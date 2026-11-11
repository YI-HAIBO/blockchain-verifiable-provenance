"""Hybrid Merkleized index: on-chain state tree + off-chain queryable index.

The off-chain index answers queries fast; every entry carries the record's
Merkle path (its on-chain commitment), so a result can be verified lazily
against the current root. An adversarial index server can return forged
results, but verification (soundness) rejects them.
"""
from __future__ import annotations

import bisect

from merkle import MerkleTree, leaf_hash, node_hash


class HybridIndex:
    """Couples an MST (chain commitment) with an ordered queryable index."""

    def __init__(self, records: list[tuple[str, str]]):
        # records sorted by key.
        self.records = sorted(records, key=lambda kv: kv[0])
        self.keys = [k for k, _ in self.records]
        self.tree = MerkleTree(self.records)
        self.root = self.tree.root
        # key -> (value, inclusion proof)
        self.paths = {
            k: self.tree.inclusion_proof(i) for i, (k, _) in enumerate(self.records)
        }
        # versioned history: one snapshot per committed root. Snapshot v is
        # the state after the v-th batch was committed. Supports time-travel
        # (query_at_version) against a target version root.
        self.history: list[dict] = []
        self._push_snapshot()

    def _push_snapshot(self):
        self.history.append({
            "version": len(self.history) + 1,
            "records": self.records,
            "keys": self.keys,
            "tree": self.tree,
            "root": self.root,
            "paths": self.paths,
        })

    def _pos(self, key: str) -> int:
        i = bisect.bisect_left(self.keys, key)
        return i if i < len(self.keys) and self.keys[i] == key else -1

    def append_batch(self, new_records: list[tuple[str, str]]):
        """Batch-incremental sync: merge new committed records and rebuild the
        MST. In this deployment the index synchronizes per batch (per-block or
        per-epoch); freshness is measured as how far the index lags the chain
        tip between batches. Returns the new root."""
        self.records = sorted(self.records + new_records, key=lambda kv: kv[0])
        self.keys = [k for k, _ in self.records]
        self.tree = MerkleTree(self.records)
        self.root = self.tree.root
        self.paths = {
            k: self.tree.inclusion_proof(i) for i, (k, _) in enumerate(self.records)
        }
        self._push_snapshot()
        return self.root

    def apply_updates(self, updates: list[tuple[str, str]]):
        """Apply a batch of state UPDATES (overwrite an existing key's value,
        or insert a new key) and rebuild the MST. This supports MUTABLE state
        ---account balances, ERC-20 holdings, contract storage---where a key's
        value changes over time, in contrast to append-only audit logs.

        The version root history (A) tracks each state; a time-travel query at
        an earlier version returns the value that was committed at that version,
        and verification against that version's root is sound: an old value
        rolled back to masquerade as the latest is rejected against the latest
        root. Returns the new root."""
        return self.apply_mutations([("set", k, v) for k, v in updates])

    def apply_mutations(self, mutations: list[tuple]):
        """Apply a batch of THREE-STATE mutations and rebuild the MST. Each
        mutation is ``('set', key, value)`` (insert or overwrite) or
        ``('del', key)`` (delete). This completes the mutable-state semantics
        (N): insert / update / delete, complementing ``apply_updates``.

        A deleted key is REMOVED from the state tree, so a point query returns
        ``None`` and a non-membership proof (see ``nonmembership_proof``)
        certifies its absence against the latest root. The version root history
        (A) tracks each state, so a time-travel query at an earlier version
        still returns the pre-deletion value, and verification against that
        version's root is sound. Returns the new root."""
        m = dict(self.records)
        for mut in mutations:
            op = mut[0]
            if op == "set":
                m[mut[1]] = mut[2]
            elif op == "del":
                m.pop(mut[1], None)
            else:
                raise ValueError(f"unknown mutation op: {op!r}")
        self.records = sorted(m.items(), key=lambda kv: kv[0])
        self.keys = [k for k, _ in self.records]
        self.tree = MerkleTree(self.records)
        self.root = self.tree.root
        self.paths = {
            k: self.tree.inclusion_proof(i) for i, (k, _) in enumerate(self.records)
        }
        self._push_snapshot()
        return self.root

    # ---- queries (off-chain, fast) --------------------------------------- #
    def query_point(self, key: str):
        i = self._pos(key)
        if i < 0:
            return None
        return self.records[i]

    def query_range(self, k1: str, k2: str) -> list[tuple[str, str]]:
        lo = bisect.bisect_left(self.keys, k1)
        hi = bisect.bisect_right(self.keys, k2)
        return self.records[lo:hi]

    def query_lineage(self, prefix: str) -> list[tuple[str, str]]:
        """Lineage: all records whose key starts with `prefix`."""
        return [r for r in self.records if r[0].startswith(prefix)]

    def lineage_certificate(self, prefix: str) -> dict:
        """Verifiable lineage (provenance graph): the records whose key starts
        with `prefix`. Because keys are sorted and prefix matches are
        CONTIGUOUS, the lineage is exactly a range over [lo, hi), so the
        authenticated range proof proves the provenance graph is COMPLETE (no
        dropped node, no false edge): the returned records are exactly the
        records in the graph rooted at `prefix`."""
        lo = bisect.bisect_left(self.keys, prefix)
        hi = bisect.bisect_left(self.keys, prefix + "\xff")  # first key past prefix
        records = [(k, v, lo + j) for j, (k, v) in enumerate(self.records[lo:hi])]
        cover = self.tree.range_proof(lo, hi)
        return {"lo": lo, "hi": hi, "records": records, "cover": cover}

    # ---- provenance certificate + lazy verification ---------------------- #
    def certificate(self, result: list[tuple[str, str]]) -> list[tuple[str, str, list]]:
        """Attach each record's Merkle path (the certificate)."""
        return [(k, v, self.paths[k]) for k, v in result]

    @staticmethod
    def verify_result(result: list[tuple[str, str, list]], root: bytes) -> bool:
        """Soundness for list/range queries: every record must verify to root
        AND the result must contain no duplicate key (each record appears once).

        Note: this provides per-record inclusion soundness and no-duplicate
        checking; completeness of the RANGE (no dropped record) is guaranteed
        by the aggregation form or by a range-proof (see paper, Section 3)."""
        seen = set()
        for k, v, proof in result:
            if k in seen:
                return False  # duplicate record -> reject
            seen.add(k)
            if not MerkleTree.verify(leaf_hash(k, v), proof, root):
                return False
        return True

    # ---- non-membership (absence) proof for deleted keys ------------------- #
    def nonmembership_proof(self, key: str):
        """Prove that ``key`` is NOT committed in the current state (e.g. a
        record that was DELETED under the tri-state semantics). Because keys
        are sorted, absence is certified by the two boundary records
        ``left = max{k' < key}`` and ``right = min{k' > key}`` together with
        their inclusion proofs: the verifier checks ``left.key < key <
        right.key`` and that both boundaries are committed to the root, so
        ``key`` cannot be in the tree. Returns ``None`` if ``key`` IS a member
        (then a membership proof is the right certificate)."""
        i = bisect.bisect_left(self.keys, key)
        if i < len(self.keys) and self.keys[i] == key:
            return None  # key is present -> not a non-membership proof
        left = None
        if i > 0:
            lk = self.records[i - 1][0]
            left = (self.records[i - 1][0], self.records[i - 1][1], self.paths[lk])
        right = None
        if i < len(self.records):
            rk = self.records[i][0]
            right = (self.records[i][0], self.records[i][1], self.paths[rk])
        return {"key": key, "left": left, "right": right}

    @staticmethod
    def verify_nonmembership(cert: dict, root: bytes) -> bool:
        """Verify an absence (non-membership) certificate: ``key`` is not in
        the committed state. Requires (i) the present boundary records verify
        to the root (soundness of the boundaries), and (ii) the boundaries
        strictly bracket ``key`` (``left.key < key < right.key``), so no leaf
        can equal ``key`` inside the tree."""
        key = cert["key"]
        for side in ("left", "right"):
            rec = cert[side]
            if rec is None:
                continue
            k, v, proof = rec
            if not MerkleTree.verify(leaf_hash(k, v), proof, root):
                return False
        # ordering: left.key < key < right.key (when both present)
        if cert["left"] is not None and not (cert["left"][0] < key):
            return False
        if cert["right"] is not None and not (key < cert["right"][0]):
            return False
        return True

    # ---- versioned (time-travel) history queries -------------------------- #
    def version_count(self) -> int:
        return len(self.history)

    def root_at(self, version: int) -> bytes:
        return self.history[version - 1]["root"]

    def query_point_at(self, key: str, version: int):
        """Point lookup as of a historical version (time-travel)."""
        snap = self.history[version - 1]
        i = bisect.bisect_left(snap["keys"], key)
        if i < len(snap["keys"]) and snap["keys"][i] == key:
            return snap["records"][i]
        return None

    def query_range_at(self, k1: str, k2: str, version: int) -> list[tuple[str, str]]:
        snap = self.history[version - 1]
        lo = bisect.bisect_left(snap["keys"], k1)
        hi = bisect.bisect_right(snap["keys"], k2)
        return snap["records"][lo:hi]

    def certificate_at(self, result: list[tuple[str, str]],
                       version: int) -> list[tuple[str, str, list]]:
        snap = self.history[version - 1]
        return [(k, v, snap["paths"][k]) for k, v in result]

    def nonmembership_proof_at(self, key: str, version: int):
        """Versioned absence proof: prove ``key`` is not committed as of a
        historical version (e.g. deleted before ``version``). Uses that
        snapshot's sorted key set and inclusion paths, verified against
        ``root_at(version)``. Returns ``None`` if ``key`` IS a member there."""
        snap = self.history[version - 1]
        keys, records, paths = snap["keys"], snap["records"], snap["paths"]
        i = bisect.bisect_left(keys, key)
        if i < len(keys) and keys[i] == key:
            return None
        left = None
        if i > 0:
            lk = records[i - 1][0]
            left = (records[i - 1][0], records[i - 1][1], paths[lk])
        right = None
        if i < len(records):
            rk = records[i][0]
            right = (records[i][0], records[i][1], paths[rk])
        return {"key": key, "left": left, "right": right}

    # ---- authenticated range (completeness) certificate ------------------- #
    def range_certificate(self, k1: str, k2: str) -> dict:
        """A verifiable range certificate: per-record (key, value, leaf index)
        plus the canonical-cover range proof. ``verify_range`` proves the
        returned records are EXACTLY the committed records in [k1,k2]: no
        record dropped (completeness), none duplicated, none extra, all
        committed to the root."""
        lo = bisect.bisect_left(self.keys, k1)
        hi = bisect.bisect_right(self.keys, k2)
        records = [(k, v, lo + j) for j, (k, v) in enumerate(self.records[lo:hi])]
        cover = self.tree.range_proof(lo, hi)
        return {"lo": lo, "hi": hi, "records": records, "cover": cover}

    def range_certificate_at(self, k1: str, k2: str, version: int) -> dict:
        snap = self.history[version - 1]
        lo = bisect.bisect_left(snap["keys"], k1)
        hi = bisect.bisect_right(snap["keys"], k2)
        records = [(k, v, lo + j) for j, (k, v) in enumerate(snap["records"][lo:hi])]
        cover = snap["tree"].range_proof(lo, hi)
        return {"lo": lo, "hi": hi, "records": records, "cover": cover}

    @staticmethod
    def verify_range(cert: dict, root: bytes) -> bool:
        """Completeness + soundness for a range query.

        Checks (i) the record leaf indices are exactly the contiguous span
        [lo,hi) (no drop, no duplicate, no extra), (ii) the canonical cover
        exactly tiles [lo,hi) (disjoint, ordered), and (iii) folding each
        cover subtree (rebuilt from the returned leaves) up via its sibling
        path reconstructs ``root``. If all three hold, the returned records are
        exactly the committed records in the range."""
        records = cert["records"]
        cover = cert["cover"]
        lo, hi = cert["lo"], cert["hi"]
        # (i) contiguous, ordered leaf indices -> no drop / dup / extra
        idxs = [r[2] for r in records]
        if idxs != list(range(lo, hi)):
            return False
        # (ii) cover must exactly tile [lo,hi) (aligned, disjoint, ordered)
        b = lo
        for i, h, _ in cover:
            if (b % (1 << h)) != 0 or i != (b >> h):
                return False
            b += (1 << h)
        if b != hi:
            return False
        # (iii) rebuild: fold each cover subtree root up to the tree root
        rec_by_idx = {r[2]: r for r in records}
        for i, h, sib in cover:
            s_lo = i * (1 << h)
            s_hi = s_lo + (1 << h)
            sub = [(rec_by_idx[idx][0], rec_by_idx[idx][1])
                   for idx in range(s_lo, s_hi)]
            sub_root = MerkleTree(sub).root
            cur = sub_root
            for side, sibling in sib:
                if side == "L":
                    cur = node_hash(sibling, cur)
                else:
                    cur = node_hash(cur, sibling)
            if cur != root:
                return False
        return True

    @staticmethod
    def range_proof_size(cert: dict) -> tuple[int, int, int]:
        """(subtree count, total hash bytes, boundary hash bytes). The boundary
        (distinct) hash set is the compressible O(log n) form."""
        subtrees = len(cert["cover"])
        all_hashes = []
        for _, _, sib in cert["cover"]:
            all_hashes.extend(sib)
        total = 32 * len(all_hashes)
        distinct = 32 * len({h for _, h in all_hashes})
        return subtrees, total, distinct

    # ---- aggregation with a verifiable certificate ----------------------- #
    def query_aggregate(self, k1: str, k2: str, group_fn, agg_init, agg_step,
                        finalize):
        """Range aggregation over [k1,k2].

        Returns ``(R, covered)`` where ``R`` is the dict of per-group FINAL
        aggregate values (``finalize`` applied) and ``covered`` is the list of
        records that participated (the certificate covers exactly these
        records). ``group_fn(k,v)`` maps a record to a group key;
        ``agg_step(cur, k, v)`` folds one record into the accumulator.
        """
        lo = bisect.bisect_left(self.keys, k1)
        hi = bisect.bisect_right(self.keys, k2)
        acc = {}
        covered = []
        for k, v in self.records[lo:hi]:
            g = group_fn(k, v)
            acc[g] = agg_step(acc.get(g, agg_init()), k, v)
            covered.append((k, v))
        R = {g: finalize(a) for g, a in acc.items()}
        return R, covered

    @staticmethod
    def verify_aggregate(R, cert, root, group_fn, agg_init, agg_step, finalize):
        """Soundness for aggregation: every covered record must verify to root
        AND the recomputed aggregation must equal the claimed result.

        Honest server: passes (completeness). Forged aggregate value or a
        tampered/inserted record: rejected (soundness).
        """
        # (1) coverage soundness: every record in the certificate is committed
        for k, v, proof in cert:
            if not MerkleTree.verify(leaf_hash(k, v), proof, root):
                return False
        # (2) aggregation soundness: recompute and compare (finalize the
        # recomputed accumulator; the claimed value is already finalized).
        R2 = {}
        for k, v, _ in cert:
            g = group_fn(k, v)
            R2[g] = agg_step(R2.get(g, agg_init()), k, v)
        for g, val in R.items():
            if finalize(R2.get(g, agg_init())) != val:
                return False
        # any group in R2 not claimed in R means the server dropped a group
        if set(R2.keys()) != set(R.keys()):
            return False
        return True

    # ---- compact (subtree) aggregation certificate ------------------------- #
    def compact_aggregate_certificate(self, k1: str, k2: str, group_fn,
                                      agg_init, agg_step, merge, finalize) -> tuple[dict, dict]:
        """COMPACT aggregation certificate: the range [k1,k2] is decomposed
        into the O(log n) canonical-cover subtrees; each subtree carries its
        OWN aggregate map {group: accumulator} (accumulators are MERGED, not
        per-record) plus the subtree root and its sibling path to the tree root.

        Because the aggregation is associative (``merge``), the per-subtree
        accumulators combine into the whole-range aggregate, so the certificate
        size is O(#subtrees x |groups|) = O(log n x |groups|) instead of
        O(|R| x |groups|) for a per-record certificate. Returns
        ``(R, cert)`` where ``R`` is the claimed FINALIZED whole-range
        aggregate and ``cert`` is the compact certificate."""
        lo = bisect.bisect_left(self.keys, k1)
        hi = bisect.bisect_right(self.keys, k2)
        cover = self.tree.range_proof(lo, hi)
        subtrees = []
        for i, h, sib in cover:
            s_lo = i * (1 << h)
            s_hi = min((i + 1) * (1 << h), self.n_records())
            acc = {}
            for idx in range(s_lo, s_hi):
                k, v = self.records[idx]
                g = group_fn(k, v)
                acc[g] = agg_step(acc.get(g, agg_init()), k, v)
            subtrees.append({
                "i": i, "h": h,
                "root": self.tree.level[h][i],
                "acc": acc, "sib": sib,
            })
        # merge all subtree accumulators, then finalize -> whole-range aggregate
        Racc = {}
        for sub in subtrees:
            for g, a in sub["acc"].items():
                Racc[g] = merge(Racc.get(g, agg_init()), a)
        R = {g: finalize(a) for g, a in Racc.items()}
        return R, {"lo": lo, "hi": hi, "subtrees": subtrees}

    @staticmethod
    def verify_compact_aggregate(R, cert, root, group_fn, agg_init, agg_step,
                                 merge, finalize) -> bool:
        """Verify a compact aggregation certificate.

        (1) Every cover-subtree root verifies to ``root`` via its sibling path
        (soundness of the subtree commitments).
        (2) Merging the per-subtree accumulators reproduces the claimed
        whole-range aggregate (aggregation soundness, associative merge).
        Together these prove R is the aggregate over EXACTLY the committed
        records in [k1,k2] (the cover is the canonical cover of the range)."""
        # (1) subtree commitments
        for sub in cert["subtrees"]:
            if not MerkleTree.verify(sub["root"], sub["sib"], root):
                return False
        # (2) aggregation soundness
        R2 = {}
        for sub in cert["subtrees"]:
            for g, a in sub["acc"].items():
                R2[g] = merge(R2.get(g, agg_init()), a)
        if set(R2.keys()) != set(R.keys()):
            return False
        for g, val in R.items():
            if finalize(R2.get(g, agg_init())) != val:
                return False
        return True

    @staticmethod
    def compact_cert_size(cert) -> tuple[int, int]:
        """(subtree count, approx certificate bytes). Each subtree contributes
        its (group -> accumulator) map plus the sibling path (32 B per hash).
        Accumulators are small ints / (sum,count) tuples; use 8 B per value."""
        subtrees = cert["subtrees"]
        hashes = 0
        vals = 0
        for sub in subtrees:
            hashes += 32 * len(sub["sib"])
            vals += 8 * len(sub["acc"])  # ~8 B per accumulator entry
        return len(subtrees), hashes + vals

    def n_records(self) -> int:
        return len(self.records)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
class FullScan:
    """(i) Full-chain scan: correct but slow (no index)."""

    def __init__(self, records):
        self.records = sorted(records, key=lambda kv: kv[0])

    def query_range(self, k1, k2):
        return [r for r in self.records if k1 <= r[0] <= k2]


class OffchainOnly:
    """(ii) Off-chain materialization WITHOUT verification (forfeits
    tamper-evidence): fast, but a forged answer is accepted silently."""

    def __init__(self, records):
        self.records = sorted(records, key=lambda kv: kv[0])

    def query_range(self, k1, k2):
        return [r for r in self.records if k1 <= r[0] <= k2]


class StandardIndexer:
    """(iii) A standard event-indexer (e.g., The Graph-style subgraph index):
    materializes the ledger into a hash/dict index for fast specific queries,
    but returns NO provenance certificate. Mirrors the event-indexing pattern;
    a forged answer is accepted (no verification is possible)."""

    def __init__(self, records):
        self.map = {k: v for k, v in records}
        self.keys = sorted(self.map)

    def query_point(self, key):
        return (key, self.map[key]) if key in self.map else None

    def query_range(self, k1, k2):
        import bisect
        lo = bisect.bisect_left(self.keys, k1)
        hi = bisect.bisect_right(self.keys, k2)
        return [(k, self.map[k]) for k in self.keys[lo:hi]]

    # no certificate() / verify_result(): tamper-evidence forfeited.
    def has_certificate(self) -> bool:
        return False


class ExternalDBBaseline:
    """(iv) A VQL/EtherQL-style baseline: the ledger is materialized into an
    off-chain database whose WHOLE-DB fingerprint (Merkle root) is periodically
    anchored on-chain. Verification is coarse-grained (whole-DB), NOT
    per-record. It exposes a replay weakness: because every historical anchor
    root remains 'valid', an adversarial server can roll the database back to a
    previously anchored state and pass whole-DB verification, whereas the
    hybrid index verifies every record against the LATEST root and rejects the
    replay."""

    def __init__(self, records, anchor_batch):
        self.records = sorted(records, key=lambda kv: kv[0])
        self.keys = [k for k, _ in self.records]
        self.anchor_batch = anchor_batch
        # whole-DB fingerprints anchored every anchor_batch records; each is a
        # (covered_count, root) pair. ANY historical root stays acceptable.
        self.anchors = []
        for i in range(0, len(records), anchor_batch):
            end = min(i + anchor_batch, len(records))
            prefix = self.records[:end]
            self.anchors.append((end, MerkleTree(prefix).root))

    def query_range(self, k1, k2):
        import bisect
        lo = bisect.bisect_left(self.keys, k1)
        hi = bisect.bisect_right(self.keys, k2)
        return self.records[lo:hi]

    @staticmethod
    def verify_whole(records, root) -> bool:
        """Whole-DB fingerprint check: recompute the Merkle root of the given
        (possibly rolled-back) state and see if it matches an anchored root."""
        return MerkleTree(records).root == root

    def replay_state(self, count):
        """Adversarial server rolls the DB back to a previously anchored state
        (the first `count` records) and returns it with that anchor's root."""
        for end, root in self.anchors:
            if end == count:
                return self.records[:count], root
        return None, None
