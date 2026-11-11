"""Merkle tree / Merkleized State Tree (MST) with inclusion proofs.

Pure-stdlib (hashlib) so the skeleton runs offline. A record is a (key, value)
pair; the leaf hash is sha256("key|value"). The MST commits the materialized
records to a root; each record has an inclusion path to that root.
"""
from __future__ import annotations

import hashlib


def h(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def leaf_hash(key: str, value: str) -> bytes:
    return h(f"{key}|{value}".encode())


def node_hash(left: bytes, right: bytes) -> bytes:
    return h(left + right)


class MerkleTree:
    """A static Merkle tree over an ordered list of records."""

    def __init__(self, records: list[tuple[str, str]]):
        # records must be sorted by key; index = position in that order.
        self.records = records
        self.n = len(records)
        self.leaves = [leaf_hash(k, v) for k, v in records]
        self.level: list[list[bytes]] = [self.leaves]
        cur = self.leaves
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur), 2):
                l = cur[i]
                r = cur[i + 1] if i + 1 < len(cur) else cur[i]
                nxt.append(node_hash(l, r))
            self.level.append(nxt)
            cur = nxt
        self.root = cur[0]

    def inclusion_proof(self, index: int) -> list[tuple[str, bytes]]:
        """Sibling path from leaf `index` to the root."""
        proof: list[tuple[str, bytes]] = []
        idx = index
        for lvl in self.level[:-1]:
            if idx % 2 == 0:
                sib = lvl[idx + 1] if idx + 1 < len(lvl) else lvl[idx]
                proof.append(("R", sib))
            else:
                proof.append(("L", lvl[idx - 1]))
            idx //= 2
        return proof

    @staticmethod
    def verify(leaf: bytes, proof: list[tuple[str, bytes]], root: bytes) -> bool:
        cur = leaf
        for side, sib in proof:
            if side == "L":
                cur = node_hash(sib, cur)
            else:
                cur = node_hash(cur, sib)
        return cur == root

    # ---- authenticated range (completeness) proof ------------------------- #
    def _node_span(self, i: int, h: int) -> tuple[int, int]:
        """Real leaf span of node (i, h): [i*2^h, min((i+1)*2^h, n))."""
        lo = i * (1 << h)
        hi = min((i + 1) * (1 << h), self.n)
        return lo, hi

    def _cover(self, i: int, h: int, lo: int, hi: int, acc: list):
        """Canonical cover: decompose [lo,hi) into disjoint aligned subtrees.

        A node is emitted only when its VIRTUAL span [i*2^h, (i+1)*2^h) is
        fully inside [lo,hi); since hi <= n this guarantees the block is a true
        power-of-two tile (never a clamped right-edge node), so the tiling
        check in the verifier (unclamped spans) stays exact."""
        vlo = i * (1 << h)
        vhi = (i + 1) * (1 << h)
        rhi = min(vhi, self.n)          # real span (clamped to n)
        if vlo >= hi or rhi <= lo:      # node disjoint from [lo,hi)
            return
        if lo <= vlo and vhi <= hi:     # virtual span fully inside -> tile
            acc.append((i, h))
            return
        if h == 0:
            acc.append((i, 0))
            return
        self._cover(2 * i, h - 1, lo, hi, acc)
        # right child may be the duplicated left child (no real node) -> skip.
        if 2 * i + 1 < len(self.level[h - 1]):
            self._cover(2 * i + 1, h - 1, lo, hi, acc)

    def subtree_sibling_path(self, i: int, h: int) -> list[tuple[str, bytes]]:
        """Sibling path from the subtree node (i,h) up to the root."""
        path: list[tuple[str, bytes]] = []
        cur = i
        for lv in range(h, len(self.level) - 1):
            lvl = self.level[lv]
            if cur % 2 == 0:
                sib = lvl[cur + 1] if cur + 1 < len(lvl) else lvl[cur]
                path.append(("R", sib))
            else:
                path.append(("L", lvl[cur - 1]))
            cur //= 2
        return path

    def range_proof(self, lo: int, hi: int) -> list[tuple[int, int, list]]:
        """Authenticated range (completeness) proof for the leaf span [lo,hi).

        Returns the canonical cover as a list of ``(i, h, sibling_path)`` where
        each ``(i,h)`` is an aligned subtree fully inside [lo,hi) and its
        ``sibling_path`` connects that subtree's root to the tree root. The
        cover is disjoint, ordered, and exactly tiles [lo,hi); together the
        sibling paths let a verifier reconstruct the root from the leaves of
        the range, proving that the returned leaves are *exactly* the
        committed leaves in [lo,hi) (no drop, no duplicate, no extra)."""
        if lo >= hi:
            return []
        acc: list[tuple[int, int]] = []
        self._cover(0, len(self.level) - 1, lo, hi, acc)
        return [(i, h, self.subtree_sibling_path(i, h)) for i, h in acc]
