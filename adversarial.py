"""Adversarial index server: an untrusted server that may forge results.

Soundness test: a forged result (wrong value / tampered record / a record
that is not in the committed state) must be REJECTED by lazy verification.
The unverifiable off-chain baseline accepts the forgery (false-accept=1),
the hybrid index rejects it (false-accept=0).
"""
from __future__ import annotations

from hybrid import HybridIndex, OffchainOnly
from merkle import leaf_hash, MerkleTree


def forge_value(server: HybridIndex, key: str) -> tuple[str, str]:
    """A server that returns a tampered value for a known key."""
    rec = server.query_point(key)
    assert rec is not None
    return (rec[0], rec[1] + "_forged")


def forge_wrong_key(server: HybridIndex, key: str) -> tuple[str, str]:
    """A server that returns a record under the wrong key (not committed)."""
    rec = server.query_point(key)
    assert rec is not None
    return (rec[0] + "_fake", rec[1])


def soundness_test(server: HybridIndex, probe_keys: list[str]) -> dict:
    """Run forgery attempts and report whether verification rejects them."""
    rejects = 0
    total = 0
    for k in probe_keys:
        for forge in (forge_value, forge_wrong_key):
            forged = forge(server, k)
            cert = [(forged[0], forged[1], server.paths.get(forged[0], []))]
            ok = HybridIndex.verify_result(cert, server.root)
            total += 1
            if not ok:
                rejects += 1
    return {"forgery_trials": total, "rejected": rejects,
            "false_accept_rate": 1.0 - rejects / total if total else 1.0}


# --------------------------------------------------------------------------- #
# Extended forgery strategies (list / range queries)
# --------------------------------------------------------------------------- #
def forge_insert_nonstate(server: HybridIndex, key: str, value: str) -> list:
    """A server inserts a record that is NOT in the committed state."""
    return [(key + "_ins", value, [])]


def forge_duplicate(server: HybridIndex, key: str) -> list:
    """A server duplicates a committed record (extra element in the result)."""
    rec = server.query_point(key)
    assert rec is not None
    p = server.paths[key]
    return [(key, rec[1], p), (key, rec[1], p)]


def forge_reorder(server: HybridIndex, keys: list[str]) -> list:
    """A server returns committed records but in the wrong order."""
    out = []
    for k in reversed(keys):
        rec = server.query_point(k)
        if rec is not None:
            out.append((k, rec[1], server.paths[k]))
    return out


def extended_soundness_test(server: HybridIndex, probe_keys: list[str]) -> dict:
    """Soundness over list/range queries: forged value, wrong key, inserted
    non-state record, and duplicated record must all be rejected."""
    def verdict(cert):
        return HybridIndex.verify_result(cert, server.root)

    def cert_of(rec):
        """Wrap a (key, value) forgery into a certificate (invalid path)."""
        k, v = rec
        return [(k, v, server.paths.get(k, []))]

    trials = {"value": 0, "wrong_key": 0, "insert": 0, "duplicate": 0}
    rejects = {"value": 0, "wrong_key": 0, "insert": 0, "duplicate": 0}
    for k in probe_keys:
        for label, forged in [
            ("value", cert_of(forge_value(server, k))),
            ("wrong_key", cert_of(forge_wrong_key(server, k))),
            ("insert", forge_insert_nonstate(server, k, "x")),
            ("duplicate", forge_duplicate(server, k)),
        ]:
            trials[label] += 1
            if not verdict(forged):
                rejects[label] += 1
    return {"trials": trials, "rejected": rejects}


# --------------------------------------------------------------------------- #
# Aggregation soundness: a forged aggregate value / dropped record / duplicate
# / reordered covered set must be rejected by verify_aggregate.
# --------------------------------------------------------------------------- #
def aggregate_soundness_test(server: HybridIndex, k1: str, k2: str,
                             group_fn, agg_init, agg_step, finalize) -> dict:
    R, covered = server.query_aggregate(k1, k2, group_fn, agg_init, agg_step,
                                        finalize)
    cert = server.certificate(covered)

    def verdict(R_, cert_):
        return HybridIndex.verify_aggregate(
            R_, cert_, server.root, group_fn, agg_init, agg_step, finalize)

    tests = {}
    tests["honest"] = verdict(R, cert)

    # (1) forged aggregate value (tamper one group's value; works for int and
    # float accumulators)
    R_fake = dict(R)
    if R_fake:
        g0 = next(iter(R_fake))
        v0 = R_fake[g0]
        if isinstance(v0, int):
            R_fake[g0] = v0 + 1
        elif isinstance(v0, float):
            R_fake[g0] = v0 + 0.5
        else:
            R_fake[g0] = f"{v0}_forged"
    tests["forged_value"] = not verdict(R_fake, cert)

    # (2) dropped a covered record (incomplete coverage -> recompute mismatch)
    cert_drop = cert[:-1] if len(cert) > 1 else []
    tests["dropped_record"] = not verdict(R, cert_drop)

    # (3) tampered covered record value
    if cert:
        k, v, p = cert[0]
        cert_tamper = [(k, v + "_tampered", p)] + cert[1:]
        tests["tampered_record"] = not verdict(R, cert_tamper)

    # (4) duplicated covered record
    if cert:
        cert_dup = cert + [cert[0]]
        tests["duplicate_record"] = not verdict(R, cert_dup)

    # (5) reordered covered records (certificate order must not matter)
    if cert:
        cert_reord = list(reversed(cert))
        tests["reorder_ok"] = verdict(R, cert_reord)

    return tests


def unverifiable_false_accept(server: OffchainOnly, probe_keys: list[str]) -> float:
    """The off-chain-only baseline has no certificate, so a forged answer is
    accepted silently -> false-accept = 1.0 by construction."""
    return 1.0
