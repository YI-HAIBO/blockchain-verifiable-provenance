"""SOTA comparison harness (scaffold for reviewer item M3).

The paper already contrasts the verifiable hybrid index against *non-verifiable*
baselines (full-chain scan, off-chain materialization, a standard event
indexer). M3 asks for a head-to-head with the *verifiable* ADS designs
vChain, GEM^2-tree, and MBFT on the same workloads.

This module makes that comparison *one implementation away*:

* :func:`run_baseline_table` already emits the non-verifiable baseline numbers
  using the classes in ``hybrid.py`` (FullScan / OffchainOnly /
  StandardIndexer).
* :func:`run_verifiable_sota` is a clearly-marked stub with the exact call
  shape each reference design must satisfy. Drop in the reference
  implementation (or a faithful re-implementation of its commit/verify API) and
  the table in the paper gains the missing vChain / GEM^2-tree / MBFT columns.

No fabricated numbers are produced: the stub returns ``None`` so the harness
never lies about SOTA latency.
"""
from __future__ import annotations

import bisect
import time
from typing import Dict, List, Optional

from hybrid import HybridIndex, FullScan, OffchainOnly, StandardIndexer
from merkle import MerkleTree


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


def run_baseline_table(recs: List[tuple], query_keys: List[str]):
    """Return latency/certificate columns for the non-verifiable baselines."""
    rows = {}
    k1, k2 = min(query_keys), max(query_keys)
    # Full-chain scan: materialize then filter (no certificate).
    fs = FullScan(recs)
    _, ms = _timed(lambda: list(fs.query_range(k1, k2)))
    rows["FullScan"] = {"ms": ms, "cert": "none"}

    # Off-chain materialization without verification.
    oc = OffchainOnly(recs)
    _, ms = _timed(lambda: list(oc.query_range(k1, k2)))
    rows["OffchainOnly"] = {"ms": ms, "cert": "none"}

    # Standard event indexer (returns no certificate).
    si = StandardIndexer(recs)
    _, ms = _timed(lambda: list(si.query_range(k1, k2)))
    rows["StandardIndexer"] = {"ms": ms, "cert": "none"}
    return rows


def run_verifiable_sota(recs: List[tuple], query_keys: List[str]) -> Dict[str, Optional[dict]]:
    """Stub for verifiable-ADS comparison.

    Each entry must be filled with a reference implementation that exposes the
    same commit/verify contract as :class:`HybridIndex`:
        idx = RefDesign(recs)
        result = idx.query_range(k1, k2)
        cert = idx.certificate(result)
        ok = idx.verify_result(cert, idx.root)
    Until then the values are ``None`` (never fabricated).
    """
    h = HybridIndex(recs)
    k1, k2 = min(query_keys), max(query_keys)
    result = h.query_range(k1, k2)
    cert = h.certificate(result)
    _, ms = _timed(lambda: h.verify_result(cert, h.root))
    out = {
        "HybridIndex": {"ms": ms, "cert": "yes", "verify": "yes"},
        # --- plug reference implementations below ---
        "vChain": None,        # boolean range queries over blockchain DBs
        "GEM2-tree": None,     # gas-efficient authenticated range queries
        "MBFT": None,          # authenticated aggregate queries
    }
    return out


def run_naive_merkle(recs: List[tuple], query_keys: List[str]) -> dict:
    """Naive (full-path) authenticated range proof.

    Every leaf in the range carries its own O(log n) inclusion path, so the
    certificate is O(|R| log n) hashes and verification is O(|R| log n). This is
    an *honest internal contrast* that isolates the value of the compact cover
    (O(log n) subtree boundaries) used by the paper; it is NOT a substitute for
    the third-party verifiable ADS designs vChain / GEM^2-tree / MBFT, which
    remain pluggable (``None``) in :func:`run_verifiable_sota`.

    Returns certificate bytes, verify latency, and the compact-cover ratio.
    """
    recs = sorted(recs, key=lambda kv: kv[0])
    keys = [k for k, _ in recs]
    k1, k2 = min(query_keys), max(query_keys)
    lo = bisect.bisect_left(keys, k1)
    hi = bisect.bisect_right(keys, k2)
    t = MerkleTree(recs)

    cert_bytes = 0
    ok = True
    t0 = time.perf_counter()
    for idx in range(lo, hi):
        leaf = t.leaves[idx]
        proof = t.inclusion_proof(idx)
        cert_bytes += sum(len(b) for _, b in proof)
        ok = ok and MerkleTree.verify(leaf, proof, t.root)
    verify_ms = (time.perf_counter() - t0) * 1000.0

    cover = t.range_proof(lo, hi)
    cover_bytes = sum(len(b) for _, _, sp in cover for _, b in sp)
    return {
        "leaves_in_range": hi - lo,
        "cert_bytes": cert_bytes,
        "cover_bytes": cover_bytes,
        "cert_ratio": (cert_bytes / cover_bytes) if cover_bytes else None,
        "verify_ms": verify_ms,
        "ok": ok,
    }


def summarize(recs: List[tuple], query_keys: List[str]) -> dict:
    return {
        "baselines": run_baseline_table(recs, query_keys),
        "verifiable_sota": run_verifiable_sota(recs, query_keys),
        "naive_merkle": run_naive_merkle(recs, query_keys),
    }


if __name__ == "__main__":
    # Smoke test with a tiny synthetic slice (no fabricated SOTA numbers).
    from workloads import e_voting
    r = e_voting(1000)
    keys = [k for k, _ in r[:50]]
    print(summarize(r, keys))
