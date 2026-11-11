"""Evaluation harness for verifiable provenance query over blockchain data.

Measures, per workload and per approach (hybrid index / full-chain scan /
off-chain-only): query latency, verification overhead (time + certificate
size), storage amplification, and adversarial false-accept rate.

Usage (offline):
    python run_eval.py --workload evoting --n 2500
    python run_eval.py --workload iot --n 5000
    python run_eval.py --workload supplychain --n 4000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from workloads import WORKLOADS
from hybrid import HybridIndex, FullScan, OffchainOnly
from adversarial import soundness_test

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000  # ms


def run_workload(name: str, n: int):
    recs = WORKLOADS[name](n)
    k_min = recs[0][0]
    k_max = recs[-1][0]

    # --- hybrid index ---
    h = HybridIndex(recs)
    # Index storage = records + Merkle tree nodes (a real design stores the
    # tree once; proofs are derived on demand, not per-record).
    data_size = sum(len(k) + len(v) for k, v in recs)
    tree_bytes = sum(len(node) for level in h.tree.level for node in level)
    index_bytes = data_size + tree_bytes

    # point / range / lineage queries
    probe = [recs[i][0] for i in range(0, n, max(1, n // 20))][:10]
    q_pt, lat_pt = timed(lambda: [h.query_point(k) for k in probe])
    rng = (recs[n // 4][0], recs[3 * n // 4][0])
    q_rg, lat_rg = timed(lambda: h.query_range(*rng))
    lineage_prefix = recs[0][0].split(":")[0] + ":"
    q_lin, lat_lin = timed(lambda: h.query_lineage(lineage_prefix))

    # verification overhead
    res = q_rg
    cert = h.certificate(res)
    t0 = time.perf_counter()
    ok = HybridIndex.verify_result(cert, h.root)
    verify_ms = (time.perf_counter() - t0) * 1000
    cert_size = sum(32 * len(p) for _, _, p in cert)  # 32-byte sha256 per hash

    # soundness under adversarial server
    ad = soundness_test(h, probe)

    # --- baselines ---
    fs = FullScan(recs)
    _, lat_scan = timed(lambda: fs.query_range(*rng))
    oc = OffchainOnly(recs)
    _, lat_oc = timed(lambda: oc.query_range(*rng))

    return {
        "workload": name, "n_records": n,
        "latency_ms": {"hybrid_point": round(lat_pt, 2),
                       "hybrid_range": round(lat_rg, 2),
                       "hybrid_lineage": round(lat_lin, 2),
                       "full_scan": round(lat_scan, 2),
                       "offchain": round(lat_oc, 2)},
        "verify_ms": round(verify_ms, 3),
        "cert_size_bytes": cert_size,
        "storage_amp": round(index_bytes / max(1, data_size), 2),
        "false_accept_hybrid": ad["false_accept_rate"],
        "false_accept_offchain": 1.0,
        "soundness_ok": ok,
        "forgery_rejected": ad["rejected"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="evoting", choices=list(WORKLOADS))
    ap.add_argument("--n", type=int, default=2500)
    args = ap.parse_args()

    r = run_workload(args.workload, args.n)
    print(json.dumps(r, indent=2))
    out = OUT / f"eval_{args.workload}.json"
    out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
