"""Full evaluation harness for verifiable provenance query over blockchain data.

Covers every dimension in the draft's Section 4 (Metrics + Baselines) plus
the query classes of Section 2:

  * aggregation / time-series queries (per-candidate tally, per-sensor avg,
    per-stage count) with a verifiable certificate (soundness + completeness)
  * freshness (batch-incremental sync lag to the chain tip)
  * standard event-indexer baseline (fast, NO certificate)
  * scaling: verify cost O(|R| log n), point O(log n), stable storage amp
  * lineage query correctness (complete, no false positives)
  * range selectivity sweep (1% / 10% / 50%)
  * extended adversarial soundness (value/wrong-key/insert/duplicate for
    lists; value/drop/tamper/duplicate/reorder for aggregates)

Usage:
    python run_full_eval.py --workload evoting --n 5000
    python run_full_eval.py --all
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from pathlib import Path

from workloads import WORKLOADS, mutable_state
from hybrid import (HybridIndex, FullScan, OffchainOnly, StandardIndexer,
                    ExternalDBBaseline)
from adversarial import extended_soundness_test, aggregate_soundness_test
import data_source as ds

# --------------------------------------------------------------------------- #
# Data-source switch. Default: high-fidelity synthetic traces (paper numbers).
# With ``--source real --data-path <csv>`` every *stateless* measure reads real
# chain exports through :mod:`data_source`; the versioned / mutable / fork
# measures use the specialized adapters (see 真实链数据导出字段规范.md).
# --------------------------------------------------------------------------- #
_SOURCE = "synthetic"
_DATA_PATH = None


def load_records(name, n):
    """Record list for ``name`` sized to ``n`` (synthetic or real)."""
    if _SOURCE == "real":
        return ds.get_data_source("real", path=_DATA_PATH).records(n, name)
    return WORKLOADS[name](n)


OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

# Per-workload aggregation spec: group key, accumulator, fold, finalize.
AGGREGATES = {
    "evoting": {
        "label": "per-candidate tally",
        "group": lambda k, v: v.split(",")[1],
        "init": lambda: 0,
        "step": lambda c, k, v: c + 1,
        "finalize": lambda c: c,
        "merge": lambda a, b: a + b,  # associative tally merge
    },
    "iot": {
        "label": "per-sensor average",
        "group": lambda k, v: k.split(":")[1],
        "init": lambda: (0.0, 0),
        "step": lambda acc, k, v: (acc[0] + float(v), acc[1] + 1),
        "finalize": lambda acc: round(acc[0] / acc[1], 3) if acc[1] else 0.0,
        "merge": lambda a, b: (a[0] + b[0], a[1] + b[1]),  # sum,count
    },
    "supplychain": {
        "label": "per-stage count",
        "group": lambda k, v: v.split(",")[0],
        "init": lambda: 0,
        "step": lambda c, k, v: c + 1,
        "finalize": lambda c: c,
        "merge": lambda a, b: a + b,
    },
}


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000  # ms


def measure_aggregate(name, n):
    """Range aggregation + verifiable certificate + adversarial soundness."""
    recs = load_records(name, n)
    h = HybridIndex(recs)
    spec = AGGREGATES[name]
    k1, k2 = recs[n // 4][0], recs[3 * n // 4][0]

    R, covered = h.query_aggregate(k1, k2, spec["group"], spec["init"],
                                   spec["step"], spec["finalize"])
    cert = h.certificate(covered)

    # correctness vs naive recompute
    naive = {}
    for k, v in recs:
        if k1 <= k <= k2:
            g = spec["group"](k, v)
            naive[g] = spec["step"](naive.get(g, spec["init"]()), k, v)
    correct = R == {g: spec["finalize"](v) for g, v in naive.items()}

    # verification (soundness of honest certificate)
    t0 = time.perf_counter()
    ok = HybridIndex.verify_aggregate(R, cert, h.root, spec["group"],
                                      spec["init"], spec["step"], spec["finalize"])
    verify_ms = (time.perf_counter() - t0) * 1000
    cert_size = sum(32 * len(p) for _, _, p in cert)

    # adversarial soundness over aggregates
    adv = aggregate_soundness_test(h, k1, k2, spec["group"], spec["init"],
                                   spec["step"], spec["finalize"])

    # latency of the aggregation itself
    _, agg_ms = timed(lambda: h.query_aggregate(k1, k2, spec["group"],
                                                spec["init"], spec["step"],
                                                spec["finalize"]))

    return {
        "workload": name, "n_records": n, "agg": spec["label"],
        "covered_records": len(covered),
        "correct": correct, "verify_ok": ok,
        "agg_ms": round(agg_ms, 2),
        "verify_ms": round(verify_ms, 3),
        "cert_size_bytes": cert_size,
        "groups": len(R),
        "adversarial": adv,
    }


def measure_freshness(name, n, batch):
    """Batch-incremental sync: how far the index lags the chain tip."""
    recs = load_records(name, n)

    # Timestamp is in the value (evoting) or in the key's last segment
    # (iot / supplychain: key = "<prefix>:<id>:<ts>").
    def ts_of(rec):
        k, v = rec
        if name == "evoting":
            return int(v.split(",")[-1])
        return int(k.split(":")[-1])

    lag_records = []
    lag_seconds = []
    h = None
    for start in range(0, n, batch):
        chunk = recs[start:start + batch]
        if h is None:
            h = HybridIndex(chunk)
        else:
            h.append_batch(chunk)
        # chain tip = last record overall; index covers up to start+batch
        pending = n - min(start + batch, n)
        lag_records.append(pending)
        if pending:
            tip_ts = ts_of(recs[-1])
            idx_ts = ts_of(recs[min(start + batch, n) - 1])
            lag_seconds.append(max(0, tip_ts - idx_ts))
    return {
        "workload": name, "n_records": n, "batch": batch,
        "batches": (n + batch - 1) // batch,
        "avg_lag_records": round(sum(lag_records) / len(lag_records), 1),
        "avg_lag_seconds": round(sum(lag_seconds) / len(lag_seconds), 1) if lag_seconds else 0,
        "max_lag_seconds": max(lag_seconds) if lag_seconds else 0,
        "final_root": h.root.hex()[:16] if h else None,
    }


def measure_scaling(name, n_list, r_size=1000, reps=3):
    """Scaling: verify O(|R| log n), point O(log n), stable storage amp.

    Each point is measured ``reps`` times and reported as mean +/- std (the
    verify cost is noisy on a shared machine, so a single-shot figure is not
    representative and can appear non-monotonic)."""
    out = []
    for n in n_list:
        recs = load_records(name, n)
        h = HybridIndex(recs)
        data_size = sum(len(k) + len(v) for k, v in recs)
        tree_bytes = sum(len(node) for level in h.tree.level for node in level)

        # point query latency (avg over 20 probes)
        probe = [recs[i][0] for i in range(0, n, max(1, n // 20))][:20]

        # fixed-size result (|R| = r_size) for range + verify
        lo = max(0, n // 2 - r_size // 2)
        hi = min(n, lo + r_size)
        k1, k2 = recs[lo][0], recs[hi - 1][0]

        lat_pts, verify_ms_list = [], []
        for _ in range(reps):
            _, lat_pt = timed(lambda: [h.query_point(k) for k in probe])
            lat_pts.append(lat_pt / len(probe))
            res, _ = timed(lambda: h.query_range(k1, k2))
            cert = h.certificate(res)
            t0 = time.perf_counter()
            ok = HybridIndex.verify_result(cert, h.root)
            verify_ms_list.append((time.perf_counter() - t0) * 1000)

        def mstd(v):
            m = sum(v) / len(v)
            sd = (sum((x - m) ** 2 for x in v) / max(1, len(v))) ** 0.5
            return round(m, 3), round(sd, 3)

        pm, psd = mstd(lat_pts)
        vm, vsd = mstd(verify_ms_list)
        res, lat_rg = timed(lambda: h.query_range(k1, k2))
        cert = h.certificate(res)

        out.append({
            "n": n, "point_ms": round(pm, 4), "point_ms_sd": round(psd, 4),
            "range_ms": round(lat_rg, 2), "result_rows": len(res),
            "verify_ms": round(vm, 2), "verify_ms_sd": round(vsd, 2),
            "verify_ok": ok, "reps": reps,
            "cert_size_bytes": sum(32 * len(p) for _, _, p in cert),
            "storage_amp": round((data_size + tree_bytes) / max(1, data_size), 2),
        })
    return {"workload": name, "r_size": r_size, "points": out}


def measure_lineage(name, n):
    """Lineage correctness: complete, no false positives."""
    recs = load_records(name, n)
    h = HybridIndex(recs)
    prefix = recs[0][0].split(":")[0] + ":"
    res, _ = timed(lambda: h.query_lineage(prefix))
    expected = [r for r in recs if r[0].startswith(prefix)]
    complete = len(res) == len(expected)
    no_false = all(r[0].startswith(prefix) for r in res)
    return {
        "workload": name, "n_records": n, "prefix": prefix,
        "returned": len(res), "expected": len(expected),
        "complete": complete, "no_false_positive": no_false,
        "lineage_ok": complete and no_false,
    }


def measure_selectivity(name, n, fracs=(0.01, 0.10, 0.50)):
    """Range latency vs selectivity for hybrid, full-scan, indexer."""
    recs = load_records(name, n)
    h = HybridIndex(recs)
    fs = FullScan(recs)
    ix = StandardIndexer(recs)
    out = []
    for f in fracs:
        span = max(1, int(n * f))
        lo = max(0, n // 2 - span // 2)
        hi = min(n, lo + span)
        k1, k2 = recs[lo][0], recs[hi - 1][0]
        _, lat_h = timed(lambda: h.query_range(k1, k2))
        _, lat_s = timed(lambda: fs.query_range(k1, k2))
        _, lat_i = timed(lambda: ix.query_range(k1, k2))
        out.append({"selectivity": f, "rows": hi - lo,
                    "hybrid_ms": round(lat_h, 2), "fullscan_ms": round(lat_s, 2),
                    "indexer_ms": round(lat_i, 2)})
    return {"workload": name, "n_records": n, "points": out}


def measure_baseline_compare(name, n):
    """Distinguishability experiment: under an adversarial server that ROLLS
    BACK (replays) a previously anchored state, which approach still verifies
    correctly? The hybrid index verifies every record against the LATEST root
    (rejects the replay); VQL/EtherQL-style whole-DB fingerprint accepts any
    historical anchor (false-accept); indexer/off-chain have no certificate."""
    recs = load_records(name, n)
    h = HybridIndex(recs)
    anchor_batch = max(100, n // 5)
    ext = ExternalDBBaseline(recs, anchor_batch)
    std = StandardIndexer(recs)
    off = OffchainOnly(recs)

    # adversarial replay: roll back to the 2nd anchor, return that state + root
    counts = [end for end, _ in ext.anchors]
    replay_count = counts[len(counts) // 2] if len(counts) > 2 else counts[-1]
    old_recs, old_root = ext.replay_state(replay_count)
    old_subset = old_recs[: max(1, len(old_recs) // 2)] if old_recs else []

    # (a) hybrid: an adversarial server returns the OLD state with its OLD
    # Merkle paths, but the verifier checks against the LATEST root -> reject.
    h_old = HybridIndex(old_recs) if old_recs else None
    hybrid_cert = ([(k, v, h_old.paths.get(k, [])) for k, v in old_subset]
                   if h_old else [])
    hybrid_reject = not HybridIndex.verify_result(hybrid_cert, h.root)

    # (b) external-DB: whole-DB fingerprint against the replayed (historical) root
    extdb_accept = ExternalDBBaseline.verify_whole(old_recs, old_root)

    # (c) indexer / off-chain: no certificate at all -> forged accepted
    return {
        "workload": name, "n_records": n, "anchor_batch": anchor_batch,
        "replay_records": len(old_recs),
        "hybrid_verify": "reject" if hybrid_reject else "accept",
        "hybrid_false_accept": 0.0 if hybrid_reject else 1.0,
        "externalDB_verify": "accept" if extdb_accept else "reject",
        "externalDB_false_accept": 0.0 if not extdb_accept else 1.0,
        "indexer_certificate": std.has_certificate(),
        "offchain_certificate": False,
    }


def measure_incremental(name, n, batch):
    """Incremental maintenance cost: per-batch rebuild vs one full rebuild."""
    recs = load_records(name, n)
    t0 = time.perf_counter()
    HybridIndex(recs)
    full_ms = (time.perf_counter() - t0) * 1000

    h = None
    inc_times = []
    for start in range(0, n, batch):
        chunk = recs[start:start + batch]
        t0 = time.perf_counter()
        if h is None:
            h = HybridIndex(chunk)
        else:
            h.append_batch(chunk)
        inc_times.append((time.perf_counter() - t0) * 1000)
    return {
        "workload": name, "n_records": n, "batch": batch,
        "batches": len(inc_times),
        "full_rebuild_ms": round(full_ms, 1),
        "avg_incremental_ms": round(sum(inc_times) / len(inc_times), 1),
        "max_incremental_ms": round(max(inc_times), 1),
    }


def measure_multi_seed(name, n, K=20):
    """Statistical adversarial soundness over K data variations: list false
    accept and aggregate reject must hold across all seeds."""
    list_fa = []
    agg_ok = []
    for k in range(K):
        nn = max(1000, n - k * (n // (K + 1)))
        recs = load_records(name, nn)
        h = HybridIndex(recs)
        probe = [recs[i][0] for i in range(0, nn, max(1, nn // 20))][:20]
        ad = extended_soundness_test(h, probe)
        total = sum(ad["trials"].values())
        rej = sum(ad["rejected"].values())
        list_fa.append(1.0 - rej / total if total else 1.0)
        spec = AGGREGATES[name]
        k1, k2 = recs[nn // 4][0], recs[3 * nn // 4][0]
        aa = aggregate_soundness_test(h, k1, k2, spec["group"], spec["init"],
                                      spec["step"], spec["finalize"])
        agg_ok.append(aa["honest"] and aa["forged_value"] and aa["dropped_record"]
                      and aa["tampered_record"] and aa["duplicate_record"])
    return {
        "workload": name, "K": K,
        "list_false_accept_max": round(max(list_fa), 4),
        "list_false_accept_avg": round(sum(list_fa) / len(list_fa), 4),
        "aggregate_ok_all_seeds": all(agg_ok),
        "seeds_ok": K if all(a and f == 0.0 for a, f in zip(agg_ok, list_fa)) else 0,
    }


def measure_query_classes(name, n):
    """All query classes (point / range / aggregate / lineage): latency and
    verification cost in one table."""
    recs = load_records(name, n)
    h = HybridIndex(recs)
    spec = AGGREGATES[name]
    mid = n // 2
    k1, k2 = recs[n // 4][0], recs[3 * n // 4][0]
    prefix = recs[0][0].split(":")[0] + ":"

    _, pt = timed(lambda: h.query_point(recs[mid][0]))
    _, rg = timed(lambda: h.query_range(k1, k2))
    R, covered = h.query_aggregate(k1, k2, spec["group"], spec["init"],
                                   spec["step"], spec["finalize"])
    cert = h.certificate(covered)
    t0 = time.perf_counter()
    HybridIndex.verify_aggregate(R, cert, h.root, spec["group"], spec["init"],
                                 spec["step"], spec["finalize"])
    verify_ms = (time.perf_counter() - t0) * 1000
    _, lin = timed(lambda: h.query_lineage(prefix))
    return {
        "workload": name, "n_records": n,
        "point_ms": round(pt, 3), "range_ms": round(rg, 3),
        "agg_ms": round(rg, 3), "lineage_ms": round(lin, 3),
        "agg_verify_ms": round(verify_ms, 1),
        "agg_covered": len(covered),
        "agg_cert_KB": round(sum(32 * len(p) for _, _, p in cert) / 1024),
    }


def measure_range_proof(name, n, frac=0.25):
    """Authenticated range (completeness) proof: honest certificate verifies
    (complete + sound); a dropped record, a duplicated record, and a tampered
    value are all rejected. Proof size decomposes into O(log n) canonical
    subtrees; the boundary (distinct) hash set is the compressible O(log n)
    form, versus the naive O(log^2 n) per-subtree sibling-path encoding."""
    recs = load_records(name, n)
    h = HybridIndex(recs)
    span = max(1, int(n * frac))
    lo = max(0, n // 2 - span // 2)
    hi = min(n, lo + span)
    k1, k2 = recs[lo][0], recs[hi - 1][0]

    cert = h.range_certificate(k1, k2)
    honest = HybridIndex.verify_range(cert, h.root)
    sub, total_b, boundary_b = HybridIndex.range_proof_size(cert)

    # adversarial: drop / duplicate / tamper must all be rejected
    drop = dict(cert); drop["records"] = cert["records"][:-1]
    drop["hi"] = cert["hi"] - 1
    dup = dict(cert); dup["records"] = cert["records"] + [cert["records"][0]]
    r0 = cert["records"][0]
    tamper = dict(cert)
    tamper["records"] = [(r0[0], r0[1] + "_X", r0[2])] + cert["records"][1:]

    _, lat_gen = timed(lambda: h.range_certificate(k1, k2))
    t0 = time.perf_counter()
    ok = HybridIndex.verify_range(cert, h.root)
    lat_verify = (time.perf_counter() - t0) * 1000

    return {
        "workload": name, "n_records": n, "range_rows": hi - lo,
        "honest_accept": honest, "verify_ok": ok,
        "rejects_drop": not HybridIndex.verify_range(drop, h.root),
        "rejects_duplicate": not HybridIndex.verify_range(dup, h.root),
        "rejects_tamper": not HybridIndex.verify_range(tamper, h.root),
        "proof_subtrees": sub,
        "proof_total_bytes": total_b,
        "proof_boundary_bytes": boundary_b,
        "gen_ms": round(lat_gen, 3),
        "verify_ms": round(lat_verify, 3),
    }


def measure_versioned(name, n, batch):
    """Versioned history queries (time-travel) + the staleness/replay attack
    that separates per-record verification from the range-proof.

    * T1 time-travel point query at a target version verifies against the
      target version root (correct).
    * T2 future-record forgery (a value claimed for a record not yet committed
      at that version) is rejected against the target root (version-scoped
      soundness).
    * T3 staleness: a range returned as of an EARLIER version, but checked with
      per-record proofs against the LATEST root, is ACCEPTED (each record is
      genuinely committed -> per-record-only verification cannot detect the
      dropped records); the RANGE-proof against the latest root REJECTS it
      (completeness gap closed).
    * T4 old-root replay: an answer certified against a historical root is
      rejected when verified against the latest root."""
    if _SOURCE == "real":
        init, batches = ds.get_data_source("real", path=_DATA_PATH).versioned_batches(name, n)
        recs = init + [r for b in batches for r in b]
        n = len(recs)
        batch = max(1, len(init))
        h = HybridIndex(init)
        for b in batches:
            h.append_batch(b)
    else:
        recs = load_records(name, n)
        h = HybridIndex(recs[:batch])
        for s in range(batch, n, batch):
            h.append_batch(recs[s:s + batch])
    m = h.version_count()
    vj = max(1, m // 2)
    lo = max(0, n // 2 - n // 4)
    hi = min(n, n // 2 + n // 4)
    k1, k2 = recs[lo][0], recs[hi - 1][0]

    # T1: time-travel point query (committed by vj) vs target root
    k = recs[vj * batch - 1][0]
    rec = h.query_point_at(k, vj)
    t1_correct = rec is not None and rec[0] == k
    t1_ok = HybridIndex.verify_result(h.certificate_at([rec], vj), h.root_at(vj)) \
        if t1_correct else False

    # T2: future-record forgery (key committed only after vj) at version vj
    kf = recs[vj * batch][0] if vj * batch < n else None
    t2_absent = h.query_point_at(kf, vj) is None if kf else True
    t2_reject = (not HybridIndex.verify_result([(kf, "forged", [])], h.root_at(vj))
                 if kf else True)

    # T3: staleness (range as-of vj verified against latest root)
    stale = h.query_range_at(k1, k2, vj)
    per_rec_paths = [(k, v, h.paths[k]) for k, v in stale]
    per_rec_accept = HybridIndex.verify_result(per_rec_paths, h.root)
    cert_latest = h.range_certificate(k1, k2)
    stale_keys = set(k for k, _ in stale)
    cert_stale = dict(cert_latest)
    cert_stale["records"] = [r for r in cert_latest["records"] if r[0] in stale_keys]
    cert_stale["hi"] = cert_stale["records"][-1][2] + 1 if cert_stale["records"] else cert_latest["lo"]
    range_reject = not HybridIndex.verify_range(cert_stale, h.root)

    # T4: old-root replay (cert against target root, checked against latest)
    cert_old = h.range_certificate_at(k1, k2, vj)
    t4_honest_at_target = HybridIndex.verify_range(cert_old, h.root_at(vj))
    t4_replay_rejected = not HybridIndex.verify_range(cert_old, h.root)

    # time-travel range latency + verify latency
    _, lat_tt = timed(lambda: h.query_range_at(k1, k2, vj))
    t0 = time.perf_counter()
    HybridIndex.verify_range(cert_old, h.root_at(vj))
    lat_tt_verify = (time.perf_counter() - t0) * 1000

    return {
        "workload": name, "n_records": n, "batch": batch, "versions": m,
        "target_version": vj, "range_rows": hi - lo,
        "T1_time_travel_correct": t1_correct and t1_ok,
        "T2_future_forgery_rejected": t2_absent and t2_reject,
        "T3_perrecord_stale_accept": per_rec_accept,
        "T3_rangeproof_stale_reject": range_reject,
        "T4_honest_at_target_root": t4_honest_at_target,
        "T4_oldroot_replay_rejected": t4_replay_rejected,
        "time_travel_ms": round(lat_tt, 3),
        "time_travel_verify_ms": round(lat_tt_verify, 3),
    }


def measure_mutable(n_accounts=2000, rounds=10, update_frac=0.5):
    """J: verifiable UPDATES over mutable account state (balances/ERC-20).

    The state is overwritten across rounds. The latest value verifies against
    the latest root (current-state soundness); a time-travel value verifies
    against its version root (historical correctness); an UPDATE-REPLAY (old
    value + old root masquerading as latest) is rejected against the latest
    root; a SKIP-VERSION certificate (v_j cert vs a later root) is rejected;
    and certificates are VERSION-BOUND (a v_j path never reconstructs a later
    root, even if the value is unchanged)."""
    if _SOURCE == "real":
        initial, batches = ds.get_data_source(
            "real", path=_DATA_PATH).mutable_batches(n_accounts, name="evoting")
        h = HybridIndex(initial)
        for b in batches:
            h.apply_updates(b)
        m = h.version_count()
        acct = h.keys[len(h.keys) // 2] if h.keys else None
        v_old = max(1, m // 2)
        v_skip = min(m, v_old + 2)
        latest = h.query_point(acct) if acct else None
        old = h.query_point_at(acct, v_old) if acct else None
        def _ok(rec, root):
            return rec is not None and HybridIndex.verify_result(
                h.certificate([rec]), root) if rec is not None else False
        replay_rej = False
        if acct and old is not None:
            replay_rej = not HybridIndex.verify_result(
                [(acct, old[1], h.paths.get(acct, []))], h.root)
        skip_rej = False
        if acct and old is not None:
            skip_rej = not HybridIndex.verify_result(
                h.certificate_at([old], v_old), h.root_at(v_skip))
        return {
            "accounts": len(initial), "rounds": len(batches), "versions": m,
            "target_version": v_old, "skip_version": v_skip,
            "build_ms": None, "latest_value_ok": _ok(latest, h.root),
            "time_travel_ok": _ok(old, h.root_at(v_old)),
            "update_replay_rejected": replay_rej,
            "skip_version_rejected": skip_rej,
            "version_bound_not_reusable": None,
            "point_ms": None, "time_travel_ms": None, "source": "real",
        }
    initial, batches = mutable_state(n_accounts, rounds, update_frac)
    t0 = time.perf_counter()
    h = HybridIndex(initial)
    for b in batches:
        h.apply_updates(b)
    build_ms = (time.perf_counter() - t0) * 1000
    m = h.version_count()

    acct = "acct:00000000"
    v_old = max(1, m // 2)
    v_skip = min(m, v_old + 2)
    latest = h.query_point(acct)
    old = h.query_point_at(acct, v_old)

    # current-state soundness
    latest_ok = HybridIndex.verify_result(h.certificate([latest]), h.root)
    # historical correctness (time-travel)
    tt_ok = HybridIndex.verify_result(h.certificate_at([old], v_old), h.root_at(v_old))
    # update-replay: old value + latest path vs latest root -> reject
    replay_rej = not HybridIndex.verify_result(
        [(acct, old[1], h.paths.get(acct, []))], h.root)
    # skip-version: v_old cert vs a later root -> reject (value changed)
    skip_rej = not HybridIndex.verify_result(
        h.certificate_at([old], v_old), h.root_at(v_skip))
    # version-binding: a static account's v_old cert does not reconstruct a
    # later root even though its value is unchanged
    acct_static = "acct:00000001"
    rec_static = h.query_point_at(acct_static, v_old)
    static_same = rec_static[1] == h.query_point(acct_static)[1]
    bound_rej = not HybridIndex.verify_result(
        h.certificate_at([rec_static], v_old), h.root_at(v_skip))

    # latencies
    _, lat_latest = timed(lambda: h.query_point(acct))
    _, lat_tt = timed(lambda: h.query_point_at(acct, v_old))

    return {
        "accounts": n_accounts, "rounds": rounds, "versions": m,
        "target_version": v_old, "skip_version": v_skip,
        "build_ms": round(build_ms, 1),
        "latest_value_ok": latest_ok,
        "time_travel_ok": tt_ok,
        "update_replay_rejected": replay_rej,
        "skip_version_rejected": skip_rej,
        "version_bound_not_reusable": bound_rej and static_same,
        "point_ms": round(lat_latest, 3),
        "time_travel_ms": round(lat_tt, 3),
    }


def measure_delete_semantics(n_accounts=2000, rounds=10, update_frac=0.5,
                             delete_frac=0.2):
    """N: tri-state mutable semantics (insert / update / DELETE / recover).

    * N1 current-state soundness for a live (never-deleted) account.
    * N2 deletion -> absence: a deleted account's point query returns None and
      a NON-MEMBERSHIP proof certifies its absence against the latest root.
    * N3 historical correctness: a deleted account's pre-deletion value
      verifies against the version root in which it was still committed.
    * N4 delete-replay: a server returns a deleted account's pre-deletion
      record (old value + old path) as the latest answer -> rejected against
      the latest root (the record is no longer in the state tree).
    * N5 genuine recovery: a re-inserted account's current value verifies
      (accepted); its pre-deletion certificate does NOT (version binding, even
    when the restored value equals the pre-deletion value), so a recovery is
    distinguishable from a replay."""
    from workloads import mutable_state_tristate
    if _SOURCE == "real":
        initial, batches = ds.get_data_source(
            "real", path=_DATA_PATH).tristate_batches(n_accounts, name="evoting")
        h = HybridIndex(initial)
        for b in batches:
            h.apply_mutations(b)
        m = h.version_count()
        latest_keys = set(h.keys)
        deleted_keys = [k for k, _ in initial if k not in latest_keys]
        dk = deleted_keys[0] if deleted_keys else None
        n1_ok = n2_ok = n3_ok = n4_rej = False
        if latest_keys:
            lk = sorted(latest_keys)[0]
            n1_ok = HybridIndex.verify_result(
                h.certificate([h.query_point(lk)]), h.root)
        if dk:
            v_pre = max(v for v in range(1, m + 1)
                        if dk in set(h.history[v - 1]["keys"]))
            n2_ok = h.query_point(dk) is None
            nm = h.nonmembership_proof(dk)
            n2_ok = n2_ok and nm is not None and \
                HybridIndex.verify_nonmembership(nm, h.root)
            pre = h.query_point_at(dk, v_pre)
            n3_ok = HybridIndex.verify_result(
                h.certificate_at([pre], v_pre), h.root_at(v_pre))
            n4_rej = not HybridIndex.verify_result(
                h.certificate_at([pre], v_pre), h.root)
        return {
            "accounts": len(initial), "rounds": len(batches), "versions": m,
            "live_at_latest": len(latest_keys),
            "deleted_at_latest": len(deleted_keys), "build_ms": None,
            "N1_live_state_ok": n1_ok, "N2_absent_query": dk is not None,
            "N2_nonmembership_ok": n2_ok, "N3_time_travel_predel_ok": n3_ok,
            "N4_delete_replay_rejected": n4_rej,
            "N5_value_restored": None, "N5_recovery_accepted": None,
            "N5_predel_replay_rejected": None, "source": "real",
        }
    initial, batches = mutable_state_tristate(n_accounts, rounds, update_frac,
                                              delete_frac)
    t0 = time.perf_counter()
    h = HybridIndex(initial)
    for b in batches:
        h.apply_mutations(b)
    build_ms = (time.perf_counter() - t0) * 1000
    m = h.version_count()
    latest_keys = set(h.keys)
    pres = {k: set(v for v in range(1, m + 1)
                   if k in set(h.history[v - 1]["keys"])) for k in latest_keys}

    # --- deleted key (absent at latest) ----------------------------------- #
    deleted_keys = [k for k, _ in initial if k not in latest_keys]
    dk = deleted_keys[0]
    v_pre = max(v for v in range(1, m + 1)
                if dk in set(h.history[v - 1]["keys"]))
    # N2: absence + non-membership proof
    absent_query = h.query_point(dk) is None
    nm = h.nonmembership_proof(dk)
    n2_ok = nm is not None and HybridIndex.verify_nonmembership(nm, h.root)
    # N3: historical correctness (pre-deletion value at v_pre)
    pre = h.query_point_at(dk, v_pre)
    n3_ok = HybridIndex.verify_result(h.certificate_at([pre], v_pre),
                                      h.root_at(v_pre))
    # N4: delete-replay rejected vs latest root
    n4_rej = not HybridIndex.verify_result(h.certificate_at([pre], v_pre), h.root)

    # --- live (never-deleted) key ----------------------------------------- #
    live_keys = [k for k in latest_keys if len(pres[k]) == m]
    lk = live_keys[0]
    n1_ok = HybridIndex.verify_result(h.certificate([h.query_point(lk)]), h.root)

    # --- recovered key (value restored, not re-updated) ------------------- #
    rec_key = v_prec = None
    for k in latest_keys:
        a = int(k.split(":")[1])
        if not (a % 5 == 0 and a % 2 == 1):
            continue  # in delete subset but not update subset -> value frozen
        for v in range(1, m):
            if v in pres[k] and (v + 1) not in pres[k] and \
               any(vv > v + 1 and vv in pres[k] for vv in range(1, m + 1)):
                rec_key, v_prec = k, v
                break
        if rec_key:
            break
    if rec_key:
        pre_r = h.query_point_at(rec_key, v_prec)
        post_r = h.query_point(rec_key)
        value_restored = pre_r[1] == post_r[1]
        n5a_ok = HybridIndex.verify_result(h.certificate([post_r]), h.root)
        n5b_rej = not HybridIndex.verify_result(
            h.certificate_at([pre_r], v_prec), h.root)
    else:
        value_restored = n5a_ok = n5b_rej = None

    return {
        "accounts": n_accounts, "rounds": rounds, "versions": m,
        "live_at_latest": len(latest_keys), "deleted_at_latest": len(deleted_keys),
        "build_ms": round(build_ms, 1),
        "N1_live_state_ok": n1_ok,
        "N2_absent_query": absent_query,
        "N2_nonmembership_ok": n2_ok,
        "N3_time_travel_predel_ok": n3_ok,
        "N4_delete_replay_rejected": n4_rej,
        "N5_value_restored": value_restored,
        "N5_recovery_accepted": n5a_ok,
        "N5_predel_replay_rejected": n5b_rej,
    }


def measure_batch_verify(name, n, sizes=(10, 50, 100, 500, 1000, 5000)):
    """O: batch verification amortization. A client issues K queries over a
    contiguous range.

    * (a) INDIVIDUAL: each record is verified with its OWN O(log n) inclusion
      path -> O(K log n) total, O(log n) per record (independent of K).
    * (b) BATCHED: ONE authenticated range-proof (canonical cover) verifies all
      K records at once -> O(K + log^2 n); the O(log n) per-record factor is
      SHARED across the batch (the cover subtrees and their sibling paths are
      common to the whole range).

    Reports the per-record verify cost under both and the amortization ratio
    (individual / batched), which grows with K: the batched per-record cost
    tends toward a constant as K grows, i.e. the O(log n) per-record
    verification overhead is amortized away for bulk audits."""
    recs = load_records(name, n)
    h = HybridIndex(recs)
    out = []
    for k in sizes:
        k = min(k, n)
        lo = max(0, n // 2 - k // 2)
        hi = min(n, lo + k)
        k1, k2 = recs[lo][0], recs[hi - 1][0]
        res = h.query_range(k1, k2)
        # (a) individual: each record's own inclusion proof verified separately
        cert_individual = h.certificate(res)
        t0 = time.perf_counter()
        ok_i = True
        for single in cert_individual:
            ok_i &= HybridIndex.verify_result([single], h.root)
        individual_ms = (time.perf_counter() - t0) * 1000
        # (b) batched: one range-proof verifies all K at once
        rc = h.range_certificate(k1, k2)
        t0 = time.perf_counter()
        ok_b = HybridIndex.verify_range(rc, h.root)
        batched_ms = (time.perf_counter() - t0) * 1000
        out.append({
            "K": len(res),
            "individual_ms": round(individual_ms, 3),
            "per_record_individual_ms": round(individual_ms / max(1, len(res)), 5),
            "batched_ms": round(batched_ms, 3),
            "per_record_batched_ms": round(batched_ms / max(1, len(res)), 5),
            "amortization_ratio": round(individual_ms / max(1e-9, batched_ms), 2),
            "individual_ok": ok_i, "batched_ok": ok_b,
        })
    return {"workload": name, "n_records": n, "points": out}


def measure_compact_aggregate(name, n):
    """S: compact (subtree) aggregation certificate. The per-record aggregate
    certificate is O(|R|) (one inclusion path per record); the compact
    certificate decomposes the range into O(log n) canonical-cover subtrees,
    each carrying its OWN aggregate map, so the certificate is
    O(log n x |groups|). Reports the size reduction and verifies the compact
    certificate is sound (reproduces the same aggregate)."""
    recs = load_records(name, n)
    h = HybridIndex(recs)
    spec = AGGREGATES[name]
    k1, k2 = recs[n // 4][0], recs[3 * n // 4][0]

    # (a) per-record certificate
    R, covered = h.query_aggregate(k1, k2, spec["group"], spec["init"],
                                   spec["step"], spec["finalize"])
    cert_rec = h.certificate(covered)
    per_record_bytes = sum(32 * len(p) for _, _, p in cert_rec)

    # (b) compact subtree certificate
    R_c, cert_c = h.compact_aggregate_certificate(
        k1, k2, spec["group"], spec["init"], spec["step"], spec["merge"],
        spec["finalize"])
    t0 = time.perf_counter()
    ok = HybridIndex.verify_compact_aggregate(
        R_c, cert_c, h.root, spec["group"], spec["init"], spec["step"],
        spec["merge"], spec["finalize"])
    verify_ms = (time.perf_counter() - t0) * 1000
    sub, compact_bytes = HybridIndex.compact_cert_size(cert_c)

    # adversarial: a forged aggregate value / a dropped subtree must be rejected
    R_fake = dict(R_c)
    if R_fake:
        g0 = next(iter(R_fake))
        v0 = R_fake[g0]
        R_fake[g0] = v0 + 1 if isinstance(v0, int) else v0 + 0.5
    forged_rej = not HybridIndex.verify_compact_aggregate(
        R_fake, cert_c, h.root, spec["group"], spec["init"], spec["step"],
        spec["merge"], spec["finalize"])
    drop_cert = dict(cert_c)
    drop_cert["subtrees"] = cert_c["subtrees"][:-1]
    drop_rej = not HybridIndex.verify_compact_aggregate(
        R_c, drop_cert, h.root, spec["group"], spec["init"], spec["step"],
        spec["merge"], spec["finalize"])

    return {
        "workload": name, "n_records": n, "covered": len(covered),
        "groups": len(R),
        "per_record_cert_bytes": per_record_bytes,
        "compact_cert_bytes": compact_bytes,
        "compression_ratio": round(per_record_bytes / max(1, compact_bytes), 1),
        "compact_subtrees": sub,
        "aggregate_equal": R == R_c,
        "compact_verify_ok": ok,
        "compact_verify_ms": round(verify_ms, 3),
        "forged_value_rejected": forged_rej,
        "dropped_subtree_rejected": drop_rej,
    }


def measure_aggregate_coverage(name, n):
    """AC: aggregate COVERAGE completeness, cross-verified with the range
    proof. ``verify_aggregate`` alone checks every covered record is committed
    and the recomputed aggregate matches --- but a server can return the
    aggregate over a SUBSET of the range (self-consistent, so it PASSES
    verify_aggregate; false-accept = 1). Cross-verifying the covered set with
    the authenticated range proof (the covered set must equal the full range
    set) closes the gap: the subset / superset aggregate is REJECTED
    (false-accept = 0)."""
    recs = load_records(name, n)
    h = HybridIndex(recs)
    spec = AGGREGATES[name]
    k1, k2 = recs[n // 4][0], recs[3 * n // 4][0]

    def recompute(records):
        acc = {}
        for k, v in records:
            g = spec["group"](k, v)
            acc[g] = spec["step"](acc.get(g, spec["init"]()), k, v)
        return {g: spec["finalize"](a) for g, a in acc.items()}

    def coverage_check(R, cert, root):
        """verify_aggregate AND range-proof cross-check (covered == full range)."""
        rc = h.range_certificate(k1, k2)
        agg_ok = HybridIndex.verify_aggregate(
            R, cert, root, spec["group"], spec["init"], spec["step"],
            spec["finalize"])
        full_keys = set(r[0] for r in rc["records"])
        covered_keys = set(k for k, _, _ in cert)
        coverage_ok = (covered_keys == full_keys)
        range_ok = HybridIndex.verify_range(rc, root)
        return agg_ok and coverage_ok and range_ok

    # (a) honest
    R, covered = h.query_aggregate(k1, k2, spec["group"], spec["init"],
                                   spec["step"], spec["finalize"])
    cert = h.certificate(covered)
    honest = coverage_check(R, cert, h.root)

    # (b) forged SUBSET aggregate (self-consistent -> verify_aggregate accepts)
    sub_cert = cert[:max(1, len(cert) // 2)]
    sub_records = [(k, v) for k, v, _ in sub_cert]
    R_sub = recompute(sub_records)
    subset_agg_accept = HybridIndex.verify_aggregate(
        R_sub, sub_cert, h.root, spec["group"], spec["init"], spec["step"],
        spec["finalize"])
    subset_cross_rej = not coverage_check(R_sub, sub_cert, h.root)

    # (c) forged SUPERSET aggregate (range + one outside record)
    lo = bisect.bisect_left(h.keys, k1)
    hi = bisect.bisect_right(h.keys, k2)
    outside_idx = lo - 1 if lo > 0 else hi
    rec_out = h.records[outside_idx]
    over_cert = cert + [(rec_out[0], rec_out[1], h.paths[rec_out[0]])]
    over_records = [(k, v) for k, v, _ in over_cert]
    R_over = recompute(over_records)
    over_agg_accept = HybridIndex.verify_aggregate(
        R_over, over_cert, h.root, spec["group"], spec["init"], spec["step"],
        spec["finalize"])
    over_cross_rej = not coverage_check(R_over, over_cert, h.root)

    return {
        "workload": name, "n_records": n, "covered": len(covered),
        "honest_coverage_ok": honest,
        "subset_aggregate_accept": subset_agg_accept,   # GAP: verify_aggregate passes
        "subset_crossverify_reject": subset_cross_rej,  # FIXED: range-proof rejects
        "superset_aggregate_accept": over_agg_accept,   # GAP
        "superset_crossverify_reject": over_cross_rej,  # FIXED
    }


def _quorum_verify(results, t, root):
    """Quorum verification over n untrusted index servers. Each server returns
    (answer, certificate) or None (offline). A server's answer is VALID only if
    its certificate verifies against ``root`` (per-server soundness). Among
    the valid answers, the majority-consensus answer is accepted iff at least
    ``t`` servers agree on it. Returns ``(accepted, winner, valid_count)``."""
    counts = {}
    winner = None
    valid_count = 0
    for R, cert in results:
        if R is None or cert is None:
            continue  # offline / no answer
        if not HybridIndex.verify_result(cert, root):
            continue  # forged -> excluded by soundness
        valid_count += 1
        key = frozenset(R)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return False, None, valid_count
    best = max(counts, key=lambda k: counts[k])
    if counts[best] >= t:
        return True, best, valid_count
    return False, None, valid_count


def measure_quorum(name, n, n_servers_list=(3, 5, 7)):
    """X: multi-indexer quorum (Byzantine) verification. The adversarial model
    is extended from a SINGLE untrusted index server to ``n`` servers of which
    up to ``f`` are malicious. Per-server soundness (certificate verification)
    excludes forged answers; majority consensus among the VALID answers makes
    the result robust to a malicious MINORITY (forged, subset, or offline).

    * X1 all honest: quorum accepts the true answer.
    * X2 honest majority + malicious minority returning a SUBSET (which passes
      per-record verification, the single-server completeness gap): quorum
      picks the honest full answer (majority).
    * X3 honest majority + malicious minority forging a value (fails per-server
      soundness): forged excluded, quorum accepts the honest answer.
    * X4 honest majority + malicious minority offline (DoS): quorum accepts
      (honest responses >= threshold).
    * X5 malicious majority: quorum accepts the (wrong) consensus --- the
      fundamental honest-majority requirement of quorum/BFT verification.
    * Quorum cost vs single-server verification (n x + majority comparison)."""
    recs = load_records(name, n)
    h = HybridIndex(recs)
    k1, k2 = recs[n // 4][0], recs[3 * n // 4][0]
    true_res = h.query_range(k1, k2)
    true_cert = h.certificate(true_res)
    subset = true_res[:max(1, len(true_res) // 2)]
    subset_cert = h.certificate(subset)
    forged_cert = [(true_res[0][0], true_res[0][1] + "_X", [])] + true_cert[1:]

    # single-server baseline verify cost
    t0 = time.perf_counter()
    HybridIndex.verify_result(true_cert, h.root)
    single_ms = (time.perf_counter() - t0) * 1000

    out = []
    honest_key = frozenset(true_res)
    for ns in n_servers_list:
        f_honest = ns // 2            # max malicious under honest majority
        t = ns // 2 + 1               # strict majority threshold
        f_mal = ns // 2 + 1           # malicious majority

        def winner(results):
            acc, win, _ = _quorum_verify(results, t, h.root)
            return acc, (win == honest_key)

        x1, w1 = winner([(true_res, true_cert)] * ns)
        x2, w2 = winner([(true_res, true_cert)] * (ns - f_honest) +
                        [(subset, subset_cert)] * f_honest)
        x3, w3 = winner([(true_res, true_cert)] * (ns - f_honest) +
                        [(None, forged_cert)] * f_honest)
        x4, w4 = winner([(true_res, true_cert)] * (ns - f_honest) +
                        [(None, None)] * f_honest)
        # malicious majority returns a CONSISTENT wrong subset (honest minority)
        x5, w5 = winner([(subset, subset_cert)] * f_mal +
                        [(true_res, true_cert)] * (ns - f_mal))

        # quorum cost: verify every cert + majority comparison
        t0 = time.perf_counter()
        _quorum_verify([(true_res, true_cert)] * ns, t, h.root)
        quorum_ms = (time.perf_counter() - t0) * 1000

        out.append({
            "servers": ns, "f_honest_majority": f_honest, "threshold": t,
            "X1_all_honest_accept": x1 and w1,
            "X2_malicious_subset_honest_wins": x2 and w2,  # honest majority
            "X3_malicious_forge_excluded": x3 and w3,      # soundness excludes
            "X4_malicious_offline_ok": x4 and w4,          # DoS-robust
            "X5_malicious_majority_defeats": x5 and not w5,  # honest-majority req.
            "quorum_verify_ms": round(quorum_ms, 3),
            "single_verify_ms": round(single_ms, 3),
            "quorum_overhead": round(quorum_ms / max(1e-9, single_ms), 1),
        })
    return {"workload": name, "n_records": n, "points": out}


def measure_attacks(name, n, batch):
    """F: extended attack surface over the versioned hybrid index.

    * range-boundary: a record OUTSIDE [k1,k2] smuggled into the answer.
    * cross-boundary cover: the canonical cover tiles a larger span.
    * skip-version range: a v_j range certificate verified against a later
      root (records in the range changed between versions -> reject)."""
    recs = load_records(name, n)
    h = HybridIndex(recs[:batch])
    for s in range(batch, n, batch):
        h.append_batch(recs[s:s + batch])
    m = h.version_count()
    v3 = min(3, m)
    v5 = min(5, m)

    keys = h.keys
    k1, k2 = keys[len(keys) // 4], keys[3 * len(keys) // 4]
    cert = h.range_certificate(k1, k2)

    # (1) range-boundary: outside record appended
    outside_idx = cert["lo"] - 1 if cert["lo"] > 0 else cert["hi"]
    rec_out = h.records[outside_idx]
    c_out = dict(cert)
    c_out["records"] = cert["records"] + [(rec_out[0], rec_out[1], outside_idx)]
    boundary_rej = not HybridIndex.verify_range(c_out, h.root)

    # (2) cross-boundary cover: tile a larger span
    c_x = dict(cert)
    c_x["lo"] = cert["lo"] - 1 if cert["lo"] > 0 else cert["lo"]
    c_x["hi"] = cert["hi"] + 1
    c_x["cover"] = h.tree.range_proof(c_x["lo"], c_x["hi"])
    cross_rej = not HybridIndex.verify_range(c_x, h.root)

    # (3) skip-version range: v3 range cert vs root_5
    cert3 = h.range_certificate_at(k1, k2, v3)
    skip_honest = HybridIndex.verify_range(cert3, h.root_at(v3))
    skip_rej = not HybridIndex.verify_range(cert3, h.root_at(v5))

    return {
        "workload": name, "n_records": n, "versions": m,
        "range_rows": cert["hi"] - cert["lo"],
        "boundary_rejected": boundary_rej,
        "cross_boundary_rejected": cross_rej,
        "skip_version_honest_at_v3": skip_honest,
        "skip_version_rejected_vs_v5": skip_rej,
    }


def measure_verify_curve(name, n, sizes=(10, 50, 100, 500, 1000, 2500, 5000)):
    """G: verification cost vs result size |R| (fixed n). Shows the
    per-record verification cost is O(log n) (nearly constant as |R| grows),
    i.e., total verify is O(|R| log n)."""
    recs = load_records(name, n)
    h = HybridIndex(recs)
    out = []
    for r in sizes:
        r = min(r, n)
        lo = max(0, n // 2 - r // 2)
        hi = min(n, lo + r)
        k1, k2 = recs[lo][0], recs[hi - 1][0]
        res = h.query_range(k1, k2)
        cert = h.certificate(res)
        t0 = time.perf_counter()
        ok = HybridIndex.verify_result(cert, h.root)
        verify_ms = (time.perf_counter() - t0) * 1000
        out.append({
            "R": len(res), "verify_ms": round(verify_ms, 3),
            "per_record_ms": round(verify_ms / max(1, len(res)), 4),
            "cert_bytes": sum(32 * len(p) for _, _, p in cert),
            "verify_ok": ok,
        })
    return {"workload": name, "n_records": n, "points": out}


def measure_portable_verify(name, n):
    """K: lightweight-client (portable) verification. A verifier holds ONLY
    the committed root and the certificate --- O(1) state, independent of the
    dataset size --- and confirms the answer. Measured with tracemalloc: the
    verifier's peak allocation during verification depends on the certificate,
    not on n. Contrasted with a full-chain-scan verifier, which must materialize
    all n records."""
    import tracemalloc
    recs = load_records(name, n)
    h = HybridIndex(recs)
    k1, k2 = recs[n // 4][0], recs[3 * n // 4][0]

    # lightweight client: only (root, cert) is transmitted / held
    rc = h.range_certificate(k1, k2)
    cert_bytes = sum(32 * len(p) for _, _, p in rc["cover"]) + \
        sum(len(r[0]) + len(r[1]) + 4 for r in rc["records"])
    root_bytes = 32

    tracemalloc.start()
    ok = HybridIndex.verify_range(rc, h.root)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # point query certificate
    pk = recs[n // 2][0]
    pcert = h.certificate([h.query_point(pk)])
    tracemalloc.start()
    ok_pt = HybridIndex.verify_result(pcert, h.root)
    cur_pt, peak_pt = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    data_bytes = sum(len(k) + len(v) for k, v in recs)  # full scan needs this

    return {
        "workload": name, "n_records": n,
        "range_rows": rc["hi"] - rc["lo"],
        "verifier_input_bytes": root_bytes + cert_bytes,
        "verifier_peak_bytes": peak,
        "verifier_peak_point_bytes": peak_pt,
        "fullscan_input_bytes": data_bytes,
        "verifier_input_vs_fullscan": round((root_bytes + cert_bytes) / max(1, data_bytes), 4),
        "range_ok": ok, "point_ok": ok_pt,
    }


def measure_lineage_verify(name, n):
    """M: verifiable lineage (provenance graph). The prefix-matched records are
    contiguous, so the range proof certifies the graph is complete (no dropped
    node, no false edge). Honest accept; a dropped node, a duplicated node, and
    a tampered value are all rejected; proof is O(log n)."""
    recs = load_records(name, n)
    h = HybridIndex(recs)
    prefix = recs[0][0].split(":")[0] + ":"
    cert = h.lineage_certificate(prefix)
    honest = HybridIndex.verify_range(cert, h.root)
    sub, total_b, boundary_b = HybridIndex.range_proof_size(cert)

    drop = dict(cert); drop["records"] = cert["records"][:-1]
    drop["hi"] = cert["hi"] - 1
    dup = dict(cert); dup["records"] = cert["records"] + [cert["records"][0]]
    r0 = cert["records"][0]
    tamper = dict(cert)
    tamper["records"] = [(r0[0], r0[1] + "_X", r0[2])] + cert["records"][1:]

    _, lat = timed(lambda: h.query_lineage(prefix))
    t0 = time.perf_counter()
    HybridIndex.verify_range(cert, h.root)
    lat_verify = (time.perf_counter() - t0) * 1000

    return {
        "workload": name, "n_records": n, "prefix": prefix,
        "lineage_nodes": cert["hi"] - cert["lo"],
        "honest_accept": honest,
        "rejects_drop_node": not HybridIndex.verify_range(drop, h.root),
        "rejects_duplicate_node": not HybridIndex.verify_range(dup, h.root),
        "rejects_tamper": not HybridIndex.verify_range(tamper, h.root),
        "proof_subtrees": sub, "proof_boundary_bytes": boundary_b,
        "query_ms": round(lat, 3), "verify_ms": round(lat_verify, 3),
    }


def measure_concurrent(name, n, batch):
    """L: interleaved write+query (mixed) workload. Records are committed in
    batches; after each commit a range query is answered and verified. Correct
    verification must hold at every step (the index state is consistent with
    the committed root), and query latency under the write pressure is
    compared with a static (pre-built) index."""
    recs = load_records(name, n)
    h = HybridIndex(recs[:batch])
    q_lat = []
    ok_all = True
    steps = 0
    for s in range(batch, n, batch):
        h.append_batch(recs[s:s + batch])
        lo = max(0, len(h.records) // 2 - batch // 2)
        hi = min(len(h.records), lo + batch)
        k1, k2 = h.records[lo][0], h.records[hi - 1][0]
        rc = h.range_certificate(k1, k2)
        t0 = time.perf_counter()
        ok = HybridIndex.verify_range(rc, h.root)
        q_lat.append((time.perf_counter() - t0) * 1000)
        ok_all &= ok
        steps += 1

    # static baseline: pre-built index, same queries
    h2 = HybridIndex(recs)
    lo = max(0, n // 2 - batch // 2); hi = min(n, lo + batch)
    k1, k2 = h2.records[lo][0], h2.records[hi - 1][0]
    rc2 = h2.range_certificate(k1, k2)
    t0 = time.perf_counter()
    ok_static = HybridIndex.verify_range(rc2, h2.root)
    static_ms = (time.perf_counter() - t0) * 1000

    return {
        "workload": name, "n_records": n, "batch": batch, "steps": steps,
        "verify_ok_all_steps": ok_all,
        "avg_query_ms": round(sum(q_lat) / len(q_lat), 3),
        "max_query_ms": round(max(q_lat), 3),
        "static_query_ms": round(static_ms, 3),
        "static_ok": ok_static,
    }


def measure_highfidelity(n=10000, accounts=2000, rounds=10):
    """I': high-fidelity synthetic data (Ethereum-style tx logs + ERC-20 state).
    Runs the core mechanism experiments (aggregation, range-proof, versioned
    time-travel, verifiable lineage, mutable updates) on realistic structures
    to show the mechanism does NOT depend on a specific synthetic distribution."""
    from workloads import eth_logs, eth_state
    out = {"tx_logs": {}, "state": {}}

    # --- (e) Ethereum-style tx logs ---------------------------------------- #
    recs = eth_logs(n)
    h = HybridIndex(recs)
    k1, k2 = recs[n // 4][0], recs[3 * n // 4][0]
    spec = {"group": lambda k, v: k.split(":")[0], "init": lambda: 0,
            "step": lambda c, k, v: c + 1, "finalize": lambda c: c}
    R, covered = h.query_aggregate(k1, k2, spec["group"], spec["init"],
                                   spec["step"], spec["finalize"])
    cert = h.certificate(covered)
    agg_ok = HybridIndex.verify_aggregate(R, cert, h.root, spec["group"],
                                          spec["init"], spec["step"],
                                          spec["finalize"])
    rc = h.range_certificate(k1, k2)
    rp_honest = HybridIndex.verify_range(rc, h.root)
    drop = dict(rc); drop["records"] = rc["records"][:-1]; drop["hi"] = rc["hi"] - 1
    rp_drop_rej = not HybridIndex.verify_range(drop, h.root)
    # versioned (time-travel) over batched appends
    hv = HybridIndex(recs[:n // 10])
    for s in range(n // 10, n, n // 10):
        hv.append_batch(recs[s:s + n // 10])
    vj = max(1, hv.version_count() // 2)
    k = recs[vj * (n // 10) - 1][0]
    v_ok = HybridIndex.verify_result(
        hv.certificate_at([hv.query_point_at(k, vj)], vj), hv.root_at(vj))
    # lineage: per-contract provenance
    prefix = recs[0][0].split(":")[0] + ":"
    lc = h.lineage_certificate(prefix)
    lin_ok = HybridIndex.verify_range(lc, h.root)
    lsub, _, lboundary = HybridIndex.range_proof_size(lc)
    out["tx_logs"] = {
        "n": n, "contracts": len({r[0].split(":")[0] for r in recs}),
        "aggregate_ok": agg_ok, "groups": len(R),
        "range_proof_honest": rp_honest,
        "range_proof_drop_rejected": rp_drop_rej,
        "time_travel_ok": v_ok, "versions": hv.version_count(),
        "lineage_ok": lin_ok, "lineage_nodes": lc["hi"] - lc["lo"],
        "lineage_proof_subtrees": lsub,
        "lineage_proof_boundary_bytes": lboundary,
    }

    # --- (f) ERC-20-style mutable state ------------------------------------ #
    initial, batches = eth_state(accounts, rounds)
    hs = HybridIndex(initial)
    for b in batches:
        hs.apply_updates(b)
    acct = initial[0][0]
    v_old = max(1, hs.version_count() // 2)
    latest = hs.query_point(acct)
    old = hs.query_point_at(acct, v_old)
    latest_ok = HybridIndex.verify_result(hs.certificate([latest]), hs.root)
    tt_ok = HybridIndex.verify_result(hs.certificate_at([old], v_old), hs.root_at(v_old))
    replay_rej = not HybridIndex.verify_result(
        [(acct, old[1], hs.paths.get(acct, []))], hs.root)
    out["state"] = {
        "accounts": accounts, "versions": hs.version_count(),
        "latest_value_ok": latest_ok, "time_travel_ok": tt_ok,
        "update_replay_rejected": replay_rej,
    }
    return out


def measure_dynamic_consistency(n_accounts=2000, rounds=10, update_frac=0.5,
                                 delete_frac=0.2):
    """V: dynamic-state unified consistency. Combines time-travel (A), verifiable
    updates (J), and tri-state deletion (N) into ONE lifecycle-consistent model:

    * V1 single-key lifecycle: a key that is inserted, updated, DELETED, then
      RECOVERED --- for EVERY version its value (or absence) verifies against
      that version's root (current, historical, and deleted states all sound).
    * V2 all-version time-travel: for a sample of keys, every version's value
      or absence verifies against its own root.
    * V3 cross-version update sequence: a never-deleted key's value is correct
      at every version and consistent (monotone), each verifies against its root.
    * V4 concurrent-update conflict: a batch updating the SAME key twice to
      different values is detected as a conflict and resolved deterministically
      (last-write-wins), the committed value verifying against the root.
    * V5 deletion-version absence: a deleted key's non-membership proof verifies
      against every version root at which it is absent."""
    from workloads import mutable_state_tristate
    if _SOURCE == "real":
        initial, batches = ds.get_data_source(
            "real", path=_DATA_PATH).tristate_batches(n_accounts, name="evoting")
        h = HybridIndex(initial)
        for b in batches:
            h.apply_mutations(b)
        m = h.version_count()
        keysets = [set(s["keys"]) for s in h.history]

        def verify_at(k, v):
            if k in keysets[v - 1]:
                rec = h.query_point_at(k, v)
                return HybridIndex.verify_result(
                    h.certificate_at([rec], v), h.root_at(v))
            nm = h.nonmembership_proof_at(k, v)
            return nm is not None and HybridIndex.verify_nonmembership(
                nm, h.root_at(v))

        sample = [initial[i][0] for i in range(0, len(initial),
                                                max(1, len(initial) // 30))][:30]
        v2_ok = all(verify_at(k, v) for k in sample for v in range(1, m + 1))
        lc_key = next((k for k, _ in initial
                       if 1 < sum(1 for v in range(1, m + 1)
                                   if k in keysets[v - 1]) < m), None)
        v1_ok = (lc_key is not None and
                 all(verify_at(lc_key, v) for v in range(1, m + 1)))
        lk = next((k for k in sample if all(k in keysets[v - 1]
                                            for v in range(1, m + 1))), None)
        v3_ok = False
        if lk:
            seq = [h.query_point_at(lk, v)[1] for v in range(1, m + 1)]
            vals = [int(x) for x in seq if str(x).lstrip("-").isdigit()]
            v3_ok = (all(seq[i] == seq[i - 1] + 1 for i in range(1, len(seq))) and
                     all(verify_at(lk, v) for v in range(1, m + 1))) if len(vals) == m else False
        batch = [("set", "acct:00000000", "200"), ("set", "acct:00000000", "300")]
        seen, conflict = {}, False
        for op, k, *rest in batch:
            if k in seen and seen[k] != rest[0]:
                conflict = True
            seen[k] = rest[0]
        h2 = HybridIndex([("acct:00000000", "100")])
        h2.apply_mutations(batch)
        lww = h2.query_point("acct:00000000")
        v4_ok = HybridIndex.verify_result(h2.certificate([lww]), h2.root)
        deleted_keys = [k for k, _ in initial if k not in keysets[m - 1]]
        dk = deleted_keys[0] if deleted_keys else None
        v5_ok = (dk is not None and all(verify_at(dk, v) for v in range(1, m + 1)
                                        if dk not in keysets[v - 1]))
        v5_absent = sum(1 for v in range(1, m + 1)
                        if dk not in keysets[v - 1]) if dk else 0
        return {
            "accounts": len(initial), "rounds": len(batches), "versions": m,
            "lifecycle_key": lc_key,
            "lifecycle_present_versions": (sum(1 for v in range(1, m + 1)
                                               if lc_key in keysets[v - 1])
                                           if lc_key else 0),
            "V1_lifecycle_consistent": v1_ok,
            "V2_all_versions_time_travel_ok": v2_ok,
            "V3_update_sequence_consistent": v3_ok,
            "V4_conflict_detected": conflict,
            "V4_last_write_wins": lww[1] == "300",
            "V4_committed_value_ok": v4_ok,
            "V5_deleted_absent_versions": v5_absent,
            "V5_absence_proof_ok": v5_ok, "source": "real",
        }
    initial, batches = mutable_state_tristate(n_accounts, rounds, update_frac,
                                              delete_frac)
    h = HybridIndex(initial)
    for b in batches:
        h.apply_mutations(b)
    m = h.version_count()
    keysets = [set(s["keys"]) for s in h.history]

    def verify_at(k, v):
        if k in keysets[v - 1]:
            rec = h.query_point_at(k, v)
            return HybridIndex.verify_result(h.certificate_at([rec], v),
                                             h.root_at(v))
        nm = h.nonmembership_proof_at(k, v)
        return nm is not None and HybridIndex.verify_nonmembership(nm, h.root_at(v))

    # V1: full lifecycle (insert -> update -> delete -> recover)
    lc_key = None
    for k, _ in initial:
        pres = [v for v in range(1, m + 1) if k in keysets[v - 1]]
        if len(pres) >= 3 and pres[0] == 1 and            any(pres[i + 1] != pres[i] + 1 for i in range(len(pres) - 1)):
            lc_key = k
            break
    v1_ok = all(verify_at(lc_key, v) for v in range(1, m + 1)) if lc_key else False
    v1_present = sum(1 for v in range(1, m + 1) if lc_key in keysets[v - 1])

    # V2: all-version time-travel for a sample of keys
    sample = [h.history[0]["records"][i][0]
              for i in range(0, len(h.history[0]["records"]),
                             max(1, len(h.history[0]["records"]) // 30))][:30]
    v2_ok = all(verify_at(k, v) for k in sample for v in range(1, m + 1))

    # V3: cross-version update sequence (never-deleted key)
    lk = next(k for k in sample if all(k in keysets[v - 1]
                                       for v in range(1, m + 1)))
    seq = [(v, h.query_point_at(lk, v)[1]) for v in range(1, m + 1)]
    vals = [int(val) for _, val in seq]
    v3_ok = all(seq[i][0] == seq[i - 1][0] + 1 and vals[i] >= vals[i - 1]
                for i in range(1, len(seq))) and         all(verify_at(lk, v) for v in range(1, m + 1))

    # V4: concurrent-update conflict (same key, two values, last-write-wins)
    batch = [("set", "acct:00000000", "200"), ("set", "acct:00000000", "300")]
    seen, conflict = {}, False
    for op, k, *rest in batch:
        if k in seen and seen[k] != rest[0]:
            conflict = True
        seen[k] = rest[0]
    h2 = HybridIndex([("acct:00000000", "100")])
    h2.apply_mutations(batch)
    lww = h2.query_point("acct:00000000")
    v4_conflict_detected = conflict
    v4_last_write_wins = lww[1] == "300"
    v4_ok = HybridIndex.verify_result(h2.certificate([lww]), h2.root)

    # V5: deletion-version absence
    deleted_keys = [k for k, _ in initial if k not in keysets[m - 1]]
    dk = deleted_keys[0]
    # absence proof must verify at every version where the key is absent
    v5_ok = all(verify_at(dk, v) for v in range(1, m + 1)
                if dk not in keysets[v - 1]) if dk else True
    v5_absent = sum(1 for v in range(1, m + 1) if dk not in keysets[v - 1])

    return {
        "accounts": n_accounts, "rounds": rounds, "versions": m,
        "lifecycle_key": lc_key, "lifecycle_present_versions": v1_present,
        "V1_lifecycle_consistent": v1_ok,
        "V2_all_versions_time_travel_ok": v2_ok,
        "V3_update_sequence_consistent": v3_ok,
        "V4_conflict_detected": v4_conflict_detected,
        "V4_last_write_wins": v4_last_write_wins,
        "V4_committed_value_ok": v4_ok,
        "V5_deleted_absent_versions": v5_absent,
        "V5_absence_proof_ok": v5_ok,
    }


def measure_reorg(n=8000, batch=1000, finality_depth=2, fork_from=3,
                 name="supplychain"):
    """Chain fork / reorg handling (M5).

    The verifier anchors to the FINALIZED canonical root set rather than to any
    root an indexer supplies (trust bootstrap, M2). Scenarios:

      R1  honest certificate against the latest FINALIZED canonical root -> accept
      R2  certificate against a root on an ABANDONED fork (stale after reorg)
          -> rejected (root is not on the finalized canonical chain)
      R3  certificate against a NON-FINALIZED canonical root (inside the reorg
          window) -> rejected (could still be reorged)
      R4  certificate against an OLD FINALIZED root -> accepted (finalized roots
          are immutable across reorgs)
    """
    if _SOURCE == "real":
        init, batches, (fv, fork_batch) = ds.get_data_source(
            "real", path=_DATA_PATH).fork_batches(name, n)
        recs = init + [r for b in batches for r in b]
        nv = len(batches) + 1
        hA = HybridIndex(init)
        for b in batches:
            hA.append_batch(b)
        hB = HybridIndex(init)
        hB.append_batch(fork_batch)
        stale_root = hB.root  # abandoned branch, not on the canonical chain
    else:
        recs = load_records("supplychain", n)
        nv = n // batch  # number of committed versions
        # Branch A (canonical) and Branch B (abandoned) share the first
        # `fork_from` versions, then diverge. B commits a DIFFERENT batch ->
        # its root is stale.
        hA = HybridIndex(recs[:batch])
        for i in range(1, nv):
            hA.append_batch(recs[i * batch:(i + 1) * batch])
        hB = HybridIndex(recs[:batch])
        for i in range(1, fork_from):
            hB.append_batch(recs[i * batch:(i + 1) * batch])
        stale_batch = [(k + "_s", v)
                       for k, v in recs[fork_from * batch:(fork_from + 1) * batch]]
        hB.append_batch(stale_batch)
        stale_root = hB.root  # not on the canonical chain

    canonical_roots = [hA.history[v - 1]["root"] for v in range(1, nv + 1)]
    # finality: the newest `finality_depth` roots are NOT yet final
    finalized = set(canonical_roots[:max(0, nv - finality_depth)])
    latest_final = canonical_roots[max(0, nv - finality_depth - 1)]
    tip = canonical_roots[-1]
    old_final = canonical_roots[0]

    def anchored(cert, root):
        if root not in finalized:
            return False
        return HybridIndex.verify_range(cert, root)

    # R1: honest certificate at the latest finalized version
    vf = max(1, nv - finality_depth)
    cert = hA.range_certificate_at(recs[0][0], recs[-1][0], vf)
    r1_ok = anchored(cert, latest_final)
    r1_size = HybridIndex.range_proof_size(cert)

    # R2: certificate from the abandoned branch (stale root) -> rejected
    certB = hB.range_certificate_at(recs[0][0], recs[-1][0], fork_from + 1)
    r2_rejected = not anchored(certB, stale_root)
    r2_root_stale = stale_root not in finalized

    # R3: certificate against the non-finalized tip -> rejected
    certT = hA.range_certificate_at(recs[0][0], recs[-1][0], nv)
    r3_rejected = not anchored(certT, tip)
    r3_root_not_final = tip not in finalized

    # R4: certificate against an old finalized root -> accepted
    certO = hA.range_certificate_at(recs[0][0], recs[-1][0], 1)
    r4_ok = anchored(certO, old_final)

    return {
        "versions": nv, "finality_depth": finality_depth,
        "canonical_roots": nv, "abandoned_roots": 1,
        "R1_finalized_tip_accept": r1_ok,
        "R1_subtrees": r1_size[0], "R1_proof_bytes": r1_size[1],
        "R2_stale_fork_reject": r2_rejected, "R2_stale_root_detected": r2_root_stale,
        "R3_nonfinalized_tip_reject": r3_rejected, "R3_tip_not_final": r3_root_not_final,
        "R4_old_finalized_accept": r4_ok,
    }


def measure_compact_cert_curve(group_counts=(4, 8, 16, 32, 64),
                               n_list=(1000, 10000, 100000, 1000000)):
    """Compact subtree aggregation certificate size bound (M6).

    Vary the number of groups |G| and the dataset size n; report the canonical
    cover subtree count and certificate bytes. Empirically the certificate is
    O(log n x |G|): subtree count ~ O(log n) and per-subtree accumulator map
    ~ O(|G|), so total bytes ~ log n x |G| (vs O(n x |G|) for a per-record
    certificate).
    """
    rows = []
    for n in n_list:
        # synthetic records with a controlled group id: value = str(group id)
        G_probe = max(group_counts)
        recs = [(f"rec:{i:08d}", str(i % G_probe)) for i in range(n)]
        h = HybridIndex(recs)
        k1, k2 = recs[n // 4][0], recs[3 * n // 4][0]
        lo = bisect.bisect_left(h.keys, k1)
        hi = bisect.bisect_right(h.keys, k2)
        covered = hi - lo
        for G in group_counts:
            group_fn = lambda k, v, g=G: str(int(v) % g)  # reduce to g groups
            acc = {}
            R, cert = h.compact_aggregate_certificate(
                k1, k2, group_fn, lambda: 0, lambda c, k, v: c + 1,
                lambda a, b: a + b, lambda c: c)
            subtrees, cert_bytes = HybridIndex.compact_cert_size(cert)
            # per-record certificate baseline: ~32 B * log2(n) path per record
            logn = max(1, n.bit_length())
            per_rec_bytes = 32 * logn + 8
            rec_cert_bytes = covered * per_rec_bytes
            rows.append({
                "n": n, "groups": G, "covered": covered,
                "subtrees": subtrees, "cert_bytes": cert_bytes,
                "compression_x": round(rec_cert_bytes / max(1, cert_bytes), 1),
                "verify_ok": HybridIndex.verify_compact_aggregate(
                    R, cert, h.root, group_fn, lambda: 0,
                    lambda c, k, v: c + 1, lambda a, b: a + b, lambda c: c),
            })
    return rows


def run_all(workload, n):
    spec = AGGREGATES[workload]
    agg = measure_aggregate(workload, n)
    fresh = measure_freshness(workload, n, batch=max(100, n // 10))
    lineage = measure_lineage(workload, n)
    sel = measure_selectivity(workload, n)
    base = measure_baseline_compare(workload, n)
    return {"workload": workload, "n_records": n,
            "aggregate": agg, "freshness": fresh,
            "lineage": lineage, "selectivity": sel,
            "baseline_compare": base}


def _safe(label, fn):
    """Run ``fn``; on failure (e.g. a real-source adapter not wired to a CSV)
    return a clear error dict instead of aborting the whole harness."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - report, never crash the run
        return {"error": f"{type(e).__name__}: {e}", "source": _SOURCE}


def measure_storage_overhead(name="evoting", n=5000):
    """M4 (real storage engine) honest proxy.

    Quantifies how much disk the authenticated MST + index needs and the
    serialize / reload / query cost a real storage engine would add. Stdlib-only
    (pickle + temp file), so it runs offline. It does not replace a RocksDB /
    LevelDB / Postgres-index measurement, but gives an honest storage footprint
    and marshalling-overhead figure (the "storage amplification" baseline is
    ``index_bytes / raw_records_bytes``).
    """
    import os
    import pickle
    import tempfile

    from hybrid import HybridIndex
    recs = load_records(name, n)
    h = HybridIndex(recs)

    t0 = time.perf_counter()
    blob = pickle.dumps({"records": h.records, "levels": h.tree.level,
                         "root": h.root})
    ser_ms = (time.perf_counter() - t0) * 1000.0

    raw_bytes = sum(len(k.encode()) + len(v.encode()) for k, v in h.records)
    idx_bytes = len(blob)

    with tempfile.NamedTemporaryFile(suffix=".mst", delete=False) as fh:
        fh.write(blob)
        disk = fh.name
    t0 = time.perf_counter()
    with open(disk, "rb") as fh:
        loaded = pickle.load(fh)
    deser_ms = (time.perf_counter() - t0) * 1000.0
    os.unlink(disk)

    h2 = HybridIndex(loaded["records"])
    lo_k, hi_k = h2.records[0][0], h2.records[-1][0]
    t0 = time.perf_counter()
    h2.query_range(lo_k, hi_k)
    qms = (time.perf_counter() - t0) * 1000.0

    return {
        "records": len(h.records),
        "raw_records_bytes": raw_bytes,
        "index_bytes": idx_bytes,
        "storage_amp": (idx_bytes / raw_bytes) if raw_bytes else None,
        "serialize_ms": round(ser_ms, 3),
        "deserialize_ms": round(deser_ms, 3),
        "query_after_reload_ms": round(qms, 3),
        "reload_total_ms": round(ser_ms + deser_ms + qms, 3),
        "source": _SOURCE,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="evoting", choices=list(WORKLOADS))
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--all", action="store_true", help="run all 3 workloads")
    ap.add_argument("--scaling", action="store_true",
                    help="run the scaling sweep (default n_list)")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "real"],
                    help="synthetic traces (paper) or a real chain export")
    ap.add_argument("--data-path", default="",
                    help="CSV export for --source real (see 真实链数据导出字段规范.md)")
    args = ap.parse_args()

    global _SOURCE, _DATA_PATH
    _SOURCE = args.source
    _DATA_PATH = args.data_path

    workloads = list(WORKLOADS) if args.all else [args.workload]
    result = {}
    # Scaling sweep target sizes; when reading a real export, cap each point at
    # the actual row count so we never probe beyond the dataset.
    if _SOURCE == "real" and _DATA_PATH:
        avail = ds.get_data_source("real", path=_DATA_PATH).row_count() or args.n
        scaling_list = [m for m in (1000, 10000, 100000, 1000000) if m <= avail]
        if not scaling_list:
            scaling_list = [avail]
    else:
        scaling_list = [1000, 10000, 100000, 1000000]
    # Every measure is wrapped: under --source real a missing/partial CSV
    # (e.g. only a flat export, no versioned/fork export) degrades to a clear
    # error dict for that experiment instead of aborting the whole run.
    for w in workloads:
        result[w] = _safe("run_all", lambda: run_all(w, args.n))
    # scaling sweep (fixed |R| = 1000), up to 1e6
    result["scaling"] = {w: _safe("scaling",
                                  lambda: measure_scaling(w, scaling_list))
                         for w in workloads}
    # statistical adversarial soundness + incremental cost + query classes
    result["multi_seed"] = {w: _safe("multi_seed",
                                     lambda: measure_multi_seed(w, args.n, K=20))
                            for w in workloads}
    result["incremental"] = {w: _safe("incremental",
                                     lambda: measure_incremental(
                                         w, args.n,
                                         batch=max(100, args.n // 10)))
                             for w in workloads}
    result["query_classes"] = {w: _safe("query_classes",
                                        lambda: measure_query_classes(w, args.n))
                               for w in workloads}
    # A+B: versioned history (time-travel) + authenticated range-proof
    result["range_proof"] = {w: _safe("range_proof",
                                      lambda: measure_range_proof(w, args.n))
                             for w in workloads}
    result["range_proof_scaling"] = {
        w: _safe("range_proof_scaling",
                 lambda: [measure_range_proof(w, nn, frac=0.25) for nn in
                          [1000, 10000, 100000, 1000000]])
        for w in workloads}
    result["versioned"] = {w: _safe("versioned",
                                   lambda: measure_versioned(
                                       w, args.n, batch=max(100, args.n // 10)))
                           for w in workloads}
    # J/F/G: mutable state (updates) + extended attacks + verify-vs-|R| curve
    result["mutable_state"] = _safe(
        "mutable_state",
        lambda: measure_mutable(n_accounts=2000, rounds=10, update_frac=0.5))
    result["attacks"] = {w: _safe("attacks",
                                  lambda: measure_attacks(
                                      w, args.n, batch=max(100, args.n // 10)))
                         for w in workloads}
    result["verify_curve"] = {w: _safe("verify_curve",
                                       lambda: measure_verify_curve(w, args.n))
                              for w in workloads}
    # K/M/L: lightweight-client verification + verifiable lineage + mixed load
    result["portable_verify"] = {w: _safe("portable_verify",
                                          lambda: measure_portable_verify(w, args.n))
                                 for w in workloads}
    result["lineage_verify"] = {w: _safe("lineage_verify",
                                         lambda: measure_lineage_verify(w, args.n))
                                for w in workloads}
    result["concurrent"] = {w: _safe("concurrent",
                                     lambda: measure_concurrent(
                                         w, args.n, batch=max(100, args.n // 10)))
                            for w in workloads}
    # I': high-fidelity synthetic data (Ethereum-style) -- synthetic-only
    if _SOURCE == "real":
        result["highfidelity"] = {
            "note": "high-fidelity Ethereum-style traces are synthetic; skipped "
                    "under --source real. Use a real ERC-20 export with "
                    "--data-path for the flat/versioned measures instead.",
            "source": "real"}
    else:
        result["highfidelity"] = _safe(
            "highfidelity",
            lambda: measure_highfidelity(n=args.n, accounts=2000, rounds=10))
    # N: tri-state delete semantics (insert/update/delete/recover)
    result["delete_semantics"] = _safe(
        "delete_semantics",
        lambda: measure_delete_semantics(
            n_accounts=2000, rounds=10, update_frac=0.5, delete_frac=0.2))
    # O: batch verification amortization (individual vs range-proof batched)
    result["batch_verify"] = {w: _safe("batch_verify",
                                       lambda: measure_batch_verify(w, args.n))
                              for w in workloads}
    # AC+S: compact subtree aggregate certificate + coverage completeness
    result["compact_aggregate"] = {w: _safe("compact_aggregate",
                                            lambda: measure_compact_aggregate(w, args.n))
                                   for w in workloads}
    result["aggregate_coverage"] = {w: _safe("aggregate_coverage",
                                             lambda: measure_aggregate_coverage(w, args.n))
                                    for w in workloads}
    # X: multi-indexer quorum (Byzantine) verification
    result["quorum"] = {w: _safe("quorum",
                                 lambda: measure_quorum(w, args.n))
                        for w in workloads}
    # V: dynamic-state unified consistency (time-travel+update+delete)
    result["dynamic_consistency"] = _safe(
        "dynamic_consistency",
        lambda: measure_dynamic_consistency(
            n_accounts=2000, rounds=10, update_frac=0.5, delete_frac=0.2))
    # M5: chain fork / reorg handling (finalized-root anchor)
    result["reorg"] = _safe(
        "reorg",
        lambda: measure_reorg(n=8000, batch=1000, finality_depth=2, fork_from=3))
    # M6: compact aggregation certificate size bound (vs n and |G|)
    result["compact_cert_curve"] = measure_compact_cert_curve()
    result["_meta"] = {"source": _SOURCE, "data_path": _DATA_PATH,
                       "workloads": workloads, "n": args.n}
    out_path = OUT / "full_eval.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "source": _SOURCE, "workloads": workloads,
                      "n": args.n, "out": str(out_path)}))
    sys.stderr.write(f"[written] {out_path}\n")


if __name__ == "__main__":
    main()
