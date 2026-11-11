# Verifiable Provenance Query over Blockchain Data — Prototype + Evaluation

Supports the ICDE 2028 reserve draft *Efficient Verifiable Provenance Query
Processing over Blockchain-Backed Databases* (Sections 3-4).

## Layout
| File | Purpose |
|---|---|
| `merkle.py` | Merkle tree / Merkleized State Tree (MST) with inclusion proofs |
| `hybrid.py` | Hybrid index (MST + off-chain ordered index + lazy verification) + baselines |
| `workloads.py` | e-voting / IoT time-series / supply-chain dataset generators |
| `adversarial.py` | Untrusted index server forgery tests (soundness) |
| `run_eval.py` | Evaluation harness; outputs latency/verification/storage/false-accept |
| `results/` | JSON output (auto-created) |

## Run (offline, stdlib only)
```bash
python run_eval.py --workload evoting     --n 2500
python run_eval.py --workload iot         --n 5000
python run_eval.py --workload supplychain --n 4000
```

## What it demonstrates (expected)
- **Hybrid index**: fast point/range/lineage queries + lazy verification that
  accepts an honest answer (soundness OK) and **rejects forged** results
  (false-accept = 0 under an adversarial index server).
- **Baselines**: full-chain scan is correct but slow; off-chain-only is fast
  but has **no certificate** so a forged answer is accepted (false-accept = 1).
- Metrics reported: per-query latency, verification time + certificate size,
  storage amplification, freshness (not measured in skeleton), false-accept.

## Notes
- Verifiable updates over mutable state use a tri-state change set
  (insert/update/delete) and are verified against the target version root; the
  `run_full_eval.py` harness measures dynamic-state consistency end to end.
- `run_full_eval.py` also covers scale (up to $n=10^6$), the authenticated
  range proof, compact subtree aggregation certificates, multi-indexer quorum
  verification, lightweight-client verification, lineage, mixed load, and
  high-fidelity Ethereum-style data.
- To use a real chain, plug a client that reads committed transactions into
  `HybridIndex` instead of the in-memory generator.
