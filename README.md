# Verifiable Provenance Query over Blockchain Data — Prototype + Evaluation

Supports the **ICDE 2027 Research Track, Round 2** submission *Efficient
Verifiable Provenance Query Processing over Blockchain-Backed Databases*
(Sections 3-4).

## Layout
| File | Purpose |
|---|---|
| `merkle.py` | Merkle tree / Merkleized State Tree (MST) with inclusion proofs |
| `hybrid.py` | Hybrid index (MST + off-chain ordered index + lazy verification) + baselines |
| `workloads.py` | e-voting / IoT time-series / supply-chain dataset generators |
| `adversarial.py` | Untrusted index server forgery tests (soundness) |
| `run_eval.py` | Evaluation harness; outputs latency/verification/storage/false-accept |
| `run_full_eval.py` | Full evaluation harness (all dimensions, 3-run mean ± SD) |
| `results/` | JSON output (auto-created) |
| `figures/` | Camera-ready figures (`fig_architecture.png`, `fig_cover.png`) + `generate_figures.py` |
| `data_source.py` | Pluggable data sources; `RealChainDataSource` auto-detects flat / ERC-20 / state / versioned / mutable / fork CSV schemas |
| `sota_harness.py` | M3 baseline harness; `run_naive_merkle` measures a naive full-path Merkle proof (25×–55× larger certificate than compact coverage); `run_verifiable_sota` reserves vChain / GEM²-tree / MBFT slots (never fabricates numbers) |
| `data/` | **Real Ethereum mainnet ERC-20 Transfer export** (flat CSV, per-row provenance audit CSV, source metadata) |

## Run (offline, stdlib only)
```bash
python run_eval.py --workload evoting     --n 2500
python run_eval.py --workload iot         --n 5000
python run_eval.py --workload supplychain --n 4000

# Full evaluation harness (all dimensions, 3-run mean ± SD, scaling to 1e6)
python run_full_eval.py --all --n 10000

# Real chain data (M1, already exported and measured)
python run_full_eval.py --source real --data-path data/real_erc20_flat.csv --n 5000 --workload supplychain
#   -> flat measures run on the CSV; versioned/mutable/fork experiments need the
#      CSV schemas in 真实链数据导出字段规范.md (they degrade to a clear error
#      dict until you provide them).
```

## What it demonstrates (expected)
- **Hybrid index**: fast point/range/lineage queries + lazy verification that
  accepts an honest answer (soundness OK) and **rejects forged** results
  (false-accept = 0 under an adversarial index server).
- **Baselines**: full-chain scan is correct but slow; off-chain-only is fast
  but has **no certificate** so a forged answer is accepted (false-accept = 1).
- Metrics reported: per-query latency, verification time + certificate size,
  storage amplification, freshness, false-accept, adversarial soundness, range
  proofs, compact certificates, quorum verification, and batch amortization.

## Notes
- Verifiable updates over mutable state use a tri-state change set
  (insert/update/delete) and are verified against the target version root; the
  `run_full_eval.py` harness measures dynamic-state consistency end to end.
- `run_full_eval.py` also covers scale (3-run mean ± SD, up to $n=10^6$), the
  authenticated range proof, compact subtree aggregation certificates, the
  **compact-certificate scaling curve** (`measure_compact_cert_curve`),
  **chain-reorganization (reorg) handling** (`measure_reorg`: finalized-anchor
  accept / abandoned-fork reject / non-finalized reject / old-finalized accept,
  all True), multi-indexer quorum verification, lightweight-client verification,
  lineage, mixed load, and high-fidelity Ethereum-style data.
- Figures are generated reproducibly: `python3 figures/generate_figures.py`
  (requires `matplotlib`), producing `figures/fig_architecture.png` and
  `figures/fig_cover.png` as referenced by the paper.
- To use a real chain, export it to a CSV per `真实链数据导出字段规范.md` and run
  `run_full_eval.py --source real --data-path <csv>`. The flat schema (A) covers
  every stateless experiment; versioned (B), mutable-op (C), and fork (D) schemas
  cover time-travel / reorg / mutable-state experiments. `data_source.py` detects
  the CSV flavour from its header and raises an actionable error otherwise — it
  never silently falls back to synthetic traces.
- **Prototype scope**: in-memory, pure-stdlib implementation (no storage engine,
  no disk I/O); sha256 hashing. It demonstrates the verifiable-query mechanisms
  and their adversarial guarantees, not a production indexing system.


## Real-chain dataset (M1)

A real on-chain dataset is included and has been measured end to end.

| Item | Value |
|---|---|
| Chain | Ethereum mainnet, ERC-20 `Transfer` logs |
| Blocks | 26,390,983–26,390,992 (chain height 26,391,103 at export) |
| Records | 6,000 events; 265 token contracts; 2,590 distinct senders |
| Exported | 2026-11-11 18:33:13 (GMT+8) |
| RPC | `https://eth.drpc.org` with fallbacks `1rpc.io/eth`, `eth-mainnet.public.blastapi.io`, `ethereum-rpc.publicnode.com` |
| Files | `data/real_erc20_flat.csv` (Schema A: `key,value`), `data/real_erc20_audit.csv` (block, tx hash, log index, contract, from, to, value, timestamp), `data/real_chain_meta.json` |

Each flat key is `erc20:<12-digit block>:<tx hash>:<log index>`; the value is
`<token contract>,<from>,<to>,<transfer amount in token base units>`. **No
synthetic or generated values are present.** Run:

```bash
python run_full_eval.py --source real --data-path data/real_erc20_flat.csv --n 5000 --workload supplychain
```

The counting aggregator groups by the first value segment, i.e. by real token
contract, so the real-chain experiment measures per-contract transfer counts.
`results/full_eval.json` currently holds this real-chain run
(`_meta.source = real`); the previous synthetic run is preserved as
`results/full_eval.synthetic.bak.json`.

Measured on real records (n=5,000, 2,501 covered): aggregation correct,
verify 21.39 ms, per-record certificate 1,040,416 B; compact certificate
5,624 B (185×) with identical aggregate; range proof 10 subtrees / 2,944 B;
false-accept 0.0 over K=20 variations; replay 0.0 (hybrid) vs 1.0 (external
DB); batch amortization 2.46×–6.16×; quorum 3/5/7 servers pass all five
checks. Versioned, mutable-state, deletion, and fork measures require
Schema B/C/D exports and are reported on synthetic traces in the paper.

See `data/真实链数据来源说明.md` for provenance and reproduction details.
