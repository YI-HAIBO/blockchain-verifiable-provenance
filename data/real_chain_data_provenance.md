# Real-Chain Data Provenance and Audit (M1 closure)

This note documents the source, schema, acquisition method and reproduction
command for the real on-chain dataset used in Section 4 (Real-chain
evaluation). All values come from Ethereum mainnet records; no synthetic or
generated values are included.

## Source

| Item | Value |
|---|---|
| Chain | Ethereum Mainnet (chain id 1) |
| Data type | ERC-20 Transfer(address,address,uint256) event logs |
| Event topic0 | 0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef |
| Primary RPC | https://eth.drpc.org |
| Fallback RPCs | https://1rpc.io/eth, https://eth-mainnet.public.blastapi.io, https://ethereum-rpc.publicnode.com |
| Block range | 26,390,983 - 26,390,992 (10 blocks) |
| Chain height at export | 26,391,103 |
| Export time | 2026-11-11 18:33:13 (GMT+8) |
| Records | 6,000 real Transfer events |
| Cardinality | 265 distinct token contracts, 2,590 distinct sender addresses |

Acquisition: eth_getLogs requests were issued in 5-block windows and filtered
to three-topic ERC-20 Transfer events, excluding four-topic ERC-721 events.
Each record was decoded into blockNumber, transactionHash, logIndex, contract
address, sender and recipient from topics[1] and topics[2], and the transfer
amount from data. Block timestamps were fetched separately with
eth_getBlockByNumber.

## Exported files

| File | Purpose |
|---|---|
| real_erc20_flat.csv | Schema A flat export (key,value), read directly by --source real |
| real_erc20_audit.csv | Per-row lineage: block_number, tx_hash, log_index, contract, from, to, value, ts |
| real_chain_meta.json | Machine-readable endpoints, block range, counts and export time |

Field semantics: key is erc20:<12-digit block>:<tx hash>:<6-digit log index>,
which is unique and lexicographically sortable. value is
<token contract>,<from>,<to>,<amount in token base units>, all taken verbatim
from chain data. The counting aggregator in run_full_eval.py groups by the
first segment of value, so the real-chain experiment measures transfer counts
grouped by token contract. The paper states this grouping explicitly and does
not describe it as supply-chain stages or voting candidates.

## Reproduction

    cd blockchain_bench
    python run_full_eval.py --source real --data-path data/real_erc20_flat.csv --n 5000 --workload supplychain

Results are written to results/full_eval.json with _meta.source = real. The
earlier synthetic run is preserved as results/full_eval.synthetic.bak.json.

## Measured results (real chain, n = 5,000)

| Metric | Value |
|---|---|
| Range aggregate coverage / groups | 2,501 records / 164 groups |
| Aggregate correctness / certificate verification | True / True |
| Aggregate / verify latency | 0.60 ms / 21.39 ms |
| Per-record certificate to compact certificate | 1,040,416 B to 5,624 B (185x, 13 subtrees) |
| Compact verification / aggregate agreement | 0.128 ms / True |
| Range proof | 10 canonical subtrees, 2,944 B, 1.861 ms; deletion, duplication and tampering all rejected |
| Adversarial robustness (K = 20) | false-accept max/avg = 0.0 / 0.0, 20/20 seeds passed |
| Replay attack | hybrid index 0.0 vs external library plus fingerprint 1.0 |
| Aggregate coverage completeness | subset and superset both rejected after cross-check |
| Batch verification amortisation | 2.46x - 6.16x (K = 10 - 5,000) |
| Quorum (3/5/7 servers) | X1-X5 all True, overhead 4.9x - 11.2x |
| Incremental sync | 8.6 ms vs full rebuild 14.7 ms |
| Light-client point verification peak | 816 B |

## Honest limitations

1. The export covers 10 consecutive blocks and 6,000 events, so the scaling
   experiment reaches only n = 1,000. The paper does not present this as a
   large-scale real dataset.
2. Time-travel, mutable-state, deletion and fork experiments require Schema
   B/C/D exports (version, op and branch fields). Only Schema A is provided
   here. Under a real-chain run those experiments degrade to explicit errors,
   and the paper reports them with synthetic traces, clearly labelled in the
   Scope and real-chain paragraphs.
3. Every number comes from a local execution. No baseline was estimated or
   filled in. Same-platform comparisons with vChain, GEM2-tree and MBFT were
   not reproduced and their table slots remain empty.
