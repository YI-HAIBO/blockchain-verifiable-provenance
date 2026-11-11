"""Pluggable data sources for the verifiable-provenance benchmark.

The hybrid index consumes records as a list of ``(key, value)`` tuples where
``key`` is a lexicographically sortable string. This module provides:

* :class:`DataSource` - the abstract interface every source must satisfy.
* :class:`SyntheticDataSource` - wraps the high-fidelity synthetic generators in
  ``workloads.py`` (e-voting / IoT / supply-chain). This is what every number
  in the paper currently uses.
* :class:`RealChainDataSource` - a *stub* that shows exactly where a real
  blockchain dataset plugs in. It is intentionally not implemented end-to-end
  because it needs a real chain export (an RPC endpoint, an ETL CSV, or a
  snapshot); the adapter shape is already correct, so M1 ("use a real chain
  dataset") is *one data drop away*.

Switching sources in ``run_full_eval.py`` only requires importing
``get_data_source`` and calling ``.records(n)`` / ``.batches(...)`` instead of
the ``workloads.*`` generators.
"""
from __future__ import annotations

import csv
import os
from typing import List, Tuple

from workloads import e_voting, iot, supply_chain, mutable_state

Record = Tuple[str, str]


class DataSource:
    """Abstract data source.

    A source must produce the same ``(key, value)`` record shape the hybrid
    index expects, with keys sorted lexicographically so that range / time-series
    and lineage queries are well defined.
    """

    kind = "abstract"

    def records(self, n: int, workload: str = "evoting") -> List[Record]:
        raise NotImplementedError

    def batches(self, n_accounts: int = 2000, rounds: int = 10):
        raise NotImplementedError

    @staticmethod
    def _sort(recs: List[Record]) -> List[Record]:
        return sorted(recs)


class SyntheticDataSource(DataSource):
    """High-fidelity synthetic traces (used for every result in the paper)."""

    kind = "synthetic"

    _MAP = {
        "evoting": e_voting,
        "iot": iot,
        "supply_chain": supply_chain,
        "supply-chain": supply_chain,
    }

    def records(self, n: int, workload: str = "evoting") -> List[Record]:
        gen = self._MAP.get(workload)
        if gen is None:
            raise ValueError(f"unknown synthetic workload: {workload!r}")
        return gen(n)

    def batches(self, n_accounts: int = 2000, rounds: int = 10):
        return mutable_state(n_accounts=n_accounts, rounds=rounds)


class RealChainDataSource(DataSource):
    """Stub for a *real* blockchain dataset.

    M1 (reviewer asks for at least one real chain dataset) is satisfied by
    implementing ``records`` here against a real export. The adapter contract is
    fixed: return ``(key, value)`` tuples with sortable keys. Two ready-made
    adapters are sketched below (ERC-20 transfers, generic block/state export).

    To enable: point ``path`` at a CSV/JSON export and implement the parser.
    Until then, calling ``records`` raises a clear, actionable error instead of
    silently returning synthetic data.
    """

    kind = "real"

    def __init__(self, path: str = "", chain: str = "ethereum"):
        self.path = path
        self.chain = chain

    # ------------------------------------------------------------------
    # Example adapter: ERC-20 Transfer logs.
    # Expected CSV columns: block_number, tx_hash, from, to, value, ts
    # (ts = unix seconds). We map each transfer to a sortable key so that
    # time-series / range queries over a token ledger are meaningful.
    # ------------------------------------------------------------------
    def from_erc20_csv(self, path: str) -> List[Record]:
        recs: List[Record] = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                # key = token event position, sortable by (block, log index)
                key = f"erc20:{int(row['block_number']):012d}:{row['tx_hash']}"
                value = f"{row['from']},{row['to']},{row['value']}"
                recs.append((key, value))
        return self._sort(recs)

    # ------------------------------------------------------------------
    # Example adapter: generic committed-state snapshot.
    # Expected CSV columns: account, balance, version, ts
    # ------------------------------------------------------------------
    def from_state_csv(self, path: str) -> List[Record]:
        recs: List[Record] = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                key = f"acct:{row['account']}"
                value = f"{row['balance']},{row['version']}"
                recs.append((key, value))
        return self._sort(recs)

    def records(self, n: int, workload: str = "evoting") -> List[Record]:
        if not self.path or not os.path.exists(self.path):
            raise NotImplementedError(
                "RealChainDataSource.records is not wired to a dataset yet. "
                "Provide a real chain export (e.g. an ERC-20 Transfer CSV) and "
                "call from_erc20_csv(path) / from_state_csv(path), then return "
                "the result here. The hybrid index consumes the same "
                "(key, value) shape, so no change to HybridIndex is required."
            )
        # Default: assume an ERC-20-style export; swap to from_state_csv for
        # account-state workloads.
        return self.from_erc20_csv(self.path)[:n]

    def batches(self, n_accounts: int = 2000, rounds: int = 10):
        raise NotImplementedError(
            "Real mutable-state batches require a real versioned state export. "
            "Reuse from_state_csv across version snapshots to build batches."
        )


def get_data_source(kind: str = "synthetic", **kwargs) -> DataSource:
    """Factory used by the evaluation harness."""
    if kind == "synthetic":
        return SyntheticDataSource()
    if kind == "real":
        return RealChainDataSource(**kwargs)
    raise ValueError(f"unknown data source kind: {kind!r}")
