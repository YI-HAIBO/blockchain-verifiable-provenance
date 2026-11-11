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
                "call from_erc20_csv(path) / from_state_csv(path) / "
                "from_flat_csv(path), then return the result here. The hybrid "
                "index consumes the same (key, value) shape, so no change to "
                "HybridIndex is required."
            )
        # Auto-detect the CSV flavour from its header so a single --data-path
        # works for ERC-20 transfers, account-state snapshots, or a generic
        # (key, value) export.
        header = self._peek_header(self.path)
        hl = set(h.strip().lower() for h in header)
        if hl & {"block_number", "tx_hash", "from", "to"}:
            loaded = self.from_erc20_csv(self.path)
        elif hl & {"account", "balance"}:
            loaded = self.from_state_csv(self.path)
        else:
            loaded = self.from_flat_csv(self.path)
        return loaded[:max(1, n)]

    # ------------------------------------------------------------------
    # Generic (key, value) flat export. Header "key,value" is optional;
    # without a header the first column is the key, the second the value.
    # Used for every *stateless* measure (aggregate / lineage / selectivity /
    # baseline / scaling / range-proof / verify-curve / portable / batch /
    # compact / coverage / quorum / concurrent / attacks).
    # ------------------------------------------------------------------
    def from_flat_csv(self, path: str) -> List[Record]:
        recs: List[Record] = []
        with open(path, newline="") as fh:
            r = csv.reader(fh)
            header = next(r, None)
            if header and header[0].strip().lower() in ("key",):
                rows = r
            else:
                fh.seek(0)
                rows = csv.reader(fh)
            for row in rows:
                if len(row) >= 2 and row[0].strip():
                    recs.append((row[0].strip(), row[1].strip()))
        return self._sort(recs)

    # ------------------------------------------------------------------
    # Versioned export: columns `version,key,value` (version = positive int).
    # Grouped ascending; version 1 becomes the initial state, versions 2..k
    # become appended batches. Powers time-travel (measure_versioned) and
    # fork/reorg handling (measure_reorg).
    # ------------------------------------------------------------------
    def from_versioned_csv(self, path: str):
        versions: dict = {}
        with open(path, newline="") as fh:
            r = csv.reader(fh)
            header = next(r, None)
            rows = r
            if header and header[0].strip().lower() in ("version",):
                pass
            else:
                fh.seek(0)
                rows = csv.reader(fh)
            for row in rows:
                if len(row) < 3:
                    continue
                v = int(row[0].strip())
                versions.setdefault(v, []).append((row[1].strip(), row[2].strip()))
        if not versions:
            raise ValueError(
                "versioned CSV needs columns: version,key,value")
        maxv = max(versions)
        initial = sorted(versions.get(1, []))
        batches = [sorted(versions.get(v, [])) for v in range(2, maxv + 1)]
        return initial, batches

    # ------------------------------------------------------------------
    # Mutable operation log: columns `op,key,value,version`
    # (op in {set, delete, del}; version optional, defaults to row order).
    # For set-only semantics (measure_mutable) deletes are dropped; for
    # tri-state semantics (measure_delete_semantics / dynamic_consistency)
    # they are kept as ('del', k, '').
    # ------------------------------------------------------------------
    def from_mutable_csv(self, path: str, tristate: bool = False):
        per_version: dict = {}
        with open(path, newline="") as fh:
            r = csv.reader(fh)
            header = next(r, None)
            rows = r
            if header and header[0].strip().lower() in ("op",):
                pass
            else:
                fh.seek(0)
                rows = csv.reader(fh)
            for row in rows:
                if len(row) < 3:
                    continue
                op = row[0].strip().lower()
                k = row[1].strip()
                v = row[2].strip() if len(row) > 2 else ""
                ver = int(row[3].strip()) if len(row) > 3 and row[3].strip() else 1
                per_version.setdefault(ver, []).append((op, k, v))
        if not per_version:
            raise ValueError(
                "mutable CSV needs columns: op,key,value[,version] (op in "
                "set/delete)")
        maxv = max(per_version)
        if tristate:
            initial = [("set", k, v) for (op, k, v) in per_version.get(1, [])]
            batches = [[(self._norm_op(op), k, v)
                        for (op, k, v) in per_version.get(vv, [])]
                       for vv in range(2, maxv + 1)]
        else:
            initial = [(k, v) for (op, k, v) in per_version.get(1, [])
                       if op in ("set", "update")]
            batches = [[(k, v) for (op, k, v) in per_version.get(vv, [])
                        if op in ("set", "update")]
                       for vv in range(2, maxv + 1)]
        return initial, batches

    @staticmethod
    def _norm_op(op):
        op = op.lower()
        return "del" if op in ("delete", "del", "remove") else "set"

    # ------------------------------------------------------------------
    # Fork export for reorg: columns `version,key,value,branch`
    # (branch in {canonical, fork}). Returns (canonical_initial,
    # canonical_batches, fork_batch) where fork_batch is the divergent batch
    # on the abandoned branch.
    # ------------------------------------------------------------------
    def from_fork_csv(self, path: str):
        canon: dict = {}
        fork = None
        with open(path, newline="") as fh:
            r = csv.reader(fh)
            header = next(r, None)
            rows = r
            if header and header[0].strip().lower() in ("version",):
                pass
            else:
                fh.seek(0)
                rows = csv.reader(fh)
            for row in rows:
                if len(row) < 4:
                    continue
                v = int(row[0].strip()); k = row[1].strip(); val = row[2].strip()
                branch = row[3].strip().lower()
                if branch == "fork":
                    fork = (v, [(k, val)])
                else:
                    canon.setdefault(v, []).append((k, val))
        if fork is None:
            raise ValueError(
                "fork CSV needs at least one branch=fork row (version,key,"
                "value,branch)")
        if not canon:
            raise ValueError("fork CSV needs canonical rows")
        maxv = max(canon)
        initial = sorted(canon.get(1, []))
        batches = [sorted(canon.get(v, [])) for v in range(2, maxv + 1)]
        return initial, batches, fork

    # ---- specialized entry points used by run_full_eval.py ----------------
    def versioned_batches(self, name: str, n: int = 8000):
        if not self.path or not os.path.exists(self.path):
            raise NotImplementedError(
                "Real versioned data needs a CSV with columns "
                "version,key,value (one row per committed record). Provide it "
                "via --data-path. See 真实链数据导出字段规范.md.")
        return self.from_versioned_csv(self.path)

    def mutable_batches(self, n_accounts: int = 2000, name: str = "evoting"):
        if not self.path or not os.path.exists(self.path):
            raise NotImplementedError(
                "Real mutable data needs a CSV with columns "
                "op,key,value[,version] (op in set/delete). Provide it via "
                "--data-path. See 真实链数据导出字段规范.md.")
        return self.from_mutable_csv(self.path, tristate=False)

    def tristate_batches(self, n_accounts: int = 2000, name: str = "evoting"):
        if not self.path or not os.path.exists(self.path):
            raise NotImplementedError(
                "Real tri-state data needs a CSV with columns "
                "op,key,value[,version] (op in set/delete). Provide it via "
                "--data-path. See 真实链数据导出字段规范.md.")
        return self.from_mutable_csv(self.path, tristate=True)

    def fork_batches(self, name: str = "supplychain", n: int = 8000):
        if not self.path or not os.path.exists(self.path):
            raise NotImplementedError(
                "Real reorg data needs a CSV with columns "
                "version,key,value,branch (branch in canonical/fork). Provide "
                "it via --data-path. See 真实链数据导出字段规范.md.")
        return self.from_fork_csv(self.path)

    @staticmethod
    def _peek_header(path: str):
        with open(path, newline="") as fh:
            return next(csv.reader(fh), [])

    def row_count(self) -> int:
        """Number of data rows in the real export (used to bound scaling sweeps
        so a real dataset of size N is never probed beyond N)."""
        if not self.path or not os.path.exists(self.path):
            return 0
        with open(self.path, newline="") as fh:
            r = csv.reader(fh)
            header = next(r, None)
            rows = r
            if header and header[0].strip().lower() in (
                    "key", "version", "op", "block_number"):
                pass
            else:
                fh.seek(0)
                rows = csv.reader(fh)
            return sum(1 for _ in rows)


def get_data_source(kind: str = "synthetic", **kwargs) -> DataSource:
    """Factory used by the evaluation harness."""
    if kind == "synthetic":
        return SyntheticDataSource()
    if kind == "real":
        return RealChainDataSource(**kwargs)
    raise ValueError(f"unknown data source kind: {kind!r}")
