"""Workload generators: blockchain e-voting, IoT sensor time-series, and a
synthetic supply-chain dataset. Each yields records as (key, value) pairs;
the key is lexicographically sortable so range/time-series queries are
meaningful. Every generator accepts an explicit ``n`` and produces exactly
``n`` records with unique, sortable keys (so scaling sweeps work up to 1e5+).
"""
from __future__ import annotations


def _seed(n: int) -> int:
    import hashlib
    return int(hashlib.sha256(str(n).encode()).hexdigest(), 16)


def e_voting(n: int = 100000, candidates: int = 5, precincts: int = 20):
    """(a) Ballot records: key='ballot:<id>', value='precinct, candidate, ts'."""
    recs = []
    base_ts = 1700000000
    for bid in range(n):
        cand = bid % candidates
        precinct = bid % precincts
        ts = base_ts + bid * 7
        recs.append((f"ballot:{bid:08d}", f"{precinct},{cand},{ts}"))
    return recs


def iot(n: int = 100000, sensors: int = 50):
    """(b) IoT sensor readings: key='iot:<sensor>:<ts>', value='reading'.
    Timestamps are globally monotonic (per-minute), so freshness (lag to the
    chain tip) is well-defined. Readings are assigned round-robin over sensors
    so each sensor's records are contiguous in time."""
    recs = []
    base_ts = 1700000000
    per = (n + sensors - 1) // sensors
    for s in range(sensors):
        for r in range(per):
            idx = s * per + r
            if idx >= n:
                break
            ts = base_ts + idx * 60  # per-minute, globally monotonic
            val = (s * 31 + r * 17) % 100
            recs.append((f"iot:{s:03d}:{ts:012d}", str(val)))
    # n is exactly respected; keys are sorted by (sensor, ts).
    return sorted(recs)[:n]


def supply_chain(n: int = 100000, products: int = 60):
    """(c) Supply-chain events: key='sc:<product>:<ts>', value='stage,actor'.
    Timestamps are globally monotonic (per-hour), so freshness is well-defined."""
    recs = []
    base_ts = 1700000000
    stages = ["mfg", "ship", "customs", "dist", "retail"]
    per = (n + products - 1) // products
    for p in range(products):
        for e in range(per):
            idx = p * per + e
            if idx >= n:
                break
            ts = base_ts + idx * 3600  # hourly, globally monotonic
            stage = stages[e % len(stages)]
            actor = f"actor{(e % 8)}"
            recs.append((f"sc:{p:03d}:{ts:012d}", f"{stage},{actor}"))
    return sorted(recs)[:n]


def mutable_state(n_accounts: int = 2000, rounds: int = 10, update_frac: float = 0.5):
    """(d) Mutable account state (e.g., token balances / ERC-20): keys are
    accounts, values are balances that CHANGE over rounds. Returns
    ``(initial, batches)`` where ``initial`` is the version-1 state and
    ``batches[r]`` is the set of (account, new_balance) updates that produce
    version ``r+2``. Balances are deterministic (a base amount plus a
    round-dependent increment) so the generator is reproducible."""
    recs = []
    base = 1000
    for a in range(n_accounts):
        recs.append((f"acct:{a:08d}", str(base + (a % 97))))
    batches = []
    for r in range(rounds):
        batch = []
        # update a deterministic subset (update_frac) of accounts each round
        step = max(1, int(round(n_accounts * update_frac)))
        for a in range(0, n_accounts, max(1, n_accounts // step)):
            # balance grows by a round-dependent amount
            new_bal = base + (a % 97) + (r + 1) * ((a % 11) + 1)
            batch.append((f"acct:{a:08d}", str(new_bal)))
        batches.append(batch)
    return sorted(recs), batches


def mutable_state_tristate(n_accounts: int = 2000, rounds: int = 10,
                           update_frac: float = 0.5, delete_frac: float = 0.2):
    """(g) Tri-state mutable ledger: INSERT (initial) -> UPDATE -> DELETE ->
    RECOVER cycles. Returns ``(initial, batches)`` where each mutation is
    ``('set', k, v)`` or ``('del', k)``.

    A deleted account's key is REMOVED from the state tree, so a point query
    returns ``None`` and a non-membership proof certifies its absence. A
    recovered account is re-inserted with its EXACT pre-deletion value on a
    later round, so the delete is not permanent; the system can then
    distinguish a genuine re-insertion (the current value + current path verify
    against the latest root) from a pre-deletion replay (the old path no longer
    reconstructs the latest root even though the value is restored). Some
    accounts stay deleted at the final version (only half of a deleted set is
    recovered each odd round). Deterministic: the update/delete subsets are
    fixed strides, so each account's lifecycle is reproducible."""
    recs = []
    base = 1000
    for a in range(n_accounts):
        recs.append((f"acct:{a:08d}", str(base + (a % 97))))
    value = {a: base + (a % 97) for a in range(n_accounts)}
    batches = []
    deleted = set()
    for r in range(rounds):
        batch = []
        # update a deterministic subset of LIVE accounts (grows each round)
        step = max(1, int(round(n_accounts * update_frac)))
        for a in range(0, n_accounts, max(1, n_accounts // step)):
            if a not in deleted:
                value[a] = base + (a % 97) + (r + 1) * ((a % 11) + 1)
                batch.append(("set", f"acct:{a:08d}", str(value[a])))
        if r % 2 == 0:
            # delete a fresh subset on even rounds (never the same stride as
            # the update subset, so the lifecycle is non-trivial)
            dstep = max(1, int(round(n_accounts * delete_frac)))
            for a in range(0, n_accounts, max(1, n_accounts // dstep)):
                if a not in deleted:
                    batch.append(("del", f"acct:{a:08d}", ""))
                    deleted.add(a)
        else:
            # recover HALF of the currently-deleted accounts (odd rounds),
            # restoring the EXACT pre-deletion value -> isolates version-binding
            to_rec = sorted(deleted)[:max(1, len(deleted) // 2)]
            for a in to_rec:
                batch.append(("set", f"acct:{a:08d}", str(value[a])))
                deleted.discard(a)
        batches.append(batch)
    return sorted(recs), batches


# --------------------------------------------------------------------------- #
# High-fidelity synthetic data: Ethereum-style structures (realistic address
# formats, skewed value distributions, globally monotonic block heights).
# --------------------------------------------------------------------------- #
def _eth_addr(seed: int) -> str:
    import hashlib
    return "0x" + hashlib.sha256(str(seed).encode()).hexdigest()[:40]


def eth_logs(n: int = 100000, contracts: int = 64, events: int = 4):
    """(e) Ethereum-style transaction logs. key = '0x<contract>:<block>:<tx>:<log>',
    value = 'from,to,value,event'. Block height is globally monotonic (~50 logs
    per block); addresses are 20-byte hex; values follow a skewed (small-tx)
    distribution. The lineage prefix '0x<contract>:' yields per-contract
    provenance, matching how auditors trace a contract's event history."""
    import hashlib
    recs = []
    base_blk = 19000000
    for i in range(n):
        c = i % contracts
        caddr = _eth_addr(c + 1000)
        blk = base_blk + i // 50
        tx = i % 50
        f = _eth_addr((i * 7 + 3) % 100000)
        t = _eth_addr((i * 13 + 5) % 100000)
        val = (i % 1000) * (10 ** 15) + (i % 97)  # skewed, in wei
        ev = i % events
        recs.append((f"{caddr}:{blk:08d}:{tx:03d}:00",
                     f"{f},{t},{val},{ev}"))
    return sorted(recs)


def eth_state(n_accounts: int = 2000, rounds: int = 10, update_frac: float = 0.5):
    """(f) ERC-20-style holder balances (mutable). key = '0x<holder>', value =
    balance in wei, overwritten across rounds (transfers). Returns
    ``(initial, batches)``."""
    holders = [_eth_addr(a + 5000) for a in range(n_accounts)]
    initial = [(h, str(10 ** 18 + (a % 997) * 10 ** 15))
               for a, h in enumerate(holders)]
    batches = []
    for r in range(rounds):
        batch = []
        step = max(1, int(round(n_accounts * update_frac)))
        for a in range(0, n_accounts, max(1, n_accounts // step)):
            bal = 10 ** 18 + (a % 997) * 10 ** 15 + (r + 1) * ((a % 13) + 1) * 10 ** 12
            batch.append((holders[a], str(bal)))
        batches.append(batch)
    return sorted(initial), batches


WORKLOADS = {
    "evoting": e_voting,
    "iot": iot,
    "supplychain": supply_chain,
}
