"""Event-by-event fulfilment simulator for the flow_* cases (questions P181-P200).

Rules: data/public/HARD_TASK_RULES.md, section "Stateful fulfilment simulations".
Money is integer euro cents. Penalties and fees are charged per unit, as the rules
describe for late, unfilled and cancellation fees and for return fees.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "datasets"

PRIO = {"CANCEL": 0, "RETURN": 1, "RECEIPT": 2, "RELEASE": 3, "DISPATCH": 4}
STAGE = os.environ.get("PROJECT5_STAGE", "public")  # "public" = Stage 1; "final" = Stage 2
FINAL_DIR = ROOT / "problem7_student_final" / "datasets"


def _read(name: str) -> pd.DataFrame:
    if STAGE == "final":
        from src.stage2_schema import STAGE2_RENAMES
        df = pd.read_csv(FINAL_DIR / f"{name}.csv", dtype=str, keep_default_na=False)
        return df.rename(columns=STAGE2_RENAMES.get(name, {}))
    return pd.read_csv(DATA / f"{name}.csv", dtype=str, keep_default_na=False)


CASES = sorted(_read("flow_cases").case_id.unique()) if STAGE == "final" else [f"FLOW_I{i:02d}" for i in range(1, 11)]


@dataclass
class FlowCase:
    case_id: str
    horizon: int
    cutoff: int
    edits: list
    orders: pd.DataFrame
    stock: pd.DataFrame
    events: pd.DataFrame      # all versions for this case
    contracts: pd.DataFrame   # all versions for this case
    warnings: list = field(default_factory=list)


def load_case(case_id: str) -> FlowCase:
    cases = _read("flow_cases")
    row = cases[cases.case_id == case_id].iloc[0]
    edits = []
    for k in ["first", "second"]:
        ev, act, val = row[f"{k}_event_id"], row[f"{k}_action"], row[f"{k}_value"]
        if ev and act:
            edits.append((ev, act, int(val)))

    orders = _read("flow_orders")
    orders = orders[orders.case_id == case_id].copy()
    for c in ["release_step", "deadline_step", "units", "unit_price_cents", "unit_cost_cents",
              "late_penalty_cents", "unfilled_penalty_cents", "cancellation_fee_cents", "return_fee_cents"]:
        orders[c] = orders[c].astype(int)

    stock = _read("flow_stock")
    stock = stock[stock.case_id == case_id].copy()
    stock["opening_units"] = stock["opening_units"].astype(int)

    events = _read("flow_events")
    events = events[events.case_id == case_id].copy()
    events["recorded_step"] = events["recorded_step"].astype(int)
    events["effective_step"] = events["effective_step"].astype(int)
    events["units"] = events["units"].replace("", "0").astype(int)
    events["restock"] = events["restock"].str.lower().eq("true")

    contracts = _read("flow_contracts")
    contracts = contracts[contracts.case_id == case_id].copy()
    contracts["recorded_step"] = contracts["recorded_step"].astype(int)
    contracts["priority"] = contracts["priority"].astype(int)
    contracts["rebate_threshold_units"] = contracts["rebate_threshold_units"].replace("", "0").astype(int)
    contracts["rebate_per_unit_cents"] = contracts["rebate_per_unit_cents"].replace("", "0").astype(int)

    return FlowCase(case_id, int(row.horizon_step), int(row.knowledge_cutoff_step), edits,
                    orders, stock, events, contracts)


def select_known(df: pd.DataFrame, key: str, cutoff: int) -> pd.DataFrame:
    """Keep, per key, the version with the greatest (recorded_step, version_id) among those recorded by the cutoff."""
    known = df[df.recorded_step <= cutoff].copy()
    known = known.sort_values([key, "recorded_step", "version_id"])
    return known.groupby(key, as_index=False).tail(1).copy()


def run(case: FlowCase, edits=()) -> dict:
    """Simulate one case. `edits` is a list of (event_id, action, value) applied to selected versions."""
    ev = select_known(case.events, "event_id", case.cutoff)
    ev = ev[ev.status != "void"].copy()  # a selected void version removes the event
    for event_id, action, value in edits:
        m = ev.event_id == event_id
        if m.sum() == 0:
            continue
        if action == "SHIFT":
            ev.loc[m, "effective_step"] = ev.loc[m, "effective_step"] + value
        elif action == "SET_UNITS":
            ev.loc[m, "units"] = value
        else:
            raise ValueError(f"unknown action {action}")
    ev = ev[ev.effective_step <= case.horizon].copy()

    con = select_known(case.contracts, "customer_id", case.cutoff)
    contract = {}
    for _, r in con.iterrows():
        if r.status == "inactive":
            contract[r.customer_id] = (99, None, 0)
        else:
            contract[r.customer_id] = (r.priority, r.rebate_threshold_units, r.rebate_per_unit_cents)

    orders = {}
    for _, o in case.orders.iterrows():
        cp = contract.get(o.customer_id)
        if cp is None:
            case.warnings.append(f"no contract for customer {o.customer_id}")
            cp = (99, None, 0)
        orders[o.order_id] = {
            "product": o.product_id, "wh": o.warehouse, "release": o.release_step, "deadline": o.deadline_step,
            "units": o.units, "price": o.unit_price_cents, "cost": o.unit_cost_cents,
            "late_pen": o.late_penalty_cents, "unfilled_pen": o.unfilled_penalty_cents,
            "cancel_fee": o.cancellation_fee_cents, "return_fee": o.return_fee_cents,
            "customer": o.customer_id, "priority": cp[0],
            "released": False, "st": ["U"] * o.units, "disp": [None] * o.units,
            "late": [False] * o.units, "restocked": [False] * o.units,
            "cancelled": 0, "returned": 0, "restock_n": 0,
        }

    free = {}
    for _, s in case.stock.iterrows():
        free[(s.product_id, s.warehouse)] = free.get((s.product_id, s.warehouse), 0) + s.opening_units

    actions = []
    for oid, o in orders.items():
        if case.orders.loc[case.orders.order_id == oid, "release_step"].iloc[0] <= case.horizon:
            actions.append((int(orders[oid]["release"]), PRIO["RELEASE"], oid, "RELEASE", None))
    for _, e in ev.iterrows():
        actions.append((int(e.effective_step), PRIO[e.event_type], e.event_id, "EVENT", e))
    actions.sort(key=lambda a: (a[0], a[1], a[2]))

    dispatch_counter = 0

    def queue_key(oid):
        o = orders[oid]
        return (o["priority"], o["deadline"], o["release"], oid)

    def allocate():
        for key in list(free.keys()):
            if free[key] <= 0:
                continue
            cands = [oid for oid, o in orders.items()
                     if o["released"] and (o["product"], o["wh"]) == key and "U" in o["st"]]
            for oid in sorted(cands, key=queue_key):
                o = orders[oid]
                for i, s in enumerate(o["st"]):
                    if free[key] <= 0:
                        break
                    if s == "U":
                        o["st"][i] = "R"
                        free[key] -= 1
                if free[key] <= 0:
                    break

    stats = {"dispatched_units": 0, "late_units": 0, "receipts": 0}
    for step, prio, tie, kind, e in actions:
        if kind == "RELEASE":
            orders[tie]["released"] = True
        elif e.event_type == "RECEIPT":
            key = (e.product_id, e.warehouse)
            free[key] = free.get(key, 0) + int(e.units)
            stats["receipts"] = stats.get("receipts", 0) + int(e.units)
        elif e.event_type == "CANCEL":
            o = orders.get(e.order_id)
            if o and o["released"]:
                n = int(e.units)
                for i, s in enumerate(o["st"]):
                    if n <= 0:
                        break
                    if s in ("U", "R"):
                        if s == "R":
                            free[(o["product"], o["wh"])] += 1
                        o["st"][i] = "C"
                        o["cancelled"] += 1
                        n -= 1
        elif e.event_type == "RETURN":
            o = orders.get(e.order_id)
            if o and o["released"]:
                n = int(e.units)
                cands = [(o["disp"][i], i) for i, s in enumerate(o["st"]) if s == "D"]
                cands.sort(reverse=True)  # latest dispatch first, then highest sequence first
                for _, i in cands:
                    if n <= 0:
                        break
                    o["st"][i] = "T"
                    o["returned"] += 1
                    n -= 1
                    if e.restock:
                        free[(o["product"], o["wh"])] += 1
                        o["restocked"][i] = True
                        o["restock_n"] += 1
        elif e.event_type == "DISPATCH":
            cap = int(e.units)
            dispatch_counter += 1
            queue = [oid for oid, o in orders.items()
                     if o["wh"] == e.warehouse and "R" in o["st"]]
            for oid in sorted(queue, key=queue_key):
                o = orders[oid]
                for i, s in enumerate(o["st"]):
                    if cap <= 0:
                        break
                    if s == "R":
                        o["st"][i] = "D"
                        o["disp"][i] = dispatch_counter
                        if step > o["deadline"]:
                            o["late"][i] = True
                            stats["late_units"] += 1
                        stats["dispatched_units"] += 1
                        cap -= 1
        allocate()

    totals = {"revenue": 0, "cost": 0, "recovery": 0, "late": 0, "unfilled": 0,
              "cancel_fee": 0, "return_fee": 0, "rebate": 0}
    retained_by_customer = {}
    for oid, o in orders.items():
        dispatched = sum(1 for s in o["st"] if s in ("D", "T"))
        returned = o["returned"]
        retained = dispatched - returned
        totals["revenue"] += retained * o["price"]
        totals["cost"] += dispatched * o["cost"]
        totals["recovery"] += o["restock_n"] * o["cost"]
        totals["late"] += sum(o["late"]) * o["late_pen"]
        if o["released"]:
            open_units = sum(1 for s in o["st"] if s in ("U", "R"))
            totals["unfilled"] += open_units * o["unfilled_pen"]
        totals["cancel_fee"] += o["cancelled"] * o["cancel_fee"]
        totals["return_fee"] += returned * o["return_fee"]
        retained_by_customer[o["customer"]] = retained_by_customer.get(o["customer"], 0) + retained

    for cust, retained in retained_by_customer.items():
        cp = contract.get(cust)
        if cp is None or cp[1] is None:
            continue
        priority, threshold, rate = cp
        if retained >= threshold:
            totals["rebate"] += retained * rate

    contribution = (totals["revenue"] - totals["cost"] + totals["recovery"] - totals["late"]
                    - totals["unfilled"] + totals["cancel_fee"] - totals["return_fee"] - totals["rebate"])

    # Invariant checks: every unit is in exactly one state; free stock never negative.
    problems = []
    for oid, o in orders.items():
        if len(o["st"]) != o["units"]:
            problems.append(f"{oid}: unit count changed")
    for key, v in free.items():
        if v < 0:
            problems.append(f"negative free stock {key}")

    stats["opening"] = int(case.stock.opening_units.sum())
    stats["restocked"] = sum(o["restock_n"] for o in orders.values())
    stats["free_end"] = sum(free.values())
    stats["reserved_end"] = sum(sum(1 for s in o["st"] if s == "R") for o in orders.values())
    stats["dispatch_ever"] = sum(sum(1 for s in o["st"] if s in ("D", "T")) for o in orders.values())
    return {"contribution_cents": int(contribution), "totals": totals, "stats": stats,
            "problems": problems, "warnings": case.warnings}


def solve_case(case_id: str) -> dict:
    case = load_case(case_id)
    base = run(case)
    what_if = run(case, edits=case.edits) if case.edits else base
    return {"case_id": case_id, "baseline_cents": base["contribution_cents"],
            "what_if_cents": what_if["contribution_cents"], "edits": case.edits,
            "problems": base["problems"] + what_if["problems"],
            "warnings": sorted(set(base["warnings"] + what_if["warnings"]))}


def run_all() -> pd.DataFrame:
    return pd.DataFrame([solve_case(c) for c in CASES])


def flow_answers() -> dict:
    df = run_all().set_index("case_id")
    out = {}
    for i, c in enumerate(CASES):
        base = df.loc[c, "baseline_cents"] / 100
        change = (df.loc[c, "what_if_cents"] - df.loc[c, "baseline_cents"]) / 100
        out[f"P{181 + 2 * i:03d}"] = f"{base:.2f}"
        out[f"P{182 + 2 * i:03d}"] = f"{change:.2f}"
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 120)
    df = run_all()
    print(df.to_string(index=False))
