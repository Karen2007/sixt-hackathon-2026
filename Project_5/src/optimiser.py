"""Exact fulfilment optimiser for the hard_opt_* cases (questions P151-P180).

Rules are from data/public/HARD_TASK_RULES.md and business_rules.md (coupled
fulfilment). Each job is either filled by exactly one feasible option or left
unfilled. The model is a mixed-integer programme solved to proven optimality
with CBC. All money is integer euro cents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pulp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "datasets"


STAGE = os.environ.get("PROJECT5_STAGE", "public")  # "public" = Stage 1; "final" = Stage 2
FINAL_DIR = ROOT / "problem7_student_final" / "datasets"


def _read(name: str) -> pd.DataFrame:
    if STAGE == "final":
        from src.stage2_schema import STAGE2_RENAMES
        df = pd.read_csv(FINAL_DIR / f"{name}.csv", dtype=str, keep_default_na=False)
        return df.rename(columns=STAGE2_RENAMES.get(name, {}))
    return pd.read_csv(DATA / f"{name}.csv", dtype=str, keep_default_na=False)


@dataclass
class CaseData:
    case_id: str
    jobs: pd.DataFrame
    options: pd.DataFrame
    stock: pd.DataFrame
    dispatch: pd.DataFrame
    suppliers: pd.DataFrame
    changes: pd.DataFrame
    warnings: list = field(default_factory=list)


def load_case(case_id: str) -> CaseData:
    def pick(name):
        df = _read(name)
        return df[df.case_id == case_id].copy()

    jobs = pick("hard_opt_jobs")
    for c in ["quantity", "ready_day", "due_day", "last_day", "late_penalty_cents_per_unit_day", "unfilled_penalty_cents"]:
        jobs[c] = jobs[c].astype(int)
    options = pick("hard_opt_options")
    for c in ["dispatch_day", "variable_cost_cents"]:
        options[c] = options[c].astype(int)
    stock = pick("hard_opt_stock")
    stock["available_units"] = stock["available_units"].astype(int)
    dispatch = pick("hard_opt_dispatch")
    dispatch["capacity_units"] = dispatch["capacity_units"].astype(int)
    dispatch["dispatch_day"] = dispatch["dispatch_day"].astype(int)
    sup = pick("hard_opt_suppliers")
    for c in ["lead_days", "activation_cost_cents", "minimum_units", "maximum_units", "active"]:
        sup[c] = sup[c].astype(int)
    changes = pick("hard_opt_changes")
    return CaseData(case_id, jobs, options, stock, dispatch, sup, changes)


def apply_changes(case: CaseData) -> CaseData:
    """Apply every change row simultaneously as an exact field replacement."""
    stock = case.stock.copy()
    disp = case.dispatch.copy()
    sup = case.suppliers.copy()
    for _, ch in case.changes.iterrows():
        val = ch["new_value"]
        if ch["table_name"] == "stock":
            m = (stock.warehouse_region == ch["warehouse_region"]) & (stock.product_id == ch["product_id"])
            if m.sum() != 1:
                case.warnings.append(f"{ch['change_id']}: stock target matched {int(m.sum())} rows")
            stock.loc[m, ch["field"]] = int(val)
        elif ch["table_name"] == "dispatch":
            m = (disp.warehouse_region == ch["warehouse_region"]) & (disp.dispatch_day.astype(int) == int(ch["dispatch_day"]))
            if m.sum() != 1:
                case.warnings.append(f"{ch['change_id']}: dispatch target matched {int(m.sum())} rows")
            disp.loc[m, ch["field"]] = int(val)
        elif ch["table_name"] == "suppliers":
            m = sup.supplier_id == ch["supplier_id"]
            if m.sum() != 1:
                case.warnings.append(f"{ch['change_id']}: supplier target matched {int(m.sum())} rows")
            sup.loc[m, ch["field"]] = int(val)
        else:
            case.warnings.append(f"{ch['change_id']}: unknown table {ch['table_name']}")
    out = CaseData(case.case_id, case.jobs, case.options, stock, disp, sup, case.changes, list(case.warnings))
    return out


def solve(case: CaseData, return_plan: bool = False):
    """Minimum total cost in cents and the solver status. Optionally the chosen plan."""
    jobs = case.jobs.set_index("job_id")
    stock = case.stock.set_index(["warehouse_region", "product_id"])["available_units"]
    cap = case.dispatch.set_index(["warehouse_region", "dispatch_day"])["capacity_units"]
    sup = case.suppliers.set_index("supplier_id")

    opts = case.options.copy()
    opts = opts.merge(sup[["lead_days", "activation_cost_cents", "minimum_units", "maximum_units", "active"]],
                      left_on="supplier_id", right_index=True, how="left", suffixes=("", "_s"))
    opts["qty"] = opts.job_id.map(jobs["quantity"])
    opts["ready"] = opts.job_id.map(jobs["ready_day"])
    opts["due"] = opts.job_id.map(jobs["due_day"])
    opts["last"] = opts.job_id.map(jobs["last_day"])
    opts["late_pen"] = opts.job_id.map(jobs["late_penalty_cents_per_unit_day"])
    opts["product"] = opts.job_id.map(jobs["product_id"])
    opts["arrival"] = opts.dispatch_day + opts.lead_days
    feasible = (
        (opts.dispatch_day >= opts.ready)
        & (opts.arrival <= opts["last"])
        & (opts.active == 1)
        & opts.lead_days.notna()
    )
    opts = opts[feasible].copy()
    opts["late_units"] = (opts.arrival - opts.due).clip(lower=0)
    opts["cost"] = opts.variable_cost_cents + opts.qty * opts.late_pen * opts.late_units

    prob = pulp.LpProblem("fulfilment", pulp.LpMinimize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in opts.index}
    u = {j: pulp.LpVariable(f"u_{j}", cat="Binary") for j in jobs.index}
    sup_ids = sorted(opts.supplier_id.unique())
    z = {s: pulp.LpVariable(f"z_{i}", cat="Binary") for i, s in enumerate(sup_ids)}

    prob += (
        pulp.lpSum(float(r.cost) * x[i] for i, r in opts.iterrows())
        + pulp.lpSum(float(jobs.loc[j, "unfilled_penalty_cents"]) * u[j] for j in jobs.index)
        + pulp.lpSum(float(sup.loc[s, "activation_cost_cents"]) * z[s] for s in sup_ids)
    )

    for j in jobs.index:
        mine = opts.index[opts.job_id == j]
        prob += pulp.lpSum(x[i] for i in mine) + u[j] == 1

    for (w, p), avail in stock.items():
        mine = opts.index[(opts.warehouse_region == w) & (opts["product"] == p)]
        if len(mine):
            prob += pulp.lpSum(float(opts.at[i, "qty"]) * x[i] for i in mine) <= float(avail)

    for (w, d), capacity in cap.items():
        mine = opts.index[(opts.warehouse_region == w) & (opts.dispatch_day == d)]
        if len(mine):
            prob += pulp.lpSum(float(opts.at[i, "qty"]) * x[i] for i in mine) <= float(capacity)

    for s in sup_ids:
        mine = opts.index[opts.supplier_id == s]
        total = pulp.lpSum(float(opts.at[i, "qty"]) * x[i] for i in mine)
        mn = float(sup.loc[s, "minimum_units"])
        mx = float(sup.loc[s, "maximum_units"])
        prob += total <= mx * z[s]
        prob += total >= mn * z[s]
        for i in mine:
            prob += x[i] <= z[s]

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    st = pulp.LpStatus[status]
    if st != "Optimal":
        raise RuntimeError(f"solver status {st} for {case.case_id}")
    cost = int(round(pulp.value(prob.objective)))
    if not return_plan:
        return cost, st
    chosen = {jid: None for jid in jobs.index}
    for i in opts.index:
        if x[i].value() and x[i].value() > 0.5:
            chosen[opts.at[i, "job_id"]] = opts.at[i, "option_id"]
    return cost, st, chosen, opts


def verify_plan(case: CaseData, chosen: dict, opts: pd.DataFrame) -> list:
    """Independent check: recompute the plan's cost and list any rule violations."""
    jobs = case.jobs.set_index("job_id")
    sup = case.suppliers.set_index("supplier_id")
    stock = case.stock.set_index(["warehouse_region", "product_id"])["available_units"]
    cap = case.dispatch.assign(dispatch_day=case.dispatch.dispatch_day.astype(int)).set_index(
        ["warehouse_region", "dispatch_day"])["capacity_units"]
    problems, total = [], 0
    stock_used, cap_used, sup_units, used_sup = {}, {}, {}, set()
    by_id = case.options.set_index("option_id")
    for jid, oid in chosen.items():
        j = jobs.loc[jid]
        if oid is None:
            total += int(j.unfilled_penalty_cents)
            continue
        o = by_id.loc[oid]
        s = sup.loc[o.supplier_id]
        dispatch, arrival = int(o.dispatch_day), int(o.dispatch_day) + int(s.lead_days)
        if dispatch < int(j.ready_day) or arrival > int(j.last_day) or int(s.active) != 1:
            problems.append(f"{jid}: infeasible option {oid}")
        q = int(j.quantity)
        total += int(o.variable_cost_cents) + q * int(j.late_penalty_cents_per_unit_day) * max(0, arrival - int(j.due_day))
        k = (o.warehouse_region, j.product_id)
        stock_used[k] = stock_used.get(k, 0) + q
        ck = (o.warehouse_region, dispatch)
        cap_used[ck] = cap_used.get(ck, 0) + q
        sup_units[o.supplier_id] = sup_units.get(o.supplier_id, 0) + q
        used_sup.add(o.supplier_id)
    for k, v in stock_used.items():
        if v > int(stock.get(k, 0)):
            problems.append(f"stock exceeded {k}")
    for k, v in cap_used.items():
        if v > int(cap.get(k, 0)):
            problems.append(f"capacity exceeded {k}")
    for sid in used_sup:
        s = sup.loc[sid]
        if not (int(s.minimum_units) <= sup_units[sid] <= int(s.maximum_units)):
            problems.append(f"supplier {sid} units {sup_units[sid]} outside limits")
        total += int(s.activation_cost_cents)
    return problems, total


def solve_case(case_id: str) -> dict:
    base = load_case(case_id)
    base_cost, base_status, plan, opts = solve(base, return_plan=True)
    problems, recomputed = verify_plan(base, plan, opts)
    if problems or recomputed != base_cost:
        raise RuntimeError(f"{case_id}: plan check failed {problems} recomputed {recomputed} vs {base_cost}")
    cf = apply_changes(base)
    cf_cost, cf_status = solve(cf)
    return {
        "case_id": case_id,
        "baseline_cents": base_cost,
        "counterfactual_cents": cf_cost,
        "n_changes": len(base.changes),
        "warnings": cf.warnings,
        "status": f"{base_status}/{cf_status}",
    }


CASES = sorted(_read("hard_opt_cases").case_id.unique()) if STAGE == "final" else [f"OI{i:02d}" for i in range(1, 16)]


def run_all() -> pd.DataFrame:
    rows = [solve_case(c) for c in CASES]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_all()
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


def optimiser_answers() -> dict:
    """P151-P180: baseline minimum (EUR) and counterfactual change (EUR, signed) per case."""
    df = run_all().set_index("case_id")
    out = {}
    for i, case in enumerate(CASES):
        base = df.loc[case, "baseline_cents"] / 100
        change = (df.loc[case, "counterfactual_cents"] - df.loc[case, "baseline_cents"]) / 100
        out[f"P{151 + 2 * i:03d}"] = f"{base:.2f}"
        out[f"P{152 + 2 * i:03d}"] = f"{change:.2f}"
    return out


if __name__ == "__main__":
    import csv
    ans = optimiser_answers()
    out = ROOT / "outputs" / "optimiser_answers_p151_p180.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["question_id", "answer"])
        for k in sorted(ans):
            w.writerow([k, ans[k]])
    print(f"written {out}")
    for k in sorted(ans):
        print(k, ans[k])
