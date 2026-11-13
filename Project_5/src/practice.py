"""Hand-coded answers to the 10 practice questions (E001-E010), used to validate the helpers.

Each function follows the rule text in data/public/business_rules.md. The
practice answers in data/public/solved_examples.csv are the check. These are
NOT yet the competition pipeline; they test that our reading of the rules is right.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import (  # noqa: E402
    REF,
    category_at,
    eligible_customers,
    fx,
    item_revenue,
    load,
    order_identity_map,
    valid_orders,
)

Q4_START, Q4_END = pd.Timestamp("2024-10-01"), pd.Timestamp("2025-01-01")
Q3_START = pd.Timestamp("2024-07-01")


def _items_with_customer(t):
    it = item_revenue(t)
    cust = t.customers.set_index("customer_id")
    it["country"] = it["customer_id"].map(cust["country"])
    it["region"] = it["customer_id"].map(cust["region"])
    it["order_cat"] = category_at(it["order_date"], it["product_id"], t)
    return it


def e001(t):
    it = _items_with_customer(t)
    q = it[(it.order_date >= Q4_START) & (it.order_date < Q4_END) & (it.country == "Germany")]
    return f"{round(q.net_eur.sum(), 2):.2f}"


def e002(t):
    """High-value at-risk real customers in France as of 2025-01-01."""
    idm = order_identity_map(t)
    vo = valid_orders(t)
    it = item_revenue(t)
    # Net revenue per raw order customer, then rolled up to identity.
    rev = it.groupby("customer_id")["net_eur"].sum()
    last = vo.groupby("customer_id")["order_date"].max()
    raw = idm.dropna(subset=["canonical_customer_id"]).copy()
    raw["revenue"] = raw["customer_id"].map(rev).fillna(0.0)
    raw["last_order"] = raw["customer_id"].map(last)
    ident = raw.groupby("identity").agg(
        revenue=("revenue", "sum"),
        last_order=("last_order", "max"),
        country=("country", "first"),
    )
    # Recompute canonical country (first row by canonical_customer_id match).
    canon = raw[raw["customer_id"] == raw["canonical_customer_id"]].set_index("identity")
    ident["country"] = canon["country"].reindex(ident.index)
    high = ident[ident.revenue >= 2500]
    # Unresolved tickets for any eligible alias of the identity, opened by reference.
    tk = t.tickets[(t.tickets["opened_at"] <= REF)]
    tk = tk[tk["customer_id"].isin(eligible_customers(t)["customer_id"])]
    unresolved = tk[tk["resolved_at"].isna() | (tk["resolved_at"] > REF)]
    alias_to_id = raw.set_index("customer_id")["identity"]
    unresolved_ids = set(unresolved["customer_id"].map(alias_to_id).dropna())
    cutoff = REF - pd.Timedelta(75, unit="D")
    at_risk = high[
        (pd.to_datetime(high.last_order) < cutoff)
        & high.index.isin(unresolved_ids)
        & high.country.eq("France")
    ]
    return str(int(len(at_risk)))


def e003(t):
    """Approved-refund rate for Laptop items ordered in Q3 2024: refund EUR / gross item EUR."""
    it = _items_with_customer(t)
    base = it[(it.order_date >= Q3_START) & (it.order_date < Q4_START) & (it.order_cat == "Laptop")]
    gross = base.gross_eur.sum()
    refunds = base.groupby(["order_id", "product_id"]).refund_eur_total.first().sum()
    return f"{round(100 * refunds / gross, 2):.2f}"


def e004(t):
    """Share of real customers whose first valid order is in 2024-03 and who ordered again within 60 days."""
    idm = order_identity_map(t)
    vo = valid_orders(t).copy()
    vo["identity"] = vo["customer_id"].map(idm.set_index("customer_id")["identity"])
    vo = vo.dropna(subset=["identity"]).sort_values(["identity", "order_date", "order_id"])
    first = vo.groupby("identity")["order_date"].first()
    cohort = first[first.dt.strftime("%Y-%m") == "2024-03"].index
    hits = 0
    for ident in cohort:
        rows = vo[vo.identity == ident]
        f = rows.order_date.iloc[0]
        later = rows.iloc[1:]
        gap = pd.to_datetime(later.order_date) - pd.Timestamp(f)
        if (gap <= pd.Timedelta(60, unit="D")).any():
            hits += 1
    return f"{round(100 * hits / len(cohort), 2):.2f}"


def e006(t):
    """EUR cost of accepted units on eligible posted receipts for warehouse Central."""
    po = t.po_lines.merge(t.suppliers[["supplier_id", "is_test_supplier", "contract_start", "contract_end"]],
                          on="supplier_id", how="left")
    po = po[
        po.status.isin(["open", "closed"])
        & (po.ordered_at <= REF)
        & ~po.is_test_supplier
        & (po.contract_start <= po.ordered_at)
        & (po.ordered_at < po.contract_end)
        & (po.warehouse_region == "Central")
    ]
    gr = t.receipts[(t.receipts.status == "posted") & t.receipts.po_line_id.isin(po.po_line_id)].merge(
        po[["po_line_id", "unit_cost_native", "currency", "ordered_at"]], on="po_line_id"
    )
    gr = gr[(gr.received_at >= gr.ordered_at) & (gr.received_at <= REF)]
    months = gr.received_at.dt.strftime("%Y-%m")
    eur = [fx(m, c, t) for m, c in zip(months, gr.currency)]
    total = (gr.accepted_units * gr.unit_cost_native * pd.Series(eur, index=gr.index)).sum()
    return f"{round(total, 2):.2f}"


def e007(t):
    """Suggested replenishment units for Central at the reference (strictly-below rule)."""
    pol = t.policies[
        (t.policies.warehouse_region == "Central") & (t.policies.valid_from <= REF) & (REF < t.policies.valid_to)
    ]
    inv = t.inventory[t.inventory.warehouse_region == "Central"].set_index("product_id")
    total = 0
    for _, p in pol.iterrows():
        pid = p.product_id
        if pid not in inv.index:
            continue
        stock_row = inv.loc[pid]
        mv = t.movements[
            (t.movements.product_id == pid)
            & (t.movements.warehouse_region == "Central")
            & (t.movements.status == "posted")
            & (t.movements.event_at <= REF)
            & (t.movements.event_at > stock_row.last_stocktake_date)
        ]
        end = stock_row.on_hand_units + mv.units_delta.sum()
        if end < p.reorder_point:
            total += max(0, p.target_units - end)
    return str(int(total))


def e008(t):
    """Recovery value in EUR of eligible inspections, for refunds whose product was in Accessory at order time."""
    vo = valid_orders(t)
    items = t.order_items[t.order_items.order_id.isin(vo.order_id)]
    pairs = set(zip(items.order_id, items.product_id))
    r = t.refunds[(t.refunds.refund_status == "approved") & (t.refunds.refund_date <= REF)]
    r = r[[ (o, p) in pairs for o, p in zip(r.order_id, r.product_id)]].copy()
    order_dates = vo.set_index("order_id")["order_date"]
    r["order_date"] = r.order_id.map(order_dates)
    r = r[r.order_date.notna()].copy()
    r["cat"] = category_at(r.order_date, r.product_id, t)
    ins = t.inspections[(t.inspections.status == "completed")].merge(
        r[["refund_id", "refund_date", "cat"]], on="refund_id", how="inner"
    )
    ins = ins[(ins.refund_date <= ins.inspected_at) & (ins.inspected_at <= REF) & (ins.cat == "Accessory")]
    total = (ins.restockable_units * ins.recovery_unit_value_eur).sum()
    return f"{round(total, 2):.2f}"


def e009(t):
    """Eligible approved worklog labour cost in EUR for 'account' tickets."""
    tk = t.tickets[(t.tickets.opened_at <= REF) & (t.tickets.issue_type == "account")]
    tk = tk[tk.customer_id.isin(eligible_customers(t).customer_id)]
    wl = t.worklogs[(t.worklogs.work_status == "approved")].merge(
        tk[["ticket_id", "opened_at"]], on="ticket_id", how="inner"
    )
    wl = wl[(wl.opened_at <= wl.worked_at) & (wl.worked_at <= REF)]
    months = wl.worked_at.dt.strftime("%Y-%m")
    eur = [fx(m, c, t) for m, c in zip(months, wl.currency)]
    cost = (wl.minutes_worked / 60 * wl.hourly_rate_native * pd.Series(eur, index=wl.index)).sum()
    return f"{round(cost, 2):.2f}"


def e010(t):
    """Restockable share (%) of units received on eligible completed inspections for 'goodwill' refunds."""
    vo = valid_orders(t)
    items = t.order_items[t.order_items.order_id.isin(vo.order_id)]
    pairs = set(zip(items.order_id, items.product_id))
    r = t.refunds[(t.refunds.refund_status == "approved") & (t.refunds.refund_date <= REF)]
    r = r[[(o, p) in pairs for o, p in zip(r.order_id, r.product_id)]]
    ins = t.inspections[t.inspections.status == "completed"].merge(
        r[["refund_id", "refund_date", "refund_reason"]], on="refund_id", how="inner"
    )
    ins = ins[(ins.refund_date <= ins.inspected_at) & (ins.inspected_at <= REF) & (ins.refund_reason == "goodwill")]
    pct = 100 * ins.restockable_units.sum() / ins.units_received.sum()
    return f"{round(pct, 2):.2f}"


QUESTIONS = {
    "E001": e001, "E002": e002, "E003": e003, "E004": e004, "E005": None,
    "E006": e006, "E007": e007, "E008": e008, "E009": e009, "E010": e010,
}


def e005(t):
    it = _items_with_customer(t)
    q = it[(it.order_date >= Q4_START) & (it.order_date < Q4_END) & (it.region == "Western Europe")]
    s = q.groupby("order_cat").net_eur.sum().sort_values(ascending=False)
    s = s.reset_index().sort_values(["net_eur", "order_cat"], ascending=[False, True])
    return ";".join(s.order_cat.head(3))


QUESTIONS["E005"] = e005


def run_all() -> pd.DataFrame:
    t = load()
    expected = pd.read_csv(
        ROOT_DATA / "solved_examples.csv", dtype=str, keep_default_na=False
    ).set_index("question_id")["answer"]
    rows = []
    for qid, fn in QUESTIONS.items():
        try:
            got = fn(t)
        except Exception as exc:  # report rather than hide
            got = f"ERROR: {exc}"
        exp = expected[qid]
        rows.append({"question_id": qid, "got": got, "expected": exp, "match": got == exp})
    return pd.DataFrame(rows)


ROOT_DATA = Path(__file__).resolve().parents[1] / "data" / "public"


if __name__ == "__main__":
    df = run_all()
    print(df.to_string(index=False))
    print(f"\nmatches: {int(df.match.sum())} / {len(df)}")
