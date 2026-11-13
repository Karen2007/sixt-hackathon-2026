"""Answers to the company questions P001-P150 (public stage).

Each question family is one parameterised function. The family functions are
shared with the practice checks in tests/test_company.py, so the same code that
reproduces E001-E010 also answers the matching P-questions.

Rules come from data/public/business_rules.md. Output formatting follows the
answer types: number -> 2 decimals, integer -> whole number, boolean -> true/false,
string -> name, date -> YYYY-MM-DD. Undefined ratios return N/A.
"""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.core import (
    REF,
    category_at,
    eligible_customers,
    fx,
    fx_series,
    item_revenue,
    load,
    order_identity_map,
    valid_orders,
)

NA = "N/A"
SLA_HOURS = {"urgent": 24, "high": 48, "medium": 72, "low": 120}
QUARTERS = {
    "Q1": ("2024-01-01", "2024-04-01"),
    "Q2": ("2024-04-01", "2024-07-01"),
    "Q3": ("2024-07-01", "2024-10-01"),
    "Q4": ("2024-10-01", "2025-01-01"),
}


def D(n: int) -> pd.Timedelta:
    return pd.Timedelta(n, unit="D")


def fmt_num(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return NA
    return f"{round(float(x), 2):.2f}"


def fmt_int(x) -> str:
    return str(int(x))


def fmt_bool(x) -> str:
    return "true" if bool(x) else "false"


def month_bounds(month: str):
    start = pd.Timestamp(month + "-01")
    return start, start + pd.DateOffset(months=1)


def quarter_bounds(q: str):
    a, b = QUARTERS[q]
    return pd.Timestamp(a), pd.Timestamp(b)


# --------------------------------------------------------------------- context

@lru_cache(maxsize=1)
def ctx() -> SimpleNamespace:
    """Shared, enriched tables. Built once and reused by every question."""
    t = load()
    c = SimpleNamespace(t=t)
    cust = t.customers.set_index("customer_id")
    acc = t.accounts.set_index("account_id")
    idm = order_identity_map(t).set_index("customer_id")
    c.customers = cust
    c.accounts = acc
    c.identity_of = idm["identity"]

    # Valid-order item revenue with customer and account attributes.
    it = item_revenue(t).copy()
    it["country"] = it["customer_id"].map(cust["country"])
    it["region"] = it["customer_id"].map(cust["region"])
    it["channel"] = it["customer_id"].map(cust["acquisition_channel"])
    it["account_id"] = it["customer_id"].map(cust["account_id"])
    it["segment"] = it["account_id"].map(acc["segment"])
    it["industry"] = it["account_id"].map(acc["industry"])
    it["identity"] = it["customer_id"].map(idm["identity"])
    it["order_cat"] = category_at(it["order_date"], it["product_id"], t)
    prod = t.products.set_index("product_id")
    it["product_name"] = it["product_id"].map(prod["product_name"])
    it["family"] = it["product_id"].map(prod["family"])
    it["month"] = it["order_date"].dt.strftime("%Y-%m")
    it["cost_eur"] = (
        it["quantity"]
        * it["product_id"].map(prod["standard_cost"])
        * fx_series(it["month"], it["product_id"].map(prod["cost_currency"]), t)
    )
    c.items = it
    c.valid = t.orders.set_index("order_id").loc[valid_orders(t)["order_id"]].copy()
    c.valid["account_id"] = c.valid["customer_id"].map(cust["account_id"])
    c.order_net = it.groupby("order_id")["net_eur"].sum()
    c.order_identity = c.valid["customer_id"].map(idm["identity"])

    # Eligible support tickets opened by the reference, with SLA result.
    elig = set(eligible_customers(t)["customer_id"])
    tk = t.tickets[(t.tickets["opened_at"] <= REF) & t.tickets["customer_id"].isin(elig)].copy()
    tk["identity"] = tk["customer_id"].map(idm["identity"])
    tk["account_id"] = tk["customer_id"].map(cust["account_id"])
    tk["country"] = tk["customer_id"].map(cust["country"])
    tk["region"] = tk["customer_id"].map(cust["region"])
    end = tk["resolved_at"].where(tk["resolved_at"] <= REF, REF)
    tk["elapsed_h"] = (end - tk["opened_at"]).dt.total_seconds() / 3600
    tk["breach"] = tk["elapsed_h"] > tk["priority"].map(SLA_HOURS)
    c.tickets = tk

    # Shipments on valid orders shipped by the reference, with late-delivery logic.
    sh = t.shipments.copy()
    sh["shipped_at"] = pd.to_datetime(sh["shipped_at"], errors="coerce")
    sh["promise"] = pd.to_datetime(sh["promised_delivery_date"], errors="coerce")
    sh["delivered"] = pd.to_datetime(sh["delivered_at"].replace("", None), errors="coerce")
    sh = sh[(sh["shipped_at"] <= REF) & sh["order_id"].isin(c.valid.index)].copy()
    sh["customer_id"] = sh["order_id"].map(c.valid["customer_id"])
    sh["region"] = sh["customer_id"].map(cust["region"])
    sh["country"] = sh["customer_id"].map(cust["country"])
    sh["account_id"] = sh["customer_id"].map(cust["account_id"])
    sh["identity"] = sh["customer_id"].map(idm["identity"])
    sh["net_w"] = sh["order_id"].map(c.order_net)
    eff = sh["delivered"].where(sh["delivered"].notna() & (sh["delivered"] <= REF), REF)
    sh["delay_days"] = (
        (eff.dt.normalize() - sh["promise"].dt.normalize()).dt.days.clip(lower=0)
    )
    sh["late"] = (sh["delay_days"] > 0) & (sh["promise"] < REF)
    c.shipments = sh

    # Web sessions and marketing touches for eligible customers, up to the reference.
    ws = t.web_sessions.copy()
    ws["session_date"] = pd.to_datetime(ws["session_date"], errors="coerce")
    ws = ws[ws["customer_id"].isin(elig) & (ws["session_date"] <= REF)].copy()
    ws["identity"] = ws["customer_id"].map(idm["identity"])
    c.sessions = ws

    mt = t.marketing_touches.copy()
    mt["touch_date"] = pd.to_datetime(mt["touch_date"], errors="coerce")
    mt = mt[mt["customer_id"].isin(elig) & (mt["touch_date"] <= REF)].copy()
    mt["identity"] = mt["customer_id"].map(idm["identity"])
    c.touches = mt

    # Identity-level valid orders (for resolved-customer questions).
    vo = c.valid.reset_index()[["order_id", "customer_id", "order_date"]].copy()
    vo["identity"] = vo["customer_id"].map(idm["identity"])
    vo["net"] = vo["order_id"].map(c.order_net).fillna(0.0)
    c.vo = vo.dropna(subset=["identity"])

    # Canonical real-customer rows (one per identity).
    canon = order_identity_map(t)
    canon = canon[canon["customer_id"] == canon["canonical_customer_id"]].set_index("identity")
    canon["account_id"] = canon["customer_id"].map(cust["account_id"])
    c.canon = canon

    return c


# ------------------------------------------------------- revenue families

def net_rev(mask_items=None) -> float:
    it = ctx().items
    return float(it.loc[mask_items, "net_eur"].sum()) if mask_items is not None else float(it["net_eur"].sum())


def rev_country_period(country: str, q: str) -> str:
    a, b = quarter_bounds(q)
    it = ctx().items
    m = (it.country == country) & (it.order_date >= a) & (it.order_date < b)
    return fmt_num(it.loc[m, "net_eur"].sum())


def rev_region_month(region: str, month: str) -> str:
    a, b = month_bounds(month)
    it = ctx().items
    m = (it.region == region) & (it.order_date >= a) & (it.order_date < b)
    return fmt_num(it.loc[m, "net_eur"].sum())


def growth_country(country: str) -> str:
    it = ctx().items
    q3, q4 = quarter_bounds("Q3"), quarter_bounds("Q4")
    r3 = it.loc[(it.country == country) & (it.order_date >= q3[0]) & (it.order_date < q3[1]), "net_eur"].sum()
    r4 = it.loc[(it.country == country) & (it.order_date >= q4[0]) & (it.order_date < q4[1]), "net_eur"].sum()
    if r3 == 0:
        return NA
    return fmt_num(100 * (r4 - r3) / r3)


def growth_bool_region(region: str, threshold_pct: float) -> str:
    it = ctx().items
    q3, q4 = quarter_bounds("Q3"), quarter_bounds("Q4")
    r3 = it.loc[(it.region == region) & (it.order_date >= q3[0]) & (it.order_date < q3[1]), "net_eur"].sum()
    r4 = it.loc[(it.region == region) & (it.order_date >= q4[0]) & (it.order_date < q4[1]), "net_eur"].sum()
    if r3 == 0:
        return NA
    return fmt_bool((r4 - r3) / r3 * 100 >= threshold_pct)


def region_target_pct(region: str, q: str) -> str:
    a, b = quarter_bounds(q)
    rev = rev_total_region(region, a, b)
    tg = ctx().t.sales_targets
    months = pd.date_range(a, b - D(1), freq="MS").strftime("%Y-%m")
    tgt = tg[(tg.region == region) & tg.month.isin(months)]["net_revenue_target_eur"].astype(float).sum()
    if tgt == 0:
        return NA
    return fmt_num(100 * rev / tgt)


def rev_total_region(region: str, a, b) -> float:
    it = ctx().items
    return float(it.loc[(it.region == region) & (it.order_date >= a) & (it.order_date < b), "net_eur"].sum())


def target_met(country: str, month: str) -> str:
    a, b = month_bounds(month)
    it = ctx().items
    rev = it.loc[(it.country == country) & (it.order_date >= a) & (it.order_date < b), "net_eur"].sum()
    tg = ctx().t.sales_targets
    tgt = tg[(tg.country == country) & (tg.month == month)]["net_revenue_target_eur"].astype(float)
    if tgt.empty:
        return NA
    return fmt_bool(rev >= float(tgt.iloc[0]))


def zscore_region_month(region: str, month: str) -> str:
    it = ctx().items
    monthly = it[it.region == region].groupby("month")["net_eur"].sum()
    months = [f"2024-{m:02d}" for m in range(1, 10)]
    series = np.array([float(monthly.get(m, 0.0)) for m in months])
    x = float(monthly.get(month, 0.0))
    sd = series.std(ddof=0)
    if sd == 0:
        return NA
    return fmt_num((x - series.mean()) / sd)


def top_zscore_region(month: str) -> str:
    scores = {}
    for r in ["Central Europe", "Southern Europe", "Western Europe"]:
        s = zscore_region_month(r, month)
        if s != NA:
            scores[r] = float(s)
    # Highest score; ties go to the alphabetically first region name (business_rules.md).
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[0][0]


def top3_categories_region(region: str, q: str) -> str:
    a, b = quarter_bounds(q)
    it = ctx().items
    m = (it.region == region) & (it.order_date >= a) & (it.order_date < b)
    s = it[m].groupby("order_cat")["net_eur"].sum().reset_index()
    s = s.sort_values(["net_eur", "order_cat"], ascending=[False, True])
    return ";".join(s["order_cat"].head(3))


def top_category_segment(segment: str, q: str) -> str:
    a, b = quarter_bounds(q)
    it = ctx().items
    m = (it.segment == segment) & (it.order_date >= a) & (it.order_date < b)
    s = it[m].groupby("order_cat")["net_eur"].sum().reset_index()
    s = s.sort_values(["net_eur", "order_cat"], ascending=[False, True])
    return s["order_cat"].iloc[0]


def top_product_region_month(region: str, month: str) -> str:
    a, b = month_bounds(month)
    it = ctx().items
    m = (it.region == region) & (it.order_date >= a) & (it.order_date < b)
    s = it[m].groupby("product_name")["net_eur"].sum().reset_index()
    s = s.sort_values(["net_eur", "product_name"], ascending=[False, True])
    return s["product_name"].iloc[0]


def top_refund_rate_category_country(country: str) -> str:
    it = ctx().items
    m = it[it.country == country]
    g = m.groupby("order_cat").agg(refund=("refund_alloc_eur", "sum"), gross=("gross_eur", "sum"))
    g = g[g.gross != 0]
    g["rate"] = g.refund / g.gross
    g = g.reset_index().sort_values(["rate", "order_cat"], ascending=[False, True])
    return g["order_cat"].iloc[0]


def refund_rate_category_period(category: str, q: str) -> str:
    a, b = quarter_bounds(q)
    it = ctx().items
    m = it[(it.order_cat == category) & (it.order_date >= a) & (it.order_date < b)]
    gross = m["gross_eur"].sum()
    if gross == 0:
        return NA
    return fmt_num(100 * m["refund_alloc_eur"].sum() / gross)


def margin_rate_category(category: str | None, q: str | None = None) -> str:
    it = ctx().items
    m = it if category is None else it[it.order_cat == category]
    if q is not None:
        a, b = quarter_bounds(q)
        m = m[(m.order_date >= a) & (m.order_date < b)]
    net = m["net_eur"].sum()
    if net == 0:
        return NA
    return fmt_num(100 * (net - m["cost_eur"].sum()) / net)


def top_margin_category(q: str) -> str:
    a, b = quarter_bounds(q)
    it = ctx().items
    m = it[(it.order_date >= a) & (it.order_date < b)]
    g = m.groupby("order_cat").agg(net=("net_eur", "sum"), cost=("cost_eur", "sum"))
    g = g[g.net != 0]
    g["rate"] = (g.net - g.cost) / g.net
    g = g.reset_index().sort_values(["rate", "order_cat"], ascending=[False, True])
    return g["order_cat"].iloc[0]


def refund_discount_correlation() -> str:
    it = ctx().items
    g = it.groupby("order_cat").agg(
        gross=("gross_eur", "sum"), disc=("line_disc_eur", "sum"), refund=("refund_alloc_eur", "sum")
    )
    x = g.disc / g.gross
    y = g.refund / g.gross
    return fmt_num(float(np.corrcoef(x, y)[0, 1]))


def median_order_net_segment(segment: str) -> str:
    c = ctx()
    accs = c.accounts[c.accounts.segment == segment].index
    orders = c.valid[c.valid["account_id"].isin(accs)]
    vals = c.order_net.reindex(orders.index).dropna()
    return fmt_num(vals.median())


def account_lifetime_revenue() -> pd.Series:
    c = ctx()
    acc = c.accounts[~c.accounts["is_test_account"]]
    rev = c.items.groupby("account_id")["net_eur"].sum()
    return rev.reindex(acc.index).fillna(0.0)


def p90_account(segment: str) -> str:
    c = ctx()
    acc = c.accounts[(~c.accounts["is_test_account"]) & (c.accounts.segment == segment)].index
    vals = account_lifetime_revenue().reindex(acc).fillna(0.0).to_numpy()
    return fmt_num(np.percentile(vals, 90, method="linear"))


def top_account_category_share(segment: str, category: str) -> str:
    c = ctx()
    accs = c.accounts[(~c.accounts["is_test_account"]) & (c.accounts.segment == segment)].index
    rev = account_lifetime_revenue().reindex(accs)
    top = rev.sort_values(ascending=False).index[0]
    it = c.items[c.items.account_id == top]
    total = it["net_eur"].sum()
    if total == 0:
        return NA
    return fmt_num(100 * it.loc[it.order_cat == category, "net_eur"].sum() / total)


def hv_accounts(segment: str, threshold: float = 7000.0) -> set:
    c = ctx()
    rev = c.items.groupby("account_id")["net_eur"].sum()
    segs = c.accounts[c.accounts.segment == segment].index
    return set(rev[(rev >= threshold)].index.intersection(segs))


def issue_mode_high_value(segment: str) -> str:
    c = ctx()
    tk = c.tickets[c.tickets.account_id.isin(hv_accounts(segment))]
    counts = tk.groupby("issue_type").size().reset_index(name="n")
    counts = counts.sort_values(["n", "issue_type"], ascending=[False, True])
    return counts["issue_type"].iloc[0]


def high_priority_breach_pct() -> str:
    tk = ctx().tickets
    h = tk[tk.priority == "high"]
    if h.empty:
        return NA
    return fmt_num(100 * h.breach.mean())


def italy_pre_ticket_top_category() -> str:
    c = ctx()
    tk = c.tickets.groupby("customer_id")["opened_at"].min()
    it = c.items[c.items.country == "Italy"].copy()
    it = it[it.customer_id.isin(tk.index)]
    it["first_ticket"] = it.customer_id.map(tk)
    it = it[it.order_date < it.first_ticket]
    s = it.groupby("order_cat")["net_eur"].sum().reset_index()
    s = s.sort_values(["net_eur", "order_cat"], ascending=[False, True])
    return s["order_cat"].iloc[0]


# -------------------------------------------------------- identity families

def at_risk_identities() -> pd.DataFrame:
    """High-value (lifetime >= 2500) and last order before ref-75d, with an unresolved eligible ticket."""
    c = ctx()
    ident = c.vo.groupby("identity").agg(lifetime=("net", "sum"), last_order=("order_date", "max"))
    ident = ident[ident.lifetime >= 2500]
    ident = ident[ident.last_order < REF - D(75)]
    unresolved = c.tickets[(c.tickets.resolved_at > REF) | c.tickets.resolved_at.isna()]
    ids = set(unresolved.identity.dropna())
    ident = ident[ident.index.isin(ids)].copy()
    ident["country"] = c.canon["country"].reindex(ident.index)
    ident["account_id"] = c.canon["account_id"].reindex(ident.index)
    return ident


def atrisk_count_country(country: str) -> str:
    ar = at_risk_identities()
    return fmt_int((ar.country == country).sum())


def at_risk_top_manager() -> str:
    c = ctx()
    ar = at_risk_identities()
    mgr = ar["account_id"].map(c.accounts["account_manager"])
    tot = ar.lifetime.groupby(mgr).sum().sort_values(ascending=False)
    return tot.index[0]


def top_email_domain() -> str:
    c = ctx()
    idm = order_identity_map(c.t).set_index("customer_id")
    domain = idm["identity"].str.split("@").str[1]
    dom_of_order = c.valid["customer_id"].map(domain)
    s = c.order_net.groupby(dom_of_order.reindex(c.order_net.index)).sum().sort_values(ascending=False)
    return s.index[0]


def active_count_region(region: str) -> str:
    c = ctx()
    vo = c.vo
    cnt = vo.groupby("identity").agg(n=("order_id", "nunique"), last_order=("order_date", "max"))
    cnt = cnt[(cnt.n >= 2) & (cnt.last_order >= REF - D(90))]
    in_region = c.canon["region"].reindex(cnt.index)
    return fmt_int((in_region == region).sum())


def unique_real_customers_country(country: str) -> str:
    c = ctx()
    return fmt_int((c.canon["country"] == country).sum())


def earliest_order_by_channel(channel: str) -> str:
    c = ctx()
    chan = c.customers["acquisition_channel"].reindex(c.valid["customer_id"]).to_numpy()
    d = c.valid["order_date"][chan == channel]
    return pd.Timestamp(d.min()).strftime("%Y-%m-%d")


def trial_conversion(channel: str) -> str:
    c = ctx()
    canon = c.canon
    base = canon[(canon.acquisition_channel == channel) & canon.trial_start_date.notna() & (canon.trial_start_date <= REF)]
    if base.empty:
        return NA
    first = c.vo.groupby("identity")["order_date"].min()
    conv = 0
    for ident, row in base.iterrows():
        f = first.get(ident)
        if f is not None and row.trial_start_date <= f <= row.trial_start_date + D(30):
            conv += 1
    return fmt_num(100 * conv / len(base))


def cohort_retention(month: str, days: int) -> str:
    c = ctx()
    vo = c.vo.sort_values(["identity", "order_date", "order_id"])
    first = vo.groupby("identity")["order_date"].first()
    cohort = first[first.dt.strftime("%Y-%m") == month].index
    if len(cohort) == 0:
        return NA
    hits = 0
    for ident in cohort:
        rows = vo[vo.identity == ident]
        f = rows.order_date.iloc[0]
        later = rows.iloc[1:]
        if len(later) and ((later.order_date - f) <= D(days)).any():
            hits += 1
    return fmt_num(100 * hits / len(cohort))


def web_direct_conversion() -> str:
    c = ctx()
    s = c.sessions[c.sessions.traffic_source == "direct"]
    ids = set(s.identity.dropna())
    if not ids:
        return NA
    hits = 0
    for ident in ids:
        sess = s[s.identity == ident]["session_date"]
        orders = c.vo[c.vo.identity == ident]["order_date"]
        ok = False
        for sd in sess:
            if ((orders >= sd) & (orders <= sd + D(14))).any():
                ok = True
                break
        hits += ok
    return fmt_num(100 * hits / len(ids))


def campaign_top_attributed(channel: str) -> str:
    c = ctx()
    touches = c.touches.copy()
    rows = []
    for _, o in c.vo.iterrows():
        cand = touches[(touches.identity == o.identity)]
        delta = (o.order_date - cand.touch_date)
        cand = cand[(delta >= pd.Timedelta(0)) & (delta <= D(30))]
        if cand.empty:
            continue
        cand = cand.sort_values(["touch_date", "campaign", "touch_id"], ascending=[False, True, True])
        best = cand.iloc[0]
        rows.append({"campaign": best.campaign, "channel": best.marketing_channel, "net": o.net})
    df = pd.DataFrame(rows)
    df = df[df.channel == channel]
    s = df.groupby("campaign")["net"].sum().reset_index().sort_values(["net", "campaign"], ascending=[False, True])
    return s["campaign"].iloc[0]


def campaign_ticket_revenue(campaign: str, issue_types: list[str]) -> str:
    c = ctx()
    camp_ids = set(c.touches[c.touches.campaign == campaign].identity.dropna())
    tk_ids = set(c.tickets[c.tickets.issue_type.isin(issue_types)].identity.dropna())
    ids = camp_ids & tk_ids
    return fmt_num(c.vo[c.vo.identity.isin(ids)]["net"].sum())


def accounts_late_breach(industry: str) -> str:
    c = ctx()
    accs = set(c.accounts[(c.accounts.industry == industry) & (~c.accounts.is_test_account)].index)
    late = set(c.shipments[c.shipments.late].account_id.dropna())
    breach = set(c.tickets[c.tickets.breach].account_id.dropna())
    return fmt_int(len(accs & late & breach))


def late_ticket_breach_pct(region: str) -> str:
    c = ctx()
    sh = c.shipments[c.shipments.late & c.shipments.delivered.notna() & (c.shipments.delivered <= REF) & (c.shipments.region == region)]
    m = sh[["identity", "delivered"]].merge(c.tickets[["ticket_id", "identity", "opened_at", "breach"]], on="identity")
    diff = (m.opened_at - m.delivered).dt.total_seconds() / 86400
    m = m[(diff >= 0) & (diff <= 14)].drop_duplicates("ticket_id")
    if m.empty:
        return NA
    return fmt_num(100 * m.breach.mean())


def carrier_wdelay(carrier: str) -> str:
    sh = ctx().shipments
    s = sh[sh.late & (sh.carrier == carrier)]
    if s.empty or s.net_w.sum() == 0:
        return NA
    return fmt_num((s.delay_days * s.net_w).sum() / s.net_w.sum())


def top_carrier_by_wdelay(region: str) -> str:
    sh = ctx().shipments
    s = sh[sh.late & (sh.region == region)]
    rows = []
    for carrier, g in s.groupby("carrier"):
        rows.append({"carrier": carrier, "w": (g.delay_days * g.net_w).sum() / g.net_w.sum()})
    df = pd.DataFrame(rows).sort_values(["w", "carrier"], ascending=[False, True])
    return df["carrier"].iloc[0]


# ------------------------------------------------------ procurement families

def _eligible_po():
    t = ctx().t
    po = t.po_lines.merge(
        t.suppliers[["supplier_id", "supplier_name", "country", "is_test_supplier", "contract_start", "contract_end"]],
        on="supplier_id", how="left",
    )
    return po[
        po.status.isin(["open", "closed"])
        & (po.ordered_at <= REF)
        & ~po.is_test_supplier
        & (po.contract_start <= po.ordered_at)
        & (po.ordered_at < po.contract_end)
    ].copy()


def _eligible_receipts():
    t = ctx().t
    po = _eligible_po()
    gr = t.receipts[t.receipts.status == "posted"].merge(
        po[["po_line_id", "product_id", "warehouse_region", "ordered_at", "unit_cost_native", "currency",
            "supplier_id", "supplier_name", "country", "ordered_units", "promised_at"]],
        on="po_line_id", how="inner",
    )
    gr = gr[(gr.received_at >= gr.ordered_at) & (gr.received_at <= REF)].copy()
    gr["fx_receipt"] = fx_series(gr.received_at.dt.strftime("%Y-%m"), gr.currency, t)
    return gr


def commitment_country(country: str) -> str:
    po = _eligible_po()
    po = po[po.country == country]
    month = po.ordered_at.dt.strftime("%Y-%m")
    eur = fx_series(month, po.currency, ctx().t)
    return fmt_num((po.ordered_units * po.unit_cost_native * eur).sum())


def outstanding_units(warehouse: str) -> str:
    po = _eligible_po()
    po = po[po.warehouse_region == warehouse].copy()
    gr = _eligible_receipts()
    acc = gr.groupby("po_line_id")["accepted_units"].sum()
    po["accepted"] = po.po_line_id.map(acc).fillna(0.0)
    rem = (po.ordered_units - po.accepted).clip(lower=0)
    return fmt_int(rem.sum())


def ontime_pct(country: str) -> str:
    po = _eligible_po()
    po = po[(po.country == country) & (po.promised_at <= REF)].copy()
    if po.empty:
        return NA
    gr = _eligible_receipts()
    ontime = gr[gr.received_at <= gr.po_line_id.map(po.set_index("po_line_id")["promised_at"])]
    acc = ontime.groupby("po_line_id")["accepted_units"].sum()
    capped = np.minimum(po.po_line_id.map(acc).fillna(0.0), po.ordered_units)
    return fmt_num(100 * capped.sum() / po.ordered_units.sum())


def rejection_pct(country: str) -> str:
    gr = _eligible_receipts()
    gr = gr[gr.country == country]
    denom = gr.accepted_units.sum() + gr.rejected_units.sum()
    if denom == 0:
        return NA
    return fmt_num(100 * gr.rejected_units.sum() / denom)


def ppv(family: str | None) -> str:
    po = _eligible_po()
    t = ctx().t
    prod = t.products.set_index("product_id")
    po["family"] = po.product_id.map(prod.family)
    if family is not None:
        po = po[po.family == family]
    month = po.ordered_at.dt.strftime("%Y-%m")
    purch = po.unit_cost_native * fx_series(month, po.currency, t)
    std = po.product_id.map(prod.standard_cost) * fx_series(month, po.product_id.map(prod.cost_currency), t)
    return fmt_num((po.ordered_units * (purch - std)).sum())


def receipt_cost(warehouse: str) -> str:
    gr = _eligible_receipts()
    gr = gr[gr.warehouse_region == warehouse]
    return fmt_num((gr.accepted_units * gr.unit_cost_native * gr.fx_receipt).sum())


def worst_supplier(country: str) -> str:
    gr = _eligible_receipts()
    gr = gr[gr.country == country]
    g = gr.groupby("supplier_name").agg(acc=("accepted_units", "sum"), rej=("rejected_units", "sum"))
    g = g[(g.acc + g.rej) > 0]
    g["ratio"] = g.rej / (g.acc + g.rej)
    g = g.reset_index().sort_values(["ratio", "supplier_name"], ascending=[False, True])
    return g["supplier_name"].iloc[0]


# -------------------------------------------------------- inventory families

def _end_stock():
    """Reconciled end stock for every inventory key: stocktake + posted deltas after it, through ref."""
    t = ctx().t
    inv = t.inventory.copy()
    mv = t.movements[t.movements.status == "posted"]
    out = []
    for _, r in inv.iterrows():
        m = mv[
            (mv.product_id == r.product_id)
            & (mv.warehouse_region == r.warehouse_region)
            & (mv.event_at > r.last_stocktake_date)
            & (mv.event_at <= REF)
        ]
        out.append(r.on_hand_units + m.units_delta.sum())
    inv["end_stock"] = out
    return inv


def recon_stock(warehouse: str) -> str:
    inv = _end_stock()
    return fmt_int(inv[inv.warehouse_region == warehouse].end_stock.sum())


def stock_value(warehouse: str) -> str:
    t = ctx().t
    inv = _end_stock()
    inv = inv[inv.warehouse_region == warehouse].copy()
    prod = t.products.set_index("product_id")
    cur = inv.product_id.map(prod.cost_currency)
    eur = fx_series(pd.Series(["2025-01"] * len(inv), index=inv.index), cur, t)
    return fmt_num((inv.end_stock * inv.product_id.map(prod.standard_cost) * eur).sum())


def _policies_at_ref(warehouse: str):
    t = ctx().t
    pol = t.policies[(t.policies.warehouse_region == warehouse) & (t.policies.valid_from <= REF) & (REF < t.policies.valid_to)]
    inv = _end_stock()
    inv = inv[inv.warehouse_region == warehouse].set_index("product_id")
    rows = []
    for _, p in pol.iterrows():
        if p.product_id in inv.index:
            rows.append({"product_id": p.product_id, "end": inv.loc[p.product_id, "end_stock"],
                         "reorder": p.reorder_point, "target": p.target_units})
    return pd.DataFrame(rows)


def below_reorder_count(warehouse: str) -> str:
    df = _policies_at_ref(warehouse)
    return fmt_int((df.end < df.reorder).sum())


def reorder_units(warehouse: str) -> str:
    df = _policies_at_ref(warehouse)
    below = df[df.end < df.reorder]
    return fmt_int((below.target - below.end).clip(lower=0).sum())


# ------------------------------------------------------------ returns families

def _valid_refunds():
    c = ctx()
    t = c.t
    items = t.order_items[t.order_items.order_id.isin(c.valid.index)]
    pairs = set(zip(items.order_id, items.product_id))
    r = t.refunds[(t.refunds.refund_status == "approved") & (t.refunds.refund_date <= REF)].copy()
    r = r[[(o, p) in pairs for o, p in zip(r.order_id, r.product_id)]]
    return r


def restockable_pct_reason(reason: str) -> str:
    r = _valid_refunds()
    r = r[r.refund_reason == reason]
    ins = ctx().t.inspections[ctx().t.inspections.status == "completed"].merge(
        r[["refund_id", "refund_date"]], on="refund_id", how="inner"
    )
    ins = ins[(ins.refund_date <= ins.inspected_at) & (ins.inspected_at <= REF)]
    if ins.units_received.sum() == 0:
        return NA
    return fmt_num(100 * ins.restockable_units.sum() / ins.units_received.sum())


def recovery_category(category: str) -> str:
    c = ctx()
    r = _valid_refunds()
    order_dates = c.valid["order_date"]
    r = r.assign(order_date=r.order_id.map(order_dates))
    r["cat"] = category_at(r.order_date, r.product_id, c.t)
    r = r[r.cat == category]
    ins = c.t.inspections[c.t.inspections.status == "completed"].merge(
        r[["refund_id", "refund_date"]], on="refund_id", how="inner"
    )
    ins = ins[(ins.refund_date <= ins.inspected_at) & (ins.inspected_at <= REF)]
    return fmt_num((ins.restockable_units * ins.recovery_unit_value_eur).sum())


# --------------------------------------------------------------- question map

def _p(fn, *args):
    def run():
        return fn(*args)
    run.plan = (fn.__name__, list(args))  # machine-readable routing, used to check the LLM router
    return run


QUESTIONS_P = {
    "P001": _p(late_ticket_breach_pct, "Southern Europe"),
    "P002": _p(carrier_wdelay, "DHL"),
    "P003": _p(top_carrier_by_wdelay, "Central Europe"),
    "P004": _p(margin_rate_category, "Phone"),
    "P005": _p(rev_country_period, "Greece", "Q2"),
    "P006": _p(refund_rate_category_period, "Accessory", "Q3"),
    "P007": _p(growth_country, "Italy"),
    "P008": _p(growth_bool_region, "Central Europe", 10),
    "P009": _p(at_risk_top_manager),
    "P010": _p(top_email_domain),
    "P011": _p(growth_country, "Austria"),
    "P012": _p(top_margin_category, "Q2"),
    "P013": _p(rev_country_period, "Germany", "Q3"),
    "P014": _p(growth_country, "Netherlands"),
    "P015": _p(region_target_pct, "Western Europe", "Q1"),
    "P016": _p(accounts_late_breach, "Education"),
    "P017": _p(rev_region_month, "Central Europe", "2024-05"),
    "P018": _p(accounts_late_breach, "Health"),
    "P019": _p(growth_bool_region, "Southern Europe", 15),
    "P020": _p(late_ticket_breach_pct, "Central Europe"),
    "P021": _p(p90_account, "Public Sector"),
    "P022": _p(zscore_region_month, "Central Europe", "2024-10"),
    "P023": _p(growth_bool_region, "Central Europe", 5),
    "P024": _p(issue_mode_high_value, "Mid-Market"),
    "P025": _p(accounts_late_breach, "Retail"),
    "P026": _p(zscore_region_month, "Western Europe", "2024-10"),
    "P027": _p(top_zscore_region, "2024-10"),
    "P028": _p(cohort_retention, "2024-02", 90),
    "P029": _p(carrier_wdelay, "GLS"),
    "P030": _p(top_category_segment, "SMB", "Q2"),
    "P031": _p(rev_region_month, "Western Europe", "2024-06"),
    "P032": _p(top_account_category_share, "Enterprise", "Accessory"),
    "P033": _p(refund_discount_correlation),
    "P034": _p(growth_country, "France"),
    "P035": _p(web_direct_conversion),
    "P036": _p(campaign_top_attributed, "Paid Search"),
    "P037": _p(top_refund_rate_category_country, "Spain"),
    "P038": _p(refund_rate_category_period, "Software", "Q2"),
    "P039": _p(rev_region_month, "Central Europe", "2024-01"),
    "P040": _p(rev_country_period, "Greece", "Q3"),
    "P041": _p(target_met, "Greece", "2024-07"),
    "P042": _p(growth_bool_region, "Western Europe", 10),
    "P043": _p(atrisk_count_country, "Spain"),
    "P044": _p(growth_country, "Poland"),
    "P045": _p(italy_pre_ticket_top_category),
    "P046": _p(atrisk_count_country, "Poland"),
    "P047": _p(zscore_region_month, "Western Europe", "2024-11"),
    "P048": _p(rev_region_month, "Western Europe", "2024-01"),
    "P049": _p(median_order_net_segment, "Mid-Market"),
    "P050": _p(refund_rate_category_period, "Software", "Q1"),
    "P051": _p(campaign_top_attributed, "Events"),
    "P052": _p(unique_real_customers_country, "Austria"),
    "P053": _p(campaign_ticket_revenue, "SpringLaunch", ["returns"]),
    "P054": _p(p90_account, "Mid-Market"),
    "P055": _p(top_carrier_by_wdelay, "Southern Europe"),
    "P056": _p(top3_categories_region, "Southern Europe", "Q1"),
    "P057": _p(top_account_category_share, "Mid-Market", "Appliance"),
    "P058": _p(top_refund_rate_category_country, "Italy"),
    "P059": _p(earliest_order_by_channel, "Referral"),
    "P060": _p(region_target_pct, "Central Europe", "Q4"),
    "P061": _p(top_zscore_region, "2024-11"),
    "P062": _p(growth_bool_region, "Central Europe", 15),
    "P063": _p(unique_real_customers_country, "Spain"),
    "P064": _p(region_target_pct, "Western Europe", "Q3"),
    "P065": _p(region_target_pct, "Southern Europe", "Q2"),
    "P066": _p(top_margin_category, "Q3"),
    "P067": _p(growth_country, "Greece"),
    "P068": _p(top_refund_rate_category_country, "Portugal"),
    "P069": _p(top_carrier_by_wdelay, "Western Europe"),
    "P070": _p(growth_country, "Portugal"),
    "P071": _p(top_product_region_month, "Central Europe", "2024-02"),
    "P072": _p(region_target_pct, "Western Europe", "Q4"),
    "P073": _p(campaign_top_attributed, "Partner"),
    "P074": _p(top_account_category_share, "Public Sector", "Accessory"),
    "P075": _p(margin_rate_category, "Software"),
    "P076": _p(zscore_region_month, "Southern Europe", "2024-12"),
    "P077": _p(issue_mode_high_value, "Enterprise"),
    "P078": _p(earliest_order_by_channel, "Organic"),
    "P079": _p(trial_conversion, "Referral"),
    "P080": _p(cohort_retention, "2024-06", 30),
    "P081": _p(region_target_pct, "Western Europe", "Q2"),
    "P082": _p(carrier_wdelay, "PostNL"),
    "P083": _p(high_priority_breach_pct),
    "P084": _p(p90_account, "SMB"),
    "P085": _p(accounts_late_breach, "Finance"),
    "P086": _p(late_ticket_breach_pct, "Western Europe"),
    "P087": _p(carrier_wdelay, "FedEx"),
    "P088": _p(target_met, "Germany", "2024-08"),
    "P089": _p(region_target_pct, "Central Europe", "Q2"),
    "P090": _p(accounts_late_breach, "Technology"),
    "P091": _p(trial_conversion, "Email"),
    "P092": _p(growth_bool_region, "Western Europe", 5),
    "P093": _p(accounts_late_breach, "Manufacturing"),
    "P094": _p(top_refund_rate_category_country, "Greece"),
    "P095": _p(target_met, "Italy", "2024-03"),
    "P096": _p(campaign_ticket_revenue, "TrialNudge", ["billing"]),
    "P097": _p(campaign_ticket_revenue, "TrialNudge", ["returns"]),
    "P098": _p(top_category_segment, "Public Sector", "Q4"),
    "P099": _p(top_product_region_month, "Southern Europe", "2024-12"),
    "P100": _p(active_count_region, "Western Europe"),
    "P101": _p(commitment_country, "Germany"),
    "P102": _p(commitment_country, "France"),
    "P103": _p(commitment_country, "Italy"),
    "P104": _p(commitment_country, "Spain"),
    "P105": _p(outstanding_units, "North"),
    "P106": _p(outstanding_units, "South"),
    "P107": _p(outstanding_units, "East"),
    "P108": _p(outstanding_units, "West"),
    "P109": _p(ontime_pct, "Germany"),
    "P110": _p(ontime_pct, "France"),
    "P111": _p(ontime_pct, "Italy"),
    "P112": _p(ontime_pct, "Spain"),
    "P113": _p(rejection_pct, "Germany"),
    "P114": _p(rejection_pct, "France"),
    "P115": _p(rejection_pct, "Italy"),
    "P116": _p(rejection_pct, "Spain"),
    "P117": _p(ppv, "Hardware"),
    "P118": _p(ppv, "Cloud"),
    "P119": _p(ppv, "Professional Services"),
    "P120": _p(ppv, None),
    "P121": _p(receipt_cost, "North"),
    "P122": _p(receipt_cost, "South"),
    "P123": _p(receipt_cost, "East"),
    "P124": _p(receipt_cost, "West"),
    "P125": _p(worst_supplier, "Germany"),
    "P126": _p(worst_supplier, "France"),
    "P127": _p(worst_supplier, "Italy"),
    "P128": _p(worst_supplier, "Spain"),
    "P129": _p(recon_stock, "North"),
    "P130": _p(recon_stock, "South"),
    "P131": _p(recon_stock, "East"),
    "P132": _p(recon_stock, "West"),
    "P133": _p(stock_value, "North"),
    "P134": _p(stock_value, "South"),
    "P135": _p(stock_value, "East"),
    "P136": _p(stock_value, "West"),
    "P137": _p(below_reorder_count, "North"),
    "P138": _p(below_reorder_count, "South"),
    "P139": _p(below_reorder_count, "East"),
    "P140": _p(below_reorder_count, "West"),
    "P141": _p(reorder_units, "North"),
    "P142": _p(reorder_units, "South"),
    "P143": _p(reorder_units, "East"),
    "P144": _p(reorder_units, "West"),
    "P145": _p(restockable_pct_reason, "billing_error"),
    "P146": _p(restockable_pct_reason, "customer_return"),
    "P147": _p(restockable_pct_reason, "defect"),
    "P148": _p(restockable_pct_reason, "late_delivery"),
    "P149": _p(recovery_category, "Laptop"),
    "P150": _p(recovery_category, "Phone"),
}
