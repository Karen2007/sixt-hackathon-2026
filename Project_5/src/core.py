"""Loader and calculation helpers for the Project 5 public stage.

Every rule here follows data/public/business_rules.md. Helpers return pandas
DataFrames so each step can be inspected. The reference time is 2025-01-01.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "public" / "datasets"
REF = pd.Timestamp("2025-01-01 00:00:00")


STAGE = os.environ.get("PROJECT5_STAGE", "public")  # "public" = Stage 1 (default); "final" = Stage 2
FINAL_DIR = ROOT / "problem7_student_final" / "datasets"  # local only; never committed (see .gitignore)


def _read(name: str) -> pd.DataFrame:
    """Read one table. Stage 2 tables are renamed to the Stage 1 column names, so the same code runs on both."""
    if STAGE == "final":
        from src.stage2_schema import STAGE2_RENAMES
        df = pd.read_csv(FINAL_DIR / f"{name}.csv", dtype=str, keep_default_na=False)
        return df.rename(columns=STAGE2_RENAMES.get(name, {}))
    return pd.read_csv(DATA_DIR / f"{name}.csv", dtype=str, keep_default_na=False)


def _dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.replace("", None), errors="coerce")


def _bool(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower().eq("true")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("", None), errors="coerce")


@lru_cache(maxsize=1)
def load() -> SimpleNamespace:
    """Read all tables and convert text columns to dates, numbers and booleans."""
    t = SimpleNamespace()

    a = _read("accounts")
    a["is_test_account"] = _bool(a["is_test_account"])
    t.accounts = a

    c = _read("customers")
    c["is_test_customer"] = _bool(c["is_test_customer"])
    c["signup_date"] = _dt(c["signup_date"])
    c["trial_start_date"] = _dt(c["trial_start_date"])
    t.customers = c

    o = _read("orders")
    o["order_date"] = _dt(o["order_date"])
    o["fraud_flag"] = _bool(o["fraud_flag"])
    o["is_duplicate"] = _bool(o["is_duplicate"])
    o["order_level_discount_native"] = _num(o["order_level_discount_native"])
    t.orders = o

    oi = _read("order_items")
    for col in ["quantity", "unit_price_native", "line_discount_native"]:
        oi[col] = _num(oi[col])
    t.order_items = oi

    p = _read("payments")
    p["payment_date"] = _dt(p["payment_date"])
    p["payment_amount_native"] = _num(p["payment_amount_native"])
    t.payments = p

    r = _read("refunds")
    r["refund_date"] = _dt(r["refund_date"])
    r["refund_amount_native"] = _num(r["refund_amount_native"])
    t.refunds = r

    pr = _read("products")
    pr["standard_cost"] = _num(pr["standard_cost"])
    t.products = pr

    h = _read("product_category_history")
    h["valid_from"] = _dt(h["valid_from"])
    h["valid_to"] = _dt(h["valid_to"])
    t.category_history = h

    fx = _read("fx_rates")
    fx["eur_rate"] = _num(fx["eur_rate"])
    t.fx = {(m, cur): rate for m, cur, rate in zip(fx["month"], fx["currency"], fx["eur_rate"])}

    tk = _read("support_tickets")
    tk["opened_at"] = _dt(tk["opened_at"])
    tk["resolved_at"] = _dt(tk["resolved_at"])
    t.tickets = tk

    wl = _read("support_worklogs")
    wl["worked_at"] = _dt(wl["worked_at"])
    wl["minutes_worked"] = _num(wl["minutes_worked"])
    wl["hourly_rate_native"] = _num(wl["hourly_rate_native"])
    t.worklogs = wl

    ri = _read("return_inspections")
    ri["inspected_at"] = _dt(ri["inspected_at"])
    for col in ["units_received", "restockable_units", "recovery_unit_value_eur"]:
        ri[col] = _num(ri[col])
    t.inspections = ri

    s = _read("suppliers")
    s["contract_start"] = _dt(s["contract_start"])
    s["contract_end"] = _dt(s["contract_end"])
    s["is_test_supplier"] = _bool(s["is_test_supplier"])
    t.suppliers = s

    po = _read("purchase_order_lines")
    po["ordered_at"] = _dt(po["ordered_at"])
    po["promised_at"] = _dt(po["promised_at"])
    po["ordered_units"] = _num(po["ordered_units"])
    po["unit_cost_native"] = _num(po["unit_cost_native"])
    t.po_lines = po

    gr = _read("goods_receipts")
    gr["received_at"] = _dt(gr["received_at"])
    gr["accepted_units"] = _num(gr["accepted_units"])
    gr["rejected_units"] = _num(gr["rejected_units"])
    t.receipts = gr

    im = _read("inventory_movements")
    im["event_at"] = _dt(im["event_at"])
    im["units_delta"] = _num(im["units_delta"])
    t.movements = im

    sp = _read("stock_policies")
    sp["valid_from"] = _dt(sp["valid_from"])
    sp["valid_to"] = _dt(sp["valid_to"])
    sp["reorder_point"] = _num(sp["reorder_point"])
    sp["target_units"] = _num(sp["target_units"])
    t.policies = sp

    wi = _read("warehouse_inventory")
    wi["on_hand_units"] = _num(wi["on_hand_units"])
    wi["last_stocktake_date"] = _dt(wi["last_stocktake_date"])
    t.inventory = wi

    t.shipments = _read("shipments")
    t.web_sessions = _read("web_sessions")
    t.marketing_touches = _read("marketing_touches")
    t.sales_targets = _read("sales_targets")
    t.sales_targets["net_revenue_target_eur"] = _num(t.sales_targets["net_revenue_target_eur"])

    return t


# ---------------------------------------------------------------- eligibility

def eligible_customers(t=None) -> pd.DataFrame:
    """Raw customers whose own flag and linked account flag are both non-test."""
    t = t or load()
    c = t.customers.merge(t.accounts[["account_id", "is_test_account"]], on="account_id", how="left")
    return c[~c["is_test_customer"] & ~c["is_test_account"].fillna(False)]


def valid_orders(t=None) -> pd.DataFrame:
    """Distinct valid orders as of the reference time (business_rules.md)."""
    t = t or load()
    elig = set(eligible_customers(t)["customer_id"])
    settled = set(
        t.payments.loc[
            (t.payments["payment_status_detail"] == "settled") & (t.payments["payment_date"] <= REF),
            "order_id",
        ]
    )
    o = t.orders
    mask = (
        (o["order_date"] <= REF)
        & o["order_status"].isin(["completed", "shipped"])
        & (o["payment_status"] == "paid")
        & o["order_id"].isin(settled)
        & ~o["fraud_flag"]
        & ~o["is_duplicate"]
        & o["customer_id"].isin(elig)
    )
    return o[mask].copy()


def fx(month: str, currency: str, t=None) -> float:
    """EUR per unit of currency for a month. EUR is always 1. Missing rate is an error."""
    if currency == "EUR":
        return 1.0
    t = t or load()
    try:
        return t.fx[(month, currency)]
    except KeyError as exc:
        raise KeyError(f"missing FX rate for {month} {currency}") from exc


def fx_series(months: pd.Series, currencies: pd.Series, t=None) -> pd.Series:
    t = t or load()
    return pd.Series(
        [fx(m, c, t) for m, c in zip(months, currencies)], index=months.index, dtype=float
    )


def category_at(dates: pd.Series, product_ids: pd.Series, t=None) -> pd.Series:
    """Product category at the given timestamps (exactly one interval must match)."""
    t = t or load()
    h = t.category_history
    out = pd.Series(index=dates.index, dtype=object)
    for idx, d, pid in zip(dates.index, dates, product_ids):
        m = h[(h["product_id"] == pid) & (h["valid_from"] <= d) & (h["valid_to"] >= d)]
        if len(m) != 1:
            raise ValueError(f"{len(m)} category intervals for {pid} at {d}")
        out.loc[idx] = m["category"].iloc[0]
    return out


# ------------------------------------------------------------ item revenue

def item_revenue(t=None) -> pd.DataFrame:
    """One row per order item of a valid order, with EUR net revenue.

    Net item revenue = gross EUR - line discount EUR - allocated order discount EUR
    - allocated approved refund EUR. Refunds use refund-month FX and are allocated
    across the order's lines for that product (business_rules.md, FX and item revenue).
    """
    t = t or load()
    vo = valid_orders(t)
    items = t.order_items[t.order_items["order_id"].isin(vo["order_id"])].merge(
        vo[["order_id", "customer_id", "order_date", "currency", "order_level_discount_native"]],
        on="order_id",
        how="inner",
    )
    items["month"] = items["order_date"].dt.strftime("%Y-%m")
    items["fx_order"] = fx_series(items["month"], items["currency"], t)

    items["gross_native"] = items["quantity"] * items["unit_price_native"]
    items["pre_native"] = items["gross_native"] - items["line_discount_native"]
    denom = items.groupby("order_id")["pre_native"].transform("sum")
    share = np.where(denom == 0, 0.0, items["pre_native"] / denom.where(denom != 0, 1.0))
    items["order_disc_native"] = items["order_level_discount_native"] * share

    items["gross_eur"] = items["gross_native"] * items["fx_order"]
    items["line_disc_eur"] = items["line_discount_native"] * items["fx_order"]
    items["order_disc_eur"] = items["order_disc_native"] * items["fx_order"]

    # Approved refunds on valid orders, converted at refund-month FX.
    r = t.refunds[
        (t.refunds["refund_status"] == "approved")
        & (t.refunds["refund_date"] <= REF)
        & t.refunds["order_id"].isin(vo["order_id"])
    ].copy()
    r["refund_eur"] = r["refund_amount_native"] * fx_series(
        r["refund_date"].dt.strftime("%Y-%m"), r["currency"], t
    )
    refund_by_key = r.groupby(["order_id", "product_id"])["refund_eur"].sum()

    items["refund_eur_total"] = [
        refund_by_key.get((o, p), 0.0) for o, p in zip(items["order_id"], items["product_id"])
    ]
    w = items["pre_native"].clip(lower=0)
    wsum = w.groupby([items["order_id"], items["product_id"]]).transform("sum")
    n_lines = items.groupby(["order_id", "product_id"])["pre_native"].transform("size")
    share_ref = np.where(wsum == 0, 1.0 / n_lines, w / wsum.where(wsum != 0, 1.0))
    items["refund_alloc_eur"] = items["refund_eur_total"] * share_ref

    items["net_eur"] = (
        items["gross_eur"] - items["line_disc_eur"] - items["order_disc_eur"] - items["refund_alloc_eur"]
    )
    return items


def order_identity_map(t=None) -> pd.DataFrame:
    """Map each eligible raw customer to its resolved real-customer identity and canonical row."""
    t = t or load()
    c = eligible_customers(t).copy()

    def norm(e: str) -> str:
        e = e.strip().lower()
        local, _, domain = e.partition("@")
        local = local.split("+", 1)[0]
        if domain in ("gmail.com", "googlemail.com"):
            local = local.replace(".", "")
            domain = "gmail.com"
        return f"{local}@{domain}"

    c["identity"] = c["email"].map(norm)
    c = c[c["signup_date"] <= REF]  # canonical candidates signed up after reference are excluded
    c = c.sort_values(["identity", "signup_date", "customer_id"])
    canon = c.groupby("identity").first()[["customer_id", "country", "region", "acquisition_channel", "trial_start_date"]]
    canon = canon.rename(columns={"customer_id": "canonical_customer_id"})

    m = eligible_customers(t)[["customer_id", "email", "country", "region", "acquisition_channel"]].copy()
    m["identity"] = m["email"].map(norm)
    m = m.merge(canon, left_on="identity", right_index=True, how="left", suffixes=("_raw", ""))
    return m
