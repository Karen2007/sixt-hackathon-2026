"""Independent cross-check of a sample of company answers, written in SQL (DuckDB).

The pandas code in src/company.py and this SQL were written separately, so a coding
mistake would have to appear in both to go unnoticed. Interpretation errors would
appear in both and are NOT caught here.
"""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "datasets"
ANSWERS = ROOT / "outputs" / "company_answers_p001_p150.csv"
REF = "DATE '2025-01-01'"

TABLES = {
    "orders": "orders", "order_items": "order_items", "payments": "payments", "refunds": "refunds",
    "products": "products", "pch": "product_category_history", "customers": "customers",
    "accounts": "accounts", "fx": "fx_rates", "targets": "sales_targets", "po": "purchase_order_lines",
    "suppliers": "suppliers", "receipts": "goods_receipts", "inv": "warehouse_inventory",
    "mv": "inventory_movements", "policies": "stock_policies", "insp": "return_inspections",
}

BASE = f"""
WITH vo AS (
  SELECT o.order_id, o.customer_id, o.order_date, o.currency, o.order_level_discount_native
  FROM orders o
  JOIN customers c ON c.customer_id = o.customer_id
  JOIN accounts a ON a.account_id = c.account_id
  WHERE o.order_date <= {REF} AND o.order_status IN ('completed','shipped') AND o.payment_status = 'paid'
    AND o.fraud_flag = false AND o.is_duplicate = false
    AND c.is_test_customer = false AND a.is_test_account = false
    AND o.order_id IN (SELECT order_id FROM payments
                       WHERE payment_status_detail = 'settled' AND payment_date <= {REF})
),
it AS (
  SELECT oi.order_item_id, oi.order_id, oi.product_id, vo.customer_id, vo.order_date,
         strftime(vo.order_date, '%Y-%m') AS m, vo.currency, oi.quantity,
         oi.quantity * oi.unit_price_native AS gross_n,
         oi.quantity * oi.unit_price_native - oi.line_discount_native AS pre_n,
         oi.line_discount_native AS ld_n, vo.order_level_discount_native AS od_n
  FROM order_items oi JOIN vo USING (order_id)
),
it2 AS (
  SELECT *, COALESCE(od_n * pre_n / NULLIF(SUM(pre_n) OVER (PARTITION BY order_id), 0), 0) AS od_share_n
  FROM it
),
fxo AS (
  SELECT it2.*, f.eur_rate AS fx_o, p.standard_cost, p.cost_currency, p.product_name, p.family
  FROM it2
  JOIN fx f ON f.month = it2.m AND f.currency = it2.currency
  JOIN products p ON p.product_id = it2.product_id
),
refsum AS (
  SELECT r.order_id, r.product_id, SUM(r.refund_amount_native * f.eur_rate) AS ref_total
  FROM refunds r
  JOIN fx f ON f.month = strftime(r.refund_date, '%Y-%m') AND f.currency = r.currency
  WHERE r.refund_status = 'approved' AND r.refund_date <= {REF}
    AND r.order_id IN (SELECT order_id FROM vo)
  GROUP BY 1, 2
),
alloc AS (
  SELECT fxo.*, COALESCE(refsum.ref_total, 0) AS ref_total,
         GREATEST(pre_n, 0) AS w,
         SUM(GREATEST(pre_n, 0)) OVER (PARTITION BY order_id, product_id) AS wsum,
         COUNT(*) OVER (PARTITION BY order_id, product_id) AS n_lines
  FROM fxo LEFT JOIN refsum USING (order_id, product_id)
),
net AS (
  SELECT alloc.*, c.country, c.region, cat.category AS order_cat,
         (gross_n * fx_o) AS gross_eur,
         (gross_n - ld_n - od_share_n) * fx_o AS sale_eur,
         ref_total * CASE WHEN wsum = 0 THEN 1.0 / n_lines ELSE w / wsum END AS refund_alloc_eur,
         quantity * standard_cost * f2.eur_rate AS cost_eur
  FROM alloc
  JOIN customers c ON c.customer_id = alloc.customer_id
  JOIN product_category AS cat ON 1 = 0
  JOIN fx f2 ON f2.month = alloc.m AND f2.currency = alloc.cost_currency
)
"""
# NOTE: the category join is supplied in the query text below via a view, see build().


def build(con):
    for name, file in TABLES.items():
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{DATA / (file + '.csv')}', header=true)")
    con.execute("""
        CREATE VIEW product_category AS
        SELECT * FROM pch
    """)


def net_sql() -> str:
    # Replace the placeholder category join with a real interval join.
    sql = BASE.replace(
        "JOIN product_category AS cat ON 1 = 0",
        "JOIN pch cat ON cat.product_id = alloc.product_id AND cat.valid_from <= alloc.order_date AND alloc.order_date <= cat.valid_to",
    )
    sql = sql.replace(
        "net AS (\n  SELECT alloc.*, c.country, c.region, cat.category AS order_cat,\n         (gross_n * fx_o) AS gross_eur,\n         (gross_n - ld_n - od_share_n) * fx_o AS sale_eur,\n         ref_total * CASE WHEN wsum = 0 THEN 1.0 / n_lines ELSE w / wsum END AS refund_alloc_eur,\n         quantity * standard_cost * f2.eur_rate AS cost_eur\n  FROM alloc",
        "net AS (\n  SELECT alloc.*, c.country, c.region, cat.category AS order_cat,\n         (gross_n * fx_o) AS gross_eur,\n         (gross_n - ld_n - od_share_n) * fx_o AS sale_eur,\n         ref_total * CASE WHEN wsum = 0 THEN 1.0 / n_lines ELSE w / wsum END AS refund_alloc_eur,\n         quantity * standard_cost * f2.eur_rate AS cost_eur\n  FROM alloc",
    )
    return sql


def q(con, body: str):
    return con.execute(net_sql() + body).fetchall()


def scalar(con, body: str):
    rows = q(con, body)
    return rows[0][0]


def fmt2(x):
    return "N/A" if x is None else f"{round(float(x), 2):.2f}"


def main():
    con = duckdb.connect()
    build(con)
    ans = pd.read_csv(ANSWERS, dtype=str, keep_default_na=False).set_index("question_id")["answer"]

    results = {}
    # Net revenue families (net_eur = sale_eur - refund_alloc_eur).
    results["P005 Greece Q2 revenue"] = fmt2(scalar(con,
        "SELECT SUM(sale_eur - refund_alloc_eur) FROM net WHERE country='Greece' AND order_date >= DATE '2024-04-01' AND order_date < DATE '2024-07-01'"))
    results["P013 Germany Q3 revenue"] = fmt2(scalar(con,
        "SELECT SUM(sale_eur - refund_alloc_eur) FROM net WHERE country='Germany' AND order_date >= DATE '2024-07-01' AND order_date < DATE '2024-10-01'"))
    results["P017 Central Europe 2024-05 revenue"] = fmt2(scalar(con,
        "SELECT SUM(sale_eur - refund_alloc_eur) FROM net WHERE region='Central Europe' AND strftime(order_date,'%Y-%m')='2024-05'"))
    q3 = scalar(con, "SELECT SUM(sale_eur - refund_alloc_eur) FROM net WHERE country='Italy' AND order_date >= DATE '2024-07-01' AND order_date < DATE '2024-10-01'")
    q4 = scalar(con, "SELECT SUM(sale_eur - refund_alloc_eur) FROM net WHERE country='Italy' AND order_date >= DATE '2024-10-01' AND order_date < DATE '2025-01-01'")
    results["P007 Italy Q3->Q4 growth"] = fmt2(100 * (q4 - q3) / q3)
    results["P006 Accessory refund rate Q3"] = fmt2(100 * scalar(con,
        "SELECT SUM(refund_alloc_eur)/SUM(gross_eur) FROM net WHERE order_cat='Accessory' AND order_date >= DATE '2024-07-01' AND order_date < DATE '2024-10-01'"))
    results["P004 Phone margin rate %"] = fmt2(100 * scalar(con,
        "SELECT (SUM(sale_eur - refund_alloc_eur) - SUM(cost_eur)) / SUM(sale_eur - refund_alloc_eur) FROM net WHERE order_cat='Phone'"))
    results["P015 Western Europe Q1 % of target"] = fmt2(100 * scalar(con,
        "SELECT SUM(sale_eur - refund_alloc_eur) FROM net WHERE region='Western Europe' AND order_date >= DATE '2024-01-01' AND order_date < DATE '2024-04-01'")
        / scalar(con, "SELECT SUM(net_revenue_target_eur) FROM targets WHERE region='Western Europe' AND month IN ('2024-01','2024-02','2024-03')"))

    # Central Europe z-score for 2024-10 against Jan-Sep monthly revenue (population sd).
    monthly = con.execute(net_sql() + """
        SELECT strftime(order_date,'%Y-%m') m, SUM(sale_eur - refund_alloc_eur) s FROM net
        WHERE region='Central Europe' GROUP BY 1""").fetchall()
    mv = {m: s for m, s in monthly}
    base = [float(mv.get(f"2024-{i:02d}", 0.0)) for i in range(1, 10)]
    import statistics
    z = (float(mv.get("2024-10", 0.0)) - statistics.fmean(base)) / statistics.pstdev(base)
    results["P022 Central Europe z 2024-10"] = fmt2(z)

    # Procurement: eligible purchase lines (open/closed, not cancelled, eligible supplier).
    elig_po = """
        SELECT po.*, s.country AS sup_country FROM po po JOIN suppliers s USING (supplier_id)
        WHERE po.status IN ('open','closed') AND po.ordered_at <= TIMESTAMP '2025-01-01 00:00:00'
          AND s.is_test_supplier = false AND s.contract_start <= po.ordered_at AND po.ordered_at < s.contract_end
    """
    res_pc = con.execute(f"""
        WITH ep AS ({elig_po})
        SELECT SUM(ordered_units*unit_cost_native*f.eur_rate) FROM ep
        JOIN fx f ON f.month = strftime(ep.ordered_at,'%Y-%m') AND f.currency = ep.currency
        WHERE sup_country='Germany'""").fetchone()[0]
    results["P101 Germany commitment EUR"] = fmt2(res_pc)

    res_ot = con.execute(f"""
        WITH ep AS ({elig_po}),
        er AS (
          SELECT r.*, ep.ordered_units, ep.promised_at, ep.sup_country FROM receipts r JOIN ep USING (po_line_id)
          WHERE r.status='posted' AND r.received_at >= ep.ordered_at AND r.received_at <= TIMESTAMP '2025-01-01 00:00:00'
        ),
        due AS (SELECT * FROM ep WHERE promised_at <= TIMESTAMP '2025-01-01 00:00:00' AND sup_country='Germany'),
        ontime AS (
          SELECT due.po_line_id, due.ordered_units, COALESCE(SUM(er.accepted_units),0) AS acc_ontime
          FROM due LEFT JOIN er ON er.po_line_id = due.po_line_id AND er.received_at <= due.promised_at
          GROUP BY 1,2
        )
        SELECT 100 * SUM(LEAST(acc_ontime, ordered_units)) / SUM(ordered_units) FROM ontime
    """).fetchone()[0]
    results["P109 Germany on-time %"] = fmt2(res_ot)

    res_rj = con.execute(f"""
        WITH ep AS ({elig_po}),
        er AS (
          SELECT r.*, ep.sup_country FROM receipts r JOIN ep USING (po_line_id)
          WHERE r.status='posted' AND r.received_at >= ep.ordered_at AND r.received_at <= TIMESTAMP '2025-01-01 00:00:00'
        )
        SELECT 100 * SUM(rejected_units) / (SUM(accepted_units) + SUM(rejected_units)) FROM er WHERE sup_country='Germany'
    """).fetchone()[0]
    results["P113 Germany rejected %"] = fmt2(res_rj)

    # Stock: end stock per (product, warehouse) and reorder units for North at the reference.
    res_ro = con.execute(f"""
        WITH endst AS (
          SELECT i.product_id, i.warehouse_region, i.on_hand_units +
                 COALESCE((SELECT SUM(m.units_delta) FROM mv m
                           WHERE m.product_id = i.product_id AND m.warehouse_region = i.warehouse_region
                             AND m.status='posted' AND m.event_at > i.last_stocktake_date
                             AND m.event_at <= TIMESTAMP '2025-01-01 00:00:00'), 0) AS end_stock
          FROM inv i
        ),
        pol AS (
          SELECT * FROM policies WHERE valid_from <= TIMESTAMP '2025-01-01 00:00:00' AND TIMESTAMP '2025-01-01 00:00:00' < valid_to
        )
        SELECT SUM(GREATEST(p.target_units - e.end_stock, 0))
        FROM endst e JOIN pol p ON p.product_id = e.product_id AND p.warehouse_region = e.warehouse_region
        WHERE e.warehouse_region = 'North' AND e.end_stock < p.reorder_point
    """).fetchone()[0]
    results["P141 North reorder units"] = str(int(res_ro))

    # Returns: restockable share for billing_error refunds on valid order/product pairs.
    res_rs = con.execute(f"""
        WITH vo AS (
          SELECT o.order_id FROM orders o
          JOIN customers c ON c.customer_id = o.customer_id JOIN accounts a ON a.account_id = c.account_id
          WHERE o.order_date <= {REF} AND o.order_status IN ('completed','shipped') AND o.payment_status='paid'
            AND o.fraud_flag=false AND o.is_duplicate=false AND c.is_test_customer=false AND a.is_test_account=false
            AND o.order_id IN (SELECT order_id FROM payments WHERE payment_status_detail='settled' AND payment_date <= {REF})
        ),
        rf AS (
          SELECT r.refund_id FROM refunds r
          WHERE r.refund_status='approved' AND r.refund_date <= {REF} AND r.refund_reason='billing_error'
            AND r.order_id IN (SELECT order_id FROM vo)
            AND EXISTS (SELECT 1 FROM order_items oi WHERE oi.order_id = r.order_id AND oi.product_id = r.product_id)
        )
        SELECT 100 * SUM(insp.restockable_units) / SUM(insp.units_received)
        FROM insp
        JOIN refunds rr ON rr.refund_id = insp.refund_id
        JOIN rf ON rf.refund_id = insp.refund_id
        WHERE insp.status = 'completed' AND insp.inspected_at <= {REF}
          AND insp.inspected_at >= rr.refund_date
    """).fetchone()[0]
    results["P145 billing_error restockable %"] = fmt2(res_rs)

    print(f"{'check':40} {'SQL':>14} {'answers CSV':>14}  match")
    mapping = {
        "P005": "P005 Greece Q2 revenue", "P013": "P013 Germany Q3 revenue", "P017": "P017 Central Europe 2024-05 revenue",
        "P007": "P007 Italy Q3->Q4 growth", "P006": "P006 Accessory refund rate Q3", "P004": "P004 Phone margin rate %",
        "P015": "P015 Western Europe Q1 % of target", "P022": "P022 Central Europe z 2024-10",
        "P101": "P101 Germany commitment EUR", "P109": "P109 Germany on-time %", "P113": "P113 Germany rejected %",
        "P141": "P141 North reorder units", "P145": "P145 billing_error restockable %",
    }
    ok = 0
    for qid, label in mapping.items():
        sql_val = results[label]
        csv_val = ans[qid]
        match = sql_val == csv_val
        ok += match
        print(f"{label:40} {sql_val:>14} {csv_val:>14}  {'OK' if match else 'DIFF'}")
    print(f"\nagreement: {ok} / {len(mapping)}")


if __name__ == "__main__":
    main()
