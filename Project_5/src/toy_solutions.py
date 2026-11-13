"""Hand-written pandas solutions for the three toy examples (DEMO001-003).

These are reference implementations, not the competition pipeline. Their job is
to prove that the data, the rules and the output format line up before any LLM
is involved.
"""

from pathlib import Path

import pandas as pd

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"


def demo001_valid_orders(examples: Path = EXAMPLES) -> int:
    """Count orders that are completed/shipped AND have >= 1 settled payment.

    Eligibility comes from payments, not from the order's 'payment_status' flag.
    Counted by distinct order_id so that an order with two payments counts once.
    """
    orders = pd.read_csv(examples / "orders.csv")
    payments = pd.read_csv(examples / "payments.csv")

    settled_ids = set(payments.loc[payments["status"] == "settled", "order_id"])
    valid = orders[
        orders["status"].isin(["completed", "shipped"])
        & orders["order_id"].isin(settled_ids)
    ]
    return int(valid["order_id"].nunique())


def demo002_net_revenue_eur(examples: Path = EXAMPLES) -> str:
    """Net revenue in EUR: discounted sale at order-month rate minus refund at refund-month rate.

    Rates are applied per event (sale vs. refund). Rounding happens only once, at the end.
    """
    row = pd.read_csv(examples / "revenue_inputs.csv").iloc[0]

    discounted_usd = (
        row["quantity"] * row["unit_price_usd"]
        - row["line_discount_usd"]
        - row["order_discount_usd"]
    )
    sale_eur = discounted_usd * row["order_eur_per_usd"]
    refund_eur = row["refund_usd"] * row["refund_eur_per_usd"]
    net = round(sale_eur - refund_eur, 2)
    return f"{net:.2f}"


def demo003_units_to_reorder(examples: Path = EXAMPLES) -> int:
    """Replenishment units under a strict-threshold rule.

    Only posted movements inside the stocktake reference window count; drafts and
    movements after the reference time are excluded. Reorder only if end stock is
    strictly below the reorder point, and then top up to the target.
    """
    moves = pd.read_csv(examples / "stock_movements.csv")
    policy = pd.read_csv(examples / "stock_policy.csv").iloc[0]

    in_scope = (moves["status"] == "posted") & (moves["within_stocktake_reference_window"])
    end_stock = policy["opening_stock"] + moves.loc[in_scope, "units_delta"].sum()

    if end_stock < policy["reorder_point"]:
        return int(policy["target_units"] - end_stock)
    return 0


def solve_all(examples: Path = EXAMPLES) -> pd.DataFrame:
    """Return the answers in the official submission format: question_id, answer."""
    return pd.DataFrame(
        {
            "question_id": ["DEMO001", "DEMO002", "DEMO003"],
            "answer": [
                str(demo001_valid_orders(examples)),
                demo002_net_revenue_eur(examples),
                str(demo003_units_to_reorder(examples)),
            ],
        }
    )


if __name__ == "__main__":
    OUTPUTS.mkdir(exist_ok=True)
    answers = solve_all()
    out = OUTPUTS / "demo_answers_pred.csv"
    answers.to_csv(out, index=False)
    print(answers.to_string(index=False))
    print(f"written: {out}")
