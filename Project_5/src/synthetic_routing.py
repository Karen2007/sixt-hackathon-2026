"""Labelled routing test set written by the team (no organiser question text).

Each of the 50 catalogue functions gets two question templates, worded by us. Slots are filled
with random values from the lists below, using a fixed seed, so the set can be regenerated.
Gold arguments are the values that were put into the slots, in the order the function expects.

The set measures whether the model maps a wording to the right function and arguments. It is a
different test from the public questions, but it is not a random sample of real questions, so its
accuracy is only an estimate for similarly worded questions.

Output: outputs/routing_synthetic_test.csv
"""

from __future__ import annotations

import random
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_router import CATALOGUE, FAMILY_KIND  # noqa: E402

# Values as they appear in the data files (checked against data/public/datasets/*.csv).
VALUES = {
    "country": ["Austria", "France", "Germany", "Greece", "Italy", "Netherlands", "Poland", "Portugal", "Spain", "Switzerland"],
    "region": ["Central Europe", "Southern Europe", "Western Europe"],
    "quarter": ["Q1", "Q2", "Q3", "Q4"],
    "month": ["2024-02", "2024-05", "2024-07", "2024-09", "2024-11"],
    "category": ["Accessory", "Appliance", "Laptop", "Networking", "Phone", "Service", "Software", "Tablet"],
    "segment": ["Enterprise", "Mid-Market", "SMB"],
    "channel": ["Email", "Events", "Organic", "Paid Search", "Partner", "Referral"],
    "warehouse": ["Central", "East", "North", "South", "West"],
    "carrier": ["DHL", "UPS", "DPD", "GLS"],
    "reason": ["billing_error", "customer_return", "defect", "goodwill", "late_delivery"],
    "industry": ["Education", "Finance", "Health", "Logistics", "Manufacturing", "Retail", "Technology"],
    "family": ["Cloud", "Hardware", "Professional Services"],
    "campaign": ["HolidayBundle", "PartnerBoost", "RetentionWave", "SecurityPush", "SpringLaunch", "TrialNudge", "UpgradePush", "Winback"],
    "issue": ["returns", "billing", "delivery"],
    "days": [14, 30, 60],
    "threshold": [5, 10, 15],
}

# (template, slot names in the order the function's arguments need). Slots named "issue" become a list.
TEMPLATES = {
    "rev_country_period": [
        ("What was the total net revenue in EUR for customers in {country} in {quarter} 2024?", ["country", "quarter"]),
        ("Give me the {quarter} 2024 net revenue, in euros, of our customers based in {country}.", ["country", "quarter"]),
    ],
    "rev_region_month": [
        ("What was net revenue in EUR for the {region} region in {month}?", ["region", "month"]),
        ("Report the euro net revenue of customers in {region} for the month {month}.", ["region", "month"]),
    ],
    "growth_country": [
        ("By what percentage did net revenue for customers in {country} change between Q3 and Q4 2024?", ["country"]),
        ("How much did {country} net revenue grow, as a percentage, from the third to the fourth quarter of 2024?", ["country"]),
    ],
    "growth_bool_region": [
        ("Was net revenue in {region} up by at least {threshold_pct} percent from Q3 to Q4 2024?", ["region", "threshold_pct"]),
        ("Did the {region} region grow its net revenue by {threshold_pct}% or more between Q3 and Q4 2024? Answer yes or no.", ["region", "threshold_pct"]),
    ],
    "region_target_pct": [
        ("What percentage of its revenue target did the {region} region reach in {quarter} 2024?", ["region", "quarter"]),
        ("Compare {region} net revenue with its targets for {quarter} 2024. What percentage was achieved?", ["region", "quarter"]),
    ],
    "target_met": [
        ("Did {country} reach its revenue target in {month}? Answer true or false.", ["country", "month"]),
        ("Was the revenue target for {country} met in the month {month}?", ["country", "month"]),
    ],
    "zscore_region_month": [
        ("What was the z-score of {region} net revenue in {month}, compared with its monthly values from January to September 2024?", ["region", "month"]),
        ("How many standard deviations from its usual level was {region} revenue in {month}?", ["region", "month"]),
    ],
    "top_zscore_region": [
        ("Which region had the highest revenue z-score in {month}?", ["month"]),
        ("In {month}, which region stood furthest above its usual revenue level?", ["month"]),
    ],
    "top3_categories_region": [
        ("List the three best-selling product categories by net revenue in {region} during {quarter} 2024.", ["region", "quarter"]),
        ("What were the top three categories in {region} for {quarter} 2024, ranked by net revenue?", ["region", "quarter"]),
    ],
    "top_category_segment": [
        ("Which product category earned the most net revenue from {segment} accounts in {quarter} 2024?", ["segment", "quarter"]),
        ("For {segment} customers, what was the leading category by revenue in {quarter} 2024?", ["segment", "quarter"]),
    ],
    "top_product_region_month": [
        ("Which product brought in the most net revenue in {region} in {month}?", ["region", "month"]),
        ("Name the best-selling product by revenue for the {region} region in {month}.", ["region", "month"]),
    ],
    "refund_rate_category_period": [
        ("What was the approved refund rate for {category} in {quarter} 2024, as a percentage of gross item revenue?", ["category", "quarter"]),
        ("Give the refund percentage of {category} sales in {quarter} 2024.", ["category", "quarter"]),
    ],
    "top_refund_rate_category_country": [
        ("Which product category had the highest approved refund rate among customers in {country}?", ["country"]),
        ("For {country}, which category has the largest share of refunded revenue?", ["country"]),
    ],
    "margin_rate_category": [
        ("What is the margin rate, as a percentage, for the {category} category across all periods?", ["category"]),
        ("How profitable is {category} overall? Give the margin rate in percent.", ["category"]),
    ],
    "top_margin_category": [
        ("Which product category had the highest margin rate in {quarter} 2024?", ["quarter"]),
        ("In {quarter} 2024, what category earned the best margin percentage?", ["quarter"]),
    ],
    "refund_discount_correlation": [
        ("What is the correlation between line discount rate and refund rate across product categories?", []),
        ("Across categories, how strongly do discount rates and refund rates move together?", []),
    ],
    "median_order_net_segment": [
        ("What is the median net revenue per valid order for {segment} accounts?", ["segment"]),
        ("Give the typical (median) order value in euros for {segment} customers.", ["segment"]),
    ],
    "p90_account": [
        ("What is the 90th percentile of lifetime net revenue per account in the {segment} segment?", ["segment"]),
        ("How much lifetime revenue does a top-decile {segment} account generate? Use the 90th percentile.", ["segment"]),
    ],
    "top_account_category_share": [
        ("For the highest-revenue {segment} account, what share of its lifetime revenue came from {category}?", ["segment", "category"]),
        ("What percentage of the biggest {segment} customer's revenue is in the {category} category?", ["segment", "category"]),
    ],
    "at_risk_top_manager": [
        ("Which account manager is responsible for the most lifetime revenue among at-risk customers?", []),
        ("Name the sales manager whose at-risk customers represent the largest lifetime revenue.", []),
    ],
    "top_email_domain": [
        ("Which normalised email domain brings in the most valid-order net revenue?", []),
        ("What is the email domain that contributes the highest net revenue from valid orders?", []),
    ],
    "atrisk_count_country": [
        ("How many high-value customers in {country} are at risk?", ["country"]),
        ("Count the at-risk high-value real customers based in {country}.", ["country"]),
    ],
    "unique_real_customers_country": [
        ("How many distinct real customers are there in {country} after merging duplicate identities?", ["country"]),
        ("Count the unique people who are real customers in {country}, with duplicate emails merged.", ["country"]),
    ],
    "active_count_region": [
        ("How many active customers does the {region} region have?", ["region"]),
        ("Count the currently active real customers in {region}.", ["region"]),
    ],
    "earliest_order_by_channel": [
        ("On what date was the first valid order placed by customers acquired through {channel}?", ["channel"]),
        ("When did the earliest valid order come from a {channel} acquisition?", ["channel"]),
    ],
    "trial_conversion": [
        ("What percentage of {channel} customers ordered within 30 days of starting their trial?", ["channel"]),
        ("Give the trial-to-order conversion rate for the {channel} channel, within a 30-day window.", ["channel"]),
    ],
    "cohort_retention": [
        ("Of the customers whose first valid order was in {month}, what percentage ordered again within {days} days?", ["month", "days"]),
        ("What share of the {month} cohort placed a second order inside {days} days?", ["month", "days"]),
    ],
    "web_direct_conversion": [
        ("What percentage of real customers with a direct web visit ordered within 14 days of that visit?", []),
        ("Give the conversion rate for direct web sessions, measured as orders within 14 days.", []),
    ],
    "campaign_top_attributed": [
        ("Which campaign generated the most attributed net revenue through the {channel} channel?", ["channel"]),
        ("Among {channel} touches, which campaign is credited with the largest revenue?", ["channel"]),
    ],
    "campaign_ticket_revenue": [
        ("What is the lifetime net revenue of customers who saw the {campaign} campaign and later raised a {issue_types} ticket?", ["campaign", "issue_types"]),
        ("Total the lifetime revenue of real customers reached by {campaign} who opened a {issue_types} support ticket.", ["campaign", "issue_types"]),
    ],
    "italy_pre_ticket_top_category": [
        ("For Italian customers, which category generated the most revenue before their first support ticket?", []),
        ("Before each Italian customer's first ticket, what category led revenue?", []),
    ],
    "issue_mode_high_value": [
        ("What is the most common ticket issue type among high-value {segment} accounts?", ["segment"]),
        ("Which issue type appears most often in support tickets from {segment} high-value customers?", ["segment"]),
    ],
    "high_priority_breach_pct": [
        ("What percentage of high-priority tickets breached their service level?", []),
        ("Give the SLA breach rate for high-priority support tickets, in percent.", []),
    ],
    "late_ticket_breach_pct": [
        ("For {region}, what percentage of tickets raised within 14 days of a late delivery breached the service level?", ["region"]),
        ("Among tickets opened within two weeks of a late shipment in {region}, what share breached SLA?", ["region"]),
    ],
    "accounts_late_breach": [
        ("How many accounts in the {industry} industry had both a late shipment and an SLA-breaching ticket?", ["industry"]),
        ("Count the {industry} accounts with at least one late delivery and at least one breached ticket.", ["industry"]),
    ],
    "carrier_wdelay": [
        ("What is the revenue-weighted average delay, in days, of late shipments handled by {carrier}?", ["carrier"]),
        ("Give the weighted mean lateness in days for {carrier} deliveries that were late.", ["carrier"]),
    ],
    "top_carrier_by_wdelay": [
        ("Which carrier has the largest revenue-weighted average delay on late shipments in {region}?", ["region"]),
        ("In {region}, which carrier is most late on average, weighting by revenue?", ["region"]),
    ],
    "commitment_country": [
        ("What is the total eligible purchase commitment, in euros, with suppliers in {country}?", ["country"]),
        ("Sum the purchase commitments for suppliers based in {country}, in EUR.", ["country"]),
    ],
    "outstanding_units": [
        ("How many units are still outstanding on eligible purchase lines for the {warehouse} warehouse?", ["warehouse"]),
        ("Count the open units not yet received for warehouse {warehouse}.", ["warehouse"]),
    ],
    "ontime_pct": [
        ("What percentage of units due from suppliers in {country} were accepted on time?", ["country"]),
        ("Give the on-time acceptance rate, in percent, for {country} suppliers.", ["country"]),
    ],
    "rejection_pct": [
        ("What percentage of received units from {country} suppliers were rejected?", ["country"]),
        ("Give the rejected-unit rate for suppliers in {country}.", ["country"]),
    ],
    "worst_supplier": [
        ("Which supplier in {country} has the highest rejection ratio?", ["country"]),
        ("Name the {country} supplier with the most rejected units relative to its deliveries.", ["country"]),
    ],
    "ppv": [
        ("What is the purchase price variance in EUR for the {family} product family?", ["family"]),
        ("How much did {family} cost over or under standard, in euros?", ["family"]),
    ],
    "receipt_cost": [
        ("What is the euro cost of accepted units on posted receipts at the {warehouse} warehouse?", ["warehouse"]),
        ("Total the cost of accepted goods received into warehouse {warehouse}, in EUR.", ["warehouse"]),
    ],
    "recon_stock": [
        ("How many units are in reconciled stock at the {warehouse} warehouse at the reference time?", ["warehouse"]),
        ("Give the reconciled stock quantity for warehouse {warehouse} at the reference date.", ["warehouse"]),
    ],
    "stock_value": [
        ("What is the standard-cost value in EUR of reconciled stock in the {warehouse} warehouse?", ["warehouse"]),
        ("How much is the {warehouse} warehouse's reconciled stock worth at standard cost?", ["warehouse"]),
    ],
    "below_reorder_count": [
        ("How many product and warehouse pairs at {warehouse} are below their reorder point?", ["warehouse"]),
        ("Count the stock lines at {warehouse} whose quantity is under the reorder level.", ["warehouse"]),
    ],
    "reorder_units": [
        ("What is the total suggested replenishment quantity for the {warehouse} warehouse?", ["warehouse"]),
        ("How many units should be reordered in total for {warehouse}?", ["warehouse"]),
    ],
    "restockable_pct_reason": [
        ("What percentage of returned units with the reason '{reason}' can be put back into stock?", ["reason"]),
        ("For returns citing '{reason}', give the share of units that are restockable, in percent.", ["reason"]),
    ],
    "recovery_category": [
        ("What is the total recovery value in EUR of inspected returns for the {category} category?", ["category"]),
        ("How much value can be recovered from inspected {category} returns, in euros?", ["category"]),
    ],
}

POOL_KEY = {"threshold_pct": "threshold", "issue_types": "issue"}


def _fill(fn: str, template: str, slots: list[str], rng: random.Random):
    values = {s: rng.choice(VALUES[POOL_KEY.get(s, s)]) for s in slots}
    text = template.format(**values)
    args = []
    for name in [x.strip() for x in CATALOGUE[fn][0].split(",") if x.strip()]:
        v = values[name]
        args.append([v] if name.endswith("_types") else v)
    return text, args


def build(n_per_function: int = 10, seed: int = 7) -> pd.DataFrame:
    assert set(TEMPLATES) == set(CATALOGUE), "every catalogue function needs templates"
    rng = random.Random(seed)
    rows = []
    for fn in CATALOGUE:
        templates = TEMPLATES[fn]
        for i in range(n_per_function):
            tpl, slots = templates[i % len(templates)]
            text, args = _fill(fn, tpl, slots, rng)
            rows.append({"question": text, "gold_function": fn, "gold_args": repr(args),
                         "answer_type": FAMILY_KIND[fn], "template_index": i % len(templates)})
    return pd.DataFrame(rows)


def similarity_to_public(df: pd.DataFrame) -> pd.Series:
    """Highest text similarity of each synthetic question to any public question (0-1)."""
    public = pd.read_csv(ROOT / "data" / "public" / "public_leaderboard_questions.csv", dtype=str)["question"].tolist()
    out = []
    for q in df.question:
        out.append(max(SequenceMatcher(None, q.lower(), p.lower()).ratio() for p in public))
    return pd.Series(out, index=df.index)


if __name__ == "__main__":
    df = build()
    df["max_similarity_to_public"] = similarity_to_public(df).round(3)
    out = ROOT / "outputs" / "routing_synthetic_test.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} with {len(df)} questions")
    print("max similarity to any public question:", df.max_similarity_to_public.max())
    print("questions above 0.8 similarity:", int((df.max_similarity_to_public > 0.8).sum()))
