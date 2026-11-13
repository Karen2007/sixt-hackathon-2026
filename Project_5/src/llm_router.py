"""LLM router for the company questions (P001-P150).

The model does NOT compute any answer. It reads a question and chooses which
calculation function to run and with which arguments, as JSON. The function then
computes the number in src/company.py. The router is checked against the
hand-written routing (QUESTIONS_P[*].plan) and every model call is logged.

Model: Qwen2.5-Coder-7B-Instruct (dense, 7,615,616,512 parameters, Apache 2.0),
GGUF Q4_K_M quantisation, run on CPU with llama-cpp-python.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_PATH = Path("/home/karen/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF (qwen2.5-coder-7b-instruct-q4_k_m.gguf)"
MODEL_PARAMS = 7_615_616_512
LOG_DIR = ROOT / "data" / "public" / "llm_logs"  # inside the gitignored data folder

CATALOGUE = {
    "rev_country_period": ("country, quarter", "Net revenue in EUR for customers in a country in a quarter (Q1-Q4 2024)."),
    "rev_region_month": ("region, month", "Net revenue in EUR for a customer region in a month (YYYY-MM)."),
    "growth_country": ("country", "Percentage growth in country net revenue from Q3 to Q4 2024."),
    "growth_bool_region": ("region, threshold_pct", "true if region net revenue grew from Q3 to Q4 2024 by at least threshold_pct percent."),
    "region_target_pct": ("region, quarter", "Region net revenue as a percentage of its summed monthly revenue targets in a quarter."),
    "target_met": ("country, month", "true if a country met or exceeded its revenue target in a month."),
    "zscore_region_month": ("region, month", "Z-score of a region's net revenue in a month against its Jan-Sep 2024 monthly values."),
    "top_zscore_region": ("month", "Region with the highest net-revenue z-score in a month."),
    "top3_categories_region": ("region, quarter", "Top 3 product categories by net revenue in a region and quarter, separated by semicolons."),
    "top_category_segment": ("segment, quarter", "Top product category by net revenue for accounts in a segment in a quarter."),
    "top_product_region_month": ("region, month", "Top product name by net revenue for a region in a month."),
    "refund_rate_category_period": ("category, quarter", "Approved-refund rate (percent of gross item revenue) for a product category in a quarter."),
    "top_refund_rate_category_country": ("country", "Product category with the highest approved-refund rate for customers in a country."),
    "margin_rate_category": ("category", "Margin rate (percent) for a product category over all time."),
    "top_margin_category": ("quarter", "Product category with the highest margin rate in a quarter."),
    "refund_discount_correlation": ("", "Pearson correlation across categories between line discount rate and refund rate."),
    "median_order_net_segment": ("segment", "Median net revenue per valid order for accounts in a segment."),
    "p90_account": ("segment", "90th percentile of account lifetime net revenue for accounts in a segment."),
    "top_account_category_share": ("segment, category", "Share (percent) of the highest-lifetime account's revenue in a segment that comes from a category."),
    "at_risk_top_manager": ("", "Account manager owning the most lifetime revenue among at-risk real customers."),
    "top_email_domain": ("", "Normalised email domain contributing the most valid-order net revenue."),
    "atrisk_count_country": ("country", "Number of high-value at-risk real customers in a country."),
    "unique_real_customers_country": ("country", "Number of unique real customers (identity-resolved) in a country."),
    "active_count_region": ("region", "Number of active real customers in a region."),
    "earliest_order_by_channel": ("channel", "Earliest valid order date for customers acquired through a channel."),
    "trial_conversion": ("channel", "Percentage of real customers in a channel who ordered within 30 days after trial start."),
    "cohort_retention": ("month, days", "Percentage of real customers whose first valid order was in a month and who ordered again within N days."),
    "web_direct_conversion": ("", "Percentage of real customers with a direct web session who ordered within 14 days after it."),
    "campaign_top_attributed": ("channel", "Campaign with the highest attributed net revenue among touches of a marketing channel."),
    "campaign_ticket_revenue": ("campaign, issue_types", "Lifetime net revenue of real customers with a touch for a campaign and a ticket of one of the issue types."),
    "italy_pre_ticket_top_category": ("", "Top category by net revenue before each Italian customer's first support ticket."),
    "issue_mode_high_value": ("segment", "Most common ticket issue type for high-value accounts in a segment."),
    "high_priority_breach_pct": ("", "Percentage of high-priority tickets that breached SLA."),
    "late_ticket_breach_pct": ("region", "Percentage of tickets within 14 days after a late delivery that breached SLA, for a region."),
    "accounts_late_breach": ("industry", "Number of accounts in an industry with at least one late shipment and at least one SLA-breaching ticket."),
    "carrier_wdelay": ("carrier", "Net-revenue-weighted average delay in days of late shipments for a carrier."),
    "top_carrier_by_wdelay": ("region", "Carrier with the highest net-revenue-weighted average delay of late shipments in a region."),
    "commitment_country": ("country", "Total eligible purchase commitment in EUR for suppliers in a country."),
    "outstanding_units": ("warehouse", "Units still outstanding across eligible purchase lines for a warehouse."),
    "ontime_pct": ("country", "Percentage of units on due purchase lines accepted on time, for suppliers in a country."),
    "rejection_pct": ("country", "Rejected-unit percentage across eligible receipts for suppliers in a country."),
    "worst_supplier": ("country", "Supplier name with the highest rejection ratio for a country."),
    "ppv": ("family", "Purchase price variance in EUR for a product family, or null for all families."),
    "receipt_cost": ("warehouse", "EUR cost of accepted units on eligible posted receipts for a warehouse."),
    "recon_stock": ("warehouse", "Total reconciled stock units at the reference time for a warehouse."),
    "stock_value": ("warehouse", "Standard-cost EUR value of reconciled stock for a warehouse."),
    "below_reorder_count": ("warehouse", "Number of product/warehouse pairs strictly below their reorder point."),
    "reorder_units": ("warehouse", "Total suggested replenishment units for a warehouse."),
    "restockable_pct_reason": ("reason", "Restockable percentage of units from eligible inspections for a refund reason."),
    "recovery_category": ("category", "Total recovery value in EUR of eligible inspections for an order-time category."),
}

# Output type of each function. Moved here from pipeline_company.py so the prompt can state it.
FAMILY_KIND = {
    "accounts_late_breach": "integer", "active_count_region": "integer", "at_risk_top_manager": "string",
    "atrisk_count_country": "integer", "below_reorder_count": "integer", "campaign_ticket_revenue": "number",
    "campaign_top_attributed": "string", "carrier_wdelay": "number", "cohort_retention": "number",
    "commitment_country": "number", "earliest_order_by_channel": "date", "growth_bool_region": "boolean",
    "growth_country": "number", "high_priority_breach_pct": "number", "issue_mode_high_value": "string",
    "italy_pre_ticket_top_category": "string", "late_ticket_breach_pct": "number", "margin_rate_category": "number",
    "median_order_net_segment": "number", "ontime_pct": "number", "outstanding_units": "integer",
    "p90_account": "number", "ppv": "number", "receipt_cost": "number", "recon_stock": "integer",
    "recovery_category": "number", "refund_discount_correlation": "number",
    "refund_rate_category_period": "number", "region_target_pct": "number", "rejection_pct": "number",
    "reorder_units": "integer", "restockable_pct_reason": "number", "rev_country_period": "number",
    "rev_region_month": "number", "stock_value": "number", "target_met": "boolean",
    "top3_categories_region": "string", "top_account_category_share": "number",
    "top_carrier_by_wdelay": "string", "top_category_segment": "string", "top_email_domain": "string",
    "top_margin_category": "string", "top_product_region_month": "string",
    "top_refund_rate_category_country": "string", "top_zscore_region": "string", "trial_conversion": "number",
    "unique_real_customers_country": "integer", "web_direct_conversion": "number", "worst_supplier": "string",
    "zscore_region_month": "number",
}

KIND_TEXT = {
    "integer": "an integer count",
    "number": "a number",
    "string": "a text value (a name or label)",
    "date": "a date",
    "boolean": "true or false",
}

# Worked examples in the prompt. Each wording was checked to be clearly different from the public
# questions (similarity below 0.5 by difflib) and from the synthetic test templates (below 0.75).
EXAMPLES = [
    ("Net revenue in euros from our Greek customers in Q1 2024?", "rev_country_period", ["Greece", "Q1"]),
    ("How much did net revenue from Hungarian customers grow, in percent, between the third and fourth quarters of 2024?", "growth_country", ["Hungary"]),
    ("Was the Northern Europe region at least 8% higher in Q4 2024 than in Q3 2024?", "growth_bool_region", ["Northern Europe", 8]),
    ("How many at-risk real customers with high value are there in Portugal?", "atrisk_count_country", ["Portugal"]),
    ("Name the supplier in Austria that rejects the most goods.", "worst_supplier", ["Austria"]),
]


def _arg_count(sig: str) -> int:
    return len([p for p in sig.split(",") if p.strip()]) if sig else 0


SYSTEM = (
    "You route business questions to one calculation function. You never compute the answer yourself. "
    "Reply with one JSON object only: {\"function\": <name>, \"args\": [<arguments in the order listed>]}. "
    "Use only function names from the catalogue. Take arguments from the question text. "
    "Match the question's wording to the output type: 'how many' asks for an integer count, "
    "'which' or 'who' asks for a name. "
    "Quarters are Q1, Q2, Q3, Q4. Months are YYYY-MM. Use null for a missing argument.\n\nCATALOGUE:\n"
    + "\n".join(
        f"- {name}({sig}) returns {KIND_TEXT[FAMILY_KIND[name]]}: {desc}"
        for name, (sig, desc) in CATALOGUE.items()
    )
    + "\n\nEXAMPLES:\n"
    + "".join(
        f"Q: {q}\nA: {{\"function\": \"{fn}\", \"args\": {json.dumps(args)}}}\n"
        for q, fn, args in EXAMPLES
    )
)


def build_grammar(kinds: set[str]) -> str:
    """GBNF grammar: the output must be one JSON object whose function name is one of the catalogue names.
    Argument counts are checked afterwards by the validator: a per-function argument grammar made
    constrained decoding too slow on CPU."""
    names = " | ".join('"\\"' + name + '\\""' for name in CATALOGUE if FAMILY_KIND[name] in kinds)
    rules = [
        'root ::= "{" ws "\\"function\\"" ws ":" ws fname ws "," ws "\\"args\\"" ws ":" ws args ws "}"',
        "fname ::= " + names,
        'args ::= "[" ws (val ws ("," ws val ws)*)? "]"',
        'val ::= str | num | "null" | strlist',
        'str ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""',
        'num ::= "-"? [0-9]+ ("." [0-9]+)?',
        'strlist ::= "[" ws (str ws ("," ws str ws)*)? "]"',
        "ws ::= [ \\t\\n]*",
    ]
    return "\n".join(rules) + "\n"




def question_prompt(question: str) -> str:
    return f"Question: {question}\nA:"


def _parse(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "function" not in obj:
        return None
    return {"function": obj.get("function"), "args": obj.get("args", [])}


class Router:
    def __init__(self, n_threads: int = 16, n_ctx: int = 4096):
        from llama_cpp import Llama, LlamaGrammar  # imported here so the rest of the project does not need it

        self.llm = Llama(model_path=str(MODEL_PATH), n_ctx=n_ctx, n_threads=n_threads,
                         verbose=False, seed=0)
        self._grammars = {}
        self._LlamaGrammar = LlamaGrammar
        self.model_sha256 = None
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / "route_calls.jsonl"

    def grammar_for(self, answer_type: str | None):
        if answer_type is None:
            return None
        if answer_type not in self._grammars:
            kinds = {answer_type}
            self._grammars[answer_type] = self._LlamaGrammar.from_string(build_grammar(kinds))
        return self._grammars[answer_type]

    def route(self, question: str, qid: str | None = None, answer_type: str | None = None) -> dict:
        t0 = time.time()
        grammar = self.grammar_for(answer_type)
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": question_prompt(question)}],
            temperature=0.0, max_tokens=120, grammar=grammar,
        )
        text = out["choices"][0]["message"]["content"]
        parsed = _parse(text)
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_id": MODEL_ID,
            "model_params": MODEL_PARAMS,
            "question_id": qid,
            "grammar_constrained": grammar is not None,
            "answer_type_constraint": answer_type,
            "prompt_tokens": out["usage"]["prompt_tokens"],
            "completion_tokens": out["usage"]["completion_tokens"],
            "seconds": round(time.time() - t0, 2),
            "raw_output": text,
            "parsed": parsed,
        }
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        return parsed
