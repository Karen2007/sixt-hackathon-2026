"""Model-routed pipeline for the company questions P001-P150.

For each question:
  1. The LLM router chooses a calculation function and arguments (JSON).
  2. Adapters fix format only: a single issue type becomes a list, "All families"
     becomes None, numeric strings become numbers.
  3. Validation: the function must exist, have the right number of arguments, and
     return the answer type the question asks for (answer_type in the question file).
  4. If validation fails, the model gets one retry with the required type stated.
  5. If it still fails, the question is FLAGGED and the hand-written routing is used
     instead. Flagged rows are labelled in the output, never silently replaced.

The number is always computed by the chosen function in src/company.py. The model
never writes an answer.
"""

from __future__ import annotations

import csv
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.company as C  # noqa: E402
from src.company import QUESTIONS_P  # noqa: E402
from src.llm_router import CATALOGUE, Router  # noqa: E402

from src.llm_router import FAMILY_KIND  # output type per function (single source of truth)

QTEXT = pd.read_csv(ROOT / "data" / "public" / "public_leaderboard_questions.csv", dtype=str).set_index("question_id")
CURRENT = pd.read_csv(ROOT / "outputs" / "company_answers_p001_p150.csv", dtype=str, keep_default_na=False).set_index("question_id")["answer"]


def _num(a):
    if isinstance(a, str) and re.fullmatch(r"-?\d+", a.strip()):
        return int(a)
    if isinstance(a, str) and re.fullmatch(r"-?\d+\.\d+", a.strip()):
        return float(a)
    return a


def adapt(plan: dict) -> dict:
    """Format-only fixes. No change of meaning."""
    fn, args = plan["function"], list(plan.get("args") or [])
    if fn == "campaign_ticket_revenue" and len(args) >= 2 and isinstance(args[1], str):
        args[1] = [args[1]]
    if fn == "ppv" and args and (args[0] is None or str(args[0]).strip().lower() in ("all families", "all", "null", "none", "")):
        args[0] = None
    args = [_num(a) for a in args]
    # A zero-argument function given a single null (e.g. at_risk_top_manager([None])) takes no arguments.
    if CATALOGUE.get(fn, ("", ""))[0].strip() == "" and args == [None]:
        args = []
    return {"function": fn, "args": args}


def _column_values(table: str, column: str) -> set:
    import src.core as K
    df = K._read(table)
    return set(df[column].astype(str).str.strip())


@lru_cache(maxsize=1)
def arg_value_sets() -> dict:
    """Values each categorical argument may take, read from the data of the active stage (core.STAGE).
    An argument outside its set makes the calculation silently return 0, so the validator rejects it."""
    cats = set(_column_values("product_category_history", "category"))
    return {
        "country": _column_values("customers", "country"),
        "region": _column_values("customers", "region"),
        "segment": _column_values("accounts", "segment"),
        "industry": _column_values("accounts", "industry"),
        "family": _column_values("products", "family"),
        "category": cats,
        "channel": _column_values("customers", "acquisition_channel") | _column_values("marketing_touches", "marketing_channel"),
        "warehouse": _column_values("warehouse_inventory", "warehouse_region") | _column_values("return_inspections", "warehouse_region"),
        "carrier": _column_values("shipments", "carrier"),
        "reason": _column_values("refunds", "refund_reason"),
        "issue_types": _column_values("support_tickets", "issue_type"),
        "campaign": _column_values("promotions", "campaign"),
        "quarter": {"Q1", "Q2", "Q3", "Q4"},
    }


def check(plan: dict, answer_type: str):
    fn = plan["function"]
    if fn not in CATALOGUE or fn not in FAMILY_KIND:
        return False, f"unknown function {fn}"
    sig = CATALOGUE[fn][0]
    names = [x.strip() for x in sig.split(",") if x.strip()]
    if len(plan["args"]) != len(names):
        return False, f"{fn} takes {len(names)} arguments, got {len(plan['args'])}"
    if FAMILY_KIND[fn] != answer_type:
        return False, f"{fn} returns {FAMILY_KIND[fn]}, question asks for {answer_type}"
    sets = arg_value_sets()
    for name, value in zip(names, plan["args"]):
        if value is None or name not in sets:
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            if str(item) not in sets[name]:
                return False, f"{fn}: {name} value {item!r} is not in the data"
    if "month" in names:
        mv = plan["args"][names.index("month")]
        if mv is not None and not re.fullmatch(r"\d{4}-\d{2}", str(mv)):
            return False, f"{fn}: month must be YYYY-MM, got {mv!r}"
    return True, "ok"


# Names whose value must lie between 0 and 100. Target-achievement percentages (region_target_pct) can
# legitimately exceed 100, so they are not in this list.
PERCENT_LIKE = ("breach", "ontime", "rejection", "restockable", "conversion", "share", "refund_rate")


def value_ok(fn: str, value, kind: str):
    """Check the answer text against its declared output kind, plus a plain range for percentages.
    The company functions return formatted text (e.g. '63.16'), so the check reads the text that is
    submitted. Reported in the output file; it does not change routing."""
    import math
    text = str(value).strip()
    if kind == "integer":
        if not re.fullmatch(r"-?\d+", text):
            return False, f"expected integer, got {text!r}"
        if int(text) < 0:
            return False, "negative count"
        return True, "ok"
    if kind == "number":
        try:
            x = float(text)
        except ValueError:
            return False, f"expected number, got {text!r}"
        if not math.isfinite(x):
            return False, "not finite"
        if any(k in fn for k in PERCENT_LIKE) and not (0 <= x <= 100):
            return False, f"percentage outside 0-100: {text}"
        return True, "ok"
    if kind == "boolean":
        ok = text.lower() in ("true", "false")
        return ok, "ok" if ok else f"expected true/false, got {text!r}"
    if kind == "string":
        return (text != "", "ok" if text else "empty text")
    if kind == "date":
        try:
            pd.Timestamp(text)
            return True, "ok"
        except (ValueError, TypeError):
            return False, f"not a date: {text!r}"
    return False, f"unknown kind {kind}"


def route_validated(router: Router, qid: str):
    text = QTEXT.loc[qid, "question"]
    atype = QTEXT.loc[qid, "answer_type"]
    plan = adapt(router.route(text, qid, answer_type=atype))
    ok, why = check(plan, atype)
    if ok:
        return plan, "model", why
    hint = f"\n(The answer must be of type {atype}. Choose the function whose output has that type.)"
    plan2 = adapt(router.route(text + hint, qid, answer_type=atype))
    ok2, why2 = check(plan2, atype)
    if ok2:
        return plan2, "model_retry", why
    return None, "flagged", f"{why} | retry: {why2}"


def run() -> pd.DataFrame:
    router = Router()
    rows = []
    for qid in sorted(QUESTIONS_P):
        plan, source, reason = route_validated(router, qid)
        if plan is not None:
            value = getattr(C, plan["function"])(*plan["args"])
            answer = str(value)
            fn_used, args_used = plan["function"], str(plan["args"])
        else:
            value = QUESTIONS_P[qid]()
            answer = str(value)
            fn_used, args_used = QUESTIONS_P[qid].plan[0], str(QUESTIONS_P[qid].plan[1])
        same = answer == CURRENT[qid]
        vok, vwhy = value_ok(fn_used, value, FAMILY_KIND[fn_used])
        rows.append({"question_id": qid, "answer": answer, "source": source, "function": fn_used,
                     "args": args_used, "reason": reason, "matches_current_answer": same,
                     "value_check_ok": vok, "value_check_reason": vwhy})
        print(qid, source, "same" if same else "DIFF", answer, flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "company_answers_model_routed.csv"
    df.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(df.source.value_counts().to_dict(), flush=True)
    print(f"matches current answers: {int(df.matches_current_answer.sum())}/{len(df)}", flush=True)
    print(f"value checks passed: {int(df.value_check_ok.sum())}/{len(df)}", flush=True)
    print("written", out, flush=True)
    return df


if __name__ == "__main__":
    run()
