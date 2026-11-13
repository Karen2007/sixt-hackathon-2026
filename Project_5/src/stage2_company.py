"""Stage 2 company answers (F001-F150): route with the LLM, validate, compute with the company functions.

Runs with the Stage 2 tables (PROJECT5_STAGE=final). There is no hand routing for Stage 2, so a question
the model cannot route to a valid function after one retry is FLAGGED and left blank, to be decided by hand.
Output: outputs/stage2_company_answers.csv (source, function, args, reason, value check).

Usage: python src/stage2_company.py
"""
import os
import sys
from pathlib import Path

os.environ["PROJECT5_STAGE"] = "final"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
import src.company as C  # noqa: E402
from src.llm_router import FAMILY_KIND, Router  # noqa: E402
from src.pipeline_company import adapt, check, value_ok  # noqa: E402

QFILE = ROOT / "problem7_student_final" / "final_questions.csv"


def route(router: Router, qid: str, text: str, atype: str):
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


def main():
    q = pd.read_csv(QFILE, dtype=str).set_index("question_id")
    ids = [qid for qid in q.index if 1 <= int(qid[1:]) <= 150]
    router = Router()
    rows = []
    for qid in ids:
        text, atype = q.loc[qid, "question"], q.loc[qid, "answer_type"]
        plan, source, reason = route(router, qid, text, atype)
        if plan is None:
            rows.append({"question_id": qid, "answer": "", "source": source, "function": "", "args": "",
                         "reason": reason, "value_check_ok": False, "value_check_reason": "not computed"})
            print(qid, "FLAGGED", reason, flush=True)
            continue
        try:
            value = str(getattr(C, plan["function"])(*plan["args"]))
        except Exception as e:  # one bad routing must not stop the run
            rows.append({"question_id": qid, "answer": "", "source": "error", "function": plan["function"],
                         "args": str(plan["args"]), "reason": f"{type(e).__name__}: {e}",
                         "value_check_ok": False, "value_check_reason": "not computed"})
            print(qid, "ERROR", type(e).__name__, flush=True)
            continue
        vok, vwhy = value_ok(plan["function"], value, FAMILY_KIND[plan["function"]])
        rows.append({"question_id": qid, "answer": value, "source": source, "function": plan["function"],
                     "args": str(plan["args"]), "reason": reason, "value_check_ok": vok, "value_check_reason": vwhy})
        print(qid, source, plan["function"], value, "" if vok else "VALUE-CHECK " + vwhy, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "stage2_company_answers.csv", index=False)
    print(out.source.value_counts().to_dict(), flush=True)
    print(f"value checks passed: {int(out.value_check_ok.sum())}/{len(out)}", flush=True)
    print("written", ROOT / "outputs" / "stage2_company_answers.csv", flush=True)


if __name__ == "__main__":
    main()
