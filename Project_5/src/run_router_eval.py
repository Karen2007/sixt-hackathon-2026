"""Route labelled questions with the LLM and score the first attempt against the gold routing.

Usage:
  python src/run_router_eval.py public      # the 150 organiser questions (gold = hand routing)
  python src/run_router_eval.py synthetic   # our 500 templated questions (outputs/routing_synthetic_test.csv)

Writes data/public/llm_logs/router_eval_<set>.csv (gitignored: the public file holds question text).
"""
import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.llm_router import Router, LOG_DIR  # noqa: E402


def norm_args(args):
    out = []
    for a in args or []:
        if isinstance(a, list):
            out.append(tuple(str(x) for x in a))
        elif isinstance(a, float) and a.is_integer():
            out.append(str(int(a)))
        else:
            out.append(str(a).strip() if a is not None else None)
    return out


def load_set(name: str) -> pd.DataFrame:
    """Columns: qid, question, answer_type, gold_function, gold_args (python literal list)."""
    if name == "public":
        from src.company import QUESTIONS_P
        q = pd.read_csv(ROOT / "data" / "public" / "public_leaderboard_questions.csv", dtype=str).set_index("question_id")
        rows = []
        for qid in sorted(QUESTIONS_P):
            gold_fn, gold_args = QUESTIONS_P[qid].plan
            rows.append({"qid": qid, "question": q.loc[qid, "question"], "answer_type": q.loc[qid, "answer_type"],
                         "gold_function": gold_fn, "gold_args": str(gold_args)})
        return pd.DataFrame(rows)
    if name == "synthetic":
        d = pd.read_csv(ROOT / "outputs" / "routing_synthetic_test.csv", dtype=str, keep_default_na=False)
        d.insert(0, "qid", [f"S{i:04d}" for i in range(len(d))])
        return d.rename(columns={"gold_function": "gold_function", "gold_args": "gold_args"})[
            ["qid", "question", "answer_type", "gold_function", "gold_args"]]
    raise SystemExit(f"unknown set {name}")


def main(name: str):
    data = load_set(name)
    router = Router()
    rows = []
    for r in data.itertuples():
        got = router.route(r.question, r.qid, answer_type=r.answer_type)
        got_fn = got["function"] if got else None
        got_args = got["args"] if got else None
        gold_args = ast.literal_eval(r.gold_args)
        ok_fn = got_fn == r.gold_function
        ok_args = norm_args(got_args) == norm_args(gold_args) if ok_fn else False
        rows.append({"question_id": r.qid, "answer_type": r.answer_type, "gold_function": r.gold_function,
                     "model_function": got_fn, "gold_args": str(gold_args), "model_args": str(got_args),
                     "function_match": ok_fn, "full_match": ok_fn and ok_args})
        print(r.qid, "fn" if ok_fn else "FN-MISS", "full" if ok_fn and ok_args else "ARG-MISS", flush=True)
    df = pd.DataFrame(rows)
    out = LOG_DIR / f"router_eval_{name}.csv"
    df.to_csv(out, index=False)
    print(f"function match {df.function_match.sum()}/{len(df)}; full match {df.full_match.sum()}/{len(df)}", flush=True)
    print("written", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "public")
