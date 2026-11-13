"""Answer-level check for the synthetic routing set.

For each synthetic question, runs the chosen function with (a) the gold arguments and (b) the arguments the
model produced after the adapter, and compares the two numbers. Also reports gold answers that are errors or
empty/zero, which would mean the test values are not in the data.

Needs data/public/llm_logs/router_eval_synthetic.csv (from src/run_router_eval.py synthetic).
Writes data/public/llm_logs/synthetic_answer_match.csv (gitignored: contains question text).
"""
import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.company as C  # noqa: E402
from src.llm_router import LOG_DIR  # noqa: E402
from src.pipeline_company import adapt  # noqa: E402

cache = {}


def run(fn: str, args) -> str:
    key = (fn, repr(args))
    if key not in cache:
        try:
            cache[key] = str(getattr(C, fn)(*args))
        except Exception as e:  # report, don't hide
            cache[key] = f"ERR:{type(e).__name__}"
    return cache[key]


def main():
    d = pd.read_csv(LOG_DIR / "router_eval_synthetic.csv", dtype=str, keep_default_na=False)
    rows = []
    for r in d.itertuples():
        gold_val = run(r.gold_function, ast.literal_eval(r.gold_args))
        if r.model_function and r.model_function in C.__dict__:
            plan = adapt({"function": r.model_function, "args": ast.literal_eval(r.model_args) if r.model_args else []})
            got_val = run(plan["function"], plan["args"])
        else:
            got_val = "NO-FUNCTION"
        rows.append({"question_id": r.question_id, "gold_function": r.gold_function, "gold_answer": gold_val,
                     "model_answer": got_val, "answer_match": gold_val == got_val})
    out = pd.DataFrame(rows)
    out.to_csv(LOG_DIR / "synthetic_answer_match.csv", index=False)
    n = len(out)
    print(f"answer-level match: {int(out.answer_match.sum())}/{n} = {100*out.answer_match.mean():.1f}%")
    bad_gold = out[out.gold_answer.str.startswith("ERR") | out.gold_answer.isin(["0", "0.0", "0.00", ""])]
    print(f"gold answers that are errors or zero (test values may be missing from data): {len(bad_gold)}")
    if len(bad_gold):
        print(bad_gold.gold_function.value_counts().to_string())


if __name__ == "__main__":
    main()
