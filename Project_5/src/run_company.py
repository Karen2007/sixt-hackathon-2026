"""Compute answers for P001-P150 and write a draft answer file to outputs/."""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.company import QUESTIONS_P  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "company_answers_p001_p150.csv"


def run() -> pd.DataFrame:
    rows, errors = [], []
    for qid in sorted(QUESTIONS_P):
        t0 = time.time()
        try:
            ans = str(QUESTIONS_P[qid]())
        except Exception as exc:
            errors.append((qid, repr(exc)))
            ans = ""
        rows.append({"question_id": qid, "answer": ans})
        print(f"{qid} {ans!r} ({time.time() - t0:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    for qid, msg in errors:
        print("ERROR", qid, msg, flush=True)
    print(f"computed {len(df)}; errors {len(errors)}; written {OUT}", flush=True)
    return df


if __name__ == "__main__":
    run()
