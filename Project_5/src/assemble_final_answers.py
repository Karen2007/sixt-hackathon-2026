"""Assemble the Stage 2 answer file (F001-F200) from the saved outputs.

Sources:
  outputs/stage2_company_answers.csv     F001-F150 (routed by the LLM, computed by company functions)
  outputs/stage2_fulfilment_answers.csv  F151-F200 (optimiser and flow solvers)

Writes outputs/final_answers_draft.csv and lists blanks. The organisers' final_answers.csv (Project_5 root)
is written only with --write, and only when there are no blanks and every value passed its checks.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = ROOT / "outputs"
TEMPLATE = ROOT / "problem7_student_final" / "final_answers.csv"


def build() -> pd.DataFrame:
    comp = pd.read_csv(ANSWERS / "stage2_company_answers.csv", dtype=str, keep_default_na=False)
    fulf = pd.read_csv(ANSWERS / "stage2_fulfilment_answers.csv", dtype=str, keep_default_na=False)
    comp = comp.rename(columns={"reason": "note"})[["question_id", "answer", "source", "value_check_ok"]]
    fulf = fulf.assign(source="solver", value_check_ok="True")[["question_id", "answer", "source", "value_check_ok"]]
    out = pd.concat([comp, fulf], ignore_index=True).sort_values("question_id").reset_index(drop=True)
    expected = [f"F{i:03d}" for i in range(1, 201)]
    assert list(out.question_id) == expected, "IDs must be F001-F200, once each"
    return out


def main():
    df = build()
    blanks = df[df.answer.str.strip() == ""]
    print("rows:", len(df), " blanks:", len(blanks), " value checks failed:", int((df.value_check_ok != "True").sum()))
    print(blanks.to_string(index=False) if len(blanks) else "no blanks")
    df.to_csv(ANSWERS / "final_answers_draft.csv", index=False)
    print("draft written:", ANSWERS / "final_answers_draft.csv")
    if "--write" in sys.argv:
        if len(blanks) or (df.value_check_ok != "True").any():
            sys.exit("refusing to write: blanks or failed checks remain")
        df[["question_id", "answer"]].to_csv(ROOT / "final_answers.csv", index=False)
        print("written", ROOT / "final_answers.csv")


if __name__ == "__main__":
    main()
