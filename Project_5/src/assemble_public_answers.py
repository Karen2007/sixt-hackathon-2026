"""Rebuild public_answers.csv (P001-P200) from the component outputs, and check it against the file in use.

Sources (all produced by the scripts in src/):
  outputs/company_answers_p001_p150.csv   company calculations (src/run_company.py)
  outputs/optimiser_answers_p151_p180.csv optimiser (src/optimiser.py)
  flow_sim.flow_answers()                 flow simulator, recomputed here (src/flow_sim.py)

Usage:
  python src/assemble_public_answers.py            # rebuild in memory and compare with public_answers.csv
  python src/assemble_public_answers.py --write    # also overwrite public_answers.csv (only after a match is reviewed)
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.flow_sim import flow_answers  # noqa: E402

TARGET = ROOT / "public_answers.csv"


def build() -> pd.DataFrame:
    company = pd.read_csv(ROOT / "outputs" / "company_answers_p001_p150.csv", dtype=str, keep_default_na=False)
    optimiser = pd.read_csv(ROOT / "outputs" / "optimiser_answers_p151_p180.csv", dtype=str, keep_default_na=False)
    flow = pd.DataFrame([{"question_id": k, "answer": v} for k, v in flow_answers().items()])
    out = pd.concat([company, optimiser, flow], ignore_index=True)
    out = out.sort_values("question_id").reset_index(drop=True)
    expected = [f"P{i:03d}" for i in range(1, 201)]
    assert list(out.question_id) == expected, "question IDs must be P001-P200, once each, in order"
    assert (out.answer.str.strip() != "").all(), "every answer must be filled"
    return out


def main():
    df = build()
    print("rebuilt rows:", len(df))
    if TARGET.exists():
        current = pd.read_csv(TARGET, dtype=str, keep_default_na=False)
        same = current.equals(df)
        print("identical to current public_answers.csv:", same)
        if not same:
            diff = current.merge(df, on="question_id", suffixes=("_current", "_rebuilt"))
            diff = diff[diff.answer_current != diff.answer_rebuilt]
            print(diff.to_string(index=False))
    if "--write" in sys.argv:
        df.to_csv(TARGET, index=False)
        print("written", TARGET, "sha256", hashlib.sha256(TARGET.read_bytes()).hexdigest())
    else:
        print("not written (use --write to overwrite public_answers.csv)")


if __name__ == "__main__":
    main()
