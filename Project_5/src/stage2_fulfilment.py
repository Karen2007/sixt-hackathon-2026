"""Stage 2 fulfilment answers (F151-F200): optimiser cases OF01-OF15 and flow cases FLOW_F01-F10.

Each question names its case in the text and asks either for the baseline value or for the change.
The case and kind are read from the question text, so no question-number mapping is needed.
Uses the same solvers as Stage 1, reading the Stage 2 tables (PROJECT5_STAGE=final).

Usage: python src/stage2_fulfilment.py   -> outputs/stage2_fulfilment_answers.csv
"""
import os
import re
import sys
from pathlib import Path

os.environ["PROJECT5_STAGE"] = "final"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
import src.flow_sim as F  # noqa: E402
import src.optimiser as O  # noqa: E402

QFILE = ROOT / "problem7_student_final" / "final_questions.csv"


def main():
    q = pd.read_csv(QFILE, dtype=str).set_index("question_id")
    wanted = [qid for qid in q.index if 151 <= int(qid[1:]) <= 200]
    opt = {c: O.solve_case(c) for c in O.CASES}
    flow = {c: F.solve_case(c) for c in F.CASES}
    rows = []
    for qid in wanted:
        text = q.loc[qid, "question"]
        m = re.search(r"\b(OF\d{2}|FLOW_F\d{2})\b", text)
        if not m:
            rows.append({"question_id": qid, "answer": "", "case": "", "kind": "", "note": "no case id found"})
            continue
        case = m.group(1)
        is_change = re.search(r"what is the change|apply (all|the)", text, re.I) is not None
        if case.startswith("OF"):
            r = opt[case]
            cents = (r["counterfactual_cents"] - r["baseline_cents"]) if is_change else r["baseline_cents"]
        else:
            r = flow[case]
            cents = (r["what_if_cents"] - r["baseline_cents"]) if is_change else r["baseline_cents"]
        problems = r.get("problems", [])
        rows.append({"question_id": qid, "answer": f"{cents / 100:.2f}", "case": case,
                     "kind": "change" if is_change else "baseline",
                     "note": "; ".join(map(str, problems)) if problems else ""})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "stage2_fulfilment_answers.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
