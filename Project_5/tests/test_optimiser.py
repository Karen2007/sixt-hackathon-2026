"""P151-P180: exact optimiser.

solve_case() already re-costs each plan and raises if the plan breaks a rule, so a
clean run is an independent check. The comparison with the saved output is a
regression check: it catches changes to the code, not errors in the rules.
"""
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from conftest import DATA, OUT, needs_data  # noqa: E402


@needs_data
@pytest.mark.parametrize("case_id", [f"OI{i:02d}" for i in range(1, 16)])
def test_case_solves_optimal_and_plan_passes_recheck(case_id):
    from src.optimiser import solve_case

    res = solve_case(case_id)  # raises RuntimeError if the plan fails the independent check
    assert res["status"] == "Optimal/Optimal"
    assert res["baseline_cents"] >= 0


@needs_data
def test_saved_optimiser_answers_are_reproduced():
    from src.optimiser import optimiser_answers

    saved = pd.read_csv(OUT / "optimiser_answers_p151_p180.csv", dtype=str).set_index("question_id")["answer"]
    got = optimiser_answers()
    assert len(got) == 30
    for qid, value in got.items():
        assert saved[qid] == value, qid
