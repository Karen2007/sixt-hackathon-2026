"""P181-P200: event-by-event fulfilment simulator.

Each run checks the conservation identity (every unit in exactly one state, free
stock never negative). The saved-output comparison is a regression check.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import OUT, needs_data  # noqa: E402


@needs_data
def test_all_flow_runs_pass_invariants():
    from src.flow_sim import run_all

    df = run_all()
    assert len(df) == 10
    assert all(len(p) == 0 for p in df["problems"]), df[df["problems"].str.len() > 0].to_string()


@needs_data
def test_saved_flow_answers_are_reproduced():
    from src.flow_sim import flow_answers

    saved = pd.read_csv(OUT / "flow_answers_p181_p200.csv", dtype=str).set_index("question_id")["answer"]
    got = flow_answers()
    assert len(got) == 20
    for qid, value in got.items():
        assert saved[qid] == value, qid
