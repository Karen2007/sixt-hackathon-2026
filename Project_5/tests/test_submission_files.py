"""Checks on the submission file itself (structure only, no answer key needed)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import ROOT  # noqa: E402


def test_public_answers_has_all_200_ids_and_no_blanks():
    df = pd.read_csv(ROOT / "public_answers.csv", dtype=str, keep_default_na=False)
    assert list(df.columns) == ["question_id", "answer"]
    assert list(df.question_id) == [f"P{i:03d}" for i in range(1, 201)]
    assert (df.answer.str.strip() != "").all()


def test_model_declaration_has_required_columns():
    df = pd.read_csv(ROOT / "MODEL_DECLARATION.csv", dtype=str, keep_default_na=False)
    required = {"model_id", "version_or_revision", "publisher", "total_parameters_billions",
                "parameter_count_source", "role", "stages_used"}
    assert required <= set(df.columns)
    assert len(df) >= 1
