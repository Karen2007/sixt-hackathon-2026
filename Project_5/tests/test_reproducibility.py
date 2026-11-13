"""Checks that the answers can be rebuilt from scripts, and that each company answer matches its declared output kind.

The kind check is not circular with the routing: it reads the submitted answer text and checks it against the
kind declared for the function that produced it, so a function whose output type is declared wrongly is caught.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import ROOT, needs_data  # noqa: E402


@needs_data
def test_assembly_reproduces_public_answers():
    from src.assemble_public_answers import build

    current = pd.read_csv(ROOT / "public_answers.csv", dtype=str, keep_default_na=False)
    assert build().equals(current)


@needs_data
def test_every_company_answer_matches_its_declared_kind():
    from src.company import QUESTIONS_P
    from src.llm_router import FAMILY_KIND
    from src.pipeline_company import value_ok

    answers = pd.read_csv(ROOT / "outputs" / "company_answers_p001_p150.csv", dtype=str, keep_default_na=False)
    problems = []
    for qid, text in zip(answers.question_id, answers.answer):
        fn, _ = QUESTIONS_P[qid].plan
        ok, why = value_ok(fn, text, FAMILY_KIND[fn])
        if not ok:
            problems.append((qid, fn, why))
    assert problems == []
