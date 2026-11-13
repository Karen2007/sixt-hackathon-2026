"""Validator and adapters for the LLM router (no model needed for these tests).

The model-dependent test is opt-in: it runs the 7.6B model on all 150 questions
(about 8 minutes on this machine). Enable with RUN_LLM_TESTS=1.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import DATA, OUT, needs_data  # noqa: E402
from src.pipeline_company import FAMILY_KIND, adapt, check  # noqa: E402
from src.company import QUESTIONS_P  # noqa: E402
from src.llm_router import CATALOGUE  # noqa: E402


def test_every_catalogue_function_has_a_declared_output_kind():
    assert set(CATALOGUE) == set(FAMILY_KIND)


def test_adapter_wraps_single_issue_type_in_list():
    plan = adapt({"function": "campaign_ticket_revenue", "args": ["SpringLaunch", "returns"]})
    assert plan["args"] == ["SpringLaunch", ["returns"]]


def test_adapter_turns_all_families_into_none():
    plan = adapt({"function": "ppv", "args": ["All families"]})
    assert plan["args"] == [None]


def test_adapter_converts_numeric_strings():
    plan = adapt({"function": "growth_bool_region", "args": ["Spain", "5"]})
    assert plan["args"] == ["Spain", 5]


def test_check_rejects_wrong_output_type():
    ok, why = check({"function": "atrisk_count_country", "args": ["Spain"]}, "string")
    assert not ok and "integer" in why


def test_check_rejects_wrong_argument_count():
    ok, why = check({"function": "at_risk_top_manager", "args": ["Spain"]}, "string")
    assert not ok and "arguments" in why


def test_check_rejects_unknown_function():
    ok, _ = check({"function": "made_up_function", "args": []}, "integer")
    assert not ok


@needs_data
def test_all_hand_routings_pass_the_validator():
    """Correct routings must never be rejected, or the validator would flag good answers."""
    from src.pipeline_company import QTEXT

    failures = []
    for qid, fam in QUESTIONS_P.items():
        fn, args = fam.plan
        ok, why = check(adapt({"function": fn, "args": list(args)}), QTEXT.loc[qid, "answer_type"])
        if not ok:
            failures.append((qid, why))
    assert failures == []


@needs_data
def test_saved_model_routed_answers_match_hand_routed_answers():
    df = pd.read_csv(OUT / "company_answers_model_routed.csv", dtype=str, keep_default_na=False)
    assert len(df) == 150
    assert (df["matches_current_answer"] == "True").all()


@needs_data
@pytest.mark.skipif(os.environ.get("RUN_LLM_TESTS") != "1", reason="set RUN_LLM_TESTS=1 to run the 7.6B model")
def test_model_routes_at_least_99_percent_of_questions_correctly():
    from src.llm_router import Router
    from src.pipeline_company import route_validated

    router = Router()
    ok = 0
    for qid in sorted(QUESTIONS_P):
        plan, _, _ = route_validated(router, qid)
        if plan is not None and plan["function"] == QUESTIONS_P[qid].plan[0]:
            ok += 1
    assert ok >= 148
