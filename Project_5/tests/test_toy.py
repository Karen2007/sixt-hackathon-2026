import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.toy_solutions import EXAMPLES, solve_all  # noqa: E402


def test_toy_answers_match_reference_csv():
    expected = pd.read_csv(EXAMPLES / "demo_answers.csv", dtype=str)
    got = solve_all()
    # Compare as strings so formatting ("67.30" vs "67.3") is checked too.
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True))


def test_demo001_counts_order_once():
    from src.toy_solutions import demo001_valid_orders

    assert demo001_valid_orders() == 1


def test_demo003_strict_threshold_no_reorder_at_equality():
    from src.toy_solutions import demo003_units_to_reorder

    assert demo003_units_to_reorder() == 0
