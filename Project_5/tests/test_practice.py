import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.practice import run_all  # noqa: E402


def test_all_ten_practice_answers_match():
    df = run_all()
    assert len(df) == 10
    assert df["match"].all(), df[~df["match"]].to_string()
