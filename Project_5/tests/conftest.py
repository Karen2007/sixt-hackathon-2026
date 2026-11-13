"""Shared paths and skip rules.

Organiser data lives in data/public/ (gitignored) and is not in the repository.
Tests that need it are skipped with a clear reason when it is absent, so the
checks that need no data still run on a fresh clone.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public"
OUT = ROOT / "outputs"

needs_data = pytest.mark.skipif(
    not (DATA / "datasets").exists(),
    reason="organiser data not present in data/public/ (gitignored)",
)
