"""Quick, read-only profile of every CSV in data/public/datasets.

Writes data/public/profile.md (inside the gitignored data folder), listing each
table with its row count and column names, plus a check of the question file.
"""
from pathlib import Path

import pandas as pd

PUB = Path(__file__).resolve().parents[1] / "data" / "public"

lines = ["# Table profile (public stage)", "", "| table | rows | columns |", "|---|---:|---|"]
for path in sorted((PUB / "datasets").glob("*.csv")):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    lines.append(f"| {path.name} | {len(df)} | {', '.join(df.columns)} |")

q = pd.read_csv(PUB / "public_leaderboard_questions.csv", dtype=str)
lines += ["", f"Questions: {len(q)} rows; types: {q['answer_type'].value_counts().to_dict()}"]
lines += ["", "Difficulty counts:", ""] + [f"- {k}: {v}" for k, v in q["difficulty"].value_counts().items()]
out = PUB / "profile.md"
out.write_text("\n".join(lines) + "\n")
print(f"tables profiled: {len(list((PUB / 'datasets').glob('*.csv')))}\nwritten: {out}")
