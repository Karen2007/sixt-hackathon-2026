"""Non-model baseline router: picks a function by word overlap with the catalogue.

Same rules as the LLM router: only functions returning the required answer type are considered.
Scores each candidate by shared content words between the question and the function's name and
description, normalised by the square root of the description length. Ties go to catalogue order.
It chooses the function only; arguments are not extracted, so the score is function accuracy.

Usage: python src/baseline_router.py     (prints accuracy on the public and synthetic sets)
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.llm_router import CATALOGUE, FAMILY_KIND  # noqa: E402
from src.run_router_eval import load_set  # noqa: E402

STOP = {"the", "and", "for", "from", "with", "that", "this", "what", "which", "how", "many", "much", "was",
        "were", "are", "is", "in", "of", "to", "by", "on", "at", "a", "an", "or", "per", "its", "their",
        "all", "any", "does", "did", "do", "between", "during", "than", "more", "most", "who", "name",
        "give", "net", "eur", "euro", "euros", "percent", "percentage", "rate", "value", "total"}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOP}


CAT_WORDS = {name: words(name.replace("_", " ") + " " + desc) for name, (sig, desc) in CATALOGUE.items()}


def choose(question: str, answer_type: str) -> str:
    q = words(question)
    best, best_score = None, -1.0
    for name in CATALOGUE:
        if FAMILY_KIND[name] != answer_type:
            continue
        c = CAT_WORDS[name]
        score = len(q & c) / (len(c) ** 0.5) if c else 0.0
        if score > best_score:
            best, best_score = name, score
    return best


def main():
    for name in ["public", "synthetic"]:
        d = load_set(name)
        pred = [choose(q, t) for q, t in zip(d.question, d.answer_type)]
        ok = [p == g for p, g in zip(pred, d.gold_function)]
        print(f"baseline function accuracy on {name}: {sum(ok)}/{len(ok)} = {100*sum(ok)/len(ok):.1f}%")


if __name__ == "__main__":
    main()
