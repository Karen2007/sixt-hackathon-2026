"""Accuracy, confidence intervals, and per-function precision and recall for the routers.

Reads the first-attempt results written by src/run_router_eval.py (LLM router) and recomputes the
keyword baseline (src/baseline_router.py) on the same questions. Prints a summary and writes
outputs/routing_metrics.csv (per-function counts, precision and recall; no question text).

Usage: python src/routing_metrics.py
"""
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline_router import choose  # noqa: E402
from src.run_router_eval import load_set  # noqa: E402
from src.llm_router import LOG_DIR  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def per_function(gold: pd.Series, pred: pd.Series) -> pd.DataFrame:
    rows = []
    for f in sorted(set(gold) | set(pred)):
        tp = int(((gold == f) & (pred == f)).sum())
        n_gold = int((gold == f).sum())
        n_pred = int((pred == f).sum())
        rows.append({"function": f, "gold_n": n_gold, "predicted_n": n_pred, "true_positive": tp,
                     "precision": tp / n_pred if n_pred else float("nan"),
                     "recall": tp / n_gold if n_gold else float("nan")})
    return pd.DataFrame(rows)


def main():
    out_rows = []
    for name in ["public", "synthetic"]:
        gold_df = load_set(name)
        llm = pd.read_csv(LOG_DIR / f"router_eval_{name}.csv", dtype=str, keep_default_na=False)
        assert list(llm.question_id) == list(gold_df.qid), "LLM results must cover the same questions in order"
        base_pred = pd.Series([choose(q, t) for q, t in zip(gold_df.question, gold_df.answer_type)])
        gold = gold_df.gold_function.reset_index(drop=True)
        llm_fn_ok = (llm.function_match == "True").sum()
        llm_full_ok = (llm.full_match == "True").sum()
        base_ok = int((base_pred == gold).sum())
        n = len(gold)
        print(f"\n== {name}: {n} questions")
        for label, k in [("LLM, correct function (first attempt)", llm_fn_ok),
                         ("LLM, exact incl. arguments (first attempt)", llm_full_ok),
                         ("Keyword baseline, correct function", base_ok)]:
            lo, hi = wilson(int(k), n)
            print(f"  {label}: {k}/{n} = {100*k/n:.1f}%  (95% CI {100*lo:.1f}-{100*hi:.1f}%)")
            out_rows.append({"set": name, "measure": label, "correct": int(k), "total": n,
                             "accuracy": round(k / n, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
        pf_llm = per_function(gold, llm.model_function.reset_index(drop=True))
        pf_base = per_function(gold, base_pred)
        for tag, pf in [("llm", pf_llm), ("baseline", pf_base)]:
            macro_p = pf.precision.mean(skipna=True)
            macro_r = pf.recall.mean(skipna=True)
            print(f"  {tag}: macro precision {macro_p:.3f}, macro recall {macro_r:.3f}, "
                  f"functions below 1.0 precision or recall: {int(((pf.precision < 1) | (pf.recall < 1)).sum())}")
            pf.insert(0, "set", name)
            pf.insert(1, "router", tag)
            out_rows.append({"set": name, "measure": f"{tag} macro precision", "correct": None, "total": None,
                             "accuracy": round(macro_p, 4), "ci_low": None, "ci_high": None})
            out_rows.append({"set": name, "measure": f"{tag} macro recall", "correct": None, "total": None,
                             "accuracy": round(macro_r, 4), "ci_low": None, "ci_high": None})
        pd.concat([pf_llm, pf_base]).to_csv(ROOT / "outputs" / f"routing_metrics_per_function_{name}.csv", index=False)
    pd.DataFrame(out_rows).to_csv(ROOT / "outputs" / "routing_metrics.csv", index=False)
    print("\nwritten outputs/routing_metrics.csv")


if __name__ == "__main__":
    main()
