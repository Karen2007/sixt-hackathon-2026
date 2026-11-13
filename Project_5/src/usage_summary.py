"""Rebuild the model-usage evidence from data/public/llm_logs/route_calls.jsonl.

Writes evidence/usage_summary.json (totals per run) and evidence/model_usage_sanitised.csv (one row per call,
no question text and no raw model output). Runs are identified by the question-ID prefix:
P = Stage 1 company questions (evaluation and pipeline), S = synthetic routing test, F = Stage 2 company questions.
Usage: python src/usage_summary.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "public" / "llm_logs" / "route_calls.jsonl"
EVID = ROOT / "evidence"
RUN = {"P": "stage1_company_questions", "S": "synthetic_routing_test", "F": "stage2_company_questions"}


def main():
    rows = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["run"] = df.question_id.str[0].map(RUN).fillna("other")
    sanitised = df[["time", "run", "question_id", "grammar_constrained", "answer_type_constraint",
                    "prompt_tokens", "completion_tokens", "seconds"]]
    sanitised.to_csv(EVID / "model_usage_sanitised.csv", index=False)
    by_run = {}
    for name, g in df.groupby("run"):
        by_run[name] = {"calls": int(len(g)), "prompt_tokens": int(g.prompt_tokens.sum()),
                        "completion_tokens": int(g.completion_tokens.sum()), "model_seconds": round(float(g.seconds.sum()))}
    summary = {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF (qwen2.5-coder-7b-instruct-q4_k_m.gguf)",
        "model_repo_revision": "13fb94bfda8c8cf22497dc57b78f391a9acb426a",
        "model_file_sha256": "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c",
        "base_model_total_parameters": int(df.model_params.iloc[0]),
        "runtime": "llama-cpp-python 0.3.16, CPU, 16 threads, temperature 0, seed 0, grammar-constrained function names",
        "hardware": "16 CPU cores, 30 GB RAM, no GPU, Linux",
        "total_calls": int(len(df)),
        "totals": {"prompt_tokens": int(df.prompt_tokens.sum()), "completion_tokens": int(df.completion_tokens.sum()),
                   "model_seconds": round(float(df.seconds.sum()))},
        "by_run": by_run,
        "first_call": str(df.time.min()), "last_call": str(df.time.max()),
        "note": "Question text and raw model output are not included; the full log stays private because it contains organiser question text. "
                "The log starts with the clean-prompt evaluation; the earlier runs are in archived log files, not counted here. "
                "The P group includes both the Stage 1 evaluation and the Stage 1 pipeline run.",
    }
    (EVID / "usage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(by_run, indent=2))
    print("totals:", summary["totals"])


if __name__ == "__main__":
    sys.exit(main())
