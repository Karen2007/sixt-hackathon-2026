# Evidence index: small-model bonus (Project 5, team SK²)

Each row is a claim in `MODEL_DECLARATION.csv`, `SUBMISSION.md` and `report/main.tex`, with the file that supports it.

| Claim | Evidence file | How to check |
| --- | --- | --- |
| Model is Qwen2.5-Coder-7B-Instruct (GGUF Q4_K_M), revision `13fb94bf…` | `hf_metadata_Qwen2.5-Coder-7B-Instruct-GGUF.json` (field `sha`) | The repository metadata returned by the Hugging Face API |
| Total parameters 7,615,616,512 (dense, BF16 base model) | `hf_metadata_Qwen2.5-Coder-7B-Instruct.json` (field `safetensors.total`) | Same API, base model repository |
| Exact model file used | `model_file_sha256.txt` (SHA-256 `509287f7…4d3c`) | `sha256sum` on the file |
| Usage: 301 calls, token counts, time per call | `usage_summary.json`, `model_usage_sanitised.csv` | One row per call: tokens, seconds, function chosen. No question text. |
| Routing quality: 149/150 first attempt, 145 exact | Measured in the routing evaluation (full log kept privately) | Re-run `src/run_router_eval.py` with the model |
| Pipeline: 149 model-routed, 1 flagged, 150/150 final match | `outputs/company_answers_model_routed.csv` (field `source`) | Column `source` shows how each answer was routed |

## Not included here, and why
- **The full call log** (`data/public/llm_logs/route_calls.jsonl`, gitignored) holds question text and raw model output. The rules forbid publishing question files, so it is available only through the organisers' channel.
- **The coding assistant's parameter count** is not published, so there is no evidence file to add. This is why the bonus depends on the organisers' ruling.
- **Superseded runs** used prompt examples that closely matched test questions. They are kept, labelled `*_superseded_leaky_examples*`, for transparency, and are not used for any reported figure.
