# Submission: Project 5 (SIXT + TUM challenge), team SK²

**Team name:** SK²

**Team members:** Karen Khudaverdian, Said Sarahna, Kyrylo Liashenko

**Project:** Project 5, answering business questions from linked company data with an AI-assisted workflow.

**Status:** Final for the Hackathon-day submission, except the code commit ID (see below). Stage 1 answers are complete; correctness is checked internally, not against a key. Stage 2 (final data) has not been released, so it is not run. Demo slides: `submission/slides.pdf`.

## Submission files and hashes

- **Public (Stage 1) answer file:** `Project_5/public_answers.csv`, 200 rows, SHA-256 `dea0ef9fa52fb624aa1de638ef07d18cc2139a75a44296d4cd0e3aa07b9a19c5`.
- **Final (Stage 2) answer file:** `Project_5/final_answers.csv`, not yet produced (final data not released).
- **Code revision used for the calculations:** 6190da5 (the code as it stood when the Stage 1 answers were calculated). The organisers record the locked commit IDs themselves at each cutoff.

## Problem and approach

Each question is answered by a deterministic calculation written in Python (pandas, with DuckDB SQL as an independent cross-check). The number is always produced by code, never by a language model.

- **Company questions (P001-P150):** 50 parameterised calculation functions cover all 150 questions. A language model routes each question to the right function and its arguments. A validator checks the choice against the answer type the question asks for, allows one retry, and flags any question it still cannot route. Flagged questions fall back to a hand-written routing, and are labelled as such.
- **Optimiser (P151-P180):** a mixed-integer programme solved to proven optimality with CBC. Each plan is re-costed and checked against the constraints by a separate function.
- **Flow simulator (P181-P200):** an event-by-event simulation with a conservation check (units in = units out) on every run.

Assumptions are documented per question family in the code (`src/company.py`, `src/optimiser.py`, `src/flow_sim.py`). The main interpretation choices were confirmed against the rule text, and one tie-break bug was fixed.

## Models and why

| Model | Open-weight or closed | Where it runs | Main metric(s) | Cost and time per item | Chosen? |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-Coder-7B-Instruct (Q4_K_M GGUF) | Open-weight, Apache-2.0 | Local CPU, llama-cpp-python | Public company questions (150): 150/150 correct function on first attempt, 146/150 exact including arguments. Synthetic routing test (500, written by us): 496/500 correct function, 480/500 exact, 491/500 final answers match gold. Pipeline: see Results. | About 3 s per question on 16 CPU threads | Yes: small, open, public size (7,615,616,512 parameters), runs without a GPU |
| Claude Haiku 5.5 (coding assistant) | Closed | Anthropic, hosted via Claude Code | Not measured. Used only to write code and documentation. | Not measured | Used for code only. Parameter count not disclosed, so it cannot establish bonus eligibility (see below). |

**Bonus: no bonus is claimed.** The model usage is stated for the organisers to assess. Model 1 (Qwen2.5-Coder-7B) has a published total of 7,615,616,512 parameters. Model 2 (the Claude Haiku 5.5 coding assistant) has no published parameter count. The usage evidence is in `MODEL_DECLARATION.csv` and the `evidence/` folder. The full usage log contains question text and stays private.

## Results

All results are internal checks. No reference answers were available to us for stage 1 beyond the 10 practice examples.

- **Practice examples (E001-E010):** 10 of 10 match the provided answers.
- **Demo examples (DEMO001-003):** all 3 match `examples/demo_answers.csv`.
- **Company questions:** 150 answered. The organisers' structure checker reports FORMAT OK. A separate SQL implementation agrees with the pandas implementation on 13 of 13 spot-checked questions.
- **Model-routed run:** all 150 questions routed by the model on the first attempt (no retries, no flags). All 150 final answers match the hand-routed answers, and all 150 pass the value checks.
- **Optimiser:** all 15 cases solve to proven optimality, and every plan passes an independent re-costing and constraint check.
- **Flow simulator:** the conservation identity holds in all 20 runs (10 baseline, 10 what-if).
- **Whole submission:** `public_answers.csv` has 200 rows, all filled, and passes the organisers' structure checker.

**What did not work and the limits:**
- On the public questions the model chose the right function on the first attempt for all 150, but the exact-match figure is 146 of 150 (four argument-format differences, corrected by the adapter). The public set is not a fair test: the catalogue descriptions were written after reading it.
- Correctness is verified only by the internal checks above, not against hidden answers. Stage 2 has not been run.
- The router's accuracy on unseen wording was measured on a synthetic test set of 500 questions we wrote from templates; it is not a random sample of real questions. Its accuracy there: 99.2% correct function, 96.0% exact, 98.2% final answers match gold. A keyword-matching baseline with no model scores 87.0% on the same questions.
- Earlier figures (149 of 150) came from a prompt that contained a worked example copied from a public question (similarity 0.99). That example was removed; the figures above come from the corrected prompt.
- Argument values outside the data make a calculation return 0 silently. The validator now checks each argument against the data, which caught wrong routings on the synthetic set.

## How to run

From a fresh clone of the repository (Python 3.14, Linux or macOS):

1. Place the organisers' package at `Project_5/data/public/` (it is gitignored and must not be committed).
2. Download the model:
   `curl -L -o ~/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf`
3. `cd Project_5 && bash setup.sh` (creates `.venv`, installs pinned requirements, runs the tests). Install llama-cpp with `CMAKE_ARGS="-DGGML_NATIVE=on"` for a CPU-tuned build.
4. Company answers, with the language model routing: `.venv/bin/python src/pipeline_company.py` writes `outputs/company_answers_model_routed.csv`.
5. Optimiser and flow answers: `.venv/bin/python src/optimiser.py` and `.venv/bin/python src/flow_sim.py`.

Environment variables needed: none. The model path is set in `src/llm_router.py`.

## Demo

Produced: `submission/slides.pdf` (five slides, within the limit of five). Source: `submission/slides.pptx`. The slides use a public toy example for the walkthrough, not a competition question. The demo feeds the presentation criterion (10 of the 35 human points).

## Next steps

- Get the organisers' ruling on whether the coding assistant counts toward the 40-point bonus.
- Produce the demo or slides.
- Run the pipeline on the stage 2 data when it is released, and recheck the rules at that point.
- Write the method report in the arXiv template, and agree on authorship and timing with the organisers before uploading.
- With more time: add an automated test that the router agrees with the hand routing on every question, and extend the validator to check value ranges as well as types.
- Use in practice: a question-to-answer workflow that a business user can run on their own data, with every number traceable to a checked calculation.

**Note:** the report, PROJECT_BREAKDOWN.md, the evidence README, the visualiser and the slides were written before the final routing run. Where they differ, the figures
  in this file are the ones to use.
