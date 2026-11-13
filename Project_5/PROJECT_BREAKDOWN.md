# Project 5 breakdown: how the answering pipeline works

Team SK² (Karen Khudaverdian, Said Sarahna, Kyrylo Liashenko). SIXT + TUM Hackathon 2026, Project 5: Enterprise Data Agent Challenge.

This document explains what the project is, how the pipeline works from question to answer, what every folder and file is for, what has been done, what the limits are, and how the Stage 2 work will fit in. It is written so that someone who has not seen the code can understand and explain it.

---

## 1. Purpose

The organisers provide a set of 37 linked business data tables (customers, orders, payments, refunds, support tickets, marketing touches, stock, purchase orders, suppliers, shipments and more), a set of written business rules, and 200 questions for Stage 1. Each question asks for one number. The task is to produce a CSV file called `public_answers.csv` with one answer per question.

The challenge is not just arithmetic. The rules include identity resolution (the same customer may appear under several emails), eligibility filters (test accounts, cancelled orders, failed payments), currency conversion by the month of the transaction, refund allocation, and time-dependent state (a stock figure at a reference time, a simulation that moves step by step). Getting one of these rules wrong changes the answer.

The project makes one central design decision:

> **A language model decides which calculation applies. Ordinary code computes the number.**

The model never produces an answer. This keeps every number traceable to a function that can be re-run, inspected and tested. It also means the model's mistakes can be caught before they reach the answer file.

---

## 2. The three question families

The 200 questions fall into three families. Each family has a different solution method, chosen to match what the question asks.

### 2.1 Company analytics: P001 to P150

These questions ask about the business: revenue by country and quarter, growth between quarters, refund rates by product category, ticket breach rates, supplier rejection rates, warehouse stock values, cohort retention, marketing attribution and so on.

They are answered by **50 calculation functions** in `src/company.py`. Each function does one well-defined calculation on the loaded tables. Many questions share a function with different arguments. For example, "net revenue for customers in one country in one quarter" is one function called with a country and a quarter.

Shared rules live in `src/core.py`: which customers are eligible, which orders are valid, how to convert currencies, how to allocate refunds to items, and how to merge duplicate customer identities by normalised email.

### 2.2 Fulfilment optimisation: P151 to P180

These questions describe jobs that need to be filled from stock or supplier options, with due dates, late penalties, unfilled penalties and dispatch capacities. The question asks for the minimum total cost of an optimal plan (P151, P153, and so on), and for the change in that cost under a set of edits (the counterfactual, which follows the even-numbered entries).

The problem is solved as a **mixed-integer linear programme**. Put simply: each job either gets one feasible option or is left unfilled, and the solver chooses the combination that minimises total cost subject to stock, capacity and supplier constraints. The solver is CBC (open source, bundled with the PuLP library). It reports whether the solution is proven optimal. All 15 cases return "Optimal".

### 2.3 Fulfilment simulation: P181 to P200

These questions describe a sequence of events over time: orders are released, stock is received, orders are cancelled, goods are dispatched, customers return items, and some returns are restocked. The question asks for the net contribution in euros after revenue, product cost, late penalties, unfilled penalties, cancellation and return fees, and rebates. The even-numbered entries ask for the change under a what-if edit (such as shifting one event later or changing a quantity).

The simulator in `src/flow_sim.py` processes events in a fixed order:

1. For each event key, keep only the version that was known by the knowledge cutoff (the latest recorded version up to that step). Drop events whose selected version is void.
2. Apply any what-if edits, then drop events after the horizon.
3. At each step, process in this priority order: cancellations, returns, receipts, releases, dispatches.
4. Reserve free stock for released orders in queue order (priority, then deadline, then release step, then order ID).
5. Dispatch units up to the warehouse capacity, recording whether each unit was late.
6. At the end, compute revenue, cost, recovery value, penalties, fees and rebates, then the contribution.

After every run, the simulator checks a **conservation identity**: opening stock plus receipts plus restocked units must equal free stock plus reserved units plus dispatched units. A run that breaks this identity is reported as a problem. All 20 runs (10 baseline, 10 what-if) pass.

---

## 3. The company pipeline in detail

The company questions go through a validated model-routing pipeline. This is the part that uses the language model. Here is what happens for one question, step by step.

**Step 1. Read the question.** The question text and its expected answer type (integer, number, date, boolean or string) come from the organisers' question file.

**Step 2. Route with the model.** `src/llm_router.py` sends the question to a local language model, Qwen2.5-Coder-7B-Instruct (7,615,616,512 parameters). The model runs on the CPU, in the GGUF format at Q4_K_M quantisation. It receives:

- a system prompt that lists the 50 functions, each with a short description and its argument list;
- three short examples of question-to-function mappings, using countries and regions that do not appear in the question set;
- the question itself.

It must reply with one JSON object: `{"function": "<name>", "args": [...]}`. Decoding is deterministic (temperature 0, seed 0), and the output is capped at 120 tokens. Every call is logged with its tokens, timing and raw output, in a private log.

**Step 3. Adapt the format.** `adapt()` in `src/pipeline_company.py` makes format-only changes. A single issue type becomes a one-element list where the function expects a list. The text "All families" becomes a null argument. Numeric strings become numbers. These changes never alter the meaning of the request.

**Step 4. Validate.** `check()` tests three things:

- the function exists in the catalogue;
- the number of arguments matches the function's signature;
- the function's output type matches the answer type the question asks for.

The output type of each function is recorded in a table (`FAMILY_KIND`). The table was built by looking at what each function actually returns, which is a limitation discussed in section 8.

**Step 5. Retry or flag.** If validation fails, the model is asked again, with the required answer type added to the question text. If the second attempt also fails, the question is **flagged**. A flagged question is answered by the hand-written routing (the mapping in `QUESTIONS_P`), and the output file records the question as `flagged`, so it is never silently replaced.

**Step 6. Calculate.** The chosen function runs on the tables and returns the number. The number is formatted and written to the output.

**Step 7. Compare.** The result is compared with the current hand-routed answer. In the final run, all 150 answers match.

### What the model's routing achieved

The routing evaluation (`src/run_router_eval.py`) ran the model on all 150 questions and compared its choices with the hand routing:

- **Correct function on the first attempt:** 149 of 150 (99.3%). The one wrong choice was a count question routed to a name lookup.
- **Exact match, including argument format:** 145 of 150. The other four differ only in format (for example, a plain string where a list is expected, or "All families" where null is expected). The adapters fix these in the pipeline.

The full pipeline run then gave:

- 149 questions routed by the model;
- 1 question flagged (the question that failed on the first attempt also failed on the retry) and answered by hand routing;
- all 150 final answers matching the hand-routed answers.

Speed: about three seconds per question on the test machine (16 CPU cores, 30 GB RAM, no GPU). The full run of 150 questions took about eight minutes.

### Why the model's routing does not change any answer

Because every final answer matches the hand routing, the model's choices do not change any number in the answer file. The model's role is checked and auditable, but the answers would be the same without it. This matters for the bonus question, which is discussed in section 10.

---

## 4. Validation and checks

Several independent checks support the answers. None of them uses the hidden Stage 1 or Stage 2 answers, because they are not available to us.

| Check | What it tests | Result |
|---|---|---|
| Toy examples (DEMO001–003) | Hand-computed examples from the organisers, with answers | 3 of 3 match |
| Practice examples (E001–E010) | Organisers' practice questions with answers | 10 of 10 match |
| SQL cross-check (`src/crosscheck_sql.py`) | 13 company questions re-computed in DuckDB SQL, independent of the pandas code | 13 of 13 agree |
| Routing evaluation | Model's function choices against the hand routing | 149 of 150 first attempt; 145 exact |
| Pipeline run | Final answers against the hand routing | 150 of 150 match |
| Optimiser re-costing | Each optimal plan re-costed independently and checked against stock, capacity and supplier limits | 15 of 15 pass |
| Flow conservation | Every simulation run keeps the stock identity | 20 of 20 pass |
| Answer-file structure | Organisers' checker: 200 rows, IDs, no blanks | FORMAT OK |
| Automated tests (`tests/`) | Validator, adapters, optimiser and simulator reproduction, answer-file structure, model declaration | 33 pass, 1 skipped |

The skipped test re-runs the model on all 150 questions. It is opt-in (`RUN_LLM_TESTS=1`), because it takes about eight minutes.

Two of the tests are regression tests: they check that the saved outputs can still be reproduced by the code. They catch accidental changes, but they cannot detect a rule that was misread from the start.

### Interpretation review

Four interpretation choices in the company questions were checked against the rule text:

- the manager for a canonical account (P009);
- the threshold for a high-value account (7,000 EUR, used in P024 and P077);
- identity-level touches and tickets (P053, P096, P097);
- the tie-break for the top z-score region and the top category (P027, P061).

Three were confirmed directly by the rule text. The tie-break was corrected: the top z-score and top category choice now uses the alphabetically first name when two are tied. The earlier code had a bug in P100 (it referred to a pandas method instead of a column), which was also fixed.

---

## 5. Folder and file reference

### Top level (`Project_5/`)

| Item | Purpose | Status |
|---|---|---|
| `public_answers.csv` | The Stage 1 answer file that is judged. 200 rows, `question_id,answer`. | Complete |
| `SUBMISSION.md` | The report required by the organisers, following their template. Includes the team, approach, models, results, how to run, demo, next steps, and the hash of the answer file. | Complete, except the commit ID |
| `MODEL_DECLARATION.csv` | The model-use declaration: identity, revision, parameter count, role, and evidence for each model. | Complete |
| `requirements.txt` | Exact versions of the Python packages, including the model runtime (llama-cpp-python 0.3.16) and the solver (PuLP 2.9.0). | Complete |
| `setup.sh` | Creates `.venv`, installs the pinned requirements, and runs the tests. | Complete |
| `PROJECT_BREAKDOWN.md` | This document. | Complete |
| `README.md`, `STUDENT_GUIDE.pdf`, `assets/` | The repository's project material, present from the start. Not written by the team. | Unchanged |

### `src/`: the code

| File | What it does |
|---|---|
| `core.py` | Loads the tables and provides shared business-rule helpers: eligible customers, valid orders, currency conversion, refund allocation, identity matching. |
| `company.py` | The 50 calculation functions and `QUESTIONS_P`, which maps each company question to its hand-chosen function and arguments. |
| `pipeline_company.py` | The validated model-routing pipeline: adapt, validate, retry, flag, calculate, compare. |
| `llm_router.py` | The language-model wrapper: the function catalogue, the prompt, JSON parsing and per-call logging. |
| `run_router_eval.py` | Runs the model on all 150 questions and scores its choices against the hand routing. |
| `run_company.py` | The hand-routed run. Writes `outputs/company_answers_p001_p150.csv`. |
| `crosscheck_sql.py` | Independent DuckDB SQL versions of 13 company questions, used to confirm the pandas code. |
| `optimiser.py` | The exact optimiser for P151–P180, including the independent plan check and the counterfactual. Its `__main__` block writes `outputs/optimiser_answers_p151_p180.csv`. |
| `flow_sim.py` | The event simulator for P181–P200, with the conservation check and what-if edits. Its `__main__` block prints the per-case results. |
| `practice.py` | Hand-coded answers to the 10 practice examples, compared with the organisers' answers. |
| `toy_solutions.py` | The three toy examples. Its `__main__` block writes `outputs/demo_answers_pred.csv`. |
| `profile_tables.py` | Produces a profile of the data tables for reference (`data/public/profile.md`). |
| `__init__.py` | Marks `src` as a package. |

### `tests/`: automated checks

| File | What it checks |
|---|---|
| `conftest.py` | Shared paths, and the rule that tests needing organiser data are skipped with a reason when the data is absent. |
| `test_toy.py` | The toy examples match their answers, and two specific rules hold. |
| `test_practice.py` | All 10 practice answers match. |
| `test_optimiser.py` | Each of the 15 optimiser cases solves to "Optimal" and passes the plan check; saved answers are reproduced. |
| `test_flow.py` | Each simulation run passes the invariants; saved answers are reproduced. |
| `test_router.py` | The validator and adapters behave as intended; every hand routing passes the validator; the saved model-routed answers match. The model test is opt-in. |
| `test_submission_files.py` | The answer file has all 200 IDs and no blanks; the model declaration has the required columns. |

### `outputs/`: calculated answers

| File | Contents |
|---|---|
| `company_answers_p001_p150.csv` | Hand-routed answers for the 150 company questions. |
| `optimiser_answers_p151_p180.csv` | Baseline and counterfactual answers for the 15 optimiser cases. |
| `flow_answers_p181_p200.csv` | Baseline and what-if answers for the 10 flow cases. No script in `src/` writes this file. It was created by a short command that calls `flow_answers()`, and its contents were verified. |
| `company_answers_model_routed.csv` | The model-routed company answers, with a `source` column (`model`, `model_retry` or `flagged`) and a `matches_current_answer` column. |
| `company_answers_model_routed_superseded_leaky_examples.csv` | An earlier model-routed run whose prompt examples were too close to test questions. Kept for transparency; not used for any reported figure. |
| `demo_answers_pred.csv` | The toy-example answers. |

### `data/`: organiser data (not committed)

The `data/` folder is excluded from version control, because the organisers' data and question files must not be published.

| Folder | Contents |
|---|---|
| `data/public/` | The Stage 1 datasets (37 CSV tables), the business rules, data dictionary, question file, templates and the organisers' preview and report template. Also the full model logs in `llm_logs/`. |
| `data/problem7_student_start/` | A copy of the organisers' package, kept for reference and for the checker script. |
| `data/final/` | Empty. Reserved for the Stage 2 data. |

`data/public/llm_logs/route_calls.jsonl` holds the full model log, including question text and raw output. It stays private and would be shared only through the organisers' channel.

### `evidence/`: model-use evidence

| File | Contents |
|---|---|
| `README.md` | An index that maps each claim about the model to its evidence file. |
| `model_file_sha256.txt` | The SHA-256 hash of the exact model file used. |
| `hf_metadata_Qwen2.5-Coder-7B-Instruct.json` | The repository metadata for the base model, including its parameter count. |
| `hf_metadata_Qwen2.5-Coder-7B-Instruct-GGUF.json` | The repository metadata for the quantised file, including the revision used. |
| `model_usage_sanitised.csv` | One row per model call: time, question identifier, tokens, timing, function chosen. No question text or raw output. |
| `usage_summary.json` | Totals and key results from the usage log. |

### `report/`: the arXiv report

| File | Contents |
|---|---|
| `main.tex` | The LaTeX source of the report, to be pasted into Overleaf. It still contains TODO markers for affiliations, a date, contributions and a code link. |
| `references.bib` | The dataset citation, copied from the organisers' template. |

### `submission/`: the demo

| File | Contents |
|---|---|
| `slides.pdf` | Five slides: title, how one answer is produced, a walkthrough of a public toy example, the validation evidence, and limits and next steps. |
| `slides.pptx` | The editable source of the slides. |

### `examples/`: public toy examples

Organiser-provided small tables (`orders.csv`, `payments.csv`, `revenue_inputs.csv`, `stock_movements.csv`, `stock_policy.csv`), the expected answers (`demo_answers.csv`) and a README explaining each example. These are invented for preparation and safe to show publicly.

### `logs/`

An empty placeholder folder. It is not used by the current code.

---

## 6. How to run everything

From a fresh copy of the repository, with the organiser data placed in `data/public/`:

1. **Set up the environment and run the tests**
   ```bash
   cd Project_5
   bash setup.sh
   ```
   This creates `.venv`, installs the pinned packages and runs the test suite. The model runtime is compiled from source; for a CPU-tuned build, install with `CMAKE_ARGS="-DGGML_NATIVE=on"`.

2. **Download the model** (about 4.7 GB, outside the repository)
   ```bash
   mkdir -p ~/models
   curl -L -o ~/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
     https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf
   ```
   The path is set in `src/llm_router.py` (`MODEL_PATH`). It is currently a fixed path and should be made configurable (see section 9).

3. **Run the company pipeline with the model** (about eight minutes)
   ```bash
   .venv/bin/python src/pipeline_company.py
   ```
   Writes `outputs/company_answers_model_routed.csv`.

4. **Run the optimiser and the simulator**
   ```bash
   .venv/bin/python src/optimiser.py     # writes outputs/optimiser_answers_p151_p180.csv
   .venv/bin/python src/flow_sim.py      # prints the per-case results
   ```

5. **Run the routing evaluation** (about eight minutes)
   ```bash
   .venv/bin/python src/run_router_eval.py
   ```

6. **Run the model-dependent test** (opt-in)
   ```bash
   RUN_LLM_TESTS=1 .venv/bin/python -m pytest tests/test_router.py
   ```

7. **Check the answer file**
   ```bash
   .venv/bin/python data/problem7_student_start/check_submission.py \
     --questions data/public/public_leaderboard_questions.csv \
     --submission public_answers.csv
   ```

The answer file `public_answers.csv` is assembled from the three output files: the hand-routed company answers (or the model-routed ones, which match them exactly), the optimiser answers and the flow answers. No script in the repository performs this assembly yet. It was done once and verified, and it is listed as an improvement in section 9.

---

## 7. What has been done, in order

1. **Set up the project.** Python environment, pinned packages, setup script, and a check of every file hash in the organisers' package.
2. **Checked the toy examples.** All three match.
3. **Built the company calculations.** The core rules module, the 50 functions, the mapping of 150 questions, and an independent SQL check on 13 of them. Four interpretation choices were reviewed, and one tie-break bug and one P100 bug were fixed.
4. **Practice examples.** All 10 match.
5. **Built the optimiser.** Exact solution of all 15 cases, with an independent plan check and the counterfactual edits.
6. **Built the simulator.** Ten cases with baseline and what-if runs, with the conservation check passing in every run.
7. **Assembled the Stage 1 answer file.** 200 rows, checked against the three outputs and against the organisers' structure checker.
8. **Added the language model.** Downloaded and checked the model, built the runtime, wrote the router, and removed prompt examples that were too close to test questions. The first run's results are kept as superseded for transparency.
9. **Ran the model-routed pipeline** with validation, one retry and flagging. The routing evaluation and the final pipeline were both re-run after the prompt fix.
10. **Wrote the tests** (33 passing) and the requirements file with pinned versions.
11. **Wrote the submission material.** The report (`SUBMISSION.md`), the model declaration, the evidence folder, the arXiv LaTeX template, and five slides.
12. **Checked the repository.** It is public, the answer file is consistent with its outputs, and the ignore rules keep the data, the environment and the caches out of version control.

Nothing has been committed yet. The commit is the last step before the submission.

---

## 8. Limits and honest caveats

These are the points to state when someone asks how reliable the answers are.

- **Correctness is only checked internally.** No hidden answers have been compared. The checks in section 4 show internal consistency, the agreement of two independent implementations on some questions, and the rules being applied as written. They do not prove that every answer is right.
- **The type table is built from our own outputs.** The `FAMILY_KIND` table, which the validator uses to check the model's choice, was inferred from what our functions return. If a function returned the wrong type, the validator would not catch it, because the table was built from the same outputs. The question file's expected answer type is independent, so the check is only partial.
- **The model's routing does not change any answer.** Every final answer matches the hand routing, so the model's contribution is checked and auditable, but the answers would be the same without it.
- **The router was tested only on public questions.** Its accuracy on new questions is unknown. The validator and fallback limit the damage, but a question routed to a wrong function that happens to return the right type would pass.
- **The regression tests compare with our own outputs.** They detect changes in the code, not errors in understanding the rules.
- **Some answer files were written by hand.** The flow answers file has no script that writes it, and the Stage 1 assembly has no script either. Both were verified, but they are not reproducible from the repository alone.
- **Question families are fixed to the public set.** The 50 functions and the 150-question mapping were written for the public questions. Stage 2 may contain questions that need new functions.
- **Timing and cost are for this machine only.** About three seconds per question on 16 CPU cores, with no API cost because the model runs locally.

---

## 9. Improvements before or after the submission

These are not needed for the 16:45 submission. They would make the project more reliable and easier to reproduce.

- **Write one script** that assembles `public_answers.csv` from the output files, and one that writes the flow answers file.
- **Make the model path configurable** through an environment variable or a config file, instead of the fixed path in `llm_router.py`.
- **Make the data paths configurable** for each stage. Several scripts currently read `data/public/` directly.
- **Derive the type table from the rules** or from the question file, instead of from our own outputs.
- **Add a routing test** against an independent set of questions, if one becomes available.
- **Add value-range checks** to the validator, for example that a percentage lies between 0 and 100.

---

## 10. Bonus and model-use declaration

The organisers offer 40 extra points for a verified small-model workflow. The rules require every generative model used for code, answers or validation to have at most 40 billion total parameters, and they require model IDs, parameter-count evidence and usage logs. Unknown sizes cannot establish eligibility.

**The team has not claimed the bonus.** The declaration states usage only:

- **Qwen2.5-Coder-7B-Instruct** is the model in the pipeline. It has a published total of 7,615,616,512 parameters. It ran locally, as a router for the 150 company questions, with 301 logged calls.
- **Claude Haiku 5.5** was the coding assistant used to write the code, tests and documents. Its parameter count is not published. It was not used at run time and did not produce any answer.

The evidence is in `evidence/`, and the declaration is in `MODEL_DECLARATION.csv`. Whether the usage is substantive enough for the bonus is for the organisers to decide. The model's routing choices match the hand routing in every final answer, which is relevant to that decision.

---

## 11. Stage 2 plan

Stage 2 is the final-stage data and question set. It has not been released. The rules say it will have 200 new scored questions with their own data snapshot, and the answers are submitted as `final_answers.csv` at the final cutoff. Stage 1 answers are locked before Stage 2 is released. The Stage 2 score counts for 65% of the dataset score, so it matters most.

When the data is released, the work is:

1. **Record the package:** archive name, SHA-256 hash, access date, dataset version and event revision, in the report and `SUBMISSION.md`.
2. **Place the data** in `data/final/`, which is excluded from version control.
3. **Make the paths and identifiers configurable** so the scripts read the Stage 2 question file and use its question IDs. The optimiser and simulator currently have fixed case lists (OI01–OI15, I01–I10) and fixed question-ID ranges.
4. **Review the question families.** Any new question type needs a new function, an interpretation check against the rules, and a catalogue entry for the router. Questions the router cannot map need hand routing.
5. **Re-run everything:** the model routing and validation, the optimiser, the simulator and the SQL cross-check where possible. This produces a new usage log.
6. **Update the tests** so they check the Stage 2 outputs, or skip the Stage 1 regression tests when Stage 2 is selected.
7. **Re-read the rules,** since the business rules and the policy version may have changed.
8. **Assemble and check** `final_answers.csv`: 200 rows in `question_id,answer` format, run through the organisers' checker. Freeze the commit before the cutoff.
9. **Update the report:** Stage 2 results, the Stage 2 dataset fields, and the usage for Stage 2.

No answer key is available for Stage 2, so only the internal checks will be possible.

---

## 12. Glossary

- **Answer type:** the kind of value a question asks for: integer, number, date, boolean or string.
- **Adapter:** a format-only change that makes the model's output match the function's expected input, without changing its meaning.
- **Catalogue:** the list of 50 functions the model can choose from, with their descriptions and argument lists.
- **Conservation identity:** in the simulator, every unit is in exactly one state, so stock is never created or lost.
- **Counterfactual or what-if:** the same scenario with specified edits applied, to see how the answer changes.
- **CBC:** an open-source solver for mixed-integer optimisation problems.
- **DuckDB:** an embedded SQL database, used here for an independent check.
- **Flagged:** a question the validator could not route, answered by hand routing and recorded as such.
- **GGUF and Q4_K_M:** a model file format and a compression level that lets a large model run on a normal computer. Compression does not change the parameter count.
- **Hand routing:** the mapping of each company question to its function, written by the team. It is the reference the model is checked against.
- **Knowledge cutoff:** in the simulator, the step up to which events are considered known.
- **Mixed-integer linear programme (MILP):** an optimisation problem in which some decisions are yes-or-no choices.
- **Quantised model:** a compressed model, stored with fewer bits per number.
- **Validator:** the code that checks the model's choice before it is used.

---

## 13. How to explain the pipeline to someone else

**The one-minute version.** We answer business questions over linked company data. A small open-weight model reads each question and chooses which of 50 tested calculations applies. A validator checks that choice against the answer type, retries once, and flags the question if it still fails. Python then computes the number. The optimisation and simulation questions use an exact solver and an event simulator, each with its own checks. The model never produces a number. On the public questions, its choices matched the hand routing in every final answer.

**A worked example** (from the public toy examples, which are safe to show):

The question asks for net revenue in EUR for a single-line order: quantity 2, unit price 50 USD, a line discount of 10 USD, an order discount of 5 USD, a rate of 0.90 EUR per USD for the order month, and an approved refund of 10 USD converted at 0.92 for the refund month.

- Discounted amount: 2 × 50 − 10 − 5 = 85 USD.
- Converted at the order-month rate: 85 × 0.90 = 76.50 EUR.
- Refund converted at the refund-month rate: 10 × 0.92 = 9.20 EUR, subtracted.
- Net revenue: 76.50 − 9.20 = **67.30 EUR**, which matches the organiser's answer.

The point of the example is that each step is a named rule, and the rate used depends on the month, so each number can be traced.

**Questions people ask, and short answers:**

- *Why use a model at all?* To read the question and choose the right calculation, which is a language task. Arithmetic over the tables is left to code, which is exact.
- *Why not let the model compute the answer?* Language models make arithmetic errors on large tables, and their answers are hard to check. Choosing among tested functions is easier to check.
- *How do you know the answers are right?* We cannot know for certain, because no hidden answers are available. We have the practice and toy answers, an independent SQL implementation for 13 questions, a re-costing check for every optimal plan, a conservation check for every simulation, and a structure check for the file.
- *What happens when the model is wrong?* The validator catches choices that are the wrong type or have the wrong arguments. A question that fails twice is flagged and answered by hand routing. A wrong choice that passes the validator would not be caught.
- *Does the model earn the bonus?* No bonus is claimed. The model's routing matches the hand routing in every final answer, so its contribution is open to question. The declaration states usage only.
- *What about the coding assistant?* It wrote the code, tests and documents. It did not run in the pipeline and did not produce any answer.
- *Can the work be reproduced?* Yes, from a fresh copy with the organiser data and the model file, using the steps in section 6.
- *What is left?* The commit and push before the deadline, the arXiv report after coordination with the organisers, Stage 2 when the data is released, and the presentation if the team is a finalist.
