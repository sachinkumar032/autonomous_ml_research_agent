# Autonomous ML Research & Experimentation Agent

**Status: Level 5 — LangGraph supervisor with conditional improvement loop**

This is the foundation of a larger project that adds LLM-driven agentic
experimentation on top of a real ML pipeline. Levels 1-3 proved the ML
fundamentals, automation, and memory work. Level 4 introduces the first
real LLM decision-making, restricted to a fixed set of tools.

## What this does right now

**Level 1 (baseline):**
1. Loads the Telco Customer Churn dataset (7,043 rows).
2. Analyzes it: shape, missing values, duplicates, class balance.
3. Preprocesses it: imputes missing values, scales numeric features,
   one-hot encodes categorical features.
4. Trains three baseline models: Logistic Regression, Random Forest, XGBoost.
5. Evaluates each on accuracy, precision, recall, F1, and ROC-AUC.
6. Picks the best model (by F1) and saves it, along with an experiment log.

**Level 2 (automatic experimentation loop):**
1. Defines 9 experiment configurations automatically: 3 hyperparameter
   variants each for Logistic Regression, Random Forest, and XGBoost.
2. Loops through every config, training and evaluating each one without
   manual intervention:
   ```python
   for config in configs:
       train()
       evaluate()
       save_result()
   ```
3. Every single experiment gets its own numbered log file
   (`experiments/experiment_001.json`, `002.json`, ...) — nothing is
   overwritten, so history accumulates across runs.
4. Builds a **leaderboard** across every experiment ever logged on disk
   (not just the current run), ranked by F1.
5. Saves the single best-performing model found across all history.

**Level 3 (experiment tracking) — new:**
1. Every config now has a deterministic **signature** — a hash of
   `(model_family, hyperparameters, dataset_version)`. Same config,
   same signature, regardless of dict key ordering.
2. Before training, the system checks: *"have I already tried this exact
   config on this exact dataset version?"* If yes, it's **skipped**
   instead of wastefully retrained.
3. `preprocessing.py` now exposes a `DATASET_VERSION` constant. Bump it
   whenever feature engineering changes, so old and new experiments are
   never silently compared as if they used the same features.
4. A new `experiment_tracker.py` module summarizes history — total
   experiments run, best/worst F1 per model family, and the overall
   best — in a compact structured shape. This is exactly what a future
   LLM agent (Level 5) will read to answer "what have I already tried?"
   and decide what to do next.

Run it twice in a row and see the difference:
```
Skipping LogReg_C0.1 — already tried as experiment #1 (F1=0.584)
...
9 experiment(s) skipped as duplicates of existing history.
```
Add one new config to `model_configs.py` and re-run — only the new one trains.

**Level 4 (LLM + tool calling) — new:**
1. Five tools are exposed to an LLM (Gemini), matching Component 14 of
   the blueprint: `analyze_dataset`, `get_leaderboard`, `get_history_summary`,
   `check_already_tried`, and `run_experiment`.
2. The LLM is given a goal in plain English (e.g. *"try a Random Forest
   with more depth if it's new"*) and decides for itself which tools to
   call, in what order, based on the results it gets back. Google's
   `google-genai` SDK handles this automatically — plain Python functions
   are passed as tools, and the SDK loops through calling Gemini,
   executing whatever tool it requests, feeding results back, until
   Gemini gives a final answer.
3. Python still does every calculation. The LLM never computes a metric
   or trains a model itself — it only decides *which* deterministic tool
   to call and interprets the structured result.
4. **Safety gate (blueprint Section 21):** reading the dataset or history
   is LOW RISK and auto-executed. Actually training a new model via
   `run_experiment` is MEDIUM RISK — the tool itself pauses and asks a
   human to type `y`/`n` before it actually runs.

Example:
```bash
python3 run_agent.py "What's the best experiment so far and why?"
python3 run_agent.py "Try Random Forest with max_depth 8 if it's not already tried"
```
The second example will make the agent call `check_already_tried` first,
then — if approved — `run_experiment`, training a real model.

Uses `gemini-3.6-flash` by default (override with `GEMINI_MODEL`). (Note:
Gemini 2.5 models are being retired by Google on October 16, 2026, so
this project uses a current-generation model.)

**Level 5 (LangGraph supervisor + conditional loop) — new:**

This is the piece the blueprint calls "the strongest demonstration of
agentic behavior" — a real feedback loop with a decision point, not
just a fixed sequence of steps:

```
Analyze Dataset -> Preprocess -> Train Baseline -> Error Analysis
                                                          |
                                                   Need Improvement?
                                              NO ---------+--------- YES
                                              |                       |
                                    Select Best Model      Create + Train Next Experiment
                                              |                       |
                                      Generate Report         (loops back to Error Analysis)
                                              |
                                             END
```

1. `src/ml/error_analysis.py` — Component 6 of the blueprint, finally
   built. Inspects false negatives/positives and looks for numeric
   feature patterns (e.g. "false negatives skew toward lower tenure"),
   not just a single score.
2. `src/agents/workflow.py` — the graph, built with LangGraph. Each
   step is a node; `decide_node` is the conditional branch point.
3. **The stopping rule** (`decide_node`) checks three things in order:
   - Has the target F1 been reached? -> stop.
   - Has the max iteration count been hit? -> stop.
   - Are there any untried configs left? -> stop if not, otherwise improve.
4. On "improve", the graph automatically trains the next untried
   config from `model_configs.py`, re-runs error analysis, and asks
   the question again — a real loop, not a fixed list.
5. On "stop", it picks the best experiment across all history, and
   generates a plain-text summary report.

Run it:
```bash
python3 run_workflow.py --target-f1 0.62 --max-iterations 4
```
Try a low `--target-f1` (e.g. 0.3) to see it stop immediately after the
baseline, or a high one (e.g. 0.95) to see it exhaust every untried
config before stopping.

## Project structure

```
autonomous-ml-research-agent/
├── data/                   # dataset
├── src/
│   ├── data/
│   │   ├── loader.py       # loads + cleans raw CSV
│   │   └── validator.py    # dataset analysis / report
│   └── ml/
│       ├── preprocessing.py     # train/test split + preprocessing pipeline
│       ├── training.py          # model definitions + training (Level 1)
│       ├── evaluation.py        # metrics + model comparison
│       ├── model_configs.py     # 9 experiment configs (Level 2)
│       ├── experiment_manager.py  # run/log/leaderboard loop, skip-duplicates (Level 2-3)
│       ├── experiment_tracker.py  # signatures, duplicate detection, history summary (Level 3)
│       └── error_analysis.py    # false positive/negative pattern detection (Level 5)
├── src/
│   ├── tools/
│   │   ├── data_tools.py    # analyze_dataset as an LLM-callable tool (Level 4)
│   │   └── ml_tools.py      # leaderboard, history, check/run experiment tools (Level 4)
│   └── agents/
│       ├── agent_tools.py       # tool wrappers w/ docstrings for Gemini auto function-calling (Level 4)
│       ├── supervisor.py        # Gemini agent loop (Level 4, active)
│       ├── supervisor_claude.py     # alternate Claude/Anthropic version (unused by default)
│       ├── tool_schemas_claude.py   # manual tool schemas for the Claude version
│       ├── graph_state.py       # shared state definition for the LangGraph workflow (Level 5)
│       └── workflow.py          # the LangGraph graph + conditional improvement loop (Level 5)
├── models/                 # saved best model (generated, gitignored)
├── experiments/            # one JSON log per experiment, auto-numbered
├── tests/                  # pytest sanity tests
├── run_pipeline.py         # Level 1 entry point (3 models, once)
├── run_experiments.py      # Level 2 entry point (9 configs, auto-logged)
└── requirements.txt
```

(The `agents/`, `tools/`, and `api/` folders are placeholders for later levels.)

## How to run it

Level 1 (3 baseline models, single run):
```bash
pip install -r requirements.txt
python3 run_pipeline.py
```

Level 2 & 3 (automatic experiments with skip-duplicate tracking):
```bash
python3 run_experiments.py
```

Level 4 (LLM agent — requires a Gemini API key):
```bash
# Get a key at aistudio.google.com/apikey, then set it:
export GEMINI_API_KEY=AIza...      # Mac/Linux
setx GEMINI_API_KEY "AIza..."       # Windows (open a new terminal after)

python3 run_agent.py "What's the best experiment so far and why?"
python3 run_agent.py "Try Random Forest with max_depth 8 if it's not already tried"
```

Level 5 (LangGraph conditional improvement loop):
```bash
python3 run_workflow.py --target-f1 0.62 --max-iterations 4
```

Run the tests:

```bash
pytest tests/ -v
```

## Results

**Level 1 baseline (no tuning):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 80.3% | 0.661 | 0.524 | 0.585 | 0.840 |
| XGBoost | 78.6% | 0.621 | 0.489 | 0.547 | 0.833 |
| Random Forest | 77.9% | 0.613 | 0.452 | 0.520 | 0.815 |

**Level 2 leaderboard (9 automatic configs, best F1 first):**

| Experiment | Model | F1 | ROC-AUC |
|---|---|---|---|
| LogReg_C1.0 | Logistic Regression | 0.585 | 0.840 |
| LogReg_C0.1 | Logistic Regression | 0.584 | 0.840 |
| LogReg_C10.0 | Logistic Regression | 0.584 | 0.840 |
| RF_n500_d10 | Random Forest | 0.571 | 0.835 |
| XGB_d3_lr0.1 | XGBoost | 0.561 | 0.836 |

Logistic Regression still wins on F1. This is expected — no
hyperparameter search has found meaningful gains yet, and class
imbalance hasn't been addressed. The point of Level 2 isn't a better
score; it's that the *search itself* now runs unattended and every
result is preserved. Error analysis and smarter experiment selection
come at higher levels.

## Roadmap

- [x] **Level 1** — Dataset, EDA, preprocessing, baseline models
- [x] **Level 2** — Automatic experimentation loop across models/configs
- [x] **Level 3** — Experiment tracking: signatures, duplicate-skipping, history summary
- [x] **Level 4** — LLM + tool calling with a human-approval gate on training
- [x] **Level 5** — LangGraph supervisor with conditional improvement loop (this repo state)
- [ ] **Level 6** — Memory + RAG for ML guidance
- [ ] **Level 5** — LangGraph supervisor with conditional improvement loop
- [ ] **Level 6** — Memory + RAG for ML guidance
- [ ] **Level 7** — FastAPI + Docker deployment
- [ ] **Level 8** — Dashboard

## Dataset

[Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) —
a standard binary classification dataset for churn prediction.
