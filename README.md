# Autonomous ML Research & Experimentation Agent

**Status: Level 2 — Automatic experimentation loop**

This is the foundation of a larger project that will eventually add
LLM-driven agentic experimentation on top of a real ML pipeline. Levels 1-2
prove the ML fundamentals and automation work before any agent logic is added.

## What this does right now

**Level 1 (baseline):**
1. Loads the Telco Customer Churn dataset (7,043 rows).
2. Analyzes it: shape, missing values, duplicates, class balance.
3. Preprocesses it: imputes missing values, scales numeric features,
   one-hot encodes categorical features.
4. Trains three baseline models: Logistic Regression, Random Forest, XGBoost.
5. Evaluates each on accuracy, precision, recall, F1, and ROC-AUC.
6. Picks the best model (by F1) and saves it, along with an experiment log.

**Level 2 (automatic experimentation loop) — new:**
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
│       └── experiment_manager.py  # automatic run/log/leaderboard loop (Level 2)
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

Level 2 (9 automatic experiments, full logging + leaderboard):
```bash
python3 run_experiments.py
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
- [x] **Level 2** — Automatic experimentation loop across models/configs (this repo state)
- [ ] **Level 3** — Experiment tracking (avoid re-running identical configs, structured history)
- [ ] **Level 4** — LLM + tool calling
- [ ] **Level 5** — LangGraph supervisor with conditional improvement loop
- [ ] **Level 6** — Memory + RAG for ML guidance
- [ ] **Level 7** — FastAPI + Docker deployment
- [ ] **Level 8** — Dashboard

## Dataset

[Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) —
a standard binary classification dataset for churn prediction.
