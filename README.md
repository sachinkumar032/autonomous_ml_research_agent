# Autonomous ML Research & Experimentation Agent

**Status: Level 1 — Normal ML pipeline (baseline)**

This is the foundation of a larger project that will eventually add
LLM-driven agentic experimentation on top of a real ML pipeline. Level 1
proves the ML fundamentals work before any agent logic is added.

## What this does right now

1. Loads the Telco Customer Churn dataset (7,043 rows).
2. Analyzes it: shape, missing values, duplicates, class balance.
3. Preprocesses it: imputes missing values, scales numeric features,
   one-hot encodes categorical features.
4. Trains three baseline models: Logistic Regression, Random Forest, XGBoost.
5. Evaluates each on accuracy, precision, recall, F1, and ROC-AUC.
6. Picks the best model (by F1) and saves it, along with an experiment log.

## Project structure

```
autonomous-ml-research-agent/
├── data/                   # dataset
├── src/
│   ├── data/
│   │   ├── loader.py       # loads + cleans raw CSV
│   │   └── validator.py    # dataset analysis / report
│   └── ml/
│       ├── preprocessing.py  # train/test split + preprocessing pipeline
│       ├── training.py       # model definitions + training
│       └── evaluation.py     # metrics + model comparison
├── models/                 # saved best model (generated, gitignored)
├── experiments/            # experiment logs (JSON)
├── tests/                  # pytest sanity tests
├── run_pipeline.py         # main entry point
└── requirements.txt
```

(The `agents/`, `tools/`, and `api/` folders are placeholders for later levels.)

## How to run it

```bash
pip install -r requirements.txt
python3 run_pipeline.py
```

Run the tests:

```bash
pytest tests/ -v
```

## Results (baseline, no tuning)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 80.3% | 0.661 | 0.524 | 0.585 | 0.840 |
| XGBoost | 78.6% | 0.621 | 0.489 | 0.547 | 0.833 |
| Random Forest | 77.9% | 0.613 | 0.452 | 0.520 | 0.815 |

Logistic Regression wins on F1 at Level 1. This is expected — no
hyperparameter tuning or class-imbalance handling has been applied yet.
Later levels will add experiment tracking, error analysis, and an agent
that decides how to improve on this.

## Roadmap

- [x] **Level 1** — Dataset, EDA, preprocessing, baseline models (this repo state)
- [ ] **Level 2** — Automatic experimentation loop across models/configs
- [ ] **Level 3** — Experiment tracking (MLflow / structured logs)
- [ ] **Level 4** — LLM + tool calling
- [ ] **Level 5** — LangGraph supervisor with conditional improvement loop
- [ ] **Level 6** — Memory + RAG for ML guidance
- [ ] **Level 7** — FastAPI + Docker deployment
- [ ] **Level 8** — Dashboard

## Dataset

[Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) —
a standard binary classification dataset for churn prediction.
