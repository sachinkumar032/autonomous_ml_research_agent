"""
experiment_manager.py
Level 2: Automatic Experimentation.

for config in configs:
    train()
    evaluate()
    save_result()

Every experiment gets its own numbered log file in experiments/, so
nothing is lost and later levels (tracking, agent reasoning) can read
this history. This module has no LLM in it — it is pure automation,
which is the point of Level 2 per the blueprint.
"""

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

EXPERIMENTS_DIR = Path("experiments")


def _next_experiment_id() -> int:
    """Looks at existing experiment_NNN.json files and returns the next id."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    existing = list(EXPERIMENTS_DIR.glob("experiment_*.json"))
    if not existing:
        return 1
    ids = []
    for f in existing:
        try:
            ids.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(ids, default=0) + 1


def run_experiment(config: dict, preprocessor, X_train, y_train, X_test, y_test,
                    evaluate_fn) -> dict:
    """Trains one config, evaluates it, saves a numbered experiment log, returns the result."""
    exp_id = _next_experiment_id()

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", config["estimator"]),
    ])

    start = time.time()
    pipeline.fit(X_train, y_train)
    train_seconds = round(time.time() - start, 2)

    metrics = evaluate_fn(pipeline, X_test, y_test)

    log = {
        "experiment_id": exp_id,
        "name": config["name"],
        "model_family": config["model_family"],
        "params": config["params"],
        "train_seconds": train_seconds,
        "metrics": metrics,
    }

    log_path = EXPERIMENTS_DIR / f"experiment_{exp_id:03d}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    return {"log": log, "pipeline": pipeline}


def run_all_experiments(configs: list[dict], preprocessor, X_train, y_train,
                         X_test, y_test, evaluate_fn) -> list[dict]:
    """Runs every config in sequence, logging each as it goes. Returns list of results."""
    results = []
    for config in configs:
        print(f"Running experiment: {config['name']} ({config['model_family']})...")
        result = run_experiment(config, preprocessor, X_train, y_train, X_test, y_test, evaluate_fn)
        f1 = result["log"]["metrics"]["f1"]
        print(f"  -> F1 = {f1:.3f}")
        results.append(result)
    return results


def load_all_experiment_logs() -> list[dict]:
    """Reads every experiment log currently on disk, sorted by experiment_id."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    logs = []
    for f in sorted(EXPERIMENTS_DIR.glob("experiment_*.json")):
        with open(f) as fh:
            logs.append(json.load(fh))
    return sorted(logs, key=lambda l: l.get("experiment_id", 0))


def leaderboard(logs: list[dict] = None, top_n: int = 10) -> pd.DataFrame:
    """Builds a ranked leaderboard across ALL experiments ever run (reads history from disk)."""
    if logs is None:
        logs = load_all_experiment_logs()

    rows = []
    for log in logs:
        row = {
            "experiment_id": log.get("experiment_id"),
            "name": log.get("name"),
            "model_family": log.get("model_family"),
            **log.get("metrics", {}),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    return df.head(top_n)
