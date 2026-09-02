"""
run_pipeline.py
Level 1 entry point: loads data, analyzes it, preprocesses it,
trains baseline models, evaluates them, and saves the best one.

Run with:  python3 run_pipeline.py
"""

import sys
import json
import joblib
from pathlib import Path

sys.path.append("src/data")
sys.path.append("src/ml")

from loader import load_dataset
from validator import analyze_dataset, print_report
from preprocessing import get_train_test
from training import get_models, train_model
from evaluation import evaluate_model, compare_models

DATA_PATH = "data/telco_churn.csv"
MODELS_DIR = Path("models")
EXPERIMENTS_DIR = Path("experiments")


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    print("=" * 50)
    print("STEP 1: Load dataset")
    print("=" * 50)
    df = load_dataset(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("=" * 50)
    print("STEP 2: Analyze dataset")
    print("=" * 50)
    report = analyze_dataset(df)
    print_report(report)
    print()

    print("=" * 50)
    print("STEP 3: Preprocess + split")
    print("=" * 50)
    X_train, X_test, y_train, y_test, preprocessor = get_train_test(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    print("=" * 50)
    print("STEP 4: Train baseline models")
    print("=" * 50)
    results = {}
    pipelines = {}
    for name, estimator in get_models().items():
        print(f"Training {name}...")
        pipe = train_model(name, estimator, preprocessor, X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test)
        results[name] = metrics
        pipelines[name] = pipe
    print()

    print("=" * 50)
    print("STEP 5: Compare models")
    print("=" * 50)
    comparison = compare_models(results)
    print(comparison)
    print()

    best_name = comparison.index[0]
    best_pipeline = pipelines[best_name]
    best_metrics = results[best_name]
    print(f"Winner: {best_name}")
    print()

    print("=" * 50)
    print("STEP 6: Save best model + experiment log")
    print("=" * 50)
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"Saved model to {model_path}")

    experiment_log = {
        "dataset": DATA_PATH,
        "rows": report["rows"],
        "models_tried": list(results.keys()),
        "results": results,
        "winner": best_name,
        "winner_metrics": best_metrics,
    }
    log_path = EXPERIMENTS_DIR / "experiment_001.json"
    with open(log_path, "w") as f:
        json.dump(experiment_log, f, indent=2)
    print(f"Saved experiment log to {log_path}")


if __name__ == "__main__":
    main()
