"""
run_experiments.py
Level 2 entry point: Automatic Experimentation.

Instead of training 3 models once (Level 1), this automatically loops
through 9 configurations (3 per algorithm family), logs every single
one as its own experiment file, then reports a leaderboard across
EVERYTHING ever run — including experiments from previous sessions.

Run with:  python3 run_experiments.py
"""

import sys
import json
import joblib
from pathlib import Path

sys.path.append("src/data")
sys.path.append("src/ml")

from loader import load_dataset
from preprocessing import get_train_test, DATASET_VERSION
from model_configs import get_experiment_configs
from evaluation import evaluate_model
from experiment_manager import run_all_experiments, leaderboard, load_all_experiment_logs
from experiment_tracker import print_history_summary

DATA_PATH = "data/telco_churn.csv"
MODELS_DIR = Path("models")


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 55)
    print("LEVEL 3: EXPERIMENT TRACKING")
    print("=" * 55)

    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = get_train_test(df)
    print(f"Data ready. Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Dataset/feature version: {DATASET_VERSION}\n")

    configs = get_experiment_configs()
    print(f"{len(configs)} configs defined. Checking history and running new ones...\n")

    results = run_all_experiments(
        configs, preprocessor, X_train, y_train, X_test, y_test, evaluate_model,
        dataset_version=DATASET_VERSION, skip_duplicates=True,
    )

    print("\n" + "=" * 55)
    print("LEADERBOARD (all experiments ever run, best F1 first)")
    print("=" * 55)
    board = leaderboard()
    print(board.to_string(index=False))

    print("\n" + "=" * 55)
    print("EXPERIMENT HISTORY SUMMARY")
    print("=" * 55)
    print_history_summary()

    # Save the single best model found across ALL history, not just this run
    best_row = board.iloc[0]
    best_exp_id = int(best_row["experiment_id"])
    print(f"\nBest experiment overall: #{best_exp_id} ({best_row['name']}, F1={best_row['f1']:.3f})")

    # Find the matching pipeline from this run (if the winner came from this run)
    matching = [r for r in results if r["log"]["experiment_id"] == best_exp_id]
    if matching:
        best_pipeline = matching[0]["pipeline"]
        joblib.dump(best_pipeline, MODELS_DIR / "best_model.joblib")
        print(f"Saved as models/best_model.joblib")
    else:
        print("Best model was from an earlier run and isn't in memory this session; "
              "re-run to regenerate it if needed.")


if __name__ == "__main__":
    main()
