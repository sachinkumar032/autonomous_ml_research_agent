"""
Basic sanity tests for the Level 1 pipeline.
Run with: pytest tests/
"""

import sys
sys.path.append("src/data")
sys.path.append("src/ml")

from loader import load_dataset
from validator import analyze_dataset
from preprocessing import get_train_test
from training import get_models, train_model
from evaluation import evaluate_model

DATA_PATH = "data/telco_churn.csv"


def test_load_dataset():
    df = load_dataset(DATA_PATH)
    assert df.shape[0] > 7000
    assert "Churn" in df.columns


def test_analyze_dataset():
    df = load_dataset(DATA_PATH)
    report = analyze_dataset(df)
    assert report["rows"] == df.shape[0]
    assert report["duplicates"] == 0


def test_train_test_split_stratified():
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = get_train_test(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_model_trains_and_scores_reasonably():
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = get_train_test(df)
    estimator = get_models()["Logistic Regression"]
    pipe = train_model("Logistic Regression", estimator, preprocessor, X_train, y_train)
    metrics = evaluate_model(pipe, X_test, y_test)
    assert metrics["accuracy"] > 0.7
    assert metrics["roc_auc"] > 0.7


# --- Level 2: automatic experimentation loop ---

def test_experiment_configs_defined():
    from model_configs import get_experiment_configs
    configs = get_experiment_configs()
    assert len(configs) >= 6
    families = {c["model_family"] for c in configs}
    assert {"Logistic Regression", "Random Forest", "XGBoost"} <= families


def test_run_single_experiment_creates_log(tmp_path, monkeypatch):
    from model_configs import get_experiment_configs
    from experiment_manager import run_experiment
    import experiment_manager

    monkeypatch.setattr(experiment_manager, "EXPERIMENTS_DIR", tmp_path)

    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = get_train_test(df)
    config = get_experiment_configs()[0]

    result = run_experiment(config, preprocessor, X_train, y_train, X_test, y_test, evaluate_model)

    assert "f1" in result["log"]["metrics"]
    saved_files = list(tmp_path.glob("experiment_*.json"))
    assert len(saved_files) == 1


def test_leaderboard_ranks_by_f1(tmp_path, monkeypatch):
    from experiment_manager import leaderboard

    fake_logs = [
        {"experiment_id": 1, "name": "A", "model_family": "X", "metrics": {"f1": 0.5, "accuracy": 0.8}},
        {"experiment_id": 2, "name": "B", "model_family": "X", "metrics": {"f1": 0.7, "accuracy": 0.8}},
    ]
    board = leaderboard(logs=fake_logs)
    assert board.iloc[0]["name"] == "B"  # higher F1 should rank first
