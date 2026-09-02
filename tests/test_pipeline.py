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


# --- Level 3: experiment tracking ---

def test_signature_is_deterministic_regardless_of_key_order():
    from experiment_tracker import config_signature
    sig_a = config_signature("XGBoost", {"max_depth": 5, "learning_rate": 0.05}, "v1")
    sig_b = config_signature("XGBoost", {"learning_rate": 0.05, "max_depth": 5}, "v1")
    assert sig_a == sig_b


def test_signature_differs_for_different_params():
    from experiment_tracker import config_signature
    sig_a = config_signature("XGBoost", {"max_depth": 5}, "v1")
    sig_b = config_signature("XGBoost", {"max_depth": 6}, "v1")
    assert sig_a != sig_b


def test_signature_differs_for_different_dataset_version():
    from experiment_tracker import config_signature
    sig_a = config_signature("XGBoost", {"max_depth": 5}, "v1")
    sig_b = config_signature("XGBoost", {"max_depth": 5}, "v2")
    assert sig_a != sig_b


def test_already_tried_detects_exact_match():
    from experiment_tracker import already_tried, config_signature
    sig = config_signature("XGBoost", {"max_depth": 5}, "v1")
    fake_logs = [{"experiment_id": 1, "signature": sig, "metrics": {"f1": 0.5}}]
    match = already_tried("XGBoost", {"max_depth": 5}, "v1", logs=fake_logs)
    assert match is not None
    assert match["experiment_id"] == 1


def test_already_tried_returns_none_for_new_config():
    from experiment_tracker import already_tried, config_signature
    sig = config_signature("XGBoost", {"max_depth": 5}, "v1")
    fake_logs = [{"experiment_id": 1, "signature": sig, "metrics": {"f1": 0.5}}]
    match = already_tried("XGBoost", {"max_depth": 99}, "v1", logs=fake_logs)
    assert match is None


def test_summarize_history_groups_by_model_family():
    from experiment_tracker import summarize_history
    fake_logs = [
        {"experiment_id": 1, "model_family": "XGBoost", "metrics": {"f1": 0.5}},
        {"experiment_id": 2, "model_family": "XGBoost", "metrics": {"f1": 0.6}},
        {"experiment_id": 3, "model_family": "Random Forest", "metrics": {"f1": 0.4}},
    ]
    summary = summarize_history(fake_logs)
    assert summary["total_experiments"] == 3
    assert summary["by_model_family"]["XGBoost"]["count"] == 2
    assert summary["by_model_family"]["XGBoost"]["best_f1"] == 0.6
    assert summary["best"]["experiment_id"] == 2
