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
