"""
ml_tools.py
More tools given to the agent, covering experiment history and
running new experiments. Per the blueprint's autonomy levels
(Section 21), reading history is LOW RISK and auto-allowed, but
actually training a new model is MEDIUM RISK and gated behind
human approval — enforced by the agent loop in supervisor.py,
not inside these functions themselves (so the functions stay
simple and reusable).
"""

import sys
sys.path.append("src/data")
sys.path.append("src/ml")

from loader import load_dataset
from preprocessing import get_train_test, DATASET_VERSION
from evaluation import evaluate_model
from experiment_manager import leaderboard as _leaderboard, run_experiment as _run_experiment
from experiment_tracker import summarize_history as _summarize_history, already_tried

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA_PATH = "data/telco_churn.csv"

_ESTIMATOR_BUILDERS = {
    "Logistic Regression": lambda p: LogisticRegression(
        C=p.get("C", 1.0), max_iter=p.get("max_iter", 1000), random_state=42
    ),
    "Random Forest": lambda p: RandomForestClassifier(
        n_estimators=p.get("n_estimators", 300), max_depth=p.get("max_depth"),
        random_state=42, n_jobs=-1,
    ),
    "XGBoost": lambda p: XGBClassifier(
        n_estimators=p.get("n_estimators", 300), max_depth=p.get("max_depth", 5),
        learning_rate=p.get("learning_rate", 0.05), eval_metric="logloss",
        random_state=42, n_jobs=-1,
    ),
}


def get_leaderboard_tool(top_n: int = 5) -> dict:
    """Returns the top N experiments ever logged, ranked by F1 score."""
    board = _leaderboard(top_n=top_n)
    if board.empty:
        return {"experiments": []}
    return {"experiments": board.to_dict(orient="records")}


def get_history_summary_tool() -> dict:
    """Returns a summary of all experiments run so far, grouped by model family."""
    return _summarize_history()


def check_already_tried_tool(model_family: str, params: dict) -> dict:
    """
    Checks whether a given (model_family, params) config has already been
    run at the current dataset version. Use this BEFORE proposing to run
    a new experiment, to avoid wasting a training run on something already tried.
    """
    match = already_tried(model_family, params, DATASET_VERSION)
    if match is None:
        return {"already_tried": False}
    return {"already_tried": True, "previous_result": match}


def run_experiment_tool(model_family: str, params: dict) -> dict:
    """
    Actually trains and evaluates ONE model configuration and logs it as
    a new experiment. This performs real training — only call this after
    confirming (via check_already_tried_tool) that the config is genuinely
    new, and only when the calling context has approved it.
    """
    if model_family not in _ESTIMATOR_BUILDERS:
        return {"error": f"Unknown model_family '{model_family}'. "
                          f"Must be one of: {list(_ESTIMATOR_BUILDERS.keys())}"}

    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = get_train_test(df)
    estimator = _ESTIMATOR_BUILDERS[model_family](params)

    config = {
        "name": f"{model_family}_agent_{'_'.join(f'{k}{v}' for k, v in params.items())}",
        "model_family": model_family,
        "params": params,
        "estimator": estimator,
    }

    result = _run_experiment(
        config, preprocessor, X_train, y_train, X_test, y_test,
        evaluate_model, dataset_version=DATASET_VERSION,
    )
    return result["log"]
