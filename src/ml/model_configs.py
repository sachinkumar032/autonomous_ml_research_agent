"""
model_configs.py
Defines several configurations per algorithm so the experiment loop
has real variation to work through automatically — not just one run
per model type. This is what makes Level 2 "automatic experimentation"
rather than "train 3 models once".
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_experiment_configs() -> list[dict]:
    """
    Returns a list of experiment configs. Each config is:
    {
        "name": human-readable experiment name,
        "model_family": "Logistic Regression" | "Random Forest" | "XGBoost",
        "params": dict of hyperparameters used (for logging),
        "estimator": the actual unfitted sklearn/xgboost estimator,
    }
    """
    configs = []

    # --- Logistic Regression variants ---
    for C in [0.1, 1.0, 10.0]:
        configs.append({
            "name": f"LogReg_C{C}",
            "model_family": "Logistic Regression",
            "params": {"C": C, "max_iter": 1000},
            "estimator": LogisticRegression(C=C, max_iter=1000, random_state=42),
        })

    # --- Random Forest variants ---
    for n_estimators, max_depth in [(200, 6), (300, None), (500, 10)]:
        configs.append({
            "name": f"RF_n{n_estimators}_d{max_depth}",
            "model_family": "Random Forest",
            "params": {"n_estimators": n_estimators, "max_depth": max_depth},
            "estimator": RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
            ),
        })

    # --- XGBoost variants ---
    for max_depth, learning_rate in [(3, 0.1), (5, 0.05), (7, 0.03)]:
        configs.append({
            "name": f"XGB_d{max_depth}_lr{learning_rate}",
            "model_family": "XGBoost",
            "params": {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_estimators": 300,
            },
            "estimator": XGBClassifier(
                n_estimators=300,
                max_depth=max_depth,
                learning_rate=learning_rate,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
        })

    return configs
