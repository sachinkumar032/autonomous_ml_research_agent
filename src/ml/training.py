"""
training.py
Trains baseline models. Component 4 of the blueprint (ML Experiment
Agent) at Level 1 = plain automation, no agent reasoning yet.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def get_models() -> dict:
    """Returns a dict of model_name -> unfitted estimator."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_model(name: str, estimator, preprocessor, X_train, y_train) -> Pipeline:
    """Wraps an estimator with the shared preprocessor and fits it."""
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


if __name__ == "__main__":
    import sys
    sys.path.append("src/data")
    from loader import load_dataset
    from preprocessing import get_train_test

    df = load_dataset("data/telco_churn.csv")
    X_train, X_test, y_train, y_test, preprocessor = get_train_test(df)

    models = get_models()
    for name, estimator in models.items():
        pipe = train_model(name, estimator, preprocessor, X_train, y_train)
        print(f"Trained: {name}")
