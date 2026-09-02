"""
evaluation.py
Component 5 of the blueprint. Uses metrics appropriate for
binary classification, not just accuracy.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def compare_models(results: dict) -> pd.DataFrame:
    """results: dict of model_name -> metrics dict"""
    table = pd.DataFrame(results).T
    table = table.sort_values("f1", ascending=False)
    table_display = table.copy()
    table_display["accuracy"] = (table_display["accuracy"] * 100).round(1).astype(str) + "%"
    for col in ["precision", "recall", "f1", "roc_auc"]:
        table_display[col] = table_display[col].round(3)
    return table_display


if __name__ == "__main__":
    import sys
    sys.path.append("src/data")
    from loader import load_dataset
    from preprocessing import get_train_test
    from training import get_models, train_model

    df = load_dataset("data/telco_churn.csv")
    X_train, X_test, y_train, y_test, preprocessor = get_train_test(df)

    results = {}
    for name, estimator in get_models().items():
        pipe = train_model(name, estimator, preprocessor, X_train, y_train)
        results[name] = evaluate_model(pipe, X_test, y_test)

    print(compare_models(results))
