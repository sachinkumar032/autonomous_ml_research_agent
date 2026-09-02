"""
validator.py
Inspects the dataset: shape, dtypes, missing values, duplicates,
numerical/categorical split, and class balance of the target.
This is the "Dataset Agent" logic from the blueprint (Component 1),
implemented as deterministic Python — no LLM needed at this level.
"""

import pandas as pd


def analyze_dataset(df: pd.DataFrame, target: str = "Churn") -> dict:
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()

    # target shouldn't be counted as a feature column
    if target in numerical_cols:
        numerical_cols.remove(target)
    if target in categorical_cols:
        categorical_cols.remove(target)

    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target": target,
        "numerical_features": len(numerical_cols),
        "categorical_features": len(categorical_cols),
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "missing_values_total": int(df.isnull().sum().sum()),
        "missing_by_column": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }

    if target in df.columns:
        counts = df[target].value_counts(normalize=True) * 100
        report["class_distribution"] = counts.round(1).to_dict()

    return report


def print_report(report: dict) -> None:
    print("Dataset Analysis")
    print(f"Rows: {report['rows']:,}")
    print(f"Columns: {report['columns']}")
    print(f"Target: {report['target']}")
    print(f"Numerical features: {report['numerical_features']}")
    print(f"Categorical features: {report['categorical_features']}")
    print(f"Missing values: {report['missing_values_total']}")
    print(f"Duplicates: {report['duplicates']}")
    if "class_distribution" in report:
        dist = ", ".join(f"{k}: {v}%" for k, v in report["class_distribution"].items())
        print(f"Class distribution: {dist}")


if __name__ == "__main__":
    from loader import load_dataset

    df = load_dataset("data/telco_churn.csv")
    report = analyze_dataset(df)
    print_report(report)
