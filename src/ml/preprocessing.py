"""
preprocessing.py
Deterministic preprocessing pipeline. This is Component 3 (Feature
Engineering) implemented as real, reproducible Python — no LLM in
the execution path, only in deciding *which* strategy to use later.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


DROP_COLS = ["customerID"]
TARGET = "Churn"

# Bump this whenever preprocessing/feature logic changes meaningfully.
# Experiment tracking uses this so old and new experiments aren't
# compared as if they used the same features.
DATASET_VERSION = "v1"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.drop_duplicates()
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = (df[TARGET] == "Yes").astype(int)  # Yes/No -> 1/0
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor


def get_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    df = clean(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    preprocessor = build_preprocessor(X_train)
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    import sys
    sys.path.append("src/data")
    from loader import load_dataset

    df = load_dataset("data/telco_churn.csv")
    X_train, X_test, y_train, y_test, preprocessor = get_train_test(df)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Train churn rate:", y_train.mean().round(3))
    print("Test churn rate:", y_test.mean().round(3))
