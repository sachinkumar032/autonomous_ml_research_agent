"""
loader.py
Loads the raw dataset from disk and does basic type cleanup.
"""

import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Load the Telco Customer Churn CSV into a DataFrame."""
    df = pd.read_csv(path)

    # TotalCharges is stored as a string in the raw file and has blank
    # entries for brand-new customers (tenure = 0). Convert to numeric.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


if __name__ == "__main__":
    df = load_dataset("data/telco_churn.csv")
    print(df.shape)
    print(df.dtypes)
