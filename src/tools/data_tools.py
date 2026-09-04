"""
data_tools.py
Component 14 of the blueprint: tools given to the agent.
These are thin, deterministic wrappers around the Level 1-3 Python
code. The LLM never computes statistics itself — it calls these tools
and reasons about the structured results they return.
"""

import sys
sys.path.append("src/data")
sys.path.append("src/ml")

from loader import load_dataset
from validator import analyze_dataset

DATA_PATH = "data/telco_churn.csv"


def analyze_dataset_tool() -> dict:
    """
    Loads the churn dataset and returns a structured analysis: row/column
    counts, missing values, duplicates, feature types, and class balance.
    """
    df = load_dataset(DATA_PATH)
    return analyze_dataset(df)
