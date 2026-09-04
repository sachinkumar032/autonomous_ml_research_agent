"""
agent_tools.py
Tool functions exposed to the Gemini agent. The google-genai SDK reads
each function's type hints and docstring (Google style, with an Args:
section) to automatically build the tool schema the model sees -- no
manual JSON schema needed, unlike the Claude/Anthropic tool-use format.

Read-only tools (analyze_dataset, get_leaderboard, get_history_summary,
check_already_tried) are safe to auto-call. run_experiment is gated
with a human-approval prompt before it actually trains anything, per
the project's autonomy levels (blueprint Section 21).
"""

import sys
import json

sys.path.append("src/data")
sys.path.append("src/ml")
sys.path.append("src/tools")

from data_tools import analyze_dataset_tool
from ml_tools import (
    get_leaderboard_tool,
    get_history_summary_tool,
    check_already_tried_tool,
    run_experiment_tool as _run_experiment_tool,
)


def analyze_dataset() -> dict:
    """Load and analyze the churn dataset: row/column counts, missing
    values, duplicates, numerical/categorical feature counts, and class
    balance. Read-only, safe to call anytime.
    """
    return analyze_dataset_tool()


def get_leaderboard(top_n: int = 5) -> dict:
    """Get the top experiments ever logged, ranked by F1 score.
    Read-only, safe to call anytime.

    Args:
        top_n: How many top experiments to return.
    """
    return get_leaderboard_tool(top_n=top_n)


def get_history_summary() -> dict:
    """Get a summary of ALL experiments run so far, grouped by model
    family, with best/worst F1 per family and the overall best
    experiment. Use this to understand what has already been tried
    before proposing new experiments. Read-only, safe to call anytime.
    """
    return get_history_summary_tool()


def check_already_tried(model_family: str, params: dict) -> dict:
    """Check whether a specific model configuration has already been
    run. ALWAYS call this before proposing to run a new experiment, so
    you don't waste a training run repeating something already tried.
    Read-only, safe to call anytime.

    Args:
        model_family: One of "Logistic Regression", "Random Forest", "XGBoost".
        params: Hyperparameters to check, e.g. {"max_depth": 5, "learning_rate": 0.05}.
    """
    return check_already_tried_tool(model_family, params)


def run_experiment(model_family: str, params: dict) -> dict:
    """Train and evaluate ONE new model configuration and log it as an
    experiment. This performs REAL training and takes real compute time.
    Only call this for a genuinely new configuration you've already
    confirmed with check_already_tried. This is a MEDIUM RISK action
    per the project's safety policy -- the human user will be asked to
    approve it before it actually executes.

    Args:
        model_family: One of "Logistic Regression", "Random Forest", "XGBoost".
        params: Hyperparameters for the chosen model family.
    """
    print(f"\n[APPROVAL NEEDED] The agent wants to train: {model_family} with {json.dumps(params)}")
    answer = input("Approve this action? (y/n): ").strip().lower()
    if answer != "y":
        return {"error": "Human did not approve this action. Do not retry it -- choose a different next step."}
    return _run_experiment_tool(model_family, params)


ALL_TOOLS = [analyze_dataset, get_leaderboard, get_history_summary, check_already_tried, run_experiment]
