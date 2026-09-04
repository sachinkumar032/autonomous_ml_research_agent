"""
graph_state.py
The shared state object that flows through every node in the Level 5
LangGraph workflow. Each node reads what it needs and returns a dict
of updates -- LangGraph merges these into the running state.
"""

from typing import TypedDict, Optional, Any


class AgentState(TypedDict, total=False):
    # Data
    dataset_report: dict
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    preprocessor: Any
    dataset_version: str

    # Current model under evaluation
    current_pipeline: Any
    current_log: dict
    current_error_report: dict

    # Loop control
    iteration: int
    max_iterations: int
    target_f1: float
    tried_configs: list  # names of configs already attempted this run
    decision: str  # "improve" | "stop"

    # Final outcome
    best_experiment: dict
    report_text: str
