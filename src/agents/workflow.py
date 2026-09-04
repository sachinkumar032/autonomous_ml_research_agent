"""
workflow.py
Level 5: LangGraph Supervisor with Conditional Improvement Loop.

Implements the diagram from the blueprint:

    START
     v
    Analyze Dataset
     v
    Preprocess + Split
     v
    Train Baseline
     v
    Error Analysis
     v
    Need Improvement? --- NO --> Select Best Model --> Generate Report --> END
     |
    YES
     v
    Create + Train Next Experiment
     v
    (back to Error Analysis)

This is the strongest demonstration of "agentic" behavior in the
project per the blueprint: state, tools, a real decision point, and a
feedback loop -- not just running models in a fixed sequence.
"""

import sys
sys.path.append("src/data")
sys.path.append("src/ml")
sys.path.append("src/agents")

from langgraph.graph import StateGraph, START, END
from sklearn.pipeline import Pipeline

from loader import load_dataset
from validator import analyze_dataset
from preprocessing import get_train_test, DATASET_VERSION
from model_configs import get_experiment_configs
from evaluation import evaluate_model
from error_analysis import analyze_errors
from experiment_manager import run_experiment, leaderboard
from experiment_tracker import already_tried

from graph_state import AgentState

DATA_PATH = "data/telco_churn.csv"
ERROR_ANALYSIS_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _train_config(config: dict, state: AgentState) -> dict:
    """
    Fits the given config's pipeline. Logs it as a new experiment only
    if it hasn't been tried before at this dataset version -- otherwise
    it still fits (so error analysis has a pipeline to inspect) but
    reuses the existing log instead of creating a duplicate file.
    """
    match = already_tried(config["model_family"], config["params"], state["dataset_version"])
    if match is not None:
        pipeline = Pipeline(steps=[
            ("preprocessor", state["preprocessor"]),
            ("model", config["estimator"]),
        ])
        pipeline.fit(state["X_train"], state["y_train"])
        return {"log": match, "pipeline": pipeline}

    result = run_experiment(
        config, state["preprocessor"], state["X_train"], state["y_train"],
        state["X_test"], state["y_test"], evaluate_model,
        dataset_version=state["dataset_version"],
    )
    return result


# --- Nodes ---

def analyze_dataset_node(state: AgentState) -> dict:
    print("\n[Node] Analyze Dataset")
    df = load_dataset(DATA_PATH)
    report = analyze_dataset(df)
    print(f"  Rows: {report['rows']}, Missing: {report['missing_values_total']}, "
          f"Class balance: {report.get('class_distribution')}")
    return {"dataset_report": report}


def preprocess_node(state: AgentState) -> dict:
    print("\n[Node] Preprocess + Split")
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = get_train_test(df)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "preprocessor": preprocessor,
        "dataset_version": DATASET_VERSION,
        "iteration": 0,
        "tried_configs": [],
    }


def train_baseline_node(state: AgentState) -> dict:
    print("\n[Node] Train Baseline")
    baseline_config = get_experiment_configs()[0]  # first config: LogReg_C0.1
    result = _train_config(baseline_config, state)
    f1 = result["log"]["metrics"]["f1"]
    print(f"  {result['log']['name']} -> F1 = {f1:.3f}")
    return {
        "current_pipeline": result["pipeline"],
        "current_log": result["log"],
        "tried_configs": state["tried_configs"] + [baseline_config["name"]],
    }


def error_analysis_node(state: AgentState) -> dict:
    print("\n[Node] Error Analysis")
    report = analyze_errors(
        state["current_pipeline"], state["X_test"], state["y_test"],
        numeric_cols=ERROR_ANALYSIS_COLS,
    )
    print(f"  FN rate: {report['false_negative_rate']}, "
          f"Patterns: {report['patterns_found'] or 'none found'}")
    return {"current_error_report": report}


def decide_node(state: AgentState) -> dict:
    print("\n[Node] Need Improvement?")
    current_f1 = state["current_log"]["metrics"]["f1"]
    target = state["target_f1"]
    iteration = state["iteration"]
    max_iter = state["max_iterations"]

    untried = [
        c for c in get_experiment_configs()
        if c["name"] not in state["tried_configs"]
    ]

    if current_f1 >= target:
        print(f"  F1 {current_f1:.3f} >= target {target}. Stopping.")
        decision = "stop"
    elif iteration >= max_iter:
        print(f"  Reached max iterations ({max_iter}). Stopping.")
        decision = "stop"
    elif not untried:
        print("  No untried configs left. Stopping.")
        decision = "stop"
    else:
        print(f"  F1 {current_f1:.3f} < target {target}, {len(untried)} configs left. Improving.")
        decision = "improve"

    return {"decision": decision, "iteration": iteration + 1}


def route_after_decision(state: AgentState) -> str:
    return state["decision"]


def create_next_experiment_node(state: AgentState) -> dict:
    print("\n[Node] Create + Train Next Experiment")
    untried = [
        c for c in get_experiment_configs()
        if c["name"] not in state["tried_configs"]
    ]
    next_config = untried[0]
    print(f"  Trying: {next_config['name']} ({next_config['model_family']})")
    result = _train_config(next_config, state)
    f1 = result["log"]["metrics"]["f1"]
    print(f"  -> F1 = {f1:.3f}")
    return {
        "current_pipeline": result["pipeline"],
        "current_log": result["log"],
        "tried_configs": state["tried_configs"] + [next_config["name"]],
    }


def select_best_model_node(state: AgentState) -> dict:
    print("\n[Node] Select Best Model")
    board = leaderboard(top_n=1)
    best = board.iloc[0].to_dict()
    print(f"  Best overall: {best['name']} ({best['model_family']}), F1={best['f1']:.3f}")
    return {"best_experiment": best}


def generate_report_node(state: AgentState) -> dict:
    print("\n[Node] Generate Report")
    best = state["best_experiment"]
    report = (
        f"AUTONOMOUS EXPERIMENTATION REPORT\n"
        f"{'=' * 40}\n"
        f"Dataset: {state['dataset_report']['rows']} rows, "
        f"target={state['dataset_report']['target']}\n"
        f"Configs tried this run: {len(state['tried_configs'])} "
        f"({', '.join(state['tried_configs'])})\n"
        f"Best experiment overall: {best['name']} ({best['model_family']})\n"
        f"Best F1: {best['f1']:.3f} | Accuracy: {best['accuracy']:.3f} | "
        f"ROC-AUC: {best['roc_auc']:.3f}\n"
        f"Latest error analysis: {state['current_error_report']['patterns_found'] or 'no strong patterns'}\n"
    )
    print(report)
    return {"report_text": report}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze_dataset", analyze_dataset_node)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("train_baseline", train_baseline_node)
    graph.add_node("error_analysis", error_analysis_node)
    graph.add_node("decide", decide_node)
    graph.add_node("create_next_experiment", create_next_experiment_node)
    graph.add_node("select_best_model", select_best_model_node)
    graph.add_node("generate_report", generate_report_node)

    graph.add_edge(START, "analyze_dataset")
    graph.add_edge("analyze_dataset", "preprocess")
    graph.add_edge("preprocess", "train_baseline")
    graph.add_edge("train_baseline", "error_analysis")
    graph.add_edge("error_analysis", "decide")

    graph.add_conditional_edges(
        "decide", route_after_decision,
        {"improve": "create_next_experiment", "stop": "select_best_model"},
    )

    graph.add_edge("create_next_experiment", "error_analysis")  # the loop
    graph.add_edge("select_best_model", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


def run_workflow(target_f1: float = 0.62, max_iterations: int = 4) -> AgentState:
    app = build_graph()
    initial_state = {"target_f1": target_f1, "max_iterations": max_iterations}
    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    final_state = run_workflow()
