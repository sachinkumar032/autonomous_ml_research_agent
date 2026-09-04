"""
Tests for Level 5: error analysis and the LangGraph workflow's
decision logic. Graph execution itself (training real models across
multiple nodes) is slower, so these focus on correctness of the
pieces that matter most: error pattern detection and the improve/stop
decision rule.
"""

import sys
sys.path.append("src/data")
sys.path.append("src/ml")
sys.path.append("src/agents")

from loader import load_dataset
from preprocessing import get_train_test
from training import get_models, train_model
from error_analysis import analyze_errors

DATA_PATH = "data/telco_churn.csv"


def test_analyze_errors_returns_expected_keys():
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = get_train_test(df)
    estimator = get_models()["Logistic Regression"]
    pipe = train_model("LogReg", estimator, preprocessor, X_train, y_train)

    report = analyze_errors(pipe, X_test, y_test, numeric_cols=["tenure"])

    assert "true_positives" in report
    assert "false_negatives" in report
    assert report["total_test_rows"] == len(X_test)
    # confusion matrix components should sum to the total
    total_check = (report["true_positives"] + report["true_negatives"]
                   + report["false_positives"] + report["false_negatives"])
    assert total_check == report["total_test_rows"]


def test_analyze_errors_detects_tenure_pattern():
    """The blueprint's known finding: false negatives skew toward lower tenure."""
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = get_train_test(df)
    estimator = get_models()["Logistic Regression"]
    pipe = train_model("LogReg", estimator, preprocessor, X_train, y_train)

    report = analyze_errors(pipe, X_test, y_test, numeric_cols=["tenure"])
    assert len(report["patterns_found"]) >= 1
    assert "tenure" in report["patterns_found"][0]


def test_decide_node_stops_when_target_met():
    from workflow import decide_node

    state = {
        "current_log": {"metrics": {"f1": 0.7}},
        "target_f1": 0.6,
        "iteration": 0,
        "max_iterations": 5,
        "tried_configs": ["LogReg_C0.1"],
    }
    result = decide_node(state)
    assert result["decision"] == "stop"


def test_decide_node_improves_when_below_target_and_configs_remain():
    from workflow import decide_node

    state = {
        "current_log": {"metrics": {"f1": 0.4}},
        "target_f1": 0.6,
        "iteration": 0,
        "max_iterations": 5,
        "tried_configs": [],
    }
    result = decide_node(state)
    assert result["decision"] == "improve"


def test_decide_node_stops_at_max_iterations():
    from workflow import decide_node

    state = {
        "current_log": {"metrics": {"f1": 0.4}},
        "target_f1": 0.9,
        "iteration": 5,
        "max_iterations": 5,
        "tried_configs": [],
    }
    result = decide_node(state)
    assert result["decision"] == "stop"


def test_decide_node_stops_when_no_configs_remain():
    from workflow import decide_node
    from model_configs import get_experiment_configs

    all_names = [c["name"] for c in get_experiment_configs()]
    state = {
        "current_log": {"metrics": {"f1": 0.1}},
        "target_f1": 0.9,
        "iteration": 0,
        "max_iterations": 100,
        "tried_configs": all_names,
    }
    result = decide_node(state)
    assert result["decision"] == "stop"
