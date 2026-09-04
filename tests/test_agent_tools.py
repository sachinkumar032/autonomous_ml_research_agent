"""
Tests for the Level 4 agent tools. These test the deterministic Python
functions the LLM calls -- no API key or network access needed, since
the LLM itself isn't being tested here, just the tools it would use.
"""

import sys
sys.path.append("src/tools")
sys.path.append("src/data")
sys.path.append("src/ml")

from data_tools import analyze_dataset_tool
from ml_tools import (
    get_leaderboard_tool,
    get_history_summary_tool,
    check_already_tried_tool,
)


def test_analyze_dataset_tool_returns_expected_shape():
    result = analyze_dataset_tool()
    assert result["rows"] > 7000
    assert result["target"] == "Churn"
    assert "class_distribution" in result


def test_get_leaderboard_tool_returns_ranked_experiments():
    result = get_leaderboard_tool(top_n=3)
    assert "experiments" in result
    if result["experiments"]:
        f1_scores = [e["f1"] for e in result["experiments"]]
        assert f1_scores == sorted(f1_scores, reverse=True)


def test_get_history_summary_tool_has_expected_keys():
    result = get_history_summary_tool()
    assert "total_experiments" in result
    assert "by_model_family" in result
    assert "best" in result


def test_check_already_tried_tool_detects_new_config():
    result = check_already_tried_tool("Random Forest", {"n_estimators": 12345, "max_depth": 99})
    assert result["already_tried"] is False
