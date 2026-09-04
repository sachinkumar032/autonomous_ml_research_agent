"""
error_analysis.py
Component 6 of the blueprint: Error Analysis Agent.

Instead of stopping at a single F1 score, this inspects which rows
the model got wrong and looks for patterns -- which feature ranges
are over-represented in false negatives/positives compared to their
share of the overall test set. This is what feeds the "Need
Improvement?" decision in the Level 5 conditional loop.
"""

import pandas as pd


def analyze_errors(pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                    numeric_cols: list[str] = None, top_n_patterns: int = 3) -> dict:
    """
    Runs the model on X_test, splits predictions into TP/TN/FP/FN, and
    looks for numeric feature ranges that are over-represented among
    false negatives (missed churners) compared to the overall test set.
    """
    y_pred = pipeline.predict(X_test)

    results = X_test.copy()
    results["actual"] = y_test.values
    results["predicted"] = y_pred
    results["outcome"] = "TN"
    results.loc[(results["actual"] == 1) & (results["predicted"] == 1), "outcome"] = "TP"
    results.loc[(results["actual"] == 0) & (results["predicted"] == 1), "outcome"] = "FP"
    results.loc[(results["actual"] == 1) & (results["predicted"] == 0), "outcome"] = "FN"

    counts = results["outcome"].value_counts().to_dict()
    total = len(results)

    fn_rows = results[results["outcome"] == "FN"]
    fp_rows = results[results["outcome"] == "FP"]

    if numeric_cols is None:
        numeric_cols = X_test.select_dtypes(include=["int64", "float64"]).columns.tolist()

    patterns = []
    for col in numeric_cols[:top_n_patterns]:
        overall_mean = results[col].mean()
        fn_mean = fn_rows[col].mean() if len(fn_rows) else None
        if fn_mean is not None and overall_mean:
            pct_diff = round(((fn_mean - overall_mean) / overall_mean) * 100, 1)
            if abs(pct_diff) > 10:  # only report meaningfully different patterns
                direction = "lower" if pct_diff < 0 else "higher"
                patterns.append(
                    f"False negatives have {direction} average {col} "
                    f"({round(fn_mean, 1)} vs overall {round(overall_mean, 1)}, {pct_diff:+.1f}%)"
                )

    return {
        "total_test_rows": total,
        "true_positives": int(counts.get("TP", 0)),
        "true_negatives": int(counts.get("TN", 0)),
        "false_positives": int(counts.get("FP", 0)),
        "false_negatives": int(counts.get("FN", 0)),
        "false_negative_rate": round(counts.get("FN", 0) / total, 3) if total else 0,
        "patterns_found": patterns,
    }


def print_error_analysis(report: dict) -> None:
    print("Error Analysis")
    print(f"  TP={report['true_positives']}  TN={report['true_negatives']}  "
          f"FP={report['false_positives']}  FN={report['false_negatives']}")
    if report["patterns_found"]:
        print("  Patterns in missed churners:")
        for p in report["patterns_found"]:
            print(f"    - {p}")
    else:
        print("  No strong feature patterns found in false negatives.")
