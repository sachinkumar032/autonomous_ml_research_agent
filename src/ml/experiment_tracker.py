"""
experiment_tracker.py
Level 3: Experiment Tracking.

This module gives the system memory of what it has already tried,
so Level 5 (the agentic loop) can eventually ask "what have I already
tried?" and avoid repeating weak approaches — exactly as described in
the blueprint's Memory section.

Two things this adds on top of Level 2's plain logging:
1. A stable signature per (model_family, params, dataset_version) so
   identical experiments can be detected and skipped.
2. A structured summary of history that reads naturally — this is the
   shape an LLM agent will eventually be given as context.
"""

import hashlib
import json
from pathlib import Path

from experiment_manager import load_all_experiment_logs, EXPERIMENTS_DIR


def config_signature(model_family: str, params: dict, dataset_version: str) -> str:
    """
    Deterministic hash identifying a specific (model + hyperparameters +
    dataset/feature version) combination. Same inputs -> same signature,
    regardless of dict key order.
    """
    payload = json.dumps(
        {"model_family": model_family, "params": params, "dataset_version": dataset_version},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def already_tried(model_family: str, params: dict, dataset_version: str,
                   logs: list[dict] = None) -> dict | None:
    """
    Returns the matching past experiment log if this exact config has
    already been run at this dataset version, otherwise None.
    """
    if logs is None:
        logs = load_all_experiment_logs()

    target_sig = config_signature(model_family, params, dataset_version)
    for log in logs:
        existing_sig = log.get("signature")
        if existing_sig is None:
            # Backward compatibility with Level 2 logs that predate signatures
            existing_sig = config_signature(
                log.get("model_family", ""), log.get("params", {}),
                log.get("dataset_version", "v1"),
            )
        if existing_sig == target_sig:
            return log
    return None


def summarize_history(logs: list[dict] = None) -> dict:
    """
    Produces a compact, structured summary of everything tried so far.
    This is the shape a future LLM agent will read to answer
    "what have I already tried, and what worked?"
    """
    if logs is None:
        logs = load_all_experiment_logs()

    if not logs:
        return {"total_experiments": 0, "by_model_family": {}, "best": None}

    by_family = {}
    for log in logs:
        fam = log.get("model_family", "unknown")
        f1 = log.get("metrics", {}).get("f1")
        if fam not in by_family:
            by_family[fam] = {"count": 0, "best_f1": None, "worst_f1": None}
        by_family[fam]["count"] += 1
        if f1 is not None:
            if by_family[fam]["best_f1"] is None or f1 > by_family[fam]["best_f1"]:
                by_family[fam]["best_f1"] = round(f1, 4)
            if by_family[fam]["worst_f1"] is None or f1 < by_family[fam]["worst_f1"]:
                by_family[fam]["worst_f1"] = round(f1, 4)

    best_log = max(logs, key=lambda l: l.get("metrics", {}).get("f1", -1))

    return {
        "total_experiments": len(logs),
        "by_model_family": by_family,
        "best": {
            "experiment_id": best_log.get("experiment_id"),
            "name": best_log.get("name"),
            "model_family": best_log.get("model_family"),
            "f1": best_log.get("metrics", {}).get("f1"),
        },
    }


def print_history_summary(logs: list[dict] = None) -> None:
    summary = summarize_history(logs)
    print(f"Total experiments logged: {summary['total_experiments']}")
    for fam, stats in summary["by_model_family"].items():
        print(f"  {fam}: {stats['count']} runs, best F1 = {stats['best_f1']}, worst F1 = {stats['worst_f1']}")
    if summary["best"]:
        b = summary["best"]
        print(f"Best overall: #{b['experiment_id']} {b['name']} ({b['model_family']}), F1 = {b['f1']:.3f}" if b['f1'] else "")
