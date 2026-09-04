"""
run_workflow.py
Level 5 entry point: runs the LangGraph supervisor with its
conditional improvement loop.

Usage:
    python3 run_workflow.py
    python3 run_workflow.py --target-f1 0.65 --max-iterations 5
"""

import sys
import argparse

sys.path.append("src/agents")
from workflow import run_workflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-f1", type=float, default=0.62,
                         help="Stop improving once F1 reaches this value.")
    parser.add_argument("--max-iterations", type=int, default=4,
                         help="Max number of improvement loops before stopping.")
    args = parser.parse_args()

    print("=" * 55)
    print("LEVEL 5: LANGGRAPH SUPERVISOR + CONDITIONAL LOOP")
    print(f"Target F1: {args.target_f1} | Max iterations: {args.max_iterations}")
    print("=" * 55)

    final_state = run_workflow(target_f1=args.target_f1, max_iterations=args.max_iterations)

    print("=" * 55)
    print("WORKFLOW COMPLETE")
    print("=" * 55)
