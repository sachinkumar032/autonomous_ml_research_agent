"""
run_agent.py
Level 4 entry point.

Requires an ANTHROPIC_API_KEY environment variable. Get one at
console.anthropic.com, then set it before running, e.g.:

    export ANTHROPIC_API_KEY=sk-ant-...      (Mac/Linux)
    setx ANTHROPIC_API_KEY "sk-ant-..."       (Windows, new terminal after)

Usage:
    python3 run_agent.py "What's the best experiment so far?"
    python3 run_agent.py "Try a Random Forest with max_depth 8 if it's new"
"""

import sys
sys.path.append("src/agents")

from supervisor import run_agent

if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Analyze the dataset, check the experiment history, and tell me "
        "what the best experiment so far is and why."
    )
    print(f"Goal: {goal}\n")
    answer = run_agent(goal)
    print("\n" + "=" * 55)
    print("FINAL ANSWER")
    print("=" * 55)
    print(answer)
