"""
supervisor.py
Level 4: LLM + Tool Calling, using Google's Gemini API.

Uses the google-genai SDK's automatic function calling: plain Python
functions are passed as tools, and the SDK handles the full loop of
calling Gemini, executing any tool calls Gemini requests, feeding the
results back, and repeating until Gemini gives a final text answer.
Python still does every calculation -- Gemini only decides which tool
to call and interprets structured results.

Safety note (blueprint Section 21): reading dataset/history is LOW
RISK and auto-executed. Actually training a new model (run_experiment)
is MEDIUM RISK -- see agent_tools.run_experiment, which pauses for a
human y/n before it trains anything.

Requires a GEMINI_API_KEY environment variable.
"""

import os
import sys

sys.path.append("src/agents")

from google import genai
from google.genai import types

from agent_tools import ALL_TOOLS

MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = (
    "You are the supervisor agent for an autonomous ML research project. "
    "The dataset is a customer churn binary classification problem. "
    "You have tools to inspect the dataset, review experiment history, "
    "check whether a configuration has already been tried, and run new "
    "experiments. Always check history before proposing a new experiment "
    "-- don't repeat configurations that have already been tried. Keep "
    "your reasoning concise. When you're done, give the user a clear "
    "final answer summarizing what you found or did."
)


def _print_call_history(response) -> None:
    """Prints each tool call the SDK made automatically, for visibility."""
    history = getattr(response, "automatic_function_calling_history", None)
    if not history:
        return
    for content in history:
        if not getattr(content, "parts", None):
            continue
        for part in content.parts:
            if getattr(part, "function_call", None):
                fc = part.function_call
                print(f"\n[Tool call] {fc.name}({dict(fc.args) if fc.args else {}})")
            elif getattr(part, "function_response", None):
                fr = part.function_response
                print(f"[Tool result] {str(fr.response)[:500]}")


def run_agent(goal: str) -> str:
    """
    Sends the goal to Gemini with tools attached. The SDK automatically
    calls whichever tools Gemini requests and loops until Gemini gives
    a final text answer.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a key from aistudio.google.com/apikey and set it before running the agent."
        )

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=goal,
        config=config,
    )

    _print_call_history(response)

    return response.text


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Analyze the dataset, check what's already been tried, and tell me "
        "what the best experiment so far is."
    )
    print(f"Goal: {goal}\n")
    answer = run_agent(goal)
    print("\n" + "=" * 55)
    print("FINAL ANSWER")
    print("=" * 55)
    print(answer)
