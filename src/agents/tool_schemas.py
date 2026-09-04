from google.genai import types


TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="analyze_dataset",
                description=(
                    "Load and analyze the churn dataset: row/column counts, "
                    "missing values, duplicates, numerical/categorical "
                    "feature counts, and class balance. Read-only, safe to "
                    "call anytime."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={},
                ),
            ),

            types.FunctionDeclaration(
                name="get_leaderboard",
                description=(
                    "Get the top experiments ever logged, ranked by F1 "
                    "score. Read-only, safe to call anytime."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "top_n": types.Schema(
                            type="INTEGER",
                            description=(
                                "How many top experiments to return. "
                                "Default 5."
                            ),
                        ),
                    },
                ),
            ),

            types.FunctionDeclaration(
                name="get_history_summary",
                description=(
                    "Get a summary of ALL experiments run so far, grouped "
                    "by model family, with best/worst F1 per family and "
                    "the overall best experiment. Use this to understand "
                    "what has already been tried before proposing new "
                    "experiments."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={},
                ),
            ),

            types.FunctionDeclaration(
                name="check_already_tried",
                description=(
                    "Check whether a specific model_family and params "
                    "configuration has already been run. ALWAYS call "
                    "this before proposing a new experiment."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "model_family": types.Schema(
                            type="STRING",
                            enum=[
                                "Logistic Regression",
                                "Random Forest",
                                "XGBoost",
                            ],
                        ),
                        "params": types.Schema(
                            type="OBJECT",
                            description=(
                                "Hyperparameters, for example "
                                '{"max_depth": 5, "learning_rate": 0.05}'
                            ),
                        ),
                    },
                    required=["model_family", "params"],
                ),
            ),

            types.FunctionDeclaration(
                name="run_experiment",
                description=(
                    "Train and evaluate ONE new model configuration and "
                    "log it as an experiment. This performs REAL training "
                    "and requires human approval."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "model_family": types.Schema(
                            type="STRING",
                            enum=[
                                "Logistic Regression",
                                "Random Forest",
                                "XGBoost",
                            ],
                        ),
                        "params": types.Schema(
                            type="OBJECT",
                            description="Hyperparameters for the chosen model family.",
                        ),
                    },
                    required=["model_family", "params"],
                ),
            ),
        ]
    )
]