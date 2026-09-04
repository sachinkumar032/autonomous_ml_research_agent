"""
knowledge_base.py
Component of the blueprint's Section 17 (RAG): a focused knowledge
layer for ML guidance -- not general web search, just curated,
short, actionable notes covering the situations this project
actually runs into. Kept as plain Python data (no external docs
scraping) so it works offline with zero setup.
"""

GUIDANCE_DOCS = [
    {
        "id": "imbalance_class_weight",
        "title": "Handling class imbalance: class weighting",
        "text": (
            "When the positive class is a minority (like churn at ~27%), a "
            "model can get decent accuracy by mostly predicting the majority "
            "class. Setting class_weight='balanced' in Logistic Regression or "
            "Random Forest re-weights the loss so minority-class errors cost "
            "more, which typically raises recall at some cost to precision. "
            "For XGBoost, use scale_pos_weight = (negative_count / positive_count)."
        ),
    },
    {
        "id": "imbalance_resampling",
        "title": "Handling class imbalance: resampling",
        "text": (
            "Oversampling the minority class (e.g. SMOTE) or undersampling the "
            "majority class can help when class weighting alone isn't enough. "
            "Resampling should only be applied to the training set, never the "
            "test set, or evaluation metrics become misleadingly optimistic."
        ),
    },
    {
        "id": "metric_choice",
        "title": "Choosing the right evaluation metric",
        "text": (
            "Accuracy is misleading on imbalanced data. For churn prediction, "
            "F1 balances precision and recall and is a reasonable default. If "
            "missing churners (false negatives) is costlier than false alarms, "
            "prioritize recall; if false alarms are costly (e.g. wasted retention "
            "offers), prioritize precision. ROC-AUC is useful for comparing "
            "models' ranking ability independent of a chosen threshold."
        ),
    },
    {
        "id": "logreg_regularization",
        "title": "Logistic Regression: tuning C",
        "text": (
            "C is the inverse of regularization strength -- lower C means "
            "stronger regularization (simpler model, less overfitting risk), "
            "higher C means the model fits the training data more closely. "
            "If train and test scores are close but both low, the model is "
            "likely underfitting -- try increasing C or adding features rather "
            "than decreasing C further."
        ),
    },
    {
        "id": "rf_tuning",
        "title": "Random Forest: key hyperparameters",
        "text": (
            "max_depth controls overfitting -- unbounded depth (None) lets "
            "trees memorize training data. n_estimators generally helps up to "
            "a point of diminishing returns; more trees rarely hurts but costs "
            "compute. If Random Forest underperforms Logistic Regression on a "
            "small/simple dataset, it's often because the decision boundary is "
            "actually close to linear and doesn't need a complex tree ensemble."
        ),
    },
    {
        "id": "xgb_tuning",
        "title": "XGBoost: learning_rate vs max_depth trade-off",
        "text": (
            "Lower learning_rate with more n_estimators generally generalizes "
            "better than high learning_rate with few trees, but takes longer "
            "to train. max_depth beyond 6-8 rarely helps on small tabular "
            "datasets and increases overfitting risk. If XGBoost underperforms "
            "simpler models, try reducing max_depth and learning_rate together "
            "before adding more trees."
        ),
    },
    {
        "id": "error_analysis_next_steps",
        "title": "Acting on error analysis findings",
        "text": (
            "If false negatives cluster around low tenure, the model may lack "
            "signal for new customers -- consider a derived feature like "
            "'tenure_bucket' or interactions between tenure and contract type. "
            "If errors cluster around a specific categorical value (e.g. "
            "month-to-month contracts), check whether that category is "
            "underrepresented in training data relative to its churn rate."
        ),
    },
    {
        "id": "overfitting_signs",
        "title": "Recognizing overfitting vs underfitting",
        "text": (
            "If a model's F1 on training data is much higher than on test "
            "data, it's overfitting -- reduce model complexity (lower "
            "max_depth, higher regularization via lower C, fewer estimators) "
            "or gather more data. If both train and test F1 are low and "
            "similar, it's underfitting -- increase model complexity or add "
            "more informative features instead."
        ),
    },
    {
        "id": "when_to_stop_experimenting",
        "title": "Deciding when to stop experimenting",
        "text": (
            "Diminishing returns are a signal to stop: if the last 2-3 "
            "experiments improved F1 by less than 0.005-0.01, further "
            "hyperparameter tweaks within the same model family are unlikely "
            "to help much. At that point, a different model family, new "
            "features, or addressing class imbalance directly is more likely "
            "to move the needle than more tuning."
        ),
    },
]
