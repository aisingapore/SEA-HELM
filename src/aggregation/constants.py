# A multi-choice task whose rarest class has fewer than this many questions gets its
# per-class support recorded in the score YAML: balanced accuracy is a macro average over
# per-class recalls, so a class with a handful of questions dominates the uncertainty no
# matter how the bootstrap is drawn.
LOW_CLASS_SUPPORT_THRESHOLD = 10

# Languages counted towards the SEA subset average.
SEA_LANGUAGES: list[str] = ["id", "vi", "th", "ta", "tl", "ms", "my"]

# Native maximum of each per-example metric key.
METRIC_NATIVE_MAX: dict[str, float] = {
    # stored as a fraction in [0, 1]
    "accuracy_score": 1.0,
    "average_criteria_score": 1.0,
    "criteria_score": 1.0,
    "generic_score": 1.0,
    "normalized_accuracy": 1.0,
    "normalized_all_stages_pass_rate": 1.0,
    "normalized_average_category_score": 1.0,
    "normalized_exact_match": 1.0,
    "normalized_f1": 1.0,
    "normalized_math_verify_score": 1.0,
    "normalized_partial_accuracy_score": 1.0,
    "normalized_pass@1": 1.0,
    "overall_lang_normalized_acc": 1.0,
    "overall_score": 1.0,
    "pass@1": 1.0,
    "prompt_level_strict_acc": 1.0,
    "weighted_win_rate": 1.0,
    # already stored as a percentage in [0, 100]
    "normalized_metricx_wmt24_scores": 100.0,  # nlg/translation/translation_metric.py
    "normalized_rougel_f1": 100.0,  # nlg/abstractive_summarization/summarization.py
    "average_cumulative_probabilities": 100.0,  # src/metrics/logprob_metric.py
}
