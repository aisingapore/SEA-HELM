import copy

from src.base_logger import get_logger

logger = get_logger(__name__)

PRAGMATICS_MIN_SCORE = 0.5


def aggregate_pragmatics_metrics(metrics: dict, lang="id") -> None:
    logger.info("---------- Task: PRAGMATICS (%s) ----------", lang.upper())
    PRAGMATICS_PHENOMENA = ["scalar_implicatures", "presuppositions"]

    # check if running logprobs
    if "logprobs" in list(metrics[lang]["linguistic-diagnostics"].keys())[0]:
        logger.info("Running aggregation for pragmatics logprobs tasks")
        PRAGMATICS_TASKS = [
            "pragmatic-single-logprobs",
            "pragmatic-pair-logprobs",
        ]
        metrics[lang]["linguistic-diagnostics"]["pragmatics-logprobs"] = {}
        subcategories = {}
        for phenomenon in PRAGMATICS_PHENOMENA:
            subcategories_probabilities = []
            for task in PRAGMATICS_TASKS:
                if (
                    phenomenon
                    in metrics[lang]["linguistic-diagnostics"][task]["subcategories"]
                ):
                    subcategories_probabilities.append(
                        metrics[lang]["linguistic-diagnostics"][task]["subcategories"][
                            phenomenon
                        ]["average_cumulative_probabilities"]
                    )
            subcategories[phenomenon] = {
                "average_cumulative_probabilities": sum(subcategories_probabilities)
                / len(subcategories_probabilities),
            }
            logger.info(
                "Average cumulative probabilities for phenomenon <%s_%s>: %f",
                lang.upper(),
                phenomenon,
                subcategories[phenomenon]["average_cumulative_probabilities"],
            )
        metrics[lang]["linguistic-diagnostics"]["pragmatics-logprobs"][
            "subcategories"
        ] = subcategories

        pragmatics_subset_scores = [
            metrics[lang]["linguistic-diagnostics"]["pragmatics-logprobs"][
                "subcategories"
            ][phenomenon]["average_cumulative_probabilities"]
            for phenomenon in PRAGMATICS_PHENOMENA
        ]
        overall_accuracy = sum(pragmatics_subset_scores) / len(pragmatics_subset_scores)
        metrics[lang]["linguistic-diagnostics"]["pragmatics-logprobs"][
            "average_cumulative_probabilities"
        ] = overall_accuracy
        logger.info(
            "Overall average cumulative probabilities for <%s_linguistic_diagnostics_pragmatics>: %f",
            lang,
            overall_accuracy,
        )
    else:
        PRAGMATICS_TASKS = [
            "pragmatic-single",
            "pragmatic-pair",
        ]

        metrics[lang]["linguistic-diagnostics"]["pragmatics"] = {}

        subcategories = {}
        for phenomenon in PRAGMATICS_PHENOMENA:
            correct_count, total_count = 0, 0

            for task in PRAGMATICS_TASKS:
                if (
                    phenomenon
                    in metrics[lang]["linguistic-diagnostics"][task]["subcategories"]
                ):
                    correct_count += metrics[lang]["linguistic-diagnostics"][task][
                        "subcategories"
                    ][phenomenon][0]
                    total_count += metrics[lang]["linguistic-diagnostics"][task][
                        "subcategories"
                    ][phenomenon][1]
            subset_accuracy = correct_count / total_count

            subcategories[phenomenon] = {
                "accuracy": subset_accuracy,
                "correct_count": correct_count,
                "total_count": total_count,
            }

            logger.info(
                "Accuracy for phenomenon <%s_%s>: %d / %d : %f",
                lang.upper(),
                phenomenon,
                correct_count,
                total_count,
                subset_accuracy,
            )

        metrics[lang]["linguistic-diagnostics"]["pragmatics"]["subcategories"] = (
            subcategories
        )

        pragmatics_subset_scores = [
            metrics[lang]["linguistic-diagnostics"]["pragmatics"]["subcategories"][
                phenomenon
            ]["accuracy"]
            for phenomenon in PRAGMATICS_PHENOMENA
        ]
        overall_accuracy = sum(pragmatics_subset_scores) / len(pragmatics_subset_scores)
        metrics[lang]["linguistic-diagnostics"]["pragmatics"]["accuracy"] = (
            overall_accuracy * 100
        )
        logger.info(
            "Overall accuracy for <%s_linguistic_diagnostics_pragmatics>: %f",
            lang,
            overall_accuracy,
        )

        # score normalization for pragmatic reasoning
        min_score = PRAGMATICS_MIN_SCORE
        max_score = 1
        normalized_accuracy = max(
            (overall_accuracy - min_score) / (max_score - min_score), 0
        )
        metrics[lang]["linguistic-diagnostics"]["pragmatics"]["normalized_accuracy"] = (
            normalized_accuracy * 100
        )
        logger.info(
            "Normalized accuracy for <%s_linguistic_diagnostics_pragmatics>: %f",
            lang,
            normalized_accuracy,
        )
    return metrics


def aggregate_lindsea_metrics(metrics: dict, lang="id") -> None:
    LINDSEA_CATEGORIES = ["mp-r", "mp-r-logprobs", "pragmatics", "pragmatics-logprobs"]

    lindsea_scores = []
    metrics[lang]["linguistic-diagnostics"]["lindsea"] = {"subcategories": {}}

    for category in LINDSEA_CATEGORIES:
        try:
            category_metrics = metrics[lang]["linguistic-diagnostics"][category]
            metrics[lang]["linguistic-diagnostics"]["lindsea"]["subcategories"][
                category
            ] = category_metrics

            if "logprob" in category:
                lindsea_scores.append(
                    category_metrics["average_cumulative_probabilities"]
                )
            else:
                lindsea_scores.append(category_metrics["normalized_accuracy"])
        except Exception:
            pass

    overall_accuracy = sum(lindsea_scores) / len(lindsea_scores)
    metrics[lang]["linguistic-diagnostics"]["lindsea"] = {
        "normalized_score": overall_accuracy,
        "subcategories": {},
    }
    logger.info(
        "Overall normalized score for <%s_lindsea>: %f\n", lang, overall_accuracy
    )

    return metrics


def aggregate_metrics(metrics: dict, config) -> None:
    logger.info("---------- Aggregation of metrics ----------")
    total_all_langs = {}
    for lang, competencies in metrics.items():
        logger.info("---------- Aggregation | Lang: %s ----------", lang.upper())
        total_lang = {}
        for competency, tasks in competencies.items():
            logger.info(
                "### Competency: %s",
                competency.upper(),
            )
            # handle special case for linguistic-diagnostics
            if competency == "linguistic-diagnostics":
                metrics = aggregate_pragmatics_metrics(metrics, lang)
                metrics = aggregate_lindsea_metrics(metrics, lang)

                total_lang[competency] = metrics[lang]["linguistic-diagnostics"][
                    "lindsea"
                ]["normalized_score"]
                metrics[lang][competency]["total"] = metrics[lang][
                    "linguistic-diagnostics"
                ]["lindsea"]["normalized_score"]
                continue

            scores = {}
            aggregations = {}

            _tasks = copy.deepcopy(tasks)
            for task, results in _tasks.items():
                metric = config["tasks"][task]["metric"]
                aggregation_group = config["tasks"][task].get("aggregation_group", None)

                if aggregation_group:
                    aggregation_type = config["tasks"][task].get(
                        "aggregation_type", "macro-average"
                    )
                    if aggregation_type == "micro-average":
                        # [IMPORTANT]
                        # For normalized accuracy tasks:
                        # Aggregation uses the normalized balanced accuracy for each subtask for multiple choice tasks
                        # this would result in a slightly different score as compared to calculating the normalized balanced accuracy from all the individual scores directly
                        num_labels = sum(
                            [
                                1 if line != "" else None
                                for line in open(
                                    config["tasks"][task]["languages"][lang]["filepath"]
                                )
                            ]
                        )
                        if aggregation_group not in aggregations:
                            aggregations[aggregation_group] = [
                                results[metric]
                            ] * num_labels
                        else:
                            aggregations[aggregation_group].extend(
                                [results[metric]] * num_labels
                            )

                        _aggregation_total = sum(aggregations[aggregation_group]) / len(
                            aggregations[aggregation_group]
                        )
                    elif aggregation_type == "macro-average":
                        if aggregation_group not in aggregations:
                            aggregations[aggregation_group] = [results[metric]]
                        else:
                            aggregations[aggregation_group].append(results[metric])

                        _aggregation_total = sum(aggregations[aggregation_group]) / len(
                            aggregations[aggregation_group]
                        )

                    scores[aggregation_group] = _aggregation_total
                    metrics[lang][competency][aggregation_group] = _aggregation_total
                else:
                    scores[task] = results[metric]

            competency_total = sum(scores.values()) / len(scores)
            total_lang[competency] = competency_total
            metrics[lang][competency]["total"] = competency_total
            logger.info(
                "Overall normalized score for <%s_%s>: %f\n",
                lang,
                competency,
                competency_total,
            )

        language_total = sum(total_lang.values()) / len(total_lang)
        total_all_langs[lang] = language_total
        metrics[lang]["total"] = language_total

        logger.info("Overall normalized score for <%s>: %f\n", lang, language_total)

    overall_total = sum(total_all_langs.values()) / len(total_all_langs)
    metrics["total"] = overall_total
    logger.info("Overall normalized score: %f\n", overall_total)
    logger.info("Aggregation of all metrics completed\n")

    return metrics
