import math
import statistics

import pandas as pd
from sklearn.metrics import accuracy_score

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class MinimalPairsMetric(SeaHelmMetric):
    """Accuracy-based metric for minimal-pair evaluations.

    This metric extracts a normalized label from each model response, maps it to
    the closest valid label, and computes:

    - Overall accuracy
    - Accuracy per linguistic phenomenon (subcategory)
    - Normalized accuracy (relative to chance)

    Args:
        dataloader (AbstractDataloader): Dataloader object containing inference results.
        task_config (TaskConfig): The task configuration.
        null_label (str, optional): Fallback label when extraction fails. Defaults to "null".
    """

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
        null_label: str = "null",
    ) -> None:
        """Initialize `MinimalPairsMetric`.

        Args:
            dataloader (AbstractDataloader): Dataloader with inference DataFrame.
            task_config (TaskConfig): The task configuration.
            null_label (str, optional): Label to use when mapping fails. Defaults to "null".
        """
        super().__init__(dataloader=dataloader, task_config=task_config)
        self.null_label = null_label

        unique_labels = set(self.dataloader.inference_df[self.label_column].to_list())
        label_string = "|".join(unique_labels)
        self.label_map = {label.lower(): label for label in unique_labels}
        self.regex_string = (
            task_config.config["languages"][self.lang]["prompt_template"]["answer_tag"]
            + rf"[\s\*]*({label_string})*"
        )
        logger.info(
            "Using the following regex to extract the model response: %s",
            self.regex_string,
        )

    def extract_response(self, response: str | None) -> str:
        """Extract and map a model response to a valid label.

        The method uses the base implementation to extract the raw answer from
        the response string, normalizes it, and maps it to the closest matching
        label in a case-insensitive manner. If no match is found, `null_label`
        is returned.

        Args:
            response (str | None): Raw model response.

        Returns:
            str: Mapped label or `null_label` if no valid label is found.
        """
        output = super().extract_response(response)
        output = self.normalize_answer(output)
        output = self.label_map.get(output, self.null_label)
        return output

    def postprocess_responses(self) -> None:
        """Postprocess model responses to prepare for metric computation.

        - Extracts and maps responses into the postprocessed response column.
        - Derives the `linguistic_phenomenon` column from metadata for grouping.
        """
        self.dataloader.inference_df[self.postprocessed_response_column] = (
            self.dataloader.inference_df[self.response_column].map(
                self.extract_response
            )
        )

        # make linguistic_phenomenon a column
        self.dataloader.inference_df["linguistic_phenomenon"] = [
            x["linguistic_phenomenon"] for x in self.dataloader.inference_df["metadata"]
        ]

    def calculate_metrics(self) -> dict:
        """Compute accuracy metrics for minimal pairs.

        Returns:
            dict: Dictionary with overall accuracy (`accuracy`), per-phenomenon
            scores (`subcategories`), and `normalized_accuracy`. Individual
            example scores are stored in the dataloader under
            `inference_df['individual_scores']`.
        """
        metric_dict = {"accuracy": None, "subcategories": {}}

        for phenomenon in self.dataloader.inference_df[
            "linguistic_phenomenon"
        ].unique():
            subset = self.dataloader.inference_df[
                self.dataloader.inference_df["linguistic_phenomenon"] == phenomenon
            ]
            subset_predictions = subset[self.postprocessed_response_column]
            subset_references = subset[self.label_column].apply(
                lambda x: self.label_map.get(x.lower())
            )
            subset_accuracy = 100 * accuracy_score(
                y_true=subset_references, y_pred=subset_predictions
            )
            metric_dict["subcategories"].update({phenomenon: subset_accuracy})
            logger.info(
                "Accuracy for phenomenon <%s>: %.2f", phenomenon, subset_accuracy
            )

        accuracy_list = metric_dict["subcategories"].values()

        overall_accuracy = (
            sum(accuracy_list) / len(accuracy_list) if len(accuracy_list) > 0 else None
        )

        metric_dict["accuracy"] = overall_accuracy
        logger.info("Overall Accuracy: %.2f", overall_accuracy)

        metric_dict["normalized_accuracy"] = (
            self.normalize_score(overall_accuracy, 1 / len(self.label_map) * 100, 100)
            * 100
        )

        predictions = self.dataloader.inference_df[self.postprocessed_response_column]
        references = self.dataloader.inference_df[self.label_column].apply(
            lambda x: self.label_map.get(x.lower())
        )

        individual_scores = predictions.eq(references, axis=0).astype(int)
        self.dataloader.inference_df["individual_scores"] = [
            {"normalized_accuracy": x} for x in individual_scores
        ]
        return metric_dict


class MinimalPairsLogProbMetric(SeaHelmMetric):
    """Probability-based metric for minimal-pair evaluations.

    This metric aggregates cumulative log-probabilities to compute probability
    scores per linguistic phenomenon and overall, and stores a per-example
    probability in `individual_scores`.

    Args:
        dataloader (AbstractDataloader): Dataloader object containing inference results.
        task_config (TaskConfig): The task configuration.
    """

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
    ) -> None:
        """Initialize `MinimalPairsLogProbMetric`.

        Args:
            dataloader (AbstractDataloader): Dataloader with inference DataFrame.
            task_config (TaskConfig): The task configuration.
        """
        super().__init__(dataloader=dataloader, task_config=task_config)

    def postprocess_responses(self) -> None:
        """Derive `linguistic_phenomenon` column from metadata for grouping."""
        self.dataloader.inference_df["linguistic_phenomenon"] = [
            x["linguistic_phenomenon"] for x in self.dataloader.inference_df["metadata"]
        ]

    def calculate_probabilites(self, logprobs: list) -> tuple[dict, list]:
        """Convert log probabilities to cumulative probabilities.

        Expects a sequence where each element corresponds to an example's
        cumulative log-probability container; the first element (index 0)
        is treated as the cumulative log-probability for that example.

        Args:
            logprobs (list): Iterable of elements where `x[0]` is the
                cumulative log-probability; elements may be NaN-like.

        Returns:
            tuple[dict, list]:
                - A metrics dictionary containing averages and standard errors
                  for cumulative log-probabilities and probabilities.
                - A list of per-example probabilities (0–100 scale).
        """
        cumulative_logprobs = [x[0] for x in logprobs]
        null_counts = sum(pd.isna(x) for x in cumulative_logprobs)
        logprobs_wo_nulls = [x for x in cumulative_logprobs if not pd.isna(x)]

        if len(logprobs_wo_nulls) > 1:
            average_cumulative_logprobs = statistics.mean(logprobs_wo_nulls)
            clt_se_cumulative_logprobs = self.calculate_stderr(logprobs_wo_nulls)
        else:
            logger.warning(
                "Num valid cumulative logprobs < 2. Setting average and stderr to 0."
            )
            average_cumulative_logprobs = 0.0
            clt_se_cumulative_logprobs = 0.0

        # replace None with 0 for subset cumulative probabilities
        cumulative_probabilities = [
            0 if pd.isna(x) else math.exp(x) * 100 for x in cumulative_logprobs
        ]

        average_probabilities = statistics.mean(cumulative_probabilities)
        clt_se_cumulative_probabilities = self.calculate_stderr(
            cumulative_probabilities
        )

        return {
            "null_count": null_counts,
            "average_cumulative_logprobs": average_cumulative_logprobs,
            "clt_se_cumulative_logprobs": clt_se_cumulative_logprobs,
            "average_cumulative_probabilities": average_probabilities,
            "clt_se_cumulative_probabilities": clt_se_cumulative_probabilities,
        }, cumulative_probabilities

    def calculate_metrics(self) -> dict:
        """Compute probability-based metrics for minimal pairs.

        Returns:
            dict: Dictionary containing per-phenomenon (`subcategories`) metrics
            and overall summary statistics. Per-example probabilities are stored
            in `inference_df['individual_scores']`.
        """
        metric_dict = {"subcategories": {}}
        cumulative_probabilities = []
        for phenomenon in self.dataloader.inference_df[
            "linguistic_phenomenon"
        ].unique():
            subset = self.dataloader.inference_df[
                self.dataloader.inference_df["linguistic_phenomenon"] == phenomenon
            ]

            subset_metric, _ = self.calculate_probabilites(
                subset["cumulative_logprobs"]
            )
            metric_dict["subcategories"].update({phenomenon: subset_metric})

            logger.info(
                "Accuracy for phenomenon <%s> ± stderr: %.2f ± %.2f",
                phenomenon,
                subset_metric["average_cumulative_probabilities"],
                subset_metric["clt_se_cumulative_probabilities"],
            )

        full_metrics, cumulative_probabilities = self.calculate_probabilites(
            self.dataloader.inference_df["cumulative_logprobs"]
        )
        metric_dict.update(full_metrics)
        logger.info(
            "Average cumulative probabilities ± stderr: %.2f ± %.2f",
            full_metrics["average_cumulative_probabilities"],
            full_metrics["clt_se_cumulative_probabilities"],
        )

        self.dataloader.inference_df["individual_scores"] = [
            {"probabilities_accuracy": x} for x in cumulative_probabilities
        ]

        return metric_dict
