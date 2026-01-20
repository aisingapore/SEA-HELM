import math
import statistics

import pandas as pd
from sklearn.metrics import accuracy_score

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class PragmaticReasoningMetric(SeaHelmMetric):
    """Metric for pragmatic reasoning accuracy.

    This metric extracts normalized categorical labels from model responses and
    computes accuracy overall and per linguistic phenomenon. It also writes
    per-example scores to the dataloader's inference DataFrame under the
    `individual_scores` column.
    """

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
        null_label: str = "null",
    ) -> None:
        """Initialize `PragmaticReasoningMetric` instance.

        Args:
            dataloader (AbstractDataloader): The dataloader for the task.
            task_config (TaskConfig): The task configuration.
            null_label (str, optional): Fallback label when extraction fails or
                predicted label is out-of-vocabulary. Defaults to "null".
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

    def extract_response(self, response: list) -> str:
        """Extract, normalize, and map the model's response to a valid label.

        The method delegates extraction to the base implementation using a
        task-specific regex, then normalizes the text and maps it to the
        canonical case of known labels. Unknown labels are mapped to
        `self.null_label`.

        Args:
            response (list): List of response strings for a row (typically a
                single-item list for single-turn tasks).

        Returns:
            str: The normalized prediction label.
        """
        output = super().extract_response(response)
        output = self.normalize_answer(output)
        output = self.label_map.get(output, self.null_label)
        return output

    def postprocess_responses(self) -> None:
        """Postprocess responses to produce cleaned predictions and metadata.

        - Populates `self.postprocessed_response_column` with cleaned labels.
        - Extracts and writes `linguistic_phenomenon` as a separate column.
        """
        self.dataloader.inference_df[self.postprocessed_response_column] = (
            self.dataloader.inference_df[self.response_column].apply(
                self.extract_response
            )
        )

        # make linguistic_phenomenon a column
        self.dataloader.inference_df["linguistic_phenomenon"] = [
            x["linguistic_phenomenon"] for x in self.dataloader.inference_df["metadata"]
        ]

    def calculate_metrics(self) -> dict:
        """Calculate accuracy metrics by linguistic phenomenon.

        Returns:
            dict: A dictionary with a `subcategories` field mapping each
                linguistic phenomenon to a tuple of
                (num_correct, num_examples). Also writes per-example
                `individual_scores` with `normalized_accuracy` (0/1).
        """
        metric_dict = {"subcategories": {}}
        for phenomenon in self.dataloader.inference_df[
            "linguistic_phenomenon"
        ].unique():
            subset = self.dataloader.inference_df[
                self.dataloader.inference_df["linguistic_phenomenon"] == phenomenon
            ]
            subset_predictions = subset[self.postprocessed_response_column]
            subset_references = subset[self.label_column]
            subset_correct = accuracy_score(
                y_true=subset_references, y_pred=subset_predictions, normalize=False
            )
            subset_size = len(subset_references)

            metric_dict["subcategories"].update(
                {phenomenon: (subset_correct, subset_size)}
            )
            logger.info(
                "Accuracy for phenomenon <%s>: %d / %d",
                phenomenon,
                subset_correct,
                subset_size,
            )

        predictions = self.dataloader.inference_df[self.postprocessed_response_column]
        references = self.dataloader.inference_df[self.label_column]
        individual_scores = predictions.eq(references, axis=0).astype(int)
        self.dataloader.inference_df["individual_scores"] = [
            {"normalized_accuracy": x} for x in individual_scores
        ]

        return metric_dict


class PragmaticReasoningLogProbMetric(SeaHelmMetric):
    """Metric for pragmatic reasoning using cumulative log-probabilities.

    This metric aggregates the cumulative log-probability assigned to the gold
    answer for each example and reports per-phenomenon averages and CLT
    standard errors. It additionally emits per-example probability-based scores
    to `individual_scores`.
    """

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
    ) -> None:
        """Initialize `PragmaticReasoningLogProbMetric` instance.

        Args:
            dataloader (AbstractDataloader): The dataloader for the task.
            task_config (TaskConfig): The task configuration.
        """
        super().__init__(dataloader=dataloader, task_config=task_config)

    def postprocess_responses(self) -> None:
        """Prepare auxiliary columns required for metric computation.

        Adds `linguistic_phenomenon` as a separate column derived from the
        `metadata` field in the inference DataFrame.
        """
        self.dataloader.inference_df["linguistic_phenomenon"] = [
            x["linguistic_phenomenon"] for x in self.dataloader.inference_df["metadata"]
        ]

    def calculate_metrics(self) -> dict:
        """Compute cumulative log-probability metrics by phenomenon.

        For each linguistic phenomenon, the method computes:
        - null_count: Number of examples with missing cumulative logprob
        - average_cumulative_logprobs: Mean cumulative logprob (excluding nulls)
        - clt_se_cumulative_logprobs: CLT standard error of those logprobs
        - average_cumulative_probabilities: Mean 100·exp(logprob) with nulls as 0
        - clt_se_cumulative_probabilities: CLT standard error of those probabilities

        It also writes per-example `individual_scores` as
        `{ "probabilities_accuracy": probability }`.

        Returns:
            dict: A dictionary with a `subcategories` field containing the
                statistics listed above for each phenomenon.
        """
        metric_dict = {"subcategories": {}}
        cumulative_probabilities = []
        for phenomenon in self.dataloader.inference_df[
            "linguistic_phenomenon"
        ].unique():
            subset = self.dataloader.inference_df[
                self.dataloader.inference_df["linguistic_phenomenon"] == phenomenon
            ]

            subset_cumulative_logprobs = [x[0] for x in subset["cumulative_logprobs"]]
            subset_null_count = sum(pd.isna(x) for x in subset_cumulative_logprobs)
            subset_logprobs_wo_nulls = [
                x for x in subset_cumulative_logprobs if not pd.isna(x)
            ]

            if len(subset_logprobs_wo_nulls) > 1:
                subset_average_cumulative_logprobs = statistics.mean(
                    subset_logprobs_wo_nulls
                )
                subset_clt_se_cumulative_logprobs = self.calculate_stderr(
                    subset_logprobs_wo_nulls
                )
            else:
                logger.warning(
                    "Num valid cumulative logprobs < 2. Setting average and stderr to 0."
                )
                subset_average_cumulative_logprobs = 0.0
                subset_clt_se_cumulative_logprobs = 0.0

            # replace None with 0 for subset cumulative probabilities
            subset_cumulative_probabilities = [
                0 if pd.isna(x) else math.exp(x) * 100
                for x in subset_cumulative_logprobs
            ]

            subset_average_cumulative_probabilities = statistics.mean(
                subset_cumulative_probabilities
            )
            subset_clt_se_cumulative_probabilities = self.calculate_stderr(
                subset_cumulative_probabilities
            )
            metric_dict["subcategories"].update(
                {
                    phenomenon: {
                        "null_count": subset_null_count,
                        "average_cumulative_logprobs": subset_average_cumulative_logprobs,
                        "clt_se_cumulative_logprobs": subset_clt_se_cumulative_logprobs,
                        "average_cumulative_probabilities": subset_average_cumulative_probabilities,
                        "clt_se_cumulative_probabilities": subset_clt_se_cumulative_probabilities,
                    }
                }
            )
            logger.info(
                "Accuracy for phenomenon <%s> ± stderr: %.2f ± %.2f",
                phenomenon,
                subset_average_cumulative_probabilities,
                subset_clt_se_cumulative_probabilities,
            )
            cumulative_probabilities.extend(subset_cumulative_probabilities)

        self.dataloader.inference_df["individual_scores"] = [
            {"probabilities_accuracy": x} for x in cumulative_probabilities
        ]

        return metric_dict
