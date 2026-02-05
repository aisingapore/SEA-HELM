import math
import statistics

import pandas as pd

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class LogProbMetric(SeaHelmMetric):
    """Metric class for calculating log probability scores."""

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
    ):
        """Initialize the LogProbMetric.

        Args:
            dataloader (AbstractDataloader): The dataloader to use.
            task_config (TaskConfig): The task configuration.
        """
        super().__init__(dataloader=dataloader, task_config=task_config)

    def postprocess_responses(self) -> None:
        """Postprocess the responses."""
        pass

    def calculate_metrics(self) -> dict:
        """Calculate the log probability scores.

        Returns:
            dict: A dictionary containing the log probability scores.
        """
        cumulative_logprobs = [
            x[0] for x in self.dataloader.inference_df["cumulative_logprobs"]
        ]

        null_count = sum(pd.isna(x) for x in cumulative_logprobs)
        _logprobs_wo_nulls = [x for x in cumulative_logprobs if not pd.isna(x)]

        if len(_logprobs_wo_nulls) > 1:
            average_cumulative_logprobs = statistics.mean(_logprobs_wo_nulls)
            clt_se_cumulative_logprobs = self.calculate_stderr(_logprobs_wo_nulls)
        else:
            logger.warning(
                "Num valid cumulative logprobs < 2. Setting average and stderr to 0."
            )
            average_cumulative_logprobs = 0.0
            clt_se_cumulative_logprobs = 0.0

        # replace None with 0 for cumulative probabilities
        cumulative_probabilities = [
            0 if pd.isna(x) else math.exp(x) * 100 for x in cumulative_logprobs
        ]

        average_probabilities = statistics.mean(cumulative_probabilities)
        clt_se_cumulative_probabilities = self.calculate_stderr(
            cumulative_probabilities
        )

        self.dataloader.inference_df["individual_scores"] = [
            {"probabilities_accuracy": x} for x in cumulative_probabilities
        ]

        logger.info(
            "Average cumulative logprobs ± stderr: %.2f ± %.2f",
            average_cumulative_logprobs,
            clt_se_cumulative_logprobs,
        )
        logger.info(
            "Average cumulative probabilities ± stderr: %.2f ± %.2f",
            average_probabilities,
            clt_se_cumulative_probabilities,
        )

        metric_dict = {
            "null_count": null_count,
            "average_cumulative_logprobs": average_cumulative_logprobs,
            "clt_se_cumulative_logprobs": clt_se_cumulative_logprobs,
            "average_cumulative_probabilities": average_probabilities,
            "clt_se_cumulative_probabilities": clt_se_cumulative_probabilities,
        }
        return metric_dict
