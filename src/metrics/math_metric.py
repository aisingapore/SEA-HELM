from math_verify import parse, verify

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class MathMetric(SeaHelmMetric):
    """Metric class for calculating math tasks scores."""

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
    ):
        """Initialize the MathMetric.

        Args:
            dataloader (AbstractDataloader): The dataloader to use.
            task_config (TaskConfig): The task configuration.
        """
        super().__init__(dataloader=dataloader, task_config=task_config)

    def safe_parse(self, x: str) -> list:
        """Safe parse the response.

        Args:
            x (str): The response to parse.

        Returns:
            list: The parsed response.
        """
        try:
            return parse(x)
        except Exception as e:
            logger.warning(f"Error parsing {x}: {e}")
            return []

    def calculate_metrics(self) -> dict:
        """Calculate the math tasks scores.

        Returns:
            dict: A dictionary containing the math tasks scores.
        """
        predictions = (
            self.dataloader.inference_df[self.postprocessed_response_column]
            .apply(self.safe_parse)
            .to_list()
        )
        references = (
            self.dataloader.inference_df[self.label_column]
            .apply(self.safe_parse)
            .to_list()
        )
        metric_dict = {}

        scores = []
        for ref, pred in zip(references, predictions, strict=True):
            score = int(verify(ref, pred))
            scores.append(score)
        normalized_scores = [self.normalize_score(x, 0, 1) for x in scores]

        math_verify_score = sum(scores) / len(scores)
        normalized_math_verify_score = self.normalize_score(math_verify_score, 0, 1)

        logger.info(
            f"Normalized Math-Verify Score: {100 * normalized_math_verify_score:.2f}"
        )
        metrics = {
            "math_verify_score": 100 * math_verify_score,
            "normalized_math_verify_score": 100 * normalized_math_verify_score,
        }

        metric_dict.update(metrics)
        self.dataloader.inference_df["individual_scores"] = [
            {"normalized_math_verify_score": x} for x in normalized_scores
        ]

        return metric_dict
