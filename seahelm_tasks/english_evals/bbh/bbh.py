from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class BBHExactMatch(SeaHelmMetric):
    """Exact match metric for BBH tasks.

    This metric computes the percentage of examples where the model's
    postprocessed prediction matches the reference label exactly. It also
    reports a normalized exact match score in the range [0, 100].

    Args:
        dataloader (AbstractDataloader): The dataloader containing
            `inference_df` with model responses and reference labels.
        task_config (TaskConfig): The task configuration.
    """

    def __init__(self, dataloader: AbstractDataloader, task_config: TaskConfig) -> None:
        super().__init__(dataloader=dataloader, task_config=task_config)

        self.regex_string: str = (
            task_config.config["languages"][self.lang]["prompt_template"]["answer_tag"]
            + r"[\s\*]*(.*)"
        )

    def calculate_metrics(self) -> dict[str, float]:
        """Calculate exact match and normalized exact match metrics.

        Returns:
            dict[str, float]: A dictionary with the following keys (values in [0, 100]):
                - "exact_match": Percentage of exact matches.
                - "normalized_exact_match": Percentage after applying normalization
                  to the exact match score.
        """
        predictions = self.dataloader.inference_df[
            self.postprocessed_response_column
        ].to_list()
        references = self.dataloader.inference_df[self.label_column].to_list()
        metric_dict: dict[str, float] = {}

        exact_match_scores: list[bool] = []
        for ref, pred in zip(references, predictions, strict=True):
            score = str(ref) == str(pred)
            exact_match_scores.append(score)

        normalized_exact_match_scores: list[float] = [
            self.normalize_score(x, 0, 1) * 100 for x in exact_match_scores
        ]

        exact_match_score = sum(exact_match_scores) / len(exact_match_scores)
        normalized_exact_match = self.normalize_score(exact_match_score, 0, 1)

        logger.info(f"Normalized Exact Match: {100 * normalized_exact_match:.2f}")
        metrics = {
            "exact_match": 100 * exact_match_score,
            "normalized_exact_match": 100 * normalized_exact_match,
        }

        metric_dict.update(metrics)
        self.dataloader.inference_df["individual_scores"] = [
            {"normalized_exact_match": x} for x in normalized_exact_match_scores
        ]

        return metric_dict
