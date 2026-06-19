from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class PathTraversalMetric(SeaHelmMetric):
    """Metric class for calculating Path Traversal scores.

    This metric evaluates how accurately a model outputs a route enclosed by
    `<Route>` and `</Route>` tags. It measures:
    - Accuracy (exact match)
    - Partial accuracy (fraction of matching lines from the top)
    - Extraction rate (successful parsing of a route block)
    - Normalized accuracy (min-max normalized accuracy)
    """

    def __init__(self, dataloader: AbstractDataloader, task_config: TaskConfig) -> None:
        """Initialize the PathTraversalMetric.

        Args:
            dataloader (AbstractDataloader): The dataloader providing inputs and labels.
            task_config (TaskConfig): The task configuration.
        """
        super().__init__(dataloader=dataloader, task_config=task_config)
        self.regex_string: str = {"en": r"(?<=<Route>)([\s\S]*)(?=</Route>)"}[self.lang]

    def extract_response(self, response: list) -> str | None:
        """Extract the route string from the model response.

        This overrides the base extraction to return `None` when parsing fails
        (instead of falling back to the original raw response).

        Args:
            response (list): Model response tokens/messages for a single example.

        Returns:
            str | None: The extracted route content between `<Route>` and
            `</Route>` if parsing succeeds; otherwise `None`.
        """
        output = super().extract_response(
            response, return_original_response_on_failure=False
        )
        return output

    def calculate_metrics(self) -> dict[str, float]:
        """Calculate metrics for path traversal.

        Compares extracted predictions against references and computes:
        - accuracy: Percentage of exact matches
        - partial_accuracy: Average fraction of matching lines from the top
        - extraction_rate: Percentage of examples successfully parsed
        - normalized_accuracy: Min-max normalized accuracy in percentage

        Returns:
            dict[str, float]: Dictionary of metric name to value.
        """
        predictions = self.dataloader.dataframe[self.postprocessed_response_column]
        references = self.dataloader.dataframe[self.label_column]

        accuracy_scores, partial_accuracy_scores, extraction_rate_scores = [], [], []
        error_reports = []

        for pred, ref in zip(predictions, references, strict=True):
            if pred is None:
                accuracy_scores.append(0)
                partial_accuracy_scores.append(0)
                extraction_rate_scores.append(0)
                error_reports.append("Parsing error")
                continue

            pred = pred.strip()
            ref = ref.strip()

            if ref == pred:
                accuracy_scores.append(1)
                partial_accuracy_scores.append(1)
                extraction_rate_scores.append(1)
                error_reports.append(None)
            else:
                error_report = None
                ref_lines = ref.split("\n")
                pred_lines = pred.split("\n")
                if len(pred_lines) != len(ref_lines):
                    logger.warning(
                        "Number of lines in prediction (%d) does not match number of lines in reference (%d).",
                        len(pred_lines),
                        len(ref_lines),
                    )

                for i, (gl, pl) in enumerate(zip(ref_lines, pred_lines, strict=False)):
                    if gl != pl:
                        error_report = f"""line: {i} | gt: {gl} | pr: {pl}"""
                        break
                i += 1

                accuracy_scores.append(0)
                partial_accuracy_scores.append(i / len(ref_lines))
                extraction_rate_scores.append(1)
                error_reports.append(error_report)

        self.dataloader.dataframe["metric_errors"] = error_reports
        self.dataloader.update_individual_scores(
            [
                {"normalized_partial_accuracy_score": self.normalize_score(x, 0, 1)}
                for x in partial_accuracy_scores
            ]
        )

        metric_dict = {
            "accuracy": 100 * sum(accuracy_scores) / len(accuracy_scores),
            "partial_accuracy": 100
            * sum(partial_accuracy_scores)
            / len(partial_accuracy_scores),
            "extraction_rate": 100
            * sum(extraction_rate_scores)
            / len(extraction_rate_scores),
            "normalized_accuracy": 100
            * self.normalize_score(sum(accuracy_scores) / len(accuracy_scores), 0, 1),
        }
        return metric_dict
