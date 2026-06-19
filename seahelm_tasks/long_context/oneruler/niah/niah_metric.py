from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class NIAHMetric(SeaHelmMetric):
    """Metric class for calculating NIAH scores."""

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
    ):
        """Initialize the NIAHMetric.

        Args:
            dataloader (AbstractDataloader): The dataloader to use.
            task_config (TaskConfig): The task configuration.
        """
        super().__init__(dataloader=dataloader, task_config=task_config)
        answer_tag = task_config.config["languages"][self.lang]["prompt_template"][
            "answer_tag"
        ]
        self.regex_string = rf"(?<={answer_tag})([\s\S]*)(?=</)"

    def extract_response(self, response: list, use_lowercase: bool = False) -> str:
        """Extract the response from the model output.

        Args:
            response (list): The list of response strings.
            use_lowercase (bool, optional): Whether to convert response to lowercase. Defaults to False.

        Returns:
            str: The extracted response string.
        """
        output = super().extract_response(
            response, return_original_response_on_failure=False
        )
        return str(output)

    def compare_response(self, pred: str, ref: list) -> tuple:
        """Compare the predicted response with the reference.

        Args:
            pred (str): The predicted response string.
            ref (list): The reference list of items.

        Returns:
            tuple: A tuple containing the score (float), sorted prediction list, and sorted reference list.
        """
        # Convert pred to list and strip all white spaces
        try:
            pred = pred.split(",")
            pred = [item.strip().lower() for item in pred]
        except Exception:
            logger.warning(f"Failed to process prediction: {pred}")
            return 0

        set_pred = set(pred)
        set_ref = set(ref)

        score = (
            len(set_pred.intersection(set_ref)) / len(set_ref)
            if len(set_ref) > 0
            else 0.0
        )

        return score, sorted(pred), sorted(ref)

    def calculate_metrics(self) -> dict:
        """Calculate the NIAH metrics.

        Returns:
            dict: A dictionary containing the calculated metrics.
        """
        predictions = self.dataloader.dataframe[self.postprocessed_response_column]
        references = self.dataloader.dataframe[self.label_column]

        accuracy_scores = []
        error_reports = []
        for pred, ref in zip(predictions, references, strict=True):
            score, pred_list, ref_list = self.compare_response(pred, ref)
            if score == 1.0:
                accuracy_scores.append(1)
                error_reports.append(None)
            else:
                accuracy_scores.append(score)
                error_report = f"'predicted': {pred_list}, 'expected': {ref_list}"
                error_reports.append(error_report)

        self.dataloader.dataframe["error_reports"] = error_reports

        strict_accuracy_scores = [1 if score == 1.0 else 0 for score in accuracy_scores]
        strict_accuracy = sum(strict_accuracy_scores) / len(strict_accuracy_scores)
        accuracy = sum(accuracy_scores) / len(accuracy_scores)

        logger.info(
            "NIAH Strict Accuracy: %.2f | NIAH Accuracy: %.2f",
            strict_accuracy * 100,
            accuracy * 100,
        )

        self.dataloader.update_individual_scores(
            [
                {"normalized_strict_accuracy": x, "normalized_accuracy": y}
                for x, y in zip(strict_accuracy_scores, accuracy_scores, strict=True)
            ]
        )
        metric_dict = {
            "strict_accuracy": 100 * strict_accuracy,
            "accuracy": 100 * accuracy,
            "normalized_strict_accuracy": 100
            * self.normalize_score(strict_accuracy, 0, 1),
        }

        return metric_dict
