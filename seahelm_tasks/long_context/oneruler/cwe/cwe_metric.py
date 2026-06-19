from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class CweMetric(SeaHelmMetric):
    """Metric class for Common Word Extraction (CWE)."""

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
    ):
        """Initialize the CweMetric.

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
            response (list): The list of responses from the model.
            use_lowercase (bool, optional): Whether to use lowercase. Defaults to False.

        Returns:
            str: The extracted response as a string.
        """
        output = super().extract_response(
            response, return_original_response_on_failure=False
        )
        return str(output)

    def compare_response(self, pred: str, ref: list) -> tuple[float, list, list]:
        """Compare the predicted list of words with the reference list of words.

        Args:
            pred (str): The predicted list of words.
            ref (list): The reference list of words.

        Returns:
            tuple[float, list, list]: A tuple containing the comparison result (float),
            the list of processed predictions, and the list of processed references.
        """
        # Lowercase and remove whitespaces
        pred = [
            predictions.strip().removesuffix("\n").lower()
            for predictions in pred.split(",")
        ]
        ref = [references.strip().removesuffix("\n").lower() for references in ref]

        set_pred = set(pred)
        set_ref = set(ref)

        score = (
            len(set_pred.intersection(set_ref)) / len(set_ref)
            if len(set_ref) > 0
            else 0.0
        )

        return score, sorted(pred), sorted(ref)

    def calculate_metrics(self) -> dict:
        """Calculate the accuracy metrics.

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
            "CWE Strict Accuracy: %.2f | CWE Accuracy: %.2f",
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
