import re

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class F1AccMetric(SeaHelmMetric):
    """Metric class for calculating F1 and accuracy scores."""

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
        null_label: str = "null",
    ):
        """Initialize the F1AccMetric.

        Args:
            dataloader (AbstractDataloader): The dataloader to use.
            task_config (TaskConfig): The task configuration.
            null_label (str, optional): The null label. Defaults to "null".
        """
        super().__init__(dataloader=dataloader, task_config=task_config)
        self.null_label = null_label

        if "global_mmlu_lite" in self.task:
            # HACK to handle tasks with too few labels in global_mmlu_lite
            if self.lang == "my":
                unique_labels = {"က", "ခ", "ဂ", "ဃ"}
            else:
                unique_labels = {"A", "B", "C", "D"}
        else:
            unique_labels = set(
                self.dataloader.inference_df[self.label_column].to_list()
            )

        # use map to convert all labels to string
        label_string = "|".join(map(str, unique_labels))
        self.label_map = {label.lower(): label for label in unique_labels}
        self.regex_string = (
            task_config.config["languages"][self.lang]["prompt_template"]["answer_tag"]
            + rf"[\s\*]*({label_string})+"
        )
        logger.info(
            "Using the following regex to extract the model response: %s",
            self.regex_string,
        )

    def extract_response(
        self,
        response: list,
        flags: re.RegexFlag = re.IGNORECASE,
        return_original_response_on_failure: bool = True,
    ) -> str | int:
        """Extract the output answer from the model's response.

        Args:
            response (list): The model's response.
            flags (re.RegexFlag, optional): Regex flags to use when extracting the answer. Defaults to 0.
            return_original_response_on_failure (bool, optional): Whether to return the original response on failure. Defaults to True.

        Returns:
            str | int: The extracted output answer.
        """
        output = super().extract_response(
            response,
            flags=flags,
            return_original_response_on_failure=return_original_response_on_failure,
        )
        output = self.normalize_answer(output)
        output = self.label_map.get(output, self.null_label)
        return output

    def calculate_metrics(self) -> dict:
        """Calculate the F1 and accuracy scores.

        Returns:
            dict: A dictionary containing the F1 and accuracy scores.
        """
        predictions = self.dataloader.inference_df[self.postprocessed_response_column]
        references = self.dataloader.inference_df[self.label_column]

        labels = list(set(references))
        null_count = sum(predictions == self.null_label)

        accuracy = balanced_accuracy_score(
            y_true=references,
            y_pred=predictions,
        )
        individual_scores = predictions.eq(references, axis=0).astype(int)
        self.dataloader.inference_df["individual_scores"] = [
            {"normalized_accuracy": x} for x in individual_scores
        ]

        avg_f1 = f1_score(
            y_true=references,
            y_pred=predictions,
            labels=labels,
            average="macro",
        )
        null_weighted_f1 = avg_f1 * (1 - null_count / len(predictions))

        macro_f1 = f1_score(
            y_true=references,
            y_pred=predictions,
            average="macro",
        )
        conf_matrix = confusion_matrix(y_true=references, y_pred=predictions)
        class_report = classification_report(y_true=references, y_pred=predictions)
        logger.info(
            f"Balanced Acc = {accuracy * 100:.2f} | Macro-F1 = {macro_f1 * 100:.2f} | Null-Weighted-F1 = {null_weighted_f1 * 100:.2f}"
        )
        logger.info("Confusion matrix:\n%s", conf_matrix)
        logger.info("Classification report:\n%s", class_report)

        metric_dict = {
            "accuracy": 100 * accuracy,
            "macro_f1": 100 * macro_f1,
            "null_weighted_f1": 100 * null_weighted_f1,
            "normalized_accuracy": 100
            * self.normalize_score(accuracy, 1 / len(self.label_map), 1),
            "null_count": null_count,
        }
        return metric_dict
