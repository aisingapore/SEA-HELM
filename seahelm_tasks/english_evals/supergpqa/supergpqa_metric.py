from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.base_logger import get_logger
from src.metrics.f1_acc_metric import F1AccMetric

logger = get_logger(__name__)


class SuperGPQAMetric(F1AccMetric):
    def calculate_metrics(self) -> dict:
        """Calculate the F1 and accuracy scores.

        Returns:
            dict: A dictionary containing the F1 and accuracy scores.
        """
        predictions = self.dataloader.dataframe[self.postprocessed_response_column]
        references = self.dataloader.dataframe[self.label_column]

        labels = list(set(references))
        null_count = sum(predictions == self.null_label)

        accuracy = balanced_accuracy_score(
            y_true=references,
            y_pred=predictions,
        )
        individual_scores = predictions.eq(references, axis=0).astype(int)
        self.dataloader.update_individual_scores(
            [{"normalized_accuracy": x} for x in individual_scores]
        )

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
            "Balanced Acc = %.2f | Macro-F1 = %.2f | Null-Weighted-F1 = %.2f",
            accuracy * 100,
            macro_f1 * 100,
            null_weighted_f1 * 100,
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

        # calculate metrics per difficulty level
        _metric_difficulty = {}
        difficulty_levels = self.dataloader.dataframe["metadata"].apply(
            lambda x: x.get("difficulty", "unknown")
        )
        for difficulty in difficulty_levels.unique():
            mask = difficulty_levels == difficulty
            if mask.sum() == 0:
                continue
            difficulty_accuracy = balanced_accuracy_score(
                y_true=references[mask],
                y_pred=predictions[mask],
            )
            difficulty_macro_f1 = f1_score(
                y_true=references[mask],
                y_pred=predictions[mask],
                average="macro",
            )
            logger.info(
                "Difficulty: %s | Balanced Acc = %.2f | Macro-F1 = %.2f",
                difficulty,
                difficulty_accuracy * 100,
                difficulty_macro_f1 * 100,
            )
            _metric_difficulty[difficulty] = {
                "accuracy": difficulty_accuracy,
                "macro_f1": difficulty_macro_f1,
            }
        metric_dict["difficulty"] = _metric_difficulty

        # calculate metrics per field
        _metric_field = {}
        fields = self.dataloader.dataframe["metadata"].apply(
            lambda x: x.get("field", "unknown")
        )
        for field in fields.unique():
            mask = fields == field
            if mask.sum() == 0:
                continue
            field_accuracy = balanced_accuracy_score(
                y_true=references[mask],
                y_pred=predictions[mask],
            )
            field_macro_f1 = f1_score(
                y_true=references[mask],
                y_pred=predictions[mask],
                average="macro",
            )
            logger.info(
                "Field: %s | Balanced Acc = %.2f | Macro-F1 = %.2f",
                field,
                field_accuracy * 100,
                field_macro_f1 * 100,
            )
            _metric_field[field] = {
                "accuracy": field_accuracy,
                "macro_f1": field_macro_f1,
            }
        metric_dict["field"] = _metric_field

        return metric_dict
