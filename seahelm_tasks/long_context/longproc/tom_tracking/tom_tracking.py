import re
import string

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


def _normalize_tom(s: str) -> str:
    """Normalize a Theory-of-Mind belief string.

    This performs the following operations, in order:
    1) Lowercase the text
    2) Remove ASCII punctuation (excluding the right single quotation mark ’)
    3) Remove specific common articles/stop-words relevant to ToM tracking
    4) Collapse repeated whitespace

    Args:
        s (str): The input belief string to normalize.

    Returns:
        str: The normalized belief string.
    """

    def remove_articles(text: str) -> str:
        return re.sub(
            r"\b(a|an|the|on|in|at|the|step|thinks|think|believes|believe|is|are|of|location|know|knows|belief)\b",
            " ",
            text,
        )

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punctuation(text: str) -> str:
        return "".join(ch for ch in text if ch not in string.punctuation and ch != "’")

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))


def _extract_belief_content(line: str) -> str | None:
    """Extract and normalize the belief content from a bullet point line.

    The expected input format is a Markdown-like bullet that starts with a hyphen,
    e.g. "- John believes the key is in the drawer". The function strips the
    leading hyphen, trims whitespace, and normalizes the belief text.

    Args:
        line (str): A single line potentially containing a belief prefixed with '-'.

    Returns:
        str | None: The normalized belief content if the line starts with '-';
        otherwise, None.
    """
    if line.startswith("-"):
        # Split on the first hyphen and get the content after it
        belief_content = line.split("-", 1)[1].strip()
        # Normalize the belief content
        belief_content = _normalize_tom(belief_content)
        return belief_content
    else:
        return None


class ToMTrackingMetric(SeaHelmMetric):
    """Metric for Theory-of-Mind (ToM) belief tracking tasks.

    This metric compares the model's predicted sequence of beliefs against the
    ground truth sequence, after extracting normalized belief contents from
    bullet-point lines. It reports exact sequence accuracy, partial accuracy up
    to the first mismatch, and a normalized accuracy.
    """

    def __init__(self, dataloader: AbstractDataloader, task_config: TaskConfig) -> None:
        """Initialize the ToMTrackingMetric.

        Args:
            dataloader (AbstractDataloader): The dataloader providing inputs and labels.
            task_config (TaskConfig): The task configuration.
        """
        super().__init__(dataloader=dataloader, task_config=task_config)

    def extract_response(
        self,
        response: list,
        flags: re.RegexFlag = 0,
        return_original_response_on_failure: bool = False,
    ) -> str:
        """Extract the model's response string from a list of turns.

        Args:
            response (list): The list of response strings (per turn). Assumes the
                first element is the relevant response for this task.
            flags (re.RegexFlag, optional): Unused. Regex flags to use when extracting the answer. Defaults to 0.
            return_original_response_on_failure (bool, optional): Unused. Whether to return the original response on failure. Defaults to False.

        Returns:
            str: The stripped response string.
        """
        output = response[0]
        return output.strip()

    def calculate_metrics(self) -> dict[str, float]:
        """Calculate ToM tracking metrics.

        The method parses bullet-point lists of beliefs from both predictions and
        references, normalizes each belief entry, and computes:
        - accuracy: exact match of the entire belief sequence
        - partial_accuracy: fraction correct up to the first mismatch
        - normalized_accuracy: accuracy scaled via ``normalize_score``

        It also writes per-example diagnostics to ``dataframe``:
        - ``error_reports`` detailing the first mismatched line (if any)
        - ``individual_scores`` containing the normalized partial accuracy

        Returns:
            dict[str, float]: A dictionary with keys ``accuracy``, ``partial_accuracy``,
            and ``normalized_accuracy``.
        """
        predictions = self.dataloader.dataframe[self.postprocessed_response_column]
        references = self.dataloader.dataframe[self.label_column]

        accuracy_scores, partial_accuracy_scores = [], []
        error_reports = []
        for pred, ref in zip(predictions, references, strict=True):
            pred_beliefs = pred.strip().split("\n")
            ref_beliefs = ref.strip().split("\n")

            # Process the lines to extract belief contents
            model_beliefs = [
                _extract_belief_content(line)
                for line in pred_beliefs
                if _extract_belief_content(line)
            ]
            ground_truth_beliefs = [
                _extract_belief_content(line)
                for line in ref_beliefs
                if _extract_belief_content(line)
            ]
            if len(model_beliefs) == len(ground_truth_beliefs) and all(
                a == b for a, b in zip(model_beliefs, ground_truth_beliefs, strict=True)
            ):
                accuracy_scores.append(1.0)
                partial_accuracy_scores.append(1.0)
                error_reports.append(None)
            else:
                accuracy_scores.append(0.0)
                first_diff = next(
                    (
                        i
                        for i, (a, b) in enumerate(
                            zip(model_beliefs, ground_truth_beliefs, strict=True)
                        )
                        if a != b
                    ),
                    None,
                )
                if first_diff is not None:
                    partial_accuracy_scores.append(
                        first_diff / len(ground_truth_beliefs)
                    )
                    error_reports.append(
                        f"""line: {first_diff} | gt: {ground_truth_beliefs[first_diff]} | pr: {model_beliefs[first_diff]}"""
                    )
                else:
                    # Handle case where there are no mismatches but lengths are different
                    partial_accuracy_scores.append(
                        min(len(model_beliefs), len(ground_truth_beliefs))
                        / len(ground_truth_beliefs)
                    )
                    error_reports.append(None)

        self.dataloader.dataframe["error_reports"] = error_reports
        self.dataloader.update_individual_scores(
            [
                {"normalized_partial_accuracy_score": self.normalize_score(x, 0, 1)}
                for x in partial_accuracy_scores
            ]
        )

        metric_dict: dict[str, float] = {
            "accuracy": 100 * sum(accuracy_scores) / len(accuracy_scores),
            "partial_accuracy": 100
            * sum(partial_accuracy_scores)
            / len(partial_accuracy_scores),
            "normalized_accuracy": 100
            * self.normalize_score(sum(accuracy_scores) / len(accuracy_scores), 0, 1),
        }
        return metric_dict
