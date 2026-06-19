import csv
import re
import string

import pandas as pd

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)

_ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(s: str) -> str:
    """Normalize text for TSV cell comparison.

    The normalization lowercases the text, strips ASCII punctuation, removes
    English articles, collapses multiple spaces, and removes all remaining
    spaces to mitigate format differences between model outputs and labels.

    Args:
        s (str): Input string.

    Returns:
        str: Normalized string suitable for equality comparison.
    """

    ## Remove articles
    def remove_articles(text: str) -> str:
        return _ARTICLES_REGEX.sub(" ", text)

    ## Fix white spaces
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    ## Remove white spaces
    def remove_white_spaces(text: str) -> str:
        return text.replace(" ", "")

    ## Remove punctuation
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    ## Lowercase the text
    def lower(text: str) -> str:
        return text.lower()

    return remove_white_spaces(white_space_fix(remove_articles(remove_punc(lower(s)))))


class HtmlToTsvMetric(SeaHelmMetric):
    """Metric for evaluating HTML-to-TSV extraction quality.

    This metric expects model responses to include a TSV block wrapped in
    triple backticks with a `tsv` language tag. It extracts the TSV content
    using a regex and compares it to the ground truth TSV by converting both
    to pandas DataFrames and computing precision, recall, and F1 scores on a
    row-wise unordered basis.

    Args:
        dataloader (AbstractDataloader): The dataloader containing inference
            results and labels.
        task_config (TaskConfig): The task configuration.
    """

    def __init__(self, dataloader: AbstractDataloader, task_config: TaskConfig) -> None:
        super().__init__(dataloader=dataloader, task_config=task_config)
        # Regex used by the base class to extract TSV content from responses
        self.regex_string: str = {"en": r"(?<=```tsv)([\s\S]*)(?=```)"}[self.lang]

    def _string_to_pd(self, text: list[str]) -> pd.DataFrame | None:
        """Convert TSV lines into a pandas DataFrame with normalized values.

        The first row is interpreted as the header. If the last row appears
        incomplete (e.g., trailing commentary or an empty row), it is dropped.
        Any rows with the wrong number of columns are replaced with "N/A".
        All cell values are normalized via `normalize_answer`.

        Args:
            text (list[str]): List of TSV lines (each line is a row string).

        Returns:
            pd.DataFrame | None: A DataFrame with normalized string values, or
                None if parsing fails.
        """
        try:
            csv_reader = csv.reader(text, delimiter="\t")

            data = list(csv_reader)
            ## Take the first line as the header
            header = data[0]
            ## Remove the header from the data
            data = data[1:]
            ## If the last line is not complete, it is likely to be an empty line or some end markers (e.g. "The above is my extracted csv" in the model response), so just remove it
            if len(data[-1]) != len(header):
                data = data[:-1]

            for idx, line in enumerate(data):
                if len(line) != len(header):
                    ## If one of the line is not complete or has extra columns than expected , fill it with N/A
                    print(f"Line {idx} is not complete or has extra columns:", line)
                    data[idx] = ["N/A" for _ in range(len(header))]

            df = pd.DataFrame(data, columns=header)
            df = df.fillna("N/A")
            ## Normalize all answers
            for column in df.columns:
                df[column] = df[column].astype(str).apply(normalize_answer)
            return df
        except Exception as e:
            print(data)
            print("Error in converting string in TSV format to pandas dataframe")
            print(e)

            return None

    def extract_response(self, response: list[str]) -> str | None:
        """Extract the TSV block from a model response.

        Uses the base class regex extraction with `return_original_response_on_failure`
        set to False, returning None if the TSV block is not found.

        Args:
            response (list[str]): The list of model responses per turn. The
                first element is used for extraction.

        Returns:
            str | None: The extracted TSV content as a string, or None if
                extraction fails.
        """
        output = super().extract_response(
            response, return_original_response_on_failure=False
        )
        return output

    def evaluate_html_to_csv_compute_metrics(
        self, prediction: str, groundtruth: str
    ) -> dict:
        """Compute precision, recall, and F1 between predicted and gold TSV.

        The computation treats rows as unordered sets: precision counts the
        fraction of predicted rows that exist in the ground truth; recall counts
        the fraction of ground-truth rows found in the prediction. F1 is the
        harmonic mean of precision and recall.

        Args:
            prediction (str): The predicted TSV text.
            groundtruth (str): The ground-truth TSV text.

        Returns:
            dict: A dictionary with keys:
                - "precision" (float)
                - "recall" (float)
                - "f1" (float)
                - "error" (None | str): Error type name if a parsing error occurs.
        """
        try:
            ## Convert the groundtruth pandas dataframe
            gt_text = groundtruth.lstrip().rstrip().split("\n")
            gt_df = self._string_to_pd(gt_text)
            ## Convert the prediction to pandas dataframe
            pred_text = prediction.lstrip().rstrip().split("\n")
            pred_df = self._string_to_pd(pred_text)
            ## Compute the precision score
            precision = 0
            for i in range(len(pred_df.index)):
                ## For each row in the prediction, check if it is in the groundtruth dataframe
                ## Note that for precision, we don't care if the prediction follows exactly the order of the groundtruth
                corr = (gt_df.eq(pred_df.iloc[i].values)).all(axis=1).any()
                precision += corr

            precision /= len(pred_df.index)

            ## Compute the UNORDERED recall score
            recall = 0
            for i in range(len(gt_df.index)):
                ## For each row in the ground truth, check if it is in the prediction dataframe
                corr = (pred_df.eq(gt_df.iloc[i].values)).all(axis=1).any()
                recall += corr
            recall /= len(gt_df.index)
            if recall + precision == 0:
                f1 = 0
            else:
                f1 = 2 * recall * precision / (recall + precision)
            return {"precision": precision, "recall": recall, "f1": f1, "error": None}
        except Exception as e:
            print("Error in evaluating HTML to CSV")
            print(e)

            return {
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "error": str(type(e).__name__),
            }

    def calculate_metrics(self) -> dict:
        """Calculate aggregated precision/recall/F1 over the dataset.

        Iterates over predictions and references, computes per-example metrics
        via `evaluate_html_to_csv_compute_metrics`, stores individual normalized
        F1 scores in the dataloader DataFrame, and aggregates metrics across
        the dataset.

        Returns:
            dict: Aggregated metrics with keys:
                - "precision" (float)
                - "recall" (float)
                - "f1" (float)
                - "normalized_f1" (float)
                - "parsing_errors" (dict[str, int])
        """
        predictions = self.dataloader.dataframe[self.postprocessed_response_column]
        references = self.dataloader.dataframe[self.label_column]

        precision_scores, recall_scores, f1_scores = [], [], []
        errors = {}
        for pred, ref in zip(predictions, references, strict=True):
            individual_metrics = self.evaluate_html_to_csv_compute_metrics(pred, ref)

            precision_scores.append(individual_metrics["precision"])
            recall_scores.append(individual_metrics["recall"])
            f1_scores.append(individual_metrics["f1"])

            # count errors by type
            error = individual_metrics["error"]
            if error is not None:
                if error in errors:
                    errors[error] += 1
                else:
                    errors[error] = 1

        self.dataloader.update_individual_scores(
            [{"normalized_f1": self.normalize_score(x, 0, 1)} for x in f1_scores]
        )

        metric_dict = {
            "precision": 100 * sum(precision_scores) / len(precision_scores),
            "recall": 100 * sum(recall_scores) / len(recall_scores),
            "f1": 100 * sum(f1_scores) / len(f1_scores),
            "normalized_f1": 100
            * self.normalize_score(sum(f1_scores) / len(f1_scores), 0, 1),
            "parsing_errors": errors,
        }
        return metric_dict
