import re

import pandas as pd

from src.base_logger import get_logger
from src.metrics.seahelm_metric import SeaHelmMetric

logger = get_logger(__name__)


class BabyVisionJudgeMetric(SeaHelmMetric):
    """Metric class for evaluating responses using LLM judge with criteria-based scoring.

    This metric uses an LLM judge to evaluate responses based on multiple criteria.
    Each criterion is evaluated based on the defined criteria system in the config file.
    The final score is the average across all criteria.
    """

    def extract_response(
        self,
        response: list,
        flags: re.RegexFlag = re.IGNORECASE,
        return_original_response_on_failure: bool = True,
    ) -> str:
        """Extract the output from the model's response.

        Args:
            response (list): The model's response.
            flags (re.RegexFlag, optional): Regex flags to use when extracting the answer. Defaults to re.IGNORECASE.
            return_original_response_on_failure (bool, optional): Whether to return the original response on failure. Defaults to True.

        Returns:
            str: The extracted output.
        """
        # try to extract the answer from the response using regex else return the response as it is
        try:
            pattern = r"\\{1,2}boxed\{((?:[^{}]|{(?:[^{}]|{.*})*})*)\}"
            matches = re.findall(pattern, response[0], flags)
            output = matches[-1].strip()  # Return content from last \boxed{}
        except Exception:
            if return_original_response_on_failure:
                output = response[0]
            else:
                return ""

        return output

    def extract_judgement(self, judge_response: str) -> str | None:
        """Extract the classification judgement from the judge's response."""
        if pd.isna(judge_response):
            return None

        # Extract text between <classification> tags
        correct_match = re.findall(r"\bTRUE\b", judge_response)
        incorrect_match = re.findall(r"\bFALSE\b", judge_response)

        if correct_match and incorrect_match:
            logger.warning(
                "Both TRUE and FALSE classifications found in judge response. Response: %s",
                judge_response,
            )
            return 0
        elif correct_match:
            return 1
        return 0

    def calculate_metrics(self) -> tuple[dict, pd.DataFrame]:
        """Calculate response quality metrics based on LLM judge evaluations.

        This method:
        1. Loads judgement data from the judgement JSONL file
        2. Extracts classifications and reasoning from judge responses
        3. Maps classifications to numerical scores (0.0 to 1.0)
        4. Aggregates scores across multiple criteria for each response
        5. Calculates overall and per-category metrics

        Returns:
            A tuple containing:
                - A dictionary with metric scores including:
                    - 'average_criteria_score': Overall average score (0-100)
                    - 'categories': Per-category average scores
                - The inference DataFrame with additional columns:
                    - 'scores': List of scores per criterion
                    - 'reasoning': List of judge reasoning per criterion
                    - 'final_scores': Average score across all criteria
                    - 'individual_scores': Dictionary with 'average_criteria_score'
        """
        # Get judgements
        llm_judgement_file_path = self.dataloader.get_judge_batch_response_filepath()
        judgement_df = pd.read_json(llm_judgement_file_path, lines=True)

        # Parse judgements
        judgement_df["score"] = judgement_df.apply(
            lambda row: self.extract_judgement(row["responses"]),
            axis=1,
        )
        self.dataloader.dataframe["score"] = judgement_df["score"]

        average_score = judgement_df["score"].mean()
        logger.info("Overall accuracy score: %.2f", average_score * 100)

        metric_dict = {"accuracy_score": average_score * 100}
        self.dataloader.update_individual_scores(
            [{"accuracy_score": x} for x in self.dataloader.dataframe["score"]]
        )

        # calculate category-wise scores
        _metric_category = {}
        categories = self.dataloader.dataframe["metadata"].apply(
            lambda x: x.get("type", "unknown")
        )
        for category in categories.unique():
            mask = categories == category
            if mask.sum() == 0:
                continue
            category_score = judgement_df.loc[mask, "score"].mean()
            logger.info(
                "Category: %s | Accuracy Score: %.2f", category, category_score * 100
            )
            _metric_category[category] = category_score * 100

        metric_dict["categories"] = _metric_category

        return metric_dict
