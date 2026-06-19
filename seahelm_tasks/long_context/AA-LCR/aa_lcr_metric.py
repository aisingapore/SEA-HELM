import re

import pandas as pd

from src.base_logger import get_logger
from src.metrics.seahelm_metric import SeaHelmMetric

logger = get_logger(__name__)


class AALCRJudgeMetric(SeaHelmMetric):
    """Metric class for evaluating responses using LLM judge with criteria-based scoring.

    This metric uses an LLM judge to evaluate responses based on multiple criteria.
    Each criterion is evaluated based on the defined criteria system in the config file.
    The final score is the average across all criteria.
    """

    def extract_judgement(self, judge_response: str) -> str | None:
        """Extract the classification judgement from the judge's response."""
        if pd.isna(judge_response):
            return None

        # Extract text between <classification> tags
        correct_match = re.findall(r"\bCORRECT\b", judge_response)
        incorrect_match = re.findall(r"\bINCORRECT\b", judge_response)

        if correct_match and incorrect_match:
            logger.warning(
                "Both CORRECT and INCORRECT classifications found in judge response. Response: %s",
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

        return metric_dict
