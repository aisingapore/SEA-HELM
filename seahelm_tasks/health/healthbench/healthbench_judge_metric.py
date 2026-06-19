import json
import re

import pandas as pd

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class HealthbenchJudgeMetric(SeaHelmMetric):
    """Metric class for evaluating responses using LLM judge with criteria-based scoring.

    This metric uses an LLM judge to evaluate responses based on multiple criteria.
    Each criterion is evaluated based on the defined criteria system in the config file.
    The final score is the average across all criteria.
    """

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
    ):
        """Initialize the HealthbenchJudgeMetric.

        Args:
            dataloader (AbstractDataloader): The dataloader containing inference data and model information.
            task_config (TaskConfig): The task configuration containing judge model settings and criteria.
        """
        super().__init__(
            dataloader=dataloader,
            task_config=task_config,
        )
        self.model_name = dataloader.model_name
        self.judge_model_name = task_config.config.judge.get("judge_model_name", "")

    def extract_json(self, judge_response: str) -> str | None:
        """Extract classification tag from judge response."""
        if pd.isna(judge_response):
            return None

        # Extract text between ```json``` tags
        match = re.findall(r"\{[^{}]*\}", judge_response, re.DOTALL)

        if match:
            json_string = match[-1].strip()
            try:
                return json.loads(json_string)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse JSON from judge response: %s.\nAttempting to fix escaped quotes.",
                    json_string,
                )

            try:
                json_strings = json_string.replace(
                    '\\"criteria_met\\"', '"criteria_met"'
                )
                return json.loads(json_strings)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse JSON from judge response after fixing escaped quotes: %s",
                    json_strings,
                )
        return {}

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

        # Create metadata dataframe
        judgement_meta_df = pd.DataFrame(
            judgement_df["custom_ids"].map(lambda x: x.split("_")).to_list(),
            columns=["id", "criteria_id"],
        )
        # Parse judgements
        judgement_df = pd.concat([judgement_df, judgement_meta_df], axis=1)

        # Process inference dataframe
        dataframe_rows = []
        for _, row in self.dataloader.dataframe.iterrows():
            question_id = str(row["id"])
            judgements = judgement_df[judgement_df["id"] == question_id]

            sorted_judgements = judgements.sort_values(by=["criteria_id"])
            criteria_scores = [int(x["points"]) for x in row["criteria"]]
            scores = []
            for judgement_row, points in zip(
                sorted_judgements.to_dict(orient="records"),
                criteria_scores,
                strict=True,
            ):
                json_data = self.extract_json(judgement_row["responses"])
                criteria_met = json_data.get("criteria_met") if json_data else None
                try:
                    scores.append(criteria_met * points)
                except Exception as e:
                    logger.warning(
                        "Non-numeric value encountered in criteria_met: %s. Defaulting to 0. Error: %s",
                        judgement_row["id"],
                        e,
                    )
                    scores.append(0.0)

            maximum_score = sum([x if x > 0 else 0 for x in criteria_scores])
            final_scores = max(0, sum(scores)) / maximum_score

            row["scores"] = scores
            row["final_scores"] = final_scores

            dataframe_rows.append(row)

        self.dataloader.dataframe = pd.DataFrame(dataframe_rows)

        # Calculate metrics
        metrics = self.get_judge_metrics(
            self.dataloader.dataframe["final_scores"],
            [x["example_tags"][0] for x in self.dataloader.dataframe["metadata"]],
        )
        self.dataloader.update_individual_scores(
            [
                {"average_criteria_score": x}
                for x in self.dataloader.dataframe["final_scores"]
            ]
        )

        return metrics

    def get_judge_metrics(self, judgement_list: list, category_list: list) -> dict:
        """Calculate aggregate metrics from judgements and categories.

        Computes both per-category and overall average scores from the judgement data.
        Logs average scores for each category and the overall average.

        Args:
            judgement_list: List of numerical scores (0.0-1.0) for each response.
            category_list: List of category labels corresponding to each judgement.

        Returns:
            A dictionary containing:
                - 'average_criteria_score': Overall average score scaled to 0-100
                - 'categories': Dictionary mapping category names to their average scores (0.0-1.0)
        """
        # Create metric dictionary
        metric_dict = {"categories": {}}
        df = pd.DataFrame({"judgement": judgement_list, "category": category_list})

        for category in set(category_list):
            # Get subset of judgements for category
            subset = df[df["category"] == category]
            subset_judgements = subset["judgement"]
            # Get average scores for category
            average_scores = (
                sum(subset_judgements) / len(subset_judgements)
                if len(subset_judgements) > 0
                else 0.0
            ) * 100
            metric_dict["categories"].update({category: average_scores})
            logger.info(
                "Average score for category <%s>: %.2f", category, average_scores
            )

        # Get overall average score
        overall_average_score = (
            sum(judgement_list) / len(judgement_list)
            if len(judgement_list) > 0
            else 0.0
        ) * 100
        metric_dict["average_criteria_score"] = overall_average_score
        logger.info("Overall average criteria score: %.2f", overall_average_score)

        return metric_dict
