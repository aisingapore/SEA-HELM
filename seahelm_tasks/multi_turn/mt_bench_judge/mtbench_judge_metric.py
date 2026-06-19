import re

import pandas as pd

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class MTBenchJudgeMetric(SeaHelmMetric):
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
        """Initialize the MTBenchJudgeMetric.

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

    def extract_classification(self, judge_response: str) -> str | None:
        """Extract classification tag from judge response.

        Parses the judge's response to extract the classification value from within
        <classification> tags. The classification indicates how well the response
        meets a specific criterion.

        Args:
            judge_response: The judge's response text containing <classification> tag.

        Returns:
            The classification string (e.g., "CLASS_EXACTLY_MET") or None if the tag
            is not found or the response is null.
        """
        if pd.isna(judge_response):
            return None

        # Extract text between <classification> tags
        match = re.findall(
            r"<classification>\s*(\w+)\s*</classification>", judge_response
        )
        if match:
            return match[-1].strip()
        return None

    def extract_reflection(self, judge_response: str) -> str | None:
        """Extract reflection text from judge response.

        Parses the judge's response to extract the reflection text from within
        <reflection> tags. This provides the judge's explanation for its classification.

        Args:
            judge_response: The judge's response text containing <reflection> tag.

        Returns:
            The reflection string with whitespace trimmed, or None if the tag is not found
            or the response is null.
        """
        if pd.isna(judge_response):
            return None

        # Extract text between <reflection> tags
        match = re.findall("<reflection>(.*?)</reflection>", judge_response)
        if match:
            return match[-1].strip()
        return None

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
            columns=["id", "turn_id", "criteria_id"],
        )
        judgement_df = pd.concat([judgement_df, judgement_meta_df], axis=1)

        # Parse judgements
        judgement_df["classification"] = judgement_df.apply(
            lambda row: self.extract_classification(row["responses"]),
            axis=1,
        )

        # parse reflection
        judgement_df["reflection"] = judgement_df.apply(
            lambda row: self.extract_reflection(row["responses"]),
            axis=1,
        )

        # Process inference dataframe
        dataframe_rows = []
        for _, row in self.dataloader.dataframe.iterrows():
            question_id = str(row["id"])
            judgements = judgement_df[judgement_df["id"] == question_id]

            scores = []
            reflections = []
            overall_turn_score = []
            for turn in range(2):
                turn_judgements = judgements[judgements["turn_id"] == f"turn{turn}"]
                sorted_judgements = turn_judgements.sort_values(by=["criteria_id"])
                turn_scores = (
                    sorted_judgements["classification"].apply(self.get_score).tolist()
                )
                scores.append(turn_scores)
                reflections.append(sorted_judgements["reflection"].tolist())
                overall_turn_score.append(
                    sum(turn_scores) / len(turn_scores) if len(turn_scores) > 0 else 0.0
                )

            row["scores"] = scores
            row["turn_scores"] = overall_turn_score
            row["reflection"] = reflections
            row["final_scores"] = (
                sum(overall_turn_score) / len(overall_turn_score)
                if len(overall_turn_score) > 0
                else 0.0
            )

            dataframe_rows.append(row)

        self.dataloader.dataframe = pd.DataFrame(dataframe_rows)

        # Calculate metrics
        metrics = self.get_judge_metrics(
            self.dataloader.dataframe["final_scores"],
            [x["category"] for x in self.dataloader.dataframe["metadata"]],
        )
        self.dataloader.update_individual_scores(
            [
                {"average_criteria_score": x}
                for x in self.dataloader.dataframe["final_scores"]
            ]
        )

        return metrics

    def get_score(self, classification: str) -> float:
        """Convert a classification string to a numerical score.

        Maps classification labels to their corresponding numerical values based on
        the definition in the config.

        Args:
            classification: The classification string from the judge's evaluation.

        Returns:
            The numerical score (0.0-1.0). Returns 0.0 if classification is None
            or not recognized.
        """
        mapping = self.task_config.config.judge.get("classifications", {})
        if classification is None:
            return 0.0
        return mapping.get(classification, 0.0)

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
            )
            metric_dict["categories"].update({category: average_scores})
            logger.info(
                "Average score for category <%s>: %.2f", category, average_scores
            )

        # Get overall average score
        overall_average_score = (
            sum(judgement_list) / len(judgement_list)
            if len(judgement_list) > 0
            else 0.0
        )
        metric_dict["average_criteria_score"] = overall_average_score * 100
        logger.info("Overall average criteria score: %.2f", overall_average_score * 100)

        return metric_dict
