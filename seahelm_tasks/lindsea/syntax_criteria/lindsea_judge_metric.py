import re

import pandas as pd

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class LindseaJudgeMetric(SeaHelmMetric):
    """Metric class for evaluating syntactic phenomena using LLM judge with criteria-based scoring.

    This metric uses an LLM judge to evaluate syntactic phenomena based on multiple criteria.
    Each criterion is evaluated on a binary scale (either UNMET or MET) corresponding
    to scores of 0.0 and 1.0 respectively. The final score is the average across all criteria.
    """

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
    ):
        """Initialize the LindseaJudgeMetric.

        Args:
            dataloader (AbstractDataloader): The dataloader containing inference data and model information.
            task_config (TaskConfig): The task configuration containing judge model settings and criteria.
        """
        super().__init__(
            dataloader=dataloader,
            task_config=task_config,
        )
        self.model_name = dataloader.model_name
        self.judge_model_name = task_config.config["judge"].get("judge_model_name", "")

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

    def calculate_metrics(self) -> dict:
        """Calculate syntactic acceptability metrics based on LLM judge evaluations using severity weighting.

        This method:
        1. Loads judgement data from the judgement JSONL file
        2. Extracts classifications and parses criteria metadata from judge responses
        3. Maps classifications to binary scores (0.0 or 1.0) and applies defined severity weights
        (high=10, medium=5, low=1)
        4. Calculates a weighted 'criteria_score' for specific linguistic phenomena and a
        'generic_score' for basic constraints
        5. Computes the 'overall_score' as the product of 'criteria_score' and 'generic_score'
        6. Calculates overall and per-category aggregate metrics

        Returns:
            A tuple containing:
                - A dictionary with metric scores including:
                    - 'average_criteria_score': Overall average score (0-100)
                    - 'categories': Per-category average scores
                - The inference DataFrame with additional columns:
                    - 'generic_score': Weighted average of the generic criteria
                    - 'criteria_score': Weighted average of the specific linguistic criteria
                    - 'overall_score': The final sentence score (criteria_score multiplied by generic_score)
                    - 'individual_scores': Dictionary detailing the generic, criteria, and overall scores
        """

        # 1. Load and parse judgements
        llm_judgement_file_path = self.dataloader.get_judge_batch_response_filepath()
        judgement_df = pd.read_json(llm_judgement_file_path, lines=True)

        # Parse metadata from custom_id (e.g., "1_criteria1" -> id = '1', criteria_id = 'criteria1')
        judgement_meta_df = pd.DataFrame(
            judgement_df["custom_ids"].map(lambda x: x.split("_")).to_list(),
            columns=["id", "criteria_id"],
        )
        judgement_df = pd.concat([judgement_df, judgement_meta_df], axis=1)

        # Pre-parse classifications and reflections for efficiency
        judgement_df["classifications"] = judgement_df.apply(
            lambda row: self.extract_classification(row["responses"]), axis=1
        )
        judgement_df["reflections"] = judgement_df.apply(
            lambda row: self.extract_reflection(row["responses"]), axis=1
        )

        # Severity Mapping
        severity_map = {"high": 10, "medium": 5, "low": 1, "generic": 1}

        def compute_weighted_score(criteria, classifications):
            generic_score = 0.0
            criteria_score = 0.0
            total_generic_severity = 0.0
            total_criteria_severity = 0.0

            for criterion, classification in zip(
                criteria, classifications, strict=True
            ):
                severity = criterion["severity"].lower()
                score = self.get_score(classification)

                if severity == "generic":
                    severity_val = severity_map["generic"]
                    generic_score += score * severity_val
                    total_generic_severity += severity_val
                else:
                    severity_val = severity_map.get(severity, 1)
                    criteria_score += score * severity_val
                    total_criteria_severity += severity_val

            weighted_generic_score = (
                (generic_score / total_generic_severity)
                if total_generic_severity > 0
                else 0.0
            )
            weighted_criteria_score = (
                (criteria_score / total_criteria_severity)
                if total_criteria_severity > 0
                else 0.0
            )

            return weighted_generic_score, weighted_criteria_score

        dataframe_rows = []
        for _, row in self.dataloader.dataframe.iterrows():
            question_id = str(row["id"])  # id = 0, 1, 2...
            item_judgements = judgement_df[judgement_df["id"] == question_id]
            item_judgements = item_judgements.sort_values("criteria_id")

            # Classifications is an array of strings: ['CLASS_MET', 'CLASS_UNMET', ...]
            row["classifications"] = item_judgements["classifications"].tolist()

            # Reflections is an array of strings: ['reflection1', 'reflection2', ...]
            row["reflections"] = item_judgements["reflections"].tolist()

            # (1) Calculate generic_score
            row["generic_score"], row["criteria_score"] = compute_weighted_score(
                row["criteria"], row["classifications"]
            )

            # (3) Calculate overall_score
            row["overall_score"] = row["criteria_score"] * row["generic_score"]

            dataframe_rows.append(row)

        self.dataloader.dataframe = pd.DataFrame(dataframe_rows)
        self.dataloader.update_individual_scores(
            [
                {
                    "average_criteria_score": row["overall_score"],
                    "generic_score": row["generic_score"],
                    "criteria_score": row["criteria_score"],
                }
                for _, row in self.dataloader.dataframe.iterrows()
            ]
        )

        # Calculate aggregate metrics using the new overall_score
        metrics = self.get_judge_metrics(
            self.dataloader.dataframe["overall_score"],
            [x["category"] for x in self.dataloader.dataframe["metadata"]],
        )

        return metrics

    def get_score(self, classification: str) -> float:
        """Convert a classification string to a numerical score.

        Maps classification labels to their corresponding numerical values based on
        the definition in the config.

        Args:
            classification: The classification string from the judge's evaluation.

        Returns:
            The numerical score (0.0 or 1.0). Returns 0.0 if classification is None
            or not recognized.
        """
        mapping = self.task_config.config["judge"].get("classifications", {})
        if classification is None:
            return 0.0
        return mapping.get(classification, 0.0)

    def get_judge_metrics(self, judgement_list: list, category_list: list) -> dict:
        """Calculate aggregate metrics from judgements and categories.

        Computes both per-category and overall average scores from the judgement data.
        Logs average scores for each category and the overall average.

        Args:
            judgement_list: List of numerical scores (0.0 or 1.0) for each generated sentence.
            category_list: List of category labels corresponding to each judgement.

        Returns:
            A dictionary containing:
                - 'average_criteria_score': Overall average score scaled to 0-100
                - 'categories': Dictionary mapping category names to their average scores (0.0-100.0)
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

        # Get overall average score for linguistic phenomenon
        overall_average_score = (
            sum(judgement_list) / len(judgement_list)
            if len(judgement_list) > 0
            else 0.0
        ) * 100
        metric_dict["average_criteria_score"] = overall_average_score
        logger.info("Overall average criteria score: %.2f", overall_average_score)

        return metric_dict
