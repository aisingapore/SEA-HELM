import os
from collections import Counter
from enum import Enum

import pandas as pd

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class PairwiseLLMJudgeMetric(SeaHelmMetric):
    """PairwiseLLMJudgeMetric class for calculating pairwise LLM judge metrics."""

    class JudgementOutcome(Enum):
        """Judgement outcome enum."""

        WIN = 1
        LOSE = 2
        TIE = 3
        ERROR = 4

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
    ):
        """Initialize the PairwiseLLMJudgeMetric.

        Args:
            dataloader (AbstractDataloader): The dataloader to use.
            task_config (TaskConfig): The task configuration.
        """
        super().__init__(
            dataloader=dataloader,
            task_config=task_config,
        )
        self.ties_allowed = task_config.config.get("ties_allowed", True)

        self.judgement_labels = task_config.config["judgement_labels"]  # Required
        if self.ties_allowed and "Tie" not in self.judgement_labels:
            raise ValueError(
                "If ties are allowed, 'Tie' must be included in the judgement_labels."
            )
        self.model_name = dataloader.model_name
        self.baseline_model = task_config.config.get("baseline_model", "")
        self.judge_model_name = task_config.config["judge"].get("judge_model_name", "")

    def get_judgement_file_name(self) -> str:
        return f"{os.path.basename(self.model_name)}_{self.task}_{self.lang}_{self.baseline_model}_{os.path.basename(self.judge_model_name)}_judgement.jsonl"

    # Implementing the abstract method from SeahelmMetric
    def calculate_metrics(self) -> tuple[dict, pd.DataFrame]:
        """
        Calculates pairwise LLM judge metrics.

        This method processes the judgements and calculates the metrics based on the judgements.
        It supports both batch and sample-by-sample API calls.

        Returns:
            tuple[dict, DataFrame]:
                - dict: A dictionary containing the metrics.
                - DataFrame: The inference dataframe with additional columns.
        """
        # Get judgements
        llm_judgement_file_path = os.path.join(
            self.dataloader.get_parent_folder(),
            self.get_judgement_file_name(),
        )
        judgement_df = pd.read_json(llm_judgement_file_path, lines=True)

        # Create metadata dataframe
        judgement_meta_df = pd.DataFrame(
            judgement_df["custom_id"].map(lambda x: x.split("_")).to_list(),
            columns=["question_id", "turn", "order"],
        )
        judgement_df = pd.concat([judgement_df, judgement_meta_df], axis=1)

        # Parse judgements
        judgement_df["verdict"] = judgement_df.apply(
            lambda row: self.parse_judgement(
                row["parsed_response"], row["order"] == "baseline-before"
            ),
            axis=1,
        )

        # Process inference dataframe
        inference_df_rows = []
        for _, row in self.dataloader.inference_df.iterrows():
            question_id = str(row["question_id"])
            judgements = judgement_df[judgement_df["question_id"] == question_id]

            for order in ["baseline-after", "baseline-before"]:
                judgements_order = judgements[judgements["order"] == order]
                row[order] = judgements_order["verdict"].to_list()

            final_judgement = []
            for turn in judgements["turn"].unique():
                _judgements = judgements[judgements["turn"] == turn]
                _judgements = _judgements["verdict"].tolist()
                verdict = self.get_final_judgement(*_judgements)
                final_judgement.append(verdict)

            row["final_judgement"] = final_judgement

            inference_df_rows.append(row)

        self.dataloader.inference_df = pd.DataFrame(inference_df_rows)

        # Calculate metrics
        metrics = self.get_judge_metrics(
            self.dataloader.inference_df["final_judgement"],
            [x["category"] for x in self.dataloader.inference_df["metadata"]],
        )
        self.dataloader.inference_df["individual_scores"] = [
            {
                "weighted_win_rate": [self.get_score(turn) for turn in x],
            }
            for x in self.dataloader.inference_df["final_judgement"]
        ]
        return metrics

    def get_score(self, judgement: int) -> float:
        """
        Get the score for a judgement.

        Args:
            judgement (int): The judgement.

        Returns:
            float: The score for the judgement.
        """
        if judgement == self.JudgementOutcome.WIN:
            return 1
        elif judgement == self.JudgementOutcome.TIE:
            return 0.5
        else:
            return 0

    def get_win_rate(self, judgement_list: list) -> float:
        """
        Get the win rate for a list of judgements.

        Args:
            judgement_list (list): The list of judgements.

        Returns:
            float: The win rate for the list of judgements.
        """
        # Calculate the win rate
        total_count = len(judgement_list)
        counts = Counter(judgement_list)
        win_rate = (
            counts[self.JudgementOutcome.WIN]
            + (counts[self.JudgementOutcome.TIE] * 0.5)
        ) / total_count
        return win_rate

    def get_judge_metrics(self, judgement_list: list, category_list: list) -> dict:
        """
        Get the judge metrics.

        Args:
            judgement_list (list): The list of judgements.
            category_list (list): The list of categories.

        Returns:
            dict: A dictionary containing the metrics.
        """
        # Create metric dictionary
        metric_dict = {"categories": {}}
        df = pd.DataFrame({"judgement": judgement_list, "category": category_list})

        for category in set(category_list):
            # Get subset of judgements for category
            subset = df[df["category"] == category]
            subset_judgements = subset["judgement"]
            subset_judgements = [
                j for judgement_pair in subset_judgements for j in judgement_pair
            ]
            # Get win rate for category
            subset_win_rate = self.get_win_rate(subset_judgements)
            metric_dict["categories"].update({category: subset_win_rate})
            logger.info(f"Win rate for category <{category}>: {subset_win_rate}")

        # Get overall win rate
        overall_win_rate = self.get_win_rate(
            [j for judgement_pair in judgement_list for j in judgement_pair]
        )
        metric_dict["win_rate"] = overall_win_rate
        logger.info(f"Overall win rate: {overall_win_rate}")

        # Get weighted win rate
        weighted_win_rate = sum(list(metric_dict["categories"].values())) / len(
            metric_dict["categories"]
        )
        metric_dict["weighted_win_rate"] = weighted_win_rate * 100
        logger.info(f"Weighted win rate: {weighted_win_rate}")

        return metric_dict

    def parse_judgement(self, judgement: str, reverse: bool) -> JudgementOutcome:
        """
        Parse LLM's judgement on a pairwise comparison to determine if the model being evaluated
        wins/loses/ties against a baseline model.

        The default order in which answers are presented to the judge is as follows:
        A: Model being evaluated
        B: Baseline model

        Therefore, a win is when the model outputs [[A]] as its judgement.
        However, when controlling for position bias, the judgement is repeated with the answers
        presented in reverse. (reverse=True)

        Args:
            judgement (str): The judgement.
            reverse (bool): Whether the judgement is presented in reverse order.

        Returns:
            JudgementOutcome: The judgement outcome.
        """
        # Parse judgement
        win = self.judgement_labels["A_wins"] in judgement
        lose = self.judgement_labels["B_wins"] in judgement
        tie = False
        if self.ties_allowed:
            tie = self.judgement_labels["Tie"] in judgement

        if (win + lose + tie) != 1:
            # Judge responded with conflicting judgements (Sum > 1)
            # Or judge did not provide judgement in correct format (Sum = 0)
            return self.JudgementOutcome.ERROR
        elif tie:  # If ties are not allowed, will never reach here
            return self.JudgementOutcome.TIE
        else:
            if reverse:
                if win:
                    return self.JudgementOutcome.LOSE
                elif lose:
                    return self.JudgementOutcome.WIN
            else:
                if win:
                    return self.JudgementOutcome.WIN
                elif lose:
                    return self.JudgementOutcome.LOSE

    def get_final_judgement(
        self, judgement_1: JudgementOutcome, judgement_2: JudgementOutcome
    ) -> JudgementOutcome:
        """
        Compare LLM judgement when pairs of answers are presented in both normal and reverse order.
        (This is to ensure that LLM judgement is consistent and not affected by position bias.)

        (1) If both judgements are different, the result is a TIE.
        (2) If the first judgement is a tie, the result is a TIE.
            (If the second judgement is also a tie, then overall the result is a tie.)
            (Even if the second judgement is not a tie, due to rule (1) above, the result will still be a tie.)
        (3) Otherwise, both judgements should agree on WIN or LOSE (so we can take either one).

        Args:
            judgement_1 (JudgementOutcome): The first judgement.
            judgement_2 (JudgementOutcome): The second judgement.

        Returns:
            JudgementOutcome: The final judgement.
        """
        # Compare judgements
        if self.ties_allowed:
            if judgement_1 == self.JudgementOutcome.TIE or judgement_1 != judgement_2:
                return self.JudgementOutcome.TIE
            else:
                return judgement_1
        else:
            if judgement_1 != judgement_2:
                return self.JudgementOutcome.ERROR
            else:
                return judgement_1
