import pandas as pd

from src.base_logger import get_logger
from src.metrics.llm_judges.pairwise_llm_judge_metric import PairwiseLLMJudgeMetric

logger = get_logger(__name__)


class PairwiseFineGrainedLLMJudgeMetric(PairwiseLLMJudgeMetric):
    def parse_judgement(self, judgement: str, reverse: bool) -> int:
        """
        Parse LLM's judgement on a pairwise comparison to determine if the model being evaluated
        wins/loses/ties against a baseline model.

        The default order in which answers are presented to the judge is as follows:
        A: Model being evaluated
        B: Baseline model

        However, when controlling for position bias, the judgement is repeated with the answers
        presented in reverse. (reverse=True)
        """
        scores = self.judgement_labels["scores"].copy()
        if reverse:
            scores.reverse()

        for label, score in zip(self.judgement_labels["labels"], scores, strict=True):
            if label in judgement:
                return score

        return None

    def get_final_judgement(self, judgement_1: int, judgement_2: int):
        """
        Sums the two judgements to get a final score.
        If either judgement is None, the final score will also be None.
        """
        if pd.isna(judgement_1) or pd.isna(judgement_2):
            return None
        return (judgement_1 + judgement_2) / 2

    def get_score(self, judgement):
        return judgement

    def get_win_rate(self, judgement_list: list):
        """
        Calculate the win rate based on the judgement scores.
        A win is defined as a score greater than 0.5 (i.e on average the model has a score better than a ties)
        """
        total_count = len(judgement_list)
        assert total_count > 0, "judgement list cannot be empty."

        wins = sum([x > 0.5 for x in judgement_list])
        ties = sum([x == 0.5 for x in judgement_list])
        win_rate = (wins + (ties * 0.5)) / total_count * 100  # Convert to percentage

        return win_rate

    def get_judge_metrics(self, judgement_list: list, category_list: list):
        metric_dict = {"categories": {}}
        df = pd.DataFrame({"judgement": judgement_list, "category": category_list})

        for category in set(category_list):
            subset = df[df["category"] == category]
            subset_judgements = [
                j for judgement_turn in subset["judgement"] for j in judgement_turn
            ]

            subset_judgements_wo_null = [j for j in subset_judgements if not pd.isna(j)]
            subset_win_rate = self.get_win_rate(subset_judgements_wo_null)
            max_score = max(self.judgement_labels["scores"]) * len(
                subset_judgements_wo_null
            )
            score = sum(subset_judgements_wo_null) / max_score * 100
            metric_dict["categories"].update(
                {
                    category: {
                        "subset_score": score,
                        "subset_win_rate": subset_win_rate,
                        "null_judgements": len(subset_judgements)
                        - len(subset_judgements_wo_null),
                    }
                }
            )
            logger.info(f"Score for category <{category}>: {score}")
            logger.info(f"Win rate for category <{category}>: {subset_win_rate}")

        overall_judgement_list = [
            j for judgement_turn in judgement_list for j in judgement_turn
        ]
        overall_judgement_list_wo_null = [
            j for j in overall_judgement_list if not pd.isna(j)
        ]
        max_score = max(self.judgement_labels["scores"]) * len(
            overall_judgement_list_wo_null
        )
        overall_score = sum(overall_judgement_list_wo_null) / max_score * 100
        metric_dict["overall_score"] = overall_score
        metric_dict["null_judgements"] = len(overall_judgement_list) - len(
            overall_judgement_list_wo_null
        )
        logger.info(f"Overall score: {overall_score}")

        overall_win_rate = self.get_win_rate(overall_judgement_list_wo_null)
        metric_dict["win_rate"] = overall_win_rate
        logger.info(f"Overall win rate: {overall_win_rate}")

        weighted_score = sum(
            [x["subset_score"] for x in metric_dict["categories"].values()]
        ) / len(metric_dict["categories"])
        metric_dict["weighted_score"] = weighted_score
        logger.info(f"Weighted score: {weighted_score}")

        weighted_win_rate = sum(
            [x["subset_win_rate"] for x in metric_dict["categories"].values()]
        ) / len(metric_dict["categories"])
        metric_dict["weighted_win_rate"] = weighted_win_rate
        logger.info(f"Weighted win rate: {weighted_win_rate}")

        return metric_dict
