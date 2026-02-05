import os

import pandas as pd
from sacrebleu.metrics import CHRF

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)

METRICX_MIN_SCORE, METRICX_MAX_SCORE = 25, 0


class TranslationMetric(SeaHelmMetric):
    """Evaluate translation quality with MetricX and ChrF++.

    This class computes quality scores for machine translation outputs.
    It supports:
    - MetricX (reference-based, optionally without references)
    - ChrF++ (character F-score)

    Args:
        dataloader (AbstractDataloader): Dataloader instance with inference results.
        task_config (TaskConfig): The task configuration.
    """

    def __init__(self, dataloader: AbstractDataloader, task_config: TaskConfig) -> None:
        super().__init__(dataloader=dataloader, task_config=task_config)
        self.use_metricx_metric: bool = task_config.config["use_metricx_metric"]
        self.use_chrf_metric: bool = task_config.config["use_chrf_metric"]
        self.regex_string: str = (
            task_config.config["languages"][self.lang]["prompt_template"]["answer_tag"]
            + r"[\s\r\n`*]*(.*)"
        )

        logger.info(
            "Using the following regex to extract the model response: %s",
            self.regex_string,
        )

    def get_judgement_file_name(self) -> str:
        return f"{os.path.basename(self.dataloader.model_name)}_{self.task}_{self.lang}_judgement.jsonl"

    def evaluate_with_metricx(
        self,
    ) -> tuple[dict[str, float], list[float]]:
        """Evaluate with MetricX and return aggregate and per-example scores.

        Returns:
            tuple[dict[str, float], list[float]]: A tuple containing:
                - metrics dict with aggregate MetricX scores
                - list of normalized scores per example (0-100 scale)
        """
        # reference scores
        judgements_filepath = os.path.join(
            self.dataloader.get_parent_folder(),
            self.get_judgement_file_name(),
        )
        judgement_df = pd.read_json(judgements_filepath, lines=True)

        # scores = judgement_df[""]
        scores = judgement_df["score"].to_list()
        metricx_wmt24_scores = sum(scores) / len(scores)
        normalized_scores = [
            self.normalize_score(x, METRICX_MIN_SCORE, METRICX_MAX_SCORE) * 100
            for x in scores
        ]
        metricx_wmt24_norm_scores = sum(normalized_scores) / len(normalized_scores)
        metrics = {
            "metricx_wmt24_scores": metricx_wmt24_scores,
            "normalized_metricx_wmt24_scores": metricx_wmt24_norm_scores,
        }
        logger.info("MetricX WMT24 score: %f", metricx_wmt24_scores)
        return metrics, normalized_scores

    def evaluate_with_chrf(
        self, references: list[str], predictions: list[str]
    ) -> dict[str, float | None]:
        """Evaluate with ChrF++ metric.

        Args:
            references (list[str]): Reference translations.
            predictions (list[str]): Model-generated translations.

        Returns:
            dict[str, float | None]: Dictionary with `chrf_score` and
            `normalized_chrf_score`.
        """
        chrf = CHRF(word_order=2)

        if len(predictions) > 0:
            scores = chrf.corpus_score(predictions, [references])
            logger.info(f"ChrF++ Score: {scores.score}")
            score = scores.score
        else:
            score = None

        return {
            "chrf_score": score,
            "normalized_chrf_score": self.normalize_score(score, 0, 1),
        }

    def calculate_metrics(self) -> dict[str, float | None]:
        """Compute and aggregate configured translation metrics.

        Returns:
            dict[str, float | None]: Aggregated metrics including MetricX and/or
            ChrF++ depending on configuration, plus a `null_count` of empty outputs.
        """
        predictions = self.dataloader.inference_df[
            self.postprocessed_response_column
        ].to_list()
        references = self.dataloader.inference_df[self.label_column].to_list()

        metric_dict: dict[str, float | None] = {}

        if self.use_metricx_metric:
            metricx_metricx, scores = self.evaluate_with_metricx()
            metric_dict.update(metricx_metricx)
            self.dataloader.inference_df["individual_scores"] = [
                {"normalized_metricx_wmt24_scores": x} for x in scores
            ]

        if self.use_chrf_metric:
            # run CHRF metrics
            chrf_metrics = self.evaluate_with_chrf(references, predictions)
            metric_dict.update(chrf_metrics)

        null_count = sum([1 for pred in predictions if pred == ""])
        metric_dict["null_count"] = null_count

        return metric_dict
