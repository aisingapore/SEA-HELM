from sacrebleu.metrics import CHRF

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.rouge_score.rouge_scorer import RougeScorer
from src.rouge_score.scoring import BootstrapAggregator
from src.task_config import TaskConfig

logger = get_logger(__name__)


class SummarizationMetric(SeaHelmMetric):
    """Metric class for abstractive summarization.

    This metric extracts the model's answer using an answer tag regex and
    computes corpus-level ROUGE-L and ChrF++ scores.

    Args:
        dataloader (AbstractDataloader): Dataloader providing inference data and labels.
        task_config (TaskConfig): The task configuration.
    """

    def __init__(self, dataloader: AbstractDataloader, task_config: TaskConfig) -> None:
        super().__init__(dataloader=dataloader, task_config=task_config)
        self.regex_string = (
            task_config.config["languages"][self.lang]["prompt_template"]["answer_tag"]
            + r"[\s\r\n`*]*(.*)"
        )
        logger.info(
            "Using the following regex to extract the model response: %s",
            self.regex_string,
        )

        if self.lang == "th":
            language = "thai"
        elif self.lang == "my":
            language = "burmese"
        else:
            language = None

        self.scorer = RougeScorer(["rougeL"], use_stemmer=False, lang=language)

        self.run_chrf = task_config.config["use_chrf_metric"]
        # self.run_bertscore = task_config.task_config["use_bertscore_metric"]
        self.run_rougeL = task_config.config["use_rougeL_metric"]

    def calculate_metrics(self) -> dict:
        """Calculate evaluation metrics for the current dataset.

        Uses the configured metric toggles to compute ROUGE-L and/or ChrF++ on the
        postprocessed predictions against references. Also records the number of
        empty predictions.

        Returns:
            dict: Aggregated metric values including:
                - null_count (int)
                - rougel_precision, rougel_recall, rougel_f1, normalized_rougel_f1 (if enabled)
                - chrf_score, normalized_chrf_score (if enabled)
        """
        predictions = self.dataloader.inference_df[
            self.postprocessed_response_column
        ].to_list()
        references = self.dataloader.inference_df[self.label_column].to_list()
        null_count = sum([x == "" for x in predictions])

        metric_dict = {"null_count": null_count}
        if self.run_rougeL:
            rougeL_metricx, scores = self.evaluate_with_rougeL(references, predictions)
            metric_dict.update(rougeL_metricx)
            self.dataloader.inference_df["individual_scores"] = [
                {"normalized_rougel_f1": x} for x in scores
            ]

        # run chrf metrics
        if self.run_chrf:
            chrf_metrics = self.evaluate_with_chrf(references, predictions)
            metric_dict.update(chrf_metrics)

        return metric_dict

    def evaluate_with_rougeL(
        self, references: list[str], predictions: list[str]
    ) -> tuple[dict, list[float]]:
        """Compute corpus ROUGE-L and per-example normalized ROUGE-L scores.

        Args:
            references (list[str]): List of ground-truth summaries.
            predictions (list[str]): List of model-generated summaries.

        Returns:
            tuple[dict, list[float]]: A tuple containing:
                - metric_dict: Corpus-level ROUGE-L metrics (precision, recall, f1, normalized f1)
                - normalized_scores: Per-example normalized ROUGE-L F1 scores in [0, 1]
        """
        rouge_score = [
            self.scorer.score(ref, pred)
            for ref, pred in zip(references, predictions, strict=True)
        ]

        if len(rouge_score) > 0:
            aggregator = BootstrapAggregator()

            for score in rouge_score:
                aggregator.add_scores(score)
            aggregates = aggregator.aggregate()
            mid_scores = aggregates["rougeL"].mid
            norm_f1_score = self.normalize_score(
                mid_scores.fmeasure, 0, 1
            )  # 1 is the max f1 score

            logger.info("Rouge-L Scores:")
            logger.info(
                f"Precision: {100 * mid_scores.precision:.2f} | Recall: {100 * mid_scores.recall:.2f} | F1: {100 * mid_scores.fmeasure:.2f}"
            )

            # calculate norm score
            logger.info(f"Norm F1 Score: {100 * norm_f1_score:.2f}")

            metric_dict = {
                "rougel_precision": 100 * mid_scores.precision,
                "rougel_recall": 100 * mid_scores.recall,
                "rougel_f1": 100 * mid_scores.fmeasure,
                "normalized_rougel_f1": 100 * norm_f1_score,
            }

        normalized_scores = [
            self.normalize_score(100 * x["rougeL"].fmeasure, 0, 1) for x in rouge_score
        ]
        return metric_dict, normalized_scores

    def calculate_max_score_for_normalization(self) -> float:
        """Compute maximum achievable ROUGE-L F1 for normalization.

        Calculates ROUGE-L by comparing each label against itself and then
        aggregates with bootstrap to obtain the corpus-level F1, which serves as
        the theoretical maximum for normalization.

        Returns:
            float: The corpus-level ROUGE-L F1 when reference equals prediction.
        """
        max_rouge_score = self.dataloader.inference_df.apply(
            lambda x: self.scorer.score(x[self.label_column], x[self.label_column]),
            axis=1,
        )
        norm_aggregator = BootstrapAggregator()

        for score in max_rouge_score:
            norm_aggregator.add_scores(score)
        aggregates = norm_aggregator.aggregate()
        mid_scores = aggregates["rougeL"].mid

        logger.info("Normalized Rouge-L Scores:")
        logger.info(
            f"Precision: {100 * mid_scores.precision:.2f} | Recall: {100 * mid_scores.recall:.2f} | F1: {100 * mid_scores.fmeasure:.2f}"
        )
        return mid_scores.fmeasure

    def evaluate_with_chrf(self, references: list[str], predictions: list[str]) -> dict:
        """Compute corpus ChrF++ and its normalized variant.

        Args:
            references (list[str]): List of ground-truth summaries.
            predictions (list[str]): List of model-generated summaries.

        Returns:
            dict: Dictionary with keys `chrf_score` and `normalized_chrf_score`.
        """
        chrf = CHRF(word_order=2)

        if len(predictions) > 0:
            scores = chrf.corpus_score(predictions, [references])
            score = scores.score
            logger.info(f"ChrF++ Score: {score}")
        else:
            score = None

        return {
            "chrf_score": score,
            "normalized_chrf_score": self.normalize_score(score, 0, 1),
        }
