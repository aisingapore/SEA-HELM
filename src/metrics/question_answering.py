import string
from collections import Counter
from typing import Any, Callable, List

from pythainlp import word_tokenize

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class QuestionAnsweringMetric(SeaHelmMetric):
    """Metric class for calculating question answering scores."""

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
    ):
        """Initialize the QuestionAnsweringMetric.

        Args:
            dataloader (AbstractDataloader): The dataloader to use.
            task_config (TaskConfig): The task configuration.
        """
        super().__init__(dataloader=dataloader, task_config=task_config)
        self.regex_string = (
            task_config.config["languages"][self.lang]["prompt_template"]["answer_tag"]
            + r"[\s\r\n`*]*(.*)"
        )
        logger.info(
            "Using the following regex to extract the model response: %s",
            self.regex_string,
        )

    def normalize_answer(self, s: str) -> str:
        """Lower text and remove punctuation and extra whitespace.

        Args:
            s (str): The string to normalize.

        Returns:
            str: The normalized string.
        """

        # Articles only apply to English
        # def remove_articles(text):
        #     return re.sub(r"\b(a|an|the)\b", " ", text)

        #########################################################################
        # TODO:
        # 1. Support stripping punctuation of non ASCII characters
        # 2. Consider whether need to strip punctuation like ')', etc.
        # 3. Better to use regex to rewrite this function
        #########################################################################
        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))

    def get_references(self) -> List[Any]:
        """Get the references from the dataloader.

        Returns:
            List[Any]: The references from the dataloader.
        """
        return self.dataloader.inference_df[self.label_column]

    def _f1_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate the F1 score.

        Args:
            prediction (str): The prediction.
            ground_truth (str): The ground truth.

        Returns:
            float: The F1 score.
        """
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(self, prediction: str, ground_truth: str) -> bool:
        """Calculate the exact match score.

        Args:
            prediction (str): The prediction.
            ground_truth (str): The ground truth.

        Returns:
            bool: The exact match score.
        """
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)

    def metric_max_over_ground_truths(
        self,
        metric_fn: Callable,
        prediction: str,
        ground_truths: List[str],
    ) -> float:
        """Calculate the maximum over the ground truths.

        Args:
            metric_fn (Callable): The metric function.
            prediction (str): The prediction.
            ground_truths (List[str]): The ground truths.

        Returns:
            float: The maximum over the ground truths.
        """
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def _evaluate(
        self,
        references: List[Any],
        predictions: List[str],
    ) -> tuple[dict, List[float]]:
        """Evaluate the predictions.

        Args:
            references (List[Any]): The references.
            predictions (List[str]): The predictions.

        Returns:
            tuple[dict, List[float]]: The evaluation results.
        """
        exact_match = []
        f1_list = []
        for ref, pred in zip(references, predictions, strict=True):
            exact_match.append(
                self.metric_max_over_ground_truths(self.exact_match_score, pred, ref)
            )
            f1_list.append(
                self.metric_max_over_ground_truths(self._f1_score, pred, ref)
            )

        total = len(f1_list)
        exact_match = 100.0 * sum(exact_match) / total if total > 0 else None
        f1 = 100.0 * sum(f1_list) / total if total > 0 else None

        normalized_f1 = self.normalize_score(f1, 0, 1)
        results = {"exact_match": exact_match, "f1": f1, "normalized_f1": normalized_f1}
        return results, f1_list

    def _tokenize_thai_text(self, text: str) -> str:
        """Tokenize the Thai text.

        Args:
            text (str): The text to tokenize.

        Returns:
            str: The tokenized text.
        """
        tokenized_text = " ".join(word_tokenize(engine="newmm", text=text))
        return tokenized_text

    def _tokenize_chinese_text(self, text: str) -> str:
        """Tokenize the Chinese text.

        Args:
            text (str): The text to tokenize.

        Returns:
            str: The tokenized text.
        """
        import jieba

        tokenized_text = " ".join(jieba.cut(text))
        return tokenized_text

    def calculate_metrics(self) -> dict:
        """Calculate the metrics.

        Returns:
            dict: The metrics.
        """
        references = self.get_references()
        predictions = self.dataloader.inference_df[self.postprocessed_response_column]

        if self.lang == "th":
            # Use PyThaiNLP newmm Thai tokenizer because Thai script does not use spaces between words
            tokenized_references = [
                [self._tokenize_thai_text(ref[0])] for ref in references
            ]
            tokenized_predictions = [
                self._tokenize_thai_text(pred) for pred in predictions
            ]

            logger.info("Tokenizing Thai text and re-evaluating...")
            results, f1_list = self._evaluate(
                tokenized_references, tokenized_predictions
            )
            logger.info(results)
        elif self.lang == "zh":
            tokenized_references = [
                [self._tokenize_chinese_text(ref[0])] for ref in references
            ]
            tokenized_predictions = [
                self._tokenize_chinese_text(pred) for pred in predictions
            ]

            logger.info("Tokenizing Chinese text and re-evaluating...")
            results, f1_list = self._evaluate(
                tokenized_references, tokenized_predictions
            )
            logger.info(results)
        else:
            results, f1_list = self._evaluate(references, predictions)
            logger.info(results)

        self.dataloader.inference_df["individual_scores"] = [
            {"normalized_f1": self.normalize_score(x, 0, 1)} for x in f1_list
        ]
        # Analyze if preds contain gold answer fully
        question_count = len(references)
        gold_in_pred = 0
        try:
            for ref, pred in zip(references, predictions, strict=True):
                answer = ref[0]
                if answer.lower().strip(".") in pred.lower().strip("."):
                    gold_in_pred += 1
        except Exception as e:
            logger.info(answer)
            logger.info(pred)
            raise ValueError("Error in analyzing gold answer in prediction.") from e

        if question_count > 0:
            logger.info(
                f"{gold_in_pred} answers out of {question_count} ({gold_in_pred * 100 / question_count:.2f}%) can be found in the model's predictions."
            )

            results.update({"found_in_prediction": gold_in_pred * 100 / question_count})

        null_count = sum(predictions == "")
        results.update({"null_count": null_count})

        return results
