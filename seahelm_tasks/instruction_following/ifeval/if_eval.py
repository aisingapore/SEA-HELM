import fast_langdetect
import pandas as pd

from seahelm_tasks.instruction_following.ifeval.instruction_checkers import (
    BulletListChecker,
    ConstrainedOptionsChecker,
    EndChecker,
    HighlightedSectionChecker,
    JSONChecker,
    NumberFrequencyChecker,
    NumberOfParagraphsChecker,
    NumberOfSentencesChecker,
    NumberOfWordsChecker,
    PlaceholderChecker,
    PostscriptChecker,
    QuotationMarkChecker,
    RepeatPromptChecker,
    ResponseLanguageChecker,
    SectionChecker,
    StartChecker,
    TitleChecker,
    TwoResponsesChecker,
    WordAbsenceChecker,
    WordExistenceChecker,
    WordFrequencyChecker,
)
from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)

CATEGORY_MAP = {
    "length_constraints:number_words": NumberOfWordsChecker,
    "length_constraints:number_sentences": NumberOfSentencesChecker,
    "length_constraints:number_paragraphs": NumberOfParagraphsChecker,
    "detectable_content:postscript": PostscriptChecker,
    "detectable_content:number_placeholders": PlaceholderChecker,
    "detectable_format:title": TitleChecker,
    "detectable_format:number_bullet_lists": BulletListChecker,
    "detectable_format:multiple_sections": SectionChecker,
    "detectable_format:number_highlighted_sections": HighlightedSectionChecker,
    "detectable_format:json_format": JSONChecker,
    "detectable_format:constrained_response": ConstrainedOptionsChecker,
    "startend:first_word": StartChecker,
    "startend:end_checker": EndChecker,
    "startend:quotation": QuotationMarkChecker,
    "keywords:existence": WordExistenceChecker,
    "keywords:forbidden_words": WordAbsenceChecker,
    "keywords:frequency": WordFrequencyChecker,
    "keywords:number_frequency": NumberFrequencyChecker,
    "combination:repeat_prompt": RepeatPromptChecker,
    "combination:two_responses": TwoResponsesChecker,
    "language:response_language": ResponseLanguageChecker,
}


class IFEvalMetric(SeaHelmMetric):
    def __init__(self, dataloader: AbstractDataloader, task_config: TaskConfig):
        super().__init__(dataloader=dataloader, task_config=task_config)

    def calculate_metrics(self):
        self.dataloader.inference_df = self.dataloader.inference_df.apply(
            self.evaluate_response, axis=1
        )
        self.dataloader.inference_df["individual_scores"] = [
            {"overall_lang_normalized_acc": x}
            for x in self.dataloader.inference_df["lang_normalized_result"]
        ]

        metric_dict = self.summarize_results(self.dataloader.inference_df)
        return metric_dict

    def postprocess_responses(self):
        self.dataloader.inference_df[self.postprocessed_response_column] = (
            self.dataloader.inference_df[self.response_column].map(lambda x: x[0])
        )

        self.dataloader.inference_df["subcategory"] = [
            x["subcategory"] for x in self.dataloader.inference_df["metadata"]
        ]

    def check_language(self, text: str, language: str):
        response_language = fast_langdetect.detect(text)["lang"]
        if language in ["id", "ms"]:
            if response_language in {
                "id",
                "ms",
            }:  # Malay is also accepted as Indonesian
                return True, response_language
        else:
            if response_language == language:
                return True, response_language

        return False, None

    def evaluate_response(self, row):
        """
        Takes in a list of dictionaries containing information on each test instance including
        - Category/Subcategory
        - Specific kwargs for the category (e.g. num_paragraphs)

        Builds instruction checkers based on the category and kwargs provided for each instance
        and evaluates the response and returns True/False for instruction following or not.
        """
        category = row["subcategory"]
        if self.lang in {"th", "lo", "km", "my"} and "number_words" in category:
            # Skip instances with length constraint by words
            # for languages that do not use whitespace to separate words
            return row

        response = row[self.postprocessed_response_column]

        checker_kwargs = row["kwargs"][0]
        checker_kwargs = {k: v for k, v in checker_kwargs.items() if v is not None}
        checker = CATEGORY_MAP[category](
            category=category,
            language=self.lang,
            response=response,
            **checker_kwargs,
        )

        row["result"] = checker.evaluate_response()
        correct_language = (
            self.lang
            if row["metadata"]["category"] != "language"
            else checker_kwargs["response_language"]
        )
        correct_language, detected_language = self.check_language(
            text=response.replace("\n", " "), language=correct_language
        )
        row["correct_language"] = correct_language
        row["detected_language"] = detected_language
        row["lang_normalized_result"] = (
            False if not row["correct_language"] else row["result"]
        )

        return row

    def summarize_results(self, inference_df: pd.DataFrame):
        evaluation_stats = {}

        if self.lang in {"th", "lo", "km", "my"}:
            inference_df = inference_df[
                inference_df["subcategory"] != "length_constraints:number_words"
            ]

        # Overall results
        # int() required to convert numpy.int64 to int for JSON serialization
        value_counts = inference_df["result"].value_counts()
        overall_pass = int(value_counts[True]) if True in value_counts else 0
        overall_fail = int(value_counts[False]) if False in value_counts else 0
        evaluation_stats["overall_count"] = overall_pass + overall_fail
        evaluation_stats["overall_pass"] = overall_pass
        evaluation_stats["overall_acc"] = (
            overall_pass / (overall_pass + overall_fail) * 100
        )
        logger.info(
            "Overall pass: %d / %d",
            evaluation_stats["overall_pass"],
            evaluation_stats["overall_count"],
        )
        logger.info("Overall accuracy: %f", evaluation_stats["overall_acc"])

        # Check language
        evaluation_stats["correct_language_rate"] = inference_df[
            "correct_language"
        ].value_counts(normalize=True)[True]
        overall_lang_normalized_pass = int(
            inference_df["lang_normalized_result"].value_counts()[True]
        )
        evaluation_stats["overall_lang_normalized_acc"] = (
            overall_lang_normalized_pass / (overall_pass + overall_fail)
        ) * 100
        logger.info(
            "Correct language rate: %f", evaluation_stats["correct_language_rate"]
        )
        logger.info(
            "Lang normalized accuracy: %f",
            evaluation_stats["overall_lang_normalized_acc"],
        )

        # Results by subcategory
        results_breakdown = inference_df.groupby("subcategory").agg({"result": "mean"})
        results_breakdown = results_breakdown.dropna()
        evaluation_stats["subcategories"] = results_breakdown["result"].to_dict()

        return evaluation_stats
