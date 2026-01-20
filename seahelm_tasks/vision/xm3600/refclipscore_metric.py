"""RefCLIPScore metric implementation for vision-language evaluation.

This module implements RefCLIPScore, a reference-aware extension of CLIPScore that evaluates
generated captions by considering both image-text alignment and similarity to reference captions.
RefCLIPScore combines:
1. CLIPScore: Measures semantic similarity between generated text and images
2. Reference similarity: Measures similarity between generated text and reference captions
3. F-score combination: Harmonically combines both scores for balanced evaluation

The implementation uses multilingual CLIP models to support cross-lingual evaluation.
"""

import os

import numpy as np
import pandas as pd

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class RefClipScoreMetric(SeaHelmMetric):
    """RefCLIPScore metric for evaluating vision-language model performance.

    RefCLIPScore is a reference-aware extension of CLIPScore that evaluates generated captions
    by combining:
    1. Image-text alignment (CLIPScore): How well the caption describes the image
    2. Reference similarity: How similar the caption is to reference captions
    3. F-score combination: Harmonic mean of the above two components

    This implementation loads pre-computed CLIP scores from the OpenClip judge that has already been
    calculated. The metric processes these judgements to compute aggregate
    statistics and normalized scores.

    Args:
        dataloader (AbstractDataloader): Dataloader containing inference data and images
        task_config (dict): Configuration dictionary containing language-specific settings
        task (str): Name of the task being evaluated
        lang (str): Language of the task
        label_column (str, optional): Column name containing reference captions. Defaults to "captions".

    Attributes:
        regex_string (str): Regex pattern for extracting model responses from inference output
        model_name (str): Name of the model being evaluated
        judge_model_name (str): Name of the judge model used for scoring
    """

    def __init__(
        self,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
        label_column: str = "captions",
    ):
        super().__init__(
            dataloader=dataloader,
            task_config=task_config,
            label_column=label_column,
        )
        self.regex_string = (
            task_config.config["languages"][self.lang]["prompt_template"]["answer_tag"]
            + r"[\s\r\n`*]*(.*)"
        )
        logger.info(
            "Using the following regex to extract the model response: %s",
            self.regex_string,
        )
        self.model_name = dataloader.model_name
        self.judge_model_name = task_config.config["judge"].get("judge_model_name", "")

    def get_judgement_file_name(self) -> str:
        return f"{os.path.basename(self.model_name)}_{self.task}_{self.lang}_{os.path.basename(self.judge_model_name)}_judgement.jsonl"

    def calculate_metrics(self):
        """Calculate RefCLIPScore and related metrics from pre-computed judgements.

        This method loads precomputed CLIP-based scores from an OpenClip judge. The workflow is:
        1. Load judgement file containing pre-computed image-text and text-text similarity scores
        2. Extract per-instance CLIPScore (image-text alignment) from judgements
        3. Extract per-instance reference similarity (text-text alignment) from judgements
        4. Compute RefCLIPScore as harmonic mean: 2 * (img_text * text_text) / (img_text + text_text)
        5. Calculate aggregate statistics (means) across all instances
        6. Normalize scores to 0-100 scale and store individual scores in dataloader

        Returns:
            dict: Dictionary containing evaluation metrics with the following keys:
                - clipscore (float): Mean CLIPScore in range [0, 1]
                - refclipscore (float): Mean RefCLIPScore in range [0, 1]
                - normalized_clipscore (float): Normalized CLIPScore (0-100 scale)
                - normalized_refclipscore (float): Normalized RefCLIPScore (0-100 scale)
                - null_count (int): Number of empty/null predictions

        Side Effects:
            - Updates self.dataloader.inference_df with individual_scores column containing
              normalized refclipscore for each instance
            - Logs CLIPScore and RefCLIPScore percentages to logger

        Example:
            >>> metric = RefClipScoreMetric(dataloader, config, task, lang)
            >>> results = metric.calculate_metrics()
            >>> print(f"RefCLIPScore: {results['refclipscore']:.3f}")
        """
        # Get judgements
        llm_judgement_file_path = os.path.join(
            self.dataloader.get_parent_folder(),
            self.get_judgement_file_name(),
        )
        judgement_df = pd.read_json(llm_judgement_file_path, lines=True)

        clipscores = []
        refclipscores = []
        for _, row in judgement_df.iterrows():
            score = row["score"]
            per_instance_image_text = score["per_instance_image_text"]
            per_instance_text_text = score["per_instance_text_text"]

            # Calculate RefCLIPScore as F-score
            refclipscore = (
                2
                * per_instance_image_text
                * per_instance_text_text
                / (per_instance_image_text + per_instance_text_text)
            )
            clipscores.append(per_instance_image_text)
            refclipscores.append(refclipscore)

        # Calculate aggregate scores
        clipscore_mean = float(np.mean(clipscores))
        refclipscore_mean = float(np.mean(refclipscores))

        # Store individual scores
        individual_scores = [
            {"refclipscore": self.normalize_score(refclip_score, 0, 1) * 100}
            for refclip_score in refclipscores
        ]

        self.dataloader.inference_df["individual_scores"] = individual_scores

        # Count null predictions
        predictions = self.dataloader.inference_df[self.postprocessed_response_column]
        null_count = sum([1 for pred in predictions if pred == ""])

        logger.info("CLIPScore: %.2f", clipscore_mean * 100)
        logger.info("RefCLIPScore: %.2f", refclipscore_mean * 100)

        metric_dict = {
            "clipscore": clipscore_mean,
            "refclipscore": refclipscore_mean,
            "normalized_clipscore": self.normalize_score(clipscore_mean, 0, 1) * 100,
            "normalized_refclipscore": self.normalize_score(refclipscore_mean, 0, 1)
            * 100,
            "null_count": null_count,
        }

        return metric_dict
