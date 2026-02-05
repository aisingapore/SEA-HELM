import os

import pandas as pd
from datasets import Image

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.judges.seahelm_judge import SeaHelmJudge
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class RefClipScoreJudge(SeaHelmJudge):
    """Judge that evaluates image captioning using RefCLIPScore.

    This judge uses a CLIP model to compute similarity scores between
    generated captions and reference images. It loads images from the dataset
    and prepares them alongside model responses for evaluation.
    """

    def __init__(
        self,
        judge_model,
        judge_config: dict,
        dataloader: AbstractDataloader,
        metric: SeaHelmMetric,
        task_config: TaskConfig,
        response_column: str = "responses",
    ):
        """Initialize the RefClipScoreJudge.

        Args:
            judge_model: The judge model serving instance (e.g., OpenAIServing, VertexAIServing).
            judge_config (dict): The configuration dictionary for the judge model.
            dataloader (AbstractDataloader): Dataloader instance containing inference data.
            metric (SeaHelmMetric): Metric instance for calculating evaluation scores.
            task_config (dict): Task configuration dictionary containing judge settings.
            response_column (str, optional): Column name in dataloader containing model
                responses to be judged. Defaults to "response".
        """
        super().__init__(
            judge_model=judge_model,
            judge_config=judge_config,
            dataloader=dataloader,
            metric=metric,
            task_config=task_config,
            response_column=response_column,
        )

        # Load images again using the dataloader
        if not self.dataloader.dataset:
            self.dataloader.load_dataset()

        # Extract image features
        images = self.dataloader.dataset.select_columns(["images"])
        images = images.map(lambda x: {"image": x["images"][0]}, num_proc=16)
        self.images = images.cast_column("image", Image(decode=True))

    def get_judgement_file_name(self) -> str:
        """Generate the filename for storing judge evaluation results.

        Returns:
            str: Formatted filename in the pattern:
                 {model_name}_{task}_{lang}_{judge_model_name}_judgement.jsonl
        """
        return f"{os.path.basename(self.model_name)}_{self.task}_{self.lang}_{os.path.basename(self.judge_model_name)}_judgement.jsonl"

    def get_judge_prompts(self, row_no: int | None = None) -> dict:
        """Retrieve judge prompts from the task configuration.

        Note: RefCLIPScore evaluation does not use text prompts, as it directly
        computes similarity between images and captions using embeddings.

        Args:
            row_no (int | None): The row number (unused). Defaults to None.

        Returns:
            dict: Empty dictionary as no text prompts are needed.
        """
        return {}

    def prepare_judge_prompts(
        self, judge_prompt: str, row: pd.Series, row_no: int
    ) -> tuple[list, list]:
        """Prepare judge input by combining model responses with images and captions.

        This method constructs conversation messages containing the model's generated
        caption (as user input), the reference image, and the ground truth caption
        (as assistant response) for RefCLIPScore evaluation.

        Args:
            judge_prompt (str): The judge prompt template (unused in this implementation).
            row (pd.Series): The data row containing model responses and captions.
            row_no (int): The row number/index for retrieving the corresponding image.

        Returns:
            tuple[list, list]: A tuple containing:
                - conversations (list): List of message dictionaries with text and image content.
                - custom_ids (list): List of row numbers as custom identifiers.
        """
        responses = self.metric.get_response(row)
        responses = self.metric.extract_response(responses)
        images = self.images[row_no]["image"]

        # Construct judge prompt messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": responses},
                    {"type": "image", "image": images},
                ],
            },
            {"role": "assistant", "content": row["captions"]},
        ]

        # Add messages and custom IDs to lists
        conversations = [messages]
        custom_ids = [row_no]
        return conversations, custom_ids
