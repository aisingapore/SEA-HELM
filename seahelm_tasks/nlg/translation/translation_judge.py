import os

import pandas as pd

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.judges.seahelm_judge import SeaHelmJudge
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class TranslationJudge(SeaHelmJudge):
    def __init__(
        self,
        judge_model,
        judge_config: dict,
        dataloader: AbstractDataloader,
        metric: SeaHelmMetric,
        task_config: TaskConfig,
        response_column: str = "responses",
    ):
        """Initialize the TranslationJudge.

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

    def get_judgement_file_name(self) -> str:
        return f"{os.path.basename(self.model_name)}_{self.task}_{self.lang}_judgement.jsonl"

    def get_judge_prompts(self, row_no: int | None = None) -> dict:
        """
        Retrieves the judge prompts from the task configuration.

        Args:
            row_no (int): The row number to get the judge prompts for
        """
        return "source: {source} candidate: {pred} reference: {ref}"

    def prepare_judge_prompts(
        self, judge_prompt: str, row: dict, row_no: int
    ) -> tuple[list, list]:
        raw_response = self.metric.get_response(row)
        response = self.metric.extract_response(raw_response)
        source = row["prompts"][0]["text"]
        ref = row["label"]

        conversations = [
            [
                {
                    "role": "user",
                    "content": judge_prompt.format(
                        source=source, pred=response, ref=ref
                    ),
                }
            ]
        ]
        if "id" in row:
            id = row["id"]
        else:
            id = ""
        custom_ids = [f"{os.path.basename(self.model_name)}_row{id}_judge"]
        return conversations, custom_ids

    def get_batch_file_name(self) -> str:
        """
        Returns the batch file name.
        """
        raise NotImplementedError(
            "get_batch_file_name method not supported by TranslationJudge."
        )

    def get_batched_llm_judgements(self) -> pd.DataFrame:
        """
        Returns the batched LLM judgements as a DataFrame.
        """
        raise NotImplementedError(
            "get_batched_llm_judgements method not supported by TranslationJudge."
        )
