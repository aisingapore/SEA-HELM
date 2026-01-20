import os

import pandas as pd

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.judges.seahelm_judge import SeaHelmJudge
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class CriteriaJudge(SeaHelmJudge):
    """Judge implementation for criteria-based evaluation of model responses.

    This judge evaluates model responses against specific criteria or key points,
    generating a separate judgement for each criterion. Commonly used for tasks
    like translation quality assessment where multiple aspects need evaluation.

    Attributes:
        judge_model: The model instance used for generating judgements.
        judge_config (dict): Configuration for the judge behavior.
        dataloader (AbstractDataloader): Data loading interface for the task.
        metric (SeaHelmMetric): Metric calculator for the task.
        task_config (TaskConfig): Task-specific configuration.
        response_column (str): Column name containing model responses.
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
        """Initialize the CriteriaJudge.

        Args:
            judge_model: Model instance for generating judgements.
            judge_config (dict): Judge-specific configuration settings.
            dataloader (AbstractDataloader): Data loader for the evaluation task.
            metric (SeaHelmMetric): Metric calculation implementation.
            task_config (TaskConfig): Configuration for the task being evaluated.
            response_column (str, optional): Name of the column containing model
                responses. Defaults to "responses".
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
        """Generate the filename for storing judge judgement results.

        The filename follows the pattern:
        <model>_<task>_<lang>_<judge>_judgement.jsonl

        Returns:
            str: Filename for the judgement results JSONL file.
        """
        return f"{os.path.basename(self.model_name)}_{self.task}_{self.lang}_{os.path.basename(self.judge_model_name)}_judgement.jsonl"

    def get_batch_file_name(self) -> str:
        """Generate the filename for storing batch API requests.

        The filename follows the pattern:
        <model>_<task>_<lang>_<judge>_batch.jsonl

        Returns:
            str: Filename for the batch requests JSONL file.
        """
        return f"{os.path.basename(self.model_name)}_{self.task}_{self.lang}_{os.path.basename(self.judge_model_name)}_batch.jsonl"

    def get_judge_prompts(self, row_no: int | None = None) -> dict:
        """Retrieve the judge prompts from the task configuration.

        Args:
            row_no (int | None, optional): Row number in the dataset. Currently unused
                but kept for interface compatibility. Defaults to None.

        Returns:
            dict: Dictionary containing judge prompt templates with keys like
                "system_prompt", "criteria_prompt", and optionally "reference_prompt".
        """
        judge_config = self.task_config.config.get("judge", {})
        return judge_config["judge_prompts"]

    def prepare_judge_prompts(
        self, judge_prompt: dict, row: pd.Series, row_no: int
    ) -> tuple[list[list], list[str]]:
        """Prepare judge prompt conversations for criteria-based evaluation.

        Creates a separate judge prompt for each criterion in the task, allowing
        independent evaluation of multiple aspects of the model response.
        Commonly used for translation tasks where each criterion represents
        a key point or quality aspect to evaluate.

        The method:
        1. Extracts the model response from the row
        2. Retrieves the list of criteria to evaluate
        3. Optionally includes reference text if configured
        4. For each criterion, formats a complete judge prompt with:
           - System prompt defining the judge's role
           - User prompt with the original task, model response, and criterion
        5. Generates unique custom IDs for tracking each criterion judgement

        Args:
            judge_prompt (dict): Dictionary containing judge prompt templates:
                - system_prompt: Instructions for the judge model
                - criteria_prompt: Template for formatting the evaluation request
                - reference_prompt: Optional template for including reference text
            row (pd.Series): Dataframe row containing:
                - responses: Model's response(s) to evaluate
                - criteria: List of criteria/key points to evaluate
                - prompts: Original task prompts
                - label: Reference answer (if using references)
                - id: Unique identifier for the example
            row_no (int): Row index in the dataframe.

        Returns:
            tuple[list[list], list[str]]: A tuple containing:
                - conversations: List of message lists, each containing system and
                  user messages for one criterion evaluation
                - custom_ids: List of custom IDs in format "<id>_criteria<n>"
                  where n is the criterion index

        Example:
            For a translation with 3 criteria, this generates 3 judge requests:
            - "example123_criteria0": Evaluate criterion 0
            - "example123_criteria1": Evaluate criterion 1
            - "example123_criteria2": Evaluate criterion 2
        """
        responses = self.metric.get_response(row)
        criteria = row["criteria"]

        conversations = []
        custom_ids = []
        # Generate judgements for each turn and baseline position

        for i, criterion in enumerate(criteria):
            if criterion.get("include_reference", False):
                reference_text_block = (
                    "\n"
                    + judge_prompt["reference_prompt"].format(reference=row["label"])
                    + "\n"
                )
            else:
                reference_text_block = ""

            messages = [
                {
                    "role": "system",
                    "content": judge_prompt["system_prompt"],
                },
                {
                    "role": "user",
                    "content": judge_prompt["criteria_prompt"].format(
                        prompt=self.task_config.config["languages"][self.lang][
                            "prompt_template"
                        ]["task_template"].format(**row["prompts"][0]),
                        translation=responses[0],
                        keyPointText=criterion["criterion"].format(**row["prompts"][0]),
                        reference_text_block=reference_text_block,
                    ),
                },
            ]
            # Add messages and custom IDs to lists
            conversations.append(messages)
            custom_ids.append(f"{row['id']}_criteria{i}")
        return conversations, custom_ids
