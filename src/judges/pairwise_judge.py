import os

import pandas as pd

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.judges.seahelm_judge import SeaHelmJudge
from src.metrics.seahelm_metric import SeaHelmMetric
from src.task_config import TaskConfig

logger = get_logger(__name__)


class PairwiseJudge(SeaHelmJudge):
    """Pairwise judge for comparing model responses against baseline models.

    This judge evaluates model responses by performing pairwise comparisons with
    baseline model outputs. It supports both with-reference and without-reference
    evaluation modes, multi-turn conversations, and position bias mitigation by
    swapping baseline positions.

    The judge generates judgements by:
    1. Comparing target model response with baseline model response for each turn
    2. Swapping baseline positions (before/after) to reduce position bias
    3. Using appropriate prompts based on whether reference answers are available
    4. Handling multi-turn conversations by building conversation context

    Required task_config fields:
    - baseline_model (str): Name of the baseline model to compare against
    - categories_with_reference (list): Categories requiring reference answers
    - judgement_labels (dict): Labels for judgement outcomes (must include "Tie" if ties_allowed)
    - judge (dict): Judge configuration with judge_prompts containing:
        - with-reference: Prompts for categories with reference answers
        - without-reference: Prompts for categories without reference answers

    Optional task_config fields:
    - ties_allowed (bool): Whether tie judgements are permitted. Defaults to True.
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
        """Initialize the PairwiseJudge.

        Args:
            judge_model: The judge model serving instance (e.g., OpenAIServing, VertexAIServing).
            judge_config (dict): The configuration dictionary for the judge model.
            dataloader (AbstractDataloader): Dataloader instance containing inference data.
            metric (SeaHelmMetric): Metric instance for calculating evaluation scores.
            task_config (dict): Task configuration dictionary containing judge settings.
            response_column (str, optional): Column name in dataloader containing model
                responses to be judged. Defaults to "response".
        Raises:
            ValueError: If ties_allowed is True but "Tie" is not in judgement_labels.
        """
        super().__init__(
            judge_model=judge_model,
            judge_config=judge_config,
            dataloader=dataloader,
            metric=metric,
            task_config=task_config,
            response_column=response_column,
        )

        self.baseline_model = task_config.config["baseline_model"]  # Required
        self.ties_allowed = task_config.config.get("ties_allowed", True)
        self.categories_with_reference = task_config.config[
            "categories_with_reference"
        ]  # Required
        self.judgement_labels = task_config.config["judgement_labels"]  # Required
        if self.ties_allowed and "Tie" not in self.judgement_labels:
            raise ValueError(
                "If ties are allowed, 'Tie' must be included in the judgement_labels."
            )

    def get_judgement_file_name(self) -> str:
        """Generate the filename for storing judge judgement results.

        The filename follows the pattern:
        <model>_<task>_<lang>_<baseline>_<judge>_judgement.jsonl

        Returns:
            str: Filename for the judgement results JSONL file.
        """
        return f"{os.path.basename(self.model_name)}_{self.task}_{self.lang}_{self.baseline_model}_{os.path.basename(self.judge_model_name)}_judgement.jsonl"

    def get_batch_file_name(self) -> str:
        """Generate the filename for storing batch API requests.

        The filename follows the pattern:
        <model>_<task>_<lang>_<baseline>_<judge>_batch.jsonl

        Returns:
            str: Filename for the batch requests JSONL file.
        """
        return f"{os.path.basename(self.model_name)}_{self.task}_{self.lang}_{self.baseline_model}_{os.path.basename(self.judge_model_name)}_batch.jsonl"

    def get_judge_prompts(self, row_no: int | None = None) -> dict:
        """
        Retrieves the judge prompts from the task configuration.
        """
        judge_config = self.task_config.config.get("judge", {})
        return judge_config["judge_prompts"]

    def prepare_judge_prompts(
        self, judge_prompt: dict, row: pd.Series, row_no: int
    ) -> tuple[list[list], list[str]]:
        """Prepare judge prompt conversations for pairwise comparison.

        Creates judge prompts for each turn and baseline position, building
        conversation context incrementally. Position bias is mitigated by
        generating judgements with baseline in both positions (before/after).

        The method:
        1. Determines if the task category requires reference answers
        2. Iterates through conversation turns
        3. For each turn, generates two prompts with swapped baseline positions
        4. Builds conversation context up to the current turn
        5. Formats prompts using system and user message templates

        Args:
            judge_prompt (dict): Dictionary containing "with-reference" and "without-reference"
                                prompt templates, each with system_prompt and prompt_template.
            row (pd.Series): Dataframe row containing:
                - responses: Target model's responses for each turn
                - baselines: Dict of baseline model responses
                - prompts: List of question prompts
                - references: Reference answers (if category uses references)
                - metadata: Dict with category information
                - question_id: Unique identifier for the question
            row_no (int): Row index in the dataframe.

        Returns:
            tuple[list[list], list[str]]: A tuple containing:
                - conversations: List of message lists, each with system and user messages
                - custom_ids: List of custom IDs in format "<question_id>_turn<n>_<position>"
                  where position is "baseline-before" or "baseline-after"

        Example:
            For a 2-turn conversation, this generates 4 judge requests:
            - turn1_baseline-before, turn1_baseline-after
            - turn2_baseline-before, turn2_baseline-after
        """
        responses = self.metric.get_response(row)
        baselines = row["baselines"][self.baseline_model]
        questions = row["prompts"]

        # Determine if this category requires reference answers
        is_with_ref = row["metadata"]["category"] in self.categories_with_reference

        # Select appropriate prompts based on reference requirement
        if is_with_ref:
            references = row["references"]
            prompts = judge_prompt["with-reference"]
        else:
            prompts = judge_prompt["without-reference"]

        conversations = []
        custom_ids = []
        # Generate judgements for each turn and baseline position
        for turn in range(len(responses)):
            for baseline_position in ["baseline-before", "baseline-after"]:
                info = {}

                # Build conversation context up to current turn
                for i in range(turn + 1):
                    info[f"question_{i + 1}"] = questions[i]["text"]

                    # Assign answers based on baseline position
                    info[f"answer_a_{i + 1}"] = (
                        responses[i]
                        if baseline_position == "baseline-after"
                        else baselines[i]
                    )

                    info[f"answer_b_{i + 1}"] = (
                        baselines[i]
                        if baseline_position == "baseline-after"
                        else responses[i]
                    )

                    # Add reference answer if required
                    if is_with_ref:
                        info[f"ref_answer_{i + 1}"] = references[i]

                # Construct judge prompt messages
                messages = [
                    {
                        "role": "system",
                        "content": prompts[turn]["system_prompt"].format(
                            **self.judgement_labels
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompts[turn]["prompt_template"].format(**info),
                    },
                ]

                # Add messages and custom IDs to lists
                conversations.append(messages)
                custom_ids.append(
                    f"{row['question_id']}_turn{i + 1}_{baseline_position}"
                )
        return conversations, custom_ids
