from typing import Any, Callable

import datasets

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)


class SeaHelmLocalDataloader(AbstractDataloader):
    """Dataloader for SEA-HELM local datasets."""

    def __init__(
        self,
        task_config: TaskConfig,
        default_num_in_context_examples: int,
        is_base_model: bool = False,
        model_name: str = "",
        run_base_path: str = "",
        inference_file_type: str = "jsonl",
    ):
        """Initialize the SeaHelmLocalDataloader.

        Args:
            task_config: TaskConfig object containing task-specific settings.
            default_num_in_context_examples (int): Default number of few-shot examples to use.
            is_base_model (bool): Whether this is a base model (vs instruction-tuned).
            model_name (str, optional): Name/path of the model being evaluated. Defaults to "".
            run_base_path (str, optional): Base path for storing inference results. Defaults to "".
            inference_file_type (str, optional): File format for inference results ('jsonl' or 'csv'). Defaults to "jsonl".
        """
        super().__init__(
            task_config,
            default_num_in_context_examples,
            is_base_model=is_base_model,
            model_name=model_name,
            run_base_path=run_base_path,
            inference_file_type=inference_file_type,
        )

    def load_dataset(self, limit: int | None = None) -> None:
        """Load the dataset from a data source as a datasets.Dataset object.

        Args:
            limit (int, optional): Optional limit on the number of instances to load. Defaults to None.
        """
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            pass
        else:
            filepath = self.specific_task_config["filepath"]

            logger.info("Drawing and preparing instances from %s", filepath)

            self.dataset = datasets.load_dataset(
                "json", split="train", data_files=filepath
            )
            if limit is not None:
                self.dataset = self.dataset.select(range(limit))

    def load_example_dataset(self, limit: int | None = None):
        """Load the example dataset from a data source as a datasets.Dataset object.

        Returns:
            datasets.Dataset: The loaded example dataset.
        """
        if self.example_dataset:
            logger.info("Example dataset already loaded, skipping loading process")
            pass
        else:
            example_filepath = self.specific_task_config["example_filepath"]
            logger.info("Drawing and preparing examples from %s", example_filepath)

            self.example_dataset = datasets.load_dataset(
                "json", split="train", data_files=example_filepath
            )

            if limit is not None:
                if len(self.example_dataset) < limit:
                    logger.warning(
                        "Not enough examples! Expected %d examples but only received %d.",
                        limit,
                        len(self.example_dataset),
                    )
                    limit = len(self.example_dataset)
                self.example_dataset = self.example_dataset.select(range(limit))

            # check if label is of type list and convert it to string
            if isinstance(self.example_dataset.features["label"], datasets.Sequence):
                self.example_dataset = self.example_dataset.map(
                    lambda x: {"label": x["label"][0]}, num_proc=16
                )

    def prepare_conversations_for_inference(
        self, turn: int, fewshot_as_multiturn: bool = False
    ) -> list[Any]:
        """Prepare the conversations for inference by formatting prompts for the given turn.

        Args:
            turn (int): Which turn to prepare the prompts for (1-indexed).
            fewshot_as_multiturn (bool, optional): Whether to treat few-shot examples as multi-turn dialogues. Defaults to False.

        Returns:
            list[Any]: The formatted conversations stored in self.conversations.
        """
        logger.info(
            "Performing inference for task '%s', turn %d with %d examples",
            self.task_name.upper(),
            turn,
            self.fewshot_num_examples,
        )

        if self.fewshot_num_examples > 0:
            if self.specific_task_config["example_filepath"]:
                self.load_example_dataset(limit=self.fewshot_num_examples)
            else:
                logger.warning(
                    "Example filepath not found! Reverting back to 0-shot instead of %d-shot.",
                    self.fewshot_num_examples,
                )

        self.dataset = self.dataset.map(
            self.get_prompt_formatter(turn, fewshot_as_multiturn),
            num_proc=16,
        )
        self.conversations = self.dataset["conversations"]
        return self.conversations

    def get_num_turns(self) -> int:
        """Get the number of turns in the dataset.

        Returns:
            int: Number of conversational turns in the task.

        Note:
            For single-turn tasks, this should return 1.
            For multi-turn tasks, this should return the number of exchanges between the user and the assistant.
        """
        return len(self.dataset[0]["prompts"])

    def get_prompt_formatter(
        self,
        turn: int,
        fewshot_as_multiturn: bool = False,
    ) -> Callable:
        """Get the prompt formatter for the given turn.

        Args:
            turn (int): Which turn to prepare the prompts for (1-indexed).
            num_examples (int): Number of few-shot examples to include.
            fewshot_as_multiturn (bool, optional): Whether to format examples as multi-turn dialogue. Defaults to False.

        Returns:
            Callable: The prompt formatter for the given turn.
        """

        def _prompt_formatter(row):
            if turn == 1:
                conversations = []
            else:
                conversations = row["conversations"]
                conversations = self.update_conversation(
                    conversations, "assistant", row["responses"][turn - 2]
                )

            values = row["prompts"][turn - 1]

            roles, contents = self.generate_formatted_conversation(
                self.specific_task_config,
                values,
                fewshot_as_multiturn=fewshot_as_multiturn,
            )
            for role, content in zip(roles, contents, strict=True):
                conversations = self.update_conversation(conversations, role, content)

            row["conversations"] = conversations
            return row

        return _prompt_formatter

    def update_conversation(
        self, conversations: list[dict], role: str, content: str
    ) -> list[dict]:
        """Update the conversation with the given role and content.

        Args:
            conversations (list[dict]): The conversation to update.
            role (str): The role to update the conversation with.
            content (str): The content to update the conversation with.

        Returns:
            list[dict]: The updated conversation.
        """
        conversations.append({"role": role, "content": content})
        return conversations
