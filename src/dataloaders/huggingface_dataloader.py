import datasets

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)


class HuggingFaceDataloader(AbstractDataloader):
    """Dataloader for HuggingFace datasets."""

    def __init__(
        self,
        task_config: TaskConfig,
        default_num_in_context_examples: int,
        is_base_model: bool = False,
        model_name: str = "",
        run_base_path: str = "",
        inference_file_type: str = "jsonl",
        num_workers: int = 16,
    ):
        """Initialize the HuggingFaceDataloader.

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
            num_workers=num_workers,
        )

    def load_dataset(self, limit: int | None = None) -> None:
        """Load the HuggingFace datasets.

        Loads the dataset for the specified language, optionally limiting the number
        of samples.

        Args:
            limit (int, optional): Optional limit on the number of instances to load. Defaults to None.

        Note:
            The dataset is stored in self.dataset after loading.
        """
        if self.dataset:
            logger.info("Dataset already loaded, skipping loading process")
            pass
        else:
            filepath = self.specific_task_config["filepath"]

            logger.info("Drawing and preparing instances from %s", filepath)

            self.dataset = datasets.load_dataset(filepath, self.lang, split="eval")
            if limit is not None:
                self.dataset = self.dataset.select(range(limit))

    def load_example_dataset(self, limit: int | None = None) -> None:
        """Load the example dataset from a data source as a datasets.Dataset object.

        Args:
            limit (int, optional): Optional limit on the number of instances to load. Defaults to None.
        """
        if self.example_dataset:
            logger.info("Example dataset already loaded, skipping loading process")
            pass
        else:
            example_filepath = self.specific_task_config["example_filepath"]
            logger.info("Drawing and preparing examples from %s", example_filepath)

            self.example_dataset = datasets.load_dataset(
                example_filepath, self.lang, split="examples"
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
                    lambda x: {"label": x["label"][0]}, num_proc=self.num_workers
                )

    def get_num_turns(self) -> int:
        """Get the number of turns in the dataset.

        Returns:
            int: Number of conversational turns in the task.
        """
        return len(self.dataset[0]["prompts"])

    @staticmethod
    def prompt_formatter(
        row: dict,
        idx: int,
        *,
        turn: int,
        specific_task_config: dict,
        fewshot_as_multiturn: bool = False,
        update_conversations_fn=None,
        generate_formatted_conversation_fn=None,
    ) -> list:
        """Static, picklable row formatter for multiprocessing.

        Returns the updated conversations list for a given dataset row.
        """
        if turn == 1:
            conversations = []
        else:
            conversations = row["conversations"]
            conversations = update_conversations_fn(
                conversations, "assistant", row["responses"][turn - 2]
            )

        values = row["prompts"][turn - 1]

        roles, text_contents = generate_formatted_conversation_fn(
            specific_task_config,
            values,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )
        for role, text_content in zip(roles, text_contents, strict=True):
            conversations = update_conversations_fn(conversations, role, text_content)

        return conversations

    @staticmethod
    def update_conversation(
        conversations: list[dict], role: str, content: str
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
