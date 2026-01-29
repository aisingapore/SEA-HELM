import base64
import functools
from abc import abstractmethod
from io import BytesIO
from multiprocessing import Pool
from typing import Any

import PIL
from tqdm import tqdm

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)


class HuggingFaceImageDataloader(AbstractDataloader):
    """Dataloader for HuggingFace image datasets."""

    def __init__(
        self,
        task_config: TaskConfig,
        default_num_in_context_examples: int,
        is_base_model: bool = False,
        model_name: str = "",
        run_base_path: str = "",
        inference_file_type: str = "jsonl",
        num_workers: int = 16,
        dropped_columns: list[str] = ["images"],  # noqa: B006
    ):
        """Initialize the HuggingFaceImageDataloader.

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

        self.num_workers = num_workers
        self.dropped_columns = dropped_columns

    @abstractmethod
    def load_dataset(self, limit: int | None = None) -> None:
        """Load the HuggingFace datasets.

        Loads the dataset for the specified language, optionally limiting the number
        of samples.

        Args:
            limit (int, optional): Optional limit on the number of instances to load. Defaults to None.

        Note:
            The dataset is stored in self.dataset after loading.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def load_example_dataset(self, limit: int | None = None) -> None:
        """Load the example dataset from a data source as a datasets.Dataset object.

        Args:
            limit (int, optional): Optional limit on the number of instances to load. Defaults to None.

        Note:
            The example dataset is stored in self.example_dataset after loading.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def prepare_conversations_for_inference(
        self, turn: int, fewshot_as_multiturn: bool = False
    ) -> list[Any]:
        """Prepare the conversations for inference by formatting prompts for the given turn.

        The formatted conversations should be in chat format compatible with the
        `apply_chat_template` method for prompt tokenization.

        Args:
            turn (int): Which turn to prepare the prompts for (0-indexed).
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

        _formatter = self.get_prompt_formatter(turn, fewshot_as_multiturn)

        with Pool(self.num_workers) as p:
            conversations = list(
                tqdm(p.imap(_formatter, self.dataset), total=len(self.dataset))
            )
        self.conversations = conversations

        return self.conversations

    def get_num_turns(self) -> int:
        """Get the number of turns in the dataset.

        Returns:
            int: Number of conversational turns in the task.
        """
        return 1

    @staticmethod
    def prompt_formatter(
        row: dict,
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
            conversations = update_conversations_fn(
                conversations, role, text_content, row["images"]
            )

        return conversations

    def get_prompt_formatter(
        self,
        turn: int,
        fewshot_as_multiturn: bool = False,
    ):
        """Return a picklable formatter callable for multiprocessing Pool.map.

        Args:
            turn (int): Which turn to prepare the prompts for (0-indexed).
            fewshot_as_multiturn (bool, optional): Whether to format examples as multi-turn dialogue. Defaults to False.

        Returns:
            Callable: The prompt formatter for the given turn.
        """
        return functools.partial(
            self.prompt_formatter,
            turn=turn,
            specific_task_config=self.specific_task_config,
            fewshot_as_multiturn=fewshot_as_multiturn,
            update_conversations_fn=self.update_conversation,
            generate_formatted_conversation_fn=self.generate_formatted_conversation,
        )

    @staticmethod
    def update_conversation(
        conversations: list,
        role: str,
        text_content: str,
        images: list[str] | None = None,
    ):
        """Update a conversation with a new message containing text and images.

        Appends a new conversation turn with the specified role, text content,
        and associated images in the multimodal format expected by
        vision-language models.

        Args:
            conversations (list): List of existing conversation turns.
            role (str): Speaker role ('user' or 'assistant').
            text_content (str): Text content for the message.
            images (list[str] | None): List of image bytes to include with the message. If None, no images will be included. Defaults to None.

        Returns:
            list: Updated conversations list with the new turn appended.
        """
        content = [{"type": "text", "text": text_content}]
        if role == "user" and images is not None:
            for image_bytes in images:
                if isinstance(image_bytes, str) or isinstance(image_bytes, bytes):
                    img_str = base64.b64encode(image_bytes).decode("utf-8")
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                        }
                    )
                elif isinstance(image_bytes, PIL.Image.Image):
                    if image_bytes.mode == "RGBA":
                        image_bytes = image_bytes.convert("RGB")

                    buffered = BytesIO()
                    image_bytes.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                        }
                    )

        conversations.append(
            {
                "role": role,
                "content": content,
            }
        )
        return conversations

    def prepare_inference_df_for_writing(self):
        """Prepare the inference DataFrame for writing.

        Drops image columns and replaces image bytes in conversations with
        placeholder text to reduce file size.
        """
        self.inference_df = self.inference_df.drop(
            columns=self.dropped_columns, errors="ignore"
        )

        def use_placeholder_image_for_conversations(conversation, placeholder="IMAGE"):
            for c in conversation:
                for t in c["content"]:
                    if t["type"] == "image_url":
                        t["image_url"] = placeholder

            return conversation

        self.inference_df["conversations"] = self.inference_df["conversations"].map(
            use_placeholder_image_for_conversations,
        )
