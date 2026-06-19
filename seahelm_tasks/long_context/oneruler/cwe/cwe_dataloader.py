import math
import random
from typing import Any, Callable

import wonderwords
from datasets import Dataset
from transformers import AutoTokenizer

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)


class CWEDataloader(AbstractDataloader):
    """Dataloader for the Common Word Extraction (CWE) task.

    This task evaluates the model's ability to identify common words from a long list of words
    within a specified token budget. It generates synthetic data where a set of common words
    are repeated multiple times while other words appear less frequently.
    """

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
        """Initialize the CWEDataloader.

        Args:
            task_config: TaskConfig object containing task-specific settings.
            default_num_in_context_examples: Default number of few-shot examples to use.
            is_base_model: Whether this is a base model (vs instruction-tuned).
            model_name: Name/path of the model being evaluated.
            run_base_path: Base path for storing inference results.
            inference_file_type: File format for inference results ('jsonl' or 'csv').
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # get context size
        self.context_size = self.task_config.config.context_size
        # get num tokens in prompt
        empty_prompt = (
            self.specific_task_config.prompt_template["preamble"]
            + self.specific_task_config.prompt_template["task_template"]
        )
        self.prompt_token_count = self.tokenizer(
            empty_prompt,
            return_length=True,
            add_special_tokens=False,
        )["length"][0]

        self.num_samples = self.task_config.config.num_samples
        self.random_seed = self.task_config.config.random_seed
        self.num_common_words = self.task_config.config.num_common_words
        self.num_repeats_common_words = self.task_config.config.num_repeats_common_words
        self.num_repeats_uncommon_words = (
            self.task_config.config.num_repeats_uncommon_words
        )

    def load_dataset(self, limit: int = None):
        """Generates the CWE dataset.

        Args:
            limit (int, optional): Optional limit on the number of instances to load.
        """
        num_samples = self.num_samples
        if limit is not None:
            if limit <= self.num_samples:
                logger.info("Using limit of %d samples as specified.", limit)
                num_samples = limit
            else:
                logger.info(
                    "Requested limit (%d) is greater than the configured number of samples (%d). Using configured number of samples.",
                    limit,
                    self.num_samples,
                )

        if self.lang == "en":
            nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
            adjs = wonderwords.random_word._get_words_from_text_file(
                "adjectivelist.txt"
            )
            verbs = wonderwords.random_word._get_words_from_text_file("verblist.txt")
            self.words = nouns + adjs + verbs
        else:
            filepath = (
                f"seahelm_tasks/long_context/oneruler/cwe/vocab/{self.lang}_words.txt"
            )
            with open(filepath, "r") as file:
                # add space to word as it affects the tokenization
                self.words = [line.strip() for line in file.readlines()]

        self.words = sorted(set(self.words))
        self.words = [" " + word for word in self.words]
        self.token_counts = self.tokenizer.batch_encode_plus(
            self.words, return_length=True, add_special_tokens=False
        )["length"]
        self.average_token_count = sum(self.token_counts) / len(self.token_counts)

        seeds = random.Random(self.random_seed).sample(range(2**31 - 1), num_samples)

        data = Dataset.from_dict(
            {
                "seed": seeds,
            }
        )
        data = data.map(
            lambda row: self.generate_single_instance(seed=row["seed"]), num_proc=32
        )
        self.dataset = data

    def estimate_num_words(self, context_size: int, num_words: int = 0) -> int:
        """Estimates the number of words that can fit within the token budget.

        Args:
            num_words (int, optional): Initial number of words. Defaults to 0.
            context_size (int | None, optional): The target token budget. If None, uses self.context_size. Defaults to None.

        Returns:
            int: Estimated number of words.
        """
        power = math.floor(math.log10(num_words)) if num_words > 0 else 0

        while True:
            random_value = random.randint(10**power, 10 ** (power + 1) - 1)
            list_tokens = self.tokenizer(
                f" {random_value}.",  # checks for num tokens for the period and the space after each word
                return_length=True,
                add_special_tokens=False,
            )["length"][0]

            additional_words = 10 ** (power + 1) - 10**power
            additional_tokens = additional_words * (
                list_tokens + self.average_token_count
            )
            if additional_tokens > context_size:
                additional_words = context_size // (
                    list_tokens + self.average_token_count
                )
                num_words += additional_words
                break
            else:
                num_words += additional_words
                context_size -= additional_tokens
                power += 1

        return int(num_words)

    def generate_single_instance(
        self, seed: int = 0, deviation: int = 100
    ) -> dict[str, Any]:
        """Generates a single CWE instance.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
            deviation (int, optional): Acceptable deviation from token budget. Defaults to 100.

        Returns:
            dict[str, Any]: A dictionary containing the generated prompt, label, and metadata.
        """
        word_list_copy = self.words.copy()

        # select common words
        random.Random(seed).shuffle(word_list_copy)
        common_words = word_list_copy[: self.num_common_words]
        uncommon_words = word_list_copy[self.num_common_words :]

        # adjust token budget for tokens used in common words
        common_words_average_token = sum(
            self.tokenizer.batch_encode_plus(
                common_words, return_length=True, add_special_tokens=False
            )["length"]
        ) / len(common_words)
        num_words = self.estimate_num_words(
            context_size=self.context_size
            - self.prompt_token_count
            - self.num_common_words
            * self.num_repeats_common_words
            * (common_words_average_token - self.average_token_count)
        )

        uncommon_words = uncommon_words * self.num_repeats_uncommon_words
        random.Random(seed).shuffle(uncommon_words)

        _common_words = common_words * self.num_repeats_common_words

        # generate final word list
        retry_count = 0
        while retry_count < 3:
            assert num_words >= len(_common_words), (
                "Context size too small to fit the common words with required repetitions."
            )
            assert num_words - len(_common_words) <= len(uncommon_words), (
                "Not enough uncommon words to fill the remaining slots."
            )
            _uncommon_words = uncommon_words[: num_words - len(_common_words)]
            words = _common_words + _uncommon_words
            random.Random(seed).shuffle(words)

            word_list = [f"{i + 1}.{word}" for i, word in enumerate(words)]
            word_str = " ".join(word_list)

            # check token length
            tokenized = self.tokenizer(word_str)
            num_tokens = len(tokenized["input_ids"])
            difference = self.context_size - self.prompt_token_count - num_tokens
            if abs(difference) < deviation:
                break
            else:
                logger.warning(
                    "Generated instance exceeds acceptable tolerance: %d tokens (expected range: %.1f - %.1f)",
                    num_tokens + self.prompt_token_count,
                    self.context_size - deviation,
                    self.context_size + deviation,
                )
                logger.warning("Regenerating instance (retry %d)...", retry_count + 1)
                num_words = self.estimate_num_words(
                    context_size=difference
                    + self.context_size
                    - self.prompt_token_count
                    - self.num_common_words
                    * self.num_repeats_common_words
                    * (common_words_average_token - self.average_token_count)
                )
                retry_count += 1

        return {
            "prompts": [{"list": word_str, "num_common_words": self.num_common_words}],
            "label": [word.strip() for word in common_words],
            "metadata": {
                "num_tokens": num_tokens + self.prompt_token_count,
            },
        }

    def prepare_conversations_for_inference(
        self, turn: int, fewshot_as_multiturn: bool = False
    ) -> list[Any]:
        """Prepare the conversations for inference by formatting prompts for the given turn.

        Args:
            turn (int): Which turn to prepare the prompts for (1-indexed).
            fewshot_as_multiturn (bool, optional): Whether to treat few-shot examples as multi-turn dialogues. Defaults to False.

        Returns:
            list[Any]: The formatted conversations.
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
            num_proc=self.num_workers,
        )

        return self.dataset["conversations"]

    def get_num_turns(self) -> int:
        """Get the number of turns in the dataset.

        Returns:
            int: Number of conversational turns in the task.

        Note:
            For single-turn tasks, this should return 1.
            For multi-turn tasks, this should return the number of exchanges between the user and the assistant.
        """
        return 1

    def get_prompt_formatter(
        self,
        turn: int,
        fewshot_as_multiturn: bool = False,
    ) -> Callable:
        """Get the prompt formatter for the given turn.

        Args:
            turn (int): Which turn to prepare the prompts for (1-indexed).
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
