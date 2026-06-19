import glob
import random
import re
import string
from typing import Any, Callable

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.task_config import TaskConfig

logger = get_logger(__name__)

_CLOSING_PUNCTUATIONS = r"'\"”’』」｣》⟩)]}）］｝»"
_OTHER_QUOTES_AND_BRACKETS = r"“”‘’『』「」｢｣《》（）［］｛｝«»"
_OTHER_PUNCTUATIONS = r"｡。，、､…：；？！။"


class NIAHDataloader(AbstractDataloader):
    """Dataloader for the Needle-In-A-Haystack (NIAH) task."""

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
        """Initialize the NIAHDataloader.

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

        self.random_seed = self.task_config.config.random_seed
        self.num_samples = self.task_config.config.num_samples
        self.min_num_keys = self.task_config.config.min_num_keys
        self.num_values = self.task_config.config.num_values
        self.num_queries = self.task_config.config.num_queries

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

        # load vocab_list
        vocab_df = pd.read_csv(
            "seahelm_tasks/long_context/oneruler/niah/vocab/100_noun_list_translated.tsv",
            sep="\t",
        )
        self.vocab_list = vocab_df[self.lang].tolist()
        self.vocab_list = [word.strip() for word in self.vocab_list]

        self.empty_needle = {
            "en": 'The special magic number for "{needle}" is: {value}. ',
            "zh": "“{needle}”的魔法数字是：{value}。",
            "ta": '"{needle}" என்ற சொல்லிற்கான சிறப்பு மந்திரச்சொல்: {value}. ',
            "vi": 'Con số ma thuật đặc biệt cho "{needle}" là: {value}. ',
        }[self.lang]

        self.and_word = {
            "en": ", and ",
            "zh": "和",
            "ta": ", மற்றும் ",
            "vi": ", và ",
        }[self.lang]

        # get num tokens in empty needle
        empty_needle_text = self.empty_needle.format(
            needle=random.Random(self.random_seed).choice(self.vocab_list),
            value=123456789,
        )
        self.empty_needle_token_count = self.tokenizer(
            empty_needle_text,
            return_length=True,
            add_special_tokens=False,
        )["length"][0]

    def split_into_paras(self, text):
        """Splits a text into paras"""
        paras = re.split("[\r|\n]", text)

        # remove empty paragraphs and those without any text characters
        paras = [
            para
            for para in paras
            if (
                para.strip(
                    string.whitespace
                    + string.punctuation
                    + _OTHER_PUNCTUATIONS
                    + _OTHER_QUOTES_AND_BRACKETS
                )
                != ""
            )
        ]

        return paras

    def split_into_sentences(
        self,
        text,
        language,
        split_at_colons: bool = False,
        split_at_semi_colons: bool = False,
    ) -> list:
        """Split the text into sentences."""
        # TODO Need to check implementation for the Thai language
        paragraphs = self.split_into_paras(text)
        sentence_endings = {
            "default": r".!?"
            + (r":" if split_at_colons else "")
            + (r";" if split_at_semi_colons else ""),
            # include the half width Chinese full stop
            "zh": r"｡。！？"
            + (r"：" if split_at_colons else "")
            + (r"；" if split_at_semi_colons else ""),
            "my": r"။.!?"
            + (r":" if split_at_colons else "")
            + (
                r";" if split_at_semi_colons else ""
            ),  # add burmese full stop while still keeping the other languages
        }

        if language in sentence_endings:
            endings = sentence_endings[language]
        else:
            endings = sentence_endings["default"]

        sentences = []
        # TODO This would fail when an ending is in quotes such as:
        # "It's so dreadful to be poor!" sighed Meg, looking down at her old dress.
        abbreviations = "".join(
            [
                r"(?<!\b[a-z]\.)",
                r"(?<!\b(?:Mr|Dr|Ms|Sr|Jr|Co|vs|Lt|St|Rd)\.)",
                r"(?<!\b(?:i\.e|e\.g|a\.m|p\.m|M\.A|Mrs|Inc|Ltd|etc|Rev|Hon|Gen|Sgt|Rep|Sen|Ave|Col)\.)",
                r"(?<!\b(?:Ph\.D|Prof|Capt|Blvd|Corp)\.)",
            ]
        )
        rexp = re.compile(
            rf"({abbreviations}(?<![\d])[{endings}]+[{re.escape(_CLOSING_PUNCTUATIONS)}\*_]*)(\s+)"
        )
        delim = r"<-split->"

        for para in paragraphs:
            para_sentences = rexp.sub(r"\1" + delim, para).split(delim)

            para_sentences = [
                s.strip()
                for s in para_sentences
                if s.strip(
                    string.whitespace
                    + string.punctuation
                    + _OTHER_PUNCTUATIONS
                    + _OTHER_QUOTES_AND_BRACKETS
                )
                != ""
            ]
            sentences.extend(para_sentences)

        return sentences

    def load_dataset(self, limit: int = None):
        """Generates the NIAH dataset.

        Args:
            limit (int, optional): Optional limit on the number of instances to load.
        """
        num_samples = self.num_samples
        if limit is not None:
            if limit <= num_samples:
                logger.info("Using limit of %d samples as specified.", limit)
                num_samples = limit
            else:
                logger.info(
                    "Requested limit (%d) is greater than the configured number of samples (%d). Using configured number of samples.",
                    limit,
                    self.num_samples,
                )

        # load sentences
        filepaths = glob.glob(
            f"seahelm_tasks/long_context/oneruler/niah/books/{self.lang}/*.txt"
        )

        books = []
        for filepath in filepaths:
            logger.info("Loading sentences from %s...", filepath)
            book = open(filepath, "r", encoding="utf-8").read()
            books.append(book)

        self.books = "\n\n".join(books)

        # replace multiple spaces with single space
        self.books = re.sub(r"[ \t]+", " ", self.books)

        # split to sentences
        tokens = self.tokenizer(self.books, add_special_tokens=False)["input_ids"]
        if (
            len(tokens)
            < self.context_size
            - self.prompt_token_count
            - self.empty_needle_token_count * self.min_num_keys
        ):
            diff = (
                self.context_size
                - self.prompt_token_count
                - self.empty_needle_token_count * self.min_num_keys
            ) // len(tokens) + 1

            logger.warning(
                "Book length (%d tokens) is less than the required context size (%d tokens). Duplicating the book %d times to reach the desired context.",
                len(tokens),
                self.context_size
                - self.prompt_token_count
                - self.empty_needle_token_count * self.min_num_keys,
                diff,
            )
            tokens = self.tokenizer(self.books * diff, add_special_tokens=False)[
                "input_ids"
            ]

        tokens = tokens[
            : self.context_size
            - self.prompt_token_count
            - self.empty_needle_token_count * self.min_num_keys
        ]
        self.text = self.tokenizer.decode(tokens)
        self.sentences = self.split_into_sentences(self.text, language=self.lang)

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

    def generate_single_instance(self, seed: int = 0) -> dict[str, Any]:
        """Generates a single CWE instance.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to 0.

        Returns:
            dict[str, Any]: A dictionary containing the generated prompt, label, and metadata.
        """
        # use sentence as insertion point. Key is always inserted before the sentence.
        labels = []
        needles = []

        _text = self.text
        rng = random.Random(seed)

        needle_list = self.vocab_list.copy()
        assert not (self.num_values is not None and self.num_queries is not None), (
            "num_values and num_queries cannot both be set."
        )
        # handle num_queries
        if self.num_queries is not None:
            assert self.num_queries <= self.min_num_keys, (
                "num_queries cannot be greater than min_num_keys."
            )

            for _ in range(self.num_queries):
                value = str(rng.randint(0, 2**31 - 1))
                needle = rng.choice(needle_list)
                needle_list.remove(needle)
                insertion_point = rng.choice(self.sentences)
                _text = _text.replace(
                    insertion_point,
                    self.empty_needle.format(needle=needle, value=value)
                    + insertion_point,
                    1,
                )
                labels.append(value)
                needles.append(needle)

            for _ in range(self.num_queries, self.min_num_keys):
                value = str(rng.randint(0, 2**31 - 1))
                needle = rng.choice(needle_list)
                needle_list.remove(needle)
                insertion_point = rng.choice(self.sentences)
                _text = _text.replace(
                    insertion_point,
                    self.empty_needle.format(needle=needle, value=value)
                    + insertion_point,
                    1,
                )
        # handle num_values
        elif self.num_values is not None:
            assert self.num_values <= self.min_num_keys, (
                "num_values cannot be greater than min_num_keys."
            )
            needle = rng.choice(needle_list)
            needle_list.remove(needle)
            if self.num_values == 0:
                labels.append("none")
                needles.append(needle)
            else:
                needles.append(needle)
                for _ in range(self.num_values):
                    value = str(rng.randint(0, 2**31 - 1))
                    insertion_point = rng.choice(self.sentences)
                    _text = _text.replace(
                        insertion_point,
                        self.empty_needle.format(needle=needle, value=value)
                        + insertion_point,
                        1,
                    )
                    labels.append(value)

            for _ in range(self.num_values, self.min_num_keys):
                value = str(rng.randint(0, 2**31 - 1))
                needle = rng.choice(needle_list)
                needle_list.remove(needle)
                insertion_point = rng.choice(self.sentences)
                _text = _text.replace(
                    insertion_point,
                    self.empty_needle.format(needle=needle, value=value)
                    + insertion_point,
                    1,
                )

        else:
            raise ValueError("Either num_values or num_queries must be set.")

        if self.lang in ["zh"]:
            quotes = "“{needle}”"
        else:
            quotes = '"{needle}"'

        if len(needles) == 1:
            needle_text = quotes.format(needle=needles[0])
        elif len(needles) == 2:
            needle_text = (
                quotes.format(needle=needles[0])
                + self.and_word
                + quotes.format(needle=needles[1])
            )
        else:
            comma = "，" if self.lang in ["zh"] else ", "
            needle_text = (
                comma.join(quotes.format(needle=needle) for needle in needles[:-1])
                + self.and_word
                + quotes.format(needle=needles[-1])
            )

        return {
            "prompts": [{"text": _text, "needle_text": needle_text}],
            "label": labels,
            "metadata": {
                "num_queries": self.num_queries,
                "num_values": self.num_values,
                "num_keys": self.min_num_keys,
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
