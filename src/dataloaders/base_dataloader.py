import functools
import os
from abc import abstractmethod
from multiprocessing import Pool
from typing import Any

import pandas as pd
import ujson
from datasets import Dataset
from tqdm import tqdm

from src.base_logger import get_logger
from src.task_config import TaskConfig

logger = get_logger(__name__)


class AbstractDataloader:
    """Abstract base class for data loaders in the SEA-HELM evaluation framework.

    This class provides a common interface and shared functionality for loading,
    preparing, and writing out datasets for all inference tasks.
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
        """Initialize the AbstractDataloader.

        Args:
            task_name: Name of the evaluation task.
            task_config: Configuration dictionary containing task-specific settings.
            lang: Language code for the task (e.g., 'en', 'zh', 'ms').
            default_num_in_context_examples: Default number of few-shot examples to use.
            is_base_model: Whether this is a base model (vs instruction-tuned).
            model_name: Name/path of the model being evaluated.
            run_base_path: Base path for storing inference results.
            inference_file_type: File format for inference results ('jsonl' or 'csv').
        """
        self.task_config = task_config
        self.task_name = task_config.task_name
        self.lang = task_config.lang
        self.specific_task_config = task_config.config["languages"][self.lang]

        self.is_base_model = is_base_model

        self.model_name = model_name
        self.run_base_path = run_base_path
        self.inference_file_type = inference_file_type

        if "fewshot_num_examples" in task_config.config:
            base_or_instruct = "base" if self.is_base_model else "instruct"
            fewshot_num_examples = task_config.config["fewshot_num_examples"][
                base_or_instruct
            ]
        else:
            fewshot_num_examples = default_num_in_context_examples
            logger.warning(
                "No fewshot_num_examples found in task config. Using default value of %d.",
                default_num_in_context_examples,
            )
        self.fewshot_num_examples = fewshot_num_examples

        self.dataset = None
        self.example_dataset = None
        self.dataframe = None
        self.num_workers = num_workers

    def get_parent_folder(self) -> str:
        """Generate the parent folder path for storing inference results.

        Args:
            task_name (str): Name of the evaluation task.
            lang (str): Language code for the task.
        """
        if "aggregation_group" in self.task_config.config:
            task = self.task_config.config["aggregation_group"]
        else:
            task = self.task_name

        parent_path = os.path.join(self.run_base_path, "inferences", self.lang, task)
        os.makedirs(parent_path, exist_ok=True)
        return parent_path

    def get_filepath(
        self, parent_path: str, parts: list[str], file_type: str = "jsonl"
    ) -> str:
        """Generate the file path for batch processing files.

        Args:
            parent_path (str): The base path for storing run files.
            model_name (str): The name of the model being used.
            parts (list[str]): List of parts to include in the file name.
            file_type (str, optional): The file extension to use. Defaults to "jsonl".

        Returns:
            str: The generated file path for the batch file.
        """
        return os.path.join(parent_path, "_".join(parts) + f".{file_type}")

    def get_inference_filepath(self, file_type: str = "jsonl") -> str:
        """Generate the file path for storing inference results.

        Args:
            file_type (str): File format ('jsonl' or 'csv').

        Returns:
            str: Full file path for storing inference results.
        """
        parent_path = self.get_parent_folder()

        return self.get_filepath(
            parent_path=parent_path,
            parts=[os.path.basename(self.model_name), self.task_name, self.lang],
            file_type=file_type,
        )

    def load_cached_inference_results(self) -> bool:
        """Load cached inference results if available.

        Returns:
            bool: True if cached inference results were loaded, False otherwise.
        """
        inference_filepath = self.get_inference_filepath(
            file_type=self.inference_file_type
        )
        if os.path.exists(inference_filepath):
            if self.inference_file_type == "jsonl":
                self.dataframe = pd.read_json(inference_filepath, lines=True)
                return True
            elif self.inference_file_type == "csv":
                self.dataframe = pd.read_csv(inference_filepath)
                return True
        return False

    ### Filepath generation functions
    def get_filepath_creator(self, suffix):
        """Return a filepath-generating function bound to a given filename suffix.

        Args:
            suffix (str): Suffix appended to the filename (e.g., 'batch', 'batch_response').

        Returns:
            Callable[[int, str], str]: A function that accepts a turn number and file type
                and returns the corresponding file path.
        """

        def _get_filepath(turn: int, file_type: str = "jsonl") -> str:
            parent_path = self.get_parent_folder()

            return self.get_filepath(
                parent_path=parent_path,
                parts=[
                    os.path.basename(self.model_name),
                    self.task_name,
                    self.lang,
                    f"turn{turn}",
                    suffix,
                ],
                file_type=file_type,
            )

        return _get_filepath

    def get_batch_filepath(self, turn: int = 1, file_type: str = "jsonl") -> str:
        """Generate the file path for storing conversations batch for a specific turn.

        Args:
            turn (int): The conversational turn number (1-indexed).
            file_type (str): File format ('jsonl' or 'csv').

        Returns:
            str: Full file path for storing conversational batches for the specified turn.
        """
        return self.get_filepath_creator("batch")(turn, file_type)

    def get_batch_response_filepath(
        self, turn: int = 1, file_type: str = "jsonl"
    ) -> str:
        """Generate the file path for storing batch responses for a specific turn.

        Args:
            turn (int): The conversational turn number (1-indexed).
            file_type (str): File format ('jsonl' or 'csv').

        Returns:
            str: Full file path for storing batch responses for the specified turn.
        """
        return self.get_filepath_creator("batch_response")(turn, file_type)

    @abstractmethod
    def load_dataset(self, limit: int | None = None) -> Dataset:
        """Load the dataset from a data source as a datasets.Dataset object.

        The dataset is stored in self.dataset. Subclasses must implement this method
        to load their specific dataset format.

        Args:
            limit (int, optional): Optional limit on the number of instances to load. Defaults to None.

        Returns:
            datasets.Dataset: The loaded dataset.

        Note:
            The loaded dataset should be compatible with the HuggingFace datasets library.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def load_example_dataset(self) -> None:
        """Load the example dataset from a data source as a datasets.Dataset object.

        Note:
            Subclasses must implement this method to load their specific example dataset format.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_num_turns(self) -> int:
        """Get the number of turns in the dataset.

        Returns:
            int: Number of conversational turns in the task.

        Note:
            For single-turn tasks, this should return 1.
            For multi-turn tasks, this should return the number of exchanges between the user and the assistant.
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
            list[Any]: The formatted conversations.
        """
        logger.info(
            "Preparing conversations for task '%s', turn %d with %d examples",
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
        _formatter = self.get_prompt_formatter(turn, fewshot_as_multiturn)

        with Pool(self.num_workers) as p:
            conversations = list(
                tqdm(
                    p.starmap(
                        _formatter,
                        zip(self.dataset, range(len(self.dataset)), strict=True),
                    ),
                    total=len(self.dataset),
                )
            )

        if "conversations" in self.dataset.column_names:
            self.dataset = self.dataset.remove_columns("conversations")
        self.dataset = self.dataset.add_column("conversations", conversations)

        return conversations

    def prepare_labels_for_inference(self):
        """Extract ground-truth labels from the dataset.

        Returns:
            list | None: The list of labels from the dataset's 'label' column,
                or None if the column does not exist.
        """
        if "label" not in self.dataset.column_names:
            logger.warning(
                "No 'label' column found in dataset. Labels will be set to None for inference."
            )
            return None
        else:
            return self.dataset["label"]

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

    def generate_formatted_conversation(
        self,
        specific_task_config: dict,
        values: dict,
        fewshot_as_multiturn: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Generate a formatted conversation with few-shot examples and task prompt.

        Creates a structured conversation format with roles and contents,
        including in-context examples and the main task prompt.

        Args:
            specific_task_config (dict): Language-specific task configuration containing
                templates and example file paths.
            values (dict): Dictionary of values to format into the task template.
            fewshot_as_multiturn (bool): Whether to format examples as multi-turn dialogue.

        Returns:
            tuple[list[str], list[str]]: (roles, contents) where roles is a list of speaker roles
                and contents is a list of corresponding message contents.
        """
        roles = []
        contents = []
        task_prompt_template = specific_task_config["prompt_template"]

        # insert multiturn examples to contents
        if hasattr(self, "example_dataset") and self.example_dataset is not None:
            in_context_examples = self.example_dataset.to_pandas()

            if not fewshot_as_multiturn:
                # TODO: add support for ICL as a single turn for chat models
                raise NotImplementedError(
                    "ICL as a single turn for chat models is not supported"
                )
            else:
                for _, row in in_context_examples.iterrows():
                    roles.append("user")
                    contents.append(
                        task_prompt_template["task_template"].format(
                            **row["prompts"][0],
                            answer_tag=task_prompt_template["answer_tag"],
                        )
                    )
                    roles.append("assistant")
                    contents.append(
                        task_prompt_template["answer_template"].format(
                            **row,
                            **row["prompts"][0],
                            answer_tag=task_prompt_template["answer_tag"],
                        )
                    )

        roles.append("user")
        contents.append(task_prompt_template["task_template"].format(**values))

        # append preamble to first user prompt
        contents[0] = (
            task_prompt_template["preamble"].format(
                answer_tag=task_prompt_template["answer_tag"]
            )
            + "\n\n"
            + contents[0]
        ).strip()

        return roles, contents

    ### Dataframe functions
    def convert_dataset_to_dataframe(self) -> None:
        """Convert the loaded dataset to a pandas DataFrame for easier manipulation."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please load the dataset first.")

        if self.dataframe is not None:
            logger.warning("DataFrame already exists. Skipping conversion.")
        else:
            self.dataframe = self.dataset.to_pandas()

    def load_model_outputs_into_dataset(self, turn):
        """Load model responses for a given turn and merge them into the dataset.

        Reads the batch response file for the specified turn and appends each
        response column to the corresponding dataset row. Columns that already
        exist in the row are converted to lists so multiple turn responses are
        accumulated.

        Args:
            turn (int): The conversational turn number (1-indexed) whose response
                file should be loaded.
        """
        responses_filepath = self.get_batch_response_filepath(turn)

        # only try to load response files if they exist, since some turns may not have generated responses yet
        if os.path.exists(responses_filepath):
            with open(responses_filepath, "r") as f:
                responses = f.readlines()
            # responses_df = pd.read_json(responses_filepath, lines=True)

        output = []
        for row, response in tqdm(
            zip(self.dataset, responses, strict=True), total=len(self.dataset)
        ):
            response_json = ujson.loads(response.strip())

            for col_name, value in response_json.items():
                if col_name == "id":
                    # remove id column since it's not needed for merging
                    continue

                if col_name in row:
                    if not isinstance(row[col_name], list):
                        row[col_name] = [row[col_name]]
                    row[col_name].append(value)
                else:
                    row[col_name] = [value]
            output.append(row)

        self.dataset = Dataset.from_list(output)

    def prepare_dataframe_for_writing(self) -> None:
        """Prepare the DataFrame for writing to file.

        Subclasses can override this method to drop unnecessary columns
        or transform data before saving inference results.

        Note:
            This is a hook method that can be customized by subclasses.
            The base implementation does nothing.
        """
        pass

    def write_out_dataframe(self) -> None:
        """Write dataframe to a file.

        Saves the pandas DataFrame to disk in the specified format
        (CSV or JSONL) after preparing it for writing.

        Raises:
            AssertionError: If file_type is not 'csv' or 'jsonl'.
        """
        self.prepare_dataframe_for_writing()
        output_filepath = self.get_inference_filepath(
            file_type=self.inference_file_type
        )

        logger.info(
            "Saving inference results for task '%s' to %s",
            self.task_name.upper(),
            output_filepath,
        )
        file_type = output_filepath.split(".")[-1]
        assert file_type in [
            "csv",
            "jsonl",
        ], "File type must be either 'csv' or 'jsonl'"

        # save outputs
        if file_type == "csv":
            self.dataframe.to_csv(output_filepath, index=False)
        elif file_type == "jsonl":
            self.dataframe.to_json(
                output_filepath, orient="records", force_ascii=False, lines=True
            )

        logger.info("Inference results saved!")

    def update_individual_scores(self, scores):
        """Write per-example scores into the dataframe

        Args:
            scores (list): Per-example score values to store in the
                'individual_scores' column of the dataframe.
        """
        self.dataframe["individual_scores"] = scores
