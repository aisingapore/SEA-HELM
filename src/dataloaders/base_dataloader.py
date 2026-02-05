import os
from abc import abstractmethod
from typing import Any, Callable

import pandas as pd
from datasets import Dataset

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
                f"No fewshot_num_examples found in task config. Using default value of {default_num_in_context_examples}."
            )
        self.fewshot_num_examples = fewshot_num_examples

        self.dataset = None
        self.example_dataset = None
        self.inference_df = None
        self.conversations = None
        self.num_turns = None

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

    def get_inference_filepath(self, file_type: str = "jsonl") -> str:
        """Generate the file path for storing inference results.

        Args:
            file_type (str): File format ('jsonl' or 'csv').

        Returns:
            str: Full file path for storing inference results.
        """
        parent_path = self.get_parent_folder()

        return os.path.join(
            parent_path,
            f"{os.path.basename(self.model_name)}_{self.task_name}_{self.lang}.{file_type}",
        )

    def read_inference_results_as_df(self) -> bool:
        """Read cached inference results from file into a DataFrame.

        Loads previously saved inference results from disk and converts
        numerical columns to strings for consistency.

        Returns:
            bool: True if results were successfully loaded, False if file doesn't exist.
        """
        file_type = self.inference_file_type
        assert file_type in [
            "csv",
            "jsonl",
        ], "File type must be either 'csv' or 'jsonl'"

        # save outputs
        input_filepath = self.get_inference_filepath(file_type=file_type)

        if os.path.exists(input_filepath) is False:
            logger.debug(
                "No cached inference results found for %s and %s at %s.",
                self.task_name,
                self.lang,
                input_filepath,
            )
            return False

        if file_type == "csv":
            self.inference_df = pd.read_csv(input_filepath)
        elif file_type == "jsonl":
            self.inference_df = pd.read_json(input_filepath, lines=True)

        # ensure that inference_df only contains
        numerical_cols = self.inference_df.select_dtypes(include="number").columns

        # Convert numerical columns to string
        for col in numerical_cols:
            self.inference_df[col] = self.inference_df[col].astype(str)
        return True

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
    def prepare_conversations_for_inference(
        self,
        turn: int,
        fewshot_as_multiturn: bool = False,
    ) -> list[Any]:
        """Prepare the conversations for inference by formatting prompts for the given turn.

        The formatted conversations should be in chat format compatible with the
        `apply_chat_template` method for prompt tokenization.

        Args:
            turn (int): Which turn to prepare the prompts for (1-indexed).
            fewshot_as_multiturn (bool): Whether to treat few-shot examples as multi-turn dialogues. Defaults to False.

        Returns:
            list[Any]: The formatted conversations stored in self.conversations.

        Note:
            Subclasses must implement this method to handle their specific prompt formatting.
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

    def get_update_function(
        self,
        column: str,
        data: list,
        active_rows: list[int] | None = None,
    ) -> Callable:
        """Create a function to update a specific column with new data.

        Args:
            column (str): Name of the column to update.
            data (list): List of data values to append to the column.
            active_rows (list[int], optional): List of row indices to update. Defaults to None.
                If not None, data is expected to be of same length and includes values for the active rows only.

        Returns:
            Callable: Update function that can be used with dataset.map().
        """
        # Re-populate data to the original length
        should_update = [
            True if i in active_rows else False for i in range(len(self.dataset))
        ]
        update_values = [
            data[active_rows.index(i)] if i in active_rows else None
            for i in range(len(self.dataset))
        ]

        def update_function(row, i):
            if should_update[i]:
                if column in row:
                    row[column].append(update_values[i])
                else:
                    row[column] = [update_values[i]]
            return row

        return update_function

    def update_column(
        self, column: str, data: list, active_rows: list | None = None
    ) -> None:
        """Update a column in the dataset with new data.

        Args:
            column (str): Name of the column to update.
            data (list): List of data values to add to the column.
            active_rows (list, optional): List of row indices to update. Defaults to None.
        """
        assert self.dataset is not None, "Dataset is not loaded."
        update_function = self.get_update_function(
            column, data, active_rows=active_rows
        )
        self.dataset = self.dataset.map(update_function, with_indices=True)

    def update_inference_df(self) -> None:
        """Update the inference DataFrame from the current dataset.

        Converts the HuggingFace dataset to a pandas DataFrame for easier manipulation.
        """
        assert self.dataset is not None, "Dataset is not loaded."
        self.inference_df = self.dataset.to_pandas()
        if "conversations" not in self.inference_df:
            self.inference_df["conversations"] = self.conversations

    def prepare_inference_df_for_writing(self) -> None:
        """Prepare the inference DataFrame for writing to file.

        Subclasses can override this method to drop unnecessary columns
        or transform data before saving inference results.

        Note:
            This is a hook method that can be customized by subclasses.
            The base implementation does nothing.
        """
        pass

    def write_out_inference_results(self) -> None:
        """Write inference results to a file.

        Saves the inference DataFrame to disk in the specified format
        (CSV or JSONL) after preparing it for writing.

        Raises:
            AssertionError: If file_type is not 'csv' or 'jsonl'.
        """
        self.prepare_inference_df_for_writing()
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
            self.inference_df.to_csv(output_filepath, index=False)
        elif file_type == "jsonl":
            self.inference_df.to_json(
                output_filepath, orient="records", force_ascii=False, lines=True
            )

        logger.info("Inference results saved!")
