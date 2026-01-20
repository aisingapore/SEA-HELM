import asyncio
import os
from abc import ABC, abstractmethod

import pandas as pd


class BaseBatchServing(ABC):
    """Abstract base class for batch serving models."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def get_run_env(self) -> dict:
        """
        Get the runtime environment information.

        Returns:
            dict: Dictionary containing the API package version.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Please implement get_run_env method."
        )

    @abstractmethod
    def prepare_llm_batches(
        self,
        llm_batch_file_path: str,
        conversations: list,
        custom_ids: list | None = None,
        **generation_kwargs,
    ) -> None:
        """Prepare the batch file for Vertex AI

        Args:
            llm_batch_file_path (str): The path to the batch file.
            conversations (list): The conversations to prepare.
            custom_ids (list, optional): The custom ids to use. Defaults to None.
            **generation_kwargs: The generation kwargs to use.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Please implement prepare_llm_batches method."
        )

    def get_batch_directory_filepath(
        self,
        task_name: str,
        lang: str,
        parent_path: str,
        file_type: str = "jsonl",
        is_response: bool = False,
    ) -> str:
        """Generate the file path for batch processing files.

        Args:
            task_name (str): The name of the task being processed.
            lang (str): The language code for the task.
            parent_path (str): The base path for storing run files.
            file_type (str, optional): The file extension to use. Defaults to "jsonl".
            is_response (bool, optional): Whether this is a response file. Defaults to False.

        Returns:
            str: The generated file path for the batch file.
        """
        if is_response:
            return os.path.join(
                parent_path,
                f"{os.path.basename(self.model_name)}_{task_name}_{lang}_batch_response.{file_type}",
            )
        else:
            return os.path.join(
                parent_path,
                f"{os.path.basename(self.model_name)}_{task_name}_{lang}_batch.{file_type}",
            )

    def generate_batch_api_responses(
        self,
        conversations: list,
        generation_kwargs: dict,
        model_name: str,
        task_name: str,
        lang: str,
        parent_path: str,
    ) -> list:
        """
        Generate batch API responses for a dataset using batch processing.

        Args:
            conversations (list): Conversations to process.
            generation_kwargs (dict): Parameters for text generation.
            model_name (str): Name of the model to use for generation.
            task_name (str): Name of the task being processed.
            lang (str): Language code for the task.
            parent_path (str): Base path for storing run files.

        Returns:
            list: List of generated outputs from the batch API.
        """
        batch_filepath = self.get_batch_directory_filepath(
            task_name=task_name,
            lang=lang,
            parent_path=parent_path,
            file_type="jsonl",
            is_response=False,
        )
        batch_output_filepath = self.get_batch_directory_filepath(
            task_name=task_name,
            lang=lang,
            parent_path=parent_path,
            file_type="jsonl",
            is_response=True,
        )

        self.prepare_llm_batches(
            llm_batch_file_path=batch_filepath,
            model=model_name,
            conversations=conversations,
            **generation_kwargs,
        )

        asyncio.run(self.abatch_generate(batch_filepath, batch_output_filepath))
        generated_outputs = pd.read_json(batch_output_filepath, lines=True).to_dict(
            "records"
        )
        return generated_outputs

    @abstractmethod
    def get_response(self, output: dict) -> str:
        """
        Gets the response from the output

        Args:
            output (dict): The output to get the response from.

        Returns:
            str: The response from the output.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Please implement get_response method."
        )

    @abstractmethod
    def get_ids_from_batch(self, batch: dict) -> str:
        """
        Extract custom IDs from batch outputs.

        Args:
            batch (dict): The batch output dictionary.
        Returns:
            string: Comma-separated string of custom IDs.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Please implement get_ids_from_batch method."
        )

    def parse_outputs(
        self,
        generated_outputs: list,
        conversations: list | None = None,
        tokenize_prompts: bool = False,
        use_logprobs: bool = False,
    ) -> dict:
        """
        Parse the generated outputs into a structured format.

        Args:
            generated_outputs (list): List of generated outputs from the model.
            conversations (list, optional): List of original conversations.
                Defaults to None.
            tokenize_prompts (bool, optional): Whether to tokenize the prompts.
                Defaults to False.

        Returns:
            dict: Dictionary containing parsed responses, errors, and optionally tokenized prompts.
        """
        responses = []
        errors = []
        tokenized_prompts = []

        for output in generated_outputs:
            responses.append(self.get_response(output))
            errors.append(None)

        if tokenize_prompts and conversations is not None:
            tokenized_prompts = self.batch_tokenize(conversations)

        outputs = {
            "responses": responses,
            "errors": errors,
            "tokenized_prompts": tokenized_prompts,
        }

        return outputs

    def cleanup(self) -> None:  # noqa: B027
        """Cleanup any resources used by the serving class."""
        pass
