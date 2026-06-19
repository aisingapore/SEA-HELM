import os
from abc import ABC, abstractmethod

import pandas as pd

from src.base_logger import get_logger

logger = get_logger(__name__)


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

    def prepare_batches(
        self,
        batch_filepath: str,
        conversations: list,
        generation_kwargs: dict,
        custom_ids: list | None = None,
    ) -> None:
        """Prepare the batch file for Batch APIs

        Args:
            batch_filepath (str): The path to the batch file.
            conversations (list): The conversations to prepare.
            custom_ids (list, optional): The custom ids to use. Defaults to None.
            **generation_kwargs: The generation kwargs to use.
        """
        requests = []
        idx = os.path.splitext(os.path.split(batch_filepath)[-1])[0]

        # Vertex AI batch API requires a specific format for the generation kwargs
        selected_generation_kwargs = {}
        for k, v in generation_kwargs.items():
            if k in self.kwargs_map:
                selected_generation_kwargs[self.kwargs_map[k]] = v
            else:
                logger.warning(
                    "Unsupported generation kwarg for %s API: %s", self.friendly_name, k
                )

        for ix, conversation in enumerate(conversations):
            custom_id = custom_ids[ix] if custom_ids else f"{idx}_{ix}"
            requests.append(
                self.create_request(
                    conversation,
                    custom_id,
                    selected_generation_kwargs,
                )
            )

        df = pd.DataFrame(requests)
        df.to_json(batch_filepath, orient="records", lines=True, force_ascii=False)

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

    def parse_output(self, output: dict, custom_id: str | None = None) -> dict:
        """Parse a single generated output into a structured dict.

        Args:
            output (dict): The generated output to parse.
            custom_id (str, optional): The custom ID associated with the input.
                Falls back to get_ids_from_batch if not provided. Defaults to None.

        Returns:
            dict: The parsed output.
        """
        try:
            parsed_output = {
                "finish_reasons": None,
                "responses": self.get_response(output),
                "reasoning_contents": None,
                "custom_ids": custom_id
                if custom_id is not None
                else self.get_ids_from_batch(output),
                "token_usages": None,
                "function_calls": None,
                "tool_calls": None,
                "logprobs": None,
                "errors": None,
            }
        except Exception as e:
            parsed_output = {
                "finish_reasons": None,
                "responses": None,
                "reasoning_contents": None,
                "custom_ids": custom_id,
                "token_usages": None,
                "function_calls": None,
                "tool_calls": None,
                "logprobs": None,
                "errors": str(e),
            }
        return parsed_output

    def cleanup(self) -> None:  # noqa: B027
        """Cleanup any resources used by the serving class."""
        pass

    def tokenize_conversations(self, conversations: list) -> list:
        """
        Tokenize a batch of conversations.

        Args:
            conversations (list): List of conversations to tokenize.

        Returns:
            list: List of tokenized conversations.
        """
        return self.batch_tokenize(conversations)
