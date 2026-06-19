from abc import ABC, abstractmethod
from typing import Any


class BaseOfflineServing(ABC):
    """Abstract base class for serving models."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def generate_completions(
        self, prompts: list[list] | list, generation_kwargs: dict | list[dict]
    ) -> list[Any]:
        """Generate responses for a given batch of prompts.

        Args:
            prompts (list[list] | list): The batch of prompts to generate responses for.
            generation_kwargs (dict | list[dict]): Additional generation kwargs.

        Returns:
            list[Any]: The generated responses.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def generate_chat_responses(
        self, conversations: list[list] | list, generation_kwargs: dict | list[dict]
    ) -> list[Any]:
        """Generate responses for a given batch of conversations.

        Args:
            conversations (list[list] | list): The batch of conversations to generate responses for.
            generation_kwargs (dict | list[dict]): Additional generation kwargs.

        Returns:
            list[Any]: The generated responses.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def get_response(self, output: dict) -> str:
        """Get the response from the output.

        Args:
            output (dict): The output to get the response from.

        Returns:
            str: The response from the output.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def parse_output(self, output, custom_id: str | None = None) -> dict:
        """
        Parse a single generated output into a structured dict.

        Args:
            output: A single generated output from the model.
            custom_id (str, optional): The custom ID associated with the input.
                Defaults to None.

        Returns:
            dict: Dictionary containing parsed response, errors, and metadata.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    def cleanup(self) -> None:  # noqa: B027
        """Cleanup any resources used by the serving class."""
        pass
