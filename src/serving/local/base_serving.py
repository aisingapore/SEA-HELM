import time
from abc import ABC, abstractmethod
from typing import Any


class BaseServing(ABC):
    """Abstract base class for serving models."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def generate(
        self, messages: list, logprobs: bool = False, **generation_kwargs
    ) -> Any:
        """Generate a response for a given message.

        Args:
            messages (list): The messages to generate a response for.
            logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
            **generation_kwargs: Additional generation kwargs.

        Returns:
            Any: The generated response.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def batch_generate(
        self, batch_messages: list[list], logprobs: bool = False, **generation_kwargs
    ) -> list[Any]:
        """Generate responses for a given batch of messages.

        Args:
            batch_messages (list[list]): The batch of messages to generate responses for.
            logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
            **generation_kwargs: Additional generation kwargs.

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

    def generate_responses(
        self,
        conversations: list,
        generation_kwargs: dict,
        labels: list | None = None,
        no_batching: bool = False,
        use_logprobs: bool = False,
    ) -> tuple[list[Any], float]:
        """Generate responses for a given dataset.

        Args:
            conversations (list): The conversations to generate responses for.
            generation_kwargs (dict): The generation parameters to use.
            labels (list, optional): The labels to generate responses for. Defaults to None.
            no_batching (bool, optional): Whether to generate responses in batch. Defaults to False.
            use_logprobs (bool, optional): Whether to return log probabilities. Defaults to False.

        Returns:
            tuple[list[Any], float]: The generated responses and the time taken to generate the responses.
        """
        start_time = time.perf_counter()
        if no_batching:
            generated_outputs = []
            for conversation in conversations:
                _output = self.generate(conversation, **generation_kwargs)
                generated_outputs.append(_output)
        else:
            if use_logprobs:
                generation_kwargs["answers"] = labels

            generated_outputs = self.batch_generate(
                conversations,
                **generation_kwargs,
            )
        end_time = time.perf_counter()
        inference_time_taken = end_time - start_time

        return generated_outputs, inference_time_taken

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
