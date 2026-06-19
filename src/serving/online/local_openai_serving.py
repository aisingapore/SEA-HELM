import asyncio
from urllib.parse import urlparse

import importlib_metadata
import openai
import requests
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.base_logger import get_logger

logger = get_logger(__name__)


class LocalOpenAIServing:
    """Serving class for any local OpenAI-compatible API endpoint.

    Wraps an AsyncOpenAI client with retry logic, semaphore-bounded concurrency,
    and helpers for both completion and chat-completion endpoints.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str = "",
        is_base_model: bool = False,
        ssl_verify: bool = True,
        max_workers: int = 100,
    ):
        """Initialize LocalOpenAIServing.

        Args:
            model_name (str): Model identifier passed to the API.
            base_url (str | None): Base URL of the OpenAI-compatible server (e.g. ``http://localhost:8000/v1``). Defaults to None.
            api_key (str): API key for authentication. Defaults to empty string.
            is_base_model (bool): Whether the model is a base (non-instruct) model. Defaults to False.
            ssl_verify (bool): Whether to verify SSL certificates. Defaults to True.
            max_workers (int): Maximum number of concurrent async requests. Defaults to 100.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.is_base_model = is_base_model
        self.api_key = api_key
        self.ssl_verify = ssl_verify
        self.max_workers = max_workers

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.additional_generation_kwargs = {}

    def load_model(self) -> None:
        """No-op for Local OpenAI serving as model is hosted externally."""
        pass

    def is_model_loaded(self) -> bool:
        """Check whether the server is running and healthy.

        Returns:
            bool: True if the /health endpoint returns HTTP 200, False otherwise.
        """
        try:
            self.client.models.list()
            return True
        except openai.AuthenticationError:
            print("❌ Authentication failed. The API key is incorrect or invalid.")
            return False
        except openai.RateLimitError:
            print(
                "⚠️ Rate limit exceeded. Check your plan and billing details on the [OpenAI platform](https://openai.com)."
            )
            return False
        except openai.APIError as e:
            print(f"⚠️ An API error occurred: {e}")
            return False
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            return False

    def get_run_env(self) -> dict:
        """
        Get the runtime environment information.

        Returns:
            dict: Dictionary containing the OpenAI version.
        """
        return {"openai_version": importlib_metadata.version("openai")}

    @retry(
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            )
        ),
        wait=wait_random_exponential(multiplier=1, min=3, max=60),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    async def _generate_single_completion(self, prompts: str, generation_kwargs: dict):
        """Send a single completion request to the API with retry logic.

        Args:
            prompts (str): The prompt string to complete.
            generation_kwargs (dict): Generation parameters forwarded to the API.
                vLLM-specific keys (``include_stop_str_in_output``, ``continue_final_message``,
                ``add_generation_prompt``) are moved to ``extra_body`` automatically.

        Returns:
            openai.types.Completion: The raw completion response object.
        """
        for additional_kwarg in [
            "include_stop_str_in_output",
            "continue_final_message",
            "add_generation_prompt",
        ]:
            if additional_kwarg in generation_kwargs:
                self.additional_generation_kwargs[additional_kwarg] = (
                    generation_kwargs.pop(additional_kwarg)
                )

        return await self.client.completions.create(
            model=self.model_name,
            prompt=prompts,
            **generation_kwargs,
            extra_body=self.additional_generation_kwargs,
        )

    async def run_generate_completions_coroutine(
        self, prompts: list[str], generation_kwargs: list[dict] | dict
    ) -> list[dict]:
        """Async coroutine that runs completion requests concurrently.

        Args:
            prompts (list[str]): List of prompt strings.
            generation_kwargs (list[dict] | dict): Per-prompt generation parameters, or a
                single dict applied to all prompts.

        Returns:
            list[dict]: List of raw completion response objects in the same order as ``prompts``.
        """
        semaphore = asyncio.Semaphore(self.max_workers)

        async def _bounded(messages, gen_kwargs):
            async with semaphore:
                return await self._generate_single_completion(messages, gen_kwargs)

        if isinstance(generation_kwargs, dict):
            generation_kwargs = [generation_kwargs] * len(prompts)
        responses = await asyncio.gather(
            *(_bounded(m, g) for m, g in zip(prompts, generation_kwargs, strict=True))
        )
        return responses

    def generate_completions(
        self, prompts: list, generation_kwargs: list[dict] | dict
    ) -> list[dict]:
        """Run completion requests for a list of prompts.

        Args:
            prompts (list): List of prompt strings.
            generation_kwargs (list[dict] | dict): Per-prompt generation parameters, or a
                single dict applied to all prompts.

        Returns:
            list[dict]: List of raw completion response objects.
        """
        responses = asyncio.run(
            self.run_generate_completions_coroutine(prompts, generation_kwargs)
        )
        return responses

    @retry(
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            )
        ),
        wait=wait_random_exponential(multiplier=1, min=3, max=60),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    async def _generate_single_chat_completion(
        self, messages: list, generation_kwargs: dict
    ):
        """Send a single chat-completion request to the API with retry logic.

        Args:
            messages (list): List of message dicts (``role`` / ``content`` pairs).
            generation_kwargs (dict): Generation parameters forwarded to the API.
                vLLM-specific keys are moved to ``extra_body`` automatically.

        Returns:
            openai.types.chat.ChatCompletion: The raw chat completion response object.
        """
        for additional_kwarg in [
            "include_stop_str_in_output",
            "continue_final_message",
            "add_generation_prompt",
        ]:
            if additional_kwarg in generation_kwargs:
                self.additional_generation_kwargs[additional_kwarg] = (
                    generation_kwargs.pop(additional_kwarg)
                )
        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **generation_kwargs,
            extra_body=self.additional_generation_kwargs,
        )

    async def run_generate_chat_responses_coroutine(
        self, batch_messages: list[list], generation_kwargs: list[dict] | dict
    ) -> list[dict]:
        """Async coroutine that runs chat-completion requests concurrently.

        Args:
            batch_messages (list[list]): List of conversations, where each conversation is a
                list of message dicts.
            generation_kwargs (list[dict] | dict): Per-conversation generation parameters, or a
                single dict applied to all conversations.

        Returns:
            list[dict]: List of raw chat completion response objects in the same order as
                ``batch_messages``.
        """
        semaphore = asyncio.Semaphore(self.max_workers)

        async def _bounded(messages, gen_kwargs):
            async with semaphore:
                return await self._generate_single_chat_completion(messages, gen_kwargs)

        if isinstance(generation_kwargs, dict):
            generation_kwargs = [generation_kwargs] * len(batch_messages)
        responses = await asyncio.gather(
            *(
                _bounded(m, g)
                for m, g in zip(batch_messages, generation_kwargs, strict=True)
            )
        )
        return responses

    def generate_chat_responses(self, conversations, generation_kwargs) -> list[dict]:
        """Run chat-completion requests for a list of conversations.

        Args:
            conversations: List of conversations, where each conversation is a list of
                message dicts.
            generation_kwargs (list[dict] | dict): Per-conversation generation parameters, or a
                single dict applied to all conversations.

        Returns:
            list[dict]: List of raw chat completion response objects.
        """
        responses = asyncio.run(
            self.run_generate_chat_responses_coroutine(conversations, generation_kwargs)
        )
        return responses

    def tokenize(self, message: list) -> dict:
        """
        Tokenize a message using the model's tokenizer.

        Args:
            message (list): The message to tokenize.

        Returns:
            dict: The tokenization response from the API.

        Raises:
            AssertionError: If base_url is not provided.
        """
        assert self.base_url is not None, (
            "Base URL is required for to get tokenized prompts"
        )

        parsed_url = urlparse(self.base_url)
        base_url = parsed_url.scheme + "://" + parsed_url.netloc

        model = "/".join(self.model_name.split("/")[1:])
        response = requests.post(
            base_url + "/tokenize",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "accept": "application/json",
            },
            json={
                "model": model,
                "messages": message,
                "add_special_tokens": False,
                "add_generation_prompt": False,
            },
        )

        return response.json()

    def batch_tokenize(self, messages: list[list]) -> list[dict]:
        """
        Tokenize multiple messages in batch.

        Args:
            messages (list[list]): List of messages to tokenize.

        Returns:
            list[dict]: List of tokenization responses.
        """
        # TODO handle cases when encode does not work
        batch_response = [self.tokenize(message) for message in messages]

        return batch_response

    def get_response(self, output) -> str:
        """
        Extract the response content from the model output.

        Args:
            output: The model output object.

        Returns:
            str: The response content string.
        """
        return output.choices[0].message.content

    def get_reasoning_content(self, output) -> str | None:
        """
        Extract the reasoning content from the model output, if available.

        Args:
            output: The model output object.

        Returns:
            str | None: The reasoning content string, or None if not available.
        """
        if hasattr(output.choices[0].message, "reasoning"):
            return output.choices[0].message.reasoning
        elif hasattr(output.choices[0].message, "reasoning_content"):
            return output.choices[0].message.reasoning_content
        else:
            return None

    def cleanup(self) -> None:
        """No-op for Local OpenAI serving as there are no local resources to clean up."""
        pass

    def parse_output(self, output: dict, custom_id: str | None = None) -> dict:
        """Parse the outputs of the generated responses.

        Args:
            output (dict): The generated output to parse.
            custom_id (str, optional): The custom ID associated with the input. Defaults to None.

        Returns:
            dict: The parsed outputs.
        """
        try:
            parsed_output = {
                "finish_reasons": output.choices[0].finish_reason,
                "responses": self.get_response(output),
                "reasoning_contents": self.get_reasoning_content(output),
                "custom_ids": custom_id,
                "token_usages": {
                    "prompt_tokens": output.usage.prompt_tokens,
                    "completion_tokens": output.usage.completion_tokens,
                    "total_tokens": output.usage.total_tokens,
                },
                "function_calls": output.choices[0].message.function_call,
                "tool_calls": output.choices[0].message.tool_calls,
                "logprobs": output.choices[0].logprobs,
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


if __name__ == "__main__":
    # start an OpenAI compatible server using vLLM with the following command
    # vllm serve google/gemma-2-9b-it --dtype bfloat16 --api-key token-abc123 --tensor-parallel-size 1 --enable-prefix-caching
    model = "google/gemma-2-9b-it"
    base_url = "http://localhost:8000/v1"
    api_key = "token-abc123"

    localOpenAIModel = LocalOpenAIServing(model, base_url, api_key, is_base_model=False)

    messages = [{"role": "user", "content": "ELI5: Why is the sky blue"}]
    # run batch generation
    response = localOpenAIModel.generate_chat_responses(
        [messages for _ in range(5)], {}
    )
    print(response)
