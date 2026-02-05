import asyncio
from urllib.parse import urlparse

import importlib_metadata
import requests
from openai import AsyncOpenAI

from src.base_logger import get_logger
from src.serving.local.base_serving import BaseServing

logger = get_logger(__name__)


class LocalOpenAIServing(BaseServing):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        is_base_model: bool = False,
        ssl_verify: bool = True,
        max_workers: int = 100,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.is_base_model = is_base_model
        self.api_key = api_key
        self.ssl_verify = ssl_verify
        self.max_workers = max_workers

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def load_model(self) -> None:
        """No-op for Local OpenAI serving as model is hosted externally."""
        pass

    def get_run_env(self) -> dict:
        """
        Get the runtime environment information.

        Returns:
            dict: Dictionary containing the OpenAI version.
        """
        return {"openai_version": importlib_metadata.version("openai")}

    def generate(
        self, messages: list, logprobs: bool = False, **generation_kwargs
    ) -> dict:
        response = asyncio.run(
            self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **generation_kwargs,
            )
        )
        return response

    async def run_batch_generate_coroutine(
        self, batch_messages: list[list], logprobs: bool = False, **generation_kwargs
    ) -> list[dict]:
        coros = []
        for messages in batch_messages:
            coros.append(
                self.client.chat.completions.create(
                    model=self.model_name, messages=messages, **generation_kwargs
                )
            )

        responses = await asyncio.gather(*coros)
        return responses

    def batch_generate(
        self, batch_messages: list[list], logprobs: bool = False, **generation_kwargs
    ) -> list[dict]:
        loop = asyncio.get_event_loop()

        responses = loop.run_until_complete(
            self.run_batch_generate_coroutine(
                batch_messages, logprobs=logprobs, **generation_kwargs
            )
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
                "add_generation_prompt": True,
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


if __name__ == "__main__":
    # start an OpenAI compatible server using vLLM with the following command
    # vllm serve google/gemma-2-9b-it --dtype bfloat16 --api-key token-abc123 --tensor-parallel-size 1 --enable-prefix-caching
    model = "google/gemma-2-9b-it"
    base_url = "http://localhost:8000/v1"
    api_key = "token-abc123"

    localOpenAIModel = LocalOpenAIServing(model, base_url, api_key, is_base_model=False)

    messages = [{"role": "user", "content": "ELI5: Why is the sky blue"}]
    # run generation
    response = localOpenAIModel.generate(messages)
    print(response)

    # run batch generation
    response = localOpenAIModel.batch_generate([messages for _ in range(5)])
    print(response)
