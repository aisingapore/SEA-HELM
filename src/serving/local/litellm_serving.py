import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlparse

import httpx
import importlib_metadata
import litellm
import requests
from litellm.exceptions import LITELLM_EXCEPTION_TYPES
from litellm.llms.vllm.completion import handler as vllm_handler
from litellm.types.utils import ModelResponse
from litellm.utils import get_optional_params

from src.serving.local.base_serving import BaseServing

litellm.drop_params = True


def batch_completion_with_retries(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages: list[list] | None = None,
    functions: list | None = None,
    function_call: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    n: int | None = None,
    stream: bool | None = None,
    stop: list[str] | None = None,
    max_tokens: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: dict | None = None,
    user: str | None = None,
    deployment_id: str | None = None,
    request_timeout: int | None = None,
    timeout: int | None = 1800,
    max_workers: int | None = 100,
    num_retries: int = 10,
    retry_strategy: str = "exponential_backoff_retry",
    # Optional liteLLM function params
    **kwargs,
) -> list:
    """
    Batch litellm.completion function for a given model with retry logic.

    This function processes multiple message conversations in parallel using ThreadPoolExecutor
    for non-VLLM models, or uses VLLM's native batch completion handler for VLLM models.

    Args:
        model (str): The model to use for generating completions.
        messages (list[list], optional): List of message conversations to use as input for
            generating completions. Each element should be a list of messages. Defaults to [].
        functions (list, optional): List of functions to use as input for generating completions.
            Defaults to None.
        function_call (str, optional): The function call to use as input for generating completions.
            Defaults to None.
        temperature (float, optional): The temperature parameter for generating completions.
            Defaults to None.
        top_p (float, optional): The top-p parameter for generating completions.
            Defaults to None.
        n (int, optional): The number of completions to generate. Defaults to None.
        stream (bool, optional): Whether to stream completions or not. Defaults to None.
        stop (list[str], optional): List of stop sequences for generating completions.
            Defaults to None.
        max_tokens (int, optional): The maximum number of tokens to generate.
            Defaults to None.
        presence_penalty (float, optional): The presence penalty for generating completions.
            Defaults to None.
        frequency_penalty (float, optional): The frequency penalty for generating completions.
            Defaults to None.
        logit_bias (dict, optional): The logit bias for generating completions.
            Defaults to None.
        user (str, optional): The user string for generating completions. Defaults to None.
        deployment_id (str, optional): The deployment ID for generating completions.
            Defaults to None.
        request_timeout (int, optional): The request timeout for generating completions.
            Defaults to None.
        timeout (int, optional): The overall timeout for the operation. Defaults to 1800.
        max_workers (int, optional): The maximum number of threads to use for parallel processing.
            Defaults to 100.
        num_retries (int, optional): The maximum number of retries to perform. Defaults to 10.
        retry_strategy (str, optional): Type of retry strategy to use.
            Defaults to "exponential_backoff_retry".
        **kwargs: Additional keyword arguments to pass to the completion function.

    Returns:
        List: A list of completion results. May contain exception objects if errors occurred.
    """
    args = locals()

    batch_messages = messages
    completions = []
    model = model
    custom_llm_provider = None
    if model.split("/", 1)[0] in litellm.provider_list:
        custom_llm_provider = model.split("/", 1)[0]
        model = model.split("/", 1)[1]
    if custom_llm_provider == "vllm":
        optional_params = get_optional_params(
            functions=functions,
            function_call=function_call,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream or False,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            # params to identify the model
            model=model,
            custom_llm_provider=custom_llm_provider,
        )
        results = vllm_handler.batch_completions(
            model=model,
            messages=batch_messages,
            custom_prompt_dict=litellm.custom_prompt_dict,
            optional_params=optional_params,
        )
    # all non VLLM models for batch completion models
    else:

        def chunks(lst: list, n: int) -> Any:
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for sub_batch in chunks(batch_messages, 100):
                for message_list in sub_batch:
                    kwargs_modified = args.copy()
                    kwargs_modified.pop("max_workers")
                    kwargs_modified["messages"] = message_list
                    original_kwargs = {}
                    if "kwargs" in kwargs_modified:
                        original_kwargs = kwargs_modified.pop("kwargs")
                    future = executor.submit(
                        litellm.completion_with_retries,
                        **kwargs_modified,
                        **original_kwargs,
                    )
                    completions.append(future)

        # Retrieve the results from the futures
        # results = [future.result() for future in completions]
        # return exceptions if any
        results = []
        for future in completions:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(exc)

    return results


class LiteLLMServing(BaseServing):
    """
    A serving class that uses LiteLLM for language model completions.

    This class provides methods for generating responses from language models using the LiteLLM
    library, supporting both single and batch requests, synchronous and asynchronous operations.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        is_base_model: bool = False,
        ssl_verify: bool = True,
        max_workers: int = 100,
    ):
        """
        Initialize the LiteLLMServing instance.

        Args:
            model (str): The model identifier to use for completions.
            base_url (str, optional): The base URL for the API endpoint. Defaults to None.
            api_key (str, optional): The API key for authentication. Defaults to None.
            is_base_model (bool, optional): Whether this is a base model that requires special
                chat template handling. Defaults to False.
            ssl_verify (bool, optional): Whether to verify SSL certificates. Defaults to True.
            max_workers (int, optional): Maximum number of worker threads for batch operations.
                Defaults to 100.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.is_base_model = is_base_model
        self.max_workers = max_workers

        if self.is_base_model:
            logging.warning(
                "Base model selected. Please ensure that the chat template has been specified in the LiteLLM config."
            )

        if not ssl_verify:
            litellm.client_session = httpx.Client(verify=False)

    def load_model(self) -> None:
        """No-op for LiteLLM serving as model is hosted externally."""
        pass

    def get_run_env(self) -> dict:
        """
        Get the runtime environment information.

        Returns:
            dict: Dictionary containing the LiteLLM version.
        """
        return {"litellm_version": importlib_metadata.version("litellm")}

    def generate(
        self,
        messages: list,
        logprobs: bool = False,
        num_retries: int = 10,
        **generation_kwargs,
    ):
        """
        Generate a completion for a single conversation.

        Args:
            messages (list): List of messages forming the conversation.
            logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
            num_retries (int, optional): Number of retries for failed requests. Defaults to 10.
            **generation_kwargs: Additional generation parameters.

        Returns:
            The completion response from the model.
        """
        response = litellm.completion_with_retries(
            model=self.model_name,
            messages=messages,
            base_url=self.base_url,
            api_key=self.api_key,
            logprobs=logprobs,
            num_retries=num_retries,
            retry_strategy="exponential_backoff_retry",
            **generation_kwargs,
        )
        return response

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

    def batch_generate(
        self,
        batch_messages: list[list],
        logprobs: bool = False,
        use_retries: bool = True,
        num_retries: int = 10,
        retry_strategy: str = "exponential_backoff_retry",
        **generation_kwargs,
    ) -> list:
        """
        Generate completions for multiple conversations in batch.

        Args:
            batch_messages (list[list]): List of message conversations to process.
            logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
            use_retries (bool, optional): Whether to use retry logic. Defaults to True.
            num_retries (int, optional): Number of retries for failed requests. Defaults to 10.
            retry_strategy (str, optional): Strategy for retries.
                Defaults to "exponential_backoff_retry".
            **generation_kwargs: Additional generation parameters.

        Returns:
            list: List of completion responses.
        """
        if use_retries:
            batch_response = batch_completion_with_retries(
                model=self.model_name,
                messages=batch_messages,
                base_url=self.base_url,
                api_key=self.api_key,
                logprobs=logprobs,
                max_workers=self.max_workers,
                num_retries=num_retries,
                retry_strategy=retry_strategy,
                **generation_kwargs,
            )
        else:
            batch_response = litellm.batch_completion(
                model=self.model_name,
                messages=batch_messages,
                base_url=self.base_url,
                api_key=self.api_key,
                logprobs=logprobs,
                max_workers=self.max_workers,
                **generation_kwargs,
            )
        return batch_response

    async def agenerate(
        self, messages: list, logprobs: bool = False, **generation_kwargs
    ):
        """
        Generate a completion asynchronously for a single conversation.

        Args:
            messages (list): List of messages forming the conversation.
            logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
            **generation_kwargs: Additional generation parameters.

        Returns:
            The async completion response from the model.
        """
        response = await litellm.acompletion(
            model=self.model_name,
            messages=messages,
            base_url=self.base_url,
            api_key=self.api_key,
            logprobs=logprobs,
            **generation_kwargs,
        )
        return response

    def get_response(self, output) -> str:
        """
        Extract the response content from the model output.

        Args:
            output: The model output object.

        Returns:
            str: The response content string.
        """
        return output["choices"][0]["message"]["content"]

    def convert_response_to_json(self, response: ModelResponse) -> dict:
        """Convert the response to a JSON serializable format.

        Args:
            response (ModelResponse): The response to convert.

        Returns:
            dict: The JSON serializable response.
        """
        return {
            "id": response.id,
            "created": response.created,
            "model": response.model,
            "system_fingerprint": response.system_fingerprint,
            "choices": [choice.to_dict() for choice in response.choices],
            "usage": response.usage.to_dict(),
        }

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
            # Handle LiteLLM Error types
            if type(output) in LITELLM_EXCEPTION_TYPES:
                responses.append(None)
                errors.append(type(output).__name__)
            else:
                responses.append(self.get_response(output))
                errors.append(None)

        if tokenize_prompts:
            tokenized_prompts = self.batch_tokenize(conversations)

        outputs = {
            "responses": responses,
            "errors": errors,
            "tokenized_prompts": tokenized_prompts,
        }

        return outputs


if __name__ == "__main__":
    # start an OpenAI compatible server using vLLM with the following command
    # vllm serve google/gemma-2-9b-it --dtype bfloat16 --api-key token-abc123 --tensor-parallel-size 1 --enable-prefix-caching
    model_name = "openai/google/gemma-2-9b-it"
    base_url = "http://localhost:8000/v1"
    api_key = "token-abc123"

    litellmModel = LiteLLMServing(model_name, base_url, api_key, is_base_model=False)

    messages = [{"role": "user", "content": "ELI5: Why is the sky blue"}]
    # run generation
    response = litellmModel.generate(messages)
    print(response)

    # run batch generation
    response = litellmModel.batch_generate([messages for _ in range(5)])
    print(response)

    # run async generation
    response = asyncio.run(litellmModel.agenerate(messages))
    print(response)
