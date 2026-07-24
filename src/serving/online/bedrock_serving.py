import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import boto3
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from src.base_logger import get_logger
from src.serving.online.base_online_serving import BaseOnlineServing

logger = get_logger(__name__)

# Generation kwargs that map directly onto the Converse ``inferenceConfig`` block.
# Converse normalises sampling parameters across providers, so unlike the batch
# (``invoke_model``) path there is no need to dispatch on the model provider.
_INFERENCE_CONFIG_MAP = {
    "max_tokens": "maxTokens",
    "max_completion_tokens": "maxTokens",
    "temperature": "temperature",
    "top_p": "topP",
}

# Generation kwargs understood by other (OpenAI/vLLM-style) serving backends that
# the Bedrock Converse API does not accept. They are dropped silently so that the
# same ``generation_kwargs`` can be passed to any serving class interchangeably.
_IGNORED_KWARGS = {
    "seed",
    "n",
    "logprobs",
    "top_logprobs",
    "use_logprobs",
    "include_stop_str_in_output",
    "continue_final_message",
    "add_generation_prompt",
    "presence_penalty",
    "frequency_penalty",
}

# Map the Bedrock Converse ``stopReason`` vocabulary onto the OpenAI-style
# ``finish_reason`` values that LocalOpenAIServing yields, so the serving classes
# stay interchangeable for consumers that gate on it (e.g. the inference
# strategies compare ``finish_reasons`` against the literal ``"stop"``). Unknown
# values pass through unchanged.
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
}

# Transient Bedrock runtime error codes that are safe to retry.
_RETRYABLE_ERROR_CODES = {
    "ThrottlingException",
    "ModelTimeoutException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
}


def _is_retryable_bedrock_error(exc: BaseException) -> bool:
    """Return True if *exc* is a transient Bedrock runtime error worth retrying.

    Args:
        exc (BaseException): The exception raised by a Converse call.

    Returns:
        bool: True for throttling/timeout/transient server errors, False otherwise.
    """
    if isinstance(exc, ClientError):
        error_code = exc.response.get("Error", {}).get("Code", "")
        return error_code in _RETRYABLE_ERROR_CODES
    return isinstance(
        exc, (ReadTimeoutError, ConnectTimeoutError, EndpointConnectionError)
    )


class BedrockServing(BaseOnlineServing):
    """Online (real-time) serving backend that uses the Amazon Bedrock Converse API.

    This class extends :class:`BaseOnlineServing` to provide Bedrock-specific
    real-time inference using the ``bedrock-runtime`` ``Converse`` API. Unlike
    the other :class:`BaseOnlineServing` subclasses (vLLM, SGLang) there is no
    local server subprocess to launch — Bedrock is a fully managed service — so
    the subprocess lifecycle (``_build_server_command``/health polling) is a
    no-op and ``load_model`` simply initialises the boto3 client.

    Inference uses the provider-agnostic Converse API, which normalises the
    request/response schema across providers (Anthropic Claude, Amazon Nova,
    Meta Llama, Mistral, ...). The methods used by the inference strategies
    (``generate_chat_responses``/``get_response``/``parse_output``/``cleanup``/
    ``tokenize``) mirror those of :class:`LocalOpenAIServing`, so the serving
    classes are interchangeable.

    This class targets :class:`DefaultInferenceStrategy` (standard single-pass
    chat/instruct evaluation). The two-phase ``BaseModelInferenceStrategy`` and
    ``LogprobsInferenceStrategy`` are not fully supported: the Converse API has
    no equivalent of vLLM's assistant-turn continuation
    (``continue_final_message``/``include_stop_str_in_output``, which are
    dropped) and exposes neither prompt token ids nor token logprobs.
    """

    def __init__(
        self,
        model_name: str,
        region: str | None = None,
        aws_profile: str | None = None,
        is_base_model: bool = False,
        api_key: str = "",  # unused; AWS credentials are resolved by boto3
        max_workers: int = 8,
        timeout: int = 3600,
        default_reasoning_effort: str | None = "medium",
        additional_model_request_fields: dict | None = None,
        **kwargs,
    ):
        """Initialize the BedrockServing instance.

        AWS credentials and region are resolved by boto3 from the standard
        environment / config chain. ``region`` and ``aws_profile`` may be passed
        explicitly (e.g. via ``--model_args``) to override the defaults.

        Args:
            model_name (str): The Bedrock model id or inference profile id to use.
                Claude 4.x models require a cross-region inference profile id
                (e.g. ``global.anthropic.claude-haiku-4-5-20251001-v1:0`` or
                ``apac.anthropic.claude-sonnet-4-6-20251115-v1:0``); older models
                may use a plain model id
                (e.g. ``anthropic.claude-3-5-sonnet-20240620-v1:0``).
            region (str, optional): AWS region for the Bedrock runtime client.
                Falls back to the boto3 default chain (``AWS_REGION`` /
                ``AWS_DEFAULT_REGION``) when not set. Defaults to None.
            aws_profile (str, optional): Named AWS profile to use. Falls back to
                the boto3 default chain (``AWS_PROFILE``) when not set. Defaults to None.
            is_base_model (bool, optional): Whether this is a base model. Stored for
                interface compatibility. Defaults to False.
            api_key (str, optional): Unused. AWS credentials are resolved by boto3.
            max_workers (int, optional): Maximum number of concurrent Converse
                requests. Kept low by default because Bedrock enforces per-account
                request-rate quotas; raise it if your account limits allow.
                Defaults to 8.
            timeout (int, optional): Read timeout (seconds) for a single Converse
                request. Defaults to 3600.
            default_reasoning_effort (str | None, optional): Controls the
                adaptive-thinking effort injected into every request.  When not
                ``None`` the initialiser adds
                ``{"thinking": {"type": "adaptive"}, "output_config": {"effort":
                <value>}}`` to ``additional_model_request_fields`` (unless a
                ``thinking`` key is already present).  Valid values are
                ``"low"``, ``"medium"``, and ``"high"``; ``None`` disables
                thinking injection entirely.  Must be set to ``None`` for
                non-thinking models such as Haiku 4.5, which reject all
                thinking-related fields.  Defaults to ``"medium"``.
            additional_model_request_fields (dict, optional): Model-specific fields
                passed verbatim as ``additionalModelRequestFields`` to every
                Converse call.  Examples:

                * ``{"top_k": 50}`` — Anthropic-specific sampling parameter.
                * Adaptive thinking (Sonnet 4.6 / Opus 4.8 / Fable 5):
                  ``{"thinking": {"type": "adaptive"}, "output_config":
                  {"effort": "medium"}}``
                * Extended thinking with explicit budget (older models):
                  ``{"thinking": {"type": "enabled", "budget_tokens": 8000}}``

                Defaults to None.
            **kwargs: Additional keyword arguments accepted for interface
                compatibility and ignored.
        """
        # NOTE: BaseOnlineServing.__init__ is intentionally not called because it
        # is tailored to local subprocess servers (it allocates a local port and
        # builds a ``http://localhost`` base_url, neither of which apply to a
        # managed cloud API). We set only the attributes the inherited helpers
        # (``cleanup``/``empty_output_dict``) rely on.
        self.model_name = model_name
        self.is_base_model = is_base_model
        self.api_key = api_key
        self.max_workers = max_workers
        self.timeout = timeout
        self.region = region
        self.aws_profile = aws_profile
        self.default_reasoning_effort = default_reasoning_effort
        self.additional_model_request_fields = additional_model_request_fields or {}
        if self.default_reasoning_effort is not None:
            if self.additional_model_request_fields.get("thinking") is None:
                self.additional_model_request_fields["thinking"] = {"type": "adaptive"}

            self.additional_model_request_fields["output_config"] = {
                "effort": self.default_reasoning_effort
            }

        self.server_kwargs = kwargs
        self.additional_generation_kwargs = {}
        self.friendly_name = "Bedrock"

        # ``process`` is referenced by the inherited ``cleanup``; there is no
        # subprocess for a managed service so it stays None.
        self.process = None

        # Imported lazily so that merely importing this module does not trigger
        # the Bedrock control-plane model listing (a network call requiring
        # credentials). It runs once, when the first Bedrock model is built, and
        # is shared with the batch serving class via its class-level cache.
        from src.serving.batch.bedrock_batch_serving import BedrockBatchServing

        # Inference profile ids are not always returned by the listing APIs, so
        # warn rather than block to avoid false negatives. The left operand primes
        # the class-level cache with this instance's AWS profile, so the
        # (profile-agnostic) is_model_name_supported reads the same list.
        if BedrockBatchServing._get_available_models(
            self.aws_profile
        ) and not BedrockBatchServing.is_model_name_supported(model_name):
            logger.warning(
                "Model %s not found in the listed Bedrock models. Proceeding anyway; "
                "ensure it is a valid model id or inference profile id.",
                model_name,
            )

        self.client = self._build_client()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @property
    def _server_name(self) -> str:
        return "Bedrock"

    def _build_server_command(self, merged_kwargs: dict) -> None:
        """No subprocess is launched for the managed Bedrock service.

        Returns:
            None: Always None so the base lifecycle never spawns a process.
        """
        return None

    def _build_client(self):
        """Create the ``bedrock-runtime`` boto3 client.

        Botocore's internal retries are disabled in favour of the explicit
        :func:`tenacity` retry on :meth:`_invoke_converse`, mirroring how the
        other online serving classes disable client-side retries.

        Returns:
            botocore.client.BaseClient: The configured ``bedrock-runtime`` client.
        """
        config = Config(
            region_name=self.region,
            read_timeout=self.timeout,
            connect_timeout=60,
            retries={"max_attempts": 1, "mode": "standard"},
        )
        session = boto3.Session(profile_name=self.aws_profile)
        return session.client("bedrock-runtime", config=config)

    def load_model(self) -> None:
        """Ensure the Bedrock runtime client is initialised.

        No model is loaded locally; this only (re)creates the boto3 client if
        required and is safe to call multiple times.
        """
        if self.is_model_loaded():
            return

        self.client = self._build_client()
        logger.info(
            "Bedrock runtime client ready for model %s (region=%s)",
            self.model_name,
            self.region or "default",
        )

    def is_model_loaded(self) -> bool:
        """Check whether the Bedrock runtime client has been initialised.

        Returns:
            bool: True if the client is ready, False otherwise.
        """
        return getattr(self, "client", None) is not None

    def get_run_env(self) -> dict:
        """Get the runtime environment information.

        Returns:
            dict: Dictionary containing the boto3 version.
        """
        return {"boto3_version": version("boto3")}

    # ------------------------------------------------------------------ #
    # Request building
    # ------------------------------------------------------------------ #
    def _convert_content(self, content: str | list[dict]) -> list[dict]:
        """Convert OpenAI-style message content into Converse content blocks.

        Args:
            content (str | list[dict]): Either a string (text-only) or a list of
                OpenAI-style content parts (multimodal).

        Returns:
            list[dict]: Converse content blocks (``{"text": ...}`` / ``{"image": ...}``).

        Raises:
            ValueError: If a content part is of an unsupported type, or an image
                is provided as a non-base64 URL.
        """
        if isinstance(content, str):
            return [{"text": content}]
        if isinstance(content, list):
            blocks = []
            for part in content:
                part_type = part.get("type")
                if part_type == "text":
                    blocks.append({"text": part["text"]})
                elif part_type == "image_url":
                    image_url = part["image_url"]["url"]
                    if not image_url.startswith("data:"):
                        raise ValueError(
                            "Bedrock Converse requires base64-encoded image data "
                            f"URLs, got a non-data URL: {image_url[:32]}..."
                        )
                    prefix, base64_data = image_url.split(";base64,", 1)
                    media_type = prefix.split("data:", 1)[1]
                    image_format = media_type.split("/")[-1].lower()
                    if image_format == "jpg":
                        image_format = "jpeg"
                    blocks.append(
                        {
                            "image": {
                                "format": image_format,
                                "source": {"bytes": base64.b64decode(base64_data)},
                            }
                        }
                    )
                else:
                    raise ValueError(
                        f"Unsupported content type for Bedrock Converse: {part_type}"
                    )
            return blocks
        raise ValueError(f"Invalid content type: {type(content)}")

    def _split_generation_kwargs(self, generation_kwargs: dict) -> tuple[dict, dict]:
        """Split generation kwargs into Converse ``inferenceConfig`` and extra fields.

        Args:
            generation_kwargs (dict): Generation parameters supplied by the caller.

        Returns:
            tuple[dict, dict]: The ``inferenceConfig`` dict and any
                ``additionalModelRequestFields`` derived from the kwargs.
        """
        inference_config = {}
        additional_fields = {}
        for key, value in generation_kwargs.items():
            if value is None:
                continue
            if key in _INFERENCE_CONFIG_MAP:
                inference_config[_INFERENCE_CONFIG_MAP[key]] = value
            elif key in ("stop", "stop_sequences", "stopSequences"):
                inference_config["stopSequences"] = (
                    [value] if isinstance(value, str) else list(value)
                )
            elif key == "top_k":
                # top_k is model-specific and not part of inferenceConfig.
                additional_fields["top_k"] = value
            elif key in _IGNORED_KWARGS:
                continue
            else:
                logger.debug(
                    "Ignoring unsupported Bedrock Converse generation kwarg: %s", key
                )
        return inference_config, additional_fields

    def _build_converse_request(
        self, conversation: list[dict], generation_kwargs: dict
    ) -> dict:
        """Build the keyword arguments for a single ``converse`` call.

        Args:
            conversation (list[dict]): OpenAI-style role/content message dicts.
            generation_kwargs (dict): Generation parameters for this request.

        Returns:
            dict: Keyword arguments to pass to ``bedrock-runtime.converse``.
        """
        system_blocks = []
        messages = []
        for message in conversation:
            role = message["role"]
            content = message["content"]
            if role == "system":
                if isinstance(content, str):
                    system_blocks.append({"text": content})
                else:
                    system_blocks.extend(self._convert_content(content))
                continue
            messages.append({"role": role, "content": self._convert_content(content)})

        inference_config, additional_fields = self._split_generation_kwargs(
            generation_kwargs
        )

        request = {"modelId": self.model_name, "messages": messages}
        if system_blocks:
            request["system"] = system_blocks
        if inference_config:
            request["inferenceConfig"] = inference_config
        merged_additional = {
            **self.additional_model_request_fields,
            **additional_fields,
        }
        if merged_additional:
            request["additionalModelRequestFields"] = merged_additional
        return request

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @retry(
        retry=retry_if_exception(_is_retryable_bedrock_error),
        wait=wait_random_exponential(multiplier=1, min=3, max=60),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def _invoke_converse(self, request: dict) -> dict:
        """Send a single Converse request with retry logic.

        Args:
            request (dict): Keyword arguments for ``bedrock-runtime.converse``.

        Returns:
            dict: The raw Converse response.
        """
        return self.client.converse(**request)

    async def run_generate_chat_responses_coroutine(
        self, batch_messages: list[list], generation_kwargs: list[dict] | dict
    ) -> list:
        """Async coroutine that runs Converse requests concurrently.

        Args:
            batch_messages (list[list]): List of conversations, where each
                conversation is a list of message dicts.
            generation_kwargs (list[dict] | dict): Per-conversation generation
                parameters, or a single dict applied to all conversations.

        Returns:
            list: List of raw Converse responses (or the raised exception for any
                request that failed) in the same order as ``batch_messages``.
        """
        loop = asyncio.get_running_loop()
        semaphore = asyncio.Semaphore(self.max_workers)
        # boto3 clients are blocking but thread-safe. ``asyncio.to_thread`` would
        # dispatch onto the loop's default executor, whose size is min(32, cpu+4)
        # and ignores ``self.max_workers``; use a dedicated executor sized to
        # ``max_workers`` so the documented concurrency knob is actually honoured.
        executor = ThreadPoolExecutor(max_workers=self.max_workers)

        async def _bounded(messages, gen_kwargs):
            async with semaphore:
                try:
                    request = self._build_converse_request(messages, gen_kwargs)
                    return await loop.run_in_executor(
                        executor, self._invoke_converse, request
                    )
                except Exception as exc:  # noqa: BLE001
                    # Return the exception so parse_output can record it per row,
                    # mirroring the per-item error handling of LiteLLMServing.
                    logger.warning("Bedrock Converse request failed: %s", exc)
                    return exc

        if isinstance(generation_kwargs, dict):
            generation_kwargs = [generation_kwargs] * len(batch_messages)
        try:
            responses = await asyncio.gather(
                *(
                    _bounded(m, g)
                    for m, g in zip(batch_messages, generation_kwargs, strict=True)
                )
            )
        finally:
            executor.shutdown(wait=False)
        return responses

    def generate_chat_responses(
        self, conversations: list[list] | list, generation_kwargs: list[dict] | dict
    ) -> list:
        """Run Converse requests for a list of conversations.

        Args:
            conversations: List of conversations, where each conversation is a list
                of message dicts.
            generation_kwargs (list[dict] | dict): Per-conversation generation
                parameters, or a single dict applied to all conversations.

        Returns:
            list: List of raw Converse responses (or exceptions for failed requests).
        """
        return asyncio.run(
            self.run_generate_chat_responses_coroutine(conversations, generation_kwargs)
        )

    def generate_completions(
        self, prompts: list, generation_kwargs: list[dict] | dict
    ) -> list:
        """Not supported: the Converse API is chat-only.

        Raises:
            NotImplementedError: Always. Use :meth:`generate_chat_responses` instead.
        """
        raise NotImplementedError(
            "generate_completions is not supported for Bedrock. "
            "Please use generate_chat_responses for BedrockServing."
        )

    # ------------------------------------------------------------------ #
    # Tokenization
    # ------------------------------------------------------------------ #
    def tokenize(self, message: list) -> None:
        """Bedrock does not expose a tokenizer API.

        Args:
            message (list): The message to tokenize (unused).

        Returns:
            None: Always None.
        """
        return None

    def batch_tokenize(self, messages: list[list]) -> list:
        """Tokenize multiple messages in batch.

        Args:
            messages (list[list]): List of messages to tokenize.

        Returns:
            list: List of None responses (Bedrock provides no tokenizer API).
        """
        logger.warning(
            "Bedrock does not provide a tokenizer API. Returning None for all tokenizations."
        )
        return [None for _ in messages]

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #
    def get_response(self, output: dict) -> str:
        """Extract the response text from a Converse response.

        Args:
            output (dict): A raw Converse response.

        Returns:
            str: The concatenated text content of the assistant message.
        """
        blocks = output["output"]["message"]["content"]
        return "".join(block["text"] for block in blocks if "text" in block)

    def get_reasoning_content(self, output: dict) -> str | None:
        """Extract the reasoning content from a Converse response, if present.

        Args:
            output (dict): A raw Converse response.

        Returns:
            str | None: The concatenated reasoning text, or None if absent.
        """
        blocks = output["output"]["message"]["content"]
        reasoning = [
            block["reasoningContent"]["reasoningText"].get("text", "")
            for block in blocks
            if "reasoningContent" in block
            and "reasoningText" in block["reasoningContent"]
        ]
        return "".join(reasoning) if reasoning else None

    def _get_tool_calls(self, output: dict) -> list | None:
        """Extract tool-use blocks from a Converse response, if present.

        Args:
            output (dict): A raw Converse response.

        Returns:
            list | None: A list of ``toolUse`` blocks, or None if there are none.
        """
        blocks = output["output"]["message"]["content"]
        tool_calls = [block["toolUse"] for block in blocks if "toolUse" in block]
        return tool_calls or None

    def parse_output(self, output: dict, custom_id: str | None = None) -> dict:
        """Parse a Converse response into the standard structured output dict.

        Args:
            output (dict): A raw Converse response, or the Exception raised while
                generating it.
            custom_id (str, optional): The custom ID associated with the input.
                Defaults to None.

        Returns:
            dict: The parsed output, matching the structure used by the other
                serving classes.
        """
        if isinstance(output, BaseException):
            return {
                "finish_reasons": None,
                "responses": None,
                "reasoning_contents": None,
                "custom_ids": custom_id,
                "token_usages": None,
                "function_calls": None,
                "tool_calls": None,
                "logprobs": None,
                "errors": type(output).__name__,
            }

        try:
            usage = output.get("usage", {})
            stop_reason = output.get("stopReason")
            parsed_output = {
                "finish_reasons": _STOP_REASON_MAP.get(stop_reason, stop_reason),
                "responses": self.get_response(output),
                "reasoning_contents": self.get_reasoning_content(output),
                "custom_ids": custom_id,
                "token_usages": {
                    "prompt_tokens": usage.get("inputTokens"),
                    "completion_tokens": usage.get("outputTokens"),
                    "total_tokens": usage.get("totalTokens"),
                },
                "function_calls": None,
                "tool_calls": self._get_tool_calls(output),
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


if __name__ == "__main__":
    # Requires valid AWS credentials with Bedrock access in the environment.
    # Claude 4.x models require a cross-region inference profile id.
    model_name = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

    bedrock_model = BedrockServing(model_name=model_name)
    bedrock_model.load_model()

    messages = [{"role": "user", "content": "ELI5: Why is the sky blue"}]

    # run batch generation
    responses = bedrock_model.generate_chat_responses(
        [messages for _ in range(5)], {"max_tokens": 256, "temperature": 0.0}
    )
    print([bedrock_model.parse_output(r) for r in responses])
