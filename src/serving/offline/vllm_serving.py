from typing import Any

import importlib_metadata
from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    load_harmony_encoding,
)

from src.base_logger import get_logger
from src.serving.offline.base_offline_serving import BaseOfflineServing

logger = get_logger(__name__)

try:
    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.inputs import TokensPrompt
    from vllm.reasoning import ReasoningParserManager
    from vllm.v1.engine.exceptions import EngineDeadError

    ACCEPTED_SAMPLING_PARAMS = set(SamplingParams.__annotations__.keys())


except ImportError:
    logger.warning("vLLM is not installed. Please install vLLM to use VLLMServing.")

REASONING_EFFORT_MAPPING = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


class VLLMServing(BaseOfflineServing):
    """
    A serving class that uses vLLM for language model completions.

    This class provides methods for generating responses from language models using the vLLM API.
    """

    def __init__(
        self,
        model_name: str,
        is_base_model: bool = False,
        dtype: str = "auto",
        enable_prefix_caching: bool = True,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int | str = "auto",
        seed: int = 1234,
        reasoning_parser: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the VLLMServing instance.

        Args:
            model_name (str): The model path or Hugging Face model ID.
            is_base_model (bool, optional): Whether this is a base model. If True, applies a
                generic base model chat template. Defaults to False.
            dtype (str, optional): The data type for model weights (e.g. "bfloat16", "float16").
                Defaults to "auto".
            enable_prefix_caching (bool, optional): Whether to enable prefix caching. Defaults to True.
            gpu_memory_utilization (float, optional): Fraction of GPU memory to use. Defaults to 0.9.
            tensor_parallel_size (int | str, optional): Number of GPUs for tensor parallelism.
                Setting to "auto" uses all available GPUs. Defaults to "auto".
            seed (int, optional): Random seed for reproducibility. Defaults to 1234.
            reasoning_parser (str | None, optional): Name of the vLLM reasoning parser to use for
                extracting chain-of-thought reasoning from model outputs. Defaults to None.
            enable_thinking (bool, optional): Whether to enable thinking mode via chat template
                kwargs (Qwen models). Defaults to None.
            thinking (bool, optional): Whether to enable thinking mode via chat template kwargs
                (DeepSeek models). Defaults to None.
            thinking_mode (str | None, optional): Thinking mode setting passed via chat template
                kwargs. Defaults to None.
            reasoning_effort (str | None, optional): Reasoning effort level ("high", "medium", "low", "none"),
                passed via generation kwargs. Defaults to None.
            **kwargs: Additional keyword arguments forwarded to the vLLM ``LLM`` constructor.
        """
        logger.info("Initializing offline VLLMServing for model: %s", model_name)
        if is_base_model:
            with open("chat_templates/base_model.jinja") as f:
                chat_template = f.read()
            self.chat_template = chat_template
        else:
            self.chat_template = None

        # handle thinking kwargs
        for thinking_kwarg in [
            "enable_thinking",
            "thinking",  # Deepseek models
            "thinking_mode",
            "reasoning_effort",  # Mistral Small/Medium
            "reasoning",  # Cohere Command A+
        ]:
            if thinking_kwarg in kwargs:
                thinking_value = kwargs.pop(thinking_kwarg)
                if thinking_value is not None:
                    self.chat_template_kwargs = {thinking_kwarg: thinking_value}
                    break
        else:
            self.chat_template_kwargs = None

        if tensor_parallel_size == "auto":
            import torch

            tensor_parallel_size = torch.cuda.device_count()

        self.model_name = model_name
        self.init_kwargs = {
            "dtype": dtype,
            "enable_prefix_caching": enable_prefix_caching,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "seed": seed,
            **kwargs,
        }
        self.is_model_loaded = False

        if "gpt-oss" in self.model_name:
            # in case of gpt-oss, we need to use the harmony encoding
            self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        self.reasoning_parser = reasoning_parser

    def load_model(self) -> None:
        """Load the vLLM model into memory.

        This method is idempotent: if the model is already loaded it returns immediately.
        On first call it initialises the ``LLM`` engine, retrieves the model's default
        generation config, and (if configured) instantiates the reasoning parser.
        """
        if self.is_model_loaded:
            # no op as model is already loaded
            return
        else:
            logger.info("Loading vLLM model: %s", self.model_name)
            self.llm = LLM(model=self.model_name, **self.init_kwargs)
            model_generation_config = (
                self.llm.llm_engine.model_config.try_get_generation_config()
            )
            accepted_keys = model_generation_config.keys() & ACCEPTED_SAMPLING_PARAMS
            self.model_default_generation_config = {
                key: model_generation_config[key] for key in accepted_keys
            }

            self.tokenizer = self.llm.get_tokenizer()
            if self.reasoning_parser is not None:
                Parser = ReasoningParserManager.get_reasoning_parser(
                    self.reasoning_parser
                )
                self.reasoning_parser = Parser(
                    self.tokenizer, chat_template_kwargs=self.chat_template_kwargs
                )
            self.is_model_loaded = True

    def get_run_env(self) -> dict:
        """Get the run environment.

        Returns:
            dict: The run environment.
        """
        return {
            "transformers_version": importlib_metadata.version("transformers"),
            "vllm_version": importlib_metadata.version("vllm"),
        }

    def prepare_generation_kwargs(
        self, generation_kwargs: dict | list[dict]
    ) -> SamplingParams | list[SamplingParams]:
        """Prepare the generation kwargs for vLLM.

        This method checks if the generation kwargs are in the correct format and converts them if necessary.

        Args:
            generation_kwargs (dict or list of dicts): The generation kwargs to prepare.

        Returns:
            SamplingParams or list of SamplingParams: The prepared generation kwargs.
        """

        if isinstance(generation_kwargs, dict):
            for key in self.model_default_generation_config:
                if key not in generation_kwargs:
                    generation_kwargs[key] = self.model_default_generation_config[key]
            return SamplingParams(**generation_kwargs)
        elif isinstance(generation_kwargs, list):
            sampling_params = []
            for gen_kwargs in generation_kwargs:
                for key in self.model_default_generation_config:
                    if key not in gen_kwargs:
                        gen_kwargs[key] = self.model_default_generation_config[key]
                sampling_params.append(SamplingParams(**gen_kwargs))
            return sampling_params
        else:
            raise ValueError("generation_kwargs must be of type dict or list of dicts")

    @staticmethod
    def is_list_of_list_of_ints(variable: Any) -> bool:
        """
        Checks if a variable is a list where every element is a list of integers.
        """
        if not isinstance(variable, list):
            return False

        for inner_list in variable:
            # Check if the inner element is a list.
            if not isinstance(inner_list, list):
                return False

            # Check if every item within the inner list is an integer.
            if not all(isinstance(item, int) for item in inner_list):
                return False

        return True

    def prepare_prompts(self, conversations: list[list] | list) -> list:
        """Normalise a batch of prompts into a format accepted by vLLM.

        If ``conversations`` is a list of token-ID lists (i.e. a list of lists of
        integers), each inner list is wrapped in a ``TokensPrompt`` object.  If
        ``conversations`` is not already a ``list``, it is converted to one.

        Args:
            conversations (list[list] | list): A batch of prompts as plain strings,
                lists of integer token IDs, or vLLM ``TokensPrompt`` objects.

        Returns:
            list: The normalised list of prompts ready to pass to ``llm.generate()``
                or ``llm.chat()``.
        """
        # check if conversations is a list of list of token ids, if so, convert to list of TokensPrompt
        if self.is_list_of_list_of_ints(conversations):
            conversations = [
                TokensPrompt(prompt_token_ids=conv) for conv in conversations
            ]
        elif not isinstance(conversations, list):
            logger.info("Converting batch_messages to type: list")
            conversations = list(conversations)

        return conversations

    def generate_chat_responses(
        self, conversations: list[list] | list, generation_kwargs: dict | list[dict]
    ) -> list | None:
        """Generate chat responses for a batch of conversations using vLLM.

        Applies the model's chat template (or a custom base-model template) and
        calls ``llm.chat()`` for standard models, or ``llm.generate()`` with
        Harmony-encoded token IDs for ``gpt-oss`` models.

        Args:
            conversations (list[list] | list): A batch of conversations, where each
                conversation is a list of message dicts with ``role`` and ``content``
                keys, or a list of pre-tokenised token-ID lists.
            generation_kwargs (dict | list[dict]): Sampling parameters passed to
                ``SamplingParams``.  Accepts an optional ``add_generation_prompt``
                key (default ``True``) and, for ``gpt-oss`` models, a
                ``reasoning_effort`` key.  Supply a single dict to use the same
                parameters for all conversations, or a list of dicts for
                per-request parameters.

        Returns:
            list | None: A list of vLLM ``RequestOutput`` objects on success.
                Returns ``None`` if a ``ValueError`` is raised during generation;
                any unfinished requests are drained before returning.

        Raises:
            EngineDeadError: Re-raised immediately if the vLLM engine dies, as
                recovery is not possible in that state.
        """
        try:
            add_generation_prompt = generation_kwargs.pop("add_generation_prompt", True)
            conversations = self.prepare_prompts(conversations)

            if "gpt-oss" in self.model_name:
                # Uses Harmony
                stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
                generation_kwargs["stop_token_ids"] = stop_token_ids

                roles_mapping = {
                    "system": Role.DEVELOPER,
                    "developer": Role.DEVELOPER,
                    "user": Role.USER,
                    "assistant": Role.ASSISTANT,
                }

                batch_prefill_ids = []

                if isinstance(generation_kwargs, dict):
                    effort = generation_kwargs.pop("reasoning_effort", "")
                    reasoning_efforts = REASONING_EFFORT_MAPPING.get(
                        effort.lower(), ReasoningEffort.MEDIUM
                    )
                else:
                    reasoning_efforts = []
                    for gen_kwargs in generation_kwargs:
                        effort = gen_kwargs.pop("reasoning_effort", "")
                        reasoning_efforts.append(
                            REASONING_EFFORT_MAPPING.get(
                                effort.lower(), ReasoningEffort.MEDIUM
                            )
                        )

                for i, sample_messages in enumerate(conversations):
                    conversation = Conversation.from_messages(
                        [
                            Message.from_role_and_content(
                                Role.SYSTEM,
                                SystemContent.new().with_reasoning_effort(
                                    reasoning_efforts[i]
                                    if isinstance(generation_kwargs, list)
                                    else reasoning_efforts
                                ),
                            ),
                        ]
                        + [
                            Message.from_role_and_content(
                                roles_mapping[message["role"]], message["content"]
                            )
                            for message in sample_messages
                        ]
                    )
                    batch_prefill_ids.append(
                        TokensPrompt(
                            prompt_token_ids=self.encoding.render_conversation_for_completion(
                                conversation, Role.ASSISTANT
                            )
                        )
                    )

                responses = self.llm.generate(
                    prompts=batch_prefill_ids,
                    sampling_params=SamplingParams(**generation_kwargs),
                )
            else:
                sampling_params = self.prepare_generation_kwargs(generation_kwargs)
                responses = self.llm.chat(
                    messages=conversations,
                    sampling_params=sampling_params,
                    chat_template=self.chat_template,
                    add_generation_prompt=add_generation_prompt,
                    chat_template_kwargs=self.chat_template_kwargs,
                )
        except EngineDeadError as e:
            logger.error("vLLM engine is dead. Re-raising the issue and exiting.")
            raise e
        except ValueError as e:
            logger.exception(e)
            if self.llm.llm_engine.has_unfinished_requests():
                logger.warning(
                    "vLLM has unfinished requests!\nWaiting for vLLM to finish requests before proceeding to the next task."
                )
                _ = self.llm.llm._run_engine()
                logger.warning(
                    "vLLM has completed all the requests. Proceeding to the next task."
                )
            return None
        return responses

    def generate_completions(
        self, prompts: list[list] | list, generation_kwargs: dict | list[dict]
    ) -> list | None:
        """Generate completions for a batch of prompts using vLLM.

        It is intended for raw text-completion or pre-tokenised inputs.

        Args:
            prompts (list[list] | list): A batch of prompts. Each element may be a
                plain string or a pre-tokenised list of integer token IDs. Lists of
                token-ID lists are automatically converted to ``TokensPrompt`` objects
                via ``prepare_prompts()``.
            generation_kwargs (dict | list[dict]): Sampling parameters passed to
                ``SamplingParams``.  Supply a single dict to use the same parameters
                for every prompt, or a list of dicts (one per prompt) for
                per-request parameters.  Default values from the model's generation
                config are merged in for any keys that are absent.

        Returns:
            list | None: A list of vLLM ``RequestOutput`` objects on success.
                Returns ``None`` if a ``ValueError`` is raised during generation;
                any unfinished requests are drained before returning.

        Raises:
            EngineDeadError: Re-raised immediately if the vLLM engine dies, as
                recovery is not possible in that state.
        """
        try:
            sampling_params = self.prepare_generation_kwargs(generation_kwargs)
            prompts = self.prepare_prompts(prompts)

            responses = self.llm.generate(
                prompts=prompts,
                sampling_params=sampling_params,
            )
        except EngineDeadError as e:
            logger.error("vLLM engine is dead. Re-raising the issue and exiting.")
            raise e
        except ValueError as e:
            logger.exception(e)
            if self.llm.llm_engine.has_unfinished_requests():
                logger.warning(
                    "vLLM has unfinished requests!\nWaiting for vLLM to finish requests before proceeding to the next task."
                )
                _ = self.llm.llm._run_engine()
                logger.warning(
                    "vLLM has completed all the requests. Proceeding to the next task."
                )
            return None
        return responses

    def get_response(self, output: Any) -> str:
        """Get the response text from a vLLM ``RequestOutput`` object.

        For ``gpt-oss`` models the token IDs are decoded via the Harmony encoding and
        wrapped appropriately with ``<think>``/``</think>`` tags.  For all other models
        the raw text of the first completion is returned.

        Args:
            output: A vLLM ``RequestOutput`` object produced by ``llm.generate()`` or
                ``llm.chat()``.

        Returns:
            str: The response text.
        """
        if "gpt-oss" in self.model_name:
            output_tokens = output.outputs[0].token_ids
            # text = output.outputs[0].text

            try:
                entries = self.encoding.parse_messages_from_completion_tokens(
                    output_tokens, Role.ASSISTANT
                )
                if len(entries) >= 2:
                    extracted_reasoning_text = entries[0].to_dict()["content"][0][
                        "text"
                    ]
                    extracted_answer_text = entries[1].to_dict()["content"][0]["text"]
                    extracted_text = (
                        "<think>"
                        + extracted_reasoning_text
                        + "</think>"
                        + extracted_answer_text
                    )
                else:
                    extracted_text = (
                        "<think></think>" + entries[0].to_dict()["content"][0]["text"]
                    )
            except Exception as e:
                text = output.outputs[0].text
                logger.error(
                    "Failed to parse GPT-OSS output: %s. Error: %s",
                    output.outputs[0].text,
                    e,
                )
                logger.warning(
                    "Trying to extract the response from the raw text output."
                )
                # try to extract the text using the sep assistantfinal
                if "assistantfinal" in text:
                    extracted_reasoning_text, extracted_answer_text = text.split(
                        "assistantfinal"
                    )
                    extracted_text = (
                        "<think>"
                        + extracted_reasoning_text
                        + "</think>"
                        + extracted_answer_text
                    )
                else:
                    extracted_text = text
            return extracted_text
        else:
            return output.outputs[0].text

    def tokenize_conversations(self, conversations: list[list]) -> None:
        """Tokenize a batch of conversations.

        Not implemented for vLLM — tokenisation is handled internally by the engine.
        Always returns ``None``.

        Args:
            conversations (list[list]): The batch of conversations to tokenize.

        Returns:
            None
        """
        return None

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        """Tokenize a batch of texts using the model's tokenizer.

        Special tokens are not added (``add_special_tokens=False``).

        Args:
            texts (list[str]): The batch of texts to tokenize.

        Returns:
            list[list[int]]: A list of token-ID sequences, one per input text.
        """
        return self.tokenizer(texts, add_special_tokens=False)["input_ids"]

    def parse_output(self, output: Any, custom_id: str | None = None) -> dict:
        """Parse a single generated output into a structured dict.

        Args:
            output: A single vLLM RequestOutput object.
            custom_id (str, optional): The custom ID associated with the input. Defaults to None.

        Returns:
            dict: The parsed output.
        """
        try:
            response_text = self.get_response(output)

            reasoning_contents = None
            if self.reasoning_parser is not None:
                reasoning_contents, response_text = (
                    self.reasoning_parser.extract_reasoning(response_text, output)
                )
            elif "gpt-oss" in self.model_name:
                # we use the <think></think> tags to extract the reasoning for gpt-oss models
                if "</think>" in response_text:
                    parts = response_text.split("</think>")
                    if len(parts) == 2:
                        reasoning_contents = parts[0].replace("<think>", "")
                        response_text = parts[1]
                else:
                    logger.warning(
                        "No </think> tag found in the %s response. Unable to extract reasoning contents.",
                        self.model_name,
                    )

            # error = "EmptyGenerationError" if output.outputs[0].text == "" else None
            # TODO: handle errors
            error = None

            parsed_output = {
                "finish_reasons": output.outputs[0].finish_reason,
                "responses": response_text,
                "reasoning_contents": reasoning_contents,
                "custom_ids": custom_id,
                "token_usages": {
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                    "total_tokens": len(output.prompt_token_ids)
                    + len(output.outputs[0].token_ids),
                },
                "function_calls": None,
                "tool_calls": None,
                "logprobs": None,
                "errors": error,
                "tokenized_prompts": output.prompt_token_ids,
                "response_tokens": output.outputs[0].token_ids,
                "stop_reasons": output.outputs[0].stop_reason,
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
                "tokenized_prompts": None,
                "response_tokens": None,
                "stop_reasons": None,
            }
        return parsed_output

    def empty_output_dict(self, custom_ids: str | None = None) -> dict:
        """Return a dict with the same structure as the output of ``parse_output()``, but with all values set to ``None``.

        This is used to fill in outputs for conversations that were skipped due to generation failures.

        Args:
            custom_ids (str | None, optional): The custom ID to include in the output dict. Defaults to None.

        Returns:
            dict: A dict with the same structure as the output of ``parse_output()``, but with all values set to ``None``.
        """
        return {
            "finish_reasons": None,
            "responses": None,
            "reasoning_contents": None,
            "custom_ids": custom_ids,
            "token_usages": None,
            "function_calls": None,
            "tool_calls": None,
            "logprobs": None,
            "errors": None,
            "tokenized_prompts": None,
            "response_tokens": None,
            "stop_reasons": None,
        }

    def cleanup(self) -> None:
        """Delete the served vLLM model and free associated memory resources.

        Safely deletes a vLLM model instance and cleans up distributed environment
        and memory. Handles both v1 and older vLLM versions by attempting to delete
        different engine components.

        Note:
            This function calls cleanup_dist_env_and_memory with shutdown_ray=True
            to ensure proper cleanup of Ray resources.
        """
        if not self.is_model_loaded:
            logger.info("Model is not loaded; no cleanup necessary.")
            return

        try:
            # v1 vllm
            del self.llm.llm_engine.engine_core
        except Exception:
            del self.llm.llm_engine.model_executor

        if hasattr(self, "llm"):
            del self.llm

        cleanup_dist_env_and_memory()
        logger.info("vLLM model %s deleted and memory freed.", self.model_name)
