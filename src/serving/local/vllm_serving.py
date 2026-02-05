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
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput

from src.base_logger import get_logger
from src.serving.local.base_serving import BaseServing

logger = get_logger(__name__)


ACCEPTED_SAMPLING_PARAMS = set(SamplingParams.__annotations__.keys())


class VLLMServing(BaseServing):
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
        enable_thinking: bool | None = None,
        thinking: bool | None = None,
        **kwargs,
    ) -> None:
        """Initialize the VLLMServing instance.

        Args:
            model (str): The model to use.
            is_base_model (bool, optional): Whether this is a base model. Defaults to False.
            dtype (str, optional): The data type to use. Defaults to "bfloat16".
            enable_prefix_caching (bool, optional): Whether to enable prefix caching. Defaults to True.
            gpu_memory_utilization (float, optional): The GPU memory utilization. Defaults to 0.9.
            tensor_parallel_size (int | str, optional): The tensor parallel size. Setting to "auto" will automatically determine the size based on available GPUs. Defaults to "auto".
            seed (int, optional): The seed to use. Defaults to 1234.
            enable_thinking (bool, optional): Whether to enable thinking (Qwen models). Defaults to None.
            thinking (bool, optional): Whether to enable thinking (DeepSeek models). Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        logger.info(f"Initializing VLLMServing for model: {model_name}")
        if is_base_model:
            with open("chat_templates/base_model.jinja") as f:
                chat_template = f.read()
            self.chat_template = chat_template
        else:
            self.chat_template = None

        if enable_thinking is not None:
            self.chat_template_kwargs = {"enable_thinking": enable_thinking}
        elif thinking is not None:
            self.chat_template_kwargs = {"thinking": thinking}
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

    def load_model(self) -> None:
        """Load the vLLM model.

        Note:
            The model is loaded during initialization, so this method does not perform any action.
        """
        if self.is_model_loaded:
            # no op as model is already loaded
            return
        else:
            logger.info(f"Loading vLLM model: {self.model_name}")
            self.llm = LLM(model=self.model_name, **self.init_kwargs)
            self.model_default_generation_config = (
                self.llm.llm_engine.model_config.try_get_generation_config()
            )
            model_generation_config = (
                self.llm.llm_engine.model_config.try_get_generation_config()
            )
            accepted_keys = model_generation_config.keys() & ACCEPTED_SAMPLING_PARAMS
            self.model_default_generation_config = {
                key: model_generation_config[key] for key in accepted_keys
            }

            self.tokenizer = self.llm.get_tokenizer()
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

    def generate(
        self, messages: list, logprobs: bool = False, **generation_kwargs
    ) -> list:
        """Generate a response.

        Args:
            messages (list): The messages to generate a response for.
            logprobs (bool, optional): Whether to generate logprobs. Defaults to False.
            **generation_kwargs: Additional keyword arguments.

        Returns:
            list: The generated response.
        """
        response = self.llm.chat(
            messages=messages,
            sampling_params=SamplingParams(**generation_kwargs),
            chat_template=self.chat_template,
            add_generation_prompt=True,
            chat_template_kwargs=self.chat_template_kwargs,
        )
        return response

    def generate_until_answer_tag(
        self,
        batch_messages: list,
        generation_kwargs: dict,
    ) -> tuple[dict, list]:
        """Generate a response until the answer tag is found.

        Args:
            batch_messages (list): The messages to generate a response for.
            generation_kwargs (dict): The generation parameters to use.

        Returns:
            tuple[dict, list]: The generated outputs (parsed responses) and the raw responses.
        """
        assert "answer_tag" in generation_kwargs, "answer_tag is required"
        _generation_kwargs = generation_kwargs.copy()
        _generation_kwargs["stop"] = _generation_kwargs.pop("answer_tag")

        # HACK
        # max_token_id = max(self.tokenizer.get_vocab().values())
        # _generation_kwargs["allowed_token_ids"] = list(range(max_token_id + 1))

        # ensure stop token is printed
        _generation_kwargs["include_stop_str_in_output"] = True

        sampling_params = SamplingParams(**_generation_kwargs)

        logger.info(
            "Sampling parameters for generation until answer tag:\n%s",
            sampling_params,
        )
        responses = self.llm.chat(
            messages=batch_messages,
            sampling_params=sampling_params,
            chat_template=self.chat_template,
            add_generation_prompt=True,
            chat_template_kwargs=self.chat_template_kwargs,
        )

        # recreate batch_messages
        outputs = self.parse_outputs(responses)
        return outputs, responses

    def calculate_logprobs(
        self,
        batch_messages: list,
        generate_to_answer_tag: bool = True,
        answers: list | None = None,
        answer_tag_separator: str = "",
        **generation_kwargs,
    ) -> list:
        """Calculate the logprobs of the correct answer.

        Args:
            batch_messages (list): The messages to calculate the logprobs for.
            generate_to_answer_tag (bool, optional): Whether to generate to the answer tag. Defaults to True.
            answers (list, optional): The answers to calculate the logprobs for. Defaults to [].
            answer_tag_separator (str, optional): The answer tag separator. Defaults to "".
            **generation_kwargs: Additional keyword arguments.

        Returns:
            list: The logprobs of the correct answer.
        """
        ids = []
        prompt_tokens = []
        answer_tokens = []
        input_lengths = []
        answers = [answer_tag_separator + answer for answer in answers]
        answer_tokens = self.tokenizer(answers, add_special_tokens=False)["input_ids"]

        if generate_to_answer_tag:
            outputs, responses = self.generate_until_answer_tag(
                batch_messages, generation_kwargs
            )

            for i in range(len(batch_messages)):
                if (
                    outputs["finish_reasons"][i] != "stop"
                    or outputs["stop_reasons"][i] is None
                ):
                    logger.info(
                        "Skipping id %d because it did not generate the correct answer tag",
                        i,
                    )
                    continue

                _prompt_tokens = (
                    outputs["tokenized_prompts"][i]
                    + outputs["response_tokens"][i]
                    + answer_tokens[i]
                )

                input_lengths.append(outputs["total_tokens"][i])
                prompt_tokens.append(TokensPrompt(prompt_token_ids=_prompt_tokens))
                ids.append(i)
        else:
            input_tokens_list = self.tokenizer.apply_chat_template(
                batch_messages,
                continue_final_message=True,
            )
            for i, input_tokens in enumerate(input_tokens_list):
                input_lengths.append(len(input_tokens))
                prompt_tokens.append(
                    TokensPrompt(prompt_token_ids=input_tokens + answer_tokens[i])
                )
                ids.append(i)

        _generation_kwargs = generation_kwargs.copy()
        _generation_kwargs["max_tokens"] = 1
        _generation_kwargs["prompt_logprobs"] = 1
        _generation_kwargs.pop("answer_tag", None)

        # HACK
        # max_token_id = max(self.tokenizer.get_vocab().values())
        # _generation_kwargs["allowed_token_ids"] = list(range(max_token_id + 1))

        _sampling_params = SamplingParams(**_generation_kwargs)

        logger.info(
            "Sampling parameters for calculating the log probs of the correct answer:\n%s",
            _sampling_params,
        )

        final_responses = self.llm.generate(
            prompts=prompt_tokens, sampling_params=_sampling_params
        )

        # parse logprobs
        # TODO find a better way to handle the extraction of the logprobs
        for id, final_response, input_len in zip(
            ids, final_responses, input_lengths, strict=True
        ):
            token_ids = final_response.prompt_token_ids[input_len:]
            logprobs = final_response.prompt_logprobs[input_len:]

            output_logprobs = []
            cumulative_logprob = 0.0
            for token, logprob in zip(token_ids, logprobs, strict=True):
                output_logprobs.append({str(token): logprob[token].logprob})
                cumulative_logprob += logprob[token].logprob

            if generate_to_answer_tag:
                responses[id].outputs[0].logprobs = output_logprobs
                responses[id].outputs[0].cumulative_logprob = cumulative_logprob
                responses[id].outputs[0].text = outputs["responses"][id] + answers[id]
                responses[id].outputs[0].token_ids = (
                    outputs["response_tokens"][id] + answer_tokens[id]
                )
            else:
                final_responses[id].outputs[0].logprobs = output_logprobs
                final_responses[id].outputs[0].cumulative_logprob = cumulative_logprob
                final_responses[id].outputs[0].text = answer_tag_separator + answers[id]
                final_responses[id].outputs[0].token_ids = answer_tokens[id]

        if generate_to_answer_tag:
            return responses
        else:
            return final_responses

    def batch_generate(
        self,
        batch_messages: list[list],
        use_logprobs: bool = False,
        generate_to_answer_tag: bool = True,
        answers: list | None = None,
        answer_tag_separator: str = "",
        **generation_kwargs,
    ) -> list:
        """Batch generate a response.

        Args:
            batch_messages (list[list]): The messages to generate a response for.
            use_logprobs (bool, optional): Whether to use logprobs. Defaults to False.
            generate_to_answer_tag (bool, optional): Whether to generate to the answer tag. Defaults to True.
            answers (list, optional): The answers to generate a response for. Defaults to [].
            answer_tag_separator (str, optional): The answer tag separator. Defaults to "".
            **generation_kwargs: Additional keyword arguments.

        Returns:
            list: The generated response.
        """
        if type(batch_messages) is not list:
            logger.info("Converting batch_messages to type: list")
            batch_messages = list(batch_messages)

        generation_kwargs.update(self.model_default_generation_config)

        if use_logprobs is True:
            responses = self.calculate_logprobs(
                batch_messages,
                generate_to_answer_tag=generate_to_answer_tag,
                answers=answers,
                answer_tag_separator=answer_tag_separator,
                **generation_kwargs,
            )
            return responses
        elif "stop" in generation_kwargs:
            outputs, responses = self.generate_until_answer_tag(
                batch_messages, generation_kwargs
            )

            sampling_params_list = []
            ids = []
            prompt_tokens = []
            for i in range(len(batch_messages)):
                if (
                    outputs["finish_reasons"][i] != "stop"
                    or outputs["stop_reasons"][i] is None
                ):
                    logger.info(
                        "Skipping id %d because it did not generate the correct answer tag",
                        i,
                    )
                    continue

                _gen_kwargs = generation_kwargs.copy()
                _gen_kwargs.pop("answer_tag", None)
                _gen_kwargs["max_tokens"] = generation_kwargs["max_tokens"] - len(
                    outputs["response_tokens"][i]
                )
                if _gen_kwargs["max_tokens"] <= 0:
                    continue

                _prompt_tokens = list(outputs["tokenized_prompts"][i]) + list(
                    outputs["response_tokens"][i]
                )
                prompt_tokens.append(TokensPrompt(prompt_token_ids=_prompt_tokens))
                sampling_params_list.append(SamplingParams(**_gen_kwargs))
                ids.append(i)

            # reset generation_kwargs to use the stop token
            final_responses = self.llm.generate(
                prompts=prompt_tokens,
                sampling_params=sampling_params_list,
            )

            for id, response in zip(ids, final_responses, strict=True):
                responses[id].outputs[0].text = (
                    responses[id].outputs[0].text + response.outputs[0].text
                )
        else:
            if "gpt-oss" in self.model_name:
                # Uses Harmony
                stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
                generation_kwargs["stop_token_ids"] = stop_token_ids

                roles_mapping = {
                    "system": Role.SYSTEM,
                    "developer": Role.DEVELOPER,
                    "user": Role.USER,
                    "assistant": Role.ASSISTANT,
                }

                batch_prefill_ids = []

                for sample_messages in batch_messages:
                    conversation = Conversation.from_messages(
                        [
                            Message.from_role_and_content(
                                Role.SYSTEM,
                                SystemContent.new().with_reasoning_effort(
                                    ReasoningEffort.MEDIUM
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
                responses = self.llm.chat(
                    messages=batch_messages,
                    sampling_params=SamplingParams(**generation_kwargs),
                    chat_template=self.chat_template,
                    add_generation_prompt=True,
                    chat_template_kwargs=self.chat_template_kwargs,
                )
        return responses

    def get_response(self, output: dict) -> str:
        """Get the response from the output.

        Args:
            output (dict): The output to get the response from.

        Returns:
            str: The response from the output.
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
                    extracted_text = text.split("assistantfinal")[1].strip()
                else:
                    extracted_text = text
            return extracted_text
        else:
            return output.outputs[0].text

    def parse_outputs(
        self,
        generated_outputs: list,
        conversations: list | None = None,
        tokenize_prompts: bool = False,
        use_logprobs: bool = False,
    ) -> dict:
        """Parse the outputs of the generated responses.

        Args:
            generated_outputs (list): The generated outputs to parse.
            conversations (list | None, optional): Conversations associated with the outputs. Defaults to None.
            tokenize_prompts (bool, optional): Whether to tokenize prompts. Defaults to False.
            use_logprobs (bool, optional): Whether logprobs were used in generation. Defaults to False.

        Returns:
            dict: The parsed outputs.
        """
        responses = []
        errors = []
        tokenized_prompts = []
        response_tokens = []
        total_tokens = []
        finish_reasons = []
        stop_reasons = []

        for output in generated_outputs:
            responses.append(self.get_response(output))
            if output.outputs[0].text == "":
                # Log empty string as an EmptyGenerationError
                errors.append("EmptyGenerationError")
            else:
                errors.append(None)
            tokenized_prompts.append(output.prompt_token_ids)
            response_tokens.append(output.outputs[0].token_ids)
            total_tokens.append(
                len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            )
            finish_reasons.append(output.outputs[0].finish_reason)
            stop_reasons.append(output.outputs[0].stop_reason)

        outputs = {
            "responses": responses,
            "errors": errors,
            "tokenized_prompts": tokenized_prompts,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "finish_reasons": finish_reasons,
            "stop_reasons": stop_reasons,
        }

        if use_logprobs:
            cumulative_logprobs = []
            logprobs = []
            for output in generated_outputs:
                logprobs.append(
                    str(output.outputs[0].logprobs)
                )  # convert logprobs to string for serialization
                cumulative_logprobs.append(output.outputs[0].cumulative_logprob)
            outputs["cumulative_logprobs"] = cumulative_logprobs
            outputs["logprobs"] = logprobs

        return outputs

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

        cleanup_dist_env_and_memory(shutdown_ray=True)
        logger.info(f"vLLM model {self.model_name} deleted and memory freed.")

    def convert_response_to_json(self, response: RequestOutput) -> dict:
        """Convert the response to a JSON serializable format.

        Args:
            response (dict): The response to convert.

        Returns:
            dict: The JSON serializable response.
        """
        return {
            "request_id": response.request_id,
            "prompt": response.prompt,
            "prompt_token_ids": response.prompt_token_ids,
            "encoder_prompt": response.encoder_prompt,
            "encoder_prompt_token_ids": response.encoder_prompt_token_ids,
            "prompt_logprobs": response.prompt_logprobs,
            "outputs": response.outputs,
            "finished": response.finished,
            "metrics": response.metrics,
            "lora_request": response.lora_request,
            "num_cached_tokens": response.num_cached_tokens,
            "multi_modal_placeholders": response.multi_modal_placeholders,
        }
