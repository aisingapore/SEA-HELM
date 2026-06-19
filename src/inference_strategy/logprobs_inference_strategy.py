import os
import time

import pandas as pd

from src.base_logger import get_logger
from src.inference_strategy.utils import check_cached_file_integrity
from src.serving.batch.base_batch_serving import BaseBatchServing
from src.serving.offline.base_offline_serving import BaseOfflineServing
from src.task_config import TaskConfig

logger = get_logger(__name__)


class LogprobsInferenceStrategy:
    """Inference strategy that computes per-token log-probabilities for candidate answers.

    Performs a two-phase generation: first generating a reasoning chain up to
    an answer tag, then scoring each candidate answer label by computing
    prompt log-probabilities for the continuation. Does not support batch
    serving backends.
    """

    def __init__(self, serving_class, task_config: TaskConfig):
        """Initialise the strategy.

        Args:
            serving_class: An instantiated serving backend. Must not be a
                ``BaseBatchServing`` instance.
            task_config: Configuration object for the current task, including
                caching preferences and generation settings.

        Raises:
            AssertionError: If ``serving_class`` is a ``BaseBatchServing`` instance.
        """
        assert not isinstance(serving_class, BaseBatchServing), (
            "BaseModelInferenceStrategy does not support batch generation."
        )
        self.serving_class = serving_class
        self.task_config = task_config

        if isinstance(serving_class, BaseOfflineServing):
            self.use_tokens_for_intermediate_steps = True
        else:
            self.use_tokens_for_intermediate_steps = False

    def run_inference(
        self,
        conversations,
        generation_kwargs,
        batch_filepath: str,
        batch_response_filepath: str,
        custom_ids: list | None = None,
        additional_kwargs: dict | None = None,
        labels: list | None = None,
    ) -> tuple[float | None, bool | None]:
        """Run two-phase inference with log-probability scoring and persist results to disk.

        Phase 1 generates tokens up to the answer tag specified in
        ``additional_kwargs['answer_tag']``. Phase 2 appends each candidate
        label (from ``labels``) to the intermediate output and computes prompt
        log-probabilities, allowing downstream metrics to rank answers by
        likelihood.

        If a cached response file already exists and
        ``task_config.use_cached_results`` is ``True``, inference is skipped
        and the cached file is used as-is.

        Args:
            conversations: List of conversation histories, each a list of
                role/content message dicts.
            generation_kwargs: Sampling parameters forwarded to the serving
                backend (e.g. ``max_tokens``, ``temperature``).
            batch_filepath: Not used by this strategy; present for interface
                compatibility.
            batch_response_filepath: Path where parsed outputs (including
                ``logprobs`` and ``cumulative_logprobs`` columns) are written
                as newline-delimited JSON.
            custom_ids: Not used by this strategy; present for interface
                compatibility.
            additional_kwargs: Auxiliary parameters. Recognised keys:
                ``answer_tag`` – stop string for phase 1 generation;
                ``answer_tag_separator`` – string prepended to each label
                before scoring.
            labels: List of candidate answer strings aligned with
                ``conversations``. Required for log-probability scoring.

        Returns:
            A tuple of:
            - ``inference_time_taken`` (``float | None``): Total wall-clock
              seconds spent on inference, or ``None`` if results were cached.
            - ``is_cached`` (``bool | None``): ``True`` if cached results were
              used, ``False`` if fresh inference was performed.
        """
        inference_time_taken: float | None = None
        is_cached: bool | None = None

        if self.task_config.use_cached_results and os.path.exists(
            batch_response_filepath
        ):
            if not check_cached_file_integrity(conversations, batch_response_filepath):
                logger.warning(
                    "Length of cached responses does not match number of conversations. Re-running inference for %s.",
                    batch_response_filepath,
                )
            else:
                logger.info("Using cached responses from %s", batch_response_filepath)
                is_cached = True

        if not is_cached:
            self.serving_class.load_model()

            inference_time_taken = 0
            # generate to answer_tag
            _generation_kwargs = generation_kwargs.copy()
            _generation_kwargs["stop"] = additional_kwargs.get("answer_tag", None)
            _generation_kwargs["include_stop_str_in_output"] = True

            start = time.perf_counter()
            responses = self.serving_class.generate_chat_responses(
                conversations, _generation_kwargs
            )
            end = time.perf_counter()
            inference_time_taken += end - start

            intermediate_outputs = [
                self.serving_class.parse_output(response) for response in responses
            ]

            # prepare for logprobs generation
            ids = []
            prompts = []
            answer_tokens = []
            input_lengths = []
            answers = [
                additional_kwargs["answer_tag_separator"] + answer for answer in labels
            ]

            if self.use_tokens_for_intermediate_steps:
                assert isinstance(self.serving_class, BaseOfflineServing), (
                    "Using tokens for intermediate steps is only supported for BaseOfflineServing."
                )
                answer_tokens = self.serving_class.tokenize(answers)

            for i in range(len(conversations)):
                if intermediate_outputs[i]["finish_reasons"] != "stop":
                    logger.info(
                        "Skipping id %d because it did not generate the correct answer tag",
                        i,
                    )
                    continue

                input_lengths.append(
                    intermediate_outputs[i]["token_usages"]["total_tokens"]
                )

                ids.append(i)
                if self.use_tokens_for_intermediate_steps:
                    _prompt = (
                        intermediate_outputs[i]["tokenized_prompts"]
                        + intermediate_outputs[i]["response_tokens"]
                        + answer_tokens[i]
                    )
                else:
                    _prompt = conversations[i] + [
                        {
                            "role": "assistant",
                            "content": intermediate_outputs[i]["responses"]
                            + answers[i],
                        }
                    ]
                prompts.append(_prompt)

            # prepare generation kwargs for logprobs generation
            _generation_kwargs = generation_kwargs.copy()
            _generation_kwargs["max_tokens"] = 1
            _generation_kwargs["prompt_logprobs"] = 1

            start = time.perf_counter()
            if self.use_tokens_for_intermediate_steps:
                logprobs_responses = self.serving_class.generate_completions(
                    prompts, _generation_kwargs
                )
            else:
                logprobs_responses = self.serving_class.generate_chat_responses(
                    prompts, _generation_kwargs
                )
            end = time.perf_counter()
            inference_time_taken += end - start

            # parse logprobs
            # TODO find a better way to handle the extraction of the logprobs
            output_logprobs = []
            cumulative_logprobs = []
            for id, logprobs_response, input_len in zip(
                ids, logprobs_responses, input_lengths, strict=True
            ):
                token_ids = logprobs_response.prompt_token_ids[input_len:]
                logprobs = logprobs_response.prompt_logprobs[input_len:]

                _logprobs = []
                cumulative_logprob = 0.0
                for token, logprob in zip(token_ids, logprobs, strict=True):
                    _logprobs.append({str(token): logprob[token].logprob})
                    cumulative_logprob += logprob[token].logprob

                output_logprobs.append(_logprobs)
                cumulative_logprobs.append(cumulative_logprob)

                # TODO This is specific to vllm outputs, find a better way to handle this
                if self.use_tokens_for_intermediate_steps:
                    responses[id].outputs[0].token_ids = (
                        intermediate_outputs["response_tokens"][id] + answer_tokens[id]
                    )

            parsed_outputs = self.serving_class.parse_outputs(responses)
            for id in ids:
                parsed_outputs[id]["responses"] = (
                    parsed_outputs[id]["responses"] + answers[id]
                )

            df = pd.DataFrame(parsed_outputs)
            df = df.assign(
                logprobs=output_logprobs, cumulative_logprobs=cumulative_logprobs
            )
            df.to_json(
                batch_response_filepath, orient="records", lines=True, force_ascii=False
            )
            is_cached = False

        return inference_time_taken, is_cached
