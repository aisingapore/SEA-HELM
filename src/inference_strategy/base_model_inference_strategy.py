import os
import time

import pandas as pd

from src.base_logger import get_logger
from src.inference_strategy.utils import check_cached_file_integrity
from src.serving.batch.base_batch_serving import BaseBatchServing
from src.serving.offline.base_offline_serving import BaseOfflineServing
from src.task_config import TaskConfig

logger = get_logger(__name__)


class BaseModelInferenceStrategy:
    """Inference strategy for models that require multi-step generation.

    Performs a two-phase generation: first generating up to an answer tag,
    then continuing from the intermediate output to produce the final response.
    Supports both token-based (offline) and message-based (online) continuation.
    Does not support batch serving backends.
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
        """Run two-phase inference and persist results to disk.

        Phase 1 generates tokens up to the answer tag specified in
        ``additional_kwargs['answer_tag']``. Phase 2 continues from the
        intermediate output to produce the final answer, respecting the
        original ``max_tokens`` budget across both phases.

        If a cached response file already exists and
        ``task_config.use_cached_results`` is ``True``, inference is skipped
        and the cached file is used as-is.

        Args:
            conversations: List of conversation histories, each a list of
                role/content message dicts.
            generation_kwargs: Sampling parameters forwarded to the serving
                backend (e.g. ``max_tokens``, ``temperature``).
            batch_filepath: Path for the raw batch request file (unused by
                this strategy but kept for interface consistency).
            batch_response_filepath: Path where parsed outputs are written as
                newline-delimited JSON.
            custom_ids: Optional list of identifiers aligned with
                ``conversations``, passed through to ``parse_output``.
            additional_kwargs: Auxiliary parameters. Recognised keys:
                ``answer_tag`` - stop string for phase 1 generation.
            labels: Not used by this strategy; present for interface
                compatibility.

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

            # generate final answer
            ids = []
            prompts = []
            sampling_params_list = []

            if self.use_tokens_for_intermediate_steps:
                assert isinstance(self.serving_class, BaseOfflineServing), (
                    "Using tokens for intermediate steps is only supported for BaseOfflineServing."
                )

            for i in range(len(conversations)):
                if intermediate_outputs[i]["finish_reasons"] != "stop":
                    logger.info(
                        "Skipping id %d because it did not generate the correct answer tag",
                        i,
                    )
                    continue

                _gen_kwargs = generation_kwargs.copy()

                if "response_tokens" in intermediate_outputs[i]:
                    _gen_kwargs["max_tokens"] = generation_kwargs["max_tokens"] - len(
                        intermediate_outputs[i]["response_tokens"]
                    )
                else:
                    _gen_kwargs["max_tokens"] = (
                        generation_kwargs["max_tokens"]
                        - intermediate_outputs[i]["token_usages"]["completion_tokens"]
                    )

                if _gen_kwargs["max_tokens"] <= 0:
                    continue

                if self.use_tokens_for_intermediate_steps:
                    _prompt = (
                        intermediate_outputs[i]["tokenized_prompts"]
                        + intermediate_outputs[i]["response_tokens"]
                    )
                else:
                    _prompt = conversations[i] + [
                        {
                            "role": "assistant",
                            "content": intermediate_outputs[i]["responses"],
                        }
                    ]
                    _gen_kwargs["continue_final_message"] = True
                    _gen_kwargs["add_generation_prompt"] = False

                sampling_params_list.append(_gen_kwargs)
                ids.append(i)
                prompts.append(_prompt)

            start = time.perf_counter()
            if self.use_tokens_for_intermediate_steps:
                final_responses = self.serving_class.generate_completions(
                    prompts, sampling_params_list
                )
            else:
                final_responses = self.serving_class.generate_chat_responses(
                    prompts, sampling_params_list
                )
            end = time.perf_counter()
            inference_time_taken += end - start

            parsed_outputs = [
                self.serving_class.parse_output(response) for response in responses
            ]
            for id, response in zip(ids, final_responses, strict=True):
                parsed_outputs[id]["responses"] = parsed_outputs[id][
                    "responses"
                ] + self.serving_class.get_response(response)

            pd.DataFrame(parsed_outputs).to_json(
                batch_response_filepath, orient="records", lines=True, force_ascii=False
            )
            is_cached = False

        return inference_time_taken, is_cached
