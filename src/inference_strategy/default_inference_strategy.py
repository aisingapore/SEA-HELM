import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src.base_logger import get_logger
from src.inference_strategy.utils import check_cached_file_integrity
from src.task_config import TaskConfig

logger = get_logger(__name__)


class DefaultInferenceStrategy:
    """Standard single-pass inference strategy.

    Generates responses for all conversations in a single call to the serving
    backend and persists the parsed outputs to disk. Supports any serving
    backend type.
    """

    def __init__(self, serving_class, task_config: TaskConfig):
        """Initialise the strategy.

        Args:
            serving_class: An instantiated serving backend.
            task_config: Configuration object for the current task, including
                caching preferences.
        """
        self.serving_class = serving_class
        self.task_config = task_config

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
        """Run single-pass inference and persist results to disk.

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
            batch_response_filepath: Path where parsed outputs are written as
                newline-delimited JSON.
            custom_ids: Optional list of identifiers aligned with
                ``conversations``, passed through to ``parse_output``.
            additional_kwargs: Not used by this strategy; present for interface
                compatibility.
            labels: Not used by this strategy; present for interface
                compatibility.

        Returns:
            A tuple of:
            - ``inference_time_taken`` (``float | None``): Wall-clock seconds
              spent on inference, or ``None`` if results were cached.
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

            ids = []
            # remove conversations whereby the model fails to produce a response
            skipped_rows = []
            for i, convo in enumerate(conversations):
                ids.append(custom_ids[i] if custom_ids else None)
                for turn in convo:
                    # TODO handle tool calls here
                    if turn["role"] == "assistant" and turn["content"] is None:
                        skipped_rows.append(i)

            # Filter out skipped conversations
            if skipped_rows:
                conversations = [
                    convo
                    for i, convo in enumerate(conversations)
                    if i not in skipped_rows
                ]

            start_time = time.perf_counter()
            generated_outputs = self.serving_class.generate_chat_responses(
                conversations, generation_kwargs
            )
            end_time = time.perf_counter()
            inference_time_taken = end_time - start_time
            if generated_outputs is None:
                logger.error("generate_chat_responses returned None.")
                parsed_outputs = []
                skipped_rows = list(range(len(ids)))
            else:
                with ThreadPoolExecutor() as executor:
                    parsed_outputs = list(
                        executor.map(
                            self.serving_class.parse_output, generated_outputs, ids
                        )
                    )

            if skipped_rows:
                logger.warning(
                    "The following rows were skipped due to generation failures: %s",
                    skipped_rows,
                )
                for i in skipped_rows:
                    # TODO check if other serving classes need to implement empty_output_dict for this to work
                    parsed_outputs.insert(
                        i,
                        self.serving_class.empty_output_dict(
                            custom_ids=custom_ids[i] if custom_ids else None
                        ),
                    )
            pd.DataFrame(parsed_outputs).to_json(
                batch_response_filepath, orient="records", lines=True, force_ascii=False
            )
            is_cached = False

        return inference_time_taken, is_cached
