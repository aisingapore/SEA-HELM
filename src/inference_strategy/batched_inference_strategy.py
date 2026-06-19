import os

import pandas as pd

from src.base_logger import get_logger
from src.inference_strategy.utils import check_cached_file_integrity
from src.serving.batch.base_batch_serving import BaseBatchServing
from src.task_config import TaskConfig

logger = get_logger(__name__)


class BatchedInferenceStrategy:
    """Inference strategy for batch-capable serving backends.

    Submits conversations as a batch request, polls for completion, and
    retries any missing responses up to a configurable maximum. Requires
    a ``BaseBatchServing`` backend (e.g. OpenAI batch API).
    """

    def __init__(self, serving_class, task_config: TaskConfig):
        """Initialise the strategy.

        Args:
            serving_class: An instantiated serving backend that inherits from
                ``BaseBatchServing``.
            task_config: Configuration object for the current task, including
                caching preferences.

        Raises:
            AssertionError: If ``serving_class`` does not inherit from
                ``BaseBatchServing``.
        """
        assert isinstance(serving_class, BaseBatchServing), (
            "BatchedInferenceStrategy requires a serving class that implements BaseBatchServing."
        )
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
        """Submit a batch inference request and persist parsed results to disk.

        Writes the full batch request to ``batch_filepath``, then submits it
        via the serving backend. If a cached response file exists and
        ``task_config.use_cached_results`` is ``True``, the batch is compared
        against the cache and only missing responses are re-requested. Retries
        up to ``max_retries`` times before logging a warning.

        Args:
            conversations: List of conversation histories, each a list of
                role/content message dicts.
            generation_kwargs: Sampling parameters forwarded to the batch
                backend.
            batch_filepath: Path where the batch request file is written.
            batch_response_filepath: Path where parsed outputs are written as
                newline-delimited JSON.
            custom_ids: Optional list of identifiers aligned with
                ``conversations``.
            additional_kwargs: Not used by this strategy; present for interface
                compatibility.
            labels: Not used by this strategy; present for interface
                compatibility.

        Returns:
            A tuple of:
            - ``inference_time_taken`` (``float | None``): Always ``None`` for
                batch strategies, as wall-clock time is not tracked.
            - ``is_cached`` (``bool | None``): ``True`` if cached results were
                used, ``False`` if a fresh batch was submitted.
        """
        inference_time_taken: float | None = None
        is_cached: bool | None = None

        self.serving_class.prepare_batches(
            batch_filepath, conversations, generation_kwargs, custom_ids=custom_ids
        )

        # Read expected batch requests to track completion
        expected_batches = pd.read_json(batch_filepath, lines=True).to_dict(
            orient="records"
        )

        if self.task_config.use_cached_results and os.path.exists(
            batch_response_filepath
        ):
            response_df = pd.read_json(batch_response_filepath, lines=True)
            if not check_cached_file_integrity(conversations, batch_response_filepath):
                logger.warning(
                    "Length of cached responses does not match number of conversations. Re-running inference for %s.",
                    batch_response_filepath,
                )
                is_cached = False
            else:
                logger.info("Using cached responses from %s", batch_response_filepath)
                is_cached = True
        else:
            response_df = pd.DataFrame()
            is_cached = False

        # Extract expected IDs
        expected_ids = {
            self.serving_class.get_ids_from_batch(batch) for batch in expected_batches
        }

        # Retry mechanism for missing batch responses
        retries = 0
        max_retries = 3

        while True:
            # Check which responses are still missing
            if not response_df.empty:
                existing_ids = {
                    self.serving_class.get_valid_ids_from_batch(batch)
                    for batch in response_df.to_dict(orient="records")
                }
            else:
                existing_ids = set()
            missing_ids = expected_ids - existing_ids

            # Exit if all responses obtained
            if not missing_ids:
                logger.info("All responses have been obtained.")
                break

            # Ensure judge model is loaded as inference needs to be run
            self.serving_class.load_model()

            # Determine batch file and output path for current iteration
            if retries == 0 and response_df.empty:
                logger.info("First run: processing all responses.")
                batch_file_path_to_use = batch_filepath
                response_file_path_to_use = batch_response_filepath
            else:
                logger.info("Missing %d responses. Retrying...", len(missing_ids))

                # Create batch file with only missing responses
                missing_batches = [
                    batch
                    for batch in expected_batches
                    if self.serving_class.get_ids_from_batch(batch) in missing_ids
                ]

                # Write missing batches to temporary file
                missing_batch_file_path = batch_filepath.replace(
                    ".jsonl", f"_missing_retry{retries}.jsonl"
                )
                pd.DataFrame(missing_batches).to_json(
                    missing_batch_file_path,
                    orient="records",
                    lines=True,
                    force_ascii=False,
                )
                batch_file_path_to_use = missing_batch_file_path

                # Set temporary output path for retry
                response_file_path_to_use = batch_response_filepath.replace(
                    ".jsonl", f"_temp_retry{retries}.jsonl"
                )

            # Generate judgements using appropriate serving backend
            self.serving_class.batch_generate(
                file_path=batch_file_path_to_use,
                output_file_path=response_file_path_to_use,
            )

            # Load and merge new judgements
            new_judgement_df = pd.read_json(response_file_path_to_use, lines=True)
            # Parse responses
            parsed_df = pd.DataFrame(
                [
                    self.serving_class.parse_output(row)
                    for _, row in new_judgement_df.iterrows()
                ]
            )
            new_judgement_df = new_judgement_df.drop(
                columns=["custom_ids", "errors", "responses"],
                errors="ignore",
            )
            new_judgement_df = pd.concat([new_judgement_df, parsed_df], axis=1)

            response_df = pd.concat([response_df, new_judgement_df], ignore_index=True)
            # Remove duplicates, keeping latest version
            response_df = response_df.drop_duplicates(subset="custom_ids", keep="last")

            # ensure the response_df follows the expected_batches order
            response_df = (
                response_df.set_index("custom_ids")
                .reindex(
                    [
                        self.serving_class.get_ids_from_batch(batch)
                        for batch in expected_batches
                    ]
                )
                .reset_index()
            )

            # Save updated responses to main file
            response_df.to_json(
                batch_response_filepath, orient="records", lines=True, force_ascii=False
            )

            # Clean up temporary files from retries
            if retries != 0:
                os.remove(batch_file_path_to_use)
                os.remove(response_file_path_to_use)

            retries += 1
            if retries >= max_retries:
                logger.warning(
                    "Reached maximum retries (%d). Some responses may still be missing.",
                    max_retries,
                )
                break

        # Final check for missing responses
        if not response_df.empty:
            existing_ids = {
                self.serving_class.get_ids_from_batch(batch)
                for batch in response_df.to_dict(orient="records")
            }
        else:
            existing_ids = set()

        missing_ids = expected_ids - existing_ids

        if missing_ids:
            logger.error(
                "Failed to obtain all responses after %d retries. Missing responses for IDs: %s",
                retries,
                ", ".join(missing_ids),
            )
        else:
            logger.info("Successfully obtained all responses.\n")

        # TODO handle tokenization of prompts
        return inference_time_taken, is_cached
