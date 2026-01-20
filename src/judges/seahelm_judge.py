import asyncio
import json
import os
from abc import abstractmethod

import pandas as pd

from src.base_logger import get_logger
from src.dataloaders.base_dataloader import AbstractDataloader
from src.metrics.seahelm_metric import SeaHelmMetric
from src.serving.batch.base_batch_serving import BaseBatchServing
from src.serving.local.base_serving import BaseServing
from src.task_config import TaskConfig

logger = get_logger(__name__)


class SeaHelmJudge:
    """Abstract base class for LLM-as-a-judge evaluation in SEA-HELM.

    This class provides the core infrastructure for using LLM judges to evaluate
    model responses. It supports both single-sample and batched API calls, with
    automatic retry mechanisms for batch processing.
    """

    def __init__(
        self,
        judge_model: BaseServing | BaseBatchServing,
        judge_config: dict,
        dataloader: AbstractDataloader,
        metric: SeaHelmMetric,
        task_config: TaskConfig,
        response_column: str = "responses",
    ):
        """Initialize the SeaHelmJudge.

        Args:
            judge_model: The judge model serving instance (e.g., OpenAIServing, VertexAIServing).
            judge_config (dict): The configuration dictionary for the judge model.
            dataloader (AbstractDataloader): Dataloader instance containing inference data.
            metric (SeaHelmMetric): Metric instance for calculating evaluation scores.
            task_config (dict): Task configuration dictionary containing judge settings.
            response_column (str, optional): Column name in dataloader containing model
                responses to be judged. Defaults to "responses".
        """
        self.judge_model = judge_model
        self.judge_config = judge_config
        self.dataloader = dataloader
        self.metric = metric
        self.task_config = task_config
        self.task = task_config.task_name
        self.lang = task_config.lang
        self.response_column = response_column

        self.model_name = dataloader.model_name
        self.judge_model_name = judge_config["judge_model_name"]

        self.use_cached_results = task_config.should_use_cached_results()
        if judge_config.get("batch_api_calls", False):
            self.judge_responses = self.get_batched_llm_judgements
        else:
            self.judge_responses = self.get_llm_judgements

    @abstractmethod
    def get_judge_prompts(self, row_no: int) -> dict:
        """Retrieve judge prompt template for a specific inference sample.

        This method should extract the appropriate judge prompt template from the task
        configuration, which may vary based on the evaluation type (e.g., with-reference
        vs without-reference, pairwise vs single-answer).

        Args:
            row_no (int): The row index in the inference dataframe to get prompts for.

        Returns:
            dict: Dictionary containing judge prompt template(s) with placeholders for
                filling in model responses, references, and other task-specific fields.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_judge_prompts method.")

    @abstractmethod
    def prepare_judge_prompts(
        self, judge_prompt: dict, row: pd.Series, row_no: int
    ) -> tuple[list[list], list[str]]:
        """Prepare judge prompts by filling templates with data from an inference row.

        This method formats the judge prompt template with actual model responses,
        references, and other task-specific fields from the inference dataframe.
        It handles creating conversation histories for the judge model and generating
        unique identifiers for tracking judgements.

        Args:
            judge_prompt (dict): Judge prompt template dictionary from get_judge_prompts().
            row (pd.Series): Single row from the inference dataframe containing model
                responses and ground truth data.
            row_no (int): The row index in the inference dataframe.

        Returns:
            tuple[list[list], list[str]]: A tuple containing:
                - list[list]: List of conversation histories, where each conversation is a
                    list of message dictionaries (e.g., [{"role": "user", "content": "..."}]).
                - list[str]: List of custom IDs corresponding to each conversation for
                    tracking and matching judgements to inference samples.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement prepare_judge_prompts method."
        )

    @abstractmethod
    def get_judgement_file_name(self) -> str:
        """Get the filename for storing judge evaluation results.

        This method defines the naming convention for the JSONL file that stores
        judge model responses and parsed judgements. The file is typically saved
        in the task's output directory alongside inference results.

        Returns:
            str: Filename (not full path) for the judgements file, typically in the
                format '<task>_<judge_model>_judgements.jsonl' or similar.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement get_judgement_file_name method."
        )

    @abstractmethod
    def get_batch_file_name(self) -> str:
        """Get the filename for storing batch API request payloads.

        This method defines the naming convention for the JSONL file that contains
        formatted batch API requests for the judge model. This file is used for
        batch processing and tracking which judgements need to be generated.

        Returns:
            str: Filename (not full path) for the batch requests file, typically in the
                format '<task>_<judge_model>_batch.jsonl' or similar.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement get_batch_file_name method."
        )

    def get_llm_judgements(self) -> pd.DataFrame:
        """Generate LLM judgements using sample-by-sample API calls.

        This method processes each inference sample individually, making separate API
        calls for each judgement. The method handles caching to avoid redundant
        API calls when rerunning evaluations.

        The process:
        1. Check for cached judgement results and load if available
        2. Otherwise, prepare all judge prompts from inference data
        3. Generate judgements using the judge model's batch_generate method
        4. Parse responses and save to cache file

        Note:
            Currently compatible with OpenAI and LiteLLM serving backends. VertexAI
            support may require additional implementation.

        Returns:
            pd.DataFrame: DataFrame containing judgement results with columns:
                - custom_id: Unique identifier linking judgement to inference sample
                - parsed_response: Extracted judgement from raw model response
                - Additional columns from raw API response (varies by backend)

        See Also:
            get_batched_llm_judgements: Alternative method using batch APIs for efficiency.
        """
        # Construct path for cached judgement results
        llm_judgement_file_path = os.path.join(
            self.dataloader.get_parent_folder(),
            self.get_judgement_file_name(),
        )

        # Load cached results if available and enabled
        if self.use_cached_results and os.path.exists(llm_judgement_file_path):
            judgement_df = pd.read_json(llm_judgement_file_path, lines=True)
        else:
            # Ensure judge model is loaded as inference needs to be run
            self.judge_model.load_model()

            # Prepare messages and IDs for batch processing
            conversations = []
            custom_ids = []
            for i, row in self.dataloader.inference_df.iterrows():
                judge_prompt = self.get_judge_prompts(i)
                _conversations, _custom_ids = self.prepare_judge_prompts(
                    judge_prompt, row, i
                )
                conversations.extend(_conversations)
                custom_ids.extend(_custom_ids)

            # Generate judgements using the judge model
            batch_responses = self.judge_model.batch_generate(
                conversations, **self.judge_config["judge_generation_kwargs"]
            )

            # Convert responses if applicable (e.g. for vLLM serving)
            # TODO Find a new solution for this that works across all serving backends
            if hasattr(self.judge_model, "convert_response_to_json"):
                batch_responses = [
                    self.judge_model.convert_response_to_json(x)
                    for x in batch_responses
                ]

            # Create dataframe with judgements and save to file
            judgement_df = pd.DataFrame(batch_responses)
            judgement_df["custom_id"] = custom_ids
            # Parse responses
            judgement_df["parsed_response"] = judgement_df.apply(
                self.judge_model.get_response, axis=1
            )

            judgement_df.to_json(
                llm_judgement_file_path, orient="records", lines=True, force_ascii=False
            )
        return judgement_df

    def prepare_llm_judgement_batches(self, llm_batch_file_path: str) -> None:
        """Format and create batch API request file for LLM judge evaluations.

        This method prepares all judge prompts from the inference data and formats them
        into a batch request file compatible with the judge model's serving backend.
        The file format varies by backend (e.g., OpenAI Batch API format, VertexAI format).

        The method:
        1. Iterates through all inference samples
        2. Generates judge prompts for each sample using prepare_judge_prompts()
        3. Extracts generation configuration from task config
        4. Calls the judge model's prepare_llm_batches() to write formatted requests

        Args:
            llm_batch_file_path (str): Absolute path where the batch request JSONL file
                will be saved. Each line contains a formatted API request.

        Note:
            The generation configuration (max_tokens, temperature, seed) is extracted
            from task_config['judge']['judge_generation_kwargs'].
        """
        conversations = []
        custom_ids = []
        for i, row in self.dataloader.inference_df.iterrows():
            judge_prompt = self.get_judge_prompts(i)
            _conversations, _custom_ids = self.prepare_judge_prompts(
                judge_prompt, row, i
            )
            conversations.extend(_conversations)
            custom_ids.extend(_custom_ids)

        # Prepare LLM batches
        self.judge_model.prepare_llm_batches(
            llm_batch_file_path,
            conversations,
            custom_ids,
            **self.judge_config["judge_generation_kwargs"],
        )

    def get_batched_llm_judgements(self) -> pd.DataFrame:
        """Generate LLM judgements using batch API calls with automatic retry mechanism.

        This method uses batch APIs for efficiency and cost-effectiveness. It includes
        robust error handling with automatic retries for missing judgements.

        The process:
        1. Prepare batch request file if not cached
        2. Load any existing judgements from previous runs
        3. Submit batch requests to the judge model's batch API
        4. Wait for completion and retrieve results
        5. Identify missing judgements and retry up to max_retries times
        6. Parse all responses and save to cache file
        7. Clean up temporary retry files

        Features:
        - Caching: Reuses existing judgements to minimize API costs
        - Retry logic: Automatically retries missing judgements up to 3 times
        - Progress tracking: Logs detailed information about missing judgements
        - Cleanup: Removes temporary files created during retries

        Returns:
            pd.DataFrame: DataFrame containing all judgement results with columns:
                - custom_id: Unique identifier linking judgement to inference sample
                - parsed_response: Extracted judgement from raw model response
                - Additional columns from raw API response (varies by backend)

        Note:
            Supports OpenAI Batch API and VertexAI serving backends. The exact
            batch format and processing depends on the judge_model's serving class.

        See Also:
            get_llm_judgements: Alternative method for sample-by-sample processing.
        """
        # Define file paths for batch processing
        llm_judgement_file_path = os.path.join(
            self.dataloader.get_parent_folder(),
            self.get_judgement_file_name(),
        )

        llm_batch_file_path = os.path.join(
            self.dataloader.get_parent_folder(),
            self.get_batch_file_name(),
        )

        # Prepare batch file if not cached
        if self.use_cached_results and os.path.exists(llm_batch_file_path):
            pass
        else:
            self.prepare_llm_judgement_batches(llm_batch_file_path)

        # Load existing judgements if available
        if self.use_cached_results and os.path.exists(llm_judgement_file_path):
            judgement_df = pd.read_json(llm_judgement_file_path, lines=True)
        else:
            judgement_df = pd.DataFrame()

        # Read expected batch requests to track completion
        with open(llm_batch_file_path, "r", encoding="utf-8") as f:
            expected_batches = [json.loads(line) for line in f]

        # Extract expected IDs
        expected_ids = {
            self.judge_model.get_ids_from_batch(batch) for batch in expected_batches
        }

        # Retry mechanism for missing judgements
        retries = 0
        max_retries = 3

        while True:
            # Check which judgements are still missing
            existing_ids = (
                set(judgement_df["custom_id"]) if not judgement_df.empty else set()
            )
            missing_ids = expected_ids - existing_ids

            # Exit if all judgements obtained
            if not missing_ids:
                logger.info("All judgements have been obtained.")
                break

            # Ensure judge model is loaded as inference needs to be run
            self.judge_model.load_model()

            # Determine batch file and output path for current iteration
            if retries == 0 and judgement_df.empty:
                logger.info("First run: processing all judgements.")
                batch_file_path_to_use = llm_batch_file_path
                judgement_file_path_to_use = llm_judgement_file_path
            else:
                logger.info(f"Missing {len(missing_ids)} judgements. Retrying...")

                # Create batch file with only missing judgements
                missing_batches = [
                    batch
                    for batch in expected_batches
                    if self.judge_model.get_ids_from_batch(batch) in missing_ids
                ]

                # Write missing batches to temporary file
                missing_batch_file_path = llm_batch_file_path.replace(
                    ".jsonl", f"_missing_retry{retries}.jsonl"
                )
                with open(missing_batch_file_path, "w", encoding="utf-8") as f:
                    for batch in missing_batches:
                        f.write(json.dumps(batch, ensure_ascii=False) + "\n")
                batch_file_path_to_use = missing_batch_file_path

                # Set temporary output path for retry
                judgement_file_path_to_use = llm_judgement_file_path.replace(
                    ".jsonl", f"_temp_retry{retries}.jsonl"
                )

            # Generate judgements using appropriate serving backend
            asyncio.run(
                self.judge_model.abatch_generate(
                    file_path=batch_file_path_to_use,
                    output_file_path=judgement_file_path_to_use,
                )
            )

            # Load and merge new judgements
            new_judgement_df = pd.read_json(judgement_file_path_to_use, lines=True)

            judgement_df = pd.concat(
                [judgement_df, new_judgement_df], ignore_index=True
            )

            # update custom_ids (not that this only affects servings other than OpenAIServing)
            judgement_df["custom_id"] = judgement_df.apply(
                self.judge_model.get_ids_from_batch, axis=1
            )
            # Parse responses
            judgement_df["parsed_response"] = judgement_df.apply(
                self.judge_model.get_response, axis=1
            )

            # Remove duplicates, keeping latest version
            judgement_df = judgement_df.drop_duplicates(subset="custom_id", keep="last")

            # Save updated judgements to main file
            judgement_df.to_json(
                llm_judgement_file_path, orient="records", lines=True, force_ascii=False
            )

            # Clean up temporary files from retries
            if retries != 0:
                os.remove(batch_file_path_to_use)
                os.remove(judgement_file_path_to_use)

            retries += 1
            if retries >= max_retries:
                logger.warning(
                    f"Reached maximum retries ({max_retries}). Some judgements may still be missing."
                )
                break

        # Final check for missing judgements
        existing_ids = (
            set(judgement_df["custom_id"]) if not judgement_df.empty else set()
        )
        missing_ids = expected_ids - existing_ids

        if missing_ids:
            logger.error(
                f"Failed to obtain all judgements after {retries} retries. Missing judgements for IDs: {missing_ids}"
            )
        else:
            logger.info("Successfully obtained all judgements.\n")

        return judgement_df
