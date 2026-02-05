import time
from multiprocessing import Pool

from src.base_logger import get_logger

logger = get_logger(__name__)


class QueueManager:
    """Manages task queues for evaluation and asynchronous judge model operations.

    This class coordinates the execution order of evaluation tasks, particularly managing
    tasks that require local or remote LLM judges. It maintains separate queues for
    evaluations and async remote judge responses, and provides a multiprocessing pool
    for parallel execution.

    The execution order is:
    1. Remote LLM judge tasks (async) - e.g., MT-Bench, Arena-Hard-v2
    2. Normal evaluation tasks
    3. Local LLM judge tasks - e.g., translation with MetricX
    4. Collection of remote LLM judge results

    Attributes:
        queued_evaluation_params (dict[str, list[dict]]): Dictionary storing evaluation
            parameters by key. Each entry is a list of dictionaries containing:
            - evaluation_params (tuple): (dataloader, task_config, task_name, lang)
            - judge_params (tuple | None): Optional judge model configuration
        remote_queue_awaiting_responses (dict[str, AsyncResult]): Dictionary tracking
            async tasks awaiting completion from remote LLM judges. Keys are task
            identifiers, values are multiprocessing.pool.AsyncResult objects.
        pool (Pool | None): Multiprocessing pool with 32 workers for parallel execution
            of async remote judge tasks.
    """

    def __init__(self):
        """Initialize QueueManager with empty queues and no active pool.

        Sets up:
        - Empty evaluation queue (queued_evaluation_params)
        - Empty remote response tracking queue (remote_queue_awaiting_responses)
        - No active multiprocessing pool (pool = None)
        """
        self.queued_evaluation_params = {}
        self.remote_queue_awaiting_responses = {}
        self.pool = None

    def add_to_evaluation_queue(
        self, key, evaluation_params: tuple, judge_configs: dict | None = None
    ):
        """Add an evaluation task to the queue.

        Stores evaluation parameters and optional judge parameters for later execution.
        Multiple tasks can be queued under the same key, allowing batching of related
        evaluations.

        Args:
            key: Unique identifier for the task (will be converted to string). Typically
                formatted as "<task_name>_<lang>".
            evaluation_params (tuple): Tuple containing evaluation parameters with structure:
                (dataloader, task_config, task_name, lang) where:
                - dataloader: Data loading object for the task
                - task_config (dict): Task configuration from config.yaml
                - task_name (str): Name of the evaluation task
                - lang (str): Language code (e.g., 'en', 'id', 'vi')
            judge_params (tuple | None, optional): Tuple containing judge model parameters
                with structure: (judge_model_name, judge_model_type, judge_init_args,
                batch_api_calls) where:
                - judge_model_name (str): Name/path of the judge model
                - judge_model_type (str): Type of judge ('vllm', 'openai', 'anthropic', etc.)
                - judge_init_args (dict): Initialization arguments for the judge model
                - batch_api_calls (bool): Whether to batch API calls for the judge
                Defaults to None (no judge model required).

        Example:
            >>> manager = QueueManager()
            >>> manager.add_to_evaluation_queue(
            ...     key="mt-bench_en",
            ...     evaluation_params=(dataloader, config, "mt-bench", "en"),
            ...     judge_params=("gpt-4", "openai", {"temperature": 0.0}, True)
            ... )
        """
        if str(key) not in self.queued_evaluation_params:
            self.queued_evaluation_params[str(key)] = []

        self.queued_evaluation_params[str(key)].append(
            {
                "evaluation_params": evaluation_params,
                "judge_configs": judge_configs,
            }
        )

    def is_queue_empty(self) -> bool:
        """Check if the evaluation queue is empty.

        Used to determine if there are remaining tasks to process before moving
        to the next execution phase.

        Returns:
            bool: True if there are no queued evaluation tasks, False otherwise.
        """
        return len(self.queued_evaluation_params.keys()) == 0

    def iterate_through_evaluation_queue(self):
        """Iterate through all queued evaluation tasks.

        Provides access to all queued tasks for processing. Each iteration yields
        a key and its associated list of evaluation parameter dictionaries.

        Yields:
            tuple[str, list[dict]]: Key-value pairs from queued_evaluation_params where:
                - key (str): Task identifier (e.g., "mt-bench_en")
                - value (list[dict]): List of dictionaries, each containing:
                    - evaluation_params (tuple): Evaluation configuration
                    - judge_params (tuple | None): Optional judge configuration

        Example:
            >>> for key, eval_list in manager.iterate_through_evaluation_queue():
            ...     for item in eval_list:
            ...         eval_params = item["evaluation_params"]
            ...         judge_params = item["judge_params"]
        """
        return self.queued_evaluation_params.items()

    def get_unique_set_of_evaluation_params(self):
        """Get unique set of evaluation parameters from all queued tasks.

        Extracts and deduplicates evaluation_params tuples from all queued tasks.
        This is useful for ensuring each unique evaluation configuration is only
        processed once, avoiding redundant computations when the same task-language
        combination appears in multiple queue entries.

        Returns:
            set[tuple]: Set of unique evaluation_params tuples, each containing:
                (dataloader, task_config, task_name, lang) where:
                - dataloader: Data loading object for the task
                - task_config (dict): Task configuration
                - task_name (str): Name of the evaluation task
                - lang (str): Language code

        Note:
            The set uses tuple hashing for deduplication. Tasks are considered
            duplicates if all four elements match.
        """
        unique_evaluation_params = set()
        for eval_params_list in self.queued_evaluation_params.values():
            for eval_params in eval_params_list:
                unique_evaluation_params.add(eval_params["evaluation_params"])

        return unique_evaluation_params

    def start_pool(self):
        """Start a multiprocessing pool for parallel task execution.

        Creates a pool with 32 worker processes for handling async operations,
        typically used for remote LLM judge tasks (e.g., MT-Bench with GPT-4,
        Arena-Hard-v2 with Claude). This allows multiple async API calls to be
        submitted in parallel while the main evaluation continues.

        Note:
            The pool should be started before adding tasks with add_to_async_queue()
            and terminated with terminate_pool() after all async tasks complete.

        Raises:
            RuntimeError: If the pool is already started (pool is not None).
        """
        self.pool = Pool(processes=32)

    def terminate_pool(self):
        """Terminate the multiprocessing pool and clean up resources.

        Performs graceful shutdown of the multiprocessing pool:
        1. Closes the pool to prevent new tasks from being submitted
        2. Waits for all worker processes to complete their current tasks
        3. Cleans up and sets the pool reference to None

        This should be called after wait_for_all_async_tasks() completes to ensure
        all async operations have finished before cleanup.

        Note:
            Safe to call even if the pool is not active (pool is None). Does nothing
            in that case.
        """
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def is_pool_started(self) -> bool:
        """Check if the multiprocessing pool is currently active.

        Used to verify that the pool is ready before submitting async tasks or
        to ensure proper cleanup has occurred after termination.

        Returns:
            bool: True if the pool exists and is active (pool is not None),
                False if the pool has not been started or has been terminated.
        """
        return self.pool is not None

    def add_to_async_queue(self, key, function, params):
        """Add an async task to the remote queue for parallel execution.

        Submits a function with parameters to the multiprocessing pool for async execution
        and tracks the AsyncResult object for later retrieval. The task will execute in
        parallel with other queued tasks while the main evaluation continues.

        Args:
            key: Unique identifier for the async task (will be converted to string).
                Typically formatted as "<task_name>_<lang>" (e.g., "mt-bench_en").
            function: Callable function to execute asynchronously. Should accept the
                parameters specified in the params argument.
            params: Tuple of parameters to pass to the function. Will be unpacked and
                passed as arguments via starmap (wrapped in a single-element list internally).

        Note:
            The pool must be started with start_pool() before calling this method.
            Results can be retrieved by waiting with wait_for_all_async_tasks().

        Example:
            >>> manager.start_pool()
            >>> manager.add_to_async_queue(
            ...     key="mt-bench_en",
            ...     function=evaluate_with_judge,
            ...     params=(model_output, judge_model, config)
            ... )
        """
        self.remote_queue_awaiting_responses[str(key)] = self.pool.starmap_async(
            function, [params]
        )

    def wait_for_all_async_tasks(self, sleep_time=2):
        """Wait for all async tasks in the remote queue to complete.

        Blocks execution until all async tasks submitted via add_to_async_queue()
        have completed. Continuously polls the remote queue, checking if AsyncResult
        objects are ready. Logs completion messages as tasks finish and removes them
        from the tracking list.

        This is typically called at the end of the evaluation pipeline to collect
        results from remote LLM judge tasks before final metric aggregation.

        Args:
            sleep_time (int, optional): Time in seconds to sleep between polling cycles
                to avoid excessive CPU usage. Defaults to 2 seconds.

        Note:
            - Logs info messages when each async task completes
            - Creates a copy of the tracking list to avoid modification during iteration
            - Blocks until ALL tasks are complete before returning

        Example:
            >>> manager.add_to_async_queue("mt-bench_en", evaluate, params1)
            >>> manager.add_to_async_queue("mt-bench_id", evaluate, params2)
            >>> # ... continue with other evaluations ...
            >>> manager.wait_for_all_async_tasks()  # Blocks until both complete
        """
        awaiting_responses_list = list(self.remote_queue_awaiting_responses.keys())
        logger.info(
            "Waiting for %d async LLM judge tasks to complete...",
            len(awaiting_responses_list),
        )
        while awaiting_responses_list:
            for key in awaiting_responses_list.copy():
                if self.remote_queue_awaiting_responses[key].ready():
                    logger.info(
                        "Completed evaluation for %s using LLM judge.",
                        key,
                    )
                    try:
                        # retrieve result to raise any exceptions encountered during execution
                        self.remote_queue_awaiting_responses[key].get()
                    except Exception as e:
                        logger.error(
                            "Error occurred during async judgement call for %s: %s",
                            key,
                            str(e),
                        )
                    awaiting_responses_list.remove(key)

            # check if awaiting_responses_list is empty
            if not awaiting_responses_list:
                break

            # sleep to wait before polling again
            time.sleep(sleep_time)  # Poll every 2 seconds
