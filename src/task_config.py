import importlib
import re
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

from src.base_logger import get_logger

if TYPE_CHECKING:
    from src.dataloaders.base_dataloader import AbstractDataloader
    from src.judges.seahelm_judge import SeaHelmJudge
    from src.metrics.seahelm_metric import SeaHelmMetric

logger = get_logger(__name__)


class TaskConfig:
    """Configuration manager for a specific evaluation task.

    This class encapsulates all configuration settings for a single task evaluation,
    including task metadata, language-specific settings, generation parameters, and
    judge model configurations. It provides methods to retrieve dataloader, metric,
    and judge classes dynamically based on the task configuration.

    Attributes:
        config (dict): Complete task configuration dictionary from config.yaml
        task_name (str): Unique identifier for the task
        lang (str): Language code for the task (e.g., 'en', 'id', 'zh')
        seed (int): Random seed for reproducible generation
        judge_configs (dict): Dictionary mapping judge identifiers to their configurations

    Example:
        ```python
        task_config = TaskConfig(
            config=yaml_config,
            task_name="nli",
            lang="en",
            seed=42,
            use_cached_results=True
        )
        dataloader = task_config.get_dataloader_class()
        metric = task_config.get_metric_class()
        ```
    """

    def __init__(
        self,
        config: dict,
        task_name: str,
        lang: str,
        seed: int,
        is_base_model: bool = False,
        use_cached_results: bool = True,
        constants: dict | None = None,
        is_reasoning_model: bool = False,
    ):
        """Initialize a TaskConfig instance.

        Args:
            config (dict): Task configuration dictionary loaded from config.yaml.
                Should contain keys like 'name', 'metric_file', 'languages', etc.
            task_name (str): Unique identifier for the task (e.g., 'nli', 'sentiment').
            lang (str): Language code following ISO 639-1 or custom codes
                (e.g., 'en', 'id', 'zh-CN').
            seed (int): Random seed for reproducible generation and evaluation.
            use_cached_results (bool, optional): Whether to use cached inference results
                from previous runs. Defaults to True.

        Side Effects:
            Modifies the config dictionary by adding the 'use_cached_results' key.

        Note:
            The judge_configs attribute is initialized as an empty dictionary and
            should be populated using prepare_judge_configs() if the task uses judges.
        """
        self.config = config
        self.use_cached_results = use_cached_results

        self.task_name = task_name
        self.lang = lang
        self.seed = seed
        self.is_base_model = is_base_model
        self.is_reasoning_model = is_reasoning_model
        self.constants = constants if constants is not None else {}

        self.judge_configs = {}

    def should_use_cached_results(self) -> bool:
        """Check if cached inference results should be used.

        Returns:
            bool: True if cached results should be used, False otherwise.
        """
        return self.use_cached_results

    def should_task_run_for_run_number(self, run_number: int) -> bool:
        """Check if the current run number is within the allowed maximum runs for a task.

        This method validates whether a task should be executed based on the run number
        and the maximum number of runs configured for the task.

        Note:
        ---------
        * max_n_runs is currently only set for SEA-HELM LogProb tasks that do not require
        Chain-of-Thought or reasoning prior to the answer tag. As such, running multiple runs
        for these task is a waste of resources as the tokens before the answer tag will be
        the same and therefore the LogProbs will be the same.

        * For reasoning models, this is not the case and a special exception is given to allow
        these models to exceed the maximum runs due to the presence of think blocks.
        ---------

        Args:
            task_config (dict): Configuration dictionary for the task containing
                potential 'max_n_runs' and 'name' keys
            run_number (int): The current run number (0-indexed)

        Returns:
            bool: True if the task should be executed, False if it should be skipped.
                Always returns True if 'max_n_runs' is not configured.
                For reasoning models, returns True even if run_number exceeds max_n_runs.
                For non-reasoning models, returns False if run_number exceeds max_n_runs.
        """
        if "max_n_runs" not in self.config:
            return True

        max_n_runs = self.config["max_n_runs"]
        if run_number >= max_n_runs:
            if self.is_reasoning_model:
                logger.warning(
                    "Run number %d (0-index) exceeds max_n_runs %d for task '%s'. Proceeding with task due to think block in reasoning models.",
                    run_number,
                    max_n_runs,
                    self.config["name"],
                )
                return True
            else:
                logger.warning(
                    "Run number %d (0-index) exceeds max_n_runs %d for task '%s'. Skipping task.",
                    run_number,
                    max_n_runs,
                    self.config["name"],
                )
                return False
        else:
            return True

    def task_uses_judges(self) -> bool:
        """Check if the task is configured to use judges.

        Returns:
            bool: True if 'use_judges' is set to True in the task configuration, False otherwise.
        """
        return self.config.get("use_judges", False)

    def prepare_judge_configs(self) -> None:
        """Extract and normalize judge configuration parameters from task config.

        Processes the judge configuration to ensure all parameters are in list format,
        enabling support for multiple judge models per task. Also injects the
        use_cached_results flag into the task configuration.

        Args:
            task_config (dict): Task configuration dictionary containing a 'judge' key
            use_cached_results (bool): Whether to use cached inference results

        Returns:
            tuple[list, list, list, list]: A tuple containing four lists:
                - judge_model_name: List of model identifiers
                - judge_model_type: List of serving types (openai, vllm, etc.)
                - judge_init_args: List of initialization argument dictionaries
                - batch_api_calls: List of boolean flags for batching

        Side Effects:
            Modifies task_config by adding 'use_cached_results' key

        Example:
            Input config:
            ```yaml
            judge:
              judge_model_name: "gpt-4"
              judge_model_type: "openai"
              judge_init_args: {"base_url": "https://api.openai.com/v1"}
              batch_api_calls: true
            ```
            Returns: (["gpt-4"], ["openai"], [{"base_url": "https://api.openai.com/v1"}], [True])
        """
        judge_config = self.config.get("judge", {})
        judge_model_name = judge_config["judge_model_name"]
        judge_model_type = judge_config["judge_model_type"]
        judge_init_args = judge_config.get("judge_init_args", {})
        batch_api_calls = judge_config.get("batch_api_calls", False)
        judge_generation_kwargs = judge_config.get("judge_generation_kwargs", {})

        # convert to list if not already
        if type(judge_model_name) is str:
            judge_model_name = [judge_model_name]
        if type(judge_model_type) is str:
            judge_model_type = [judge_model_type]

        if type(judge_init_args) is DictConfig:
            judge_init_args = OmegaConf.to_object(judge_init_args)
        if type(judge_init_args) is dict:
            judge_init_args = [judge_init_args]
        if type(batch_api_calls) is bool:
            batch_api_calls = [batch_api_calls]

        if type(judge_generation_kwargs) is DictConfig:
            judge_generation_kwargs = OmegaConf.to_object(judge_generation_kwargs)
        if type(judge_generation_kwargs) is dict:
            judge_generation_kwargs = [judge_generation_kwargs]

        for jm_name, jm_type, jm_args, batch_api, jm_gen_kwargs in zip(
            judge_model_name,
            judge_model_type,
            judge_init_args,
            batch_api_calls,
            judge_generation_kwargs,
            strict=True,
        ):
            jm_gen_kwargs["seed"] = self.seed
            self.judge_configs[str((jm_name, jm_type, jm_args, batch_api))] = {
                "judge_model_name": jm_name,
                "judge_model_type": jm_type,
                "judge_init_args": jm_args,
                "batch_api_calls": batch_api,
                "judge_generation_kwargs": jm_gen_kwargs,
                "judge_seed": self.seed,
            }

    def get_dataloader_class(self) -> "type[AbstractDataloader]":
        """Retrieve the appropriate dataloader class for a specific task.

        Looks up the task configuration to determine if a custom dataloader is specified.
        If not, defaults to SeaHelmLocalDataloader. The dataloader class is dynamically
        imported using the file path and class name from the task config.

        Args:
            task_name (str): Name of the task (must exist in self.config["tasks"])

        Returns:
            type[AbstractDataloader]: Dataloader class that inherits from AbstractDataloader

        Example:
            Task config with custom dataloader:
            ```yaml
            my_task:
              dataloader_file: "seahelm_tasks/custom/my_dataloader.py"
              dataloader_class: "MyCustomDataloader"
            ```
            Returns: MyCustomDataloader class

            Task config without custom dataloader:
            Returns: SeaHelmLocalDataloader (default)
        """
        if "dataloader_file" in self.config:
            dataloader_file = self.config["dataloader_file"]
            dataloader_path = dataloader_file.removesuffix(".py").replace("/", ".")
            dataloader_class = self.config["dataloader_class"]
            Dataloader = getattr(
                importlib.import_module(dataloader_path), dataloader_class
            )
        else:
            from src.dataloaders.seahelm_local_dataloader import SeaHelmLocalDataloader

            Dataloader = SeaHelmLocalDataloader

        return Dataloader

    def get_judge_class(self) -> "type[SeaHelmJudge]":
        """Retrieve the Judge class for evaluating a specific task.

        Dynamically imports and returns the judge class specified in the task
        configuration. The judge class implements the logic for sending model
        outputs to a judge model and collecting judgments.

        Args:
            task_name (str): Name of the task requiring judge-based evaluation

        Returns:
            type[SeaHelmJudge]: Judge class that inherits from SeaHelmJudge

        Raises:
            KeyError: If task configuration is missing judge_file or judge_class
            ImportError: If the judge module cannot be imported
            AttributeError: If the specified class doesn't exist in the module

        Example:
            Task config:
            ```yaml
            translation-en-xx:
              judge_file: seahelm_tasks/nlg/translation/translation_judge.py
              judge_class: TranslationJudge
            ```
            Returns: TranslationJudge class
        """
        judge_file = self.config["judge_file"]
        judge_path = judge_file.removesuffix(".py").replace("/", ".")
        judge_class = self.config["judge_class"]
        Judge = getattr(importlib.import_module(judge_path), judge_class)

        return Judge

    def get_metric_class(self) -> "type[SeaHelmMetric]":
        """Retrieve the metric computation class for a specific task.

        Dynamically imports and returns the metric class specified in the task
        configuration.

        Args:
            task_name (str): Name of the task (must exist in self.config["tasks"])

        Returns:
            type[SeaHelmMetric]: Metric class that inherits from SeaHelmMetric

        Raises:
            KeyError: If task configuration is missing metric_file or metric_class
            ImportError: If the metric module cannot be imported
            AttributeError: If the specified class doesn't exist in the module

        Example:
            Task config:
            ```yaml
            nli:
              metric_file: src/metrics/f1_acc_metric.py
              metric_class: F1AccMetric
            ```
            Returns: F1AccMetric class
        """
        metric_file = self.config["metric_file"]
        metric_path = metric_file.removesuffix(".py").replace("/", ".")
        metric_class = self.config["metric_class"]
        Metric = getattr(importlib.import_module(metric_path), metric_class)

        return Metric

    def get_generation_kwargs(
        self,
    ) -> dict:
        """Get generation kwargs for the language model based on task configuration.

        This method constructs a dictionary of generation parameters for the language model
        based on the task configuration and evaluation settings. It handles different
        configurations for logprobs mode and base models, including setting up answer tags,
        stop tokens, and other generation parameters.

        Args:
            specific_task_config (dict): Configuration dictionary for the specific task
                containing prompt templates, max tokens, and other task-specific settings.
            use_logprobs (bool, optional): Whether to use logprobs during generation.
                When True, enables logprobs and sets up answer tag generation. Defaults to False.
            generate_to_answer_tag (bool, optional): Whether to generate up to the answer tag.
                Used for controlling generation behavior. Defaults to True.

        Returns:
            dict: Dictionary containing generation parameters that can include:
                - seed: Random seed for reproducible generation
                - max_tokens: Maximum number of tokens to generate
                - use_logprobs: Whether to use logprobs
                - generate_to_answer_tag: Whether to generate to answer tag
                - answer_tag: The answer tag string
                - answer_tag_separator: Separator string after answer tag
                - stop: List of stop tokens for generation

        Note:
            For logprobs mode, the method attempts to extract answer_tag_separator from
            the answer template using regex. If extraction fails, it falls back to
            the configured separator or defaults to a single space.
        """
        specific_task_config = self.config["languages"][self.lang]
        use_logprobs = self.config.get("use_logprobs", False)
        generate_to_answer_tag = self.config.get("generate_to_answer_tag", True)

        generation_kwargs = {"seed": self.seed}

        if "max_tokens" in specific_task_config:
            generation_kwargs["max_tokens"] = specific_task_config["max_tokens"]

        if use_logprobs:
            generation_kwargs["use_logprobs"] = True

            generation_kwargs["generate_to_answer_tag"] = generate_to_answer_tag
            generation_kwargs["answer_tag"] = specific_task_config["prompt_template"][
                "answer_tag"
            ]

            # TODO consider moving the answer_tag_separator to the prompt template
            try:
                answer_tag_separator = re.search(
                    "(?<={answer_tag})(.*)(?={label})",
                    specific_task_config["prompt_template"]["answer_template"],
                ).group(1)
            except Exception:
                # try to get an answer tag separator from the prompt template if it fails to extract it from the answer template
                answer_tag_separator = specific_task_config["prompt_template"].get(
                    "answer_tag_separator", " "
                )
            generation_kwargs["answer_tag_separator"] = answer_tag_separator
            logger.info('Answer tag separator: "' + answer_tag_separator + '"')
        elif self.is_base_model:
            generation_kwargs["generate_to_answer_tag"] = generate_to_answer_tag
            generation_kwargs["answer_tag"] = specific_task_config["prompt_template"][
                "answer_tag"
            ]

            generation_kwargs["stop"] = self.constants["few_shot_stop_tokens"]

        return generation_kwargs
