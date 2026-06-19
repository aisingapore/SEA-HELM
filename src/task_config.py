import importlib
import re
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

from src.base_logger import get_logger

if TYPE_CHECKING:
    from src.dataloaders.base_dataloader import AbstractDataloader
    from src.metrics.seahelm_metric import SeaHelmMetric

logger = get_logger(__name__)


class TaskConfig:
    """Configuration manager for a specific evaluation task.

    This class encapsulates all configuration settings for a single task evaluation,
    including task metadata, language-specific settings, generation parameters, and
    judge model configurations. It provides methods to retrieve dataloader, and metric
    classes dynamically based on the task configuration.

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
        reasoning_generation_kwargs: dict | None = None,
        sandbox_type: str | None = None,
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
            is_base_model (bool, optional): Whether the model being evaluated is a base model
                without reasoning capabilities. Defaults to False.
            constants (dict, optional): A dictionary of constant values that can be used in generation kwargs
                or other parts of the evaluation. Defaults to None.
            is_reasoning_model (bool, optional): Whether the model being evaluated is a reasoning model
                that generates think blocks. Defaults to False.
            reasoning_generation_kwargs (dict, optional): Additional generation kwargs specific to reasoning models,
                such as max_think_tokens. Defaults to None.
            sandbox_type (str, optional): The type of sandbox to use for evaluation (e.g., 'enroot', 'singularity').
                Defaults to None, which means auto-detect the sandbox.

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
        self.reasoning_generation_kwargs = (
            reasoning_generation_kwargs
            if reasoning_generation_kwargs is not None
            else {}
        )
        self.sandbox_type = sandbox_type

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

        Side Effects:
            Modifies task_config by adding 'use_cached_results' key

        Example:
            Input config:
            ```yaml
            judge:
              judge_model_name: "gpt-4"
              judge_model_type: "openai"
              judge_init_args: {"base_url": "https://api.openai.com/v1"}
            ```
            Returns: (["gpt-4"], ["openai"], [{"base_url": "https://api.openai.com/v1"}], [True])
        """
        judge_config = self.config.get("judge", {})
        judge_model_name = judge_config["judge_model_name"]
        judge_model_type = judge_config["judge_model_type"]
        judge_init_args = judge_config.get("judge_init_args", {})
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

        if type(judge_generation_kwargs) is DictConfig:
            judge_generation_kwargs = OmegaConf.to_object(judge_generation_kwargs)
        if type(judge_generation_kwargs) is dict:
            judge_generation_kwargs = [judge_generation_kwargs]

        for jm_name, jm_type, jm_args, jm_gen_kwargs in zip(
            judge_model_name,
            judge_model_type,
            judge_init_args,
            judge_generation_kwargs,
            strict=True,
        ):
            jm_gen_kwargs["seed"] = self.seed
            self.judge_configs[str((jm_name, jm_type, jm_args))] = {
                "judge_model_name": jm_name,
                "judge_model_type": jm_type,
                "judge_init_args": jm_args,
                "judge_generation_kwargs": jm_gen_kwargs,
                "judge_seed": self.seed,
            }

    def get_strategy(self, default_strategy: str | None = None) -> str:
        """Get the inference strategy name from the task configuration.

        Returns:
            str: The name of the inference strategy to use for this task, as specified
                in the task configuration under the 'inference_strategy' key. If not
                specified, returns the provided default_strategy.
        """
        return self.config.get("inference_strategy", default_strategy)

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

    def get_generation_kwargs(self) -> dict:
        """Get generation kwargs for the language model based on task configuration.

        This method constructs a dictionary of generation parameters for the language model
        based on the task configuration and evaluation settings. It handles different
        configurations for logprobs mode and base models, including setting up answer tags,
        stop tokens, and other generation parameters.

        Returns:
            dict: Dictionary containing generation parameters that can include:
                - seed: Random seed for reproducible generation
                - max_tokens: Maximum number of tokens to generate
                - use_logprobs: Whether to use logprobs
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

        generation_kwargs = {"seed": self.seed}

        if "max_tokens" in specific_task_config:
            generation_kwargs["max_tokens"] = specific_task_config["max_tokens"]

        # Update generation kwargs for reasoning models
        # Follows the kwargs for DeepSeek models
        if self.is_reasoning_model:
            # TODO: max_think_tokens is not defined
            generation_kwargs["max_tokens"] += self.reasoning_generation_kwargs[
                "max_think_tokens"
            ]
            logger.info(
                "Model is a reasoning model. Increasing the max_tokens to '%d'.",
                generation_kwargs["max_tokens"],
            )

        additional_kwargs = {}
        if use_logprobs:
            additional_kwargs["use_logprobs"] = True
            additional_kwargs["answer_tag"] = specific_task_config["prompt_template"][
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
            additional_kwargs["answer_tag_separator"] = answer_tag_separator
            logger.info('Answer tag separator: "' + answer_tag_separator + '"')
        elif self.is_base_model:
            additional_kwargs["answer_tag"] = specific_task_config["prompt_template"][
                "answer_tag"
            ]

            generation_kwargs["stop"] = self.constants["few_shot_stop_tokens"]

        return generation_kwargs, additional_kwargs
