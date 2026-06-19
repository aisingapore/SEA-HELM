import argparse
import glob
import json
import os
import sys
from datetime import datetime as dt
from typing import Any

import pandas as pd
from omegaconf import ListMergeMode, OmegaConf

# Add main directory to path for imports. Importing LiteLLM used to add os.cwd() to path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aggregate_metrics import aggregate_metrics
from src.base_logger import get_logger, setup_root_logger
from src.collect_env import get_pretty_env_info
from src.dataloaders.base_dataloader import AbstractDataloader
from src.inference_strategy import get_inference_strategy_class
from src.queue_manager import QueueManager
from src.serving import (
    get_serving_class,
    is_serving_type_remote,
)
from src.task_config import TaskConfig
from src.utils import get_git_commit_hash, simple_parse_args_string

logger = get_logger(__name__)


class SeaHelmEvaluation:
    """Orchestrates the complete evaluation pipeline for language models on SEA-HELM tasks.

    This class manages the end-to-end evaluation process including:
    - Loading task configurations and datasets
    - Running model inference across multiple tasks and languages
    - Managing LLM-as-a-judge evaluations (both local and remote)
    - Computing metrics and aggregating results
    - Handling caching and result persistence

    The evaluation pipeline supports different model types (base, instruction-tuned,
    reasoning, vision) and multiple serving backends (vLLM, OpenAI, Anthropic, etc.).
    """

    def __init__(
        self,
        llm: Any,
        tasks_configuration: str | list,
        output_dir: str,
        model_name: str,
        reasoning_generation_config_file: str = "seahelm_tasks/reasoning_generation_config.yaml",
        task_config_file: str = "seahelm_tasks/task_config.yaml",
        constants_file: str = "seahelm_tasks/constants.yaml",
        task_folder: str = "seahelm_tasks",
        inference_strategy: str | None = None,
        is_base_model: bool = False,
        is_vision_model: bool = False,
        is_reasoning_model: bool = False,
        inference_file_type: str = "jsonl",
        tokenize_prompts: bool = True,
        skip_task: list | None = None,
        limit: int | None = None,
        seed: int = 1234,
        no_batching: bool = True,
        ignore_missing_files: bool = False,
        run_number: int = 1,
        datetime: str | None = None,
        sandbox_type: str | None = None,
    ):
        """Initialize SeaHelmEvaluation instance with configurations and model settings.

        Sets up the evaluation environment by loading task configurations, initializing
        directory structures, and configuring model-specific settings. This includes
        determining which tasks to run based on model type.

        Args:
            llm (Any): Language model instance for inference (can be None for cached-only runs)
            tasks_configuration (str | list): Task set name(s) from task_config.yaml (e.g., "seahelm", "english_evals")
            output_dir (str): Root directory for saving results, configs, logs, and inferences
            model_name (str): Model identifier or path (used for directory naming and identification)
            reasoning_generation_config_file (str, optional): Path to reasoning model config.
                Defaults to "seahelm_tasks/reasoning_generation_config.yaml".
            task_config_file (str, optional): Path to task set definitions.
                Defaults to "seahelm_tasks/task_config.yaml".
            constants_file (str, optional): Path to constants configuration.
                Defaults to "seahelm_tasks/constants.yaml".
            task_folder (str, optional): Root directory containing task definitions.
                Defaults to "seahelm_tasks".
            is_base_model (bool, optional): If True, skips instruction-following tasks and uses
                base model chat template. Defaults to False.
            is_vision_model (bool, optional): If True, enables vision-specific functions (currently there are none).
                Defaults to False.
            is_reasoning_model (bool, optional): If True, adds thinking tokens and increases
                max_tokens for reasoning. Defaults to False.
            inference_file_type (str, optional): File format for saving inferences ("csv" or "jsonl").
                Defaults to "jsonl".
            tokenize_prompts (bool, optional): Whether to tokenize and save prompts.
                Auto-disabled for VertexAI/Anthropic. Defaults to True.
            skip_task (list | None, optional): List of task names to skip. If None, uses
                default skip list based on model type. Defaults to None.
            limit (int | None, optional): Maximum examples per task for testing.
                Should not be used in production runs. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to 1234.
            no_batching (bool, optional): If True, processes examples one at a time.
                Defaults to True.
            ignore_missing_files (bool, optional): If True, continues execution when
                cached results are missing. Defaults to False.
            run_number (int, optional): Run identifier for multiple runs (0-indexed).
                Defaults to 1.
            datetime (str | None, optional): ISO format timestamp for file naming.
                If None, uses current datetime. Defaults to None.
            sandbox_type (str | None, optional): Type of sandbox to use for code execution.
                Defaults to None.

        Raises:
            AssertionError: If tasks_configuration is not found in task_config.yaml
        """
        constants = OmegaConf.load(constants_file)
        self.constants = OmegaConf.to_object(constants)

        logger.info(
            "%s\nEvaluating %s as %s%s%s model...\n%s",
            "<>" * 50,
            model_name,
            "base" if is_base_model else "instruction-tuned",
            " vision" if is_vision_model else "",
            " reasoning" if is_reasoning_model else "",
            "<>" * 50,
        )

        self.model_name = model_name
        self.is_base_model = is_base_model
        self.is_vision_model = is_vision_model
        self.is_reasoning_model = is_reasoning_model
        self.inference_strategy = inference_strategy
        self.sandbox_type = sandbox_type

        config, self.task_list_by_lang, task_alias = self.load_config_from_folders(
            task_folder
        )
        task_config = OmegaConf.load(task_config_file)

        self.run_base_path = os.path.join(
            output_dir, os.path.basename(model_name), f"run_{run_number}"
        )

        # add seed and run_number to config
        self.seed = seed
        self.run_number = run_number

        seed_config = OmegaConf.create({"seed": seed, "run_number": run_number})
        self.config = OmegaConf.merge(config, seed_config)

        # load tasks to run from configuration file
        if isinstance(tasks_configuration, list):
            tasks = []
            for _task_config in tasks_configuration:
                tasks.append(task_config.get(_task_config, None))
            self.tasks = OmegaConf.merge(
                *tasks, list_merge_mode=ListMergeMode.EXTEND_UNIQUE
            )
        else:
            self.tasks = task_config.get(tasks_configuration, None)
        assert self.tasks is not None, (
            f"Unable to find tasks_configuration in task_config.yaml. Received {self.tasks_configuration}."
        )

        # convert self.tasks back to dictionary
        self.tasks = OmegaConf.to_object(self.tasks)

        if datetime is not None:
            self.datetime = datetime
        else:
            self.datetime = dt.now().isoformat()

        # convert "all" case to list of task
        for lang, tasks in self.tasks.items():
            if isinstance(self.tasks[lang], str):
                assert tasks == "all", (
                    f"The only string allowed for definition of tasks is 'all', {self.tasks[lang]} was defined instead for {lang}. Please use a list of tasks if you only want to run a subset of tasks."
                )

                self.tasks[lang] = self.task_list_by_lang[lang]
            else:
                for alias, value in task_alias.items():
                    if alias in tasks:
                        self.tasks[lang].remove(alias)
                        self.tasks[lang].extend(value)

        if self.is_reasoning_model:
            reasoning_config = OmegaConf.load(reasoning_generation_config_file)
            self.config = OmegaConf.merge(self.config, reasoning_config)

        # This ought to be set in each task config, but this sets the default number of in-context examples to use for fewshot testing.
        # If used later, a warning is raised.
        if self.is_base_model:
            self.default_num_in_context_examples = self.constants[
                "base_num_in_context_examples"
            ]
        else:
            self.default_num_in_context_examples = self.constants[
                "instruct_num_in_context_examples"
            ]

        # TODO deprecated fewshot_as_multiturn
        self.fewshot_as_multiturn = True
        self.inference_file_type = inference_file_type

        self.tokenize_prompts = tokenize_prompts

        self.limit = limit
        self.no_batching = no_batching
        self.ignore_missing_files = ignore_missing_files

        _default_skip_task = {
            "base_models": self.constants["base_models_skip_tasks"],
            "instruct_models": self.constants["instruct_models_skip_tasks"],
        }
        if skip_task is None:
            self.skip_task = _default_skip_task[
                "base_models" if is_base_model else "instruct_models"
            ]
        else:
            self.skip_task = skip_task

        # add env info to config file
        run_env = {
            "env_info": get_pretty_env_info(),
            "seahelm_git_hash": get_git_commit_hash(),
        }
        if llm is not None:
            run_env.update(llm.get_run_env())

        self.config.run_env = run_env

        # add model args to config file
        run_args = {}
        for key in [
            "model_name",
            "inference_strategy",
            "is_base_model",
            "is_vision_model",
            "is_reasoning_model",
            "fewshot_as_multiturn",
            "tokenize_prompts",
            "skip_task",
            "limit",
            "no_batching",
            "sandbox_type",
        ]:
            run_args[key] = self.__dict__[key]

        self.config.run_args = run_args

        # remove unneeded task configurations
        tasks = self.config.tasks.copy()
        for task in tasks:
            languages = self.config.tasks[task]["languages"]
            for lang in languages.copy():
                if lang not in self.tasks:
                    del self.config.tasks[task]["languages"][lang]
                    continue

                if task not in self.tasks[lang]:
                    del self.config.tasks[task]["languages"][lang]

            if self.config.tasks[task]["languages"] == {}:
                del self.config.tasks[task]
                continue

        # save config folder
        self.save_config()

    def load_config_from_folders(self, folder: str) -> tuple[OmegaConf, dict, dict]:
        """Load and merge configuration files from a folder structure.

        Recursively searches for config.yaml files in the given folder and its subdirectories,
        then merges them into a single configuration object. Also creates task lists organized
        by language and task aliases for aggregated tasks.

        Args:
            folder (str): Path to the folder containing configuration files

        Returns:
            tuple: A tuple containing:
                - output_config (OmegaConf): Merged configuration object with all tasks
                - task_list_by_lang (dict): Dictionary mapping languages to lists of task names
                - task_alias (dict): Dictionary mapping aggregation groups to lists of task names
        """
        config_files = glob.glob(f"{folder}/**/config.yaml", recursive=True)

        output_config = OmegaConf.create({})
        task_list_by_lang = {}
        task_alias = {}
        config_list = []
        for config_file in config_files:
            config = OmegaConf.load(config_file)
            for task_name in config:
                config_list.append({task_name: config.get(task_name)})

                # create task list by language
                for lang in config[task_name]["languages"]:
                    if lang not in task_list_by_lang:
                        task_list_by_lang[lang] = [task_name]
                    else:
                        task_list_by_lang[lang].append(task_name)

                # create task alias for aggregated tasks
                if "aggregation_group" in config[task_name]:
                    if config[task_name]["aggregation_group"] not in task_alias:
                        task_alias[config[task_name]["aggregation_group"]] = [task_name]
                    else:
                        task_alias[config[task_name]["aggregation_group"]].append(
                            task_name
                        )

        output_config = OmegaConf.merge(*config_list)
        OmegaConf.resolve(output_config)
        output_config = OmegaConf.create({"tasks": output_config})

        return output_config, task_list_by_lang, task_alias

    @staticmethod
    def create_seahelm_run_folders(
        output_dir: str, model_name: str, run_number: int
    ) -> None:
        """Create the directory structure for storing evaluation outputs.

        Sets up the required folder hierarchy for a model evaluation run:
        - inferences/: Stores raw model outputs by language and task
        - results/: Contains metric JSONs and aggregated scores
        - configs/: Saves run configurations and task settings
        - logs/: Holds execution logs

        This is a static method called before SeaHelmEvaluation initialization to
        ensure logging directories exist from the start.

        Args:
            output_dir (str): Root directory for all evaluation outputs
            model_name (str): Model identifier used in path construction (basename extracted)
            run_number (int): Run identifier for organizing multiple evaluation runs

        Example:
            >>> SeaHelmEvaluation.create_seahelm_run_folders(
            ...     "./results", "meta-llama/Llama-3-8B-Instruct", 1
            ... )
            Creates: ./results/Llama-3-8B-Instruct/run_1/{inferences,results,configs,logs}/
        """
        run_base_path = os.path.join(
            output_dir, os.path.basename(model_name), f"run_{run_number}"
        )
        os.makedirs(os.path.join(run_base_path, "inferences"), exist_ok=True)
        os.makedirs(os.path.join(run_base_path, "results"), exist_ok=True)
        os.makedirs(os.path.join(run_base_path, "configs"), exist_ok=True)
        os.makedirs(os.path.join(run_base_path, "logs"), exist_ok=True)

    def get_config_filepath(self) -> str:
        """Get the timestamped configuration file path for the current run.

        Constructs the full path to where the run configuration YAML will be saved,
        including the model name and ISO datetime stamp for uniqueness.

        Returns:
            str: Absolute path to the configuration YAML file in the format:
                {run_base_path}/configs/{model_basename}_run_config_{datetime}.yaml

        Example:
            >>> evaluator.get_config_filepath()
            './results/Llama-3-8B/run_1/configs/Llama-3-8B_run_config_2025-12-04T10:30:45.123456.yaml'
        """
        config_filepath = os.path.join(
            self.run_base_path,
            "configs",
            f"{os.path.basename(self.model_name)}_run_config_{self.datetime}.yaml",
        )
        return config_filepath

    def save_config(self) -> None:
        """Save the complete evaluation configuration to a timestamped YAML file.

        Persists all configuration settings including:
        - Task configurations and language mappings
        - Model parameters and generation settings
        - Runtime arguments (seeds, limits, flags)
        - Environment information (Python version, package versions, git hash)
        - Model-specific run environment details

        This creates a YAML file in {run_base_path}/configs/ with the current
        datetime stamp in the filename.
        """
        config_filepath = self.get_config_filepath()
        logger.info(
            """---------- Configuration saving ----------
Saving run config to output folder...
Filepath: %s""",
            config_filepath,
        )

        with open(config_filepath, "w") as fp:
            OmegaConf.save(config=self.config, f=fp)
        logger.info("Config file saved!\n")

    def _check_validity_of_task(self, task: str, lang: str) -> bool:
        """Check if a task is valid and should be executed.

        This method validates whether a task should be run by checking:
        1. If the task is in the skip_task list
        2. If the task exists in the defined tasks for the specified language

        Args:
            task (str): The name of the task to validate
            lang (str): The language code for the task

        Returns:
            bool: True if the task is valid and should be executed, False otherwise
        """
        if task in self.skip_task:
            logger.info(
                "Task in skip task list: %s. Skipping task '%s' for lang '%s'.",
                self.skip_task,
                task,
                lang,
            )
            return False

        if task not in self.task_list_by_lang[lang]:
            logger.error(
                "Task '%s' is not found in the list of defined tasks for lang '%s'.",
                task,
                lang,
            )
            return False

        return True

    def setup_dataloader(self, task_config):
        Dataloader = task_config.get_dataloader_class()
        dataloader = Dataloader(
            task_config,
            self.default_num_in_context_examples,
            is_base_model=self.is_base_model,
            model_name=self.model_name,
            run_base_path=self.run_base_path,
            inference_file_type=self.inference_file_type,
        )

        return dataloader

    def run_single_task_inference(
        self,
        model: Any,
        task_config: TaskConfig,
        limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Run inference for a single task and language combination.

        This method handles the complete inference pipeline for a specific task, including:
        - Loading and preparing the dataset
        - Setting up generation parameters
        - Running inference through the model
        - Processing and cleaning responses (especially for reasoning models)
        - Saving results to disk

        Args:
            model (Any): The model to use for inference
            task_config (TaskConfig): Configuration object for the specific task
            limit (int, optional): Maximum number of examples to process. Defaults to None.

        Returns:
            tuple[pd.DataFrame, dict]: A tuple containing:
                - dataframe: DataFrame with inference results
                - cache_status: Dictionary containing cache status information including:
                    - inference_time_taken: Total time taken for inference
                    - is_cached: Boolean indicating if cached results were used

        Raises:
            EngineDeadError: If the vLLM engine becomes unresponsive
            ValueError: If there are issues with the inference process
            Exception: For any other unexpected errors during inference
        """
        task_name = task_config.task_name
        lang = task_config.lang

        logger.info(
            "---------- Inference | Lang: %s | Task: %s ----------\nTesting Competency: %s",
            lang.upper(),
            task_name.upper(),
            task_config.config["competency"].upper(),
        )
        # run inference strategy
        cache_status = {
            task_name: {
                "inference_time_taken": [],
                "is_cached": [],
            }
        }

        try:
            is_logprobs = "logprobs" in task_name
            strategy = task_config.get_strategy(self.inference_strategy)
            InferenceStrategy = get_inference_strategy_class(
                model,
                strategy,
                is_base_model=self.is_base_model,
                is_logprobs=is_logprobs,
            )
            inference_strategy = InferenceStrategy(
                serving_class=model,
                task_config=task_config,
            )

            # load dataset
            dataloader = self.setup_dataloader(task_config)
            dataloader.load_dataset(limit)
            n_turns = dataloader.get_num_turns()

            # run inference strategy
            for turn in range(1, n_turns + 1):
                # Set up generation kwargs
                batch_filepath = dataloader.get_batch_filepath(
                    turn=turn, file_type="jsonl"
                )
                batch_response_filepath = dataloader.get_batch_response_filepath(
                    turn=turn, file_type="jsonl"
                )

                generation_kwargs, additional_kwargs = (
                    task_config.get_generation_kwargs()
                )
                conversations = dataloader.prepare_conversations_for_inference(
                    turn, self.fewshot_as_multiturn
                )

                labels = dataloader.prepare_labels_for_inference()

                inference_time_taken, is_cached = inference_strategy.run_inference(
                    conversations,
                    generation_kwargs,
                    batch_filepath,
                    batch_response_filepath,
                    custom_ids=None,
                    additional_kwargs=additional_kwargs,
                    labels=labels,
                )
                cache_status[task_name]["inference_time_taken"].append(
                    inference_time_taken
                )
                cache_status[task_name]["is_cached"].append(is_cached)

                # update dataloader with parsed responses
                # TODO Handle logprobs case
                dataloader.load_model_outputs_into_dataset(turn)
            dataloader.convert_dataset_to_dataframe()

            logger.info("Inference for task '%s' completed!\n", task_name.upper())
        except Exception as e:
            logger.error(
                "Failed to run inference for task %s and lang %s", task_name, lang
            )
            logger.exception(e)
            dataloader = None

        return dataloader, cache_status

    def run_single_judgement(
        self,
        model: Any,
        dataloader: AbstractDataloader,
        task_config: TaskConfig,
        judge_config: dict,
    ) -> None:
        """Execute inference using the judge model for a single task and language.

        Initializes the appropriate Judge and Metric classes, then runs the judge
        model on the dataloader's inference results.

        Args:
            model: Serving instance for the LLM judge (can be any serving type)
            dataloader (AbstractDataloader): Dataloader containing model inference results
            task_config (TaskConfig): Task configuration including judge settings and generation kwargs
            judge_config (dict): Configuration for initializing the judge model, including:

        Side Effects:
            - Updates dataloader.dataframe with judge evaluations
            - Writes updated results to disk via dataloader
            - Logs evaluation progress

        Exception Handling:
            Catches and logs all exceptions, allowing evaluation to continue for
            remaining tasks even if one task's judgment fails.
        """
        task_name = task_config.task_name
        lang = task_config.lang
        try:
            logger.info(
                "--------- Judgement | Lang: %s | Task: %s ----------",
                lang.upper(),
                task_name.upper(),
            )

            InferenceStrategy = get_inference_strategy_class(
                model, judge_config.get("judge_inference_strategy", None)
            )
            inference_strategy = InferenceStrategy(
                serving_class=model,
                task_config=task_config,
            )

            # initialise metric class as we need the response extraction function
            Metric = task_config.get_metric_class()
            metric = Metric(
                dataloader=dataloader,
                task_config=task_config,
            )

            judge_batch_filepath = dataloader.get_judge_batch_filepath()
            judge_batch_response_filepath = (
                dataloader.get_judge_batch_response_filepath()
            )

            judge_generation_kwargs = judge_config["judge_generation_kwargs"]
            conversations, custom_ids = dataloader.prepare_conversations_for_judgements(
                metric
            )

            # TODO save out judge inference times and cache status
            _, _ = inference_strategy.run_inference(
                conversations=conversations,
                generation_kwargs=judge_generation_kwargs,
                batch_filepath=judge_batch_filepath,
                batch_response_filepath=judge_batch_response_filepath,
                custom_ids=custom_ids,
                additional_kwargs=None,
            )

            logger.info("Judgement for task '%s' completed!\n", task_name.upper())
        except Exception as e:
            logger.error(
                "Failed to run judgement for task %s and lang %s", task_name, lang
            )
            logger.exception(e)

    def update_metrics(
        self,
        new_metric_json: dict,
        metrics: dict,
        task_config: dict,
    ) -> dict:
        """Update the metrics dictionary with new evaluation results.

        This method adds new metric results to the existing metrics dictionary,
        organizing them by language and competency. It ensures the proper nested
        structure is maintained in the metrics dictionary.

        Args:
            new_metric_json (dict): New metric results to add to the metrics dictionary
            metrics (dict): Existing metrics dictionary to update
            task_config (dict): Configuration for the task being evaluated
            lang (str): Language code for the evaluation results

        Returns:
            dict: Updated metrics dictionary with the new results added
        """
        competency = task_config.config["competency"]
        lang = task_config.lang
        if lang not in metrics:
            metrics.update({lang: {competency: {}}})
        elif competency not in metrics[lang]:
            metrics[lang].update({competency: {}})
        for key, value in new_metric_json.items():
            if key not in metrics[lang][competency]:
                metrics[lang][competency].update({key: value})
            else:
                metrics[lang][competency][key].update(value)

        return metrics

    def run_single_task_evaluation(
        self,
        dataloader: AbstractDataloader,
        task_config: dict,
    ) -> dict:
        """Compute metrics for a single task using the appropriate metric class.

        Initializes the metric evaluator and runs the calculate_metrics() method on
        the dataloader's inference results. Also counts errors in the responses and
        includes them in the returned metrics.

        Args:
            dataloader (AbstractDataloader): Dataloader containing inference results and ground truth
            task_config (dict): Task configuration including metric specifications
            task_name (str): Name of the task being evaluated
            lang (str): Language code (e.g., "en", "id", "vi")

        Returns:
            dict: Nested dictionary with structure {task_name: {metric_name: value, ...}}
                Always includes an 'errors' key with the count of failed inferences.
                On evaluation failure, returns {task_name: {primary_metric: 0, "error": msg}}

        Side Effects:
            - Updates dataloader with computed metrics
            - Writes updated inference results to disk
            - Logs evaluation progress and any errors

        Example:
            >>> metric_json = evaluator.run_single_task_evaluation(
            ...     dataloader, task_config, "nli", "id"
            ... )
            >>> metric_json
            {"nli": {"accuracy": 0.75, "f1": 0.73, "errors": 2}}
        """
        task_name = task_config.task_name
        lang = task_config.lang
        try:
            logger.info(
                "--------- Evaluation | Lang: %s | Task: %s ----------",
                lang.upper(),
                task_name.upper(),
            )
            Metric = task_config.get_metric_class()
            logger.info("Evaluating '%s' using %s", task_name.upper(), Metric.__name__)
            evaluation_metric = Metric(
                dataloader=dataloader,
                task_config=task_config,
            )

            metric_json = evaluation_metric.evaluate_responses()
            dataloader.write_out_dataframe()

            # TODO move this elsewhere
            # metric_json[task_name]["errors"] = get_error_count(
            #     dataloader.dataframe["errors"]
            # )

            logger.info("Evaluation for task '%s' completed!\n", task_name.upper())
        except Exception as e:
            logger.error(
                "Failed to run evaluation for task %s and lang %s", task_name, lang
            )
            logger.exception(e)
            logger.warning(
                "Setting metric %s to 0 for task %s",
                task_config.config["metric"],
                task_name,
            )
            metric_json = {
                task_name: {
                    task_config.config["metric"]: 0,
                    "error": "Failed to run evaluation for task",
                },
            }

        return metric_json

    def write_metric_to_file(self, metrics: dict) -> None:
        """Persist evaluation metrics to a timestamped JSON file.

        Saves the complete metrics dictionary to a JSON file with proper formatting
        (2-space indentation). The file is written to the results directory with a datetime stamp.

        Args:
            metrics (dict): Nested dictionary containing all evaluation metrics with structure:
                {lang: {competency: {task: {metric: value, ...}, ...}, ...}, ...}

        Side Effects:
            Creates or overwrites the JSON file at the path returned by
            get_results_json_filepath()

        Raises:
            IOError: If there are issues writing to the specified file path
            OSError: If the results directory doesn't exist or lacks write permissions

        Example File Structure:
            ```json
            {
              "en": {
                "nlu": {
                  "nli": {"accuracy": 0.75, "errors": 0}
                }
              }
            }
            ```
        """
        json_filepath = self.get_results_json_filepath()

        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def get_results_json_filepath(self) -> str:
        """Construct the timestamped filepath for the evaluation results JSON.

        Generates the full path where the aggregated metrics will be saved,
        incorporating the model name and ISO datetime stamp for uniqueness.

        Returns:
            str: Absolute path to the results JSON file in the format:
                {run_base_path}/results/{model_basename}_seahelm_results_{datetime}.json

        Example:
            >>> evaluator.get_results_json_filepath()
            './results/Llama-3-8B/run_1/results/Llama-3-8B_seahelm_results_2025-12-04T10:30:45.123456.json'
        """
        json_filepath = os.path.join(
            self.run_base_path,
            "results",
            f"{os.path.basename(self.model_name)}_seahelm_results_{self.datetime}.json",
        )
        return json_filepath

    def invalidate_judge_cache(
        self, dataloader: AbstractDataloader, task_name: str, lang: str
    ) -> None:
        """Invalidate cached judge inference results if the corresponding task inference results are not cached."""

        if hasattr(dataloader, "get_judge_batch_response_filepath"):
            judge_response_filepath = dataloader.get_judge_batch_response_filepath()
            if os.path.exists(judge_response_filepath):
                logger.warning(
                    "Inference results for task %s and lang %s is not cached, but found existing judge inference results at %s. Renaming judge inference results to ensure correct evaluation.",
                    task_name,
                    lang,
                    judge_response_filepath,
                )
                os.replace(
                    judge_response_filepath,
                    judge_response_filepath + ".invalid",
                )

    def run_evaluation(self, llm: Any, use_cached_results: bool = True) -> None:
        """Run the complete evaluation pipeline for all configured tasks and languages.

        This method orchestrates the entire evaluation workflow across multiple tasks
        by coordinating inference, judging, and metric computation. It implements a
        sophisticated execution order to optimize resource usage and enable async
        remote judge evaluations.

        Execution Order:
            1. Remote LLM Judge Tasks (mt-bench, mental-health-safety, arena-hard-v2):
                - Run inference on the model being evaluated
                - Submit outputs to remote judge APIs asynchronously
                - Continue with other tasks while waiting for judge responses

            2. Normal Tasks (all non-judge tasks):
                - Run inference and compute metrics immediately
                - Write results incrementally to JSON

            3. Local LLM Judge Tasks (deferred):
                - Free memory by cleaning up the main LLM
                - Load local judge models and run judgments
                - Compute metrics from judge evaluations

            4. Remote LLM Judge Collection:
                - Collect async judge responses
                - Compute final metrics from remote judgments

            5. Aggregation:
                - Aggregate metrics across tasks, competencies, and languages
                - Write final results with SEA average scores

        Args:
            llm (Any): Language model instance to evaluate (or None for cached-only runs)
            use_cached_results (bool, optional): If True, loads cached inference results
                when available instead of re-running inference. If False, always runs
                fresh inference even if cache exists. Defaults to True.

        Side Effects:
            - Runs model inference and saves results to inferences/ directory
            - Writes incremental metrics to results JSON after each task
            - Logs detailed progress information
            - May cleanup/reload models to manage memory
            - Creates async process pool for remote judge evaluations

        Note:
            Tasks are filtered based on:
            - Task validity (_check_validity_of_task)
            - Run number limits (_check_run_number)
            - Skip task list (self.skip_task)

            Results are persisted incrementally to allow resumption if interrupted.
        """
        metrics = {}

        normal_queue = []
        remote_queue = QueueManager()
        local_queue = QueueManager()

        for lang, tasks_by_lang in self.tasks.items():
            for task_name in tasks_by_lang:
                if not self._check_validity_of_task(task_name, lang):
                    continue

                task_config = TaskConfig(
                    config=self.config["tasks"][task_name],
                    task_name=task_name,
                    lang=lang,
                    seed=self.seed,
                    is_base_model=self.is_base_model,
                    use_cached_results=use_cached_results,
                    constants=self.constants,
                    is_reasoning_model=self.is_reasoning_model,
                    reasoning_generation_kwargs=self.config.get(
                        "reasoning_generation_kwargs", None
                    ),
                    sandbox_type=self.sandbox_type,
                )

                if not task_config.should_task_run_for_run_number(self.run_number):
                    continue

                should_run_first = False
                if task_config.task_uses_judges():
                    task_config.prepare_judge_configs()

                    for judge_config in task_config.judge_configs.values():
                        if is_serving_type_remote(judge_config["judge_model_type"]):
                            should_run_first = True
                            break

                if should_run_first:
                    # Iterate through to run remote LLM judgement task inferences first
                    logger.info(
                        "Running inferences for lang %s, task %s first due to remote LLM judge...",
                        lang.upper(),
                        task_name.upper(),
                    )
                    dataloader, cache_status = self.run_single_task_inference(
                        model=llm,
                        task_config=task_config,
                        limit=self.limit,
                    )
                    if not all(cache_status[task_name]["is_cached"]):
                        self.invalidate_judge_cache(dataloader, task_name, lang)

                    metrics = self.update_metrics(cache_status, metrics, task_config)

                    # Creates the parameters for the judge LLM to be called later
                    for key, judge_config in task_config.judge_configs.items():
                        if is_serving_type_remote(judge_config["judge_model_type"]):
                            if not remote_queue.is_pool_started():
                                remote_queue.start_pool()
                            judge_model = get_serving_class(
                                model_name=judge_config["judge_model_name"],
                                model_type=judge_config["judge_model_type"],
                                seed=self.seed,
                                **judge_config["judge_init_args"],
                            )
                            # NOTE This assumes that the judge model is a small object that can be easily serialized and passed to the process pool. If the judge model is large or has non-serializable components, this approach may need to be revised to initialize the judge model within the async function instead of passing it as a parameter.
                            remote_queue.add_to_async_queue(
                                key=(lang, task_name, judge_config["judge_model_name"]),
                                function=self.run_single_judgement,
                                params=(
                                    judge_model,
                                    dataloader,
                                    task_config,
                                    judge_config,
                                ),
                            )
                            remote_queue.add_to_evaluation_queue(
                                key=key,
                                evaluation_params=(dataloader, task_config),
                                judge_configs=None,
                            )
                        else:
                            local_queue.add_to_evaluation_queue(
                                key=key,
                                evaluation_params=(dataloader, task_config),
                                judge_configs=judge_config,
                            )
                else:
                    normal_queue.append(task_config)

        # proceed with the rest of tasks (local LLM judges and normal tasks)
        for task_config in normal_queue:
            logger.info("Running evaluation for local LLM judges and normal tasks...")
            # inference_strategy = task_config.get_inference_strategy()
            dataloader, cache_status = self.run_single_task_inference(
                model=llm,
                task_config=task_config,
                limit=self.limit,
            )
            metrics = self.update_metrics(cache_status, metrics, task_config)

            if task_config.task_uses_judges():
                if not all(cache_status[task_config.task_name]["is_cached"]):
                    self.invalidate_judge_cache(dataloader, task_config.task_name, lang)

                logger.info(
                    "Deferring evaluation for task %s to run last as it uses a local judge model.",
                    task_config.task_name.upper(),
                )

                for key, judge_config in task_config.judge_configs.items():
                    local_queue.add_to_evaluation_queue(
                        key=key,
                        evaluation_params=(dataloader, task_config),
                        judge_configs=judge_config,
                    )
            else:
                metric_json = self.run_single_task_evaluation(dataloader, task_config)
                metrics = self.update_metrics(metric_json, metrics, task_config)
                # Write out metrics to file
                self.write_metric_to_file(metrics)

        if not local_queue.is_queue_empty():
            logger.info("Running judging for local LLM judges...")
            # Runs any serving cleanup to free up memory. Currently only affect VLLMServing
            llm.cleanup()

            for _, tasks in local_queue.iterate_through_evaluation_queue():
                judge_config = tasks[0]["judge_configs"]
                judge_model = get_serving_class(
                    model_name=judge_config["judge_model_name"],
                    model_type=judge_config["judge_model_type"],
                    seed=self.seed,
                    **judge_config["judge_init_args"],
                )

                for task in tasks:
                    dataloader, task_config = task["evaluation_params"]
                    self.run_single_judgement(
                        model=judge_model,
                        dataloader=dataloader,
                        task_config=task_config,
                        judge_config=task["judge_configs"],
                    )

                # clear judge model
                judge_model.cleanup()

            logger.info("Local judgement completed for local LLM judges.")

        if not local_queue.is_queue_empty():
            logger.info("Running evaluation for local LLM judges...")
            for task in local_queue.get_unique_set_of_evaluation_params():
                dataloader, task_config = task

                metric_json = self.run_single_task_evaluation(dataloader, task_config)
                metrics = self.update_metrics(metric_json, metrics, task_config)
                # Write out metrics to file
                self.write_metric_to_file(metrics)
            logger.info("Evaluation completed for local LLM judges.")

        if not remote_queue.is_queue_empty():
            remote_queue.wait_for_all_async_tasks()
            logger.info("All remote LLM judge tasks have completed.")

            logger.info("Running evaluation for remote LLM judges...")
            for task in remote_queue.get_unique_set_of_evaluation_params():
                (dataloader, task_config) = task
                metric_json = self.run_single_task_evaluation(dataloader, task_config)
                metrics = self.update_metrics(metric_json, metrics, task_config)
                # Write out metrics to file
                self.write_metric_to_file(metrics)
            logger.info("Evaluation completed for remote LLM judges.")

            remote_queue.terminate_pool()

        metrics = aggregate_metrics(metrics, config=self.config)
        self.write_metric_to_file(metrics)

        logger.info("Ending evaluation...")


if __name__ == "__main__":
    # python src/seahelm_evaluation.py --tasks seahelm --output_dir results --model_type vllm --model_name google/gemma-2-9b-it --model_args "dtype=bfloat16,enable_prefix_caching=True,gpu_memory_utilization=0.95,tensor_parallel_size=1"
    parser = argparse.ArgumentParser(description="Process configuration file path.")
    parser.add_argument(
        "--tasks",
        action="append",
        help='Evaluation task configuration. Default is "seahelm". Accepted values: seahelm, all_tasks',
        required=True,
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to output model to", required=True
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of model serving [vLLM, OpenAI, LiteLLM, VertexAI]",
        required=True,
    )
    parser.add_argument(
        "--model_name", type=str, help="Path to the model directory", required=True
    )
    parser.add_argument(
        "--model_args",
        type=str,
        help="Model args to pass to the model (e.g vLLM [tensor_parallel_size, pipeline_parallel_size, max_model_len, ...])",
    )
    parser.add_argument(
        "--is_base_model",
        action="store_true",
        help="Include this flag if the model is a base model. The model's chat template (if available) will be not be applied.",
    )
    parser.add_argument(
        "--is_vision_model",
        action="store_true",
        help="Include this flag if the model is a vision model",
    )
    parser.add_argument(
        "--is_reasoning_model",
        action="store_true",
        help="Include this flag if the model is a reasoning model",
    )
    parser.add_argument(
        "--rerun_cached_results",
        action="store_true",
        help="Include this flag if you want to rerun cached results",
    )
    parser.add_argument(
        "--skip_tokenize_prompts",
        action="store_true",
        help="Include this flag to skip the tokenization of prompts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for reproducibility. Default is 1234.",
    )
    parser.add_argument(
        "--skip_tasks",
        type=str,
        default=None,
        help="Comma separated list of tasks that should be skipped. Default is None.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples per task (only use this for testing)",
    )
    parser.add_argument(
        "--no_batching",
        action="store_true",
        help="Include this flag to disable batching for model inference",
    )
    parser.add_argument(
        "--ignore_missing_files",
        action="store_true",
        help="Include this flag to ignore missing files during evaluation",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=0,
        help="A label from {0,...,N-1} for N non-deterministic runs. Please assign a distinct seed for each run.",
    )
    parser.add_argument(
        "--sandbox_type",
        type=str,
        default=None,
        help="Type of sandbox to use for code execution.",
    )

    args = parser.parse_args()
    tasks_configuration = args.tasks
    output_dir = args.output_dir
    model_name = args.model_name
    model_type = args.model_type
    is_base_model = args.is_base_model
    is_vision_model = args.is_vision_model
    is_reasoning_model = args.is_reasoning_model
    limit = args.limit
    seed = args.seed
    run_number = args.run
    no_batching = args.no_batching
    skip_tokenize_prompts = args.skip_tokenize_prompts
    ignore_missing_files = args.ignore_missing_files
    sandbox_type = args.sandbox_type
    skip_tasks = (
        args.skip_tasks.split(",") if args.skip_tasks is not None else args.skip_tasks
    )

    if args.model_args is not None:
        model_args = simple_parse_args_string(args.model_args)
    else:
        model_args = {}

    # Setup logging
    current_datetime = dt.now().isoformat()

    SeaHelmEvaluation.create_seahelm_run_folders(
        output_dir=output_dir, model_name=model_name, run_number=run_number
    )
    log_path = f"{output_dir}/{os.path.basename(model_name)}/run_{run_number}/logs/logfile_{current_datetime}.log"
    setup_root_logger(filepath=log_path)

    # Setup
    llm = get_serving_class(
        model_type=model_type,
        model_name=model_name,
        is_base_model=is_base_model,
        seed=seed,
        **model_args,
    )

    seahelm_eval = SeaHelmEvaluation(
        llm,
        tasks_configuration,
        output_dir,
        model_name,
        is_base_model=is_base_model,
        is_vision_model=is_vision_model,
        is_reasoning_model=is_reasoning_model,
        inference_file_type="jsonl",
        skip_task=skip_tasks,
        limit=limit,
        seed=seed,
        no_batching=no_batching,
        ignore_missing_files=ignore_missing_files,
        tokenize_prompts=not skip_tokenize_prompts,
        run_number=run_number,
        datetime=current_datetime,
        sandbox_type=sandbox_type,
    )
    seahelm_eval.run_evaluation(
        llm=llm, use_cached_results=not args.rerun_cached_results
    )
