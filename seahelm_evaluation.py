import argparse
import glob
import importlib
import json
import logging
import os
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import datasets
import litellm
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.collect_env import get_pretty_env_info

from base_logger import setup_root_logger, get_logger
from constants import (
    BASE_MODELS_SKIP_TASKS,
    FEW_SHOT_STOP_TOKENS,
    INSTRUCT_MODELS_SKIP_TASKS,
)
from seahelm_tasks.aggregate_metrics import aggregate_metrics
from serving import LiteLLMServing, OpenAIServing, VLLMServing
from serving.openai_serving import OPENAI_MODELS
from utils import get_error_count, get_git_commit_hash, simple_parse_args_string

litellm.drop_params = True
logger = get_logger(__name__)


class SeaHelmEvaluation:
    def __init__(
        self,
        llm,
        tasks_configuration: str,
        output_dir: str,
        model_name: str,
        generation_config_file: str = "seahelm_tasks/generation_config.yaml",
        reasoning_generation_config_file: str = "seahelm_tasks/reasoning_generation_config.yaml",
        task_config_file: str = "seahelm_tasks/task_config.yaml",
        task_folder: str = "seahelm_tasks",
        is_base_model: bool = False,
        is_vision_model: bool = False,
        is_reasoning_model: bool = False,
        num_in_context_examples: int = None,
        fewshot_as_multiturn: bool = False,
        inference_file_type: str = "csv",
        tokenize_prompts: bool = True,
        skip_task: list = None,
        limit: int = None,
        no_batching: bool = True,
    ):
        self.output_dir = output_dir
        self.setup_seahelm_logging(output_dir, model_name)

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

        config, self.task_list_by_lang, task_alias = self.load_config_from_folders(
            task_folder
        )
        generation_config = OmegaConf.load(generation_config_file)
        task_config = OmegaConf.load(task_config_file)

        # load tasks to run from configuration file
        self.tasks = task_config.get(tasks_configuration, None)
        assert (
            self.tasks is not None
        ), f"Unable to find tasks_configuration in task_config.yaml. Received {self.tasks_configuration}."
        # convert self.tasks back to dictionary
        self.tasks = OmegaConf.to_object(self.tasks)

        self.datetime = datetime.now().isoformat()

        # convert "all" case to list of task
        for lang, tasks in self.tasks.items():
            if isinstance(self.tasks[lang], str):
                assert (
                    tasks == "all"
                ), f"The only string allowed for definition of tasks is 'all', {self.tasks[lang]} was defined instead for {lang}. Please use a list of tasks if you only want to run a subset of tasks."

                self.tasks[lang] = self.task_list_by_lang[lang]
            else:
                for alias, value in task_alias.items():
                    if alias in tasks:
                        self.tasks[lang].remove(alias)
                        self.tasks[lang].extend(value)

        self.config = OmegaConf.merge(config, generation_config)
        if self.is_reasoning_model:
            reasoning_config = OmegaConf.load(reasoning_generation_config_file)
            self.config = OmegaConf.merge(self.config, reasoning_config)

        # Automatically set num_in_context_examples if None
        if num_in_context_examples == None:
            # 0 for zero-shot testing of instruction-tuned models
            # 5 for few-shot testing of base models
            if self.is_base_model:
                from constants import BASE_NUM_IN_CONTEXT_EXAMPLES

                self.num_in_context_examples = BASE_NUM_IN_CONTEXT_EXAMPLES
            else:
                from constants import INSTRUCT_NUM_IN_CONTEXT_EXAMPLES

                self.num_in_context_examples = INSTRUCT_NUM_IN_CONTEXT_EXAMPLES
        else:
            self.num_in_context_examples = num_in_context_examples

        self.fewshot_as_multiturn = fewshot_as_multiturn
        self.inference_file_type = inference_file_type

        self.tokenize_prompts = tokenize_prompts
        self.limit = limit
        self.no_batching = no_batching

        _default_skip_task = {
            "base_models": BASE_MODELS_SKIP_TASKS,
            "instruct_models": INSTRUCT_MODELS_SKIP_TASKS,
        }
        if skip_task == None:
            self.skip_task = _default_skip_task[
                "base_models" if is_base_model else "instruct_models"
            ]
        else:
            self.skip_task = skip_task

        if OPENAI_MODELS == set():
            logger.warning("No valid OpenAI models found. Skipping task: mt-bench")
            self.skip_task.extend(["mt-bench"])

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
            "is_base_model",
            "is_vision_model",
            "num_in_context_examples",
            "fewshot_as_multiturn",
            "tokenize_prompts",
            "skip_task",
            "limit",
            "no_batching",
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

        # save config folder
        self.save_config(output_dir, model_name)

    def load_config_from_folders(self, folder):
        config_files = glob.glob(f"{folder}/**/config.yaml", recursive=True)

        output_config = OmegaConf.create({})
        task_list_by_lang = {}
        task_alias = {}
        for config_file in config_files:
            config = OmegaConf.load(config_file)
            for task_name in config:
                # update main config file
                OmegaConf.update(
                    output_config,
                    f"tasks.{task_name}",
                    config.get(task_name),
                    merge=True,
                )

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

        return output_config, task_list_by_lang, task_alias

    def setup_seahelm_logging(self, output_dir, model_name):
        os.makedirs(f"{output_dir}/{os.path.basename(model_name)}", exist_ok=True)

        folder = f"{output_dir}/{os.path.basename(model_name)}/inference"
        logger.info(
            "---------- Preparation of output folder ----------\nPreparing output folder ...\nFolder: %s",
            folder,
        )
        os.makedirs(
            f"{output_dir}/{os.path.basename(model_name)}/inference", exist_ok=True
        )
        logger.info("Completed preparation of output folder!\n")

    def save_config(self, output_dir, model_name):
        config_filepath = f"{output_dir}/{os.path.basename(model_name)}/{os.path.basename(model_name)}_run_config_{self.datetime}.yaml"
        logger.info(
            """---------- Configuration saving ----------
Saving run config to output folder...
Filepath: %s""",
            config_filepath,
        )

        with open(config_filepath, "w") as fp:
            OmegaConf.save(config=self.config, f=fp)
        logger.info("Config file saved!\n")

    def _check_if_task_should_run(self, task: str, lang: str) -> bool:
        if task in self.skip_task:
            logger.info(
                f"Task in skip task list: %s. Skipping task '%s' for lang '%s'.",
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

    def get_generation_kwargs(
        self, task_config: dict, specific_task_config: dict
    ) -> dict:
        generation_kwargs = {}
        if "temperature" in task_config:
            generation_kwargs["temperature"] = task_config["temperature"]

        if "max_tokens" in specific_task_config:
            generation_kwargs["max_tokens"] = specific_task_config["max_tokens"]

        for key, value in self.config["generation_kwargs"].items():
            generation_kwargs[key] = value

        if self.is_base_model:
            generation_kwargs["stop"] = FEW_SHOT_STOP_TOKENS

        return generation_kwargs

    def generate_formatted_conversation(
        self,
        specific_task_config,
        values,
        num_examples,
        fewshot_as_multiturn: bool = False,
    ):
        roles = []
        contents = []
        task_prompt_template = specific_task_config["prompt_template"]["template"]

        fewshot_examples = ""

        if num_examples > 0:
            if "example_filepath" not in specific_task_config:
                logger.warning(
                    "Example filepath not found! Reverting back to 0-shot instead of %d-shot.",
                    num_examples,
                )
            else:
                in_context_examples = pd.read_json(
                    specific_task_config["example_filepath"], lines=True
                ).loc[: num_examples - 1]
                if len(in_context_examples) < num_examples:
                    logger.warning(
                        "Not enough examples! Expected %d examples but only received %d.",
                        num_examples,
                        len(in_context_examples),
                    )

                if fewshot_as_multiturn:
                    task_prompt_label = specific_task_config["prompt_template"][
                        "fewshot_label"
                    ]
                    for _, row in in_context_examples.iterrows():
                        roles.append("user")
                        contents.append(
                            task_prompt_template.format(
                                fewshot_examples="", **row["prompts"][0]
                            )
                        )
                        roles.append("assistant")
                        contents.append(task_prompt_label.format(**row))
                else:
                    examples_prompt_template = specific_task_config["prompt_template"][
                        "fewshot_example"
                    ]
                    fewshot_examples = "".join(
                        [
                            examples_prompt_template.format(
                                label=row["label"], **row["prompts"][0]
                            )
                            for _, row in in_context_examples.iterrows()
                        ]
                    )

        roles.append("user")
        contents.append(
            task_prompt_template.format(fewshot_examples=fewshot_examples, **values)
        )

        return roles, contents

    def update_conversation(self, conversations, role, content):
        if self.is_vision_model:
            content = [{"type": "text", "text": content}]

        conversations.append({"role": role, "content": content})
        return conversations

    def get_prompt_formatter(
        self,
        specific_task_config: dict,
        turn: int,
        num_examples: int,
        fewshot_as_multiturn: bool = False,
    ):
        def _prompt_formatter(row):
            if turn == 1:
                conversations = []
            else:
                conversations = row["conversations"]
                conversations = self.update_conversation(
                    conversations, "assistant", row["responses"][turn - 2]
                )

            values = row["prompts"][turn - 1]

            roles, contents = self.generate_formatted_conversation(
                specific_task_config,
                values,
                num_examples,
                fewshot_as_multiturn=fewshot_as_multiturn,
            )
            for role, content in zip(roles, contents):
                conversations = self.update_conversation(conversations, role, content)

            row["conversations"] = conversations
            return row

        return _prompt_formatter

    def get_update_function(self, column, data):
        def update_function(row, i):
            if column in row:
                row[column].append(data[i])
            else:
                row[column] = [data[i]]
            return row

        return update_function

    def update_reasoning_generation_kwargs(self, generation_kwargs):
        generation_kwargs["max_tokens"] += self.config["reasoning_generation_kwargs"][
            "max_think_tokens"
        ]
        logger.info(
            "Model is a reasoning model. Increasing the max_tokens to '%d'.",
            generation_kwargs["max_tokens"],
        )
        generation_kwargs["temperature"] = self.config["reasoning_generation_kwargs"][
            "temperature"
        ]
        logger.info(
            "Model is a reasoning model. Setting the temperature to '%f'.",
            generation_kwargs["temperature"],
        )
        return generation_kwargs

    def run_single_task_inference(
        self,
        llm,
        task_config: str,
        task_name: str,
        lang: str,
        limit: int = None,
        use_cached_results: bool = False,
    ):
        inference_times = []
        is_cached = False
        if llm is None:
            assert (
                use_cached_results == True
            ), "use_cached_results must be set to True if model_type is None."

        logger.info(
            "---------- Inference | Lang: %s | Task: %s ----------\nTesting Competency: %s",
            lang.upper(),
            task_name.upper(),
            task_config["competency"].upper(),
        )
        if use_cached_results:
            inference_df = self.read_inference_results(
                task_name,
                lang,
                file_type=self.inference_file_type,
            )
            if inference_df is not None:
                is_cached = True
                logger.info(
                    "Using cached results for Task: %s | Lang: %s\n",
                    task_name.upper(),
                    lang.upper(),
                )
                return inference_df, inference_times, is_cached
            elif llm is None:
                assert (
                    inference_df is not None
                ), f"Unable to load cached results for task {task_name} and lang {lang}. (When model_type is None)"

        try:
            specific_task_config = task_config["languages"][lang]
            filepath = specific_task_config["filepath"]

            logger.info(f"Drawing and preparing instances from %s", filepath)

            dataset = datasets.load_dataset("json", split="train", data_files=filepath)
            if limit is not None:
                dataset = dataset.select(range(limit))

            logger.info(
                f"Performing inference for task '%s' with %d examples",
                task_name.upper(),
                self.num_in_context_examples,
            )

            # assume that all rows have the same number of turns
            n_turns = len(dataset[0]["prompts"])
            for turn in range(1, n_turns + 1):
                dataset = dataset.map(
                    self.get_prompt_formatter(
                        specific_task_config,
                        turn,
                        self.num_in_context_examples,
                        self.fewshot_as_multiturn,
                    ),
                    num_proc=16,
                )

                # Set up generation kwargs
                generation_kwargs = self.get_generation_kwargs(
                    task_config, specific_task_config
                )

                # Update generation kwargs for reasoning models
                # Follows the kwargs for DeepSeek models
                if self.is_reasoning_model:
                    generation_kwargs = self.update_reasoning_generation_kwargs(
                        generation_kwargs
                    )

                if isinstance(llm, OpenAIServing):
                    generated_outputs = llm.generate_openai_batched_responses(
                        dataset,
                        generation_kwargs,
                        output_dir,
                        model_name,
                        task_name,
                        lang,
                    )
                else:
                    generated_outputs, inference_time_taken = llm.generate_responses(
                        dataset, generation_kwargs, no_batching=self.no_batching
                    )
                    inference_times.append(inference_time_taken)

                responses, errors, tokenized_prompts = llm.parse_outputs(
                    generated_outputs,
                    conversations=dataset["conversations"],
                    tokenize_prompts=self.tokenize_prompts,
                )

                if self.is_reasoning_model:
                    dataset = dataset.map(
                        self.get_update_function("responses_with_thinking", responses),
                        with_indices=True,
                    )

                    # remove think portion
                    clean_responses = []
                    for response in responses:
                        # skip None responses
                        if response is None:
                            clean_responses.append(response)
                        else:
                            response = response.split(
                                self.config["reasoning_generation_kwargs"][
                                    "end_think_token"
                                ]
                            )[-1]
                            clean_responses.append(response.strip())

                    dataset = dataset.map(
                        self.get_update_function("responses", clean_responses),
                        with_indices=True,
                    )
                else:
                    dataset = dataset.map(
                        self.get_update_function("responses", responses),
                        with_indices=True,
                    )

                dataset = dataset.map(
                    self.get_update_function("errors", errors), with_indices=True
                )

                if self.tokenize_prompts:
                    dataset = dataset.map(
                        self.get_update_function(
                            "tokenized_prompts", tokenized_prompts
                        ),
                        with_indices=True,
                    )

            inference_df = dataset.to_pandas()
            self.write_out_inference_results(inference_df, task_name, lang)

            logger.info("Inference for task '%s' completed!\n", task_name.upper())
            return inference_df, inference_times, is_cached

        except Exception as e:
            logger.error(
                "Failed to run inference for task %s and lang %s", task_name, lang
            )
            logger.exception(e)
            return None, None, is_cached

    def get_inference_filepath(
        self, task_name: str, lang: str, file_type: str = "jsonl"
    ):
        return os.path.join(
            self.output_dir,
            os.path.basename(self.model_name),
            "inference",
            f"{os.path.basename(self.model_name)}_{task_name}_{lang}.{file_type}",
        )

    def write_out_inference_results(
        self, inference_df: pd.DataFrame, task_name: str, lang: str
    ):
        output_filepath = self.get_inference_filepath(
            task_name=task_name, lang=lang, file_type=self.inference_file_type
        )

        logger.info(
            "Saving inference results for task '%s' to %s",
            task_name.upper(),
            output_filepath,
        )
        file_type = output_filepath.split(".")[-1]
        assert file_type in [
            "csv",
            "jsonl",
        ], "File type must be either 'csv' or 'jsonl'"

        # save outputs
        if file_type == "csv":
            inference_df.to_csv(output_filepath, index=False)
        elif file_type == "jsonl":
            inference_df.to_json(
                output_filepath, orient="records", force_ascii=False, lines=True
            )

        logger.info("Inference results saved!")

    def read_inference_results(
        self,
        task_name: str,
        lang: str,
        file_type: str = "jsonl",
    ):
        assert file_type in [
            "csv",
            "jsonl",
        ], "File type must be either 'csv' or 'jsonl'"

        # save outputs
        input_filepath = self.get_inference_filepath(
            task_name, lang, file_type=file_type
        )

        if os.path.exists(input_filepath) is False:
            logger.debug(
                f"No cached inference results found for %s and %s at %s.",
                task_name,
                lang,
                input_filepath,
            )
            return None

        if file_type == "csv":
            inference_df = pd.read_csv(input_filepath)
        elif file_type == "jsonl":
            inference_df = pd.read_json(input_filepath, lines=True)
        return inference_df

    def get_metric_class(self, task_name):
        metric_file = self.config["tasks"][task_name]["metric_file"]
        metric_path = metric_file.strip(".py").replace("/", ".")
        metric_class = self.config["tasks"][task_name]["metric_class"]
        Metric = getattr(importlib.import_module(metric_path), metric_class)

        return Metric

    def run_single_task_evaluation(
        self,
        inference_df: pd.DataFrame,
        metrics: dict,
        task_config: str,
        task_name: str,
        lang: str,
        inference_time_taken: list[int] = None,
        is_cached: bool = False,
    ):
        try:
            logger.info(
                "--------- Evaluation | Lang: %s | Task: %s ----------",
                lang.upper(),
                task_name.upper(),
            )
            Metric = self.get_metric_class(task_name=task_name)

            logger.info("Evaluating '%s' using %s", task_name.upper(), Metric.__name__)
            evaluation_metric = Metric(
                inference_df=inference_df,
                task_config=task_config,
                task=task_name,
                lang=lang,
            )

            metric_json, inference_df = evaluation_metric.evaluate_responses()
            # save scores in inference_df
            self.write_out_inference_results(inference_df, task_name, lang)

            metric_json[task_name]["errors"] = get_error_count(inference_df["errors"])
            metric_json[task_name]["inference_time_taken"] = inference_time_taken
            metric_json[task_name]["is_cached"] = is_cached

            logger.info("Evaluation for task '%s' completed!\n", task_name.upper())
        except Exception as e:
            logger.error(
                "Failed to run evaluation for task %s and lang %s", task_name, lang
            )
            logger.exception(e)
            logger.warning(
                "Setting metric %s to 0 for task %s", task_config["metric"], task_name
            )
            metric_json = {
                task_name: {
                    task_config["metric"]: 0,
                    "error": "Failed to run evaluation for task",
                },
            }

        competency = task_config["competency"]
        if lang not in metrics:
            metrics.update({lang: {competency: {}}})
        elif competency not in metrics[lang]:
            metrics[lang].update({competency: {}})
        metrics[lang][competency].update(metric_json)

        return metrics

    def write_metric_to_file(self, metrics, json_filepath):
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def run_evaluation(self, llm, use_cached_results=True):
        metrics = {}
        json_filepath = os.path.join(
            self.output_dir,
            os.path.basename(self.model_name),
            f"{os.path.basename(self.model_name)}_seahelm_results_{self.datetime}.json",
        )

        mt_bench_inferences = []
        tasks_to_run_first = ["mt-bench"]
        task_to_run_first = tasks_to_run_first[0]

        tasks_to_run_evaluation_last = [
            "translation-en-xx",
            "translation-xx-en",
            "translation-id-xx",
            "translation-xx-id",
        ]
        # iterate through to run mt-bench inferences first
        for lang, tasks_by_lang in self.tasks.items():
            if task_to_run_first in tasks_by_lang:
                task_name = task_to_run_first
                if not self._check_if_task_should_run(task_name, lang):
                    continue

                task_config = self.config["tasks"][task_name]
                # Append additional info needed for MT Bench to task_config
                task_config["model_name"] = self.model_name
                task_config["output_dir"] = self.output_dir
                task_config["use_cached_results"] = use_cached_results

                inference_df, inference_time_taken, is_cached = (
                    self.run_single_task_inference(
                        llm,
                        task_config,
                        task_name,
                        lang,
                        limit=self.limit,
                        use_cached_results=use_cached_results,
                    )
                )
                mt_bench_inferences.append(
                    (
                        inference_df,
                        task_config,
                        task_name,
                        lang,
                        inference_time_taken,
                        is_cached,
                    )
                )

        if mt_bench_inferences:
            from seahelm_tasks.multi_turn.mt_bench.mt_bench import (
                evaluate_mt_bench_task,
            )

            logger.info(
                "Starting %s evaluation using multiprocessing", task_to_run_first
            )
            _evaluate_mt_bench_task = partial(
                evaluate_mt_bench_task,
                Metric=self.get_metric_class(task_name=task_name),
            )
            # Create a multiprocessing Pool
            pool = Pool(processes=8)
            # Start the mt-bench evaluation async
            mt_bench_results_async = pool.starmap_async(
                _evaluate_mt_bench_task, mt_bench_inferences
            )
        else:
            pool = None
            mt_bench_results_async = None

        translation_tasks = []
        # proceed with the rest of tasks (non-mt-bench)
        for lang, tasks_by_lang in self.tasks.items():
            for task_name in tasks_by_lang:
                if task_name in tasks_to_run_first:
                    continue

                # Skip task if it shouldn't be run
                if not self._check_if_task_should_run(task_name, lang):
                    continue

                task_config = self.config["tasks"][task_name]
                inference_df, inference_time_taken, is_cached = (
                    self.run_single_task_inference(
                        llm,
                        task_config,
                        task_name,
                        lang,
                        limit=self.limit,
                        use_cached_results=use_cached_results,
                    )
                )

                if task_name in tasks_to_run_evaluation_last:
                    translation_tasks.append(
                        (
                            inference_df,
                            task_config,
                            task_name,
                            lang,
                            inference_time_taken,
                            is_cached,
                        )
                    )
                else:
                    metrics = self.run_single_task_evaluation(
                        inference_df,
                        metrics,
                        task_config,
                        task_name,
                        lang,
                        inference_time_taken=inference_time_taken,
                        is_cached=is_cached,
                    )
                    # Write out metrics to file
                    self.write_metric_to_file(metrics, json_filepath)

        if translation_tasks:
            # delete vllm instance to free up memory for the MetricX model
            if isinstance(llm, VLLMServing):
                from utils import delete_vllm_model_and_free_memory

                delete_vllm_model_and_free_memory(llm)

            for _tasks in translation_tasks:
                (
                    inference_df,
                    task_config,
                    task_name,
                    lang,
                    inference_time_taken,
                    is_cached,
                ) = _tasks

                metrics = self.run_single_task_evaluation(
                    inference_df,
                    metrics,
                    task_config,
                    task_name,
                    lang,
                    inference_time_taken=inference_time_taken,
                    is_cached=is_cached,
                )
                # Write out metrics to file
                self.write_metric_to_file(metrics, json_filepath)

        # wait for mt-bench evaluation to complete
        if mt_bench_results_async:
            logger.info("Waiting for mt-bench evaluation to complete")
            mt_bench_results = mt_bench_results_async.get()
            pool.close()
            pool.join()

            # Collect metrics from all processes
            for (
                metric_json,
                inference_df,
                competency,
                task_name,
                lang,
            ) in mt_bench_results:

                # save scores in inference_df
                self.write_out_inference_results(inference_df, task_name, lang)
                if lang not in metrics:
                    metrics[lang] = {competency: {}}
                elif competency not in metrics[lang]:
                    metrics[lang][competency] = {}
                metrics[lang][competency].update(metric_json)

            # Write out metrics to file
            self.write_metric_to_file(metrics, json_filepath)

        metrics = aggregate_metrics(metrics, config=self.config)
        self.write_metric_to_file(metrics, json_filepath)

        logger.info("Ending evaluation...")


if __name__ == "__main__":
    # python seahelm_evaluation.py --tasks seahelm --output_dir results --model_type vllm --model_name google/gemma-2-9b-it --model_args "dtype=bfloat16,enable_prefix_caching=True,gpu_memory_utilization=0.95,tensor_parallel_size=1"
    parser = argparse.ArgumentParser(description="Process configuration file path.")
    parser.add_argument(
        "--tasks",
        type=str,
        help='Evaluation task configuration. Default is "seahelm". Accepted values: seahelm, all_tasks',
        required=True,
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to output model to", required=True
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of model serving [vLLM, OpenAI, LiteLLM]",
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
        "--fewshot_as_multiturn",
        action="store_true",
        help="Include this flag if you want to run fewshot testing using multiturn conversation",
    )
    parser.add_argument(
        "--num_in_context_examples",
        type=int,
        default=None,
        help="Number of examples to use for fewshot testing. Max of 5 examples. Set to None to use default number of examples.",
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

    args = parser.parse_args()
    tasks_configuration = args.tasks
    output_dir = args.output_dir
    model_name = args.model_name
    model_type = args.model_type
    is_base_model = args.is_base_model
    is_vision_model = args.is_vision_model
    is_reasoning_model = args.is_reasoning_model
    num_in_context_examples = args.num_in_context_examples
    fewshot_as_multiturn = args.fewshot_as_multiturn
    limit = args.limit
    no_batching = args.no_batching
    skip_tokenize_prompts = args.skip_tokenize_prompts
    skip_tasks = (
        args.skip_tasks.split(",") if args.skip_tasks is not None else args.skip_tasks
    )

    if args.model_args is not None:
        model_args = simple_parse_args_string(args.model_args)
    else:
        model_args = {}

    # Setup logging
    os.makedirs(f"{output_dir}/{os.path.basename(model_name)}", exist_ok=True)
    log_path = f"{output_dir}/{os.path.basename(model_name)}/logfile.log"
    setup_root_logger(filepath=log_path)

    # Setup
    assert model_type.lower() in [
        "litellm",
        "vllm",
        "openai",
        "none",
    ], f"""model_type should be one of ["litellm", "vllm", "openai"]. Received {model_type} instead."""
    if model_type.lower() == "litellm":
        # typical model args: "api_provider=openai,base_url=http://localhost:8000/v1,api_key=token-abc123"
        logging.info(
            f"Loading model {model_name} using {model_args['api_provider'].upper()}..."
        )
        llm = LiteLLMServing(
            model=f"{model_args['api_provider']}/{model_name}",
            **model_args,
        )
    elif model_type.lower() == "openai":
        llm = OpenAIServing(
            model=model_name,
            is_base_model=is_base_model,
        )
    elif model_type.lower() == "vllm":
        # typical model args: "dtype=bfloat16,enable_prefix_caching=True,gpu_memory_utilization=0.95,tensor_parallel_size=1"
        logging.info(f"Loading model {model_name} using vLLMs...")
        llm = VLLMServing(
            model=model_name,
            is_base_model=is_base_model,
            seed=1234,
            tokenizer_mode="mistral" if model_name.startswith("mistralai") else "auto",
            load_format="mistral" if model_name.startswith("mistralai") else "auto",
            config_format="mistral" if model_name.startswith("mistralai") else "auto",
            **model_args,
        )
    elif model_type.lower() == "none":
        logging.info(
            f"Model type is set to None. Please ensure that the model inferences are in the correct folder and format."
        )
        llm = None

    seahelm_eval = SeaHelmEvaluation(
        llm,
        tasks_configuration,
        output_dir,
        model_name,
        is_base_model=is_base_model,
        is_vision_model=is_vision_model,
        is_reasoning_model=is_reasoning_model,
        num_in_context_examples=num_in_context_examples,
        fewshot_as_multiturn=fewshot_as_multiturn,
        inference_file_type="jsonl",
        skip_task=skip_tasks,
        limit=limit,
        no_batching=no_batching,
        tokenize_prompts=not skip_tokenize_prompts,
    )
    seahelm_eval.run_evaluation(
        llm=llm, use_cached_results=not args.rerun_cached_results
    )
