import json
import os
import re
from multiprocessing import Pool

import dateutil.parser as dparser
import pandas as pd
import yaml
from elo_utils import COLORS, EloPrintWrapper

# Ideally, found in the config file for each model. This is a temporary fix.
NUM_TURNS_FOR_MULTI_TURN_TASKS: dict[str, int] = {"mt-bench": 2}


def load_inference_file(
    inference_file_path: str,
    results_directory_path: str,
    model_run_path: str,
    model_name: str,
    r: int,
    lang: str,
    competency: str,
    task: str,
    metric: str,
) -> tuple[
    pd.DataFrame | None, dict[str, object], dict[str, object], list[dict[str, object]]
]:
    """Load a single inference file and extract per-sample scores.

    This function reads a JSONL inference file produced by SEA-HELM, extracts
    the per-sample metric values, and returns a DataFrame alongside a compact
    overview and structured logs.

    Args:
        inference_file_path (str): Full path to the inference JSONL file.
        results_directory_path (str): Root directory containing model results.
        model_run_path (str): Relative model run path (e.g., "model_x/run_0").
        model_name (str): The model identifier (e.g., "org/model").
        r (int): Run index for the SEA-HELM evaluation.
        lang (str): Language code of the task.
        competency (str): Competency category for the task.
        task (str): Task name.
        metric (str): Metric key to extract from each record's
            "individual_scores" field.

    Returns:
        tuple: A tuple of four elements:
            - DataFrame | None: DataFrame with a single column "individual_scores";
              None if the file cannot be read.
            - dict: Overview metadata for this (model, task, language) slice.
            - dict: Failure metadata when loading fails; empty on success.
            - list[dict]: List of structured log messages for user feedback.
    """
    logs = []
    try:
        individual_scores = []
        with open(inference_file_path) as file:
            for line in file:
                json_line = json.loads(line)

                # HACK skip if-eval subcategory for num_words as it is not used in the evaluation
                if task == "if-eval":
                    if lang in ["th", "my"]:
                        if (
                            json_line["metadata"]["subcategory"]
                            == "length_constraints:number_words"
                        ):
                            logs.append(
                                {
                                    "message": f"Warning: Skipping if-eval subcategory for num_words in {inference_file_path}",
                                    "color": COLORS.WARNING,
                                }
                            )
                            continue
                individual_scores.append(json_line["individual_scores"][metric])
        this_df = pd.DataFrame({"individual_scores": individual_scores})
        this_df = this_df.dropna(
            subset=["individual_scores"]
        )  # TODO review this again as it could lead to potential silent misalignment

        num_samples = len(this_df)
        results_overview = {
            "model": model_name,
            "task": task,
            "competency": competency,
            "languages": lang,
            "metric": metric,
            "num_samples": num_samples,
            "file_path": inference_file_path,
            "num_turns": NUM_TURNS_FOR_MULTI_TURN_TASKS.get(task, 1),
        }
        return this_df, results_overview, {}, logs
    except Exception as e:
        directory_path = os.path.join(
            results_directory_path,
            model_run_path,
            "inference",
        )
        logs.append(
            {
                "message": f"Warning: Couldn't read file for {model_name}, run {r}, {task}, {lang}. Check directory {directory_path}. Error: {type(e).__name__}:{e}",
                "color": COLORS.WARNING,
            }
        )
        failed_loads = {
            "model_name": model_name,
            "task": task,
            "language": lang,
            "run": r,
            "directory_path": directory_path,
            "error": f"{type(e).__name__}:{e}",
            "success": "",
        }
        return None, {}, failed_loads, logs


class EloResultsLoader:
    """Loader for collating SEA-HELM inference outputs for ELO evaluation.

    This class scans a results directory, locates the latest SEA-HELM config per
    model, and loads per-task, per-language inference files into nested
    DataFrames suitable for ELO pipelines.

    Args:
        results_directory_path (str): Root directory containing model runs.
        output_directory_path (str): Directory to write summaries and logs to.
        ontological_category (str): Category being ranked (e.g., "task",
            "language", or "competency").
        ontological_label (str): Specific label within the category (e.g., a
            task name or language code).
        metric_override_dict (dict, optional): Optional overrides mapping task
            -> metric name. Defaults to {}.
        ignore_task (set|dict, optional): Tasks to skip. Membership test is
            used; both a set or a dict of tasks as keys are supported. Defaults
            to {}.
        ignore_language (set, optional): Languages to skip. Defaults to set().
        ignore_task_lang (dict|set, optional): Mapping task -> set/list of
            languages to skip for that task. Defaults to set().
        ignore_model (set, optional): Model names to skip. Defaults to set().
        num_seahelm_runs (int, optional): Number of SEA-HELM runs per model to
            consider. Defaults to 1.
        tasks_config_filepath (str | None, optional): If provided, use this
            configuration file instead of auto-detecting the latest. Defaults to
            None.
        num_processes (int, optional): Number of processes for parallel file
            loading. Defaults to 32.
        elo_printer (EloPrintWrapper | None, optional): Printer for colored
            output. Defaults to None.
    """

    def __init__(
        self,
        results_directory_path: str,
        output_directory_path: str,
        ontological_category: str,
        ontological_label: str,
        metric_override_dict: dict[str, str] = None,
        ignore_task: dict[str, object] | set[str] = None,
        ignore_language: set[str] = None,
        ignore_task_lang: dict[str, set[str] | list[str]] | set[str] = None,
        ignore_model: set[str] = None,
        num_seahelm_runs: int = 1,
        tasks_config_filepath: str | None = None,
        num_processes: int = 32,
        elo_printer: EloPrintWrapper | None = None,
    ) -> None:
        self.results_directory_path = results_directory_path
        self.output_directory_path = output_directory_path
        self.metric_override_dict = (
            metric_override_dict if metric_override_dict is not None else set()
        )
        self.num_seahelm_runs = (
            num_seahelm_runs  # Number of parallel runs for each seahelm task
        )

        self.ignore_task = ignore_task if ignore_task is not None else set()
        self.ignore_language = ignore_language if ignore_language is not None else set()
        self.ignore_task_lang = (
            ignore_task_lang if ignore_task_lang is not None else set()
        )
        self.ignore_model = ignore_model if ignore_model is not None else set()
        self.ontological_category = ontological_category
        self.ontological_label = ontological_label
        self.tasks_config_filepath = tasks_config_filepath

        self.num_processes = num_processes
        self.elo_printer = elo_printer if elo_printer is not None else EloPrintWrapper()

    def load_results(
        self,
    ) -> tuple[
        dict[str, dict[str, dict[str, dict[str, pd.DataFrame]]]],
        pd.DataFrame,
        list[str],
    ]:
        """Load inference results into nested DataFrames keyed by model/run/task/lang.

        When ``tasks_config_filepath`` is provided, the configuration for tasks,
        competencies, languages, and metrics is read from that file. Otherwise,
        the latest config under each model's ``run_0/configs`` directory is used.

        Returns:
            tuple: A 3-tuple consisting of:
                - dict: Nested mapping
                  ``{model_name: {run_label: {task: {lang: DataFrame}}}}``.
                - DataFrame: A results overview with per-file metadata
                  (model/task/competency/language/metric/num_samples/file_path/num_turns).
                - list[str]: List of model names encountered.
        """
        if self.tasks_config_filepath:
            self.elo_printer.print(
                "Loading tasks config file from",
                self.tasks_config_filepath,
                color=COLORS.OKBLUE,
            )
            with open(self.tasks_config_filepath) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

        # Else, prepare to load the latest config file for the each model in the results directory
        def _get_last_config_file(model_path: str) -> str | None:
            """
            Extracts the latest config from the run_0 results directory of a model.
            Run 0 is selected because logprobs tasks are run only once regardless of the number of runs.

            Returns:
            last_config_file (str): The name of the latest config file found in the configs directory.
            """
            last_date = None
            last_config_file = None
            for f in os.listdir(
                os.path.join(
                    self.results_directory_path, model_path, "run_0", "configs"
                )
            ):  # Assuming run_0 always exists
                if f.endswith(".yaml"):
                    this_date_string = re.findall(r"(?<=run_config_)(.*).yaml", f)
                    if len(this_date_string) == 0:
                        continue
                    this_date = dparser.parse(this_date_string[0])
                    if last_date is None or this_date > last_date:
                        last_date = this_date
                        last_config_file = f

            if last_config_file is None:
                return None

            return os.path.join(
                self.results_directory_path,
                model_path,
                "run_0",
                "configs",
                last_config_file,
            )

        self.elo_printer.print(
            "Loading each model name and its latest run's config file, to extract task, competency, languages, and metric, and counting the number of samples in each inference file......",
            color=COLORS.OKBLUE,
        )

        def _subdirs(path: str) -> list[str]:
            """Yield directory names not starting with '.' under given path."""
            dirs: list[str] = []
            for entry in os.scandir(path):
                if not entry.name.startswith(".") and entry.is_dir():
                    dirs.append(entry.name)
            return dirs

        models: list[str] = []
        results_dfs: dict[str, dict[str, dict[str, dict[str, pd.DataFrame]]]] = {}
        results_overview: list[dict[str, object]] = []
        failed_loads: list[dict[str, object]] = []

        pool = Pool(self.num_processes)
        inference_file_paths: list[str] = []
        model_run_paths: list[str] = []
        model_names: list[str] = []
        runs: list[int] = []
        langs: list[str] = []
        competencies: list[str] = []
        tasks: list[str] = []
        metrics: list[str] = []

        for d in _subdirs(self.results_directory_path):
            for f in _subdirs(os.path.join(self.results_directory_path, d)):
                model_name = os.path.join(d, f)
                models.append(model_name)
                results_dfs[model_name] = {}

                # If not already provided, load the model's latest config file (from run 0) as logprobs tasks are run only once, regardless of the number of runs specified.
                config_file_name = (
                    self.tasks_config_filepath
                    if self.tasks_config_filepath
                    else _get_last_config_file(model_name)
                )
                if config_file_name is None:
                    self.elo_printer.print(
                        f"WARNING: Skipping loading results for {model_name}. No config file found.",
                        color=COLORS.WARNING,
                    )
                    continue
                # load latest run config file into memory
                with open(
                    config_file_name,
                    "r",
                ) as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)

                assert config is not None, "Config file cannot be None."

                for r in range(self.num_seahelm_runs):
                    results_dfs[model_name][f"run_{r}"] = {}

                    model_run_path = os.path.join(model_name, f"run_{r}")
                    if model_name in self.ignore_model:
                        self.elo_printer.print(
                            f"Skipping {model_name} as it is in ignore_model list.",
                            color=COLORS.WARNING,
                        )
                        continue

                    def _valid_tasks(config=config) -> list[str]:
                        """Optimises loading by skipping tasks that are not required"""
                        tasks: list[str] = []
                        for task in config["tasks"]:
                            if task in self.ignore_task:
                                continue
                            elif (
                                self.ontological_category == "task"
                                and self.ontological_label != task
                            ):
                                continue
                            elif (
                                self.ontological_category == "competency"
                                and config["tasks"][task]["competency"]
                                != self.ontological_label
                            ):
                                continue
                            tasks.append(task)
                        return tasks

                    def _valid_languages(task: str, config=config) -> list[str]:
                        """Optimises loading by skipping languages that are not required"""
                        languages: list[str] = []
                        for lang in config["tasks"][task]["languages"]:
                            if (
                                lang in self.ignore_task_lang.get(task, [])
                                or lang in self.ignore_language
                            ):
                                continue
                            elif (
                                self.ontological_category == "language"
                                and self.ontological_label != lang
                            ):
                                continue
                            languages.append(lang)
                        return languages

                    def _get_metric(task: str, config=config) -> str:
                        if "logprobs" in task:  # TODO: remove this temporary fix
                            return "probabilities_accuracy"
                        elif self.metric_override_dict.get(task, None) is not None:
                            return self.metric_override_dict[task]
                        else:
                            return config["tasks"][task]["metric"]

                    # load results for each task and language
                    for task in _valid_tasks():
                        if "logprobs" in task and r > 0:
                            # logprobs tasks are run only once, regardless of the number of runs specified.
                            continue
                        results_dfs[model_name][f"run_{r}"][task] = {}
                        for lang in _valid_languages(task):
                            inference_file_path = os.path.join(
                                self.results_directory_path,
                                model_run_path,
                                "inferences",
                                lang,
                                config["tasks"][task]["aggregation_group"]
                                if "aggregation_group" in config["tasks"][task]
                                else task,
                                f"{model_name.split('/')[1]}_{task}_{lang}.jsonl",
                            )
                            inference_file_paths.append(inference_file_path)
                            model_run_paths.append(model_run_path)
                            model_names.append(model_name)
                            runs.append(r)
                            langs.append(lang)
                            competencies.append(config["tasks"][task]["competency"])
                            tasks.append(task)
                            metrics.append(_get_metric(task))

        results = pool.starmap(
            load_inference_file,
            zip(
                inference_file_paths,
                [self.results_directory_path] * len(inference_file_paths),
                model_run_paths,
                model_names,
                runs,
                langs,
                competencies,
                tasks,
                metrics,
                strict=True,
            ),
        )
        pool.close()

        for result, model_name, lang, task, r in zip(
            results, model_names, langs, tasks, runs, strict=True
        ):
            this_df, result_overview, failed_runs, logs = result
            if this_df is not None:
                results_dfs[model_name][f"run_{r}"][task][lang] = this_df
                results_overview.append(result_overview)
            else:
                failed_loads.append(failed_runs)

            # print out all logs
            for log in logs:
                self.elo_printer.print(log["message"], color=log["color"])

        results_overview = pd.DataFrame(results_overview)

        failed_loads = pd.DataFrame(failed_loads)
        failed_loads.to_csv(
            os.path.join(
                self.output_directory_path,
                f"{self.ontological_category}={self.ontological_label}_failed_loads.csv",
            ),
            index=False,
        )

        self.elo_printer.print("Complete.", color=COLORS.OKBLUE)

        return results_dfs, results_overview, models
