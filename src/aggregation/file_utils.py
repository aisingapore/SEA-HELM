import glob
import os
import re
from datetime import datetime
from multiprocessing import Pool
from typing import (
    Any,
    NamedTuple,
    Optional,
    Sequence,
    TypeAlias,
    TypedDict,
)

import pandas as pd
import ujson
import yaml


class PerTaskScores(TypedDict):
    scores: list[list[float]]
    length: list[int]
    labels: list[list[Any]]
    metric: list[Optional[str]]


TaskScoresByLang: TypeAlias = dict[str, dict[str, dict[str, PerTaskScores]]]


class TaskRef(NamedTuple):
    """Everything needed to locate and read one task-language-run score file.

    One record per (task, language, run) triple, so the fields travel together
    through the multiprocessing pool instead of as parallel lists that have to be
    kept in step by hand.
    """

    folder: str
    model: str
    run: int
    task: str
    lang: str
    competency: str
    aggregation_group: Optional[str]
    metric: Optional[str]


def model_dir(folder: str, model: str) -> str:
    """Return the directory holding every run of `model` plus its aggregated scores."""
    return os.path.join(folder, model)


def run_dir(folder: str, model: str, run: int) -> str:
    """Return the directory holding one run's configs and inferences."""
    return os.path.join(model_dir(folder, model), f"run_{run}")


def config_glob(folder: str, model: str, run: int) -> str:
    """Return the glob matching one run's task configuration files."""
    return os.path.join(run_dir(folder, model, run), "configs", "*.yaml")


def inference_path(ref: TaskRef) -> str:
    """Return the JSONL of per-question scores for one task-language-run.

    A task belonging to an aggregation group is written under the group's directory
    rather than its own.
    """
    subfolder = ref.aggregation_group if ref.aggregation_group else ref.task
    return os.path.join(
        run_dir(ref.folder, ref.model, ref.run),
        "inferences",
        ref.lang,
        subfolder,
        f"{ref.model}_{ref.task}_{ref.lang}.jsonl",
    )


def scores_path(folder: str, model: str) -> str:
    """Return the aggregated score YAML written for `model`."""
    return os.path.join(model_dir(folder, model), f"{model}_scores.yaml")


def extract_datetime_from_filename(filename: str) -> Optional[datetime]:
    """Extract a datetime from a filename using the specified regex patterns.

    Args:
        filename (str): Filepath or filename string.

    Returns:
        Optional[datetime]: Parsed datetime if a match is found; otherwise None.
    """
    patterns = [
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6})"  # e.g., 2025-06-22T00:10:41.690531
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            dt_str = match.group(1)
            return datetime.fromisoformat(dt_str)
    return None


def get_latest_file(file_list: Sequence[str]) -> Optional[str]:
    """Return the filepath with the most recent embedded datetime.

    Args:
        file_list (Sequence[str]): Filepaths to examine.

    Returns:
        Optional[str]: Path with the latest datetime match or None if none match.
    """
    files_with_dt: list[tuple[datetime, str]] = []
    for f in file_list:
        dt = extract_datetime_from_filename(f)
        if dt:
            files_with_dt.append((dt, f))
    if not files_with_dt:
        return None
    latest = max(files_with_dt, key=lambda x: x[0])
    return latest[1]


def load_config_files(
    folder: str,
    model: str,
    run: int,
) -> tuple[dict[str, Any], list[TaskRef]]:
    """Load the latest task configuration file and prepare it for the multiprocessing run.

    Args:
        folder (str): Root results folder path.
        model (str): Model directory name.
        run (int): Run index whose config to load.

    Returns:
        tuple[dict[str, Any], list[TaskRef]]: The config, and one `TaskRef` per
        (task, language) pair it declares for this run.
    """
    files = glob.glob(config_glob(folder, model, run))
    latest_file = get_latest_file(files)
    config: dict[str, Any] = yaml.safe_load(open(latest_file, "r"))

    refs = [
        TaskRef(
            folder=folder,
            model=model,
            run=run,
            task=task,
            lang=lang,
            competency=task_config["competency"],
            aggregation_group=task_config.get("aggregation_group"),
            metric=task_config["metric"],
        )
        for task, task_config in config["tasks"].items()
        for lang in task_config["languages"].keys()
    ]
    return config, refs


def load_task_scores(
    folder: str, model: str, run_numbers: int, pool: Pool
) -> tuple[list[dict[str, Any]], TaskScoresByLang]:
    """Load all per-task, per-language scores across multiple runs using a pool.

    Args:
        folder (str): Root results folder path.
        model (str): Model directory name.
        run_numbers (int): Number of runs to aggregate (0..run_numbers-1).
        pool (Pool): Multiprocessing Pool for parallel I/O.

    Returns:
        tuple[list[dict[str, Any]], TaskScoresByLang]:
        configs: latest config per run; task_scores: nested lang -> competency -> task with scores/lengths/labels/metric.
    """
    task_scores: TaskScoresByLang = {}
    arguments = pool.starmap(
        load_config_files,
        [(folder, model, run) for run in range(run_numbers)],
    )
    configs = [config for config, _ in arguments]
    refs = [ref for _, run_refs in arguments for ref in run_refs]

    try:
        results = pool.map(get_individual_scores, refs)

        for ref, (values, count, labels) in zip(refs, results, strict=True):
            task_entry = (
                task_scores.setdefault(ref.lang, {})
                .setdefault(ref.competency, {})
                .setdefault(
                    ref.task,
                    PerTaskScores(scores=[], length=[], labels=[], metric=[]),
                )
            )
            task_entry["scores"].append(values)
            task_entry["length"].append(count)
            task_entry["labels"].append(labels)
            task_entry["metric"].append(ref.metric)

    except Exception:
        print(f"Error loading task scores for {model} over {run_numbers} run(s)")
        raise

    return configs, task_scores


def get_individual_scores(ref: TaskRef) -> tuple[list[float], int, list[str]]:
    """Load per-question scores and labels for a single task-language-run.

    Reads the file named by `inference_path`.

    Args:
        ref (TaskRef): The task-language-run to read, including the metric key to pull out
            of each row's `individual_scores` (None means the file stores a bare value with
            no metric name) and the aggregation group that overrides the task subfolder.

    Returns:
        tuple[list[float], int, list[str]]: (values, count, labels) where values are numeric
        scores, count is number of items, and labels are per-item labels if present.

    Raises:
        KeyError: If the run stores no value under `ref.metric`. The key is never inferred
            from the file as a fallback: a run written before a metric was renamed holds a
            different quantity under a different name, and quietly reading that instead
            averages two different metrics together across runs of the same task.
    """
    task, lang, metric = ref.task, ref.lang, ref.metric
    filepath = inference_path(ref)

    values: list[float] = []
    labels: list[str] = []
    if os.path.exists(filepath):
        with open(filepath) as f:
            lines = f.readlines()
        # Load the first line to check the stored structure against the config's metric
        try:
            line_0 = ujson.loads(lines[0])["individual_scores"]
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            raise e

        if isinstance(line_0, dict):
            if metric not in line_0:
                raise KeyError(
                    f"{filepath}: the config resolves task {task!r} to metric {metric!r}, "
                    f"but individual_scores holds {sorted(line_0)}. Either the run predates "
                    f"the current metric definition, or the config names an aggregate that "
                    f"is not stored per question"
                )
        elif metric is not None:
            raise KeyError(
                f"{filepath}: the config resolves task {task!r} to metric {metric!r}, but "
                f"individual_scores holds a bare {type(line_0).__name__}."
            )

        for line in lines:
            individual_row = ujson.loads(line)
            # HACK to omit number words subcategory in IF-Eval for thai and burmese
            if task == "if-eval" and lang in ["th", "my"]:
                if (
                    individual_row["metadata"]["subcategory"]
                    == "length_constraints:number_words"
                ):
                    # print(
                    #     f"Warning: Skipping if-eval subcategory for num_words in {filepath}"
                    # )
                    continue
            if metric is not None:
                value = individual_row["individual_scores"][metric]
            else:
                value = individual_row["individual_scores"]

            # a row may store one value or a list of them; a missing value scores 0
            row_values = value if isinstance(value, list) else [value]
            values.extend(0.0 if pd.isna(v) else float(v) for v in row_values)

            if "label" in individual_row:
                # load labels for use in the balanced accuracy calculation
                labels.append(individual_row["label"])

    return values, len(values), labels
