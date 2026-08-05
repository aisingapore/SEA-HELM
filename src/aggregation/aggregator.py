import statistics
from multiprocessing import Pool
from typing import Any, Optional, Sequence

import numpy as np
import yaml

from src.aggregation.bootstrap import run_bootstraps
from src.aggregation.constants import (
    LOW_CLASS_SUPPORT_THRESHOLD,
    METRIC_NATIVE_MAX,
    SEA_LANGUAGES,
)
from src.aggregation.file_utils import TaskScoresByLang, load_task_scores, scores_path


def _new_competency_node() -> dict[str, Any]:
    """Return an empty competency node, the shape shared by both aggregation trees."""
    return {"tasks": {}, "aggregation_groups": {}}


class MultiRunAggregator:
    def __init__(
        self,
        folder: str,
        model_name: str,
        n_runs: int,
        num_bootstraps: int = 2000,
        num_workers: int = 32,
        ci_percentiles: tuple[float, float] = (2.5, 97.5),
        seed: int = 94370244,
        omit_competencies: Optional[Sequence[str]] = None,
        omit_tasks: Optional[Sequence[str]] = None,
        sea_languages: Sequence[str] = SEA_LANGUAGES,
        metric_scales: Optional[dict[str, float]] = None,
    ):
        """
        Args:
            folder (str): Results folder holding the per-run outputs.
            model_name (str): Model whose runs are aggregated.
            n_runs (int): Number of runs to load.
            num_bootstraps (int): Number of bootstrap samples `run_aggregation` computes.
            num_workers (int): Worker processes in the bootstrap pool.
            ci_percentiles (tuple[float, float]): Percentiles of the reported interval.
            seed (int): RNG seed for reproducibility.
            omit_competencies (Optional[Sequence[str]]): Competency names to exclude.
            omit_tasks (Optional[Sequence[str]]): Task names to exclude in addition to
                `run_args.skip_task`.
            sea_languages (Sequence[str]): Language codes included in the SEA average.
            metric_scales (Optional[dict[str, float]]): Native maximum per metric key,
                passed to `get_native_max` to extend `METRIC_NATIVE_MAX` for an
                unrecognised metric.
        """
        self.folder = folder
        self.model_name = model_name

        self.n_runs = n_runs
        self.num_bootstraps = num_bootstraps
        self.ci_percentiles = ci_percentiles

        self.omit_competencies = list(omit_competencies or [])
        self.omit_tasks = list(omit_tasks or [])
        self.sea_languages = list(sea_languages)
        self.metric_scales = metric_scales

        self.seed = seed
        self.pool = Pool(processes=num_workers)

    def calculate_mean_and_ci(
        self, scores: Sequence[float], prefix: str = ""
    ) -> dict[str, Any]:
        """Calculate the mean and 95% interval of a set of scores.

        Args:
            scores (Sequence[float]): Numeric values per bootstrap/run.
            prefix (str): string prefix for keys (e.g., "sea_").

        Returns:
            dict[str, Any]: A mapping with keys `{prefix}mean`, `{prefix}ci`.
        """
        mean = statistics.mean(scores)
        return {
            f"{prefix}mean": mean.tolist() if isinstance(mean, np.generic) else mean,
            f"{prefix}ci": np.percentile(scores, self.ci_percentiles).tolist(),
        }

    def _record(
        self,
        scores_node: dict[str, Any],
        means_node: dict[str, Any],
        per_bootstrap: list[float],
        prefix: str = "",
    ) -> None:
        """Write one node's summary stats and the bootstrap vector they came from.

        Every level of the tree -- task, aggregation group, competency, language, overall,
        and the SEA subset -- is summarised the same way: the mean and interval go into the
        dumped `aggregated_scores` tree, while the full per-bootstrap vector stays in the
        parallel `aggregated_means` tree so a parent can average over it without the vectors
        reaching the score YAML.

        Args:
            scores_node (dict[str, Any]): Node of the score tree to summarise into.
            means_node (dict[str, Any]): Matching node of the bootstrap-vector tree.
            per_bootstrap (list[float]): This node's score in each bootstrap replicate.
            prefix (str): Key prefix, used for the "sea_" subset aggregate.
        """
        scores_node.update(self.calculate_mean_and_ci(per_bootstrap, prefix))
        means_node[f"{prefix}mean_list"] = per_bootstrap

    def _rollup(
        self,
        scores_node: dict[str, Any],
        means_node: dict[str, Any],
        children: Sequence[Sequence[float]],
        prefix: str = "",
    ) -> list[float]:
        """Fold a node's children into its own bootstrap vector, record it, and return it.

        Every level above a task is the same fold: average the children's per-bootstrap
        vectors elementwise, summarise the result into the node, and hand the vector back so
        the parent can fold it in turn. Tasks are the leaves, already recorded by the
        per-task loop.

        Args:
            scores_node (dict[str, Any]): Node of the score tree to summarise into.
            means_node (dict[str, Any]): Matching node of the bootstrap-vector tree.
            children (Sequence[Sequence[float]]): One per-bootstrap vector per child.
            prefix (str): Key prefix, used for the "sea_" subset aggregate.

        Returns:
            list[float]: This node's score in each bootstrap replicate.
        """
        per_bootstrap = np.mean(children, axis=0).tolist()
        self._record(scores_node, means_node, per_bootstrap, prefix)
        return per_bootstrap

    @staticmethod
    def get_native_max(
        metric: Optional[str],
        metric_scales: Optional[dict[str, float]] = None,
    ) -> float:
        """Return the native maximum of `metric`'s per-question scores.

        Callers derive the 0-100 multiplier from this as `100.0 / native_max`.

        Args:
            metric (Optional[str]): Metric key found under `individual_scores`, or None when
                the JSONL stores a bare value with no metric name.
            metric_scales (Optional[dict[str, float]]): Native maximum per metric key,
                extending or overriding `METRIC_NATIVE_MAX`. Use this to aggregate a result
                folder whose metric the table does not cover.

        Returns:
            float: 1.0 for a metric stored as a fraction in [0, 1], 100.0 for one already
            stored as a percentage in [0, 100].

        Raises:
            KeyError: If the metric's native scale is unknown. This is deliberately fatal
                rather than inferred from the observed values, because a wrong guess silently
                rescales a whole task by 100x.
        """
        scales = dict(METRIC_NATIVE_MAX)
        if metric_scales:
            scales.update(metric_scales)

        if metric not in scales:
            raise KeyError(
                f"Unknown native scale for metric {metric!r}. Add it to METRIC_NATIVE_MAX "
                "(1.0 if individual_scores holds a fraction, 100.0 if it already holds a "
                f"percentage), or pass metric_scales={{{metric!r}: <native max>}} to "
                f"aggregate_scores. Known metrics: {sorted(k for k in scales if k)}."
            )

        return scales[metric]

    def aggregate_scores(
        self,
        configs: Sequence[dict[str, Any]],
        data: TaskScoresByLang,
        n_bootstraps: int = 2000,
    ) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
        """Aggregate per-task and per-language results into competency and overall scores.

        Applies bootstrap sampling, balanced accuracy for multi-choice tasks, and random-chance normalization.
        Also computes SEA-language subset aggregates.

        Args:
            configs (Sequence[dict[str, Any]]): Per-run configuration objects (first referenced for schema).
            data (TaskScoresByLang): Collected scores by `lang -> competency -> task`.
            n_bootstraps (int): Number of bootstrap samples to compute.

        Returns:
            tuple[dict[str, Any], list[tuple[str, str, str]]]:
            aggregated_scores with per-task/competency/language and overall stats, and
            incomplete_tasks as (lang, competency, task) for missing runs.
        """
        generator = np.random.default_rng(self.seed)
        aggregated_scores: dict[str, Any] = {lang: {} for lang in data.keys()}
        aggregated_means: dict[str, Any] = {lang: {} for lang in data.keys()}
        incomplete_tasks: list[tuple[str, str, str]] = []
        omit_tasks = set(self.omit_tasks) | set(configs[0]["run_args"]["skip_task"])

        # Iterate through each task and its configuration and calculate mean and stderr
        for task, task_config in configs[0]["tasks"].items():
            if task in omit_tasks:
                continue

            competency = task_config["competency"]
            if competency in self.omit_competencies:
                continue

            aggregation_group = task_config.get("aggregation_group", None)

            for lang in task_config["languages"].keys():
                competency_node = aggregated_scores[lang].setdefault(
                    competency, _new_competency_node()
                )
                competency_means_node = aggregated_means[lang].setdefault(
                    competency, _new_competency_node()
                )
                task_node = competency_node["tasks"].setdefault(task, {})
                task_means_node = competency_means_node["tasks"].setdefault(task, {})

                task_scores = data[lang][competency][task]["scores"]
                task_length = data[lang][competency][task]["length"][0]
                task_labels = data[lang][competency][task]["labels"]
                valid_tasks = [1 if x != [] else 0 for x in task_scores]
                num_valid_tasks = sum(valid_tasks)

                if num_valid_tasks == 0:
                    task_node.update({"mean": 0, "ci": [0, 0]})
                    task_means_node["mean_list"] = [0] * n_bootstraps

                # incomplete scores are given a score of 0
                if task_config.get("use_logprobs", False):
                    is_task_incomplete = False
                    scores = task_scores[0]
                else:
                    is_task_incomplete = np.any([x == [] for x in task_scores])

                    # remove empty scores from the calculation of the per-question mean
                    scores = [x for x in task_scores if x != []]
                    scores = np.array(scores).mean(axis=0).tolist()

                labels = task_labels[0]
                try:
                    # heuristic: if number of unique labels is less than half the number of questions, treat as multi-choice task
                    is_multi_choice = (
                        labels != [] and len(set(labels)) < len(labels) / 2
                    )
                except TypeError:
                    is_multi_choice = False

                # Ensure that the scale of the scores are the same.
                task_metrics = {
                    metric
                    for metric in data[lang][competency][task]["metric"]
                    if metric is not None
                }
                if len(task_metrics) > 1:
                    raise ValueError(
                        f"Run configs disagree on the metric for task {task!r} "
                        f"({lang}): "
                        f"{sorted(task_metrics)}. Cannot choose a score scale."
                    )
                # With no valid run there is nothing to scale; such a task already fails
                # further down, so avoid masking that with a scale lookup error.
                native_max = (
                    self.get_native_max(next(iter(task_metrics)), self.metric_scales)
                    if num_valid_tasks and task_metrics
                    else 1.0
                )

                # bootstrap the questions
                seeds = generator.integers(0, 2**32 - 1, size=n_bootstraps)
                bootstrap_means = run_bootstraps(
                    scores, labels, is_multi_choice, native_max, seeds, self.pool
                )
                self._record(task_node, task_means_node, bootstrap_means)
                if is_task_incomplete:
                    incomplete_tasks.append((lang, competency, task))
                    task_node["is_incomplete"] = True

                if is_multi_choice:
                    # A rare class dominates the uncertainty of a macro-averaged score, so
                    # record the supports when one of them is thin enough to matter.
                    class_support = {
                        str(label): int(count)
                        for label, count in zip(
                            *np.unique(labels, return_counts=True), strict=True
                        )
                    }
                    if min(class_support.values()) < LOW_CLASS_SUPPORT_THRESHOLD:
                        task_node["low_class_support"] = class_support

                if aggregation_group:
                    aggregation_group = aggregation_group.replace("-logprobs", "")

                    task_node["remarks"] = (
                        f"Using scores for aggregation group {aggregation_group.upper()} instead of individual task scores."
                    )
                    task_node["ignore"] = True
                    task_node["aggregation_group"] = aggregation_group

                    competency_node["aggregation_groups"].setdefault(
                        aggregation_group, {}
                    )
                    competency_means_node["aggregation_groups"].setdefault(
                        aggregation_group, {}
                    )[task] = {
                        "mean_list": task_means_node["mean_list"],
                        "length": task_length,
                    }

        # Calculate mean and stderr for each language, competency, and aggregation group
        overall_scores: list[list[float]] = []
        for lang, lang_node in aggregated_scores.items():
            lang_means_node = aggregated_means[lang]
            lang_scores: list[list[float]] = []
            for competency, competency_node in lang_node.items():
                competency_means_node = lang_means_node[competency]
                competency_scores: list[list[float]] = [
                    competency_means_node["tasks"][task]["mean_list"]
                    for task, task_node in competency_node["tasks"].items()
                    if not task_node.get("ignore", False)
                ]

                group_scores_nodes = competency_node["aggregation_groups"]
                group_means_nodes = competency_means_node["aggregation_groups"]
                for aggregation_group, per_task_means in group_means_nodes.items():
                    group_children = [x["mean_list"] for x in per_task_means.values()]
                    group_means_nodes[aggregation_group] = {}
                    competency_scores.append(
                        self._rollup(
                            group_scores_nodes[aggregation_group],
                            group_means_nodes[aggregation_group],
                            group_children,
                        )
                    )

                lang_scores.append(
                    self._rollup(
                        competency_node, competency_means_node, competency_scores
                    )
                )

            overall_scores.append(self._rollup(lang_node, lang_means_node, lang_scores))

        self._rollup(aggregated_scores, aggregated_means, overall_scores)
        self._rollup(
            aggregated_scores,
            aggregated_means,
            [
                aggregated_means[lang]["mean_list"]
                for lang in data.keys()
                if lang in self.sea_languages
            ],
            prefix="sea_",
        )

        if incomplete_tasks:
            aggregated_scores["is_incomplete"] = True

        return aggregated_scores, incomplete_tasks

    def run_aggregation(self) -> None:
        configs, data = load_task_scores(
            self.folder, self.model_name, self.n_runs, self.pool
        )
        scores, incomplete_tasks = self.aggregate_scores(
            configs, data, n_bootstraps=self.num_bootstraps
        )
        if scores.get("is_incomplete", False):
            print(f"Overall incomplete data detected for {self.model_name}!")
            for lang, competency, task in incomplete_tasks:
                print(f"Incomplete task: {task} for {lang}, competency: {competency}")

        with open(scores_path(self.folder, self.model_name), "w") as f:
            yaml.dump(
                scores,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
