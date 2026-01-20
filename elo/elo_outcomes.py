import math
import os
from typing import Any

import numpy as np
import pandas as pd
from elo_utils import COLORS, EloTqdmWrapper


class EloOutcomes:
    """Compute head-to-head outcomes used for Elo scoring.

    This helper aggregates per-sample scores across multiple models, runs,
    tasks, and languages to produce balanced sets of pairwise outcomes
    for Elo computations.

    Attributes:
        results_overview (pd.DataFrame): Overview per model/task/language with
            fields like `num_samples` and `num_turns`.
        results_dfs (dict): Nested mapping of
            {model -> {run -> {task -> {lang -> pd.DataFrame}}}} containing
            per-sample results.
        models (list[str]): Ordered list of model names participating.
        deterministic (bool): Whether sampling should be reproducible.
        seed (int): RNG seed used when `deterministic` is True.
        absolute_difference_tolerance (float): Absolute tolerance within which
            two scores are considered a tie.
        num_seahelm_runs (int): Number of SeaHelm evaluation runs available per
            model.
        output_directory_path (str): Directory to write failure CSVs.
        elo_printer (Any): Printer/logger with a `print` method for progress.
    """

    def __init__(
        self,
        results_overview: pd.DataFrame,
        results_dfs: dict,
        models: list[str],
        deterministic: bool,
        seed: int,
        absolute_difference_tolerance: float,
        num_seahelm_runs: int,
        output_directory_path: str,
        elo_printer: Any,
    ) -> None:
        """Initialize `EloOutcomes` instance.

        Args:
            results_overview (pd.DataFrame): High-level overview per
                model/task/language with consistent `num_samples` and
                `num_turns` across models.
            results_dfs (dict): Nested results mapping
                {model -> {run -> {task -> {lang -> pd.DataFrame}}}}.
            models (list[str]): Ordered list of model names.
            deterministic (bool): If True, sampling is reproducible using `seed`.
            seed (int): RNG seed used when deterministic.
            absolute_difference_tolerance (float): Absolute tolerance to decide
                ties when comparing two scores.
            num_seahelm_runs (int): Number of SeaHelm runs per model.
            output_directory_path (str): Output directory to write failure CSVs.
            elo_printer (Any): Printer/logger with a `print` method.
        """
        self.results_overview = results_overview
        self.results_dfs = results_dfs
        self.models = models
        self.m__model = dict(enumerate(self.models))
        self.deterministic = deterministic
        self.seed = seed
        self.elo_printer = elo_printer
        self.absolute_difference_tolerance = absolute_difference_tolerance
        self.num_seahelm_runs = num_seahelm_runs
        self.output_directory_path = output_directory_path

    def count_num_head_to_head(
        self, num_turns: int, num_table_samples: int, num_models: int
    ) -> int:
        """Compute total number of head-to-head contests available.

        Args:
            num_turns (int): Number of turns per sample for the task.
            num_table_samples (int): Number of samples available in the table.
            num_models (int): Number of participating models.

        Returns:
            int: Total number of pairwise contests across runs, turns and samples.
        """
        return (
            self.num_seahelm_runs
            * num_turns
            * num_table_samples
            * num_models
            * (num_models - 1)
            // 2
        )

    def _get_outcome(self, x: float | int, y: float | int) -> float:
        """Compute outcome for a head-to-head comparison.

        Args:
            x (float | int): Score for model A.
            y (float | int): Score for model B.

        Returns:
            float: 1.0 if A wins, 0.0 if B wins, 0.5 for a tie within tolerance.
        """
        if np.isclose(x, y, atol=self.absolute_difference_tolerance):
            return 0.5
        elif x > y:
            return 1
        else:
            return 0

    @staticmethod
    def convert_paired_sample_ix_to_a_b_ix(
        pair_table_ix: int, num_runs: int, m: int, t: int = 1
    ) -> tuple[int, int, int, int, int]:
        """Converts a the index of a head-to-head contest into a tuple of (sample index, model A, model B,
        turn index).

        For example, with a task_lang (eg: mt-bench_indonesian) that has n=100 samples, m=3 models, and t=2 turns,
        There would be n*Combinations(m,2)*t = 100*3*2 = 600 possible head-to-head contests.
        pair_table_ix would be sampled from the range(600), to uniformly weigh each possible head-to-head contest.
        This function would take a pair_table_ix and convert it into a tuple of (sample index, model A, model B, turn index), so they can be read from the results dataframe.

        Args:
            pair_table_ix (int): Index of a head-to-head contest sampled from the indices of all possible
        head-to-head contests
            num_runs (int): Number of SeaHelm runs.
            m (int): Number of models.
            t (int, optional): Number of turns per sample. Defaults to 1.

        Returns:
            tuple[int, int, int, int, int]: A tuple containing the `(run_ix, sample index, model A, model B, turn index)`.
        """
        run_ix = pair_table_ix % num_runs
        pair_table_ix = pair_table_ix // num_runs
        turn_ix = pair_table_ix % t
        pair_table_ix = pair_table_ix // t

        def f(m):
            return m * (m - 1) // 2

        def strict_floor(x):
            """Returns the floor of x, but if x is an integer, returns floor(x) - 1."""
            return math.floor(x) if x != math.floor(x) else math.floor(x) - 1

        ix = pair_table_ix // (f(m))
        p = pair_table_ix % (f(m))
        a1 = strict_floor((1 + math.sqrt(9 + 8 * p)) / 2)
        a2 = math.ceil((-1 + math.sqrt(9 + 8 * p)) / 2)
        assert a1 == a2  # Redundant but retained for explainabalility
        a = a1
        b = int(p - f(a))
        return run_ix, ix, a, b, turn_ix

    def get_task_lang_outcomes(
        self,
        task: str,
        lang: str,
        required_num_head_to_head_contests: int = 100,
    ) -> pd.DataFrame:
        """Generate outcomes for a specific `(task, lang)` pair.

        This samples head-to-head comparisons across runs, samples, models and
        turns, and returns a DataFrame with the computed outcome per contest.

        Args:
            task (str): Task name.
            lang (str): Language code.
            required_num_head_to_head_contests (int, optional): Number of
                contests to sample. May be sampled with replacement if fewer
                are available. Defaults to 100.

        Returns:
            pd.DataFrame: DataFrame with columns
                `[task, lang, run, sample_ix, modelA, modelB, modelA_res, modelB_res, outcome]`.

        Raises:
            AssertionError: If `num_samples` or `num_turns` differ across models
                for the given `(task, lang)`.
        """
        relevant_overview = self.results_overview[
            (self.results_overview.task == task)
            & (self.results_overview.languages == lang)
        ]

        valid_models = relevant_overview.model.unique()
        num_models = len(valid_models)
        assert relevant_overview.num_samples.nunique() == 1, (
            f"Number of samples not same for all models for {task}-{lang}"
        )

        assert relevant_overview.num_turns.nunique() == 1, (
            f"Number of turns not same for all models for {task}-{lang}"
        )
        num_turns = relevant_overview["num_turns"].iloc[0]

        def _get_number_table_samples() -> int:
            """Return number of available samples for `(task, lang)`.

            Returns:
                int: Number of samples present in any available model/run.
            """
            for m in range(len(self.models)):
                try:
                    return self.results_dfs[self.m__model[m]]["run_0"][task][
                        lang
                    ].shape[0]
                except KeyError:
                    continue
            self.elo_printer.print(
                f"No model found with the task {task} and language {lang} combination. Returning 0 samples.",
                color=COLORS.ERROR,
            )
            return 0

        num_table_samples = _get_number_table_samples()
        available_num_head_to_head_contests = self.count_num_head_to_head(
            num_turns, num_table_samples, num_models
        )

        if self.deterministic:
            random_generator = np.random.default_rng(self.seed)
        else:
            random_generator = np.random.default_rng()

        if available_num_head_to_head_contests < required_num_head_to_head_contests:
            sampled_pair_indices = random_generator.choice(
                available_num_head_to_head_contests,
                required_num_head_to_head_contests,
                replace=True,
            )
            self.elo_printer.print(
                f"Warning: Only {available_num_head_to_head_contests} head-to-head contests available for {task}-{lang}, but {required_num_head_to_head_contests} required. Sampling with replacement.",
                color=COLORS.WARNING,
            )
        else:
            sampled_pair_indices = random_generator.choice(
                available_num_head_to_head_contests,
                required_num_head_to_head_contests,
                replace=False,
            )

        elo_results = []
        failed_run = []

        for sampled_pair_ix in sampled_pair_indices:
            run_ix, ix, a, b, turn_ix = self.convert_paired_sample_ix_to_a_b_ix(
                num_runs=self.num_seahelm_runs,
                pair_table_ix=sampled_pair_ix,
                m=num_models,
                t=num_turns,
            )

            def _get_individual_scores(
                model_ix: int,
                turn_ix: int | None = None,
                run_ix: int | None = None,
                ix: int | None = None,
            ) -> float | int:
                """Read an individual numeric score from results tables.

                For logprob tasks, returns a Bernoulli sample using score/100.

                Args:
                    model_ix (int): Model index in `self.models`.
                    turn_ix (int | None): Optional turn index for multi-turn tasks.

                Returns:
                    float | int: The numeric score for the requested entry, or -1 on failure.
                """
                if "logprobs" in task:
                    override_run_ix = 0
                else:
                    override_run_ix = run_ix

                try:
                    if turn_ix is not None:
                        returned_score = (
                            1
                            * self.results_dfs[self.m__model[model_ix]][
                                f"run_{override_run_ix}"
                            ][task][lang].at[ix, "individual_scores"][turn_ix]
                        )
                    else:
                        returned_score = (
                            1
                            * self.results_dfs[self.m__model[model_ix]][
                                f"run_{override_run_ix}"
                            ][task][lang].at[ix, "individual_scores"]
                        )

                    if "logprobs" in task:
                        return 1 * (
                            np.random.random() < returned_score / 100
                        )  # Bernoulli sampling with probability = returned_score/100
                    else:
                        return returned_score
                except Exception as e:
                    failed_run.append(
                        {
                            "model_name": self.m__model[model_ix],
                            "task": task,
                            "language": lang,
                            "run": run_ix,
                            "ix": ix,
                            "turn_ix": turn_ix,
                            "directory_path": "directory_path",
                            "error": f"{type(e).__name__}:{e}",
                            "success": "",
                        }
                    )
                    return -1

            if num_turns == 1:
                resA = _get_individual_scores(a, run_ix=run_ix, ix=ix)
                resB = _get_individual_scores(b, run_ix=run_ix, ix=ix)
                sample_alias = f"{run_ix}-{ix}"
            else:
                resA = _get_individual_scores(a, turn_ix, run_ix=run_ix, ix=ix)
                resB = _get_individual_scores(b, turn_ix, run_ix=run_ix, ix=ix)
                sample_alias = f"{run_ix}-{ix}-{turn_ix}"

            elo_results.append(
                {
                    "task": task,
                    "lang": lang,
                    "run": run_ix,
                    "sample_ix": sample_alias,
                    "modelA": a,
                    "modelB": b,
                    "modelA_res": resA,
                    "modelB_res": resB,
                    "outcome": self._get_outcome(resA, resB),
                }
            )

        failed_run = pd.DataFrame(failed_run)
        if not failed_run.empty:
            failed_run.to_csv(
                os.path.join(
                    self.output_directory_path,
                    f"failed_runs_{task}_{lang}.csv",
                ),
                index=False,
            )

        return pd.DataFrame(elo_results)

    def _get_samples_per_task(
        self,
        num_samples: int,
        lang: str,
        comp: str,
        print_type: str,
        print_allocations: bool,
    ) -> dict[str, int]:
        """Compute per-task sample allocations for a language/competency.

        Distributes `num_samples` approximately evenly across tasks that match
        the provided language and competency.

        Args:
            num_samples (int): Total samples to allocate across tasks.
            lang (str): Language code to filter tasks.
            comp (str): Competency to filter tasks.
            print_type (str): Label used in allocation printouts.
            print_allocations (bool): Whether to print allocation details.

        Returns:
            dict[str, int]: Mapping of `{task: allocated_samples}`.
        """
        _samples_per_task = {}
        m = len(self.m__model)

        tasks = self.results_overview[
            (self.results_overview.languages == lang)
            & (self.results_overview.competency == comp)
        ].task.unique()
        num_samples_per_task = math.ceil(num_samples / len(tasks))

        self.elo_printer.print(f"\t{print_type}={num_samples}", show=print_allocations)
        for task in tasks:
            n = self.results_overview[
                (self.results_overview.languages == lang)
                & (self.results_overview.competency == comp)
                & (self.results_overview.task == task)
            ].num_samples.iloc[0]
            actual_samples_available = self.count_num_head_to_head(
                2 if task == "mt-bench" else 1, n, m
            )

            self.elo_printer.print(
                f"\t\t{task}={num_samples_per_task} [available = {actual_samples_available}]",
                "*" if actual_samples_available < num_samples_per_task else "",
                show=print_allocations,
            )
            _samples_per_task[task] = num_samples_per_task
        return _samples_per_task

    def _get_outcomes(
        self, samples_per_task: dict[str, dict[str, int]], show_progress: bool = True
    ) -> pd.DataFrame:
        """Collect outcomes for a set of per-task allocations.

        Args:
            samples_per_task (dict[str, dict[str, int]]): Mapping of
                `{lang: {task: num_contests}}` to sample for each task.
            show_progress (bool, optional): Whether to display progress bars.
                Defaults to True.

        Returns:
            pd.DataFrame: Concatenated outcomes DataFrame for all requested contests.
        """
        success_samples = 0
        failed_samples = 0
        expected_samples = 0

        outcomes_df = []

        length = sum([len(samples_per_task[lang]) for lang in samples_per_task.keys()])
        elo_pbar = EloTqdmWrapper(length=length, show_progress=show_progress)
        for lang in samples_per_task.keys():
            for task in samples_per_task[lang]:
                elo_pbar.set_description(f"Processing {task} {lang}")
                this_outcomes_df = self.get_task_lang_outcomes(
                    task,
                    lang,
                    samples_per_task[lang][task],
                )
                outcomes_df.append(this_outcomes_df)
                success_samples += len(this_outcomes_df)

                expected_samples += samples_per_task[lang][task]
                elo_pbar.set_description(
                    f"Success={success_samples}/Expected={expected_samples}, Failed={failed_samples}"
                )
                elo_pbar.update(1)
        elo_pbar.close()

        outcomes_df = pd.concat(outcomes_df).reset_index(drop=True)
        return outcomes_df

    def get_task_outcomes(
        self,
        task: str,
        num_head_to_head_contests: int,
        num_tabs: int = 0,
        print_allocations: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute outcomes for a single task across its languages.

        Args:
            task (str): Task name.
            num_head_to_head_contests (int): Total contests to allocate across languages.
            num_tabs (int, optional): Indentation for printer. Defaults to 0.
            print_allocations (bool, optional): Whether to print allocation details.
                Defaults to False.
            show_progress (bool, optional): Whether to display progress bars.
                Defaults to True.

        Returns:
            pd.DataFrame: Outcomes DataFrame for the requested task.
        """
        languages = self.results_overview[
            self.results_overview.task == task
        ].languages.unique()
        num_samples_per_lang = math.ceil(num_head_to_head_contests / len(languages))

        self.elo_printer.print("=" * 80, show=print_allocations)
        self.elo_printer.print(
            "\t" * num_tabs + "Sample Allocations for Task Elo score for ",
            task,
            show=print_allocations,
        )
        self.elo_printer.print(
            "\t" * num_tabs + "TOTAL_SAMPLES=",
            num_head_to_head_contests,
            show=print_allocations,
        )
        m = len(self.m__model)

        samples_per_lang = {}
        for lang in languages:
            n = self.results_overview[
                (self.results_overview.languages == lang)
                & (self.results_overview.task == task)
            ].num_samples.iloc[0]
            actual_samples_available = self.count_num_head_to_head(
                2 if task == "mt-bench" else 1, n, m
            )
            self.elo_printer.print(
                f"\t\t{lang}={num_samples_per_lang} [available = {actual_samples_available}]",
                "*" if actual_samples_available < num_samples_per_lang else "",
                show=print_allocations,
            )
            samples_per_lang[lang] = {task: num_samples_per_lang}
        self.elo_printer.print("=" * 80, show=print_allocations)

        # Get outcomes for each task
        outcomes_df = self._get_outcomes(samples_per_lang, show_progress=show_progress)
        return outcomes_df

    def get_lang_outcomes(
        self,
        lang: str,
        num_head_to_head_contests: int,
        num_tabs: int = 0,
        print_allocations: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute outcomes for a single language across its competencies/tasks.

        Args:
            lang (str): Language code.
            num_head_to_head_contests (int): Total contests to allocate across competencies.
            num_tabs (int, optional): Indentation for printer. Defaults to 0.
            print_allocations (bool, optional): Whether to print allocation details.
                Defaults to False.
            show_progress (bool, optional): Whether to display progress bars.
                Defaults to True.

        Returns:
            pd.DataFrame: Outcomes DataFrame for the requested language.
        """
        # Calculate the distribution of samples per task, with equal weightage to each competency
        competencies = self.results_overview[
            self.results_overview.languages == lang
        ].competency.unique()
        num_samples_per_comp = math.ceil(num_head_to_head_contests / len(competencies))

        self.elo_printer.print("=" * 80, show=print_allocations)
        self.elo_printer.print(
            "\t" * num_tabs + "Sample Allocations for Language Elo score for ",
            lang,
            show=print_allocations,
        )
        self.elo_printer.print(
            "\t" * num_tabs + "TOTAL_SAMPLES=",
            num_head_to_head_contests,
            show=print_allocations,
        )

        samples_per_task = {lang: {}}
        for comp in competencies:
            samples_per_task[lang].update(
                self._get_samples_per_task(
                    num_samples_per_comp, lang, comp, comp, print_allocations
                )
            )

        # Get outcomes for each task
        outcomes_df = self._get_outcomes(samples_per_task, show_progress=show_progress)

        return outcomes_df

    def get_comp_outcomes(
        self,
        competency: str,
        num_head_to_head_contests: int,
        num_tabs: int = 0,
        print_allocations: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute outcomes for a single competency across its languages/tasks.

        Args:
            competency (str): Competency name.
            num_head_to_head_contests (int): Total contests to allocate across languages.
            num_tabs (int, optional): Indentation for printer. Defaults to 0.
            print_allocations (bool, optional): Whether to print allocation details.
                Defaults to False.
            show_progress (bool, optional): Whether to display progress bars.
                Defaults to True.

        Returns:
            pd.DataFrame: Outcomes DataFrame for the requested competency.
        """
        # Calculate the distribution of samples per task, with equal weightage to each language
        languages = self.results_overview[
            self.results_overview.competency == competency
        ].languages.unique()
        num_samples_per_lang = math.ceil(num_head_to_head_contests / len(languages))

        self.elo_printer.print("=" * 80, show=print_allocations)
        self.elo_printer.print(
            "\t" * num_tabs + "Sample Allocations for Competency Elo score for ",
            competency,
            show=print_allocations,
        )
        self.elo_printer.print(
            "\t" * num_tabs + "TOTAL_SAMPLES=",
            num_head_to_head_contests,
            show=print_allocations,
        )

        samples_per_task = {}
        for lang in languages:
            samples_per_task[lang] = self._get_samples_per_task(
                num_samples_per_lang, lang, competency, lang, print_allocations
            )
        self.elo_printer.print("=" * 80, show=print_allocations)

        # Get outcomes for each task
        outcomes_df = self._get_outcomes(samples_per_task, show_progress=show_progress)

        return outcomes_df

    def get_global_outcomes(
        self,
        balance_by: str,
        num_head_to_head_contests: int,
        print_allocations: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute global outcomes balanced by a category.

        Args:
            balance_by (str): One of ["competency", "task", "language"].
            num_head_to_head_contests (int): Total contests to allocate across entities.
            print_allocations (bool, optional): Whether to print allocation details.
                Defaults to False.
            show_progress (bool, optional): Whether to display progress bars.
                Defaults to True.

        Returns:
            pd.DataFrame: Outcomes DataFrame aggregated across the chosen dimension.

        Raises:
            AssertionError: If `balance_by` is not one of the accepted values.
        """
        assert balance_by in [
            "competency",
            "task",
            "language",
        ], (
            f'Invalid category of {balance_by} provided to rebalance Global Elo. The valid options are ["competency", "task", "language"]'
        )

        if balance_by == "competency":
            entities = self.results_overview.competency.unique()
        elif balance_by == "task":
            entities = self.results_overview.task.unique()
        elif balance_by == "language":
            entities = self.results_overview.languages.unique()

        num_samples_per_entity = math.ceil(num_head_to_head_contests / len(entities))

        self.elo_printer.print("<>" * 40, show=print_allocations)
        self.elo_printer.print(
            "Sample Allocations for Global Elo score balanced by",
            balance_by,
            "with TOTAL_SAMPLES=",
            num_head_to_head_contests,
            show=print_allocations,
        )
        for entity in entities:
            self.elo_printer.print(
                f"\t{entity}={num_samples_per_entity}", show=print_allocations
            )

        outcomes_df = []
        elo_pbar = EloTqdmWrapper(length=len(entities), show_progress=show_progress)

        for entity in entities:
            if balance_by == "competency":
                this_outcomes_df = self.get_comp_outcomes(
                    entity,
                    num_samples_per_entity,
                    num_tabs=1,
                    print_allocations=print_allocations,
                    show_progress=False,
                )
            elif balance_by == "task":
                this_outcomes_df = self.get_task_outcomes(
                    entity,
                    num_samples_per_entity,
                    num_tabs=1,
                    print_allocations=print_allocations,
                    show_progress=False,
                )
            elif balance_by == "language":
                this_outcomes_df = self.get_lang_outcomes(
                    entity,
                    num_samples_per_entity,
                    num_tabs=1,
                    print_allocations=print_allocations,
                    show_progress=False,
                )
            outcomes_df.append(this_outcomes_df)
            elo_pbar.update(1)
        elo_pbar.close()

        outcomes_df = pd.concat(outcomes_df).reset_index(drop=True)
        return outcomes_df
