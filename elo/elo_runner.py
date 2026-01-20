import argparse
import json
import math
import os
from datetime import datetime
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from elo_outcomes import EloOutcomes
from elo_results_loader import EloResultsLoader
from elo_utils import COLORS, EloPrintWrapper
from sklearn.linear_model import LogisticRegression


class EloRunner:
    """Calculate ELO scores for SEA-HELM model results.

    This class loads SEA-HELM results, samples head-to-head matchups,
    computes ELO scores (MLE or regular), estimates uncertainty via bootstrapping,
    and writes comprehensive outputs (scores, outcomes, bootstrap samples, p-values).

    The runner supports filtering/overrides and different ontological granularities
    (task, competency, language, global).
    """

    def __init__(
        self,
        results_directory_path: str,
        output_directory_path: str,
        ontological_category: str,
        ontological_label: str,
        metric_override_dict: dict | None = None,
        ignore_task: dict | None = None,
        ignore_language: set | None = None,
        ignore_task_lang: set | None = None,
        ignore_model: set | None = None,
        elo_method: str = "mle",
        scale: int = 400,
        base: int = 10,
        init_rating: int = 1000,
        k: int = 4,
        keep_ties: bool = True,
        num_bootstrap: int = 30,
        num_head_to_head_contests: int = 20000,
        alpha_normal: float = 0.05,
        num_jobs: int = 14,
        deterministic: bool = False,
        seed: int | None = None,
        absolute_difference_tolerance: float = 1e-4,
        num_seahelm_runs: int = 1,
        tasks_config_filepath: str | None = None,
        optional_suffix: str = "",
        override_ontological_category: str | None = None,
        override_ontological_label: str | None = None,
    ) -> None:
        """Initialize an `EloRunner` instance.

        Args:
            results_directory_path (str): Path containing SEA-HELM results (model inferences).
            output_directory_path (str): Path to write ELO artifacts and logs.
            ontological_category (str): One of {"task", "competency", "language", "global"}.
            ontological_label (str): Label within the chosen ontological category to balance on.
            metric_override_dict (dict | None): Optional mapping to override default metrics per task.
            ignore_task (dict | None): Mapping of tasks to ignore (optionally by language).
            ignore_language (set | None): Languages to ignore entirely.
            ignore_task_lang (set | None): Specific (task, language) combinations to ignore.
            ignore_model (set | None): Models to exclude from the ELO analysis.
            elo_method (str): "mle" or "regular". Defaults to "mle" (logistic regression based).
            scale (int): ELO scale factor used in odds calculation. Defaults to 400.
            base (int): Logarithm base used in ELO odds. Defaults to 10.
            init_rating (int): Initial rating for all models. Defaults to 1000.
            k (int): Learning rate for regular ELO updates. Used only in regular ELO. Defaults to 4.
            keep_ties (bool): Whether to include ties in ELO. Defaults to True.
            num_bootstrap (int): Number of bootstrap samples. Defaults to 30.
            num_head_to_head_contests (int): Total sampled head-to-head matches per bootstrap. Defaults to 20000.
            alpha_normal (float): Significance threshold for Shapiro-Wilk normality tests. Defaults to 0.05.
            num_jobs (int): Parallel jobs for logistic regression in MLE ELO. Defaults to 14.
            deterministic (bool): Deterministic sampling; requires `seed`. Defaults to False.
            seed (int | None): Seed used when deterministic is True.
            absolute_difference_tolerance (float): Tolerance to consider two scores equal. Defaults to 1e-4.
            num_seahelm_runs (int): Number of parallel SEA-HELM runs to consider. Defaults to 1.
            tasks_config_filepath (str | None): Optional path to tasks config to align with.
            optional_suffix (str): Optional suffix appended to experiment name for disambiguation.
            override_ontological_category (str | None): Override category for this run only.
            override_ontological_label (str | None): Override label for this run only.
        """
        self.results_directory_path = results_directory_path
        self.output_directory_path = output_directory_path
        self.deterministic = False
        self.num_seahelm_runs = (
            num_seahelm_runs  # Number of parallel runs for each seahelm task
        )
        self.ontological_category = ontological_category
        self.ontological_label = ontological_label
        if deterministic:
            self.on_determinstic_mode(seed=seed)
        else:
            self.seed = None

        self.latest_exp_type = None
        self.latest_run_outcome_df = None
        self.latest_run_mle_elo_results = None

        # elo settings
        self.elo_method = elo_method
        self.num_bootstrap = num_bootstrap
        self.num_head_to_head_contests = num_head_to_head_contests
        self.keep_ties = keep_ties
        self.scale = scale
        self.base = base
        self.init_rating = init_rating
        self.k = k  # only used in regular ELO
        self.num_jobs = num_jobs  # only used in MLE ELO for logistic regression
        self.alpha_normal = alpha_normal  # Significance level for rejecting the null hypothesis that the distribution of bootstrapped elo scores is Normal.

        # overrides
        self.metric_override_dict = metric_override_dict
        self.ignore_task = ignore_task
        self.ignore_language = ignore_language
        self.ignore_task_lang = ignore_task_lang
        self.ignore_model = ignore_model

        if override_ontological_category is not None:
            self.this_ontological_category = override_ontological_category
        else:
            self.this_ontological_category = self.ontological_category

        if override_ontological_label is not None:
            self.this_ontological_label = override_ontological_label
        else:
            self.this_ontological_label = self.ontological_label

        self.optional_suffix = optional_suffix

        def _get_latest_exp_type():
            if self.deterministic:
                return f"{self.this_ontological_category}={self.this_ontological_label}_num_contests={self.num_head_to_head_contests}_method={self.elo_method}_num_bootstrap={self.num_bootstrap}_keep_ties={self.keep_ties}_deterministic={self.deterministic}{self.optional_suffix}"
            else:
                return f"{self.this_ontological_category}={self.this_ontological_label}_num_contests={self.num_head_to_head_contests}_method={self.elo_method}_num_bootstrap={self.num_bootstrap}_keep_ties={self.keep_ties}{self.optional_suffix}"

        self.latest_exp_type = _get_latest_exp_type()

        if not os.path.exists(self.output_directory_path):
            os.makedirs(self.output_directory_path, exist_ok=True)
        exp_path = os.path.join(self.output_directory_path, self.latest_exp_type)
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)

        # Generate timestamp for log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"elo_runner_{timestamp}.log"

        self.elo_printer = EloPrintWrapper(
            log_file=os.path.join(
                output_directory_path, self.latest_exp_type, log_filename
            )
        )
        self.elo_printer.print("EloRunner Configuration:", color=COLORS.OKBLUE)
        self.elo_printer.print(self)

        # loaded in load_results()
        results_loader = EloResultsLoader(
            results_directory_path,
            output_directory_path,
            ontological_category,
            ontological_label,
            metric_override_dict=metric_override_dict,
            ignore_task=ignore_task,
            ignore_language=ignore_language,
            ignore_task_lang=ignore_task_lang,
            ignore_model=ignore_model,
            num_seahelm_runs=num_seahelm_runs,
            tasks_config_filepath=tasks_config_filepath,
            elo_printer=self.elo_printer,
        )
        self.results_dfs, self.results_overview, self.models = (
            results_loader.load_results()
        )
        self.m__model = dict(enumerate(self.models))

        self.elo_outcomes = EloOutcomes(
            results_overview=self.results_overview,
            results_dfs=self.results_dfs,
            models=self.models,
            deterministic=self.deterministic,
            seed=self.seed,
            absolute_difference_tolerance=absolute_difference_tolerance,
            num_seahelm_runs=num_seahelm_runs,
            output_directory_path=output_directory_path,
            elo_printer=self.elo_printer,
        )

    def __repr__(self) -> str:
        """Return a readable representation of the `EloRunner` configuration.

        Returns:
            str: Multiline string enumerating instance attributes (omits the printer).
        """
        attrs = []
        for key, value in self.__dict__.items():
            if key == "elo_printer":
                # skip elo_printer
                continue
            elif hasattr(value, "__len__") and not isinstance(value, str):
                # For collections, show type and length
                if isinstance(value, dict):
                    str_repr = f"{key}={{"
                    for k, v in value.items():
                        str_repr += f"\n    {k}: {repr(v)}"
                    str_repr += "}"
                    attrs.append(str_repr)
                elif isinstance(value, (list, tuple, set)):
                    attrs.append(f"{key}=[{','.join(repr(v) for v in value)}]")
                else:
                    attrs.append(f"{key}={repr(value)}")
            else:
                attrs.append(f"{key}={repr(value)}")

        return "EloRunner(\n    " + ",\n    ".join(attrs) + "\n)"

    def on_deterministic_mode(self, seed: int) -> None:
        """Enable deterministic sampling mode.

        Args:
            seed (int): Random seed to control sampling deterministically.

        Returns:
            None
        """
        assert seed is not None, "Seed must be provided for deterministic mode."
        self.seed = (
            seed  # seed is not actually used here - used in the sampling process
        )
        self.deterministic = True

    def off_determinstic_mode(self) -> None:
        """Disable deterministic sampling mode.

        Returns:
            None
        """
        self.deterministic = False

    def get_win_rates(self, df: pd.DataFrame) -> dict[int, float]:
        """Compute per-model win rates from outcomes.

        Args:
            df (pd.DataFrame): Outcomes with columns {"modelA", "modelB", "outcome"} where
                outcome in {0 (B wins), 0.5 (tie), 1 (A wins)}.

        Returns:
            dict[int, float]: Mapping from model index to win rate in [0, 1].
        """
        win_rates = {}

        for i in range(len(self.m__model)):
            model_count = df["modelA"].value_counts().get(i, 0) + df[
                "modelB"
            ].value_counts().get(i, 0)
            win_count = df[df["outcome"] == 1]["modelA"].value_counts().get(i, 0) + df[
                df["outcome"] == 0
            ]["modelB"].value_counts().get(i, 0)

            if model_count == 0:
                self.elo_printer.print(
                    f"Model {self.m__model[i]} has no wins or losses. Setting win rate to 0.",
                    color=COLORS.WARNING,
                )
                win_rates[i] = 0.0
            else:
                win_rates[i] = win_count / model_count

        return win_rates

    def get_regular_elo_score(
        self, df: pd.DataFrame, keep_ties: bool = False
    ) -> dict[int, float]:
        """Compute regular (online update) ELO scores.

        Args:
            df (pd.DataFrame): Outcomes with columns {"modelA", "modelB", "outcome"}.
            keep_ties (bool): Whether to include tie games as 0.5 outcomes. Defaults to False (ignore ties).

        Returns:
            dict[int, float]: Mapping model index to ELO score.
        """
        m = len(self.m__model)
        elo_scores = dict.fromkeys(range(m), self.init_rating)

        df = df.sample(frac=1, random_state=self.seed)

        for i in range(len(df)):
            k = self.k * (1 - i / len(df))

            if not keep_ties and df.outcome[i] == 0.5:
                continue
            p_A = 1 / (
                1
                + self.base
                ** (
                    (elo_scores[df.modelB.iloc[i]] - elo_scores[df.modelA.iloc[i]])
                    / self.scale
                )
            )
            p_B = 1 - p_A

            elo_scores[df.modelA.iloc[i]] += k * (df.outcome.iloc[i] - p_A)
            elo_scores[df.modelB.iloc[i]] += k * ((1 - df.outcome.iloc[i]) - p_B)

        return elo_scores

    def get_mle_elo_score(
        self, df: pd.DataFrame, keep_ties: bool = False
    ) -> dict[int, float]:
        """Compute MLE-based ELO scores via logistic regression.

        Treats each head-to-head as a pairwise comparison between models with
        features indicating the active participants. Optionally expands ties into
        two balanced outcomes.

        Args:
            df (pd.DataFrame): Outcomes with columns {"modelA", "modelB", "outcome"}.
            keep_ties (bool): Whether to include ties as balanced outcomes. Defaults to False.

        Returns:
            dict[int, float]: Mapping model index to ELO score.
        """
        m = len(self.m__model)

        if not keep_ties:
            non_ties = df[df.outcome != 0.5]

            X_A = 1 * pd.get_dummies(non_ties["modelA"])
            for i in range(m):
                if i not in X_A.columns:
                    X_A[i] = 0
            X_A = X_A[sorted(X_A)]

            X_B = 1 * pd.get_dummies(non_ties["modelB"])
            for i in range(m):
                if i not in X_B.columns:
                    X_B[i] = 0
            X_B = X_B[sorted(X_B.columns)]
            X = (X_A - X_B).to_numpy()

            Y = non_ties["outcome"]
        else:

            def _process_mle_tie_outcomes(df, num_models):
                X = []
                Y = []
                for i in range(len(df)):
                    row = df.iloc[i]
                    a = row["modelA"]
                    b = row["modelB"]
                    this_outcome = row["outcome"]

                    this_X = np.zeros((2, num_models))
                    this_Y = np.zeros(2)
                    if this_outcome == 0:  # B wins, 2 losses for A
                        this_X[0, a] = 1
                        this_X[0, b] = -1
                        this_Y[0] = 0

                        this_X[1, a] = -1
                        this_X[1, b] = 1
                        this_Y[1] = 1
                    elif this_outcome == 1:  # A wins, 2 wins for A
                        this_X[0, a] = 1
                        this_X[0, b] = -1
                        this_Y[0] = 1

                        this_X[1, a] = -1
                        this_X[1, b] = 1
                        this_Y[1] = 0
                    else:  # Tie, 1 win each
                        this_X[0, a] = 1
                        this_X[0, b] = -1
                        this_Y[0] = 1

                        this_X[1, a] = -1
                        this_X[1, b] = 1
                        this_Y[1] = 1

                    X.append(this_X)
                    Y.append(this_Y)
                X = np.vstack(X)
                Y = np.hstack(Y)
                return X, Y

            X, Y = _process_mle_tie_outcomes(df, m)

        assert len(X) == len(Y), "Length of X and Y should be the same"
        lr = LogisticRegression(
            fit_intercept=False, penalty=None, tol=1e-4, n_jobs=self.num_jobs
        )
        lr.fit(X, Y)

        elo_scores = self.scale * lr.coef_[0] / math.log(self.base) + self.init_rating
        elo_dict = dict(zip(range(m), elo_scores, strict=True))
        return elo_dict

    def run_pairwise_t_tests(self, bootstrap_df: pd.DataFrame) -> pd.DataFrame:
        """Run dependent two-sample t-tests across bootstrap ELO samples.

        Performs Shapiro-Wilk normality checks (with Bonferroni correction),
        then computes paired t-tests for each model pair.

        Args:
            bootstrap_df (pd.DataFrame): Rows are bootstrap samples; columns are model names.

        Returns:
            pd.DataFrame: Symmetric matrix of p-values for each model pair.
        """
        self.elo_printer.print("=" * 80)
        self.elo_printer.print(
            "Running Dependent 2-Sample Pairwise T-tests", color=COLORS.OKBLUE
        )

        bootstrap_df = bootstrap_df[
            bootstrap_df.agg(["mean"])
            .sort_values(by="mean", axis=1, ascending=False)
            .columns
        ]  # Sort columns by mean
        models = bootstrap_df.columns
        num_models = len(models)

        self.elo_printer.print(
            "Running Normality Tests using Shapiro-Wilks, as T-tests assume underlying distribution is normal\nH0: Normal; H1: Not Normal",
            color=COLORS.OKBLUE,
        )
        for model in models:
            _, p = stats.shapiro(bootstrap_df[model])
            if p >= (
                self.alpha_normal / len(models)
            ):  # Bonferroni correction as we are running multiple tests
                pass
            else:
                self.elo_printer.print(
                    f"Model: {model}, p-value: {p}: Normality assumption rejected at Alpha={self.alpha_normal}",
                    color=COLORS.WARNING,
                )

        result_combinations = np.zeros((num_models, num_models))
        ix__models = dict(enumerate(models))

        self.elo_printer.print(
            "Running dependent pairwise T-tests.\nH0: Mean(ModelA)=Mean(ModelB); H1: Mean(ModelA)!=Mean(ModelB)",
            color=COLORS.OKBLUE,
        )
        for a, b in combinations(range(num_models), 2):
            # Run paired t-test for all combinations of models
            _, p_value = stats.ttest_rel(
                bootstrap_df[ix__models[a]], bootstrap_df[ix__models[b]]
            )
            result_combinations[a, b] = p_value
        self.elo_printer.print("Complete.", color=COLORS.OKBLUE)
        self.elo_printer.print("=" * 80)

        result_combinations = (
            result_combinations + result_combinations.T
        )  # Test is symmetrical
        result_combinations = pd.DataFrame(result_combinations)
        result_combinations.columns = models
        result_combinations.index = models
        return result_combinations

    def run_bootstrap_experiment(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run the full bootstrap experiment and write out results.

        For each bootstrap iteration:
        - Sample head-to-head outcomes according to the chosen ontology
        - Compute ELO (MLE or regular)
        - Track win/tie rates

        After all bootstraps, compute pairwise p-values and write artifacts to disk.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - ELO summary table (mean/CI/win rate)
                - Outcomes dataframe (the last sampled outcomes)
                - Bootstrap sample ELO scores
                - Pairwise t-test p-value matrix
        """
        get_ontological_outcomes = {
            "task": self.elo_outcomes.get_task_outcomes,
            "competency": self.elo_outcomes.get_comp_outcomes,
            "language": self.elo_outcomes.get_lang_outcomes,
            "global": self.elo_outcomes.get_global_outcomes,
        }

        elo_scores = []
        win_rates_list = []
        ties_rates_list = []

        for n in range(self.num_bootstrap):
            print_allocations = (
                True if n == 0 else False
            )  # Print allocations only for the first bootstrap
            outcomes_df = get_ontological_outcomes[self.this_ontological_category](
                self.this_ontological_label,
                num_head_to_head_contests=self.num_head_to_head_contests,
                print_allocations=print_allocations,
            )

            self.elo_printer.print(
                f"Evaluating on {len(outcomes_df)}/{self.num_head_to_head_contests} head to head contests",
                show=print_allocations,
            )

            if self.elo_method == "regular":
                this_elo = self.get_regular_elo_score(
                    outcomes_df, keep_ties=self.keep_ties
                )
            else:
                this_elo = self.get_mle_elo_score(outcomes_df, keep_ties=self.keep_ties)

            win_rates = self.get_win_rates(outcomes_df)
            this_tie_rate = 100 * (outcomes_df.outcome == 0.5).sum() / len(outcomes_df)

            win_rates_list.append(win_rates)
            elo_scores.append(this_elo)

            ties_rates_list.append(this_tie_rate)

        elo_scores = pd.DataFrame(elo_scores)
        elo_scores.columns = [self.m__model[i] for i in range(len(self.m__model))]
        pairwise_p_values = self.run_pairwise_t_tests(elo_scores)
        win_rates_list = pd.DataFrame(win_rates_list)
        win_rates_list.columns = [self.m__model[i] for i in range(len(self.m__model))]

        df1 = win_rates_list.agg(["mean", "std"]).T
        df2 = elo_scores.agg(["mean", "std"]).T

        result_df = df1.join(
            df2, lsuffix="_win_rate", rsuffix="_mle_elo_score"
        ).sort_values("mean_mle_elo_score", ascending=False)[
            ["mean_mle_elo_score", "std_mle_elo_score", "mean_win_rate", "std_win_rate"]
        ]
        assert self.num_bootstrap > 1, (
            "Number of bootstrap samples should be greater than 1 for confidence intervals"
        )
        result_df["std_mle_elo_score"] *= np.sqrt(
            self.num_bootstrap / (self.num_bootstrap - 1)
        )  # Unbiased estimate of population's standard deviation
        result_df["2.5_percentile"] = result_df["mean_mle_elo_score"] + stats.norm.ppf(
            0.025
        ) * result_df["std_mle_elo_score"] / np.sqrt(self.num_bootstrap)
        result_df["97.5_percentile"] = result_df["mean_mle_elo_score"] + stats.norm.ppf(
            0.975
        ) * result_df["std_mle_elo_score"] / np.sqrt(self.num_bootstrap)
        result_df = result_df[
            [
                "2.5_percentile",
                "mean_mle_elo_score",
                "97.5_percentile",
                "std_mle_elo_score",
                "mean_win_rate",
                "std_win_rate",
            ]
        ]

        # Keeps track of the latest experiment run
        self.latest_run_outcome_df = outcomes_df
        self.latest_run_mle_elo_results = result_df

        self.elo_printer.print(
            "Average Tie Rate = ", round(np.mean(ties_rates_list), 2), "%"
        )
        self.elo_printer.print(
            "Average Missing Rate = ",
            round(
                100
                * (
                    (outcomes_df["modelA_res"] == -1)
                    + (outcomes_df["modelB_res"] == -1)
                ).mean(),
                2,
            ),
            "%",
        )

        def _save_results(df, name, filepath):
            df.to_csv(filepath)
            self.elo_printer.print(f"{name} saved to", filepath, color=COLORS.OKGREEN)

        exp_path = os.path.join(self.output_directory_path, self.latest_exp_type)

        _save_results(
            result_df,
            "Elo results",
            os.path.join(exp_path, "elo_result.csv"),
        )
        _save_results(
            outcomes_df,
            "Game outcomes",
            os.path.join(exp_path, "game_outcomes.csv"),
        )
        _save_results(
            elo_scores,
            "Bootstrap samples",
            os.path.join(exp_path, "bootstrap_samples.csv"),
        )
        _save_results(
            pairwise_p_values,
            "Pairwise T-Test P values",
            os.path.join(exp_path, "pairwise_p_values.csv"),
        )

        return result_df, outcomes_df, elo_scores, pairwise_p_values


def get_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the ELO runner.

    Returns:
        argparse.ArgumentParser: Configured parser for `elo_runner.py`.
    """
    parser = argparse.ArgumentParser(description="Configurations for ELO Runner")

    # Arguments for EloRunner object
    parser.add_argument(
        "-r",
        "--results_directory_path",
        type=str,
        help="Path to results directory which contains models inferences",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_directory_path",
        type=str,
        help="Path to output directory where the Elo results will be saved",
        required=True,
    )
    parser.add_argument(
        "--scale",
        help="Scale of the Elo in the odds calculation - 400 is default",
        type=int,
        required=False,
        default=400,
    )
    parser.add_argument(
        "--base",
        help="Base of the logarithm used in Elo - 10 is default",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "--init_rating",
        help="Initial rating for Elo - 1000 is default",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--k",
        help="K = learning rate for Elo - 4 is default. Only used in regular Elo",
        type=int,
        required=False,
        default=4,
    )

    # Arguments for bootstrap experiment
    parser.add_argument(
        "-c",
        "--ontological_category",
        help="Ontological category to run the Elo on - task, competency, language or global",
        type=str,
        choices=["task", "competency", "language", "global"],
        required=False,
        default="global",
    )
    parser.add_argument(
        "-l",
        "--ontological_label",
        help="Ontological label within ontological category to run/balance the Elo on.",
        type=str,
        required=False,
        default="language",
    )
    parser.add_argument(
        "-s",
        "--total_samples",
        help="Total number of head-to-head matches to run the Elo on. Default is 20000.",
        type=int,
        required=False,
        default=20000,
    )
    parser.add_argument(
        "-n",
        "--num_bootstrap",
        help="Number of bootstrap samples to run the Elo on. Default is 30.",
        type=int,
        required=False,
        default=30,
    )
    parser.add_argument(
        "-m",
        "--elo_method",
        help="Method to calculate Elo - mle or regular.",
        type=str,
        choices=["mle", "regular"],
        required=False,
        default="mle",
    )
    parser.add_argument(
        "--num_jobs",
        help="Number of jobs to run the logistic regression in MLE Elo. Default is 14.",
        type=int,
        required=False,
        default=14,
    )
    parser.add_argument(
        "-R",
        "--num_seahelm_runs",
        help="Number of parallel SEA-HELM runs to consider for the Elo calculation. Default is 1.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--alpha_normal",
        help="Significance level for rejecting the null hypothesis that the distribution of bootstrapped elo scores is Normal. Default is 0.05",
        type=float,
        required=False,
        default=0.05,
    )
    parser.add_argument(
        "-d",
        "--deterministic",
        help="Samples bootstraps determinstically. Supply a seed if you wish.",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        help="Random seed for sampling bootstraps determinstically.",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--overrides_filename",
        help="Path to a JSON file containing overrides for the EloRunner object. Default is None.",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--keep_ties",
        action="store_true",
        help="Includes ties in the Elo calculation. Ties are ignored if not set.",
    )
    parser.add_argument(
        "--atol",
        help="Absolute tolerance for concluding two scores are equal. Default is 1e-4.",
        type=float,
        required=False,
        default=1e-4,
    )
    return parser


def get_overrides(overrides_filepath: str | None = None) -> dict[str, Any]:
    """Load optional overrides from a JSON file.

    Args:
        overrides_filepath (str | None): Path to a JSON file with keys such as
            METRICS_OVERRIDE_DICT, IGNORE_TASK, IGNORE_TASK_LANG, IGNORE_LANG, IGNORE_MODEL.

    Returns:
        dict[str, Any]: The overrides mapping, or an empty dict if unavailable.
    """
    if overrides_filepath is None:
        return {}
    try:
        with open(overrides_filepath, "r") as f:
            overrides = json.load(f)
    except Exception:
        print(
            f"Given overrides file of {overrides_filepath} not found. Returning empty overrides."
        )
        overrides = {}
    return overrides


if __name__ == "__main__":
    args = get_parser().parse_args()
    config = vars(args)
    overrides = get_overrides(config.get("overrides_filename", None))

    elo = EloRunner(
        results_directory_path=config["results_directory_path"],
        output_directory_path=config["output_directory_path"],
        ontological_category=config["ontological_category"],
        ontological_label=config["ontological_label"],
        metric_override_dict=overrides.get("METRICS_OVERRIDE_DICT", {}),
        ignore_task=overrides.get("IGNORE_TASK", {}),
        ignore_task_lang=overrides.get("IGNORE_TASK_LANG", {}),
        ignore_language=overrides.get("IGNORE_LANG", {}),
        ignore_model=overrides.get("IGNORE_MODEL", {}),
        scale=config["scale"],
        base=config["base"],
        init_rating=config["init_rating"],
        k=config["k"],
        num_jobs=config["num_jobs"],
        alpha_normal=config["alpha_normal"],
        deterministic=config["deterministic"],
        seed=config["seed"],
        absolute_difference_tolerance=config["atol"],
        num_seahelm_runs=config["num_seahelm_runs"],
        num_bootstrap=config["num_bootstrap"],
        num_head_to_head_contests=config["total_samples"],
        elo_method=config["elo_method"],
        keep_ties=config["keep_ties"],
    )

    elo.run_bootstrap_experiment()
