"""Bradley-Terry rating system, fitted with L-BFGS-B."""

import math
import multiprocessing as mp
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit
from scipy.stats import norm

from .data_utils import PairDataset


def lbfgs_minimize(
    loss_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    max_iter: int = 1000,
    gtol: float = 1e-6,
    ftol: float = 1e-9,
    verbose: bool = False,
) -> tuple[np.ndarray, OptimizeResult]:
    """Minimizes a function returning ``(value, gradient)`` with SciPy's L-BFGS-B.

    Stops as soon as either ``max|proj g| <= gtol`` or the relative loss change
    ``(f_k - f_k+1) / max(|f_k|, |f_k+1|, 1) <= ftol``.

    Returns:
        ``(x, result)``: the minimizer and the SciPy ``OptimizeResult``.
    """
    result = minimize(
        loss_and_grad,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": max_iter,
            "maxfun": max(15000, 10 * max_iter),
            "gtol": gtol,
            "ftol": ftol,
        },
    )
    if verbose:
        print(f"L-BFGS finished in {result.nit} iterations.")
        print(f"  Final Loss: {result.fun:.6f}")
        print(f"  Grad Norm:  {np.abs(result.jac).max():.2e} (tol={gtol})")
        print(f"  Converged:  {result.success} ({result.message})")
    return result.x, result


def pairwise_matrix(
    vals: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray, n: int
) -> np.ndarray:
    """Accumulates per-pair values into an (n, n) matrix of competitor interactions.

    Each value ``v`` for the pair (a, b) adds ``+v`` to both diagonal entries and ``-v``
    to both off-diagonal ones, giving a weighted graph Laplacian of the matchup graph.
    """
    # intp keeps the flattened n * n indices from wrapping
    idx_a = np.asarray(idx_a, dtype=np.intp)
    idx_b = np.asarray(idx_b, dtype=np.intp)
    flat_idx = np.concatenate(
        [idx_a * n + idx_a, idx_b * n + idx_b, idx_a * n + idx_b, idx_b * n + idx_a]
    )
    flat_vals = np.concatenate([vals, vals, -vals, -vals])
    return np.bincount(flat_idx, flat_vals, n * n).reshape(n, n)


# top level so that it is pickle-able for multiprocessing
def _fit_bootstrap_sample(
    boot_counts: np.ndarray, model: "BradleyTerry", dataset: PairDataset
) -> np.ndarray:
    """Refits ratings from scratch on one multinomial resample of the aggregated rows."""
    boot_dataset = replace(
        dataset, counts=boot_counts, opt_weights=boot_counts * dataset.weights
    )
    return model.solve(boot_dataset, np.zeros(model.n_competitors))


@dataclass(eq=False)
class BradleyTerry:
    """Bradley-Terry ratings with sandwich or bootstrap confidence intervals.

    Attributes:
        n_competitors: Number of competitors being rated.
        scale: Rating points per factor of ``base`` in the odds (400 is Elo-style).
        base: Odds base for the rating scale.
        init_rating: Rating assigned to a competitor with a zero fitted score.
        hessian_reg: Ridge added to the Hessian diagonal before inverting it. The BT
            objective is invariant to a constant shift, so the raw Hessian is singular.
        max_iter, ftol, gtol: L-BFGS-B stopping criteria, see ``lbfgs_minimize``.
        verbose: Print a summary after each fit.
    """

    n_competitors: int
    scale: float = 400.0
    base: float = 10.0
    init_rating: float = 1000.0
    hessian_reg: float = 1e-5
    max_iter: int = 1000
    ftol: float = 1e-9
    gtol: float = 1e-9
    verbose: bool = False

    def __post_init__(self):
        self.alpha = self.scale / math.log(self.base)
        self.ratings = np.zeros(self.n_competitors)
        self.fitted = False

    def rescale(self, ratings: np.ndarray) -> np.ndarray:
        """Maps fitted log-odds scores onto the reported rating scale."""
        return ratings * self.alpha + self.init_rating

    @staticmethod
    def loss_and_grad(
        ratings: np.ndarray, dataset: PairDataset
    ) -> tuple[float, np.ndarray]:
        """Weight-averaged negative log-likelihood and its gradient wrt ``ratings``."""
        idx_a, idx_b = dataset.pairs[:, 0], dataset.pairs[:, 1]
        weights, outcomes = dataset.opt_weights, dataset.outcomes
        rating_diffs = ratings[idx_a] - ratings[idx_b]
        weight_sum = np.sum(weights)

        # np.logaddexp(0, x) is softplus, computed stably
        loss = (
            -np.sum(
                weights * (outcomes * rating_diffs - np.logaddexp(0.0, rating_diffs))
            )
            / weight_sum
        )
        # d(loss)/d(rating_diff), scattered onto the two competitors of each pair
        resid = -weights * (outcomes - expit(rating_diffs)) / weight_sum
        n = len(ratings)
        grad = np.bincount(idx_a, resid, n) - np.bincount(idx_b, resid, n)
        return loss, grad

    @staticmethod
    def loss_function(ratings: np.ndarray, dataset: PairDataset) -> float:
        """The loss alone, for inspection; ``fit`` goes through ``loss_and_grad``."""
        return BradleyTerry.loss_and_grad(ratings, dataset)[0]

    def solve(self, dataset: PairDataset, initial_ratings: np.ndarray) -> np.ndarray:
        """Returns the ratings minimizing the loss on ``dataset``, leaving ``self`` alone."""
        ratings, _ = lbfgs_minimize(
            lambda r: self.loss_and_grad(r, dataset),
            initial_ratings,
            max_iter=self.max_iter,
            gtol=self.gtol,
            ftol=self.ftol,
            verbose=self.verbose,
        )
        return ratings

    def fit(self, dataset: PairDataset) -> "BradleyTerry":
        """Fits ``self.ratings`` on ``dataset``, warm-starting from their current value."""
        self.ratings = self.solve(dataset, self.ratings)
        self.fitted = True
        return self

    def hessian_and_grad_cov(
        self, dataset: PairDataset, hessian_reg: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Builds the two matrices the sandwich estimator needs, at the fitted ratings.

        Returns:
            ``(hessian, grad_cov)``: the loss Hessian with ``hessian_reg`` added to its
            diagonal, and the covariance of the per-pair gradient contributions.
        """
        idx_a, idx_b = dataset.pairs[:, 0], dataset.pairs[:, 1]
        probs = expit(self.ratings[idx_a] - self.ratings[idx_b])
        grad_vals = (dataset.outcomes - probs) * dataset.opt_weights
        hess_vals = probs * (1.0 - probs) * dataset.opt_weights

        n = self.n_competitors
        hessian = pairwise_matrix(hess_vals, idx_a, idx_b, n) + np.eye(n) * hessian_reg
        # variance per aggregated row, hence the division by its effective count
        grad_cov = pairwise_matrix(grad_vals**2 / dataset.ess_counts, idx_a, idx_b, n)
        return hessian, grad_cov

    def sandwich_cis(
        self, dataset: PairDataset, significance_level: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Central-limit confidence intervals from the sandwich covariance estimator."""
        # Unweighted data makes the loss a plain mean over battles, shrinking the Hessian
        # by a factor of N; scale the ridge by N so it stays equally weak either way.
        is_unweighted = np.allclose(dataset.weights, 1.0)
        reg_factor = np.sum(dataset.counts) if is_unweighted else 1.0
        hessian, grad_cov = self.hessian_and_grad_cov(
            dataset, self.hessian_reg * reg_factor
        )

        hessian_inv = np.linalg.inv(hessian)
        variances = np.diag(hessian_inv @ grad_cov @ hessian_inv)
        interval_widths = norm.ppf(1 - significance_level / 2) * np.sqrt(variances)
        return (
            self.rescale(self.ratings - interval_widths),
            self.rescale(self.ratings + interval_widths),
            variances * (self.alpha**2),
        )

    def bootstrap_cis(
        self,
        dataset: PairDataset,
        significance_level: float,
        num_bootstrap: int,
        seed: int,
        n_jobs: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Percentile confidence intervals from refitting resampled datasets."""
        total_battles = np.sum(dataset.counts)
        rng = np.random.default_rng(seed)
        boot_counts = rng.multinomial(
            int(total_battles), dataset.counts / total_battles, size=num_bootstrap
        )

        worker_args = [(counts, self, dataset) for counts in boot_counts]
        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        if n_jobs == 1:
            samples = [_fit_bootstrap_sample(*args) for args in worker_args]
        else:
            with mp.Pool(processes=n_jobs) as pool:
                samples = pool.starmap(_fit_bootstrap_sample, worker_args)

        # [num_bootstrap, n_competitors]
        scaled_samples = self.rescale(np.stack(samples))
        return (
            np.quantile(scaled_samples, significance_level / 2.0, axis=0),
            np.quantile(scaled_samples, 1.0 - significance_level / 2.0, axis=0),
            np.var(scaled_samples, axis=0),
        )

    def compute_ratings_and_cis(
        self,
        dataset: PairDataset,
        significance_level: float = 0.05,
        ci_method: str = "sandwich",
        num_bootstrap: int = 100,
        seed: int = 42,
        n_jobs: int = -1,
    ) -> dict[str, Any]:
        """Fits the model if needed, then returns rescaled ratings with intervals.

        Args:
            significance_level: 0.05 gives 95% intervals.
            ci_method: ``"sandwich"`` for the central-limit estimate, ``"bootstrap"`` for
                resampling.
            num_bootstrap, seed, n_jobs: Bootstrap-only. ``n_jobs=1`` runs the refits
                sequentially, anything else uses that many worker processes (-1 for all
                cores); results are identical either way for a given ``seed``.
        """
        if not self.fitted:
            self.fit(dataset)

        if ci_method == "sandwich":
            lower, upper, variances = self.sandwich_cis(dataset, significance_level)
        elif ci_method == "bootstrap":
            lower, upper, variances = self.bootstrap_cis(
                dataset, significance_level, num_bootstrap, seed, n_jobs
            )
        else:
            raise ValueError(f"Unknown ci_method: {ci_method}")

        return {
            "competitors": dataset.competitors,
            "ratings": self.rescale(self.ratings),
            "rating_lower": lower,
            "rating_upper": upper,
            "variances": variances,
        }
