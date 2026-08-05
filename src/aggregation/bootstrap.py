import statistics
from multiprocessing import Pool
from typing import Any, Sequence

import numpy as np

from src.aggregation.probabilistic_accuracy import (
    expected_balanced_accuracy_score,
)


def normalize_scores(scores: float, min_score: float, max_score: float) -> float:
    """Normalize a score to [0, 1] given min and max, floor at 0.

    Args:
        scores (float): Raw score.
        min_score (float): Minimum expected score (e.g., random chance).
        max_score (float): Maximum possible score.

    Returns:
        float: Normalized score in [0, 1].
    """
    normalized_scores = max((scores - min_score) / (max_score - min_score), 0)
    return normalized_scores


def stratified_bootstrap_index(
    labels: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Resample question positions with replacement *within* each class.

    Every replicate therefore keeps the class supports of the original sample, which an iid
    bootstrap does not: with 4 minority questions out of 67, about 1.6% of iid replicates
    contain none of them at all.

    Balanced accuracy is a macro average over per-class recalls and is by construction
    independent of class prevalence, so holding the supports fixed removes noise rather
    than hiding it.

    Args:
        labels (np.ndarray): Per-question true labels, shape (n_samples,).
        generator (np.random.Generator): Random number generator for reproducibility.

    Returns:
        np.ndarray: Positions into `labels`, shape (n_samples,), grouped by class.
    """
    return np.concatenate(
        [
            generator.choice(positions, size=positions.size, replace=True)
            for positions in (
                np.flatnonzero(labels == label) for label in np.unique(labels)
            )
        ]
    )


def run_bootstraps(
    scores: Sequence[float],
    labels: Sequence[Any],
    is_multi_choice: bool,
    native_max: float,
    seeds: Sequence[int],
    pool: Pool,
) -> list[float]:
    """Run bootstrap samples and return the mean scores.

    Args:
        scores (Sequence[float]): Per-question scores for this task, in native units.
        labels (Sequence[Any]): True labels for multi-choice tasks.
        is_multi_choice (bool): Whether the task is multi-choice.
        native_max (float): The metric's native maximum, from `get_native_max`.
        seeds (Sequence[int]): Random seeds for reproducibility.

    Returns:
        list[float]: Mean scores for each bootstrap sample.
    """
    return pool.starmap(
        run_single_bootstrap,
        [
            (
                scores,
                labels,
                is_multi_choice,
                native_max,
                np.random.default_rng(seed),
            )
            for seed in seeds
        ],
    )


def run_single_bootstrap(
    scores: Sequence[float],
    labels: Sequence[Any],
    is_multi_choice: bool,
    native_max: float,
    generator: np.random.Generator,
) -> float:
    """Run a single bootstrap sample and return the mean score.

    Args:
        scores (Sequence[float]): Per-question scores for this task, in native units.
        labels (Sequence[Any]): True labels for multi-choice tasks.
        is_multi_choice (bool): Whether the task is multi-choice.
        native_max (float): The metric's native maximum, from `get_native_max`; both the
            0-1 rescaling and the 0-100 scale-up are derived from it here. It must come
            from the metric name and never from the resampled values: deriving it from the
            data (`max(scores) <= 1`) made the scale flip between replicates whenever a
            replicate happened to hold no value above 1, mixing 1x and 100x replicates
            inside a single task.
        generator (np.random.Generator): Random number generator for reproducibility.

    Returns:
        float: Mean score for this bootstrap sample.
    """
    if is_multi_choice:
        # stratified: balanced accuracy is a macro average over classes, so the class
        # supports are held fixed rather than resampled (see stratified_bootstrap_index)
        labels_array = np.array(labels)
        if labels_array.size != len(scores):
            # Stratifying indexes by class, so the index is sized by the labels rather
            # than the scores. Mismatched lengths would silently mis-pair the two (an iid
            # index raised IndexError instead), so refuse outright. This happens when
            # individual_scores stores a list per question but only one label is read.
            raise ValueError(
                f"A multi-choice task needs one label per score, got "
                f"{labels_array.size} labels for {len(scores)} scores."
            )
        bootstrap_index = stratified_bootstrap_index(labels_array, generator)
    else:
        bootstrap_index = generator.integers(0, len(scores), size=len(scores))

    bootstrap_scores = np.array(scores)[bootstrap_index]

    if is_multi_choice:
        bootstrap_labels = labels_array[bootstrap_index]
        # rescale the bootstrap scores to [0, 1] before computing the expected balanced accuracy
        confidence = bootstrap_scores / native_max
        # y_pred is the same as y_true because we are calculating the expected balanced accuracy
        # score based on the probabilities of correct answers
        bootstrap_accuracy = expected_balanced_accuracy_score(
            bootstrap_labels,
            bootstrap_labels,
            confidence=confidence,
        )
        len_labels = len(set(labels))
        bootstrap_mean = normalize_scores(bootstrap_accuracy, 1 / len_labels, 1) * 100

    else:
        bootstrap_mean = statistics.mean(bootstrap_scores)
        # no-op for non-multi-choice tasks, but we still normalize to [0, 100] for consistency
        bootstrap_mean = normalize_scores(bootstrap_mean, 0, 1)

        # Put the score on a 0-100 scale using the metric's declared native range.
        bootstrap_mean = bootstrap_mean * (100.0 / native_max)
        bootstrap_mean = bootstrap_mean.tolist()

    return bootstrap_mean
