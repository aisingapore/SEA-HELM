"""Probability-aware extension of ``sklearn.metrics.balanced_accuracy_score``.

sklearn's balanced accuracy consumes hard predictions: it builds a confusion
matrix and averages the per-class recall read off its diagonal. A model that
picks the correct class with probability 0.99 and one that scrapes through at
0.34 therefore score identically.

The functions here replace the hard confusion matrix with its *expected*
counterpart under the model's predictive distribution,

    C[i, j] = sum_n w_n * 1[y_n == labels[i]] * P[n, j]

so each sample contributes the probability mass it placed on every class rather
than a single count. The per-class score becomes an expected recall

    recall_i = sum_{n: y_n == i} w_n * P[n, i] / sum_{n: y_n == i} w_n

and balanced accuracy remains their unweighted mean. Feeding one-hot
probabilities reproduces ``sklearn.metrics.balanced_accuracy_score`` exactly, so
this is a strict generalisation.

Note that the denominator is the class *support* and not the row sum of ``C``.
Rows whose probabilities sum to less than one (an abstention, e.g. a null or
unparseable response) therefore lose credit instead of being renormalised.
"""

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

# tolerance for probability range / simplex checks against float round-off
_TOL = 1e-6

Remainder = Literal["uniform", "abstain"]


def _check_same_length(name_a: str, n_a: int, name_b: str, n_b: int) -> None:
    """Raise if two inputs that must be row-aligned disagree on their sample count."""
    if n_a != n_b:
        raise ValueError(
            f"{name_a} and {name_b} have inconsistent lengths: {n_a} vs {n_b}."
        )


def _check_labels_not_empty(labels: np.ndarray) -> None:
    """Raise if there is no class to score."""
    if labels.size == 0:
        raise ValueError("labels must not be empty.")


def _warn_abstentions(count: int, subject: str) -> None:
    """Warn that some rows carry no usable distribution and so earn no credit.

    Args:
        count (int): Number of affected rows.
        subject (str): What was wrong with them, phrased as a clause that follows the
            count and precedes "and are treated as abstentions".
    """
    warnings.warn(
        f"{count} {subject} and are treated as abstentions (zero probability for "
        "every class).",
        stacklevel=3,
    )


def _weighted_indicator(
    y_true: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray
) -> np.ndarray:
    """Return the one-hot encoding of ``y_true`` over ``labels``, scaled by the weights.

    Column ``j`` holds each sample's weight where its true class is ``labels[j]`` and
    zero elsewhere. Summing a column therefore gives that class's support, and
    multiplying against a probability matrix gives the mass its samples placed on each
    class -- the two quantities both the confusion matrix and the recall are built from.

    Args:
        y_true (np.ndarray): Ground truth labels of shape (n_samples,).
        labels (np.ndarray): The classes indexing the columns.
        sample_weight (np.ndarray): Per-sample weights of shape (n_samples,).

    Returns:
        np.ndarray: Weighted indicator of shape (n_samples, n_classes).
    """
    indicator = (y_true[:, None] == labels[None, :]).astype(float)
    return indicator * sample_weight[:, None]


def probabilities_from_confidence(
    y_pred: npt.ArrayLike,
    confidence: npt.ArrayLike,
    *,
    labels: npt.ArrayLike,
    remainder: Remainder = "uniform",
) -> np.ndarray:
    """Turn hard predictions plus a per-sample confidence into a class distribution.

    Use this when the model only exposes a probability for the label it actually
    emitted (e.g. ``exp(cumulative_logprob)`` of the extracted answer) rather
    than a distribution over all classes.

    Args:
        y_pred (npt.ArrayLike): Predicted labels of shape (n_samples,).
        confidence (npt.ArrayLike): Probability assigned to each predicted label,
            shape (n_samples,), each in [0, 1]. ``NaN`` is read as a full
            abstention (a row of zeros), matching how null responses surface as
            missing logprobs.
        labels (npt.ArrayLike): The classes indexing the columns of the result.
        remainder (Remainder, optional): How to spread the leftover ``1 - p``
            mass. ``"uniform"`` spreads it evenly over the other classes -- the
            maximum-entropy reading of "the model was only p sure". ``"abstain"``
            discards it, which makes the score a lower bound. With a single class
            there is nowhere to spread it and both behave alike. Defaults to
            ``"uniform"``.

    Returns:
        np.ndarray: Probabilities of shape (n_samples, n_classes), aligned to
            ``labels``.

    Raises:
        ValueError: If lengths disagree, ``labels`` is empty, ``confidence``
            falls outside [0, 1], or ``remainder`` is unknown.
    """
    if remainder not in ("uniform", "abstain"):
        raise ValueError(
            f"remainder must be 'uniform' or 'abstain', got {remainder!r}."
        )

    labels = np.asarray(labels)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=float)

    _check_labels_not_empty(labels)
    _check_same_length("y_pred", y_pred.shape[0], "confidence", confidence.shape[0])

    is_null = np.isnan(confidence)
    if np.any(is_null):
        _warn_abstentions(int(is_null.sum()), "confidence value(s) are NaN")
    finite = np.where(is_null, 0.0, confidence)
    if np.any((finite < -_TOL) | (finite > 1 + _TOL)):
        raise ValueError("confidence must lie in [0, 1].")
    finite = np.clip(finite, 0.0, 1.0)

    n_samples, n_classes = y_pred.shape[0], labels.shape[0]
    label_to_column = {label: column for column, label in enumerate(labels)}
    columns = np.array([label_to_column.get(label, -1) for label in y_pred])

    # A prediction outside `labels` (e.g. the null label) carries no recoverable
    # distribution, so it abstains entirely regardless of `remainder`.
    unknown = columns < 0
    if np.any(unknown):
        _warn_abstentions(int(unknown.sum()), "prediction(s) are not in labels")
    abstained = unknown | is_null

    proba = np.zeros((n_samples, n_classes), dtype=float)
    if remainder == "uniform" and n_classes > 1:
        # (1 - p) spread over the n_classes - 1 classes that were not predicted
        proba[~abstained] = ((1 - finite[~abstained]) / (n_classes - 1))[:, None]
    rows = np.flatnonzero(~abstained)
    proba[rows, columns[rows]] = finite[rows]
    return proba


def _as_proba_matrix(
    y_proba: npt.ArrayLike,
    labels: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Validate ``y_proba`` and return it as a dense (n_samples, n_classes) array.

    Args:
        y_proba (npt.ArrayLike): Predicted probabilities. A 2D array aligned to
            ``labels``, a DataFrame whose columns are the labels, or -- for two
            classes only -- a 1D array holding P(``labels[1]``).
        labels (np.ndarray): The classes indexing the columns.
        n_samples (int): Expected number of rows.

    Returns:
        np.ndarray: Column-aligned probabilities of shape (n_samples, n_classes),
            with ``NaN`` rows zeroed out.

    Raises:
        ValueError: If the shape, column labels, or probability values are invalid.
    """
    if isinstance(y_proba, pd.DataFrame):
        missing = [label for label in labels if label not in y_proba.columns]
        if missing:
            raise ValueError(
                f"y_proba is missing a column for the following labels: {missing}."
            )
        y_proba = y_proba.loc[:, labels].to_numpy(dtype=float)

    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        if labels.shape[0] != 2:
            raise ValueError(
                "1D y_proba is only supported for 2 classes (read as the "
                f"probability of labels[1]); got {labels.shape[0]} classes. Pass "
                "a (n_samples, n_classes) array instead."
            )
        proba = np.column_stack([1 - proba, proba])
    if proba.ndim != 2:
        raise ValueError(f"y_proba must be 1D or 2D, got {proba.ndim} dimensions.")
    if proba.shape != (n_samples, labels.shape[0]):
        raise ValueError(
            "y_proba has shape "
            f"{proba.shape}, expected {(n_samples, labels.shape[0])}. Its "
            "columns must be aligned to `labels`."
        )

    # A NaN anywhere in a row means no usable distribution: abstain for that row.
    null_rows = np.isnan(proba).any(axis=1)
    if np.any(null_rows):
        _warn_abstentions(int(null_rows.sum()), "row(s) of y_proba contain NaN")
        proba = np.where(null_rows[:, None], 0.0, proba)

    if np.any(proba < -_TOL):
        raise ValueError("y_proba must not contain negative probabilities.")
    row_sums = proba.sum(axis=1)
    if np.any(row_sums > 1 + _TOL):
        raise ValueError(
            "y_proba rows must sum to at most 1; the largest row sum is "
            f"{row_sums.max():.6f}."
        )
    return np.clip(proba, 0.0, None)


def _resolve_inputs(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike | None,
    y_proba: npt.ArrayLike | None,
    confidence: npt.ArrayLike | None,
    labels: npt.ArrayLike | None,
    sample_weight: npt.ArrayLike | None,
    remainder: Remainder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalise the accepted input combinations into arrays.

    Args:
        y_true (npt.ArrayLike): Ground truth labels.
        y_pred (npt.ArrayLike | None): Predicted labels, if any.
        y_proba (npt.ArrayLike | None): Predicted probabilities, if any.
        confidence (npt.ArrayLike | None): Per-sample confidence in ``y_pred``.
        labels (npt.ArrayLike | None): Classes indexing the probability columns.
        sample_weight (npt.ArrayLike | None): Sample weights.
        remainder (Remainder): Passed through to
            :func:`probabilities_from_confidence`.

    Returns:
        tuple: ``(y_true, labels, proba, sample_weight)`` as arrays.

    Raises:
        ValueError: If the input combination is invalid or lengths disagree.
    """
    if (y_pred is None) == (y_proba is None):
        raise ValueError("Pass exactly one of y_pred or y_proba.")
    if y_proba is not None and confidence is not None:
        raise ValueError(
            "confidence only applies to y_pred; y_proba already carries the "
            "probabilities."
        )

    y_true = np.asarray(y_true)
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got {y_true.ndim} dimensions.")
    n_samples = y_true.shape[0]

    if labels is not None:
        labels = np.asarray(labels)
        _check_labels_not_empty(labels)
        if labels.size != np.unique(labels).size:
            raise ValueError("labels must not contain duplicates.")
        unseen = set(np.unique(y_true).tolist()) - set(labels.tolist())
        if unseen:
            raise ValueError(
                f"y_true contains labels not present in `labels`: {sorted(unseen)}."
            )
    elif isinstance(y_proba, pd.DataFrame):
        labels = np.asarray(y_proba.columns)
    elif y_pred is not None:
        # mirrors sklearn: the label set is the union of truth and prediction
        labels = np.unique(np.concatenate([y_true, np.asarray(y_pred).ravel()]))
    else:
        labels = np.unique(y_true)

    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float).ravel()
        _check_same_length("y_true", n_samples, "sample_weight", sample_weight.shape[0])
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must not be negative.")

    if y_pred is not None:
        y_pred = np.asarray(y_pred)
        _check_same_length("y_true", n_samples, "y_pred", y_pred.shape[0])
        if confidence is None:
            # hard predictions: one-hot rows recover the sklearn behaviour
            confidence = np.ones(n_samples)
        proba = probabilities_from_confidence(
            y_pred, confidence, labels=labels, remainder=remainder
        )
    else:
        proba = _as_proba_matrix(y_proba, labels, n_samples)

    return y_true, labels, proba, sample_weight


def expected_confusion_matrix(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike | None = None,
    *,
    y_proba: npt.ArrayLike | None = None,
    confidence: npt.ArrayLike | None = None,
    labels: npt.ArrayLike | None = None,
    sample_weight: npt.ArrayLike | None = None,
    remainder: Remainder = "uniform",
) -> np.ndarray:
    """Compute the expected (soft) confusion matrix under the predictive distribution.

    ``C[i, j]`` is the weighted probability mass that samples of true class
    ``labels[i]`` put on class ``labels[j]``. With one-hot probabilities this is
    exactly ``sklearn.metrics.confusion_matrix``.

    Args:
        y_true (npt.ArrayLike): Ground truth labels of shape (n_samples,).
        y_pred (npt.ArrayLike | None, optional): Predicted labels. Mutually
            exclusive with ``y_proba``. Defaults to None.
        y_proba (npt.ArrayLike | None, optional): Predicted probabilities of
            shape (n_samples, n_classes) aligned to ``labels``, a DataFrame keyed
            by label, or a 1D array of P(``labels[1]``) in the two-class case.
            Defaults to None.
        confidence (npt.ArrayLike | None, optional): Probability of each
            ``y_pred``; see :func:`probabilities_from_confidence`. Defaults to
            None, i.e. hard predictions.
        labels (npt.ArrayLike | None, optional): Classes to index rows and
            columns. Defaults to the labels observed in the inputs.
        sample_weight (npt.ArrayLike | None, optional): Sample weights. Defaults
            to None.
        remainder (Remainder, optional): How ``confidence`` leftover mass is
            spread. Defaults to ``"uniform"``.

    Returns:
        np.ndarray: Matrix of shape (n_classes, n_classes).
    """
    y_true, labels, proba, sample_weight = _resolve_inputs(
        y_true, y_pred, y_proba, confidence, labels, sample_weight, remainder
    )
    return _weighted_indicator(y_true, labels, sample_weight).T @ proba


def expected_recall_score(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike | None = None,
    *,
    y_proba: npt.ArrayLike | None = None,
    confidence: npt.ArrayLike | None = None,
    labels: npt.ArrayLike | None = None,
    sample_weight: npt.ArrayLike | None = None,
    remainder: Remainder = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the per-class expected recall and the classes it was computed for.

    Classes absent from ``y_true`` have no support and are dropped, mirroring how
    ``balanced_accuracy_score`` handles them.

    Args:
        y_true (npt.ArrayLike): Ground truth labels of shape (n_samples,).
        y_pred (npt.ArrayLike | None, optional): Predicted labels. Defaults to None.
        y_proba (npt.ArrayLike | None, optional): Predicted probabilities.
            Defaults to None.
        confidence (npt.ArrayLike | None, optional): Probability of each
            ``y_pred``. Defaults to None.
        labels (npt.ArrayLike | None, optional): Classes to score. Defaults to
            the labels observed in the inputs.
        sample_weight (npt.ArrayLike | None, optional): Sample weights. Defaults
            to None.
        remainder (Remainder, optional): How ``confidence`` leftover mass is
            spread. Defaults to ``"uniform"``.

    Returns:
        tuple[np.ndarray, np.ndarray]: The expected recall per class and the
            corresponding labels, both of shape (n_present_classes,).
    """
    y_true, labels, proba, sample_weight = _resolve_inputs(
        y_true, y_pred, y_proba, confidence, labels, sample_weight, remainder
    )
    weighted = _weighted_indicator(y_true, labels, sample_weight)

    # numerator: mass placed on the true class; denominator: class support. This is the
    # diagonal of `expected_confusion_matrix` over its row sums, computed elementwise
    # rather than through the matmul so the two share no floating-point summation order.
    hit = (weighted * proba).sum(axis=0)
    support = weighted.sum(axis=0)

    present = support > 0
    if not np.any(present):
        raise ValueError("No class in `labels` has any support in y_true.")
    if not np.all(present):
        warnings.warn(
            "The following labels are not present in y_true and are excluded "
            f"from the average: {labels[~present].tolist()}",
            stacklevel=2,
        )
    return hit[present] / support[present], labels[present]


def expected_balanced_accuracy_score(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike | None = None,
    *,
    y_proba: npt.ArrayLike | None = None,
    confidence: npt.ArrayLike | None = None,
    labels: npt.ArrayLike | None = None,
    sample_weight: npt.ArrayLike | None = None,
    adjusted: bool = False,
    remainder: Remainder = "uniform",
) -> float:
    """Compute the balanced accuracy, optionally weighted by prediction probability.

    Balanced accuracy is the unweighted mean of the per-class recalls. This
    version averages *expected* recalls instead: a sample of class ``c``
    contributes the probability the model assigned to ``c`` rather than 0 or 1.
    Passing only ``y_pred`` (hard labels) reproduces
    ``sklearn.metrics.balanced_accuracy_score``.

    The best value is 1 and the worst 0 when ``adjusted=False``. A model
    predicting a uniform distribution scores ``1 / n_classes``, so the
    chance-adjusted rescaling is unchanged from sklearn's.

    When ``confidence`` is a sample's fraction of correct runs, the score is identical
    to running sklearn over all the runs pooled: balanced accuracy is linear in the
    per-sample indicator and every run shares the same class supports, so averaging
    runs into a probability first results in the same value.

    This function allows for an accuracy score to be calculated directed from the
    model's logits or probabilities, rather than the hard predictions.

    Args:
        y_true (npt.ArrayLike): Ground truth labels of shape (n_samples,).
        y_pred (npt.ArrayLike | None, optional): Predicted labels of shape
            (n_samples,). Mutually exclusive with ``y_proba``. Combine with
            ``confidence`` when only the emitted label's probability is known.
            Defaults to None.
        y_proba (npt.ArrayLike | None, optional): Predicted probabilities of
            shape (n_samples, n_classes) aligned to ``labels``, a DataFrame keyed
            by label, or a 1D array of P(``labels[1]``) in the two-class case.
            Rows may sum to less than 1, in which case the missing mass earns no
            credit. Defaults to None.
        confidence (npt.ArrayLike | None, optional): Probability assigned to each
            ``y_pred``, shape (n_samples,). ``NaN`` counts as a full abstention.
            Defaults to None, i.e. treat ``y_pred`` as certain.
        labels (npt.ArrayLike | None, optional): Classes to score, and the order
            of the ``y_proba`` columns. Defaults to the labels observed in the
            inputs.
        sample_weight (npt.ArrayLike | None, optional): Sample weights. Defaults
            to None.
        adjusted (bool, optional): When True, rescale so that chance performance
            scores 0 while perfect performance stays at 1. Defaults to False.
        remainder (Remainder, optional): How to spread the ``1 - confidence``
            mass across the classes that were not predicted: ``"uniform"``
            (maximum entropy) or ``"abstain"`` (discard it, giving a lower
            bound). Ignored when ``y_proba`` is given. Defaults to ``"uniform"``.

    Returns:
        float: The (expected) balanced accuracy score.

    Raises:
        ValueError: If the input combination, shapes, or probability values are
            invalid.

    Examples:
        Hard labels behave exactly like sklearn::

            >>> y_true = [0, 1, 0, 0, 1, 0]
            >>> y_pred = [0, 1, 0, 0, 0, 1]
            >>> expected_balanced_accuracy_score(y_true, y_pred)
            0.625

        Hedging costs credit -- the same calls made less confidently score lower::

            >>> round(expected_balanced_accuracy_score(
            ...     y_true, y_pred, confidence=[0.9] * 6), 4)
            0.6
            >>> round(expected_balanced_accuracy_score(
            ...     y_true, y_pred, confidence=[0.4] * 6), 4)
            0.475

        Full distributions are scored directly::

            >>> import numpy as np
            >>> proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
            >>> round(expected_balanced_accuracy_score(
            ...     [0, 1, 0], y_proba=proba, labels=[0, 1]), 4)
            0.7
    """
    per_class, _ = expected_recall_score(
        y_true,
        y_pred,
        y_proba=y_proba,
        confidence=confidence,
        labels=labels,
        sample_weight=sample_weight,
        remainder=remainder,
    )
    score = float(per_class.mean())
    if adjusted:
        chance = 1 / per_class.shape[0]
        score = (score - chance) / (1 - chance)
    return score
