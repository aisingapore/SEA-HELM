"""Dataset container for pairwise-comparison (Bradley-Terry) data."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

# A drawn comparison: equally likely either way, so it contributes half credit to each side.
# This is the exact value `keep_ties=False` filters on, so anything that should be treated as a
# draw has to map to it rather than merely close to it.
TIE_OUTCOME = 0.5

# Ceiling on the `(a, b, outcome)` key space `from_indices` will count densely rather than
# sort. One int64 bin per key, so this is ~400 MB at the limit and reached only past ~8,000
# competitors, an order of magnitude more than any leaderboard here holds.
_MAX_AGGREGATION_BINS = 50_000_000

OUTCOME_MAP = {
    "model_a": 1.0,
    "model_b": 0.0,
    "tie": TIE_OUTCOME,
    "both_bad": TIE_OUTCOME,
}


def default_outcome_map(outcome: str) -> float:
    """Maps a str winner value to a float outcome; anything unrecognised scores as a tie."""
    return OUTCOME_MAP.get(outcome, TIE_OUTCOME)


def get_matchups_and_competitors(
    df, competitor_cols: Sequence[str] = ("model_a", "model_b")
) -> tuple[np.ndarray, list[str]]:
    """Maps the two str competitor columns to integer indices into a sorted competitor list."""
    n_rows = len(df)
    competitor_indices, competitors = pd.factorize(
        pd.concat([df[competitor_cols[0]], df[competitor_cols[1]]]), sort=True
    )
    competitor_indices = np.asarray(competitor_indices, dtype=np.int32)
    matchups = np.column_stack(
        [competitor_indices[:n_rows], competitor_indices[n_rows:]]
    )
    return matchups, competitors.tolist()


@dataclass(eq=False)
class PairDataset:
    """Pairwise comparisons, aggregated by (competitor a, competitor b, outcome) triplet.

    All array fields are indexed by aggregated row, i.e. have length ``n_pairs``.

    Attributes:
        competitors: Competitor names; index into this list is the competitor id.
        pairs: Competitor id of a and b, shape (n_pairs, 2).
        outcomes: Outcome in [0, 1], from a's point of view.
        counts: Raw number of rows behind each aggregate. Drives the bootstrap.
        weights: Mean per-row weight of the aggregate. All 1.0 for unweighted data,
            which is what the sandwich estimator checks to scale its ridge.
        opt_weights: Summed per-row weight of the aggregate. The loss weights.
        ess_counts: Kish effective count ``(sum w)^2 / (sum w^2)`` per aggregate; equals
            ``counts`` when row weights are constant within the aggregate.
    """

    competitors: list[str]
    pairs: np.ndarray
    outcomes: np.ndarray
    counts: np.ndarray
    weights: np.ndarray
    opt_weights: np.ndarray
    ess_counts: np.ndarray

    @property
    def n_pairs(self) -> int:
        return len(self.outcomes)

    @property
    def n_competitors(self) -> int:
        return len(self.competitors)

    @classmethod
    def from_pandas(
        cls,
        df,
        competitor_cols: Sequence[str] = ("model_a", "model_b"),
        outcome_col: str = "winner",
        outcome_map: Callable[[str], float] = default_outcome_map,
        reweighted: bool = False,
        min_pair_count: int = 50,
        row_weight_col: Optional[str] = None,
        keep_ties: bool = True,
    ) -> "PairDataset":
        """Builds a PairDataset from a dataframe of individual comparisons.

        Args:
            reweighted: Apply inverse-propensity weights so that heavily sampled pairs
                do not dominate the fit.
            min_pair_count: Floor on a pair's effective count in the IPW denominator,
                which caps how much a rarely seen pair can be upweighted.
            row_weight_col: Optional column of per-row weights in [0, 1] expressing each
                row's information value. When given, a pair's effective count becomes the
                sum of its row weights rather than its raw row count, so both reweighting
                layers work in the same "effective sample size" currency.
            keep_ties: Include drawn comparisons, those whose mapped outcome is exactly
                ``TIE_OUTCOME``. Kept by default, because a draw is evidence that two
                competitors are indistinguishable on an item rather than an absence of
                evidence; pass False to rate competitors on decisive comparisons alone.

                Dropped rows are removed *before* the rows are aggregated, so ``counts``,
                ``ess_counts`` and the inverse-propensity denominator are all computed over
                the decisive subset -- a pair's effective sample size should not count
                comparisons the fit never sees. That makes ``keep_ties=False`` identical to
                filtering the dataframe beforehand, including its effect on ``competitors``:
                a competitor whose every comparison was a draw drops out of the dataset
                rather than being handed a rating nothing identifies. Outcomes strictly
                between 0 and 1 that are not exactly ``TIE_OUTCOME`` are decisive and
                survive, so partial credit is not mistaken for a draw.

        Raises:
            ValueError: If dropping ties leaves no comparisons to fit.
        """
        outcomes = np.asarray(df[outcome_col].map(outcome_map).to_numpy(), dtype=float)
        if not keep_ties:
            decisive = outcomes != TIE_OUTCOME
            if not decisive.any():
                raise ValueError(
                    f"All {len(outcomes)} comparisons are ties, so keep_ties=False leaves "
                    "nothing to fit. Either keep them, or check that the outcome map "
                    f"resolves decisive rows to something other than {TIE_OUTCOME}."
                )
            df = df[decisive]
            outcomes = outcomes[decisive]

        matchups, competitors = get_matchups_and_competitors(df, competitor_cols)
        if row_weight_col is not None:
            row_weights = np.asarray(df[row_weight_col].to_numpy())
        else:
            row_weights = np.ones(len(df))

        # Aggregate identical (a, b, outcome) rows, tracking raw counts (for the
        # bootstrap), summed weights (loss and IPW) and summed squared weights (ESS).
        unique_rows, row_to_agg, counts = np.unique(
            np.column_stack([matchups, outcomes]),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        # numpy 2.0.0 returned a column vector here; later versions return 1-D
        row_to_agg = row_to_agg.reshape(-1)
        n_agg = len(unique_rows)
        sum_weights = np.bincount(row_to_agg, row_weights, n_agg)
        sum_sq_weights = np.bincount(row_to_agg, row_weights**2, n_agg)

        # An aggregate whose rows all have zero weight has no effective sample size; fall back to its count.
        has_weight = sum_sq_weights > 0
        ess_counts = np.where(
            has_weight,
            sum_weights**2 / np.where(has_weight, sum_sq_weights, 1.0),
            counts,
        )

        pairs = unique_rows[:, :2].astype(np.int32)
        if reweighted:
            # Effective count of an unordered pair = its summed row weights, shared by
            # every aggregate (i.e. every outcome) belonging to that pair.
            _, agg_to_pair = np.unique(
                np.sort(pairs, axis=1), axis=0, return_inverse=True
            )
            agg_to_pair = agg_to_pair.reshape(-1)
            eff_pair_counts = np.bincount(agg_to_pair, sum_weights)[agg_to_pair]
            pair_weights = 1.0 / np.maximum(eff_pair_counts, min_pair_count)
        else:
            pair_weights = np.ones(n_agg)

        return cls(
            competitors=competitors,
            pairs=pairs,
            outcomes=unique_rows[:, 2],
            counts=counts,
            weights=(sum_weights / counts) * pair_weights,
            opt_weights=sum_weights * pair_weights,
            ess_counts=ess_counts,
        )

    @classmethod
    def from_indices(
        cls,
        competitor_a,
        competitor_b,
        outcomes,
        competitor_names: Sequence[str],
        keep_ties: bool = True,
    ) -> "PairDataset":
        """Builds a PairDataset from integer competitor ids rather than a dataframe.

        Produces exactly what ``from_pandas`` produces for unweighted, un-reweighted data --
        same ``competitors``, same row order, same counts -- without ever materialising the
        competitor names per row. Millions of comparisons are the normal case here, and the
        dataframe route spends most of its time turning integers into strings and then
        sorting them back into integers: the names become Arrow string columns, ``factorize``
        maps them back to ids, and the ``(a, b, outcome)`` aggregation is a lexsort over a
        float64 matrix. This aggregates by counting a compound integer key instead.

        Args:
            competitor_a, competitor_b: Competitor ids per comparison, indices into
                ``competitor_names``.
            outcomes: Outcome in [0, 1] per comparison, from a's point of view. Already
                mapped, so there is no ``outcome_map`` to apply.
            competitor_names: Every competitor an id may refer to, **sorted by name**. Ids
                absent from the comparisons are dropped and the rest renumbered, so the
                result matches ``from_pandas``'s ``factorize(..., sort=True)`` -- which is
                what lets ratings be aligned by name across bootstrap replicates.
            keep_ties: As in ``from_pandas``, and applied at the same point: ties are dropped
                before the ids are renumbered, so a competitor whose every comparison was a
                draw leaves the dataset rather than being handed a rating nothing identifies.

        Raises:
            ValueError: If dropping ties leaves no comparisons to fit.
        """
        ids_a = np.asarray(competitor_a, dtype=np.int64)
        ids_b = np.asarray(competitor_b, dtype=np.int64)
        outcomes = np.asarray(outcomes, dtype=float)
        if not keep_ties:
            decisive = outcomes != TIE_OUTCOME
            if not decisive.any():
                raise ValueError(
                    f"All {len(outcomes)} comparisons are ties, so keep_ties=False leaves "
                    "nothing to fit. Either keep them, or check that the outcomes hold "
                    f"something other than {TIE_OUTCOME}."
                )
            ids_a, ids_b, outcomes = (
                ids_a[decisive],
                ids_b[decisive],
                outcomes[decisive],
            )

        # Renumber to the competitors actually present, keeping the order of
        # `competitor_names` -- sorted, so this is `factorize(sort=True)` on the names.
        present = np.zeros(len(competitor_names), dtype=bool)
        present[ids_a] = True
        present[ids_b] = True
        renumber = np.cumsum(present) - 1
        ids_a, ids_b = renumber[ids_a], renumber[ids_b]
        competitors = [
            name
            for name, keep in zip(competitor_names, present.tolist(), strict=True)
            if keep
        ]
        n_competitors = len(competitors)

        # Aggregate identical (a, b, outcome) rows. The distinct outcomes are a handful of
        # values -- 1, 0.5 and 0 for a tie-tolerant comparison -- so the triplet fits in one
        # integer key, and `bincount` groups it in a pass instead of a sort. The key is
        # ordered (a, b, outcome) with the outcome codes ascending in outcome value, which is
        # the row order `np.unique(..., axis=0)` produces in `from_pandas`.
        levels, codes = np.unique(outcomes, return_inverse=True)
        codes = codes.reshape(-1)
        n_bins = n_competitors * n_competitors * len(levels)
        if n_bins <= _MAX_AGGREGATION_BINS:
            keys = (ids_a * n_competitors + ids_b) * len(levels) + codes
            counted = np.bincount(keys, minlength=n_bins)
            unique_keys = np.flatnonzero(counted)
            counts = counted[unique_keys]
        else:
            # More competitors than the dense key space is worth; sort instead.
            keys = (ids_a * n_competitors + ids_b) * len(levels) + codes
            unique_keys, counts = np.unique(keys, return_counts=True)
        rows, outcome_codes = np.divmod(unique_keys, len(levels))
        pair_a, pair_b = np.divmod(rows, n_competitors)

        # Every row weighs 1, so the summed weight of an aggregate is its count and its Kish
        # effective count is its count as well.
        float_counts = counts.astype(float)
        return cls(
            competitors=competitors,
            pairs=np.column_stack([pair_a, pair_b]).astype(np.int32),
            outcomes=levels[outcome_codes],
            counts=counts,
            weights=np.ones(len(counts)),
            opt_weights=float_counts,
            ess_counts=float_counts,
        )
