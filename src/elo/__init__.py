"""Bradley-Terry ratings from pairwise comparisons, in NumPy/SciPy.

The rating model and its pairwise dataset are re-exported here. `BTDataLoader`, which builds
those datasets from a results tree, is imported from `src.elo.bt_dataloader` directly so that
reading ratings does not pull in the loader's pandas/yaml/tqdm dependencies.
"""

from .bradley_terry import BradleyTerry, lbfgs_minimize, pairwise_matrix
from .data_utils import (
    TIE_OUTCOME,
    PairDataset,
    default_outcome_map,
    get_matchups_and_competitors,
)

__all__ = [
    "TIE_OUTCOME",
    "BradleyTerry",
    "PairDataset",
    "default_outcome_map",
    "get_matchups_and_competitors",
    "lbfgs_minimize",
    "pairwise_matrix",
]
