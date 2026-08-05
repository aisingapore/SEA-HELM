"""Bradley-Terry pairwise datasets from SEA-HELM multi-run inference outputs.

Where `MultiRunAggregator` folds several runs of one model into absolute scores, and
`EloResultsLoader`/`EloOutcomes` sample head-to-head contests from individual runs, this
module does both at once for the rating model in `src.elo.bradley_terry`:

1. every item's score is averaged across the runs of a model, so a contest compares two
   run-averaged scores rather than two single-run draws;
2. contests are allocated across languages, competencies and tasks with the same
   equal-share-per-level scheme `EloOutcomes` uses, then sampled within each cell -- and
   when ties are dropped the share is granted in *decisive* contests, so a cell that ties
   on most of its comparisons is drawn from proportionally harder rather than quietly
   handing its weight to the cells that tie least (see `BTDataLoader.tie_budget`);
   `pair_budget` then says how a cell spends its share across the pairs inside it, which is
   what decides whether the *competitors* come out balanced as well as the cells;
3. two averaged scores closer than `tolerance` on a common [0, 1] scale are a tie.

The unit of work is a *cell*: one (task, language) pair. `load` reads every model's runs
into one run-averaged score matrix per cell and optionally caches them; `build` turns those
matrices into a `PairDataset` for one view of the data. Loading is the expensive half
(hundreds of GB of JSONL), so it happens once and every view, tolerance and bootstrap
replicate is derived from the cached matrices.

Only the `{leaf}_{task}_{lang}.jsonl` score files are read. Their `*_batch_response.jsonl`
neighbours hold raw generations with no `individual_scores` and are never opened.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd
import ujson
import yaml
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.aggregation.constants import METRIC_NATIVE_MAX, SEA_LANGUAGES  # noqa: E402

# Items are identified by their position in the score file so that the same item can be
# recognised across models and runs. A row holding a list of values contributes one item per
# element, packed into the low bits of the key.
_ELEM_BITS = 16
_MAX_ELEMS = 1 << _ELEM_BITS

# `run_config_<iso timestamp>.yaml`, as written by the eval pipeline.
_CONFIG_TIMESTAMP = re.compile(r"run_config_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)")

# Which levels get an equal share of the contest budget, outermost first. The task name is
# appended to every entry, so members of an aggregation group split that group's share.
_VIEW_LEVELS: dict[tuple[str, Optional[str]], tuple[str, ...]] = {
    ("global", "language"): ("lang", "competency", "task_node"),
    ("global", "competency"): ("competency", "lang", "task_node"),
    ("global", "task"): ("task_node", "lang"),
    ("sea", "language"): ("lang", "competency", "task_node"),
    ("sea", "competency"): ("competency", "lang", "task_node"),
    ("sea", "task"): ("task_node", "lang"),
    ("language", None): ("competency", "task_node"),
    ("competency", None): ("lang", "task_node"),
    ("task", None): ("lang",),
}

# A competitor whose sampled contest count falls this far below the median is reported: its
# rating is far less certain than the others' and the interval alone can understate that.
_THIN_COVERAGE_FRACTION = 0.1

# How far apart the realised contest counts may drift under `pair_budget="equal"` before it
# is reported. The split is exact by construction, so any spread means pairs were pinned to
# what they hold; a few percent is the rounding of the last contest in each cell.
_PAIR_BALANCE_SPREAD = 0.05

# How a cell's budget is spent when ties are dropped; see `BTDataLoader.tie_budget`.
_TIE_BUDGETS = ("none", "inflate", "decisive")

# How a cell's budget is spent across the pairs inside it; see `BTDataLoader.pair_budget`.
_PAIR_BUDGETS = ("pool", "equal")

# Floor on the decisive rate the tie-aware allocation divides by, so a cell that ties on
# almost everything cannot claim more than 1 / _MIN_DECISIVE_RATE times its base share and
# starve the rest of the view. A cell decisive on *nothing* gets no budget at all rather than
# the largest one: no number of draws from it survives into a tie-dropping fit.
_MIN_DECISIVE_RATE = 0.01

# Items per block when counting a cell's decisive comparisons. The intermediate is
# (block, pairs) wide, so this keeps it to a few tens of MB at ~4,300 pairs and speeds up the
# computation as compared to a single pass over the whole matrix.
_DECISIVE_CHUNK = 256

# Pairs per block when drawing a per-pair budget by ranking. That draw needs a whole column
# at once -- it ranks every item within a pair -- so it blocks the other way round from
# `_DECISIVE_CHUNK`, and the intermediate is (items, block) wide.
_PAIR_CHUNK = 256

# Share of what a pair holds above which its quota is drawn by ranking rather than by
# rejection; see `BTDataLoader._draw_per_pair`. Rejection is cheaper by two orders of
# magnitude at the few contests per pair a real budget grants, and degrades to coupon
# collecting only as the quota approaches the pool, so the split sits well away from both.
_DENSE_PAIR_FRACTION = 0.25

# Standard deviations of headroom a rejection round draws on top of its shortfall, so that the
# round finishes the pair even when its rejects run against it. A round costs a pass over every
# pair still short, so buying headroom in candidates -- two gathers and a compare each -- is
# far cheaper than buying it in rounds.
_DRAW_SIGMA = 3.0

# Candidates added on top of that, so a pair short by one or two draws more than a couple and
# does not depend on those surviving.
_DRAW_PAD = 4


def get_native_max(
    metric: Optional[str], metric_scales: Optional[Mapping[str, float]] = None
) -> float:
    """Return the native maximum of `metric`'s per-item scores.

    Args:
        metric: Metric key found under `individual_scores`, or None when the file stores a
            bare value with no metric name.
        metric_scales: Native maximum per metric key, extending or overriding
            `METRIC_NATIVE_MAX`.

    Returns:
        1.0 for a metric stored as a fraction in [0, 1], 100.0 for one already stored as a
        percentage.

    Raises:
        KeyError: If the metric's native scale is unknown. Fatal rather than inferred: the
            tie tolerance is applied to the rescaled score, so guessing 1.0 for a
            percentage metric would turn every genuine difference into a tie.
    """
    scales = dict(METRIC_NATIVE_MAX)
    if metric_scales:
        scales.update(metric_scales)
    if metric not in scales:
        raise KeyError(
            f"Unknown native scale for metric {metric!r}. Add it to METRIC_NATIVE_MAX "
            "(1.0 if individual_scores holds a fraction, 100.0 if it already holds a "
            f"percentage), or pass metric_scales={{{metric!r}: <native max>}}. "
            f"Known metrics: {sorted(k for k in scales if k)}."
        )
    return scales[metric]


class CellSpec(NamedTuple):
    """Identity and configuration of one (task, language) cell."""

    task: str
    lang: str
    competency: str
    aggregation_group: Optional[str]
    metric: Optional[str]

    @property
    def task_node(self) -> str:
        """The name this cell is balanced under: its group if it has one, else its task."""
        return self.aggregation_group or self.task

    @property
    def subfolder(self) -> str:
        """The inference subfolder holding this cell's score file."""
        return self.aggregation_group or self.task


class ModelSpec(NamedTuple):
    """One competitor and where its runs live.

    `competitor` leads with the model type because the same org/leaf appears under more than
    one sweep directory -- the instruct and reasoning sweeps of a model share a directory
    name and a score-file name, and are distinguished only by which sweep they sit in.
    """

    competitor: str
    model_type: str
    sweep_dir: str
    org: str
    leaf: str
    path: str


class _LoadRequest(NamedTuple):
    """One (model, cell) unit of work: every run of one model for one cell."""

    model: ModelSpec
    cell: CellSpec
    n_runs: int
    native_max: float
    use_logprobs: bool
    apply_if_eval_hack: bool


class _LoadResult(NamedTuple):
    """Run-averaged scores for one (model, cell)."""

    competitor: str
    cell: CellSpec
    items: np.ndarray  # int64 item keys, sorted
    values: np.ndarray  # float32 run-averaged, rescaled scores
    n_runs_used: np.ndarray  # int16 finite runs behind each value
    failures: list[dict[str, Any]]


class _Contests(NamedTuple):
    """The contests drawn from one cell, as competitor ids rather than names.

    Ids index the loader's own sorted competitor list, not the cell's columns, so cells can be
    concatenated without a join. Names are never materialised per contest: at a few million
    contests a view, turning ids into strings and back costs more than everything else `build`
    does put together.
    """

    competitor_a: np.ndarray  # int64 competitor id
    competitor_b: np.ndarray  # int64 competitor id
    outcome: np.ndarray  # float64, 1.0 / 0.5 / 0.0 from a's point of view


@dataclass
class _Cell:
    """Run-averaged scores for every competitor that can compete in one cell.

    Attributes:
        spec: Which task and language this cell is.
        competitors: Competitor names, sorted; a column index into `scores`.
        items: Item keys retained, sorted. An item is kept only when every competitor in
            `competitors` has a finite score for it, so a contest always compares the two
            models on the same question.
        scores: Run-averaged, rescaled scores, shape (n_items, n_competitors).
        n_runs_used: Finite runs behind each score, same shape.
    """

    spec: CellSpec
    competitors: list[str]
    items: np.ndarray
    scores: np.ndarray
    n_runs_used: np.ndarray

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_competitors(self) -> int:
        return len(self.competitors)


def _subdirs(path: str) -> list[str]:
    """Return sorted directory names under `path`, ignoring dotfiles."""
    if not os.path.isdir(path):
        return []
    return sorted(
        entry.name
        for entry in os.scandir(path)
        if entry.is_dir() and not entry.name.startswith(".")
    )


def _latest_config(configs_dir: str) -> Optional[str]:
    """Return the newest `run_config_<timestamp>.yaml` in `configs_dir`, by timestamp."""
    best: Optional[tuple[datetime, str]] = None
    for name in os.listdir(configs_dir) if os.path.isdir(configs_dir) else []:
        match = _CONFIG_TIMESTAMP.search(name)
        if not name.endswith(".yaml") or match is None:
            continue
        try:
            stamp = datetime.fromisoformat(match.group(1))
        except ValueError:
            continue
        if best is None or stamp > best[0]:
            best = (stamp, os.path.join(configs_dir, name))
    return None if best is None else best[1]


def _score_path(model: ModelSpec, cell: CellSpec, run: int) -> str:
    """Return the score file for one (model, cell, run).

    The filename carries only the leaf model name, so the instruct and reasoning sweeps of
    a model produce identically named files distinguished only by their sweep directory.
    """
    return os.path.join(
        model.path,
        f"run_{run}",
        "inferences",
        cell.lang,
        cell.subfolder,
        f"{model.leaf}_{cell.task}_{cell.lang}.jsonl",
    )


def _read_run(
    path: str, cell: CellSpec, native_max: float, apply_if_eval_hack: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Read one score file into (item keys, rescaled values).

    A row holding a list contributes one item per element. Missing and non-finite values are
    dropped rather than scored zero, so a model that failed an item simply does not compete
    on it. Item keys are built from the raw line number, so skipping a line never shifts the
    identity of the lines after it.

    Raises:
        KeyError: If the file does not store the metric the config resolves this cell to.
    """
    keys: list[int] = []
    values: list[float] = []
    skip_if_eval = (
        apply_if_eval_hack and cell.task == "if-eval" and cell.lang in ("th", "my")
    )

    with open(path) as handle:
        for line_no, line in enumerate(handle):
            row = ujson.loads(line)
            if skip_if_eval:
                # Retained from both existing loaders: this subcategory is not part of the
                # reported if-eval score for these two languages.
                if (
                    row.get("metadata", {}).get("subcategory")
                    == "length_constraints:number_words"
                ):
                    continue

            scores = row["individual_scores"]
            if isinstance(scores, dict):
                if cell.metric not in scores:
                    raise KeyError(
                        f"{path}: the config resolves task {cell.task!r} to metric "
                        f"{cell.metric!r}, but individual_scores holds {sorted(scores)}. "
                        "Either the run predates the current metric definition, or the "
                        "config names an aggregate that is not stored per item."
                    )
                value = scores[cell.metric]
            elif cell.metric is not None:
                raise KeyError(
                    f"{path}: the config resolves task {cell.task!r} to metric "
                    f"{cell.metric!r}, but individual_scores holds a bare "
                    f"{type(scores).__name__}."
                )
            else:
                value = scores

            elements = value if isinstance(value, list) else [value]
            if len(elements) >= _MAX_ELEMS:
                raise ValueError(
                    f"{path} line {line_no}: {len(elements)} values in one row exceeds the "
                    f"{_MAX_ELEMS} an item key can encode."
                )
            for elem_no, element in enumerate(elements):
                if element is None:
                    continue
                numeric = float(element)
                if not math.isfinite(numeric):
                    continue
                keys.append((line_no << _ELEM_BITS) | elem_no)
                values.append(numeric / native_max)

    return (
        np.asarray(keys, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


def _load_model_cell(request: _LoadRequest) -> _LoadResult:
    """Read every run of one (model, cell) and average each item over the runs it appears in.

    Averaging over the runs an item actually has -- rather than requiring all of them --
    mirrors how `MultiRunAggregator` drops empty runs before taking the per-item mean. The
    number of runs behind each value travels with it so thin coverage can be reported.
    """
    model, cell = request.model, request.cell
    # Logprob tasks are evaluated once regardless of how many runs were requested, so
    # averaging is a no-op over the single run that exists.
    runs = range(1) if request.use_logprobs else range(request.n_runs)

    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    failures: list[dict[str, Any]] = []

    for run in runs:
        path = _score_path(model, cell, run)
        try:
            keys, values = _read_run(
                path, cell, request.native_max, request.apply_if_eval_hack
            )
        except FileNotFoundError:
            failures.append(
                {
                    "competitor": model.competitor,
                    "task": cell.task,
                    "language": cell.lang,
                    "run": run,
                    "path": path,
                    "error": "FileNotFoundError",
                }
            )
            continue
        except (KeyError, ValueError):
            # A metric the file does not store, or a row too wide to key, is a
            # configuration error rather than a flaky run: reading a different quantity
            # under a different name would silently average two metrics together.
            raise
        except Exception as error:  # a malformed run must not sink the whole load
            failures.append(
                {
                    "competitor": model.competitor,
                    "task": cell.task,
                    "language": cell.lang,
                    "run": run,
                    "path": path,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        for key, value in zip(keys.tolist(), values.tolist(), strict=True):
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1

    if not totals:
        return _LoadResult(
            competitor=model.competitor,
            cell=cell,
            items=np.empty(0, dtype=np.int64),
            values=np.empty(0, dtype=np.float32),
            n_runs_used=np.empty(0, dtype=np.int16),
            failures=failures,
        )

    items = np.fromiter(totals.keys(), dtype=np.int64, count=len(totals))
    order = np.argsort(items, kind="stable")
    items = items[order]
    sums = np.fromiter(totals.values(), dtype=np.float64, count=len(totals))[order]
    used = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))[order]

    return _LoadResult(
        competitor=model.competitor,
        cell=cell,
        items=items,
        values=(sums / used).astype(np.float32),
        n_runs_used=used.astype(np.int16),
        failures=failures,
    )


def _partition(total: int, shares: Sequence[float]) -> list[int]:
    """Split `total` into integers proportional to `shares`, summing to exactly `total`.

    Largest-remainder rather than the per-level `math.ceil` the existing allocator uses,
    which overshoots: a 200,000-contest request there realises 200,126.
    """
    if total <= 0 or not shares:
        return [0] * len(shares)
    weight = float(sum(shares))
    if weight <= 0:
        return [0] * len(shares)

    exact = [total * share / weight for share in shares]
    counts = [int(math.floor(value)) for value in exact]
    remainder = total - sum(counts)
    if remainder:
        # Ties broken by index so the split is reproducible.
        order = sorted(range(len(shares)), key=lambda i: (-(exact[i] - counts[i]), i))
        for i in order[:remainder]:
            counts[i] += 1
    return counts


def _partition_capped(
    total: int, shares: Sequence[float], capacities: Sequence[int]
) -> list[int]:
    """Split `total` proportionally to `shares` without exceeding any `capacities` entry.

    Water-filling: a cell that cannot absorb its share is pinned to its capacity and the
    surplus is re-split over the cells that still have room, until nothing overflows. Each
    pass fixes at least one cell, so it terminates in at most `len(shares)` passes.

    Returns less than `total` only when every cell is at capacity, which means the view has
    fewer distinct contests than were asked for; `_sample_cell` reports that per cell.
    """
    counts = [0] * len(shares)
    pinned = [False] * len(shares)
    remaining = total
    while True:
        pool = [i for i in range(len(shares)) if not pinned[i]]
        if not pool or remaining <= 0:
            return counts
        split = _partition(remaining, [shares[i] for i in pool])
        overflowing = [
            i for i, count in zip(pool, split, strict=True) if count > capacities[i]
        ]
        if not overflowing:
            for i, count in zip(pool, split, strict=True):
                counts[i] = count
            return counts
        for i in overflowing:
            counts[i] = capacities[i]
            pinned[i] = True
            remaining -= capacities[i]


def _split_evenly(
    budget: int, capacities: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Split `budget` as evenly as possible over pairs, capping each at its `capacities`.

    The same water-filling `_partition_capped` does, specialised to equal shares and written
    over arrays: raise a common level until the budget runs out, and hand a pair that cannot
    reach the level only what it holds, which frees its surplus for the pairs that can.

    The leftover units are handed out *at random* among the pairs that still have room, not
    by index. Every pair's exact share is identical here, so largest-remainder would degrade
    to index order, and `np.triu_indices` puts the same competitors at the front of that
    order in every cell -- reintroducing, one contest at a time, exactly the systematic
    imbalance an equal pair budget exists to remove.

    Returns less than `budget` only when every pair is at capacity, i.e. the cell holds fewer
    distinct comparisons than it was allocated; `_sample_cell` reports that.
    """
    counts = np.zeros(len(capacities), dtype=np.int64)
    if not len(capacities) or budget <= 0:
        return counts
    if budget >= int(capacities.sum()):
        return capacities.astype(np.int64)

    low, high = 0, int(capacities.max())
    while low < high:
        level = (low + high + 1) // 2
        if int(np.minimum(capacities, level).sum()) <= budget:
            low = level
        else:
            high = level - 1
    counts = np.minimum(capacities, low).astype(np.int64)

    # The level is maximal, so raising it by one would overshoot: the leftover is strictly
    # smaller than the number of pairs with room, and one extra unit each is always enough.
    leftover = budget - int(counts.sum())
    room = np.flatnonzero(capacities > counts)
    if leftover > 0 and room.size:
        counts[rng.choice(room, size=min(leftover, room.size), replace=False)] += 1
    return counts


def _assign_shares(
    cells: Sequence[_Cell], levels: Sequence[str]
) -> dict[CellSpec, float]:
    """Give every cell its fraction of the budget, splitting evenly at each level.

    Each level divides its parent's share equally among the distinct values it takes:
    a language with two competencies gives each half, regardless of how many items either
    holds. Aggregation groups enter as one `task_node`.
    """
    if not cells:
        return {}
    if not levels:
        share = 1.0 / len(cells)
        return {cell.spec: share for cell in cells}

    groups: dict[Any, list[_Cell]] = {}
    for cell in cells:
        groups.setdefault(_level_value(cell, levels[0]), []).append(cell)

    per_group = 1.0 / len(groups)
    shares: dict[CellSpec, float] = {}
    for group in groups.values():
        for spec, share in _assign_shares(group, levels[1:]).items():
            shares[spec] = share * per_group
    return shares


def _level_value(cell: _Cell, level: str) -> Any:
    """Return the value of one balancing level for a cell."""
    if level == "task_node":
        return cell.spec.task_node
    return getattr(cell.spec, level)


@dataclass
class BTDataLoader:
    """Builds Bradley-Terry datasets from a SEA-HELM results tree.

    Reading dominates the cost so `load` happens once and caches the run-averaged scores.
    Everything that shapes a dataset rather than the scores behind it (`tolerance`, the view,
    the contest budget, the bootstrap) is applied in `build`, and can be changed without
    touching the tree again.

    Attributes:
        sweep_dirs: Path to a sweep directory -> model type, e.g.
            `{"results/instruct-folder": "instruct", "results/reasoning-folder": "reasoning"}`.
            A sweep directory holds `<org>/<leaf>/run_<r>/`, and the paths need not share a
            results directory. Only the listed sweeps are scanned, so this doubles as a model
            filter, and the type a path maps to labels every competitor beneath it.
        n_runs: Runs per model to average.
        tolerance: Scores within this distance on the rescaled [0, 1] scale are a tie. The
            run-average of a binary metric moves in steps of `1 / n_runs`, so a tolerance
            below half that step means only exactly equal averages tie; the tolerance does
            its real work on continuous metrics.
        n_contests: Default contest budget per `build` call.
        tasks_config_filepath: Read the task schema from this file instead of each model's
            latest `run_config_*.yaml`, so every competitor is read against one schema.
        metric_override_dict: Task -> metric, overriding the config's declared metric.
        metric_scales: Native maximum per metric, extending `METRIC_NATIVE_MAX`.
        ignore_model: Competitor names to exclude.
        ignore_task: Task names to exclude.
        ignore_language: Language codes to exclude.
        ignore_task_lang: Task -> languages to exclude for that task alone.
        omit_competencies: Competency names to exclude.
        exclude_groups: Groups of competitors that must not face each other, for variants of
            one base model that no naming rule would catch (quantisations, candidate
            checkpoints).
        sea_languages: Languages in the `sea` view. Currently set to the shared 7-language
            constant.
        keep_ties: Include tie contests, passed through to `PairDataset.from_indices`, which
            drops them before aggregating so the counts and effective counts cover only the
            contests the fit sees. If ties are frequent, dropping them discards a lot of the
            data and can leave a competitor with no contests at all in a narrow view. Setting
            it False also engages `tie_budget`, since otherwise the cells that tie least would
            take a share of the fit that the balancing scheme never granted them.
        tie_budget: How a cell's contest budget is spent once ties are dropped. Ignored
            entirely when `keep_ties` is True, where every drawn contest reaches the fit.

            - "none": spend the budget on draws and let the ties fall out. A cell's share of
              the fit is then its share of the budget times its decisive rate, so tasks that
              tie rarely -- the continuous metrics -- silently take weight from the tasks
              that tie often. This is the unadjusted behaviour, kept for reproducing runs
              made before the other two existed.
            - "inflate": divide each share by the cell's decisive rate, so the contests that
              survive the drop land on the share the balancing scheme asked for. Right in
              expectation, off by the draw.
            - "decisive": draw from the cell's decisive comparisons alone, so the budget is
              spent entirely on contests the fit will see and the share is met exactly. Costs
              a pass over the cell's whole (item, pair) space per build -- and per bootstrap
              replicate, since resampling the items resamples the decisive set with them.
        pair_budget: How a cell's budget is spent across the pairs inside it. `tie_budget`
            balances the *cells*; this is what balances the *competitors*, and the two are
            independent.

            - "pool": draw uniformly from the cell's pool, so a pair's share of the cell is
              its share of the comparisons in it. With ties dropped that pool is the decisive
              comparisons, and a comparison is decisive only when the two competitors
              disagree -- so a competitor's contest count becomes a measure of how often it
              disagrees with the field. On a benchmark most models pass, the weakest model
              is the one disagreeing, and ends up with several times the contests of the
              strongest. This is the default.
            - "equal": split the cell's budget evenly over its pairs and draw each pair's
              quota from that pair's own comparisons. Every competitor sits in exactly
              `n_competitors - 1` of a cell's pairs, so equal per pair is equal per
              competitor, and the cell balance is untouched -- all three hold at once. It
              also spends the budget where the resolution is missing: the top of a leaderboard
              is packed because near-equal models rarely disagree, and this draws their scarce
              disagreements as heavily as anyone else's.

              Note that it changes the estimand. Under "pool" a pair weighs on the fit in
              proportion to how often the two disagree; under "equal" every pair weighs the same,
              so a pair apart on 2% of items counts for what one apart on 60% counts for.
        apply_if_eval_hack: Skip the `length_constraints:number_words` if-eval subcategory
            for Thai and Burmese, as both existing loaders do.
        seed: Base seed for contest sampling.
        num_processes: Worker processes for reading.
        show_progress: Show a progress bar over the (competitor, cell) reads.
        cache_dir: Directory for the run-averaged score cache. None disables caching.
        output_dir: Directory for the diagnostics CSVs. None disables writing them.
    """

    sweep_dirs: Mapping[str, str]
    n_runs: int = 8
    tolerance: float = 0.01
    n_contests: int = 1_000_000
    tasks_config_filepath: Optional[str] = None
    metric_override_dict: Mapping[str, str] = field(default_factory=dict)
    metric_scales: Mapping[str, float] = field(default_factory=dict)
    ignore_model: Iterable[str] = field(default_factory=set)
    ignore_task: Iterable[str] = field(default_factory=set)
    ignore_language: Iterable[str] = field(default_factory=set)
    ignore_task_lang: Mapping[str, Iterable[str]] = field(default_factory=dict)
    omit_competencies: Iterable[str] = field(default_factory=set)
    exclude_groups: Sequence[Sequence[str]] = field(default_factory=list)
    sea_languages: Sequence[str] = tuple(SEA_LANGUAGES)
    keep_ties: bool = True
    tie_budget: str = "inflate"
    pair_budget: str = "pool"
    apply_if_eval_hack: bool = True
    seed: int = 94370244
    num_processes: int = 32
    show_progress: bool = True
    cache_dir: Optional[str] = None
    output_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tie_budget not in _TIE_BUDGETS:
            raise ValueError(
                f"tie_budget must be one of {', '.join(_TIE_BUDGETS)}, got "
                f"{self.tie_budget!r}."
            )
        if self.pair_budget not in _PAIR_BUDGETS:
            raise ValueError(
                f"pair_budget must be one of {', '.join(_PAIR_BUDGETS)}, got "
                f"{self.pair_budget!r}."
            )
        self.ignore_model = set(self.ignore_model)
        self.ignore_task = set(self.ignore_task)
        self.ignore_language = set(self.ignore_language)
        self.omit_competencies = set(self.omit_competencies)
        self.cells: dict[CellSpec, _Cell] = {}
        self.models: list[ModelSpec] = []
        self.failed_loads: pd.DataFrame = pd.DataFrame()
        # (cell spec, tolerance) -> decisive comparisons per pair in the cell. Keyed on the
        # tolerance because it defines what a tie is, and memoised because `allocate` runs
        # once per bootstrap replicate. Counts rather than rates, so the decisive capacity a
        # "decisive" budget is capped at is exact rather than a rounded product, and held per
        # pair rather than summed because that is the capacity `pair_budget="equal"` caps
        # against; the cell total is the sum. ~34 kB per cell at ~4,300 pairs.
        self._decisive_counts: dict[tuple[CellSpec, float], np.ndarray] = {}
        self._names: Optional[list[str]] = None
        self._name_ids: dict[str, int] = {}
        self._column_ids: dict[CellSpec, np.ndarray] = {}
        self._excluded: set[frozenset[str]] = {
            frozenset(pair)
            for group in self.exclude_groups
            for pair in _unordered_pairs(group)
        }

    # ---------------------------------------------------------------- discovery

    def discover_models(self) -> list[ModelSpec]:
        """Find every competitor under the configured sweep directories.

        The tree is `<sweep_dir>/<org>/<leaf>/run_<r>/`, and a leaf counts as a competitor
        only once it has a `run_0`.
        """
        models: list[ModelSpec] = []
        for sweep_dir, model_type in self.sweep_dirs.items():
            if not os.path.isdir(sweep_dir):
                raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")
            for org in _subdirs(sweep_dir):
                for leaf in _subdirs(os.path.join(sweep_dir, org)):
                    path = os.path.join(sweep_dir, org, leaf)
                    if not os.path.isdir(os.path.join(path, "run_0")):
                        continue
                    competitor = f"{model_type}/{org}/{leaf}"
                    if (
                        competitor in self.ignore_model
                        or f"{org}/{leaf}" in self.ignore_model
                    ):
                        continue
                    models.append(
                        ModelSpec(
                            competitor=competitor,
                            model_type=model_type,
                            sweep_dir=sweep_dir,
                            org=org,
                            leaf=leaf,
                            path=path,
                        )
                    )
        if not models:
            raise ValueError(
                f"No competitors found under sweep directories {sorted(self.sweep_dirs)}."
            )
        return models

    def competitor_table(self) -> pd.DataFrame:
        """Return competitor names alongside their type, org and leaf, for joining."""
        return pd.DataFrame(
            [
                {
                    "competitor": model.competitor,
                    "model_type": model.model_type,
                    "model": f"{model.org}/{model.leaf}",
                    "sweep_dir": model.sweep_dir,
                    "org": model.org,
                    "leaf": model.leaf,
                }
                for model in self.models
            ]
        )

    def _read_config(self, model: ModelSpec) -> dict[str, Any]:
        """Return the task schema this model's scores are read against."""
        path = self.tasks_config_filepath or _latest_config(
            os.path.join(model.path, "run_0", "configs")
        )
        if path is None:
            raise FileNotFoundError(
                f"No run_config_<timestamp>.yaml under {model.path}/run_0/configs. Pass "
                "tasks_config_filepath to read every competitor against one schema."
            )
        with open(path) as handle:
            config = yaml.safe_load(handle)
        if not config or "tasks" not in config:
            raise ValueError(f"{path} does not declare any tasks.")
        return config

    def _cell_specs(self, config: dict[str, Any]) -> list[tuple[CellSpec, bool]]:
        """Return the cells a config declares, paired with whether they use logprobs."""
        specs: list[tuple[CellSpec, bool]] = []
        skipped = set(config.get("run_args", {}).get("skip_task", []) or [])
        for task, task_config in config["tasks"].items():
            if task in self.ignore_task or task in skipped:
                continue
            competency = task_config["competency"]
            if competency in self.omit_competencies:
                continue
            ignored_langs = set(self.ignore_task_lang.get(task, ()))
            metric = self.metric_override_dict.get(task, task_config.get("metric"))
            use_logprobs = (
                bool(task_config.get("use_logprobs", False)) or "logprobs" in task
            )
            for lang in task_config["languages"]:
                if lang in self.ignore_language or lang in ignored_langs:
                    continue
                specs.append(
                    (
                        CellSpec(
                            task=task,
                            lang=lang,
                            competency=competency,
                            aggregation_group=task_config.get("aggregation_group"),
                            metric=metric,
                        ),
                        use_logprobs,
                    )
                )
        return specs

    # ------------------------------------------------------------------- loading

    def load(self, use_cache: bool = True) -> "BTDataLoader":
        """Read every competitor's runs into one run-averaged score matrix per cell.

        Args:
            use_cache: Read from and write to `cache_dir` when it is set. The cache holds the
                run-averaged scores, so changing `n_runs`, the metric overrides or the task
                schema invalidates it, while changing `tolerance`, the view or the contest
                budget does not.

        Returns:
            self, so a load can be chained into a build.
        """
        self.models = self.discover_models()
        # Everything derived from the cells is about to be replaced along with them: the
        # decisive counts were counted over the old scores, and the ids index the old columns.
        self._decisive_counts.clear()
        self._column_ids.clear()
        self._names, self._name_ids = None, {}
        cache_path = self._cache_path() if (use_cache and self.cache_dir) else None
        if cache_path and os.path.exists(cache_path):
            self._load_cache(cache_path)
            return self

        requests: list[_LoadRequest] = []
        for model in tqdm(
            self.models, desc="Discovering models", disable=not self.show_progress
        ):
            for spec, use_logprobs in self._cell_specs(self._read_config(model)):
                requests.append(
                    _LoadRequest(
                        model=model,
                        cell=spec,
                        n_runs=self.n_runs,
                        native_max=get_native_max(spec.metric, self.metric_scales),
                        use_logprobs=use_logprobs,
                        apply_if_eval_hack=self.apply_if_eval_hack,
                    )
                )

        by_cell: dict[CellSpec, list[_LoadResult]] = {}
        failures: list[dict[str, Any]] = []
        progress = tqdm(
            total=len(requests),
            desc="Reading cells",
            unit="cell",
            disable=not self.show_progress,
        )
        with progress:
            if self.num_processes > 1:
                with Pool(processes=self.num_processes) as pool:
                    results = pool.imap_unordered(
                        _load_model_cell, requests, chunksize=8
                    )
                    for result in results:
                        self._collect(result, by_cell, failures)
                        progress.update()
            else:
                for request in requests:
                    self._collect(_load_model_cell(request), by_cell, failures)
                    progress.update()

        self.failed_loads = pd.DataFrame(failures)
        self.cells = {
            spec: cell for spec, cell in map(_align_cell, by_cell.items()) if cell
        }
        if not self.cells:
            raise ValueError(
                "No cell has two competitors with items in common. Check the results tree "
                "layout and the task schema."
            )
        if cache_path:
            self._write_cache(cache_path)
        return self

    @staticmethod
    def _collect(
        result: _LoadResult,
        by_cell: dict[CellSpec, list[_LoadResult]],
        failures: list[dict[str, Any]],
    ) -> None:
        """Route one worker result into the per-cell bucket and the failure log."""
        failures.extend(result.failures)
        if result.items.size:
            by_cell.setdefault(result.cell, []).append(result)

    # -------------------------------------------------------------------- cache

    def _cache_key(self) -> str:
        """Hash the inputs that determine the run-averaged scores."""
        payload = ujson.dumps(
            {
                "sweep_dirs": {
                    os.path.abspath(path): model_type
                    for path, model_type in sorted(self.sweep_dirs.items())
                },
                "n_runs": self.n_runs,
                "config": self.tasks_config_filepath,
                "metric_override": dict(sorted(self.metric_override_dict.items())),
                "metric_scales": dict(sorted(self.metric_scales.items())),
                "ignore_model": sorted(self.ignore_model),
                "ignore_task": sorted(self.ignore_task),
                "ignore_language": sorted(self.ignore_language),
                "ignore_task_lang": {
                    k: sorted(v) for k, v in sorted(self.ignore_task_lang.items())
                },
                "omit_competencies": sorted(self.omit_competencies),
                "if_eval_hack": self.apply_if_eval_hack,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"bt_scores_{self._cache_key()}.npz")

    def _write_cache(self, path: str) -> None:
        """Persist the run-averaged scores as float32, one array group per cell."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        index: list[dict[str, Any]] = []
        ordered = sorted(self.cells.items(), key=lambda kv: (kv[0].lang, kv[0].task))
        for order, (spec, cell) in enumerate(ordered):
            arrays[f"items_{order}"] = cell.items
            arrays[f"scores_{order}"] = cell.scores.astype(np.float32)
            arrays[f"runs_{order}"] = cell.n_runs_used.astype(np.int16)
            index.append(
                {
                    "task": spec.task,
                    "lang": spec.lang,
                    "competency": spec.competency,
                    "aggregation_group": spec.aggregation_group,
                    "metric": spec.metric,
                    "competitors": cell.competitors,
                }
            )
        arrays["index"] = np.asarray(ujson.dumps(index))
        np.savez_compressed(path, **arrays)

    def _load_cache(self, path: str) -> None:
        """Restore the run-averaged scores written by `_write_cache`."""
        with np.load(path, allow_pickle=False) as data:
            index = ujson.loads(str(data["index"]))
            self.cells = {}
            for order, entry in enumerate(index):
                spec = CellSpec(
                    task=entry["task"],
                    lang=entry["lang"],
                    competency=entry["competency"],
                    aggregation_group=entry["aggregation_group"],
                    metric=entry["metric"],
                )
                self.cells[spec] = _Cell(
                    spec=spec,
                    competitors=list(entry["competitors"]),
                    items=data[f"items_{order}"],
                    scores=data[f"scores_{order}"],
                    n_runs_used=data[f"runs_{order}"],
                )

    # -------------------------------------------------------------------- views

    def select_cells(
        self,
        category: str = "global",
        label: Optional[str] = None,
        balance_by: str = "language",
    ) -> list[_Cell]:
        """Return the cells one view covers.

        Args:
            category: "global", "sea", "language", "competency" or "task".
            label: The language, competency or task to restrict to. Required for those three
                categories, ignored for "global" and "sea". A "task" label may name either a
                task or an aggregation group.
            balance_by: For "global" and "sea", which level gets equal shares at the top:
                "language", "competency" or "task".
        """
        if category in ("global", "sea"):
            if balance_by not in ("language", "competency", "task"):
                raise ValueError(
                    f"balance_by must be language, competency or task, got {balance_by!r}."
                )
        elif label is None:
            raise ValueError(f"category {category!r} requires a label.")

        cells = list(self.cells.values())
        if category == "sea":
            allowed = set(self.sea_languages)
            cells = [cell for cell in cells if cell.spec.lang in allowed]
        elif category == "language":
            cells = [cell for cell in cells if cell.spec.lang == label]
        elif category == "competency":
            cells = [cell for cell in cells if cell.spec.competency == label]
        elif category == "task":
            cells = [
                cell for cell in cells if label in (cell.spec.task, cell.spec.task_node)
            ]
        elif category != "global":
            raise ValueError(f"Unknown category {category!r}.")

        if not cells:
            raise ValueError(f"No cells match category={category!r} label={label!r}.")
        return sorted(cells, key=lambda cell: (cell.spec.lang, cell.spec.task))

    def allocate(
        self,
        cells: Sequence[_Cell],
        category: str,
        balance_by: str,
        n_contests: int,
    ) -> dict[CellSpec, int]:
        """Split `n_contests` across cells, equal share per balancing level.

        With `keep_ties=False` a drawn contest that turns out to be a tie is discarded
        before the fit, so a cell's share of the *budget* is not its share of the
        *evidence*: at the tolerances used here a discrete task ties on half its
        comparisons while a continuous one ties on under a tenth, which silently moves
        weight from the former to the latter. `tie_budget` says what to do about it --
        "inflate" divides each share by the cell's decisive rate so the surviving contests
        land on the granted share, "decisive" spends the budget on decisive comparisons
        alone so they land on it exactly, and "none" leaves the distortion in place.

        Either way the budget is capped at what a cell can supply -- its whole space under
        "inflate", its decisive pool under "decisive" -- and the surplus redistributed, so
        adjusting for the ties never pushes a cell into drawing one contest twice.
        """
        levels = _VIEW_LEVELS[
            (category, balance_by if category in ("global", "sea") else None)
        ]
        shares = _assign_shares(cells, (*levels, "task"))
        ordered = [cell.spec for cell in cells]
        if self.keep_ties or self.tie_budget == "none":
            counts = _partition(n_contests, [shares[spec] for spec in ordered])
            return dict(zip(ordered, counts, strict=True))

        if self.tie_budget == "decisive":
            # Every contest drawn is one the fit keeps, so the share stands unadjusted and
            # what a cell can supply is its decisive pool rather than its whole space.
            weights = [shares[spec] for spec in ordered]
            capacity = [self._decisive_count(cell) for cell in cells]
        else:
            rates = {cell.spec: self._decisive_rate(cell) for cell in cells}
            weights = [
                0.0
                if rates[spec] <= 0.0
                else shares[spec] / max(rates[spec], _MIN_DECISIVE_RATE)
                for spec in ordered
            ]
            capacity = [cell.n_items * len(self._cell_pairs(cell)[0]) for cell in cells]
        counts = _partition_capped(n_contests, weights, capacity)

        # A capped cell gets less than the scheme granted it, so the balance is no longer
        # exactly the one `_assign_shares` describes. Rare at a sane budget and worth
        # knowing about when it is not: capped everywhere means the view is weighted by how
        # big its cells are rather than by language and competency.
        capped = [
            spec
            for spec, count, room in zip(ordered, counts, capacity, strict=True)
            if room and count >= room
        ]
        if capped:
            pool = (
                "decisive comparisons" if self.tie_budget == "decisive" else "contests"
            )
            warnings.warn(
                f"{len(capped)} of {len(cells)} cells cannot supply their tie-adjusted "
                f"share and were capped at the {pool} they hold "
                f"({', '.join(f'{spec.task}/{spec.lang}' for spec in capped[:5])}"
                f"{', ...' if len(capped) > 5 else ''}). Their weight in the fit is below "
                "the balancing scheme's; lower n_contests to restore it.",
                stacklevel=3,
            )
        return dict(zip(ordered, counts, strict=True))

    def _count_decisive(
        self, scores: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray
    ) -> np.ndarray:
        """Decisive comparisons per pair in a score matrix, at `self.tolerance`.

        Takes the matrix rather than the cell for the same reason `_decisive_positions` does:
        an item bootstrap replaces it, and the counts have to follow the items drawn.
        """
        counts = np.zeros(len(idx_a), dtype=np.int64)
        for start in range(0, len(scores), _DECISIVE_CHUNK):
            block = scores[start : start + _DECISIVE_CHUNK]
            difference = block[:, idx_a] - block[:, idx_b]
            counts += np.count_nonzero(np.abs(difference) > self.tolerance, axis=0)
        return counts

    def _pair_decisive_counts(self, cell: _Cell) -> np.ndarray:
        """`_count_decisive` over a cell's own items, memoised."""
        key = (cell.spec, self.tolerance)
        cached = self._decisive_counts.get(key)
        if cached is None:
            idx_a, idx_b = self._cell_pairs(cell)
            cached = self._count_decisive(cell.scores, idx_a, idx_b)
            self._decisive_counts[key] = cached
        return cached

    def _decisive_count(self, cell: _Cell) -> int:
        """Comparisons in a cell's (item, pair) space that are not a tie at `self.tolerance`.

        Counted exactly over the whole space rather than estimated from a draw: this scales
        the contest budget, and an estimate would make the allocation -- and so every
        rating -- depend on the sampling seed.
        """
        return int(self._pair_decisive_counts(cell).sum())

    def _decisive_rate(self, cell: _Cell) -> float:
        """Fraction of a cell's (item, pair) space that is not a tie at `self.tolerance`."""
        total = cell.n_items * len(self._cell_pairs(cell)[0])
        return 0.0 if total == 0 else self._decisive_count(cell) / total

    def _decisive_positions(
        self, scores: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray
    ) -> np.ndarray:
        """Flat (item, pair) indices of the decisive comparisons in a score matrix.

        Takes the matrix rather than the cell because an item bootstrap replaces it, and the
        decisive set has to follow the items that were actually drawn. Indices are row-major
        over (items, pairs), the same encoding `_sample_cell` divmods back apart.
        """
        blocks: list[np.ndarray] = []
        for start in range(0, len(scores), _DECISIVE_CHUNK):
            block = scores[start : start + _DECISIVE_CHUNK]
            difference = block[:, idx_a] - block[:, idx_b]
            found = np.flatnonzero(np.abs(difference) > self.tolerance)
            if found.size:
                blocks.append(found + start * len(idx_a))
        return np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)

    # -------------------------------------------------------------------- build

    def build(
        self,
        category: str = "global",
        label: Optional[str] = None,
        balance_by: str = "language",
        n_contests: Optional[int] = None,
        seed: Optional[int] = None,
        bootstrap: str = "none",
    ):
        """Sample contests for one view and return them as a `PairDataset`.

        Args:
            category, label, balance_by: The view, as in `select_cells`.
            n_contests: Contest budget, defaulting to `self.n_contests`.
            seed: Sampling seed, defaulting to `self.seed`. Vary it across bootstrap
                replicates.
            bootstrap: "none" for the point estimate; "contests" to draw a fresh contest
                sample, which is what the existing pipeline's bootstrap loop varies;
                "items" to additionally resample each cell's item pool with replacement,
                which is the question-level bootstrap and gives wider, more honest
                intervals.

        Returns:
            (PairDataset, cell summary DataFrame). The summary carries one row per cell with
            its budget, item and competitor counts, tie rate and run coverage.
        """
        from src.elo.data_utils import (  # local: keeps import cost off `load`
            PairDataset,
        )

        if bootstrap not in ("none", "contests", "items"):
            raise ValueError(
                f"bootstrap must be none, contests or items, got {bootstrap!r}."
            )
        cells = self.select_cells(category, label, balance_by)
        budgets = self.allocate(
            cells,
            category,
            balance_by,
            self.n_contests if n_contests is None else n_contests,
        )
        rng = np.random.default_rng(self.seed if seed is None else seed)

        drawn: list[_Contests] = []
        summary: list[dict[str, Any]] = []
        for cell in cells:
            contests, row = self._sample_cell(cell, budgets[cell.spec], rng, bootstrap)
            summary.append(row)
            if contests is not None:
                drawn.append(contests)

        if not drawn:
            raise ValueError(
                "No contests were sampled. Every cell was either empty or allocated a "
                "budget of zero -- raise n_contests, or, with keep_ties=False, check that "
                "some cell is decisive at this tolerance: a view in which every comparison "
                "is a tie is allocated nothing, because nothing in it would reach the fit."
            )
        dataset = PairDataset.from_indices(
            np.concatenate([contests.competitor_a for contests in drawn]),
            np.concatenate([contests.competitor_b for contests in drawn]),
            np.concatenate([contests.outcome for contests in drawn]),
            competitor_names=self._competitor_names(),
            keep_ties=self.keep_ties,
        )
        cell_summary = pd.DataFrame(summary)
        self._check_graph(dataset)
        return dataset, cell_summary

    def bootstrap_ratings(
        self,
        n_replicates: int = 2000,  # set to 2000 to calculate 95% confidence intervals
        bootstrap: str = "items",
        seed: Optional[int] = None,
        n_contests: Optional[int] = None,
        **view: Any,
    ) -> pd.DataFrame:
        """Refit on resampled datasets and return one row of ratings per replicate.

        Ratings are indexed by competitor *name*, not position: a `PairDataset`'s competitors
        are the ones that won a contest in that replicate, so a competitor missing from one
        would silently shift every rating after it if the rows were stacked positionally.

        Replicates are independent -- each draws from its own seed -- so they parallelise
        exactly. Sampling is the whole cost; the fit itself is milliseconds.

        Args:
            n_replicates: Replicates to draw.
            bootstrap: "contests" to vary only the contest draw, "items" to resample each
                cell's item pool with replacement as well, which is the question-level
                interval. "none" would return `n_replicates` identical rows.
            seed: Seed for the replicate seeds, defaulting to `self.seed`.
            n_contests: Contest budget per replicate, defaulting to `self.n_contests`.
            **view: Passed to `build`: `category`, `label`, `balance_by`.

        Returns:
            A (replicate, competitor) frame of rescaled ratings, with NaN where a competitor
            was absent from a replicate.
        """
        seeds = (
            np.random.default_rng(self.seed if seed is None else seed)
            .integers(0, 2**32 - 1, size=n_replicates)
            .tolist()
        )
        # Warm the memoised decisive counts here rather than in each worker: they are what the
        # allocation is computed from, identical across replicates, and a fork inherits them.
        if not self.keep_ties and self.tie_budget != "none":
            for cell in self.select_cells(**view):
                self._pair_decisive_counts(cell)

        task = (self, bootstrap, n_contests, view)
        rows: list[tuple[list[str], np.ndarray]] = []
        progress = tqdm(
            total=n_replicates,
            desc=f"Bootstrap ({bootstrap})",
            unit="replicate",
            disable=not self.show_progress,
        )
        with progress:
            if self.num_processes > 1:
                context = multiprocessing.get_context(
                    "fork"
                    if "fork" in multiprocessing.get_all_start_methods()
                    else None
                )
                with context.Pool(
                    processes=min(self.num_processes, n_replicates),
                    initializer=_init_bootstrap_worker,
                    initargs=(task,),
                ) as pool:
                    for row in pool.imap_unordered(
                        _bootstrap_replicate, seeds, chunksize=1
                    ):
                        rows.append(row)
                        progress.update()
            else:
                _init_bootstrap_worker(task)
                for replicate_seed in seeds:
                    rows.append(_bootstrap_replicate(replicate_seed))
                    progress.update()

        return pd.DataFrame(
            [pd.Series(ratings, index=names) for names, ratings in rows]
        )

    def _competitor_names(self) -> list[str]:
        """Every competitor a cell may hold, sorted, as the id space contests are drawn in.

        Sorted because `PairDataset.from_indices` renumbers to the competitors present in the
        order given, and a rating table aligned by name across bootstrap replicates needs that
        order to be the same one `pd.factorize(sort=True)` would have produced.
        """
        if self._names is None:
            self._names = sorted(
                {name for cell in self.cells.values() for name in cell.competitors}
            )
            self._name_ids = {name: index for index, name in enumerate(self._names)}
        return self._names

    def _competitor_ids(self, cell: _Cell) -> np.ndarray:
        """Global competitor id per column of a cell's score matrix, memoised."""
        cached = self._column_ids.get(cell.spec)
        if cached is None:
            self._competitor_names()
            cached = np.fromiter(
                (self._name_ids[name] for name in cell.competitors),
                dtype=np.int64,
                count=cell.n_competitors,
            )
            self._column_ids[cell.spec] = cached
        return cached

    def _sample_cell(
        self,
        cell: _Cell,
        budget: int,
        rng: np.random.Generator,
        bootstrap: str,
    ) -> tuple[Optional[_Contests], dict[str, Any]]:
        """Draw one cell's contests from the space it is allowed to draw from.

        That space is the cell's whole (item, pair) grid, or, under
        `tie_budget="decisive"`, the decisive comparisons within it. Drawing uniformly
        without replacement from a subset is the same distribution as drawing from the
        whole grid and rejecting the ties until the budget is met, without the loop.

        Under `pair_budget="equal"` the draw is stratified by pair instead of uniform over
        that space, which is what balances the competitors; see `_draw_per_pair`.
        """
        pair_a, pair_b = self._cell_pairs(cell)
        row: dict[str, Any] = {
            "task": cell.spec.task,
            "lang": cell.spec.lang,
            "competency": cell.spec.competency,
            "aggregation_group": cell.spec.aggregation_group,
            "task_node": cell.spec.task_node,
            "metric": cell.spec.metric,
            "n_items": cell.n_items,
            "n_competitors": cell.n_competitors,
            "n_pairs": len(pair_a),
            "budget": budget,
            "sampled": 0,
            "available": cell.n_items * len(pair_a),
            # The decisive comparisons a "decisive" budget draws from, which is what
            # `available` means for it; NaN under the other budgets, which use the grid.
            "decisive_available": float("nan"),
            "with_replacement": False,
            # Pairs that hold fewer comparisons than an even split of the cell's budget, and
            # so were pinned to what they hold with the surplus spread over the rest. NaN
            # under `pair_budget="pool"`, which does not split by pair at all.
            "pairs_capped": float("nan"),
            "tie_rate": float("nan"),
            # The exact rate the allocation was computed against, of which `tie_rate` --
            # measured on the draw -- is the check. NaN when no adjustment was made.
            "alloc_decisive_rate": (
                float("nan")
                if self.keep_ties or self.tie_budget == "none"
                else self._decisive_rate(cell)
            ),
            "min_runs_used": int(cell.n_runs_used.min()) if cell.n_items else 0,
            "max_runs_used": int(cell.n_runs_used.max()) if cell.n_items else 0,
        }
        available = row["available"]
        if budget <= 0 or available == 0:
            return None, row

        scores = cell.scores
        if bootstrap == "items":
            # Resample the questions themselves, so the interval reflects the item pool
            # being a sample rather than the population.
            scores = scores[rng.integers(0, cell.n_items, size=cell.n_items)]

        decisive_only = not self.keep_ties and self.tie_budget == "decisive"

        if self.pair_budget == "equal":
            # Capacities come from the same score matrix the draw uses, so an item bootstrap
            # caps against the pool it actually resampled; the memoised per-cell counts are
            # only valid for the point estimate.
            if not decisive_only:
                capacity = np.full(len(pair_a), len(scores), dtype=np.int64)
            elif bootstrap == "items":
                capacity = self._count_decisive(scores, pair_a, pair_b)
            else:
                capacity = self._pair_decisive_counts(cell)
            if decisive_only:
                pool_size = int(capacity.sum())
                row["decisive_available"] = pool_size
                if pool_size == 0:
                    return None, row

            quotas = _split_evenly(budget, capacity, rng)
            row["pairs_capped"] = int(np.count_nonzero(quotas >= capacity))
            item_idx, pair_idx = self._draw_per_pair(
                scores, pair_a, pair_b, quotas, rng, decisive_only, capacity=capacity
            )
            if not len(item_idx):
                return None, row
        else:
            if not decisive_only:
                pool, pool_size, kind = None, available, "contests"
            else:
                # Recomputed per call rather than memoised: an item bootstrap replaces the
                # score matrix, and the decisive set follows the items actually drawn.
                pool = self._decisive_positions(scores, pair_a, pair_b)
                pool_size, kind = int(pool.size), "decisive comparisons"
                row["decisive_available"] = pool_size
                if pool_size == 0:
                    return None, row

            replace = budget > pool_size
            row["with_replacement"] = replace
            if replace:
                warnings.warn(
                    f"{cell.spec.task}/{cell.spec.lang}: {pool_size} {kind} available for a "
                    f"budget of {budget}; sampling with replacement.",
                    stacklevel=3,
                )
                draw = rng.integers(0, pool_size, size=budget)
            else:
                draw = rng.choice(pool_size, size=budget, replace=False)

            flat = draw if pool is None else pool[draw]
            item_idx, pair_idx = np.divmod(flat, len(pair_a))

        score_a = scores[item_idx, pair_a[pair_idx]]
        score_b = scores[item_idx, pair_b[pair_idx]]
        outcome = _outcomes(score_a, score_b, self.tolerance)

        ids = self._competitor_ids(cell)
        # The realised count, which an equal pair budget can leave below the allocation when
        # the cell holds fewer comparisons than it was granted. Identical to the budget under
        # "pool", which makes up any shortfall with replacement instead.
        row["sampled"] = int(len(item_idx))
        row["tie_rate"] = float(np.mean(outcome == 0.5))
        contests = _Contests(
            competitor_a=ids[pair_a[pair_idx]],
            competitor_b=ids[pair_b[pair_idx]],
            outcome=outcome,
        )
        return contests, row

    def _draw_per_pair(
        self,
        scores: np.ndarray,
        idx_a: np.ndarray,
        idx_b: np.ndarray,
        quotas: np.ndarray,
        rng: np.random.Generator,
        decisive_only: bool,
        capacity: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw `quotas[p]` distinct comparisons from pair p's own items, for every pair.

        The quotas are a handful of contests against a pool of hundreds of items -- a few
        contests per pair per cell at any sane budget -- so the work here is proportional to
        the contests drawn, not to the cell's (item, pair) grid. Ranking every item to keep
        the top few would be ~100x the necessary work: `_draw_sparse` draws candidates and
        rejects, and only pairs whose quota is a large fraction of what they hold go through
        `_draw_dense`, which ranks.

        Args:
            capacity: Comparisons pair p may be drawn from -- its decisive count under
                `decisive_only`, else the item count. Recomputed when omitted; `_sample_cell`
                already holds it, because it is what the quotas were capped against.
        """
        n_pairs = len(idx_a)
        n_items = len(scores)
        quotas = np.asarray(quotas, dtype=np.int64)
        empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        if not n_pairs or not n_items or not quotas.any():
            return empty

        if capacity is None:
            capacity = (
                self._count_decisive(scores, idx_a, idx_b)
                if decisive_only
                else np.full(n_pairs, n_items, dtype=np.int64)
            )
        capacity = np.asarray(capacity, dtype=np.int64)
        wanted = np.minimum(quotas, capacity)

        # Rejection costs a draw per contest plus the rejects, ranking costs a key per item
        # per pair. Which is cheaper turns on the quota's share of the pair's pool, and the
        # crossover is nowhere near either end, so the exact split hardly matters.
        dense = wanted > _DENSE_PAIR_FRACTION * capacity
        active = wanted > 0
        shared = (scores, idx_a, idx_b, wanted, capacity)
        sparse_items, sparse_pairs = self._draw_sparse(
            *shared, np.flatnonzero(active & ~dense), rng, decisive_only
        )
        dense_items, dense_pairs = self._draw_dense(
            *shared, np.flatnonzero(active & dense), rng, decisive_only
        )
        return (
            np.concatenate([sparse_items, dense_items]),
            np.concatenate([sparse_pairs, dense_pairs]),
        )

    def _draw_sparse(
        self,
        scores: np.ndarray,
        idx_a: np.ndarray,
        idx_b: np.ndarray,
        wanted: np.ndarray,
        capacity: np.ndarray,
        chosen: np.ndarray,
        rng: np.random.Generator,
        decisive_only: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw each chosen pair's quota by rejection, for quotas well under what a pair holds.

        Draws candidate items uniformly, reads the two scores that candidate compares, and
        keeps it if it is decisive and not already drawn for that pair. Rejecting the ties is
        what makes an accepted candidate uniform over the pair's *decisive* items, and
        rejecting the repeats is what makes the quota distinct -- taking each pair's first
        `wanted` survivors in the order they were generated is sequential sampling without
        replacement, spelled as a few vectorised rounds instead of a loop over pairs.

        Rounds are needed because a round's yield is random; each one draws for the shortfall
        that is left, over-drawing by the pair's own reject rate so the shortfall collapses
        quickly, and widening that factor if a straggler somehow survives several rounds.
        """
        n_items = len(scores)
        shortfall = wanted[chosen].copy()
        left = idx_a[chosen]
        right = idx_b[chosen]
        # Exactly the acceptance rate of the tie rejection: capacity is the pair's decisive
        # count, so this is the share of items that survive it.
        keep_rate = (
            np.maximum(capacity[chosen] / n_items, 1.0 / n_items)
            if decisive_only
            else np.ones(len(chosen))
        )

        drawn_items: list[np.ndarray] = []
        drawn_slots: list[np.ndarray] = []
        # Encoded (slot, item) of everything accepted so far, sorted, so a later round rejects
        # a repeat with a `searchsorted` rather than a membership test over the whole set.
        accepted = np.empty(0, dtype=np.int64)
        overdraw = 1.0
        while True:
            active = np.flatnonzero(shortfall > 0)
            if not active.size:
                break

            # Enough candidates that a round's *worst* plausible yield still fills the
            # shortfall, not merely its mean: a round costs a pass over every active pair, so
            # one that leaves a single pair short has cost more than the candidates it saved.
            # Rejections are binomial in the pair's own decisive rate, hence the sqrt term.
            need = shortfall[active]
            budgets = np.ceil(
                (need + _DRAW_SIGMA * np.sqrt(need) + _DRAW_PAD)
                / keep_rate[active]
                * overdraw
            ).astype(np.int64)
            slot = np.repeat(active, budgets)
            item = rng.integers(0, n_items, size=slot.size)

            if decisive_only:
                difference = (
                    scores[item, left[slot]].astype(np.float64)
                    - scores[item, right[slot]]
                )
                survives = np.abs(difference) > self.tolerance
                slot, item = slot[survives], item[survives]

            # Keep each candidate's first appearance, in generation order: `unique` reports
            # the first index of each key, and sorting those indices restores that order.
            key = slot * n_items + item
            _, first = np.unique(key, return_index=True)
            first.sort()
            slot, item, key = slot[first], item[first], key[first]

            if accepted.size:
                nearest = np.searchsorted(accepted, key)
                fresh = accepted[np.minimum(nearest, accepted.size - 1)] != key
                slot, item, key = slot[fresh], item[fresh], key[fresh]
            if not slot.size:
                overdraw *= 2
                continue

            # Take each pair's first `shortfall` survivors. The stable sort groups the pairs
            # while leaving generation order within a pair, so the rank is the position in a
            # sequence of distinct uniform draws -- which is the without-replacement draw.
            order = np.argsort(slot, kind="stable")
            slot, item, key = slot[order], item[order], key[order]
            starts = np.searchsorted(slot, np.arange(len(shortfall)))
            rank = np.arange(len(slot)) - starts[slot]
            taken = rank < shortfall[slot]
            slot, item, key = slot[taken], item[taken], key[taken]

            drawn_items.append(item)
            drawn_slots.append(slot)
            shortfall -= np.bincount(slot, minlength=len(shortfall))
            # Both sides hold distinct keys, so a sort of the concatenation is their union --
            # and the whole accepted set is a few tens of thousands of keys, where deriving it
            # with `unique` or testing membership with `isin` costs more than the draw does.
            accepted = np.sort(np.concatenate([accepted, key]))
            # A shortfall that survives a round is a pair whose rejects ran worse than its
            # decisive rate implies; doubling the over-draw ends it in one or two more.
            overdraw *= 2

        if not drawn_items:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(drawn_items), chosen[np.concatenate(drawn_slots)]

    def _draw_dense(
        self,
        scores: np.ndarray,
        idx_a: np.ndarray,
        idx_b: np.ndarray,
        wanted: np.ndarray,
        capacity: np.ndarray,
        chosen: np.ndarray,
        rng: np.random.Generator,
        decisive_only: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw each chosen pair's quota by ranking, for quotas near what a pair holds.

        Gives every item a random key and keeps the `wanted[p]` smallest per pair, which is a
        draw without replacement whatever the quota's share of the pool -- where rejection
        degrades to coupon collecting once a pair is nearly exhausted. Non-decisive items are
        pushed to `inf` so they rank last.

        Blocks over pairs rather than items because a pair's items have to be ranked
        together.
        """
        items: list[np.ndarray] = []
        pairs: list[np.ndarray] = []
        n_items = len(scores)
        for start in range(0, len(chosen), _PAIR_CHUNK):
            columns = chosen[start : start + _PAIR_CHUNK]
            block = wanted[columns]
            deepest = int(block.max())
            if deepest == 0:
                continue

            keys = rng.random((n_items, len(columns)))
            if decisive_only:
                difference = scores[:, idx_a[columns]] - scores[:, idx_b[columns]]
                keys[np.abs(difference) <= self.tolerance] = np.inf
            ranked = np.argpartition(keys, deepest - 1, axis=0)[:deepest]
            within = np.argsort(
                np.take_along_axis(keys, ranked, axis=0), axis=0, kind="stable"
            )
            ranked = np.take_along_axis(ranked, within, axis=0)

            # Row r of `ranked` is the r-th pick of every pair in the block, so keeping the
            # rows below each pair's own quota is what makes the depth per pair unequal.
            taken = np.arange(deepest)[:, None] < block[None, :]
            items.append(ranked[taken])
            pairs.append(np.broadcast_to(columns, ranked.shape)[taken])

        if not items:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(items), np.concatenate(pairs)

    def _cell_pairs(self, cell: _Cell) -> tuple[np.ndarray, np.ndarray]:
        """Return the local index pairs that may compete in a cell."""
        idx_a, idx_b = np.triu_indices(cell.n_competitors, k=1)
        if not self._excluded:
            return idx_a, idx_b
        keep = np.fromiter(
            (
                frozenset((cell.competitors[a], cell.competitors[b]))
                not in self._excluded
                for a, b in zip(idx_a.tolist(), idx_b.tolist(), strict=True)
            ),
            dtype=bool,
            count=len(idx_a),
        )
        return idx_a[keep], idx_b[keep]

    def _check_graph(self, dataset) -> None:
        """Warn on thin coverage and refuse a matchup graph that is not connected.

        Bradley-Terry ratings are only jointly identified on a connected graph: with two
        components the ridge on the Hessian still produces numbers, but the two groups'
        ratings are not comparable and nothing in the output says so.
        """
        n = dataset.n_competitors
        missing = set(self.competitor_table()["competitor"]) - set(dataset.competitors)
        if missing:
            warnings.warn(
                f"{len(missing)} competitors won no sampled contests and are absent from "
                f"the dataset: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}",
                stacklevel=3,
            )

        seen = np.zeros(n, dtype=bool)
        adjacency: list[list[int]] = [[] for _ in range(n)]
        for a, b in dataset.pairs:
            adjacency[a].append(b)
            adjacency[b].append(a)
        stack = [0]
        seen[0] = True
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
        if not seen.all():
            unreached = [dataset.competitors[i] for i in np.flatnonzero(~seen)]
            raise ValueError(
                "The matchup graph is not connected, so the ratings of these competitors "
                f"are not comparable to the rest: {unreached[:10]}. Raise n_contests or "
                "relax the exclusions."
            )

        contests = np.bincount(
            dataset.pairs.reshape(-1),
            weights=np.repeat(dataset.counts, 2),
            minlength=n,
        )
        median = float(np.median(contests))
        thin = [
            dataset.competitors[i]
            for i in np.flatnonzero(contests < median * _THIN_COVERAGE_FRACTION)
        ]
        if thin:
            warnings.warn(
                f"{len(thin)} competitors have under {_THIN_COVERAGE_FRACTION:.0%} of the "
                f"median contest count ({median:.0f}): {thin[:5]}",
                stacklevel=3,
            )

        # An equal pair budget promises equal contests per competitor, so a spread is a
        # report on where the data would not support it rather than a property of the scheme.
        if self.pair_budget == "equal" and contests.size and contests.max() > 0:
            spread = (contests.max() - contests.min()) / contests.max()
            if spread > _PAIR_BALANCE_SPREAD:
                leanest = dataset.competitors[int(np.argmin(contests))]
                warnings.warn(
                    f"pair_budget='equal' realised contest counts spanning "
                    f"{contests.min():.0f}-{contests.max():.0f} ({spread:.1%} apart), not "
                    f"the equal split it grants: some pairs hold fewer comparisons than "
                    f"their share and were capped at what they hold, leaving "
                    f"{leanest!r} leanest. See the cell summary's `pairs_capped`.",
                    stacklevel=3,
                )

    # ------------------------------------------------------------- diagnostics

    def write_diagnostics(self, cell_summary: pd.DataFrame, prefix: str) -> None:
        """Write the per-cell summary and the failed-load log to `output_dir`."""
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        cell_summary.to_csv(
            os.path.join(self.output_dir, f"{prefix}_cell_summary.csv"), index=False
        )
        self.failed_loads.to_csv(
            os.path.join(self.output_dir, f"{prefix}_failed_loads.csv"), index=False
        )


# The loader, bootstrap mode, budget and view one bootstrap worker draws from. Module-level
# because a `Pool` worker cannot be handed a bound method, and set once per worker rather than
# per replicate so that a fork shares the loader's score matrices rather than pickling ~30 MB
# of them for every seed.
_BOOTSTRAP_TASK: Optional[tuple["BTDataLoader", str, Optional[int], dict[str, Any]]] = (
    None
)


def _init_bootstrap_worker(
    task: tuple["BTDataLoader", str, Optional[int], dict[str, Any]],
) -> None:
    """Install one bootstrap worker's loader and view."""
    global _BOOTSTRAP_TASK
    _BOOTSTRAP_TASK = task


def _bootstrap_replicate(seed: int) -> tuple[list[str], np.ndarray]:
    """Sample and fit one replicate, returning its competitor names and rescaled ratings."""
    from src.elo.bradley_terry import BradleyTerry

    assert _BOOTSTRAP_TASK is not None, "worker was not initialised"
    loader, bootstrap, n_contests, view = _BOOTSTRAP_TASK
    with warnings.catch_warnings():
        # Every replicate warns about the same thin cells, and with one process per replicate
        # each would say so again; the point estimate has reported them already.
        warnings.simplefilter("once")
        dataset, _ = loader.build(
            seed=int(seed), bootstrap=bootstrap, n_contests=n_contests, **view
        )
    # A fresh model per replicate, so a previous fit's ratings never warm-start this one.
    model = BradleyTerry(n_competitors=dataset.n_competitors)
    return dataset.competitors, model.rescale(model.fit(dataset).ratings)


def _outcomes(score_a: np.ndarray, score_b: np.ndarray, tolerance: float) -> np.ndarray:
    """Map score pairs to 1.0 (a wins), 0.5 (tie) or 0.0 (b wins)."""
    difference = score_a.astype(np.float64) - score_b.astype(np.float64)
    outcome = np.where(difference > 0.0, 1.0, 0.0)
    return np.where(np.abs(difference) <= tolerance, 0.5, outcome)


def _unordered_pairs(items: Sequence[str]) -> Iterable[tuple[str, str]]:
    """Yield every unordered pair drawn from `items`."""
    for i, first in enumerate(items):
        for second in items[i + 1 :]:
            yield first, second


def _align_cell(
    entry: tuple[CellSpec, list[_LoadResult]],
) -> tuple[CellSpec, Optional[_Cell]]:
    """Reduce one cell's per-model results to a single aligned score matrix.

    An item is kept only when every competitor in the cell has a finite run-average for it,
    so a contest always compares two models on the same question. A competitor with no items
    simply does not compete in this cell, and a cell left with fewer than two competitors or
    no items is dropped.
    """
    spec, results = entry
    results = sorted(
        (result for result in results if result.items.size),
        key=lambda result: result.competitor,
    )
    if len(results) < 2:
        return spec, None

    items = results[0].items
    for result in results[1:]:
        items = np.intersect1d(items, result.items, assume_unique=True)
        if items.size == 0:
            return spec, None

    scores = np.empty((items.size, len(results)), dtype=np.float32)
    runs = np.empty((items.size, len(results)), dtype=np.int16)
    for column, result in enumerate(results):
        take = np.searchsorted(result.items, items)
        scores[:, column] = result.values[take]
        runs[:, column] = result.n_runs_used[take]

    return spec, _Cell(
        spec=spec,
        competitors=[result.competitor for result in results],
        items=items,
        scores=scores,
        n_runs_used=runs,
    )
