"""Train/val/test splits by whole run (splitting windows of one run leaks between sets)."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import NamedTuple

import numpy as np

from heatpde.data.io import RunInfo

Predicate = Callable[[RunInfo], bool]


class Split(NamedTuple):
    train: list[RunInfo]
    val: list[RunInfo]
    test: list[RunInfo]


def matches(criteria: Mapping[str, Collection]) -> Predicate:
    """``matches({"material": ["brass"], "period": [30]})``: every listed field in its values.

    Empty criteria match nothing.
    """
    return lambda i: bool(criteria) and all(getattr(i, k) in v for k, v in criteria.items())


def split_by(infos: Sequence[RunInfo], test: Predicate, val: Predicate = lambda _: False) -> Split:
    split = Split([], [], [])
    for i in infos:
        (split.test if test(i) else split.val if val(i) else split.train).append(i)
    return split


def holdout_periods(
    infos: Sequence[RunInfo], test: Collection[float], val: Collection[float] = ()
) -> Split:
    return split_by(infos, lambda i: i.period in test, lambda i: i.period in val)


def holdout_materials(
    infos: Sequence[RunInfo], test: Collection[str | None], val: Collection[str | None] = ()
) -> Split:
    return split_by(infos, lambda i: i.material in test, lambda i: i.material in val)


def random_split(
    infos: Sequence[RunInfo], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0
) -> Split:
    idx = np.random.default_rng(seed).permutation(len(infos))
    n_test, n_val = round(test_frac * len(infos)), round(val_frac * len(infos))
    pick = [infos[k] for k in idx]
    return Split(pick[n_test + n_val :], pick[n_test : n_test + n_val], pick[:n_test])
