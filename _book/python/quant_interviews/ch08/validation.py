from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Hashable, Iterable, Literal

import pandas as pd


IndexValue = Hashable


@dataclass(frozen=True)
class Fold:
    train_index: tuple[IndexValue, ...]
    test_index: tuple[IndexValue, ...]
    validation_index: tuple[IndexValue, ...] = ()
    embargo_index: tuple[IndexValue, ...] = ()
    raw_train_index: tuple[IndexValue, ...] = ()
    purged_index: tuple[IndexValue, ...] = ()
    interval_semantics: str = "half_open"


@dataclass(frozen=True)
class NestedFold:
    outer: Fold
    inner: tuple[Fold, ...]


def walk_forward_splits(
    n_obs: int,
    *,
    min_train: int,
    test_size: int,
    step: int | None = None,
    rolling_window: int | None = None,
    allow_overlapping_tests: bool = False,
) -> list[Fold]:
    if min_train <= 0 or test_size <= 0 or n_obs < min_train + test_size:
        raise ValueError("Invalid split sizes.")
    governed_step = test_size if step is None else step
    if governed_step <= 0:
        raise ValueError("step must be strictly positive.")
    if not allow_overlapping_tests and governed_step < test_size:
        raise ValueError("step must be at least test_size for disjoint test folds.")
    if rolling_window is not None:
        if rolling_window <= 0:
            raise ValueError("rolling_window must be positive.")
        if rolling_window < min_train:
            raise ValueError("rolling_window cannot be shorter than min_train.")

    folds: list[Fold] = []
    test_start = min_train
    while test_start + test_size <= n_obs:
        train_start = (
            0
            if rolling_window is None
            else max(0, test_start - rolling_window)
        )
        train = tuple(range(train_start, test_start))
        test = tuple(range(test_start, test_start + test_size))
        folds.append(Fold(train_index=train, test_index=test))
        test_start += governed_step
    return folds


def nested_walk_forward_splits(
    n_obs: int,
    *,
    outer_min_train: int,
    outer_test_size: int,
    inner_min_train: int,
    inner_validation_size: int,
    outer_step: int | None = None,
    rolling_window: int | None = None,
    intervals: pd.DataFrame | None = None,
    embargo: timedelta = timedelta(0),
) -> list[NestedFold]:
    """Create and, when intervals are supplied, purge every outer/inner fold."""

    if embargo < timedelta(0):
        raise ValueError("embargo must be nonnegative.")
    if intervals is not None and len(intervals) != n_obs:
        raise ValueError("intervals must contain exactly n_obs rows.")

    outer_folds = walk_forward_splits(
        n_obs,
        min_train=outer_min_train,
        test_size=outer_test_size,
        step=outer_step,
        rolling_window=rolling_window,
    )
    result: list[NestedFold] = []
    for raw_outer in outer_folds:
        outer = raw_outer
        if intervals is not None:
            outer = purged_embargo_split(
                intervals,
                raw_outer.test_index,
                candidate_train_indices=raw_outer.train_index,
                embargo=embargo,
                mode="walk_forward",
            )
            assert_no_interval_overlap(intervals, outer)
        outer_train = outer.train_index
        relative = walk_forward_splits(
            len(outer_train),
            min_train=inner_min_train,
            test_size=inner_validation_size,
        )
        governed_inner: list[Fold] = []
        for fold in relative:
            raw_train = tuple(outer_train[index] for index in fold.train_index)
            validation = tuple(outer_train[index] for index in fold.test_index)
            if intervals is None:
                governed_inner.append(
                    Fold(
                        train_index=raw_train,
                        test_index=(),
                        validation_index=validation,
                        raw_train_index=raw_train,
                    )
                )
            else:
                purged = purged_embargo_split(
                    intervals,
                    validation,
                    candidate_train_indices=raw_train,
                    embargo=embargo,
                    mode="walk_forward",
                )
                assert_no_interval_overlap(intervals, purged)
                governed_inner.append(
                    Fold(
                        train_index=purged.train_index,
                        test_index=(),
                        validation_index=validation,
                        embargo_index=purged.embargo_index,
                        raw_train_index=purged.raw_train_index,
                        purged_index=purged.purged_index,
                    )
                )
        inner = tuple(governed_inner)
        if not inner:
            raise ValueError("Outer training data cannot support an inner fold.")
        result.append(NestedFold(outer=outer, inner=inner))
    return result


def _validated_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    if not {"start", "end"}.issubset(intervals.columns):
        raise ValueError("intervals must contain start and end columns.")
    if intervals.index.has_duplicates:
        raise ValueError("interval index must be unique.")
    result = intervals.loc[:, ["start", "end"]].copy()
    result["start"] = pd.to_datetime(result["start"], utc=True, errors="raise")
    result["end"] = pd.to_datetime(result["end"], utc=True, errors="raise")
    if result[["start", "end"]].isna().any().any():
        raise ValueError("Intervals cannot contain missing timestamps.")
    if (result["start"] >= result["end"]).any():
        raise ValueError("Intervals use half-open semantics and require start < end.")
    return result


def _overlaps_any(
    start: pd.Timestamp,
    end: pd.Timestamp,
    test_intervals: pd.DataFrame,
) -> bool:
    return bool(
        (
            (test_intervals["start"] < end)
            & (test_intervals["end"] > start)
        ).any()
    )


def purged_embargo_split(
    intervals: pd.DataFrame,
    test_indices: Iterable[IndexValue],
    *,
    candidate_train_indices: Iterable[IndexValue] | None = None,
    embargo: timedelta = timedelta(0),
    mode: Literal["walk_forward", "combinatorial"] = "walk_forward",
) -> Fold:
    """Purge against the exact union of half-open test-label intervals."""

    if embargo < timedelta(0):
        raise ValueError("embargo must be nonnegative.")
    if mode not in {"walk_forward", "combinatorial"}:
        raise ValueError("mode must be walk_forward or combinatorial.")
    data = _validated_intervals(intervals)
    test = tuple(dict.fromkeys(test_indices))
    if not test:
        raise ValueError("test_indices must be nonempty.")
    missing_test = [index for index in test if index not in data.index]
    if missing_test:
        raise KeyError(f"Unknown test indices: {missing_test!r}")
    test_intervals = data.loc[list(test)]
    first_test_start = test_intervals["start"].min()

    if candidate_train_indices is None:
        candidates = [
            index
            for index, row in data.iterrows()
            if index not in test
            and (
                mode == "combinatorial"
                or row["start"] < first_test_start
            )
        ]
    else:
        candidates = list(dict.fromkeys(candidate_train_indices))
        missing_candidates = [
            index for index in candidates if index not in data.index
        ]
        if missing_candidates:
            raise KeyError(f"Unknown training indices: {missing_candidates!r}")
        candidates = [index for index in candidates if index not in test]
        if mode == "walk_forward" and any(
            data.loc[index, "start"] >= first_test_start for index in candidates
        ):
            raise ValueError("Walk-forward training candidates must precede test.")

    kept: list[IndexValue] = []
    purged: list[IndexValue] = []
    embargoed_from_train: list[IndexValue] = []
    embargo_intervals = [
        (row.end, row.end + embargo)
        for row in test_intervals.itertuples()
        if embargo > timedelta(0)
    ]
    for index in candidates:
        row = data.loc[index]
        if _overlaps_any(row["start"], row["end"], test_intervals):
            purged.append(index)
            continue
        in_embargo = any(
            embargo_start <= row["start"] < embargo_end
            for embargo_start, embargo_end in embargo_intervals
        )
        if in_embargo:
            embargoed_from_train.append(index)
            continue
        kept.append(index)
    # In a pure walk-forward fold, current training rows precede the test and a
    # post-test embargo cannot remove them.  We still publish the future rows
    # made ineligible for subsequent folds, so embargo evidence is not a hidden
    # no-op.  In combinatorial mode the same rows may already be candidates.
    post_test_embargo = [
        index
        for index, row in data.iterrows()
        if index not in test
        and any(
            embargo_start <= row["start"] < embargo_end
            for embargo_start, embargo_end in embargo_intervals
        )
    ]
    embargoed = tuple(
        dict.fromkeys([*embargoed_from_train, *post_test_embargo])
    )
    return Fold(
        train_index=tuple(kept),
        test_index=test,
        embargo_index=embargoed,
        raw_train_index=tuple(candidates),
        purged_index=tuple(purged),
    )


def assert_no_interval_overlap(intervals: pd.DataFrame, fold: Fold) -> None:
    data = _validated_intervals(intervals)
    test_intervals = data.loc[list(fold.test_index)]
    for index in fold.train_index:
        row = data.loc[index]
        if _overlaps_any(row["start"], row["end"], test_intervals):
            raise AssertionError(
                f"Training interval {index!r} overlaps the test union."
            )


def assert_nested_temporal_integrity(
    intervals: pd.DataFrame,
    folds: Iterable[NestedFold],
) -> None:
    """Fail unless every outer and inner certificate is chronological and clean."""

    data = _validated_intervals(intervals)
    for outer_number, nested in enumerate(folds):
        assert_no_interval_overlap(data, nested.outer)
        if nested.outer.train_index and not (
            data.loc[list(nested.outer.train_index), "start"].max()
            < data.loc[list(nested.outer.test_index), "start"].min()
        ):
            raise AssertionError(f"Outer fold {outer_number} is not chronological.")
        outer_test = set(nested.outer.test_index)
        for inner_number, inner in enumerate(nested.inner):
            if outer_test.intersection(inner.train_index):
                raise AssertionError("Inner training contains an outer-test row.")
            if outer_test.intersection(inner.validation_index):
                raise AssertionError("Inner validation contains an outer-test row.")
            proxy = Fold(
                train_index=inner.train_index,
                test_index=inner.validation_index,
            )
            assert_no_interval_overlap(data, proxy)
            if inner.train_index and not (
                data.loc[list(inner.train_index), "start"].max()
                < data.loc[list(inner.validation_index), "start"].min()
            ):
                raise AssertionError(
                    f"Inner fold {outer_number}.{inner_number} is not chronological."
                )
