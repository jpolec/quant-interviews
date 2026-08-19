from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Mapping, Sequence


class WindowMode(str, Enum):
    ROLLING = "ROLLING"
    EXPANDING = "EXPANDING"


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train: tuple[date, ...]
    validation: tuple[date, ...]
    test: tuple[date, ...]
    embargo: tuple[date, ...]
    mode: WindowMode

    def __post_init__(self) -> None:
        if isinstance(self.fold_id, bool) or self.fold_id < 1:
            raise ValueError("fold_id must be a positive integer")
        if not isinstance(self.mode, WindowMode):
            raise TypeError("mode must be WindowMode")
        if not self.train or not self.test:
            raise ValueError("train and test windows must be non-empty")
        all_sessions = self.train + self.validation + self.embargo + self.test
        if tuple(sorted(all_sessions)) != all_sessions:
            raise ValueError("fold sessions are not chronologically ordered")
        if len(set(all_sessions)) != len(all_sessions):
            raise ValueError("fold windows overlap")


def walk_forward_folds(
    sessions: Sequence[date],
    *,
    train_sessions: int,
    validation_sessions: int,
    test_sessions: int,
    embargo_sessions: int = 0,
    step_sessions: int | None = None,
    mode: WindowMode = WindowMode.ROLLING,
) -> tuple[Fold, ...]:
    if not isinstance(mode, WindowMode):
        raise TypeError("mode must be WindowMode")
    ordered = tuple(sessions)
    if ordered != tuple(sorted(set(ordered))):
        raise ValueError("sessions must be strictly increasing and unique")
    if min(train_sessions, test_sessions) <= 0:
        raise ValueError("train_sessions and test_sessions must be positive")
    if min(validation_sessions, embargo_sessions) < 0:
        raise ValueError("validation and embargo lengths must be non-negative")
    step = test_sessions if step_sessions is None else step_sessions
    if step <= 0:
        raise ValueError("step_sessions must be positive")
    first_test_start = (
        train_sessions + validation_sessions + embargo_sessions
    )
    folds: list[Fold] = []
    fold_id = 1
    test_start = first_test_start
    while test_start + test_sessions <= len(ordered):
        validation_end = test_start - embargo_sessions
        validation_start = validation_end - validation_sessions
        if mode is WindowMode.ROLLING:
            train_start = validation_start - train_sessions
        else:
            train_start = 0
        train_end = validation_start
        if train_start < 0:
            break
        folds.append(
            Fold(
                fold_id=fold_id,
                train=ordered[train_start:train_end],
                validation=ordered[validation_start:validation_end],
                embargo=ordered[validation_end:test_start],
                test=ordered[test_start : test_start + test_sessions],
                mode=mode,
            )
        )
        fold_id += 1
        test_start += step
    if not folds:
        raise ValueError("INSUFFICIENT_HISTORY: policy produces zero folds")
    return tuple(folds)


def purge_overlapping_supports(
    candidate_times: Sequence[date],
    support_start: Mapping[date, date],
    support_end: Mapping[date, date],
    *,
    protected_intervals: Sequence[tuple[date, date]],
) -> tuple[date, ...]:
    """Retain candidates whose complete half-open support avoids protected sets."""

    ordered = tuple(candidate_times)
    if ordered != tuple(sorted(set(ordered))):
        raise ValueError("candidate_times must be strictly increasing and unique")
    if not protected_intervals:
        raise ValueError("protected_intervals must be non-empty")
    protected = tuple(protected_intervals)
    if any(end <= start for start, end in protected):
        raise ValueError("protected intervals must be non-empty and half-open")
    missing = [
        timestamp
        for timestamp in ordered
        if timestamp not in support_start or timestamp not in support_end
    ]
    if missing:
        raise ValueError(f"missing support endpoint for {missing[:3]}")
    retained: list[date] = []
    for timestamp in ordered:
        start = support_start[timestamp]
        end = support_end[timestamp]
        if end <= start:
            raise ValueError("candidate support must be non-empty and half-open")
        overlaps = any(
            start < protected_end and end > protected_start
            for protected_start, protected_end in protected
        )
        if not overlaps:
            retained.append(timestamp)
    return tuple(retained)


def purge_overlapping_labels(
    feature_times: Sequence[date],
    label_end: Mapping[date, date],
    *,
    test_start: date,
    test_end: date,
) -> tuple[date, ...]:
    """Compatibility wrapper for label support [feature_time, label_end)."""

    return purge_overlapping_supports(
        feature_times,
        {timestamp: timestamp for timestamp in feature_times},
        label_end,
        protected_intervals=((test_start, test_end),),
    )
