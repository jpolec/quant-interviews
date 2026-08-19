from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True)
class InterviewScore:
    interviewer_id: str
    case_id: str
    score: float
    anchor_score: float | None = None
    anchor_version: str | None = None

    def __post_init__(self) -> None:
        if not self.interviewer_id or not self.case_id:
            raise ValueError("interviewer and case identity are required")
        if not isfinite(self.score) or not 0.0 <= self.score <= 5.0:
            raise ValueError("interviewer score must lie in [0, 5]")
        if self.anchor_score is not None and (
            not isfinite(self.anchor_score) or not 0.0 <= self.anchor_score <= 5.0
        ):
            raise ValueError("anchor score must lie in [0, 5]")
        if (self.anchor_score is None) != (self.anchor_version is None):
            raise ValueError("anchor score and version must be supplied together")
        if self.anchor_version is not None and not self.anchor_version.strip():
            raise ValueError("anchor_version must be non-empty")


@dataclass(frozen=True)
class InterviewerDiagnostic:
    interviewer_id: str
    observations: int
    mean_signed_difference: float
    root_mean_squared_difference: float
    reference: str


def interviewer_diagnostics(
    records: list[InterviewScore],
    *,
    minimum_observations: int = 1,
) -> tuple[InterviewerDiagnostic, ...]:
    if not records:
        raise ValueError("at least one interview score is required")
    if minimum_observations <= 0:
        raise ValueError("minimum_observations must be positive")
    identities = [(record.interviewer_id, record.case_id) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("interviewer-case observations must be unique")
    by_case: dict[str, list[InterviewScore]] = defaultdict(list)
    for record in records:
        by_case[record.case_id].append(record)
    authoritative: dict[str, tuple[float, str] | None] = {}
    for case_id, case_records in by_case.items():
        anchors = {
            (record.anchor_score, record.anchor_version)
            for record in case_records
            if record.anchor_score is not None
        }
        if len(anchors) > 1:
            raise ValueError(f"case {case_id} has contradictory anchors")
        authoritative[case_id] = next(iter(anchors)) if anchors else None

    residuals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        anchor = authoritative[record.case_id]
        if anchor is not None:
            reference = anchor[0]
            reference_name = f"independent_anchor@{anchor[1]}"
        else:
            peers = [
                peer.score
                for peer in by_case[record.case_id]
                if peer.interviewer_id != record.interviewer_id
            ]
            if not peers:
                raise ValueError(
                    f"case {record.case_id} has neither anchor nor peer reference"
                )
            reference = sum(peers) / len(peers)
            reference_name = "leave_one_out_panel"
        residuals[(record.interviewer_id, reference_name)].append(
            record.score - reference
        )
    diagnostics: list[InterviewerDiagnostic] = []
    for (interviewer_id, reference_name), values in sorted(residuals.items()):
        if len(values) < minimum_observations:
            continue
        mean = sum(values) / len(values)
        rmse = sqrt(sum(value * value for value in values) / len(values))
        diagnostics.append(
            InterviewerDiagnostic(
                interviewer_id=interviewer_id,
                observations=len(values),
                mean_signed_difference=mean,
                root_mean_squared_difference=rmse,
                reference=reference_name,
            )
        )
    if not diagnostics:
        raise ValueError("no interviewer has enough valid calibration observations")
    return tuple(diagnostics)


def pairwise_agreement(
    scores_a: list[float],
    scores_b: list[float],
    tolerance: float = 0.5,
) -> float:
    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("aligned non-empty score vectors required")
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    values = scores_a + scores_b
    if any(not isfinite(score) or not 0.0 <= score <= 5.0 for score in values):
        raise ValueError("scores must lie in [0, 5]")
    return sum(abs(a - b) <= tolerance for a, b in zip(scores_a, scores_b)) / len(scores_a)


def standard_error(values: list[float]) -> float:
    if len(values) < 2 or any(not isfinite(value) for value in values):
        raise ValueError("standard_error requires at least two finite values")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return sqrt(variance / len(values))
