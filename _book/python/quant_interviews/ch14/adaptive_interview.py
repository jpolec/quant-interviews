from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class Question:
    question_id: str
    competency: str
    difficulty: int
    time_minutes: int

    def __post_init__(self) -> None:
        if (
            not self.question_id
            or not self.competency
            or not 1 <= self.difficulty <= 5
            or self.time_minutes <= 0
        ):
            raise ValueError("invalid question metadata")


@dataclass(frozen=True)
class Evidence:
    competency: str
    score: float
    confidence: float

    def __post_init__(self) -> None:
        if (
            not self.competency
            or not isfinite(self.score)
            or not isfinite(self.confidence)
            or not 0.0 <= self.score <= 5.0
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("invalid evidence")


def competency_need(targets: dict[str, float], evidence: Iterable[Evidence]) -> dict[str, float]:
    if any(not key or not isfinite(target) or not 0.0 <= target <= 5.0 for key, target in targets.items()):
        raise ValueError("targets must be named finite values in [0, 5]")
    weighted: dict[str, tuple[float, float]] = {}
    for item in evidence:
        if item.competency not in targets:
            raise ValueError("evidence competency is outside the role target set")
        total, weight = weighted.get(item.competency, (0.0, 0.0))
        weighted[item.competency] = (total + item.score * item.confidence, weight + item.confidence)
    need: dict[str, float] = {}
    for competency, target in targets.items():
        total, weight = weighted.get(competency, (0.0, 0.0))
        estimate = total / weight if weight else 0.0
        confidence = min(1.0, weight)
        need[competency] = max(0.0, target - estimate) + (1.0 - confidence)
    return need


def choose_next_question(
    questions: list[Question],
    asked: set[str],
    targets: dict[str, float],
    evidence: list[Evidence],
    remaining_minutes: int,
) -> Question | None:
    if remaining_minutes < 0:
        raise ValueError("remaining_minutes must be non-negative")
    identifiers = [question.question_id for question in questions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("question IDs must be unique")
    if any(question.competency not in targets for question in questions):
        raise ValueError("question competency is outside the role target set")
    if not asked <= set(identifiers):
        raise ValueError("asked contains an unknown question ID")
    needs = competency_need(targets, evidence)
    eligible = [
        q for q in questions
        if q.question_id not in asked and q.time_minutes <= remaining_minutes
    ]
    if not eligible:
        return None
    # Deterministic information-rate proxy: close the largest calibrated gap
    # per minute, then prefer the shorter question and stable lexical ID.
    return min(
        eligible,
        key=lambda q: (
            -needs[q.competency] / q.time_minutes,
            q.time_minutes,
            q.question_id,
        ),
    )
