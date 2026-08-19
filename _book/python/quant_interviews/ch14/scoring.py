from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from math import isfinite, sqrt
import re
from typing import Literal, Mapping

from .takehome_case import canonical_hash


HEX64 = re.compile(r"^[0-9a-f]{64}$")
CanonicalState = Literal[
    "ACCEPT",
    "CONDITIONAL_ACCEPT",
    "PANEL_REVIEW",
    "REJECT",
    "INSUFFICIENT_EVIDENCE",
]


@dataclass(frozen=True)
class ScoreCard:
    correctness: float
    reasoning: float
    implementation: float
    judgment: float
    communication: float

    def __post_init__(self) -> None:
        if any(
            not isfinite(value) or not 0.0 <= value <= 5.0
            for value in self.as_dict().values()
        ):
            raise ValueError("scores must be finite and lie in [0, 5]")

    def as_dict(self) -> dict[str, float]:
        return {
            "correctness": self.correctness,
            "reasoning": self.reasoning,
            "implementation": self.implementation,
            "judgment": self.judgment,
            "communication": self.communication,
        }


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    competency: str
    observation: str
    quality: float
    source_hash: str
    observed_at: str
    scorer_id: str

    def __post_init__(self) -> None:
        if (
            not self.evidence_id
            or not self.competency
            or not self.observation
            or not self.scorer_id
        ):
            raise ValueError("evidence identity, competency, observation and scorer required")
        if HEX64.fullmatch(self.source_hash) is None:
            raise ValueError("evidence source_hash must be a lowercase SHA-256")
        try:
            observed = datetime.fromisoformat(self.observed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("evidence observed_at must be ISO date-time") from error
        if observed.tzinfo is None:
            raise ValueError("evidence observed_at must include a timezone")
        if not isfinite(self.quality) or not 0.0 <= self.quality <= 1.0:
            raise ValueError("evidence quality must lie in [0, 1]")


@dataclass(frozen=True)
class CompletenessCertificate:
    weighted_completeness: float
    missing_evidence: tuple[str, ...]
    category_quality: Mapping[str, float]
    extra_evidence_categories: tuple[str, ...]


@dataclass(frozen=True)
class HiringDecision:
    weighted_score: float
    completeness: float
    hard_floor_violations: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    dissent: tuple[str, ...]
    integrity_failures: tuple[str, ...]
    state: CanonicalState
    recommendation: str
    policy_version: str
    decision_hash: str


def weighted_score(card: ScoreCard, weights: Mapping[str, float]) -> float:
    values = card.as_dict()
    if set(weights) != set(values):
        raise ValueError("weights must cover every competency exactly once")
    if any(not isfinite(weight) or weight < 0.0 for weight in weights.values()):
        raise ValueError("weights must be finite and non-negative")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("weight sum must be positive")
    return sum(values[name] * weights[name] for name in values) / total


def evidence_completeness(
    required_quality: Mapping[str, float],
    required_weights: Mapping[str, float],
    evidence: tuple[EvidenceItem, ...],
) -> CompletenessCertificate:
    if set(required_quality) != set(required_weights) or not required_quality:
        raise ValueError("required quality and weights must cover the same categories")
    if any(
        not isfinite(floor) or not 0.0 <= floor <= 1.0
        for floor in required_quality.values()
    ):
        raise ValueError("evidence quality floors must lie in [0, 1]")
    if any(
        not isfinite(weight) or weight <= 0.0 for weight in required_weights.values()
    ):
        raise ValueError("evidence weights must be finite and strictly positive")
    if len({item.evidence_id for item in evidence}) != len(evidence):
        raise ValueError("evidence IDs must be unique")

    observed: dict[str, float] = {}
    for item in evidence:
        observed[item.competency] = max(
            observed.get(item.competency, 0.0), item.quality
        )
    missing = tuple(
        competency
        for competency, floor in sorted(required_quality.items())
        if observed.get(competency, 0.0) < floor
    )
    total_weight = sum(required_weights.values())
    passed_weight = sum(
        required_weights[competency]
        for competency, floor in required_quality.items()
        if observed.get(competency, 0.0) >= floor
    )
    completeness = passed_weight / total_weight
    extra = tuple(sorted(set(observed) - set(required_quality)))
    return CompletenessCertificate(
        weighted_completeness=completeness,
        missing_evidence=missing,
        category_quality={
            competency: observed.get(competency, 0.0)
            for competency in sorted(required_quality)
        },
        extra_evidence_categories=extra,
    )


def _decision_time(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("decision_time is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("decision_time must be ISO date-time") from error
    if parsed.tzinfo is None:
        raise ValueError("decision_time must include a timezone")
    return parsed.isoformat()


def hiring_decision(
    *,
    card: ScoreCard,
    weights: Mapping[str, float],
    hard_floors: Mapping[str, float],
    required_evidence_quality: Mapping[str, float],
    required_evidence_weights: Mapping[str, float],
    evidence: tuple[EvidenceItem, ...],
    scorer_ids: tuple[str, ...],
    decision_time: str,
    policy_version: str,
    dissent: tuple[str, ...] = (),
    integrity_failures: tuple[str, ...] = (),
    pass_score: float = 3.5,
    strong_hire_score: float = 4.0,
    panel_lower: float = 3.25,
    panel_upper: float = 3.75,
    prior_packet_hash: str | None = None,
) -> HiringDecision:
    thresholds = (pass_score, strong_hire_score, panel_lower, panel_upper)
    if not all(isfinite(value) for value in thresholds):
        raise ValueError("decision thresholds must be finite")
    if not (
        0.0
        <= panel_lower
        < pass_score
        < panel_upper
        < strong_hire_score
        <= 5.0
    ):
        raise ValueError(
            "thresholds require panel_lower < pass_score < panel_upper "
            "< strong_hire_score in [0, 5]"
        )
    if not policy_version.strip():
        raise ValueError("policy_version is required")
    if (
        len(scorer_ids) < 2
        or len(set(scorer_ids)) != len(scorer_ids)
        or any(not value.strip() for value in scorer_ids)
    ):
        raise ValueError("at least two unique scorer identities are required")
    if any(not value.strip() for value in dissent + integrity_failures):
        raise ValueError("dissent and integrity failures must be non-empty strings")
    if prior_packet_hash is not None and HEX64.fullmatch(prior_packet_hash) is None:
        raise ValueError("prior_packet_hash must be a lowercase SHA-256")

    scores = card.as_dict()
    unknown_floors = sorted(set(hard_floors) - set(scores))
    if unknown_floors:
        raise ValueError(f"unknown hard-floor competencies: {unknown_floors}")
    if any(
        not isfinite(floor) or not 0.0 <= floor <= 5.0
        for floor in hard_floors.values()
    ):
        raise ValueError("hard floors must lie in [0, 5]")
    violations = tuple(
        competency
        for competency, floor in sorted(hard_floors.items())
        if scores[competency] < floor
    )
    completeness = evidence_completeness(
        required_evidence_quality,
        required_evidence_weights,
        evidence,
    )
    score = weighted_score(card, weights)

    # Policy precedence is deliberate: reject -> insufficient evidence ->
    # panel review -> strong accept -> conditional accept -> reject.
    if integrity_failures or violations:
        state: CanonicalState = "REJECT"
    elif completeness.missing_evidence:
        state = "INSUFFICIENT_EVIDENCE"
    elif dissent or panel_lower <= score <= panel_upper:
        state = "PANEL_REVIEW"
    elif score >= strong_hire_score:
        state = "ACCEPT"
    elif score >= pass_score:
        state = "CONDITIONAL_ACCEPT"
    else:
        state = "REJECT"
    recommendation = {
        "ACCEPT": "STRONG_HIRE",
        "CONDITIONAL_ACCEPT": "HIRE",
        "PANEL_REVIEW": "PANEL_REVIEW",
        "REJECT": "NO_HIRE",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
    }[state]
    normalized_time = _decision_time(decision_time)
    packet: dict[str, object] = {
        "card": asdict(card),
        "decision_time": normalized_time,
        "dissent": list(dissent),
        "evidence": [asdict(item) for item in evidence],
        "evidence_certificate": asdict(completeness),
        "hard_floors": dict(hard_floors),
        "integrity_failures": list(integrity_failures),
        "policy_version": policy_version,
        "prior_packet_hash": prior_packet_hash,
        "recommendation": recommendation,
        "required_evidence_quality": dict(required_evidence_quality),
        "required_evidence_weights": dict(required_evidence_weights),
        "score_weights": dict(weights),
        "scorer_ids": list(scorer_ids),
        "state": state,
        "thresholds": {
            "panel_lower": panel_lower,
            "panel_upper": panel_upper,
            "pass_score": pass_score,
            "strong_hire_score": strong_hire_score,
        },
        "weighted_score": score,
    }
    digest = canonical_hash(packet)
    return HiringDecision(
        weighted_score=score,
        completeness=completeness.weighted_completeness,
        hard_floor_violations=violations,
        missing_evidence=completeness.missing_evidence,
        dissent=dissent,
        integrity_failures=integrity_failures,
        state=state,
        recommendation=recommendation,
        policy_version=policy_version,
        decision_hash=digest,
    )


def mean_and_se(scores: list[float]) -> tuple[float, float]:
    if not scores or any(
        not isfinite(score) or not 0.0 <= score <= 5.0 for score in scores
    ):
        raise ValueError("scores must be a non-empty finite vector in [0, 5]")
    mean = sum(scores) / len(scores)
    if len(scores) == 1:
        return mean, float("inf")
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    return mean, sqrt(variance / len(scores))
