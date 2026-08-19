from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

DecisionStatus = Literal["PASS", "CONDITIONAL_PASS", "REJECT"]
T = TypeVar("T")


@dataclass(frozen=True)
class ConventionSchema:
    currency: str
    rate_compounding: str
    day_count: str
    discounting_curve: str
    projection_curve: str | None = None
    credit_convention: str | None = None
    exercise_convention: str | None = None

    def __post_init__(self) -> None:
        required = {
            "currency": self.currency,
            "rate_compounding": self.rate_compounding,
            "day_count": self.day_count,
            "discounting_curve": self.discounting_curve,
        }
        optional = {
            "projection_curve": self.projection_curve,
            "credit_convention": self.credit_convention,
            "exercise_convention": self.exercise_convention,
        }
        if any(not value.strip() for value in required.values()):
            raise ValueError("required convention identifiers must be non-empty")
        if any(
            value is not None and not value.strip()
            for value in optional.values()
        ):
            raise ValueError("optional convention identifiers cannot be blank")


@dataclass(frozen=True)
class TolerancePolicy:
    absolute: float
    relative: float
    conditional_multiplier: float = 10.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.absolute)
            or not math.isfinite(self.relative)
            or not math.isfinite(self.conditional_multiplier)
            or self.absolute < 0.0
            or self.relative < 0.0
        ):
            raise ValueError("tolerances must be finite and non-negative")
        if self.conditional_multiplier < 1.0:
            raise ValueError(
                "conditional_multiplier must be finite and at least one"
            )

    def limit(self, scale: float = 1.0) -> float:
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("tolerance scale must be finite and positive")
        return self.absolute + self.relative * abs(scale)


@dataclass(frozen=True)
class InvariantResult:
    name: str
    observed: float | str | bool
    expected: float | str | bool
    tolerance: float | None
    passed: bool


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    observed: float | str | None = None
    limit: float | str | None = None


@dataclass(frozen=True)
class EvidenceProvenance:
    run_id: str
    issued_at: str
    owner: str
    policy_version: str
    artifact_hashes: tuple[str, ...]
    exception_scope: str | None = None
    exception_expires_at: str | None = None

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.run_id,
                self.issued_at,
                self.owner,
                self.policy_version,
            )
        ):
            raise ValueError("evidence provenance fields must be non-empty")
        issued = datetime.fromisoformat(self.issued_at.replace("Z", "+00:00"))
        if issued.tzinfo is None:
            raise ValueError("issued_at must include an explicit UTC offset")
        if not self.artifact_hashes or any(
            len(value) != 64
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.artifact_hashes
        ):
            raise ValueError("artifact_hashes must contain lowercase SHA-256 values")
        has_scope = self.exception_scope is not None
        has_expiry = self.exception_expires_at is not None
        if has_scope != has_expiry:
            raise ValueError("exception scope and expiry must be supplied together")
        if self.exception_scope is not None and not self.exception_scope.strip():
            raise ValueError("exception_scope cannot be blank")
        if self.exception_expires_at is not None:
            expiry = datetime.fromisoformat(
                self.exception_expires_at.replace("Z", "+00:00")
            )
            if expiry.tzinfo is None:
                raise ValueError(
                    "exception_expires_at must include an explicit UTC offset"
                )
            if expiry <= issued:
                raise ValueError("exception expiry must be later than issued_at")


@dataclass(frozen=True)
class EvidenceCertificate(Generic[T]):
    status: DecisionStatus
    result: T
    invariants: tuple[InvariantResult, ...]
    violations: tuple[Violation, ...]
    conventions: ConventionSchema
    tolerances: TolerancePolicy
    input_manifest_hash: str
    implementation_version: str
    provenance: EvidenceProvenance | None = None

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "CONDITIONAL_PASS", "REJECT"}:
            raise ValueError("invalid evidence decision status")
        if (
            len(self.input_manifest_hash) != 64
            or self.input_manifest_hash.lower() != self.input_manifest_hash
            or any(
                character not in "0123456789abcdef"
                for character in self.input_manifest_hash
            )
        ):
            raise ValueError("input_manifest_hash must be a lowercase SHA-256")
        if not self.implementation_version.strip():
            raise ValueError("implementation_version must be non-empty")
        if self.status != "PASS" and not self.violations:
            raise ValueError("non-PASS certificates require an explicit violation")
        if self.provenance is None:
            object.__setattr__(
                self,
                "provenance",
                EvidenceProvenance(
                    run_id=(
                        f"{self.implementation_version}:"
                        f"{self.input_manifest_hash[:16]}"
                    ),
                    issued_at="2026-07-30T16:00:00+02:00",
                    owner="CH07 deterministic evidence fixture owner",
                    policy_version=f"{self.implementation_version}.policy-v1",
                    artifact_hashes=(self.input_manifest_hash,),
                ),
            )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, float):
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        if value != value:
            return "NaN"
    return value


def stable_manifest_hash(payload: Any) -> str:
    serialised = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def certificate_manifest(
    *,
    inputs: Any,
    algorithm_config: Any,
    policy: Any,
    conventions: Any,
    implementation_version: str,
) -> dict[str, Any]:
    """Build the complete semantic payload signed by a certificate hash."""

    if not implementation_version.strip():
        raise ValueError("implementation_version must be non-empty")
    return {
        "inputs": inputs,
        "algorithm_config": algorithm_config,
        "policy": policy,
        "conventions": conventions,
        "implementation_version": implementation_version,
    }


def decision_from_invariants(
    invariants: tuple[InvariantResult, ...],
    *,
    conditional: bool = False,
) -> DecisionStatus:
    if any(not invariant.passed for invariant in invariants):
        return "REJECT"
    return "CONDITIONAL_PASS" if conditional else "PASS"
