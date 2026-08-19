from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from hashlib import sha256
import json
from math import isfinite
import re
from typing import Mapping, Sequence


REQUIRED_MANIFEST_FIELDS = (
    "candidate_id",
    "schema_version",
    "code_commit",
    "data_hash",
    "config_hash",
    "environment_hash",
    "result_hash",
)
SUPPORTED_MANIFEST_VERSIONS = frozenset({"1.0"})
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MAX_SAFE_INTEGER = 2**53 - 1


@dataclass(frozen=True)
class SubmissionManifest:
    candidate_id: str
    schema_version: str
    code_commit: str
    data_hash: str
    config_hash: str
    environment_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        if IDENTIFIER.fullmatch(self.candidate_id) is None:
            raise ValueError("candidate_id has an invalid canonical format")
        if self.schema_version not in SUPPORTED_MANIFEST_VERSIONS:
            raise ValueError("unsupported manifest schema version")
        if GIT_COMMIT.fullmatch(self.code_commit) is None:
            raise ValueError("code_commit must be a lowercase Git SHA-1 or SHA-256")
        for field in ("data_hash", "config_hash", "environment_hash", "result_hash"):
            if HEX64.fullmatch(getattr(self, field)) is None:
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class TakeHomeDecision:
    weighted_score: float
    fatal_reasons: tuple[str, ...]
    reproducibility_floor_pass: bool
    decision_supported: bool
    component_count: int
    status: str


def _serialize_number(value: int | float) -> str:
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise ValueError("JCS integers must be exactly representable as IEEE-754")
        return str(value)
    if not isfinite(value):
        raise ValueError("canonical payloads cannot contain NaN or infinity")
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    representation = repr(value).lower()
    if 1.0e-6 <= magnitude < 1.0e21:
        if "e" in representation:
            return format(Decimal(representation), "f")
        return representation[:-2] if representation.endswith(".0") else representation
    if "e" not in representation:
        representation = format(value, ".17e")
    mantissa, exponent_text = representation.split("e")
    if mantissa.endswith(".0"):
        mantissa = mantissa[:-2]
    exponent = int(exponent_text)
    sign = "+" if exponent >= 0 else ""
    return f"{mantissa}e{sign}{exponent}"


def _string(value: str) -> str:
    try:
        value.encode("utf-8")
        value.encode("utf-16be")
    except UnicodeError as error:
        raise ValueError("canonical strings cannot contain lone surrogates") from error
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _canonical(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return _serialize_number(value)
    if isinstance(value, float):
        return _serialize_number(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical mappings require string keys")
        try:
            keys = sorted(value, key=lambda key: key.encode("utf-16be"))
        except UnicodeError as error:
            raise ValueError("canonical keys cannot contain lone surrogates") from error
        return "{" + ",".join(
            _string(key) + ":" + _canonical(value[key]) for key in keys
        ) + "}"
    raise TypeError(f"unsupported canonical type: {type(value).__name__}")


def canonical_bytes(payload: Mapping[str, object]) -> bytes:
    """RFC 8785/JCS canonical JSON bytes for the supported JSON data model."""

    return _canonical(payload).encode("utf-8")


def canonical_hash(payload: Mapping[str, object]) -> str:
    return sha256(canonical_bytes(payload)).hexdigest()


def validate_manifest(
    manifest: SubmissionManifest,
    expected: Mapping[str, str],
    *,
    artifact_bytes: Mapping[str, bytes] | None = None,
) -> tuple[str, ...]:
    manifest_values = asdict(manifest)
    unknown = sorted(set(expected) - set(REQUIRED_MANIFEST_FIELDS))
    if unknown:
        raise ValueError(f"unknown manifest fields: {unknown}")
    issues: list[str] = []
    for field in REQUIRED_MANIFEST_FIELDS:
        actual = manifest_values[field]
        expected_value = expected.get(field)
        if expected_value is None:
            issues.append(f"{field}: expected value missing")
        elif actual != expected_value:
            issues.append(f"{field}: expected {expected_value}, got {actual}")
    if artifact_bytes is not None:
        expected_artifacts = {
            "data": "data_hash",
            "config": "config_hash",
            "environment": "environment_hash",
            "result": "result_hash",
        }
        unknown_artifacts = sorted(set(artifact_bytes) - set(expected_artifacts))
        if unknown_artifacts:
            raise ValueError(f"unknown artifact byte fields: {unknown_artifacts}")
        for artifact, hash_field in expected_artifacts.items():
            content = artifact_bytes.get(artifact)
            if content is None:
                issues.append(f"{artifact}: artifact bytes missing")
            elif not isinstance(content, bytes):
                raise TypeError(f"{artifact}: artifact content must be bytes")
            else:
                actual_hash = sha256(content).hexdigest()
                if actual_hash != manifest_values[hash_field]:
                    issues.append(
                        f"{hash_field}: artifact digest {actual_hash} "
                        f"does not match {manifest_values[hash_field]}"
                    )
    return tuple(issues)


def score_takehome(
    *,
    reproducibility: float,
    research_quality: float,
    implementation: float,
    communication: float,
    leakage: bool,
    fabricated_result: bool,
    core_claim_reproduced: bool,
    decision_supported: bool,
    pass_score: float = 3.5,
    reproducibility_floor: float = 3.0,
    penalty: float = 0.0,
) -> TakeHomeDecision:
    values = (reproducibility, research_quality, implementation, communication)
    if not all(isfinite(value) and 0.0 <= value <= 5.0 for value in values):
        raise ValueError("four take-home scores must be finite and lie in [0, 5]")
    if not all(
        isfinite(value) for value in (pass_score, reproducibility_floor, penalty)
    ):
        raise ValueError("decision thresholds must be finite")
    if (
        not 0.0 <= pass_score <= 5.0
        or not 0.0 <= reproducibility_floor <= 5.0
        or penalty < 0.0
    ):
        raise ValueError("invalid decision threshold or penalty")
    weighted = max(
        0.0,
        0.25 * reproducibility
        + 0.35 * research_quality
        + 0.25 * implementation
        + 0.15 * communication
        - penalty,
    )
    fatal: list[str] = []
    if leakage:
        fatal.append("leakage")
    if fabricated_result:
        fatal.append("fabricated_result")
    if not core_claim_reproduced:
        fatal.append("non_reproducible_core_claim")
    floor_pass = reproducibility >= reproducibility_floor
    if fatal:
        status = "REJECT"
    elif not floor_pass or not decision_supported:
        status = "INSUFFICIENT_EVIDENCE"
    elif weighted >= pass_score:
        status = "ACCEPT"
    else:
        status = "REJECT"
    return TakeHomeDecision(
        weighted_score=weighted,
        fatal_reasons=tuple(fatal),
        reproducibility_floor_pass=floor_pass,
        decision_supported=decision_supported,
        component_count=4,
        status=status,
    )
