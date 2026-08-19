from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


GateState = Literal["PASS", "REJECT"]
DecisionState = Literal["PASS", "CONDITIONAL_PASS", "REJECT"]
Operator = Literal["<", "<=", ">", ">=", "=="]


def canonical_value(value: Any) -> Any:
    """Convert evidence to strict, deterministic JSON without lossy rounding."""

    if is_dataclass(value):
        return canonical_value(asdict(value))
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(column) for column in value.columns],
            "index": [canonical_value(item) for item in value.index],
            "rows": [
                [canonical_value(item) for item in row]
                for row in value.itertuples(index=False, name=None)
            ],
        }
    if isinstance(value, pd.Series):
        return {
            "name": None if value.name is None else str(value.name),
            "index": [canonical_value(item) for item in value.index],
            "values": [canonical_value(item) for item in value.to_numpy()],
        }
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            raise ValueError("Evidence timestamps must be timezone-aware.")
        return value.tz_convert("UTC").isoformat()
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": canonical_value(value.tolist()),
        }
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Evidence cannot contain NaN or infinity.")
        return {"$float64_hex": value.hex()}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Evidence mappings require string keys.")
        return {
            key: canonical_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported evidence value: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def evidence_digest(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class GatePolicy:
    gate_id: str
    metric: str
    operator: Operator
    limit: float | bool
    unit: str
    hard: bool
    policy_version: str


@dataclass(frozen=True)
class GateRecord:
    gate_id: str
    metric: str
    value: float | bool
    operator: Operator
    limit: float | bool
    unit: str
    hard: bool
    policy_version: str
    evidence_sha256: str
    state: GateState
    reason: str


def evaluate_gate(
    policy: GatePolicy,
    value: float | bool,
    *,
    evidence: Any,
) -> GateRecord:
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Gate values must be finite.")
    predicates = {
        "<": lambda left, right: left < right,
        "<=": lambda left, right: left <= right,
        ">": lambda left, right: left > right,
        ">=": lambda left, right: left >= right,
        "==": lambda left, right: left == right,
    }
    passed = bool(predicates[policy.operator](value, policy.limit))
    return GateRecord(
        gate_id=policy.gate_id,
        metric=policy.metric,
        value=value,
        operator=policy.operator,
        limit=policy.limit,
        unit=policy.unit,
        hard=policy.hard,
        policy_version=policy.policy_version,
        evidence_sha256=evidence_digest(evidence),
        state="PASS" if passed else "REJECT",
        reason=(
            f"{policy.metric}={value!r} {policy.operator} {policy.limit!r}"
            if passed
            else f"predicate failed: {policy.metric}={value!r} "
            f"not {policy.operator} {policy.limit!r}"
        ),
    )


def reduce_gate_records(
    required_gate_ids: Sequence[str],
    records: Sequence[GateRecord],
    *,
    waived_gate_ids: Sequence[str] = (),
) -> tuple[DecisionState, str]:
    required = tuple(required_gate_ids)
    if len(required) != len(set(required)) or not required:
        raise ValueError("Required gate IDs must be unique and nonempty.")
    by_id = {record.gate_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Gate records contain duplicate IDs.")
    missing = sorted(set(required) - set(by_id))
    if missing:
        return "REJECT", f"missing required gates: {', '.join(missing)}"
    unknown_waivers = sorted(set(waived_gate_ids) - set(required))
    if unknown_waivers:
        raise ValueError(f"Waivers reference unknown gates: {unknown_waivers}")
    hard_failures = [
        record.gate_id
        for record in records
        if record.gate_id in required and record.state == "REJECT" and record.hard
    ]
    if hard_failures:
        return "REJECT", f"hard gate failures: {', '.join(sorted(hard_failures))}"
    unwaived = [
        record.gate_id
        for record in records
        if record.gate_id in required
        and record.state == "REJECT"
        and record.gate_id not in waived_gate_ids
    ]
    if unwaived:
        return "REJECT", f"unwaived gate failures: {', '.join(sorted(unwaived))}"
    if waived_gate_ids:
        return "CONDITIONAL_PASS", "enumerated soft-gate waivers applied"
    return "PASS", "all required gates passed"


def seal_evidence_bundle(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Sign the complete immutable payload; the identifier is not self-hashed."""

    required = {
        "raw_data",
        "splits",
        "parameters",
        "policies",
        "results",
        "source_hashes",
        "environment",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Evidence payload is incomplete: {missing}")
    canonical = canonical_value(dict(payload))
    return {
        "schema_version": "ch08-evidence-bundle-v1",
        "bundle_id": evidence_digest(canonical),
        "payload": canonical,
    }


def verify_evidence_bundle(bundle: Mapping[str, Any]) -> None:
    if bundle.get("schema_version") != "ch08-evidence-bundle-v1":
        raise ValueError("Unknown evidence-bundle schema.")
    expected = evidence_digest(bundle["payload"])
    if bundle.get("bundle_id") != expected:
        raise ValueError("Evidence bundle digest mismatch.")
