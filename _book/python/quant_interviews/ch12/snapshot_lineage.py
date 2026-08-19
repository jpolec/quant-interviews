from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import math
from pathlib import Path
import unicodedata
from typing import Any, Mapping


CANONICAL_FORMAT = "QJ-CANONICAL-JSON-1"
HASH_ALGORITHM = "sha256"


@dataclass(frozen=True)
class Artifact:
    logical_uri: str
    materialized_path: str
    sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        if not self.logical_uri or not self.materialized_path or not self.media_type:
            raise ValueError("artifact identity and resolver hint are required")
        _require_digest(self.sha256, "artifact sha256")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("artifact size must be non-negative")


@dataclass(frozen=True)
class VerificationResult:
    status: str
    manifest_integrity: bool
    verified_artifacts: int
    missing_artifacts: tuple[str, ...]
    changed_artifacts: tuple[str, ...]


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not canonical")
        return {"$float_hex": value.hex()}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite decimals are not canonical")
        return {"$decimal": format(value.normalize(), "f")}
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical datetimes must be timezone-aware")
        normalized = value.astimezone(timezone.utc).isoformat(
            timespec="microseconds"
        )
        return {"$datetime_utc": normalized.replace("+00:00", "Z")}
    if isinstance(value, date):
        return {"$date": value.isoformat()}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical mapping keys must be strings")
        normalized_items: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized_items:
                raise ValueError("canonical-equivalent mapping keys collide")
            normalized_items[normalized_key] = _normalize(item)
        return dict(sorted(normalized_items.items()))
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise TypeError(f"unsupported canonical type: {type(value).__name__}")


def canonical_bytes(payload: Any) -> bytes:
    envelope = {
        "canonical_format": CANONICAL_FORMAT,
        "payload": _normalize(payload),
    }
    return json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return sha256(canonical_bytes(payload)).hexdigest()


def hash_file(
    path: Path,
    *,
    logical_uri: str,
    media_type: str = "application/octet-stream",
) -> Artifact:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return Artifact(
        logical_uri=logical_uri,
        materialized_path=str(path),
        sha256=digest.hexdigest(),
        size_bytes=size,
        media_type=media_type,
    )


def build_manifest(
    files: Mapping[str, Path],
    *,
    config: Mapping[str, Any],
    seed: int,
    implementation_version: str,
    environment: Mapping[str, str],
    output_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if seed < 0 or not implementation_version:
        raise ValueError("seed and implementation_version are invalid")
    if len(files) != len(set(files)):
        raise ValueError("logical artifact URIs must be unique")
    if output_hashes is not None:
        for key, value in output_hashes.items():
            if not key:
                raise ValueError("output logical URI must be non-empty")
            _require_digest(value, f"output hash {key}")
    artifacts = [
        asdict(hash_file(path, logical_uri=logical_uri))
        for logical_uri, path in sorted(files.items())
    ]
    body: dict[str, Any] = {
        "schema_version": 1,
        "canonical_format": CANONICAL_FORMAT,
        "hash_algorithm": HASH_ALGORITHM,
        "artifacts": artifacts,
        "config": dict(config),
        "seed": seed,
        "implementation_version": implementation_version,
        "environment": dict(environment),
        "output_hashes": dict(output_hashes or {}),
    }
    body["manifest_hash"] = canonical_hash(_manifest_identity(body))
    return body


def verify_manifest_integrity(manifest: Mapping[str, Any]) -> bool:
    try:
        candidate = dict(manifest)
        expected = candidate.pop("manifest_hash", None)
        _require_digest(expected, "manifest_hash")
        artifacts = candidate.get("artifacts")
        if not isinstance(artifacts, list):
            return False
        parsed = [Artifact(**raw) for raw in artifacts]
        uris = [item.logical_uri for item in parsed]
        if len(uris) != len(set(uris)):
            return False
        outputs = candidate.get("output_hashes", {})
        if not isinstance(outputs, Mapping):
            return False
        for key, value in outputs.items():
            if not isinstance(key, str) or not key:
                return False
            _require_digest(value, f"output hash {key}")
        return canonical_hash(_manifest_identity(candidate)) == expected
    except (KeyError, TypeError, ValueError):
        return False


def verify_artifact_bytes(
    manifest: Mapping[str, Any],
    *,
    resolver: Mapping[str, Path] | None = None,
    observed_outputs: Mapping[str, Any] | None = None,
) -> VerificationResult:
    integrity = verify_manifest_integrity(manifest)
    missing: list[str] = []
    changed: list[str] = []
    verified = 0
    if not integrity:
        return VerificationResult("REJECT", False, 0, (), ("manifest",))
    artifacts = [Artifact(**raw) for raw in manifest["artifacts"]]
    for artifact in artifacts:
        path = (
            Path(artifact.materialized_path)
            if resolver is None
            else resolver.get(artifact.logical_uri, Path("/__missing__"))
        )
        if not path.is_file():
            missing.append(artifact.logical_uri)
            continue
        observed = hash_file(
            path,
            logical_uri=artifact.logical_uri,
            media_type=artifact.media_type,
        )
        if (
            observed.sha256 != artifact.sha256
            or observed.size_bytes != artifact.size_bytes
        ):
            changed.append(artifact.logical_uri)
        else:
            verified += 1
    declared_outputs = manifest.get("output_hashes", {})
    if declared_outputs:
        if observed_outputs is None:
            changed.extend(f"output:{key}" for key in declared_outputs)
        else:
            for key, expected in declared_outputs.items():
                if key not in observed_outputs or not verify_output_equivalence(
                    expected,
                    observed_outputs[key],
                ):
                    changed.append(f"output:{key}")
    passed = integrity and not missing and not changed
    return VerificationResult(
        status="PASS" if passed else "REJECT",
        manifest_integrity=integrity,
        verified_artifacts=verified,
        missing_artifacts=tuple(sorted(missing)),
        changed_artifacts=tuple(sorted(changed)),
    )


def verify_output_equivalence(expected_hash: str, output: Any) -> bool:
    _require_digest(expected_hash, "expected_hash")
    return canonical_hash(output) == expected_hash


def _manifest_identity(body: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(body)
    identity["artifacts"] = [
        {
            key: value
            for key, value in dict(raw).items()
            if key != "materialized_path"
        }
        for raw in identity.get("artifacts", [])
    ]
    return identity


def _require_digest(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set(
        "0123456789abcdef"
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
