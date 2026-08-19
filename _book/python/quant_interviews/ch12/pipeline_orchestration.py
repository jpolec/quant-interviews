from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from math import ceil, isfinite, log2
from threading import RLock
from typing import Iterable, Mapping


class RunState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    PUBLISHED = "PUBLISHED"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dependencies: tuple[str, ...]
    max_retries: int = 0
    cpu_units: int = 1
    memory_mb: int = 256
    checkpointable: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task name is required")
        if (
            self.max_retries < 0
            or self.cpu_units <= 0
            or self.memory_mb <= 0
        ):
            raise ValueError("invalid retry or resource policy")
        if self.name in self.dependencies:
            raise ValueError("task cannot depend on itself")


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    input_hash: str
    code_version: str
    config_hash: str
    task_order: tuple[str, ...]
    state: RunState
    output_hash: str | None = None
    task_specs: tuple[TaskSpec, ...] = ()
    environment_hash: str = "0" * 64
    partition_certificates: tuple[tuple[str, str, str], ...] = ()
    schema_version: str = "run-manifest-v0.10"

    def __post_init__(self) -> None:
        if not self.run_id or not self.code_version or not self.schema_version:
            raise ValueError("run manifest identity is required")
        for field, value in (
            ("input_hash", self.input_hash),
            ("config_hash", self.config_hash),
            ("environment_hash", self.environment_hash),
        ):
            _require_digest(value, field)
        if self.task_specs:
            if topological_order(self.task_specs) != self.task_order:
                raise ValueError("task_order does not match the declared DAG")
        if self.output_hash is not None:
            _require_digest(self.output_hash, "output_hash")
        if self.state in {RunState.SUCCEEDED, RunState.PUBLISHED}:
            if self.output_hash is None or not self.partition_certificates:
                raise ValueError("successful run requires output and partition evidence")
            if any(status != "PASS" for _, _, status in self.partition_certificates):
                raise ValueError("all required partitions must pass")

    @property
    def semantic_run_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "input_hash": self.input_hash,
            "code_version": self.code_version,
            "config_hash": self.config_hash,
            "environment_hash": self.environment_hash,
            "tasks": [
                {
                    "name": task.name,
                    "dependencies": task.dependencies,
                    "max_retries": task.max_retries,
                    "cpu_units": task.cpu_units,
                    "memory_mb": task.memory_mb,
                    "checkpointable": task.checkpointable,
                }
                for task in sorted(self.task_specs, key=lambda item: item.name)
            ],
            "partition_certificates": self.partition_certificates,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class PublicationResult:
    status: str
    run_id: str
    semantic_run_hash: str
    output_hash: str


class PublicationRegistry:
    def __init__(self) -> None:
        self.lock = RLock()
        self.records: dict[str, PublicationResult] = {}


def topological_order(tasks: Iterable[TaskSpec]) -> tuple[str, ...]:
    rows = tuple(tasks)
    names = [task.name for task in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate task names")
    deps = {task.name: set(task.dependencies) for task in rows}
    unknown = {
        dependency
        for values in deps.values()
        for dependency in values
        if dependency not in deps
    }
    if unknown:
        raise ValueError(f"unknown dependencies: {sorted(unknown)}")
    result: list[str] = []
    ready = sorted(name for name, values in deps.items() if not values)
    while ready:
        name = ready.pop(0)
        result.append(name)
        for other in sorted(deps):
            if name in deps[other]:
                deps[other].remove(name)
                if (
                    not deps[other]
                    and other not in result
                    and other not in ready
                ):
                    ready.append(other)
        ready.sort()
    if len(result) != len(rows):
        raise ValueError("cycle detected")
    return tuple(result)


def retry_delay(
    attempt: int,
    *,
    base_seconds: float = 1.0,
    cap_seconds: float = 60.0,
    deterministic_jitter: float = 0.0,
) -> float:
    if attempt < 1:
        raise ValueError("attempt is one-based")
    if (
        not isfinite(base_seconds)
        or not isfinite(cap_seconds)
        or not isfinite(deterministic_jitter)
        or base_seconds <= 0.0
        or cap_seconds <= 0.0
        or not 0.0 <= deterministic_jitter <= 1.0
    ):
        raise ValueError("invalid retry policy")
    maximum_exponent = max(0, ceil(log2(cap_seconds / base_seconds)) + 1)
    raw = base_seconds * (2 ** min(attempt - 1, maximum_exponent))
    return min(cap_seconds, raw * (1.0 + deterministic_jitter))


def create_run_manifest(
    *,
    run_id: str,
    input_manifest: Mapping[str, object],
    code_version: str,
    config: Mapping[str, object],
    tasks: Iterable[TaskSpec],
    environment: Mapping[str, object] | None = None,
) -> RunManifest:
    if not run_id or not code_version:
        raise ValueError("run_id and code_version are required")
    task_rows = tuple(tasks)
    order = topological_order(task_rows)
    encoded_input = json.dumps(
        input_manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    encoded_config = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    encoded_environment = json.dumps(
        dict(environment or {}),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return RunManifest(
        run_id=run_id,
        input_hash=sha256(encoded_input).hexdigest(),
        code_version=code_version,
        config_hash=sha256(encoded_config).hexdigest(),
        task_order=order,
        state=RunState.PENDING,
        task_specs=task_rows,
        environment_hash=sha256(encoded_environment).hexdigest(),
    )


def mark_succeeded(
    run: RunManifest,
    *,
    output_hash: str,
    partition_certificates: Iterable[tuple[str, str, str]],
) -> RunManifest:
    if run.state is not RunState.PENDING and run.state is not RunState.RUNNING:
        raise ValueError("only pending/running run can succeed")
    rows = tuple(sorted(partition_certificates))
    if not rows or len({(task, partition) for task, partition, _ in rows}) != len(rows):
        raise ValueError("partition certificates must be nonempty and unique")
    return RunManifest(
        run_id=run.run_id,
        input_hash=run.input_hash,
        code_version=run.code_version,
        config_hash=run.config_hash,
        task_order=run.task_order,
        state=RunState.SUCCEEDED,
        output_hash=output_hash,
        task_specs=run.task_specs,
        environment_hash=run.environment_hash,
        partition_certificates=rows,
        schema_version=run.schema_version,
    )


def publish_atomically(
    run: RunManifest,
    *,
    output_hash: str,
    registry: PublicationRegistry,
) -> PublicationResult:
    if run.state is not RunState.SUCCEEDED:
        raise ValueError("only a successful run can be published")
    _require_digest(output_hash, "output_hash")
    if run.output_hash != output_hash:
        raise ValueError("published output must match the completed run")
    semantic_hash = run.semantic_run_hash
    with registry.lock:
        prior = registry.records.get(semantic_hash)
        if prior is not None:
            if prior.output_hash != output_hash:
                raise ValueError("same semantic run produced a different output")
            return PublicationResult(
                "DUPLICATE_SAME_OUTPUT",
                run.run_id,
                semantic_hash,
                output_hash,
            )
        result = PublicationResult(
            "PUBLISHED",
            run.run_id,
            semantic_hash,
            output_hash,
        )
        registry.records[semantic_hash] = result
        return result


def _require_digest(value: str, field: str) -> None:
    if len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
