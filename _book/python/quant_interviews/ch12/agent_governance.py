from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from threading import RLock
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator, FormatChecker


HEX = frozenset("0123456789abcdef")


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _digest(value: str | None, field: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or len(value) != 64 or set(value) - HEX:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class AuthorizationStatus(str, Enum):
    AUTHORIZED_READ_ONLY = "AUTHORIZED_READ_ONLY"
    AUTHORIZED_MUTATION = "AUTHORIZED_MUTATION"


@dataclass(frozen=True)
class ToolPolicy:
    name: str
    policy_version: str
    allowed_roles: frozenset[str]
    allowed_tenants: frozenset[str]
    allowed_portfolios: frozenset[str]
    allowed_actions: frozenset[str]
    mutating: bool
    requires_human_approval: bool
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.name or not self.policy_version:
            raise ValueError("tool policy identity is required")
        if not self.allowed_roles or not self.allowed_tenants:
            raise ValueError("tool policy scope must not be empty")
        if self.mutating and not self.requires_human_approval:
            raise ValueError("mutating tools must require human approval")
        Draft202012Validator.check_schema(dict(self.input_schema))
        Draft202012Validator.check_schema(dict(self.output_schema))


@dataclass(frozen=True)
class ToolCall:
    tool: str
    action: str
    actor_id: str
    actor_role: str
    tenant_id: str
    portfolio_id: str
    request_id: str
    idempotency_key: str
    expected_state_hash: str | None
    arguments_json: bytes

    def __post_init__(self) -> None:
        fields = (
            self.tool,
            self.action,
            self.actor_id,
            self.actor_role,
            self.tenant_id,
            self.portfolio_id,
            self.request_id,
            self.idempotency_key,
        )
        if not all(isinstance(value, str) and value.strip() for value in fields):
            raise ValueError("tool-call identity and scope are required")
        _digest(self.expected_state_hash, "expected_state_hash", optional=True)
        decoded = json.loads(self.arguments_json)
        if not isinstance(decoded, dict) or _canonical(decoded) != self.arguments_json:
            raise ValueError("arguments_json must be a canonical JSON object")

    @classmethod
    def create(
        cls,
        *,
        tool: str,
        action: str,
        actor_id: str,
        actor_role: str,
        tenant_id: str,
        portfolio_id: str,
        request_id: str,
        idempotency_key: str,
        expected_state_hash: str | None,
        arguments: Mapping[str, object],
    ) -> ToolCall:
        return cls(
            tool,
            action,
            actor_id,
            actor_role,
            tenant_id,
            portfolio_id,
            request_id,
            idempotency_key,
            expected_state_hash,
            _canonical(dict(arguments)),
        )

    @property
    def arguments(self) -> dict[str, object]:
        result = json.loads(self.arguments_json)
        assert isinstance(result, dict)
        return result

    @property
    def arguments_hash(self) -> str:
        return sha256(self.arguments_json).hexdigest()

    @property
    def scoped_idempotency_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.tenant_id,
            self.portfolio_id,
            self.actor_id,
            self.tool,
            self.action,
            self.idempotency_key,
        )

    @property
    def envelope_hash(self) -> str:
        return sha256(
            _canonical(
                {
                    "tool": self.tool,
                    "action": self.action,
                    "actor_id": self.actor_id,
                    "actor_role": self.actor_role,
                    "tenant_id": self.tenant_id,
                    "portfolio_id": self.portfolio_id,
                    "request_id": self.request_id,
                    "idempotency_key": self.idempotency_key,
                    "expected_state_hash": self.expected_state_hash,
                    "arguments_hash": self.arguments_hash,
                }
            )
        ).hexdigest()


@dataclass(frozen=True)
class HumanApproval:
    approval_id: str
    request_id: str
    approver_id: str
    approved_at: datetime
    policy_version: str
    actor_id: str
    tenant_id: str
    portfolio_id: str
    tool: str
    action: str
    arguments_hash: str
    expected_state_hash: str
    not_before: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        text = (
            self.approval_id,
            self.request_id,
            self.approver_id,
            self.policy_version,
            self.actor_id,
            self.tenant_id,
            self.portfolio_id,
            self.tool,
            self.action,
        )
        if not all(isinstance(value, str) and value.strip() for value in text):
            raise ValueError("approval identity and scope are required")
        approved = _utc(self.approved_at, "approved_at")
        start = _utc(self.not_before, "not_before")
        end = _utc(self.expires_at, "expires_at")
        if not start <= approved <= end:
            raise ValueError("approval time must lie inside its validity interval")
        _digest(self.arguments_hash, "arguments_hash")
        _digest(self.expected_state_hash, "expected_state_hash")

    @classmethod
    def create(
        cls,
        *,
        approval_id: str,
        approver_id: str,
        call: ToolCall,
        policy_version: str,
        approved_at: datetime,
        not_before: datetime,
        expires_at: datetime,
    ) -> HumanApproval:
        if call.expected_state_hash is None:
            raise ValueError("mutating approval requires expected state")
        return cls(
            approval_id,
            call.request_id,
            approver_id,
            approved_at,
            policy_version,
            call.actor_id,
            call.tenant_id,
            call.portfolio_id,
            call.tool,
            call.action,
            call.arguments_hash,
            call.expected_state_hash,
            not_before,
            expires_at,
        )

    @property
    def approval_hash(self) -> str:
        return sha256(
            _canonical(
                {
                    "approval_id": self.approval_id,
                    "request_id": self.request_id,
                    "approver_id": self.approver_id,
                    "approved_at": _utc(self.approved_at, "approved_at").isoformat(),
                    "policy_version": self.policy_version,
                    "actor_id": self.actor_id,
                    "tenant_id": self.tenant_id,
                    "portfolio_id": self.portfolio_id,
                    "tool": self.tool,
                    "action": self.action,
                    "arguments_hash": self.arguments_hash,
                    "expected_state_hash": self.expected_state_hash,
                    "not_before": _utc(self.not_before, "not_before").isoformat(),
                    "expires_at": _utc(self.expires_at, "expires_at").isoformat(),
                }
            )
        ).hexdigest()


@dataclass(frozen=True)
class AuthorizationCertificate:
    status: AuthorizationStatus
    request_id: str
    policy_version: str
    approval_id: str | None
    input_hash: str
    expected_state_hash: str | None
    approval_hash: str | None


@dataclass(frozen=True)
class ExecutionEvidence:
    request_id: str
    tool: str
    action: str
    actor_id: str
    tenant_id: str
    portfolio_id: str
    policy_version: str
    tool_version: str
    authorization_status: AuthorizationStatus
    approval_id: str | None
    input_hash: str
    expected_state_hash: str | None
    output_hash: str
    staged_invariant_passed: bool
    evidence_hash: str


@dataclass(frozen=True)
class IdempotencyRecord:
    result: Mapping[str, object]
    evidence: ExecutionEvidence


@dataclass(frozen=True)
class OutboxMessage:
    scoped_key: tuple[str, str, str, str, str, str]
    result: Mapping[str, object]
    evidence: ExecutionEvidence
    dispatched: bool = False
    receipt_hash: str | None = None


class IdempotencyStore:
    """In-process transactional reference; production uses a durable unique key."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.records: dict[
            tuple[str, str, str, str, str, str], IdempotencyRecord
        ] = {}
        self.outbox: dict[
            tuple[str, str, str, str, str, str], OutboxMessage
        ] = {}


def _validate_schema(value: Mapping[str, object], schema: Mapping[str, object]) -> None:
    validator = Draft202012Validator(dict(schema), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(dict(value)), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        location = ".".join(str(item) for item in first.absolute_path) or "$"
        raise ValueError(f"schema validation failed at {location}: {first.message}")


def authorize(
    call: ToolCall,
    policy: ToolPolicy,
    approval: HumanApproval | None = None,
    *,
    decision_time: datetime | None = None,
    authoritative_state_hash: str | None = None,
) -> AuthorizationCertificate:
    if call.tool != policy.name:
        raise ValueError("policy/tool mismatch")
    if call.actor_role not in policy.allowed_roles:
        raise PermissionError("role not authorized")
    if call.tenant_id not in policy.allowed_tenants:
        raise PermissionError("tenant not authorized")
    if policy.allowed_portfolios and call.portfolio_id not in policy.allowed_portfolios:
        raise PermissionError("portfolio not authorized")
    if call.action not in policy.allowed_actions:
        raise PermissionError("action not authorized")
    _validate_schema(call.arguments, policy.input_schema)
    if policy.mutating:
        if approval is None or decision_time is None:
            raise PermissionError("bound human approval and decision_time required")
        if call.expected_state_hash is None or authoritative_state_hash is None:
            raise PermissionError("expected and authoritative state hashes are required")
        _digest(authoritative_state_hash, "authoritative_state_hash")
        if authoritative_state_hash != call.expected_state_hash:
            raise PermissionError("authoritative state changed before mutation")
        current = _utc(decision_time, "decision_time")
        bindings = (
            approval.request_id == call.request_id,
            approval.policy_version == policy.policy_version,
            approval.actor_id == call.actor_id,
            approval.tenant_id == call.tenant_id,
            approval.portfolio_id == call.portfolio_id,
            approval.tool == call.tool,
            approval.action == call.action,
            approval.arguments_hash == call.arguments_hash,
            approval.expected_state_hash == call.expected_state_hash,
            _utc(approval.approved_at, "approved_at") <= current,
            _utc(approval.not_before, "not_before") <= current,
            current < _utc(approval.expires_at, "expires_at"),
        )
        if not all(bindings):
            raise PermissionError("approval is not bound, current and in scope")
        status = AuthorizationStatus.AUTHORIZED_MUTATION
        approval_id = approval.approval_id
        approval_hash = approval.approval_hash
    else:
        if approval is not None:
            raise ValueError("read-only call must not consume an approval")
        status = AuthorizationStatus.AUTHORIZED_READ_ONLY
        approval_id = None
        approval_hash = None
    return AuthorizationCertificate(
        status,
        call.request_id,
        policy.policy_version,
        approval_id,
        call.envelope_hash,
        call.expected_state_hash,
        approval_hash,
    )


def execute_tool(
    call: ToolCall,
    policy: ToolPolicy,
    implementation: Callable[..., Mapping[str, object]],
    *,
    tool_version: str,
    approval: HumanApproval | None = None,
    decision_time: datetime | None = None,
    authoritative_state_hash: str | None = None,
    precondition: Callable[[ToolCall], bool] | None = None,
    staged_invariant: Callable[[Mapping[str, object]], bool] | None = None,
    local_outbox_commit: Callable[[Mapping[str, object]], None] | None = None,
    idempotency_store: IdempotencyStore | None = None,
) -> tuple[Mapping[str, object], ExecutionEvidence]:
    if not tool_version:
        raise ValueError("tool_version is required")
    if policy.mutating and (
        precondition is None
        or staged_invariant is None
        or local_outbox_commit is None
        or idempotency_store is None
    ):
        raise ValueError(
            "mutations require precondition, staged invariant, local outbox commit "
            "and durable store"
        )
    certificate = authorize(
        call,
        policy,
        approval,
        decision_time=decision_time,
        authoritative_state_hash=authoritative_state_hash,
    )
    store = idempotency_store or IdempotencyStore()
    with store.lock:
        prior = store.records.get(call.scoped_idempotency_key)
        if prior is not None:
            if prior.evidence.input_hash != certificate.input_hash:
                raise ValueError("scoped idempotency-key conflict")
            return prior.result, prior.evidence
        if precondition is not None and not precondition(call):
            raise PermissionError("deterministic precondition failed")
        # The implementation is a pure staged calculation.  For a mutation the
        # callback may only enqueue an outbox row in the same local transaction;
        # it must not perform the external effect.
        result = dict(implementation(**call.arguments))
        _validate_schema(result, policy.output_schema)
        staged_invariant_passed = (
            True if staged_invariant is None else bool(staged_invariant(result))
        )
        if not staged_invariant_passed:
            raise RuntimeError("deterministic staged invariant failed")
        output_hash = sha256(_canonical(result)).hexdigest()
        evidence_body = {
            "request_id": call.request_id,
            "tool": call.tool,
            "action": call.action,
            "actor_id": call.actor_id,
            "tenant_id": call.tenant_id,
            "portfolio_id": call.portfolio_id,
            "policy_version": policy.policy_version,
            "tool_version": tool_version,
            "authorization_status": certificate.status.value,
            "approval_id": certificate.approval_id,
            "approval_hash": certificate.approval_hash,
            "input_hash": certificate.input_hash,
            "expected_state_hash": certificate.expected_state_hash,
            "output_hash": output_hash,
            "staged_invariant_passed": staged_invariant_passed,
        }
        evidence = ExecutionEvidence(
            request_id=call.request_id,
            tool=call.tool,
            action=call.action,
            actor_id=call.actor_id,
            tenant_id=call.tenant_id,
            portfolio_id=call.portfolio_id,
            policy_version=policy.policy_version,
            tool_version=tool_version,
            authorization_status=certificate.status,
            approval_id=certificate.approval_id,
            input_hash=certificate.input_hash,
            expected_state_hash=certificate.expected_state_hash,
            output_hash=output_hash,
            staged_invariant_passed=staged_invariant_passed,
            evidence_hash=sha256(_canonical(evidence_body)).hexdigest(),
        )
        if local_outbox_commit is not None:
            local_outbox_commit(result)
        store.records[call.scoped_idempotency_key] = IdempotencyRecord(
            result=result,
            evidence=evidence,
        )
        if policy.mutating:
            store.outbox[call.scoped_idempotency_key] = OutboxMessage(
                call.scoped_idempotency_key,
                result,
                evidence,
            )
        return result, evidence


def dispatch_outbox(
    store: IdempotencyStore,
    scoped_key: tuple[str, str, str, str, str, str],
    external_executor: Callable[
        [Mapping[str, object], tuple[str, str, str, str, str, str]],
        Mapping[str, object],
    ],
    actual_postcondition: Callable[[Mapping[str, object]], bool],
) -> Mapping[str, object]:
    """Dispatch one local outbox message through an idempotent external executor.

    The scoped key is passed to the executor.  A crash after the external effect
    but before receipt persistence is recovered by retrying that same key; the
    external boundary, not this process, must deduplicate it.
    """

    with store.lock:
        message = store.outbox.get(scoped_key)
        if message is None:
            raise KeyError("outbox message not found")
        if message.dispatched:
            raise ValueError("outbox message was already dispatched")
        result = dict(message.result)
    receipt = dict(external_executor(result, scoped_key))
    if not actual_postcondition(receipt):
        raise RuntimeError("actual external post-condition failed")
    receipt_hash = sha256(_canonical(receipt)).hexdigest()
    with store.lock:
        current = store.outbox.get(scoped_key)
        if current != message:
            raise RuntimeError("outbox state changed during dispatch")
        store.outbox[scoped_key] = OutboxMessage(
            message.scoped_key,
            message.result,
            message.evidence,
            True,
            receipt_hash,
        )
    return receipt


def execute_read_only(
    call: ToolCall,
    policy: ToolPolicy,
    implementation: Callable[..., Mapping[str, object]],
    *,
    tool_version: str,
) -> dict[str, Any]:
    if policy.mutating:
        raise PermissionError("read-only executor rejects mutating tools")
    result, evidence = execute_tool(
        call,
        policy,
        implementation,
        tool_version=tool_version,
    )
    return {"result": result, "evidence": evidence}
