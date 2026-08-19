from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from math import isfinite
import re
from types import MappingProxyType
from typing import Iterable, Mapping

from .takehome_case import canonical_bytes


HEX64 = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_EVENT_SCHEMA = 1
REDUCER_VERSION = "journal-reducer-v1"
SUPPORTED_CURRENCIES = frozenset({"USD", "EUR", "GBP", "JPY", "CHF"})
REPLAY_DOMAIN_SEPARATOR = b"ch14-journal-hash-chain-v1"
EMPTY_REPLAY_COMMITMENT = sha256(REPLAY_DOMAIN_SEPARATOR).hexdigest()


def _hash(value: str, *, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _utc(value: str, *, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class Event:
    event_id: str
    sequence: int
    kind: str
    instrument_id: str
    currency: str
    quantity: float = 0.0
    price: float = 0.0
    amount: float = 0.0
    schema_version: int = SUPPORTED_EVENT_SCHEMA

    def __post_init__(self) -> None:
        if not self.event_id or self.sequence <= 0:
            raise ValueError("event identity and positive durable sequence are required")
        if self.schema_version != SUPPORTED_EVENT_SCHEMA:
            raise ValueError("unsupported event schema version")
        if self.kind not in {"fill", "fee"}:
            raise ValueError("unsupported event kind for this journal")
        if self.currency not in SUPPORTED_CURRENCIES:
            raise ValueError("unsupported event currency")
        if not all(
            isfinite(value) for value in (self.quantity, self.price, self.amount)
        ):
            raise ValueError("event monetary fields must be finite")
        if self.kind == "fill":
            if (
                not self.instrument_id
                or self.quantity == 0.0
                or self.price <= 0.0
                or self.amount != 0.0
            ):
                raise ValueError(
                    "fill requires instrument, nonzero quantity, positive price "
                    "and zero fee amount"
                )
        elif (
            self.instrument_id
            or self.quantity != 0.0
            or self.price != 0.0
            or self.amount < 0.0
        ):
            raise ValueError(
                "fee requires empty instrument, zero quantity/price and nonnegative amount"
            )


@dataclass(frozen=True)
class JournalState:
    positions: Mapping[tuple[str, str], float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    cash: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    processed_payload_hashes: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    sequence_owners: Mapping[int, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    replay_commitment: str = EMPTY_REPLAY_COMMITMENT
    last_sequence: int = 0
    event_count: int = 0

    def __post_init__(self) -> None:
        _hash(self.replay_commitment, name="replay_commitment")
        if self.last_sequence < 0 or self.event_count < 0:
            raise ValueError("journal counters must be non-negative")
        if any(
            not instrument
            or currency not in SUPPORTED_CURRENCIES
            or not isfinite(value)
            for (instrument, currency), value in self.positions.items()
        ):
            raise ValueError("journal positions are invalid")
        if any(
            currency not in SUPPORTED_CURRENCIES or not isfinite(value)
            for currency, value in self.cash.items()
        ):
            raise ValueError("journal cash is invalid")
        if any(
            not event_id or HEX64.fullmatch(payload_hash) is None
            for event_id, payload_hash in self.processed_payload_hashes.items()
        ):
            raise ValueError("processed event map is invalid")
        if any(
            not isinstance(sequence, int)
            or sequence <= 0
            or not event_id
            for sequence, event_id in self.sequence_owners.items()
        ):
            raise ValueError("durable sequence map is invalid")
        if set(self.sequence_owners.values()) != set(
            self.processed_payload_hashes
        ):
            raise ValueError("sequence and processed-event maps disagree")
        if self.event_count != len(self.processed_payload_hashes):
            raise ValueError("event_count disagrees with processed-event map")
        expected_last = max(self.sequence_owners, default=0)
        if self.last_sequence != expected_last:
            raise ValueError("last_sequence disagrees with durable sequence map")
        if set(self.sequence_owners) != set(range(1, self.last_sequence + 1)):
            raise ValueError("durable sequence map contains a gap")


@dataclass(frozen=True)
class FoldResult:
    state: JournalState
    applied_event_ids: tuple[str, ...]
    status: str

    @property
    def positions(self) -> Mapping[tuple[str, str], float]:
        return self.state.positions

    @property
    def cash(self) -> Mapping[str, float]:
        return self.state.cash

    @property
    def processed_payload_hashes(self) -> Mapping[str, str]:
        return self.state.processed_payload_hashes

    @property
    def replay_hash(self) -> str:
        return self.state.replay_commitment


@dataclass(frozen=True)
class HashedApprovalRecord:
    commitment: str
    issued_at: str
    expires_at: str
    approver_ids: tuple[str, ...]
    authorization_receipt_hash: str
    policy_version: str
    tool_version: str


def canonical_event_payload(event: Event) -> str:
    return canonical_bytes(asdict(event)).decode("utf-8")


def event_payload_hash(event: Event) -> str:
    return sha256(canonical_bytes(asdict(event))).hexdigest()


def fold_events(
    events: Iterable[Event],
    *,
    prior_state: JournalState | None = None,
) -> FoldResult:
    state = prior_state or JournalState()
    positions = dict(state.positions)
    cash = dict(state.cash)
    processed = dict(state.processed_payload_hashes)
    sequence_owners = dict(state.sequence_owners)
    commitment = state.replay_commitment
    last_sequence = state.last_sequence
    event_count = state.event_count
    applied: list[str] = []

    for event in sorted(events, key=lambda item: (item.sequence, item.event_id)):
        payload = canonical_bytes(asdict(event))
        payload_hash = sha256(payload).hexdigest()
        prior_hash = processed.get(event.event_id)
        if prior_hash is not None:
            if prior_hash != payload_hash:
                raise ValueError("event identity conflict: same ID, different payload")
            continue
        owner = sequence_owners.get(event.sequence)
        if owner is not None and owner != event.event_id:
            raise ValueError("duplicate durable sequence")
        if event.sequence != last_sequence + 1:
            raise ValueError("incremental journal events must be contiguous")

        next_positions = dict(positions)
        next_cash = dict(cash)
        if event.kind == "fill":
            key = (event.instrument_id, event.currency)
            position = next_positions.get(key, 0.0) + event.quantity
            cash_value = (
                next_cash.get(event.currency, 0.0)
                - event.quantity * event.price
            )
            if not isfinite(position) or not isfinite(cash_value):
                raise FloatingPointError("event would create non-finite state")
            next_positions[key] = position
            next_cash[event.currency] = cash_value
        else:
            cash_value = next_cash.get(event.currency, 0.0) - event.amount
            if not isfinite(cash_value):
                raise FloatingPointError("event would create non-finite cash")
            next_cash[event.currency] = cash_value

        next_commitment = sha256(
            bytes.fromhex(commitment)
            + payload
            + f"|schema={event.schema_version}|reducer={REDUCER_VERSION}".encode(
                "utf-8"
            )
        ).hexdigest()
        # Atomic in-memory commit after all event post-conditions pass.
        positions = next_positions
        cash = next_cash
        processed[event.event_id] = payload_hash
        sequence_owners[event.sequence] = event.event_id
        commitment = next_commitment
        last_sequence = event.sequence
        event_count += 1
        applied.append(event.event_id)

    next_state = JournalState(
        positions=MappingProxyType(dict(sorted(positions.items()))),
        cash=MappingProxyType(dict(sorted(cash.items()))),
        processed_payload_hashes=MappingProxyType(dict(sorted(processed.items()))),
        sequence_owners=MappingProxyType(dict(sorted(sequence_owners.items()))),
        replay_commitment=commitment,
        last_sequence=last_sequence,
        event_count=event_count,
    )
    return FoldResult(next_state, tuple(applied), "PASS")


def replay_hash(events: Iterable[Event]) -> str:
    return fold_events(events).replay_hash


def approval_commitment(
    *,
    proposal_hash: str,
    portfolio_snapshot_hash: str,
    risk_result_hash: str,
    policy_version: str,
    approver_ids: tuple[str, ...],
    issued_at: str,
    expires_at: str,
    tool_version: str,
    authorization_receipt_hash: str,
) -> HashedApprovalRecord:
    hashes = {
        "proposal_hash": _hash(proposal_hash, name="proposal_hash"),
        "portfolio_snapshot_hash": _hash(
            portfolio_snapshot_hash, name="portfolio_snapshot_hash"
        ),
        "risk_result_hash": _hash(risk_result_hash, name="risk_result_hash"),
        "authorization_receipt_hash": _hash(
            authorization_receipt_hash, name="authorization_receipt_hash"
        ),
    }
    issued = _utc(issued_at, name="issued_at")
    expires = _utc(expires_at, name="expires_at")
    if issued >= expires:
        raise ValueError("approval requires issued_at < expires_at")
    if (
        len(approver_ids) < 2
        or len(set(approver_ids)) != len(approver_ids)
        or any(not identifier.strip() for identifier in approver_ids)
    ):
        raise ValueError("dual control requires two distinct approver identities")
    if not policy_version.strip() or not tool_version.strip():
        raise ValueError("approval policy and tool versions are required")
    payload: dict[str, object] = {
        **hashes,
        "approver_ids": sorted(approver_ids),
        "issued_at": issued.isoformat(),
        "expires_at": expires.isoformat(),
        "policy_version": policy_version,
        "tool_version": tool_version,
    }
    return HashedApprovalRecord(
        commitment=sha256(canonical_bytes(payload)).hexdigest(),
        issued_at=issued.isoformat(),
        expires_at=expires.isoformat(),
        approver_ids=tuple(sorted(approver_ids)),
        authorization_receipt_hash=hashes["authorization_receipt_hash"],
        policy_version=policy_version,
        tool_version=tool_version,
    )


def approval_gate(
    *,
    proposed_risk: float,
    limit: float,
    approval: HashedApprovalRecord,
    current_commitment: str,
    decision_time: str,
    coverage: float,
    minimum_coverage: float,
    evidence_complete: bool,
) -> str:
    values = (proposed_risk, limit, coverage, minimum_coverage)
    if (
        not all(isfinite(value) for value in values)
        or limit <= 0.0
        or proposed_risk < 0.0
        or not 0.0 <= coverage <= 1.0
        or not 0.0 <= minimum_coverage <= 1.0
    ):
        return "INVALID_INPUT"
    try:
        _hash(current_commitment, name="current_commitment")
        now = _utc(decision_time, name="decision_time")
        issued = _utc(approval.issued_at, name="approval.issued_at")
        expires = _utc(approval.expires_at, name="approval.expires_at")
        _hash(
            approval.authorization_receipt_hash,
            name="authorization_receipt_hash",
        )
    except ValueError:
        return "INVALID_INPUT"
    if current_commitment != approval.commitment:
        return "BLOCK_STALE_APPROVAL"
    if not issued <= now < expires:
        return "BLOCK_EXPIRED_APPROVAL"
    if len(set(approval.approver_ids)) < 2:
        return "BLOCK_NO_DUAL_CONTROL"
    if not evidence_complete:
        return "BLOCK_INCOMPLETE_EVIDENCE"
    if coverage < minimum_coverage:
        return "BLOCK_COVERAGE"
    if proposed_risk > limit:
        return "BLOCK_LIMIT"
    return "PASS"
