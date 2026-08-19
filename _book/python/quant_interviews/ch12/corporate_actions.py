from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum


class ActionType(str, Enum):
    SPLIT = "SPLIT"
    CASH_DIVIDEND = "CASH_DIVIDEND"
    SPINOFF = "SPINOFF"
    RIGHTS = "RIGHTS"
    SYMBOL_CHANGE = "SYMBOL_CHANGE"
    DELISTING = "DELISTING"


class ActionStatus(str, Enum):
    ANNOUNCED = "ANNOUNCED"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"


class ApplicationStatus(str, Enum):
    APPLIED = "APPLIED"
    PENDING_RECOVERY = "PENDING_RECOVERY"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class Position:
    instrument_id: str
    quantity: Decimal
    currency: str
    cost_basis: Decimal
    source_lot_id: str

    def __post_init__(self) -> None:
        if (
            not self.instrument_id
            or not self.currency
            or not self.source_lot_id
            or not self.quantity.is_finite()
            or not self.cost_basis.is_finite()
            or self.quantity < 0
            or self.cost_basis < 0
        ):
            raise ValueError("invalid position")


@dataclass(frozen=True)
class CashPosting:
    account: str
    currency: str
    amount: Decimal
    action_id: str

    def __post_init__(self) -> None:
        if not self.account or not self.currency or not self.action_id:
            raise ValueError("cash-posting identity is required")
        if not self.amount.is_finite() or self.amount < 0:
            raise ValueError("cash-posting amount must be finite and non-negative")


@dataclass(frozen=True)
class CorporateAction:
    action_id: str
    version_id: str
    source_id: str
    parent_instrument_id: str
    action_type: ActionType
    announced_at: datetime
    known_from: datetime
    known_to: datetime | None
    ex_date: date
    record_date: date | None
    payable_date: date | None
    currency: str
    status: ActionStatus
    ratio: Decimal = Decimal("1")
    cash_amount: Decimal = Decimal("0")
    child_instrument_id: str | None = None
    successor_instrument_id: str | None = None
    recovery_amount: Decimal | None = None
    parent_cost_basis_fraction: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        if not all(
            value
            for value in (
                self.action_id,
                self.version_id,
                self.source_id,
                self.parent_instrument_id,
                self.currency,
            )
        ):
            raise ValueError("corporate-action identity fields are required")
        if self.known_to is not None and self.known_to <= self.known_from:
            raise ValueError("invalid knowledge interval")
        _require_aware_utc(self.announced_at, "announced_at")
        _require_aware_utc(self.known_from, "known_from")
        if self.known_to is not None:
            _require_aware_utc(self.known_to, "known_to")
        if self.announced_at > self.known_from:
            raise ValueError("knowledge cannot precede the announcement")
        ordered_dates = [self.ex_date]
        if self.record_date is not None:
            ordered_dates.append(self.record_date)
        if self.payable_date is not None:
            ordered_dates.append(self.payable_date)
        if ordered_dates != sorted(ordered_dates):
            raise ValueError("ex, record and payable dates must be chronological")
        if not self.ratio.is_finite() or self.ratio <= 0:
            raise ValueError("ratio must be positive")
        if not self.cash_amount.is_finite() or self.cash_amount < 0:
            raise ValueError("cash amount must be non-negative")
        if (
            self.recovery_amount is not None
            and (
                not self.recovery_amount.is_finite()
                or self.recovery_amount < 0
            )
        ):
            raise ValueError("recovery amount must be non-negative")
        if (
            not self.parent_cost_basis_fraction.is_finite()
            or not Decimal("0") <= self.parent_cost_basis_fraction <= Decimal("1")
        ):
            raise ValueError("parent cost-basis fraction must lie in [0, 1]")
        if self.action_type in {ActionType.SPINOFF, ActionType.RIGHTS}:
            if self.child_instrument_id is None:
                raise ValueError("child instrument is required")
        if self.action_type is ActionType.SPINOFF and not (
            Decimal("0") < self.parent_cost_basis_fraction < Decimal("1")
        ):
            raise ValueError("spin-off requires an explicit nontrivial basis allocation")
        if self.action_type is ActionType.SYMBOL_CHANGE:
            if self.successor_instrument_id is None:
                raise ValueError("successor instrument is required")


@dataclass(frozen=True)
class ActionApplication:
    status: ApplicationStatus
    parent_position: Position | None
    child_positions: tuple[Position, ...]
    cash_postings: tuple[CashPosting, ...]
    identity_transition: tuple[str, str] | None
    raw_market_price_adjusted: bool
    reason: str
    application_key: tuple[str, str, str, str] | None = None
    cost_basis_conserved: bool = True


def apply_action(
    position: Position,
    action: CorporateAction,
    *,
    portfolio_id: str,
    effective_date: date,
    knowledge_cutoff: datetime,
    application_ledger: set[tuple[str, str, str, str]],
) -> ActionApplication:
    if not portfolio_id:
        raise ValueError("portfolio_id is required")
    _require_aware_utc(knowledge_cutoff, "knowledge_cutoff")
    if position.instrument_id != action.parent_instrument_id:
        raise ValueError("action and position instrument mismatch")
    if position.currency != action.currency:
        raise ValueError("currency conversion requires a separate FX posting")
    application_key = (
        portfolio_id,
        position.source_lot_id,
        action.action_id,
        action.version_id,
    )
    if application_key in application_ledger:
        return ActionApplication(
            ApplicationStatus.REJECTED,
            position,
            (),
            (),
            None,
            False,
            "duplicate corporate-action application",
            application_key,
        )
    eligible = (
        action.status is ActionStatus.CONFIRMED
        and effective_date >= action.ex_date
        and action.announced_at <= knowledge_cutoff
        and action.known_from <= knowledge_cutoff
        and (action.known_to is None or knowledge_cutoff < action.known_to)
    )
    if not eligible:
        return ActionApplication(
            ApplicationStatus.REJECTED,
            position,
            (),
            (),
            None,
            False,
            "action is not confirmed, effective and known at the cutoff",
            application_key,
        )
    application_ledger.add(application_key)
    if action.action_type is ActionType.SPLIT:
        adjusted = Position(
            position.instrument_id,
            position.quantity * action.ratio,
            position.currency,
            position.cost_basis,
            position.source_lot_id,
        )
        return ActionApplication(
            ApplicationStatus.APPLIED,
            adjusted,
            (),
            (),
            None,
            False,
            "quantity adjusted; raw observed market price remains external",
            application_key,
        )
    if action.action_type is ActionType.CASH_DIVIDEND:
        posting = CashPosting(
            "dividend_receivable",
            action.currency,
            position.quantity * action.cash_amount,
            action.action_id,
        )
        return ActionApplication(
            ApplicationStatus.APPLIED,
            position,
            (),
            (posting,),
            None,
            False,
            "receivable posted once; no synthetic price subtraction",
            application_key,
        )
    if action.action_type in {ActionType.SPINOFF, ActionType.RIGHTS}:
        assert action.child_instrument_id is not None
        child_fraction = (
            Decimal("0")
            if action.action_type is ActionType.RIGHTS
            else Decimal("1") - action.parent_cost_basis_fraction
        )
        allocated_cost = position.cost_basis * child_fraction
        parent_basis = position.cost_basis - allocated_cost
        parent = Position(
            position.instrument_id,
            position.quantity,
            position.currency,
            parent_basis,
            position.source_lot_id,
        )
        child = Position(
            action.child_instrument_id,
            position.quantity * action.ratio,
            action.currency,
            allocated_cost,
            f"{position.source_lot_id}:{action.action_id}",
        )
        return ActionApplication(
            ApplicationStatus.APPLIED,
            parent,
            (child,),
            (),
            None,
            False,
            "child security created and cost basis allocated conservatively",
            application_key,
            parent.cost_basis + child.cost_basis == position.cost_basis,
        )
    if action.action_type is ActionType.SYMBOL_CHANGE:
        assert action.successor_instrument_id is not None
        return ActionApplication(
            ApplicationStatus.APPLIED,
            position,
            (),
            (),
            (position.instrument_id, action.successor_instrument_id),
            False,
            "ticker/symbol alias updated; canonical instrument remains unchanged",
            application_key,
        )
    if action.action_type is ActionType.DELISTING:
        if action.recovery_amount is None:
            return ActionApplication(
                ApplicationStatus.PENDING_RECOVERY,
                position,
                (),
                (),
                None,
                False,
                "recovery is unresolved; no cash is invented",
                application_key,
            )
        posting = CashPosting(
            "delisting_receivable",
            action.currency,
            position.quantity * action.recovery_amount,
            action.action_id,
        )
        return ActionApplication(
            ApplicationStatus.APPLIED,
            None,
            (),
            (posting,),
            None,
            False,
            "terminal receivable posted at the governed recovery amount",
            application_key,
        )
    raise ValueError(f"unsupported action type: {action.action_type}")


def theoretical_split_reference_price(
    raw_pre_action_price: Decimal,
    ratio: Decimal,
) -> Decimal:
    if (
        not raw_pre_action_price.is_finite()
        or not ratio.is_finite()
        or raw_pre_action_price < 0
        or ratio <= 0
    ):
        raise ValueError("invalid split reference inputs")
    return raw_pre_action_price / ratio


def total_return(
    previous_price: Decimal,
    current_price: Decimal,
    cash_distributions: tuple[Decimal, ...] = (),
    distributed_security_values: tuple[Decimal, ...] = (),
) -> Decimal:
    if (
        not previous_price.is_finite()
        or not current_price.is_finite()
        or previous_price <= 0
        or current_price < 0
    ):
        raise ValueError("invalid price")
    if any(
        not value.is_finite() or value < 0
        for value in cash_distributions + distributed_security_values
    ):
        raise ValueError("distribution values must be non-negative")
    economic_value = (
        current_price
        + sum(cash_distributions, Decimal("0"))
        + sum(distributed_security_values, Decimal("0"))
    )
    return economic_value / previous_price - Decimal("1")


def _require_aware_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field} must be normalized to UTC")
