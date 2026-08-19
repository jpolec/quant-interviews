from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Iterable


ZERO = Decimal("0")


class Account(str, Enum):
    CASH = "CASH"
    CAPITAL = "CAPITAL"
    SECURITY_INVENTORY = "SECURITY_INVENTORY"
    FEE_EXPENSE = "FEE_EXPENSE"
    REALIZED_GAIN = "REALIZED_GAIN"
    REALIZED_LOSS = "REALIZED_LOSS"
    RECEIVABLE = "RECEIVABLE"


class NavStatus(str, Enum):
    PASS = "PASS"
    INVALID_OR_STALE_INPUT = "INVALID_OR_STALE_INPUT"


@dataclass(frozen=True)
class Posting:
    account: Account
    currency: str
    debit: Decimal = ZERO
    credit: Decimal = ZERO
    instrument_id: str | None = None
    lot_id: str | None = None

    def __post_init__(self) -> None:
        if not self.currency:
            raise ValueError("currency is required")
        if not self.debit.is_finite() or not self.credit.is_finite():
            raise ValueError("posting amounts must be finite")
        if self.debit < ZERO or self.credit < ZERO:
            raise ValueError("posting amounts must be non-negative")
        if (self.debit == ZERO) == (self.credit == ZERO):
            raise ValueError("exactly one of debit and credit must be positive")


@dataclass(frozen=True)
class JournalEntry:
    event_id: str
    booked_at: datetime
    settlement_at: datetime
    description: str
    postings: tuple[Posting, ...]

    def __post_init__(self) -> None:
        if not self.event_id or not self.description:
            raise ValueError("event_id and description are required")
        if not self.postings:
            raise ValueError("a journal entry requires postings")
        _require_aware_utc(self.booked_at, "booked_at")
        _require_aware_utc(self.settlement_at, "settlement_at")
        if self.settlement_at < self.booked_at:
            raise ValueError("settlement cannot precede booking")
        currencies = {posting.currency for posting in self.postings}
        for currency in currencies:
            debit = sum(
                (
                    posting.debit
                    for posting in self.postings
                    if posting.currency == currency
                ),
                ZERO,
            )
            credit = sum(
                (
                    posting.credit
                    for posting in self.postings
                    if posting.currency == currency
                ),
                ZERO,
            )
            if debit != credit:
                raise ValueError(
                    f"unbalanced journal entry in {currency}: {debit} != {credit}"
                )


@dataclass(frozen=True)
class Lot:
    lot_id: str
    instrument_id: str
    currency: str
    quantity: Decimal
    unit_cost: Decimal
    acquired_at: datetime
    settlement_at: datetime
    source_event_id: str

    def __post_init__(self) -> None:
        if not self.lot_id or not self.instrument_id or not self.currency:
            raise ValueError("lot identity fields are required")
        if (
            not self.quantity.is_finite()
            or not self.unit_cost.is_finite()
            or self.quantity <= ZERO
            or self.unit_cost < ZERO
        ):
            raise ValueError("lot quantity and cost are invalid")
        _require_aware_utc(self.acquired_at, "acquired_at")
        _require_aware_utc(self.settlement_at, "settlement_at")
        if self.settlement_at < self.acquired_at:
            raise ValueError("lot settlement cannot precede acquisition")


@dataclass(frozen=True)
class LedgerState:
    entries: tuple[JournalEntry, ...] = ()
    lots: tuple[Lot, ...] = ()

    def __post_init__(self) -> None:
        event_ids = [entry.event_id for entry in self.entries]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("journal event_id values must be unique")
        lot_ids = [lot.lot_id for lot in self.lots]
        if len(lot_ids) != len(set(lot_ids)):
            raise ValueError("lot_id values must be unique")


@dataclass(frozen=True)
class Mark:
    instrument_id: str
    currency: str
    price: Decimal
    known_at: datetime
    effective_at: datetime | None = None
    source_id: str = "UNSPECIFIED"
    version_id: str = "UNSPECIFIED"
    row_id: str = "UNSPECIFIED"

    def __post_init__(self) -> None:
        if (
            not self.instrument_id
            or not self.currency
            or not self.price.is_finite()
            or self.price < ZERO
        ):
            raise ValueError("invalid mark")
        _require_aware_utc(self.known_at, "known_at")
        effective = self.known_at if self.effective_at is None else self.effective_at
        _require_aware_utc(effective, "effective_at")
        if not self.source_id or not self.version_id or not self.row_id:
            raise ValueError("mark lineage is required")


@dataclass(frozen=True)
class FxRate:
    currency: str
    base_currency: str
    rate: Decimal
    known_at: datetime
    effective_at: datetime | None = None
    source_id: str = "UNSPECIFIED"
    version_id: str = "UNSPECIFIED"
    row_id: str = "UNSPECIFIED"

    def __post_init__(self) -> None:
        if (
            not self.currency
            or not self.base_currency
            or not self.rate.is_finite()
            or self.rate <= ZERO
        ):
            raise ValueError("invalid FX rate")
        _require_aware_utc(self.known_at, "known_at")
        effective = self.known_at if self.effective_at is None else self.effective_at
        _require_aware_utc(effective, "effective_at")
        if not self.source_id or not self.version_id or not self.row_id:
            raise ValueError("FX lineage is required")


@dataclass(frozen=True)
class NavCertificate:
    status: NavStatus
    base_currency: str
    nav: Decimal | None
    cash_by_currency: tuple[tuple[str, Decimal], ...]
    security_value_by_currency: tuple[tuple[str, Decimal], ...]
    missing_inputs: tuple[str, ...]
    stale_inputs: tuple[str, ...]
    as_of: datetime


def _append_entry(state: LedgerState, entry: JournalEntry) -> LedgerState:
    if any(existing.event_id == entry.event_id for existing in state.entries):
        raise ValueError(f"duplicate journal event: {entry.event_id}")
    return replace(state, entries=state.entries + (entry,))


def open_cash(
    state: LedgerState,
    *,
    event_id: str,
    currency: str,
    amount: Decimal,
    booked_at: datetime,
) -> LedgerState:
    if not amount.is_finite() or amount <= ZERO:
        raise ValueError("opening cash must be finite and positive")
    entry = JournalEntry(
        event_id=event_id,
        booked_at=booked_at,
        settlement_at=booked_at,
        description="opening cash",
        postings=(
            Posting(Account.CASH, currency, debit=amount),
            Posting(Account.CAPITAL, currency, credit=amount),
        ),
    )
    return _append_entry(state, entry)


def buy(
    state: LedgerState,
    *,
    event_id: str,
    lot_id: str,
    instrument_id: str,
    currency: str,
    quantity: Decimal,
    price: Decimal,
    fee: Decimal,
    trade_at: datetime,
    settlement_at: datetime,
) -> LedgerState:
    if any(lot.lot_id == lot_id for lot in state.lots):
        raise ValueError(f"duplicate lot_id: {lot_id}")
    if (
        not quantity.is_finite()
        or not price.is_finite()
        or not fee.is_finite()
        or quantity <= ZERO
        or price <= ZERO
        or fee < ZERO
    ):
        raise ValueError("invalid buy")
    _require_aware_utc(trade_at, "trade_at")
    _require_aware_utc(settlement_at, "settlement_at")
    if settlement_at < trade_at:
        raise ValueError("settlement cannot precede trade")
    gross = quantity * price
    postings = [
        Posting(
            Account.SECURITY_INVENTORY,
            currency,
            debit=gross,
            instrument_id=instrument_id,
            lot_id=lot_id,
        ),
        Posting(Account.CASH, currency, credit=gross + fee),
    ]
    if fee > ZERO:
        postings.append(Posting(Account.FEE_EXPENSE, currency, debit=fee))
    entry = JournalEntry(
        event_id,
        trade_at,
        settlement_at,
        f"buy {instrument_id}",
        tuple(postings),
    )
    lot = Lot(
        lot_id,
        instrument_id,
        currency,
        quantity,
        price,
        trade_at,
        settlement_at,
        event_id,
    )
    updated = _append_entry(state, entry)
    return replace(updated, lots=updated.lots + (lot,))


def sell_fifo(
    state: LedgerState,
    *,
    event_id: str,
    instrument_id: str,
    currency: str,
    quantity: Decimal,
    price: Decimal,
    fee: Decimal,
    trade_at: datetime,
    settlement_at: datetime,
) -> LedgerState:
    if (
        not quantity.is_finite()
        or not price.is_finite()
        or not fee.is_finite()
        or quantity <= ZERO
        or price <= ZERO
        or fee < ZERO
    ):
        raise ValueError("invalid sell")
    _require_aware_utc(trade_at, "trade_at")
    _require_aware_utc(settlement_at, "settlement_at")
    if settlement_at < trade_at:
        raise ValueError("settlement cannot precede trade")
    remaining = quantity
    cost = ZERO
    used_by_lot: dict[str, Decimal] = {}
    eligible = sorted(
        (
            lot
            for lot in state.lots
            if lot.instrument_id == instrument_id
            and lot.currency == currency
            and lot.acquired_at <= trade_at
            and lot.settlement_at <= trade_at
        ),
        key=lambda lot: (lot.acquired_at, lot.lot_id),
    )
    for lot in eligible:
        if remaining <= ZERO:
            break
        used = min(lot.quantity, remaining)
        cost += used * lot.unit_cost
        remaining -= used
        used_by_lot[lot.lot_id] = used
    if remaining > ZERO:
        raise ValueError("insufficient settled position")
    new_lots: list[Lot] = []
    for lot in state.lots:
        used = used_by_lot.get(lot.lot_id, ZERO)
        if used == ZERO:
            new_lots.append(lot)
        elif lot.quantity > used:
            new_lots.append(replace(lot, quantity=lot.quantity - used))
    proceeds = quantity * price
    if fee > proceeds:
        raise ValueError("fee exceeds sale proceeds")
    gain = proceeds - cost
    postings = [
        Posting(Account.CASH, currency, debit=proceeds - fee),
        Posting(
            Account.SECURITY_INVENTORY,
            currency,
            credit=cost,
            instrument_id=instrument_id,
        ),
    ]
    if fee > ZERO:
        postings.append(Posting(Account.FEE_EXPENSE, currency, debit=fee))
    if gain >= ZERO:
        if gain > ZERO:
            postings.append(
                Posting(Account.REALIZED_GAIN, currency, credit=gain)
            )
    else:
        postings.append(
            Posting(Account.REALIZED_LOSS, currency, debit=-gain)
        )
    entry = JournalEntry(
        event_id,
        trade_at,
        settlement_at,
        f"sell {instrument_id}",
        tuple(postings),
    )
    updated = _append_entry(state, entry)
    return replace(updated, lots=tuple(new_lots))


def account_balances(
    entries: Iterable[JournalEntry],
    account: Account,
) -> dict[str, Decimal]:
    balances: dict[str, Decimal] = {}
    for entry in entries:
        for posting in entry.postings:
            if posting.account is account:
                balances[posting.currency] = (
                    balances.get(posting.currency, ZERO)
                    + posting.debit
                    - posting.credit
                )
    return balances


def nav(
    state: LedgerState,
    *,
    marks: Iterable[Mark],
    fx_rates: Iterable[FxRate],
    base_currency: str,
    as_of: datetime,
    stale_after: timedelta,
) -> NavCertificate:
    _require_aware_utc(as_of, "as_of")
    if stale_after < timedelta(0):
        raise ValueError("stale_after must be non-negative")
    missing: list[str] = []
    stale: list[str] = []
    ambiguous: list[str] = []
    eligible_entries = tuple(
        entry
        for entry in state.entries
        if entry.booked_at <= as_of and entry.settlement_at <= as_of
    )
    eligible_lots = tuple(
        lot
        for lot in state.lots
        if lot.acquired_at <= as_of and lot.settlement_at <= as_of
    )
    mark_map = _select_marks(marks, as_of, ambiguous)
    fx_map = _select_fx(fx_rates, base_currency, as_of, ambiguous)
    cash = account_balances(eligible_entries, Account.CASH)
    security: dict[str, Decimal] = {}
    for lot in eligible_lots:
        mark = mark_map.get(lot.instrument_id)
        if mark is None or mark.currency != lot.currency:
            missing.append(f"mark:{lot.instrument_id}:{lot.currency}")
            continue
        effective = mark.known_at if mark.effective_at is None else mark.effective_at
        if as_of - effective > stale_after:
            stale.append(f"mark:{lot.instrument_id}")
            continue
        security[lot.currency] = (
            security.get(lot.currency, ZERO) + lot.quantity * mark.price
        )
    currencies = set(cash) | set(security)
    total = ZERO
    for currency in currencies:
        amount = cash.get(currency, ZERO) + security.get(currency, ZERO)
        if currency == base_currency:
            rate = Decimal("1")
        else:
            fx = fx_map.get(currency)
            if fx is None:
                missing.append(f"fx:{currency}/{base_currency}")
                continue
            effective = fx.known_at if fx.effective_at is None else fx.effective_at
            if as_of - effective > stale_after:
                stale.append(f"fx:{currency}/{base_currency}")
                continue
            rate = fx.rate
        converted = amount * rate
        if not converted.is_finite():
            ambiguous.append(f"nonfinite:{currency}")
            continue
        total += converted
    status = (
        NavStatus.PASS
        if not missing and not stale and not ambiguous and total.is_finite()
        else NavStatus.INVALID_OR_STALE_INPUT
    )
    return NavCertificate(
        status=status,
        base_currency=base_currency,
        nav=total if status is NavStatus.PASS else None,
        cash_by_currency=tuple(sorted(cash.items())),
        security_value_by_currency=tuple(sorted(security.items())),
        missing_inputs=tuple(sorted(set(missing + ambiguous))),
        stale_inputs=tuple(sorted(set(stale))),
        as_of=as_of,
    )


def _select_marks(
    marks: Iterable[Mark],
    as_of: datetime,
    ambiguity: list[str],
) -> dict[str, Mark]:
    grouped: dict[str, list[Mark]] = {}
    for mark in marks:
        effective = mark.known_at if mark.effective_at is None else mark.effective_at
        if effective <= as_of and mark.known_at <= as_of:
            grouped.setdefault(mark.instrument_id, []).append(mark)
    selected: dict[str, Mark] = {}
    for instrument, values in grouped.items():
        rank = max(
            (
                item.effective_at or item.known_at,
                item.known_at,
                item.version_id,
            )
            for item in values
        )
        top = [
            item
            for item in values
            if (item.effective_at or item.known_at, item.known_at, item.version_id)
            == rank
        ]
        if len({(item.currency, item.price) for item in top}) != 1:
            ambiguity.append(f"ambiguous-mark:{instrument}")
        else:
            selected[instrument] = min(top, key=lambda item: item.row_id)
    return selected


def _select_fx(
    rates: Iterable[FxRate],
    base_currency: str,
    as_of: datetime,
    ambiguity: list[str],
) -> dict[str, FxRate]:
    grouped: dict[str, list[FxRate]] = {}
    for rate in rates:
        effective = rate.known_at if rate.effective_at is None else rate.effective_at
        if (
            rate.base_currency == base_currency
            and effective <= as_of
            and rate.known_at <= as_of
        ):
            grouped.setdefault(rate.currency, []).append(rate)
    selected: dict[str, FxRate] = {}
    for currency, values in grouped.items():
        rank = max(
            (item.effective_at or item.known_at, item.known_at, item.version_id)
            for item in values
        )
        top = [
            item
            for item in values
            if (item.effective_at or item.known_at, item.known_at, item.version_id)
            == rank
        ]
        if len({item.rate for item in top}) != 1:
            ambiguity.append(f"ambiguous-fx:{currency}/{base_currency}")
        else:
            selected[currency] = min(top, key=lambda item: item.row_id)
    return selected


def _require_aware_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field} must be normalized to UTC")
