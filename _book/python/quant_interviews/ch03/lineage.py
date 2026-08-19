"""Canonical CH03 quote serialization shared by Python and SQL validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib


_DECIMAL_QUANTUM = Decimal("0.00000001")
_ROW_FIELDS = (
    "instrument_id",
    "expiry",
    "strike",
    "option_type",
    "quote_convention",
    "bid",
    "ask",
    "valid_from",
    "valid_to",
    "known_at",
    "superseded_at",
    "revision_id",
    "quote_payload_hash",
)
def canonical_length_prefixed(value: str | None) -> str:
    """Encode text without delimiter ambiguity; encode SQL NULL as ``-1:``."""

    if value is None:
        return "-1:"
    if not isinstance(value, str):
        raise TypeError("canonical length-prefix values must be strings or None")
    return f"{len(value)}:{value}"


def canonical_decimal(value: object) -> str:
    """Return a finite decimal with exactly eight fractional digits."""

    if isinstance(value, bool):
        raise ValueError("boolean is not a canonical decimal")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("finite canonical decimal required") from error
    if not number.is_finite():
        raise ValueError("finite canonical decimal required")
    try:
        quantized = number.quantize(_DECIMAL_QUANTUM, rounding=ROUND_HALF_UP)
    except InvalidOperation as error:
        raise ValueError("decimal is outside the canonical 8dp range") from error
    return format(quantized, "f")


def canonical_date(value: object) -> str:
    """Return an ISO calendar date in ``YYYY-MM-DD`` form."""

    if isinstance(value, datetime):
        result = value.date()
    elif isinstance(value, date):
        result = value
    elif isinstance(value, str):
        try:
            result = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("canonical date must use YYYY-MM-DD") from error
    else:
        raise TypeError("canonical date must be a date or ISO string")
    return result.isoformat()


def canonical_timestamp(value: object) -> str:
    """Return an aware instant in UTC with fixed microsecond precision."""

    if isinstance(value, str):
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            instant = datetime.fromisoformat(candidate)
        except ValueError as error:
            raise ValueError("canonical timestamp must be ISO-8601") from error
    elif isinstance(value, datetime):
        instant = value
    else:
        raise TypeError("canonical timestamp must be a datetime or ISO string")
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("canonical timestamp must include a UTC offset")
    return (
        instant.astimezone(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )


def canonical_quote_payload(bid: object, ask: object) -> str:
    """Serialize the economically meaningful quote payload."""

    return (
        f"bid={canonical_length_prefixed(canonical_decimal(bid))};"
        f"ask={canonical_length_prefixed(canonical_decimal(ask))}"
    )


def quote_payload_hash(bid: object, ask: object) -> str:
    """Return lowercase SHA-256 of the canonical bid/ask payload."""

    return hashlib.sha256(
        canonical_quote_payload(bid, ask).encode("utf-8")
    ).hexdigest()


def _canonical_snapshot_values(row: Mapping[str, object]) -> dict[str, str | None]:
    missing = set(_ROW_FIELDS[:-1]).difference(row)
    if missing:
        raise ValueError(f"snapshot row is missing fields: {sorted(missing)}")
    instrument_id = row["instrument_id"]
    option_type = row["option_type"]
    quote_convention = row["quote_convention"]
    revision = row["revision_id"]
    if (
        not isinstance(instrument_id, str)
        or not instrument_id
        or not isinstance(option_type, str)
        or not option_type
        or not isinstance(quote_convention, str)
        or not quote_convention
        or isinstance(revision, bool)
    ):
        raise ValueError("nonempty quote keys and an integer revision are required")
    try:
        revision_integer = int(revision)
    except (TypeError, ValueError) as error:
        raise ValueError("revision_id must be a nonnegative base-10 integer") from error
    if revision_integer < 0 or str(revision_integer) != str(revision):
        raise ValueError("revision_id must be a nonnegative base-10 integer")

    bid = canonical_decimal(row["bid"])
    ask = canonical_decimal(row["ask"])
    if Decimal(bid) > Decimal(ask):
        raise ValueError("canonical quote requires bid <= ask")
    payload_hash = quote_payload_hash(bid, ask)
    supplied_hash = row.get("quote_payload_hash")
    if supplied_hash is not None and supplied_hash != payload_hash:
        raise ValueError("quote_payload_hash does not match canonical bid/ask payload")

    valid_to = row["valid_to"]
    superseded_at = row["superseded_at"]
    values: dict[str, str | None] = {
        "instrument_id": instrument_id,
        "expiry": canonical_date(row["expiry"]),
        "strike": canonical_decimal(row["strike"]),
        "option_type": option_type,
        "quote_convention": quote_convention,
        "bid": bid,
        "ask": ask,
        "valid_from": canonical_timestamp(row["valid_from"]),
        "valid_to": None if valid_to is None else canonical_timestamp(valid_to),
        "known_at": canonical_timestamp(row["known_at"]),
        "superseded_at": (
            None if superseded_at is None else canonical_timestamp(superseded_at)
        ),
        "revision_id": str(revision_integer),
        "quote_payload_hash": payload_hash,
    }
    if values["valid_to"] is not None and values["valid_from"] >= values["valid_to"]:
        raise ValueError("economic-valid interval must be nonempty")
    if (
        values["superseded_at"] is not None
        and values["known_at"] >= values["superseded_at"]
    ):
        raise ValueError("knowledge interval must be nonempty")
    return values


def canonical_snapshot_row(row: Mapping[str, object]) -> str:
    """Serialize one quote row with fixed field tags and ordering."""

    values = _canonical_snapshot_values(row)
    return ";".join(
        f"{field}={canonical_length_prefixed(values[field])}"
        for field in _ROW_FIELDS
    )


def canonical_snapshot_hash(rows: Sequence[Mapping[str, object]]) -> str:
    """Hash rows after text-key and natural numeric strike/revision ordering."""

    if not rows:
        raise ValueError("snapshot must contain at least one quote row")
    normalized = [_canonical_snapshot_values(row) for row in rows]
    normalized.sort(
        key=lambda row: (
            row["instrument_id"],
            row["expiry"],
            Decimal(row["strike"]),
            row["option_type"],
            row["quote_convention"],
            int(row["revision_id"]),
        )
    )
    payload = "".join(
        ";".join(
            f"{field}={canonical_length_prefixed(row[field])}"
            for field in _ROW_FIELDS
        )
        for row in normalized
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
