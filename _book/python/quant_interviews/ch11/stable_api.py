from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from enum import Enum
import json
import math
import re
from typing import Literal

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IMPLEMENTATION_HASH = sha256(
    b"quant_interviews.ch11.stable_api:price_cash_instrument:v0.10"
).hexdigest()
PRICING_CODE_VERSION = "ch11.price_cash.v0.10.0"
DEFAULT_CONFIG_HASH = sha256(
    b"cash-pricing-config:v1|quantity=positive|currency=exact|pit=dual-clock"
).hexdigest()


class PricingErrorCode(str, Enum):
    MISSING_INPUT = "MISSING_INPUT"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    FUTURE_VALID_TIME = "FUTURE_VALID_TIME"
    LOOK_AHEAD = "LOOK_AHEAD"


class PricingErrorCategory(str, Enum):
    DATA = "DATA"
    CONTRACT = "CONTRACT"
    TEMPORAL = "TEMPORAL"


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("decimal value must be finite")
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _utf16_sort_key(value: str) -> bytes:
    return value.encode("utf-16-be", errors="surrogatepass")


def _canonical_text(payload: object) -> str:
    if payload is None:
        return "null"
    if payload is True:
        return "true"
    if payload is False:
        return "false"
    if isinstance(payload, int):
        return str(payload)
    if isinstance(payload, str):
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if isinstance(payload, (list, tuple)):
        return "[" + ",".join(_canonical_text(value) for value in payload) + "]"
    if isinstance(payload, dict):
        if any(not isinstance(key, str) for key in payload):
            raise TypeError("canonical JSON object keys must be strings")
        return "{" + ",".join(
            _canonical_text(key) + ":" + _canonical_text(payload[key])
            for key in sorted(payload, key=_utf16_sort_key)
        ) + "}"
    raise TypeError(
        "restricted canonical JSON accepts only null, booleans, integers, "
        "strings, arrays and string-keyed objects"
    )


def canonical_json(payload: object) -> bytes:
    """Canonical UTF-8 JSON for the API's restricted JCS data model.

    Object keys use the RFC 8785 UTF-16 code-unit ordering. Floating-point
    values are deliberately excluded; economic decimals cross the boundary as
    normalized strings.
    """

    return _canonical_text(payload).encode("utf-8")


def _require_sha256(value: str, field: str) -> None:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, order=True)
class Quote:
    instrument_id: str
    price: Decimal
    currency: str

    def __post_init__(self) -> None:
        if not self.instrument_id or not self.currency:
            raise ValueError("instrument_id and currency are required")
        if len(self.currency) != 3 or not self.currency.isalpha() or not self.currency.isupper():
            raise ValueError("currency must be a three-letter uppercase code")
        if not isinstance(self.price, Decimal):
            try:
                object.__setattr__(self, "price", Decimal(str(self.price)))
            except InvalidOperation as error:
                raise ValueError("invalid price") from error
        if not self.price.is_finite() or self.price <= 0:
            raise ValueError("cash-instrument quote must be finite and positive")


@dataclass(frozen=True)
class MarketSnapshot:
    snapshot_id: str
    valid_time: datetime
    knowledge_time: datetime
    quotes: tuple[Quote, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported schema_version")
        if not self.snapshot_id:
            raise ValueError("snapshot_id is required")
        _utc_text(self.valid_time)
        _utc_text(self.knowledge_time)
        copied = tuple(sorted(tuple(self.quotes)))
        if len({quote.instrument_id for quote in copied}) != len(copied):
            raise ValueError("duplicate instrument quote")
        object.__setattr__(self, "quotes", copied)

    def _semantic_payload(self) -> dict[str, object]:
        return {
            "knowledge_time": _utc_text(self.knowledge_time),
            "quotes": [
                {
                    "currency": quote.currency,
                    "instrument_id": quote.instrument_id,
                    "price": _decimal_text(quote.price),
                }
                for quote in self.quotes
            ],
            "schema_version": self.schema_version,
            "valid_time": _utc_text(self.valid_time),
        }

    def semantic_content_hash(self) -> str:
        return sha256(canonical_json(self._semantic_payload())).hexdigest()

    def artifact_hash(self) -> str:
        payload = {"snapshot_id": self.snapshot_id, **self._semantic_payload()}
        return sha256(canonical_json(payload)).hexdigest()

    def quote(self, instrument_id: str, *, correlation_id: str = "snapshot-query") -> Quote:
        for quote in self.quotes:
            if quote.instrument_id == instrument_id:
                return quote
        raise PricingError(
            PricingErrorCode.MISSING_INPUT,
            f"missing quote: {instrument_id}",
            correlation_id=correlation_id,
            details=(("instrument_id", instrument_id),),
        )


@dataclass(frozen=True)
class PriceRequest:
    request_id: str
    idempotency_key: str
    instrument_id: str
    quantity: float
    currency: str
    side: Literal["long", "short"]
    valuation_time: datetime
    knowledge_cutoff: datetime
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported request schema_version")
        if not self.request_id or not self.idempotency_key or not self.instrument_id:
            raise ValueError("request identity fields are required")
        if not math.isfinite(self.quantity) or self.quantity <= 0:
            raise ValueError("quantity must be a finite positive magnitude")
        if not self.currency:
            raise ValueError("currency is required")
        if len(self.currency) != 3 or not self.currency.isalpha() or not self.currency.isupper():
            raise ValueError("currency must be a three-letter uppercase code")
        if self.side not in {"long", "short"}:
            raise ValueError("invalid side")
        _utc_text(self.valuation_time)
        _utc_text(self.knowledge_cutoff)

    def content_hash(self) -> str:
        payload = {
            "currency": self.currency,
            "idempotency_key": self.idempotency_key,
            "instrument_id": self.instrument_id,
            "knowledge_cutoff": _utc_text(self.knowledge_cutoff),
            "quantity": _decimal_text(Decimal(str(self.quantity))),
            "request_id": self.request_id,
            "schema_version": self.schema_version,
            "side": self.side,
            "valuation_time": _utc_text(self.valuation_time),
        }
        return sha256(canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class PriceResult:
    value: Decimal
    currency: str
    request_hash: str
    snapshot_content_hash: str
    snapshot_artifact_hash: str
    model_version: str
    schema_version: int
    status: Literal["PASS"]
    code_version: str = PRICING_CODE_VERSION
    config_hash: str = DEFAULT_CONFIG_HASH
    implementation_hash: str = IMPLEMENTATION_HASH
    output_hash: str = ""

    def __post_init__(self) -> None:
        value = self.value
        if not isinstance(value, Decimal):
            try:
                value = Decimal(str(value))
            except InvalidOperation as error:
                raise ValueError("invalid result value") from error
            object.__setattr__(self, "value", value)
        if not value.is_finite():
            raise ValueError("result value must be finite")
        if len(self.currency) != 3 or not self.currency.isalpha() or not self.currency.isupper():
            raise ValueError("result currency must be a three-letter uppercase code")
        if not self.model_version or not self.code_version:
            raise ValueError("model_version and code_version are required")
        if self.schema_version != 1 or self.status != "PASS":
            raise ValueError("unsupported result schema/status")
        _require_sha256(self.request_hash, "request_hash")
        _require_sha256(self.snapshot_content_hash, "snapshot_content_hash")
        _require_sha256(self.snapshot_artifact_hash, "snapshot_artifact_hash")
        _require_sha256(self.config_hash, "config_hash")
        _require_sha256(self.implementation_hash, "implementation_hash")
        payload = {
            "code_version": self.code_version,
            "config_hash": self.config_hash,
            "currency": self.currency,
            "implementation_hash": self.implementation_hash,
            "model_version": self.model_version,
            "request_hash": self.request_hash,
            "schema_version": self.schema_version,
            "snapshot_artifact_hash": self.snapshot_artifact_hash,
            "snapshot_content_hash": self.snapshot_content_hash,
            "status": self.status,
            "value": _decimal_text(value),
        }
        expected = sha256(canonical_json(payload)).hexdigest()
        if self.output_hash:
            _require_sha256(self.output_hash, "output_hash")
            if self.output_hash != expected:
                raise ValueError("output_hash does not match the result payload")
        else:
            object.__setattr__(self, "output_hash", expected)


class PricingError(RuntimeError):
    _POLICY = {
        PricingErrorCode.MISSING_INPUT: (PricingErrorCategory.DATA, False),
        PricingErrorCode.CURRENCY_MISMATCH: (PricingErrorCategory.CONTRACT, False),
        PricingErrorCode.FUTURE_VALID_TIME: (PricingErrorCategory.TEMPORAL, False),
        PricingErrorCode.LOOK_AHEAD: (PricingErrorCategory.TEMPORAL, False),
    }

    def __init__(
        self,
        code: PricingErrorCode | str,
        message: str,
        *,
        correlation_id: str,
        details: tuple[tuple[str, str], ...] = (),
        cause_code: str | None = None,
    ) -> None:
        try:
            parsed_code = PricingErrorCode(code)
        except ValueError as error:
            raise ValueError("unsupported pricing error code") from error
        if not message or not correlation_id:
            raise ValueError("pricing error message and correlation_id are required")
        if cause_code is not None and not cause_code:
            raise ValueError("cause_code must be non-empty when supplied")
        if any(not key or not value for key, value in details):
            raise ValueError("pricing error details require non-empty keys and values")
        normalized_details = tuple(sorted(details))
        if len({key for key, _ in normalized_details}) != len(normalized_details):
            raise ValueError("pricing error detail keys must be unique")
        super().__init__(message)
        category, retryable = self._POLICY[parsed_code]
        self.code = parsed_code.value
        self.category = category.value
        self.retryable = retryable
        self.correlation_id = correlation_id
        self.details = normalized_details
        self.cause_code = cause_code

    def evidence_hash(self) -> str:
        return sha256(
            canonical_json(
                {
                    "category": self.category,
                    "cause_code": self.cause_code,
                    "code": self.code,
                    "correlation_id": self.correlation_id,
                    "details": [list(item) for item in self.details],
                    "message": str(self),
                    "retryable": self.retryable,
                    "schema_version": 1,
                }
            )
        ).hexdigest()


def price_cash_instrument(
    request: PriceRequest,
    snapshot: MarketSnapshot,
    model_version: str = "1.0.0",
    config_hash: str = DEFAULT_CONFIG_HASH,
) -> PriceResult:
    if not model_version:
        raise ValueError("model_version is required")
    _require_sha256(config_hash, "config_hash")
    quote = snapshot.quote(request.instrument_id, correlation_id=request.request_id)
    if quote.currency != request.currency:
        raise PricingError(
            PricingErrorCode.CURRENCY_MISMATCH,
            f"quote currency {quote.currency} differs from request {request.currency}",
            correlation_id=request.request_id,
            details=(("quote_currency", quote.currency), ("request_currency", request.currency)),
        )
    if snapshot.valid_time > request.valuation_time:
        raise PricingError(
            PricingErrorCode.FUTURE_VALID_TIME,
            "snapshot is effective after valuation time",
            correlation_id=request.request_id,
            details=(("snapshot_id", snapshot.snapshot_id),),
        )
    if snapshot.knowledge_time > request.knowledge_cutoff:
        raise PricingError(
            PricingErrorCode.LOOK_AHEAD,
            "snapshot was not known by the request cutoff",
            correlation_id=request.request_id,
            details=(("snapshot_id", snapshot.snapshot_id),),
        )
    direction = Decimal("1") if request.side == "long" else Decimal("-1")
    value = direction * Decimal(str(request.quantity)) * quote.price
    return PriceResult(
        value=value,
        currency=request.currency,
        request_hash=request.content_hash(),
        snapshot_content_hash=snapshot.semantic_content_hash(),
        snapshot_artifact_hash=snapshot.artifact_hash(),
        model_version=model_version,
        code_version=PRICING_CODE_VERSION,
        config_hash=config_hash,
        schema_version=1,
        status="PASS",
    )
