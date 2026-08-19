from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Mapping


ZERO = Decimal("0")


class BreakState(str, Enum):
    MATCH = "MATCH"
    MATERIAL = "MATERIAL"
    MISSING_INTERNAL = "MISSING_INTERNAL"
    MISSING_EXTERNAL = "MISSING_EXTERNAL"


@dataclass(frozen=True)
class Break:
    key: str
    internal: Decimal | None
    external: Decimal | None
    tolerance: Decimal
    state: BreakState
    unit: str = "UNSPECIFIED"
    policy_version: str = "UNSPECIFIED"

    def __post_init__(self) -> None:
        if not self.key or not self.unit or not self.policy_version:
            raise ValueError("break identity, unit and policy are required")
        if not self.tolerance.is_finite() or self.tolerance < ZERO:
            raise ValueError("break tolerance must be finite and non-negative")
        if self.internal is not None and not self.internal.is_finite():
            raise ValueError("internal value must be finite")
        if self.external is not None and not self.external.is_finite():
            raise ValueError("external value must be finite")

    @property
    def difference(self) -> Decimal | None:
        if self.internal is None or self.external is None:
            return None
        return self.internal - self.external

    @property
    def material(self) -> bool:
        return self.state is not BreakState.MATCH


@dataclass(frozen=True)
class ServiceLevelResult:
    observed_bad_fraction: Decimal
    remaining_budget_fraction: Decimal
    burn_rate: Decimal
    within_budget: bool
    bad_minutes: Decimal
    window_minutes: Decimal
    slo_id: str
    policy_version: str
    input_hash: str

    @property
    def consumed_fraction(self) -> Decimal:
        return self.observed_bad_fraction

    @property
    def remaining_fraction(self) -> Decimal:
        return self.remaining_budget_fraction


def reconcile(
    internal: Mapping[str, Decimal],
    external: Mapping[str, Decimal],
    tolerances: Mapping[str, Decimal],
    *,
    default_tolerance: Decimal | None = None,
    expected_keys: tuple[str, ...] | None = None,
    unit: str = "UNSPECIFIED",
    policy_version: str = "UNSPECIFIED",
) -> tuple[Break, ...]:
    keys = (
        sorted(set(internal) | set(external))
        if expected_keys is None
        else sorted(set(expected_keys))
    )
    result: list[Break] = []
    for key in keys:
        tolerance = tolerances.get(key, default_tolerance)
        if tolerance is None:
            raise ValueError(f"missing tolerance for {key}")
        if not tolerance.is_finite() or tolerance < 0:
            raise ValueError(f"invalid tolerance for {key}")
        left = internal.get(key)
        right = external.get(key)
        if left is not None and not left.is_finite():
            raise ValueError(f"non-finite internal value for {key}")
        if right is not None and not right.is_finite():
            raise ValueError(f"non-finite external value for {key}")
        if left is None:
            state = BreakState.MISSING_INTERNAL
        elif right is None:
            state = BreakState.MISSING_EXTERNAL
        elif abs(left - right) > tolerance:
            state = BreakState.MATERIAL
        else:
            state = BreakState.MATCH
        result.append(Break(key, left, right, tolerance, state, unit, policy_version))
    return tuple(result)


def service_level(
    error_budget: Decimal,
    bad_minutes: Decimal,
    window_minutes: Decimal,
    *,
    slo_id: str,
    policy_version: str,
    input_hash: str,
) -> ServiceLevelResult:
    if (
        not error_budget.is_finite()
        or not bad_minutes.is_finite()
        or not window_minutes.is_finite()
        or not ZERO <= error_budget <= Decimal("1")
        or window_minutes <= ZERO
        or not ZERO <= bad_minutes <= window_minutes
    ):
        raise ValueError("invalid SLO inputs")
    if not slo_id or not policy_version:
        raise ValueError("SLO identity and policy version are required")
    if len(input_hash) != 64 or set(input_hash) - set("0123456789abcdef"):
        raise ValueError("input_hash must be SHA-256")
    observed = bad_minutes / window_minutes
    if error_budget == ZERO:
        burn_rate = ZERO if observed == ZERO else Decimal("Infinity")
    else:
        burn_rate = observed / error_budget
    return ServiceLevelResult(
        observed_bad_fraction=observed,
        remaining_budget_fraction=error_budget - observed,
        burn_rate=burn_rate,
        within_budget=observed <= error_budget,
        bad_minutes=bad_minutes,
        window_minutes=window_minutes,
        slo_id=slo_id,
        policy_version=policy_version,
        input_hash=input_hash,
    )
