from __future__ import annotations

from dataclasses import dataclass
from math import copysign, fma, isfinite, sqrt
from typing import Iterable


@dataclass(frozen=True)
class Tolerance:
    absolute: float = 1e-12
    relative: float = 1e-10

    def __post_init__(self) -> None:
        if (
            not isfinite(self.absolute)
            or not isfinite(self.relative)
            or self.absolute < 0
            or self.relative < 0
        ):
            raise ValueError("tolerances must be non-negative")


def almost_equal(a: float, b: float, tolerance: Tolerance = Tolerance()) -> bool:
    left = float(a)
    right = float(b)
    if not isfinite(left) or not isfinite(right):
        return False
    return abs(left - right) <= max(
        tolerance.absolute,
        tolerance.relative * max(abs(left), abs(right)),
    )


def kahan_sum(values: Iterable[float]) -> float:
    total = 0.0
    compensation = 0.0
    for value in values:
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError("values must be finite")
        adjusted = numeric - compensation
        updated = total + adjusted
        compensation = (updated - total) - adjusted
        total = updated
    return total


def stable_quadratic_roots(a: float, b: float, c: float) -> tuple[float, float]:
    if not all(isfinite(value) for value in (a, b, c)):
        raise ValueError("coefficients must be finite")
    if a == 0.0:
        raise ValueError("a must be non-zero")
    scale = max(abs(a), abs(b), abs(c))
    scaled_a, scaled_b, scaled_c = a / scale, b / scale, c / scale
    discriminant = fma(scaled_b, scaled_b, -4.0 * scaled_a * scaled_c)
    roundoff_floor = 32.0 * float.fromhex("0x1.0p-52") * (
        scaled_b * scaled_b + abs(4.0 * scaled_a * scaled_c)
    )
    if discriminant < -roundoff_floor:
        raise ValueError("real roots do not exist")
    root = sqrt(max(discriminant, 0.0))
    q = -0.5 * (scaled_b + copysign(root, scaled_b))
    if q == 0.0:
        repeated = -scaled_b / (2.0 * scaled_a)
        return repeated, repeated
    x1 = q / scaled_a
    x2 = scaled_c / q
    return (x1, x2) if x1 <= x2 else (x2, x1)
