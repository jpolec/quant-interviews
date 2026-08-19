from __future__ import annotations

from dataclasses import dataclass
from math import ceil, expm1, fsum, isfinite, log1p


@dataclass(frozen=True)
class CollisionCertificate:
    items: int
    buckets: int
    expected_colliding_pairs: float
    exact_any_collision_probability: float
    poisson_any_collision_probability: float
    approximation_understates_exact: bool
    uniform_independent_hashes: bool
    status: str


@dataclass(frozen=True)
class HashSizingCertificate:
    items: int
    maximum_probability: float
    buckets: int
    risk_at_buckets: float
    risk_at_previous_bucket: float
    status: str


def expected_pair_collisions(items: int, buckets: int) -> float:
    if items < 0 or buckets <= 0:
        raise ValueError("invalid hash dimensions")
    return items * (items - 1) / (2.0 * buckets)


def probability_any_collision_exact(items: int, buckets: int) -> float:
    if items < 0 or buckets <= 0:
        raise ValueError("invalid hash dimensions")
    if items > buckets:
        return 1.0
    log_no_collision = fsum(log1p(-k / buckets) for k in range(items))
    return -expm1(log_no_collision)


def probability_any_collision_approx(items: int, buckets: int) -> float:
    return -expm1(-expected_pair_collisions(items, buckets))


def collision_certificate(
    items: int,
    buckets: int,
    *,
    uniform_independent_hashes: bool = True,
) -> CollisionCertificate:
    if items < 0 or buckets <= 0 or items > buckets:
        raise ValueError("certificate requires 0 <= items <= buckets")
    exact = probability_any_collision_exact(items, buckets)
    approximation = probability_any_collision_approx(items, buckets)
    ordered = (
        isfinite(exact)
        and isfinite(approximation)
        and 0.0 <= approximation <= exact + 1e-15 <= 1.0 + 1e-15
    )
    return CollisionCertificate(
        items=items,
        buckets=buckets,
        expected_colliding_pairs=expected_pair_collisions(items, buckets),
        exact_any_collision_probability=exact,
        poisson_any_collision_probability=approximation,
        approximation_understates_exact=approximation <= exact + 1e-15,
        uniform_independent_hashes=uniform_independent_hashes,
        status="PASS" if ordered and uniform_independent_hashes else "REJECT",
    )


def required_buckets(items: int, maximum_probability: float) -> int:
    """Approximate sizing followed by an exact monotone post-check."""
    if items < 2 or not 0.0 < maximum_probability < 1.0:
        raise ValueError("invalid sizing inputs")
    denominator = -2.0 * log1p(-maximum_probability)
    approximation = ceil(items * (items - 1) / denominator)
    lower = max(items - 1, approximation - 1)
    upper = max(approximation + items, items)
    while probability_any_collision_exact(items, upper) > maximum_probability:
        increment = max(items, upper - lower)
        lower = upper
        upper += increment
    while lower + 1 < upper:
        midpoint = (lower + upper) // 2
        if (
            probability_any_collision_exact(items, midpoint)
            <= maximum_probability
        ):
            upper = midpoint
        else:
            lower = midpoint
    return upper


def required_buckets_certificate(
    items: int,
    maximum_probability: float,
    *,
    uniform_independent_hashes: bool = True,
) -> HashSizingCertificate:
    buckets = required_buckets(items, maximum_probability)
    risk = probability_any_collision_exact(items, buckets)
    previous = probability_any_collision_exact(items, buckets - 1)
    passed = (
        uniform_independent_hashes
        and risk <= maximum_probability
        and previous > maximum_probability
    )
    return HashSizingCertificate(
        items=items,
        maximum_probability=maximum_probability,
        buckets=buckets,
        risk_at_buckets=risk,
        risk_at_previous_bucket=previous,
        status="PASS" if passed else "REJECT",
    )
