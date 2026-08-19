from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Factor:
    name: str
    median: float
    log_sigma: float

    def __post_init__(self) -> None:
        if (
            not self.name
            or not isfinite(self.median)
            or not isfinite(self.log_sigma)
            or self.median <= 0.0
            or self.log_sigma < 0.0
        ):
            raise ValueError(
                "factors require a name, positive finite median and "
                "non-negative finite log uncertainty"
            )


@dataclass(frozen=True)
class NotionalEstimate:
    currency: str
    market_scope: str
    counting_convention: str
    point_estimate: float
    q025: float
    median: float
    q975: float
    variance_contributions: tuple[float, ...]
    contribution_names: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class DecompositionEstimate:
    """Typed evidence needed before two Fermi decompositions are compared."""

    median: float
    log_sigma: float
    unit: str
    market_scope: str
    counting_convention: str
    method: str
    source_ids: tuple[str, ...]
    as_of: str

    def __post_init__(self) -> None:
        if (
            not isfinite(self.median)
            or self.median <= 0.0
            or not isfinite(self.log_sigma)
            or self.log_sigma < 0.0
            or not all(
                value.strip()
                for value in (
                    self.unit,
                    self.market_scope,
                    self.counting_convention,
                    self.method,
                    self.as_of,
                )
            )
            or not self.source_ids
            or any(not value.strip() for value in self.source_ids)
            or len(set(self.source_ids)) != len(self.source_ids)
        ):
            raise ValueError("decomposition evidence is incomplete or invalid")


@dataclass(frozen=True)
class ReconciliationResult:
    log_difference: float
    tolerance: float
    shared_source_ids: tuple[str, ...]
    statistically_compatible: bool
    independently_sourced: bool
    status: str


def point_estimate(factors: Mapping[str, float]) -> float:
    """Return the Fermi approximation from representative factors."""
    if not factors:
        raise ValueError("at least one factor is required")
    result = 1.0
    for value in factors.values():
        if not isfinite(value) or value <= 0.0:
            raise ValueError("all multiplicative factors must be positive and finite")
        result *= value
        if not isfinite(result):
            raise OverflowError("multiplicative estimate overflowed")
    return result


def exact_trading_notional(
    shares_outstanding: Sequence[float],
    prices: Sequence[float],
    turnover_rates: Sequence[float],
) -> float:
    """Compute sum_i Q_i P_i tau_i, counting each trade once."""
    shares = np.asarray(shares_outstanding, dtype=float)
    prices_array = np.asarray(prices, dtype=float)
    turnover = np.asarray(turnover_rates, dtype=float)
    if (
        shares.ndim != 1
        or shares.shape != prices_array.shape
        or shares.shape != turnover.shape
        or shares.size == 0
        or not np.all(np.isfinite(shares))
        or not np.all(np.isfinite(prices_array))
        or not np.all(np.isfinite(turnover))
        or np.any(shares <= 0.0)
        or np.any(prices_array <= 0.0)
        or np.any(turnover < 0.0)
    ):
        raise ValueError("issuer inputs must be finite, aligned and economically valid")
    if not np.any(turnover > 0.0):
        raise ValueError("at least one issuer must have positive turnover")
    try:
        with np.errstate(over="raise", invalid="raise"):
            issuer_notionals = shares * prices_array * turnover
            result = float(np.sum(issuer_notionals))
    except FloatingPointError as error:
        raise OverflowError("issuer notional overflowed") from error
    if not isfinite(result) or result <= 0.0:
        raise OverflowError("aggregate notional must be positive and finite")
    return result


def market_cap_turnover_notional(
    shares_outstanding: Sequence[float],
    prices: Sequence[float],
    turnover_rates: Sequence[float],
) -> tuple[float, float, float]:
    """Return market cap, market-cap-weighted turnover and exact notional."""
    shares = np.asarray(shares_outstanding, dtype=float)
    prices_array = np.asarray(prices, dtype=float)
    turnover = np.asarray(turnover_rates, dtype=float)
    notional = exact_trading_notional(shares, prices_array, turnover)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            market_values = shares * prices_array
            market_cap = float(np.sum(market_values))
            weighted_turnover = float(np.dot(market_values, turnover) / market_cap)
    except FloatingPointError as error:
        raise OverflowError("market-cap decomposition overflowed") from error
    if not all(isfinite(value) and value > 0.0 for value in (market_cap, weighted_turnover)):
        raise OverflowError("market-cap decomposition must be positive and finite")
    return market_cap, weighted_turnover, notional


def _validated_covariance(
    factors: Sequence[Factor],
    log_covariance: FloatArray | None,
) -> FloatArray:
    count = len(factors)
    if count == 0:
        raise ValueError("at least one factor is required")
    names = [factor.name for factor in factors]
    if len(names) != len(set(names)):
        raise ValueError("factor names must be unique stable identifiers")
    if log_covariance is None:
        covariance = np.diag([factor.log_sigma**2 for factor in factors])
    else:
        covariance = np.asarray(log_covariance, dtype=float)
        if covariance.shape != (count, count):
            raise ValueError("log covariance shape does not match factors")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("log covariance must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
            raise ValueError("log covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-12:
            raise ValueError("log covariance must be positive semidefinite")
        expected_variances = np.array(
            [factor.log_sigma**2 for factor in factors],
            dtype=float,
        )
        if not np.allclose(
            np.diag(covariance),
            expected_variances,
            atol=1e-12,
            rtol=1e-8,
        ):
            raise ValueError("covariance diagonal must match factor log sigmas")
    return covariance


def sample_estimate(
    factors: Sequence[Factor],
    draws: int = 100_000,
    seed: int = 7,
    log_covariance: FloatArray | None = None,
) -> FloatArray:
    """Sample a jointly lognormal multiplicative estimate."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    covariance = _validated_covariance(factors, log_covariance)
    means = np.array([log(factor.median) for factor in factors], dtype=float)
    rng = np.random.default_rng(seed)
    log_draws = rng.multivariate_normal(
        means,
        covariance,
        size=draws,
        check_valid="raise",
    )
    samples = np.exp(np.sum(log_draws, axis=1))
    if not np.all(np.isfinite(samples)):
        raise OverflowError("sampled estimate is not finite")
    return samples


def summarize(samples: FloatArray) -> dict[str, float]:
    values = np.asarray(samples, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("samples must be a non-empty finite positive vector")
    q025, q50, q975 = np.quantile(values, [0.025, 0.50, 0.975])
    return {
        "q025": float(q025),
        "median": float(q50),
        "q975": float(q975),
        "mean": float(np.mean(values)),
    }


def analytic_lognormal_summary(
    factors: Sequence[Factor],
    log_covariance: FloatArray | None = None,
    z_value: float = 1.959963984540054,
) -> dict[str, float]:
    """Return exact median and central-normal quantiles of the product."""
    if not isfinite(z_value) or z_value <= 0.0:
        raise ValueError("z_value must be positive and finite")
    covariance = _validated_covariance(factors, log_covariance)
    log_median = sum(log(factor.median) for factor in factors)
    variance = float(np.sum(covariance))
    if variance < -1e-12:
        raise ValueError("aggregate log variance must be non-negative")
    sigma = sqrt(max(variance, 0.0))
    median = exp(log_median)
    result = {
        "q025": exp(log_median - z_value * sigma),
        "median": median,
        "q975": exp(log_median + z_value * sigma),
        "mean": exp(log_median + variance / 2.0),
    }
    if not all(isfinite(value) and value > 0.0 for value in result.values()):
        raise OverflowError("analytic lognormal summary overflowed")
    return result


def uncertainty_contributions(
    factors: Sequence[Factor],
    log_covariance: FloatArray | None = None,
) -> dict[str, float]:
    """Allocate Var(log V) through Cov(log X_j, log V)."""
    covariance = _validated_covariance(factors, log_covariance)
    aggregate_covariance = covariance @ np.ones(len(factors))
    total_variance = float(np.sum(aggregate_covariance))
    if total_variance <= 0.0:
        raise ValueError("uncertainty allocation is not applicable at zero variance")
    contributions = aggregate_covariance / total_variance
    return {
        factor.name: float(contribution)
        for factor, contribution in zip(factors, contributions)
    }


def log_elasticities(factors: Mapping[str, float]) -> dict[str, float]:
    """Return elasticities; these are not uncertainty contributions."""
    point_estimate(factors)
    return {name: 1.0 for name in factors}


def reconcile_log_estimates(
    first_median: float,
    second_median: float,
    first_log_sigma: float,
    second_log_sigma: float,
    log_covariance: float = 0.0,
    z_value: float = 1.959963984540054,
) -> tuple[float, float, str]:
    inputs = (
        first_median,
        second_median,
        first_log_sigma,
        second_log_sigma,
        log_covariance,
        z_value,
    )
    if not all(isfinite(value) for value in inputs):
        raise ValueError("reconciliation inputs must be finite")
    if first_median <= 0.0 or second_median <= 0.0:
        raise ValueError("medians must be positive")
    if first_log_sigma < 0.0 or second_log_sigma < 0.0:
        raise ValueError("log sigmas must be non-negative")
    covariance_bound = first_log_sigma * second_log_sigma
    if abs(log_covariance) > covariance_bound + 1e-12:
        raise ValueError("log covariance violates the Cauchy bound")
    variance = (
        first_log_sigma**2
        + second_log_sigma**2
        - 2.0 * log_covariance
    )
    if variance < -1e-12 or z_value <= 0.0:
        raise ValueError("invalid covariance or quantile")
    delta = abs(log(first_median) - log(second_median))
    tolerance = z_value * max(variance, 0.0) ** 0.5
    return delta, tolerance, "PASS" if delta <= tolerance else "REJECT"


def reconcile_decompositions(
    first: DecompositionEstimate,
    second: DecompositionEstimate,
    log_covariance: float = 0.0,
    z_value: float = 1.959963984540054,
) -> ReconciliationResult:
    """Reconcile compatible units while exposing shared-source dependence."""
    if (
        first.unit != second.unit
        or first.market_scope != second.market_scope
        or first.counting_convention != second.counting_convention
    ):
        raise ValueError("decomposition units, scope and counting convention must match")
    delta, tolerance, statistical_status = reconcile_log_estimates(
        first.median,
        second.median,
        first.log_sigma,
        second.log_sigma,
        log_covariance,
        z_value,
    )
    shared = tuple(sorted(set(first.source_ids) & set(second.source_ids)))
    independent = not shared and first.method != second.method
    statistically_compatible = statistical_status == "PASS"
    return ReconciliationResult(
        log_difference=delta,
        tolerance=tolerance,
        shared_source_ids=shared,
        statistically_compatible=statistically_compatible,
        independently_sourced=independent,
        status="PASS" if statistically_compatible and independent else "REJECT",
    )


def governed_equity_notional_fixture(
    *,
    draws: int = 20_000,
    seed: int = 3,
) -> NotionalEstimate:
    """Return a reproducible USD consolidated-market Fermi estimate."""
    if isinstance(draws, bool) or draws <= 0 or isinstance(seed, bool):
        raise ValueError("draws must be positive and seed must be an integer")
    factors = (
        Factor("listed_equity_market_cap_usd", 48_000_000_000_000.0, 0.10),
        Factor("daily_turnover_of_market_cap", 0.0038, 0.20),
        Factor("off_exchange_and_venue_coverage", 1.12, 0.08),
    )
    correlation = np.array(
        [
            [1.0, -0.20, 0.05],
            [-0.20, 1.0, 0.15],
            [0.05, 0.15, 1.0],
        ],
        dtype=float,
    )
    sigmas = np.array([factor.log_sigma for factor in factors])
    covariance = correlation * np.outer(sigmas, sigmas)
    # The primary certificate is analytic. ``draws`` and ``seed`` remain in the
    # API only for the separately reported Monte Carlo diagnostic/figure.
    summary = analytic_lognormal_summary(factors, covariance)
    contributions = uncertainty_contributions(factors, covariance)
    return NotionalEstimate(
        currency="USD",
        market_scope="US consolidated listed equities including off-exchange prints",
        counting_convention="single-counted traded notional",
        point_estimate=point_estimate(
            {factor.name: factor.median for factor in factors}
        ),
        q025=summary["q025"],
        median=summary["median"],
        q975=summary["q975"],
        variance_contributions=tuple(contributions.values()),
        contribution_names=tuple(contributions),
        status="PASS",
    )
