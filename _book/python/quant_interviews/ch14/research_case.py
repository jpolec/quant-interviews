from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, isfinite, log, pi, sqrt
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad


EULER_GAMMA = 0.5772156649015329
ONE_SIDED_95_Z = 1.6448536269514722


@dataclass(frozen=True)
class SelectionCertificate:
    observed_sharpe: float
    effective_trials: float
    sample_years: float
    sharpe_standard_error: float
    standard_error_method: str
    dependence_method: str
    expected_null_maximum: float
    selection_penalty: float
    adjusted_sharpe: float
    adjusted_lower_bound: float
    hurdle: float
    status: Literal["SELECTION_ADJUSTMENT_PASS", "SELECTION_ADJUSTMENT_REJECT"]


@dataclass(frozen=True)
class EconomicCertificate:
    gross_alpha: float
    turnover_value: float
    turnover_horizon: Literal["annual"]
    turnover_convention: Literal["one_way"]
    linear_cost: float
    nonlinear_impact: float
    net_alpha: float
    active_risk: float
    net_information_ratio: float
    status: Literal["ECONOMIC_PASS", "ECONOMIC_REJECT"]


@dataclass(frozen=True)
class ResearchLaunchDecision:
    required_gates: tuple[str, ...]
    failed_gates: tuple[str, ...]
    decision: Literal["PASS", "REJECT"]


def _finite(*values: float) -> bool:
    return all(isfinite(value) for value in values)


def annualized_sharpe(
    returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    sample = np.asarray(returns, dtype=float)
    if (
        sample.ndim != 1
        or sample.size < 3
        or not np.all(np.isfinite(sample))
        or periods_per_year <= 0
    ):
        raise ValueError("returns must be a finite vector with at least three values")
    volatility = float(np.std(sample, ddof=1))
    if volatility <= 0.0:
        raise ValueError("sample volatility must be positive")
    result = float(np.mean(sample) / volatility * sqrt(periods_per_year))
    if not isfinite(result):
        raise FloatingPointError("annualized Sharpe is non-finite")
    return result


def normal_max_centering(effective_trials: float) -> float:
    """Return the large-M Gaussian extreme-value centering term b_M."""

    if not isfinite(effective_trials) or effective_trials < 1.0:
        raise ValueError("effective_trials must be finite and at least one")
    if effective_trials == 1.0:
        return 0.0
    scale = sqrt(2.0 * log(effective_trials))
    return scale - (log(log(effective_trials)) + log(4.0 * pi)) / (2.0 * scale)


def _normal_pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def expected_max_standard_normal(effective_trials: float) -> float:
    """Return E[max Z_i] by numerical integration of its exact density.

    ``effective_trials`` may be non-integral when supplied by a governed
    dependence model; the density formula is then used as a documented smooth
    continuation of the integer-trial result.
    """

    if not isfinite(effective_trials) or effective_trials < 1.0:
        raise ValueError("effective_trials must be finite and at least one")
    if effective_trials == 1.0:
        return 0.0
    value, error = quad(
        lambda z: z
        * effective_trials
        * _normal_pdf(z)
        * _normal_cdf(z) ** (effective_trials - 1.0),
        -10.0,
        10.0,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=300,
    )
    if not isfinite(value) or error > 1.0e-8:
        raise FloatingPointError("Gaussian-maximum quadrature did not converge")
    return float(value)


def selection_adjusted_score(
    observed_sharpe: float,
    effective_trials: float,
    sample_years: float,
    *,
    sharpe_standard_error: float,
    standard_error_method: str,
    dependence_method: str,
    confidence_multiplier: float = ONE_SIDED_95_Z,
    hurdle: float = 0.0,
) -> SelectionCertificate:
    values = (
        observed_sharpe,
        sample_years,
        sharpe_standard_error,
        confidence_multiplier,
        hurdle,
    )
    if not _finite(*values) or sample_years <= 0.0:
        raise ValueError("selection inputs must be finite and years positive")
    if sharpe_standard_error <= 0.0 or confidence_multiplier < 0.0:
        raise ValueError("standard error must be positive and multiplier non-negative")
    if not standard_error_method.strip() or not dependence_method.strip():
        raise ValueError("standard-error and dependence methods are required")
    expected_maximum = expected_max_standard_normal(effective_trials)
    penalty = expected_maximum * sharpe_standard_error
    adjusted = observed_sharpe - penalty
    lower_bound = adjusted - confidence_multiplier * sharpe_standard_error
    status: Literal[
        "SELECTION_ADJUSTMENT_PASS", "SELECTION_ADJUSTMENT_REJECT"
    ] = (
        "SELECTION_ADJUSTMENT_PASS"
        if lower_bound > hurdle
        else "SELECTION_ADJUSTMENT_REJECT"
    )
    return SelectionCertificate(
        observed_sharpe=observed_sharpe,
        effective_trials=effective_trials,
        sample_years=sample_years,
        sharpe_standard_error=sharpe_standard_error,
        standard_error_method=standard_error_method,
        dependence_method=dependence_method,
        expected_null_maximum=expected_maximum,
        selection_penalty=penalty,
        adjusted_sharpe=adjusted,
        adjusted_lower_bound=lower_bound,
        hurdle=hurdle,
        status=status,
    )


def benjamini_hochberg_qvalues(
    p_values: NDArray[np.float64],
    alpha: float = 0.05,
    *,
    method: Literal["BH_POSITIVE_DEPENDENCE", "BY_ARBITRARY_DEPENDENCE"] = (
        "BH_POSITIVE_DEPENDENCE"
    ),
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
        raise ValueError("p_values must be a non-empty finite vector")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    if not isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if method not in {"BH_POSITIVE_DEPENDENCE", "BY_ARBITRARY_DEPENDENCE"}:
        raise ValueError("unsupported false-discovery policy")

    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    dependence_multiplier = (
        1.0
        if method == "BH_POSITIVE_DEPENDENCE"
        else float(np.sum(1.0 / np.arange(1, ranked.size + 1)))
    )
    raw = ranked * ranked.size * dependence_multiplier / np.arange(
        1, ranked.size + 1
    )
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_values = np.empty_like(adjusted)
    q_values[order] = adjusted
    return q_values, q_values <= alpha


def net_information_ratio(
    gross_alpha_annual: float,
    active_risk_annual: float,
    turnover_value: float,
    cost_per_unit_turnover: float,
    nonlinear_impact_annual: float = 0.0,
    *,
    turnover_horizon: Literal["annual"] = "annual",
    turnover_convention: Literal["one_way"] = "one_way",
) -> EconomicCertificate:
    values = (
        gross_alpha_annual,
        active_risk_annual,
        turnover_value,
        cost_per_unit_turnover,
        nonlinear_impact_annual,
    )
    if not _finite(*values):
        raise ValueError("economic inputs must be finite")
    if turnover_horizon != "annual" or turnover_convention != "one_way":
        raise ValueError("turnover must be annual and one_way for this certificate")
    if (
        active_risk_annual <= 0.0
        or turnover_value < 0.0
        or cost_per_unit_turnover < 0.0
        or nonlinear_impact_annual < 0.0
    ):
        raise ValueError("risk must be positive and cost inputs non-negative")
    linear_cost = turnover_value * cost_per_unit_turnover
    net_alpha = gross_alpha_annual - linear_cost - nonlinear_impact_annual
    ratio = net_alpha / active_risk_annual
    return EconomicCertificate(
        gross_alpha=gross_alpha_annual,
        turnover_value=turnover_value,
        turnover_horizon=turnover_horizon,
        turnover_convention=turnover_convention,
        linear_cost=linear_cost,
        nonlinear_impact=nonlinear_impact_annual,
        net_alpha=net_alpha,
        active_risk=active_risk_annual,
        net_information_ratio=ratio,
        status="ECONOMIC_PASS" if ratio > 0.0 else "ECONOMIC_REJECT",
    )


def research_launch_gate(
    selection: SelectionCertificate,
    economics: EconomicCertificate,
    *,
    point_in_time_pass: bool,
    holdout_untouched: bool,
    capacity_pass: bool,
    operational_readiness_pass: bool,
) -> ResearchLaunchDecision:
    states = {
        "selection_adjustment": selection.status == "SELECTION_ADJUSTMENT_PASS",
        "point_in_time": point_in_time_pass,
        "untouched_holdout": holdout_untouched,
        "after_cost_economics": economics.status == "ECONOMIC_PASS",
        "capacity": capacity_pass,
        "operational_readiness": operational_readiness_pass,
    }
    failed = tuple(name for name, passed in states.items() if not passed)
    return ResearchLaunchDecision(
        required_gates=tuple(states),
        failed_gates=failed,
        decision="REJECT" if failed else "PASS",
    )


def bounded_fold_stability(fold_sharpes: NDArray[np.float64]) -> float:
    values = np.asarray(fold_sharpes, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("fold_sharpes must contain at least two finite values")
    magnitude = abs(float(np.mean(values)))
    dispersion = float(np.std(values, ddof=1))
    result = magnitude / (magnitude + dispersion + np.finfo(float).eps)
    if not 0.0 <= result <= 1.0:
        raise FloatingPointError("bounded stability escaped [0, 1]")
    return result
