"""Economic-utility calculations for Chapter 1."""

from __future__ import annotations

from math import isfinite, log


def research_utility_breakdown(
    gross_alpha: float,
    trading_cost: float,
    variance: float,
    risk_aversion: float,
    *,
    tail_penalty: float = 0.0,
    model_penalty: float = 0.0,
    capacity_scale: float = 1.0,
) -> dict[str, float]:
    """Translate gross evidence into deployable certainty equivalent."""

    if not all(
        isfinite(value)
        for value in (
            gross_alpha,
            trading_cost,
            variance,
            risk_aversion,
            tail_penalty,
            model_penalty,
            capacity_scale,
        )
    ):
        raise ValueError("utility inputs must be finite")
    if (
        trading_cost < 0.0
        or variance < 0.0
        or risk_aversion < 0.0
        or tail_penalty < 0.0
        or model_penalty < 0.0
    ):
        raise ValueError("cost and risk penalties must be nonnegative")
    if not 0.0 <= capacity_scale <= 1.0:
        raise ValueError("capacity_scale must lie in [0, 1]")

    net_alpha = gross_alpha - trading_cost
    variance_penalty = 0.5 * risk_aversion * variance
    certainty_equivalent = (
        net_alpha
        - variance_penalty
        - tail_penalty
        - model_penalty
    )
    return {
        "gross_alpha": gross_alpha,
        "trading_cost": trading_cost,
        "net_alpha": net_alpha,
        "variance_penalty": variance_penalty,
        "tail_penalty": tail_penalty,
        "model_penalty": model_penalty,
        "certainty_equivalent": certainty_equivalent,
        "capacity_adjusted_value": capacity_scale * certainty_equivalent,
    }


def chapter_utility_examples() -> dict[str, dict[str, float]]:
    """Return the governed Signal A/B inputs and reconciled utility results."""

    specifications = {
        "Signal A": {
            "gross_alpha": 0.06,
            "trading_cost": 0.01,
            "variance": 0.06**2,
            "risk_aversion": 3.0,
            "tail_penalty": 0.001,
            "model_penalty": 0.0005,
            "capacity_scale": 1.0,
        },
        "Signal B": {
            "gross_alpha": 0.12,
            "trading_cost": 0.09,
            "variance": 0.12**2,
            "risk_aversion": 3.0,
            "tail_penalty": 0.004,
            "model_penalty": 0.002,
            "capacity_scale": 0.25,
        },
    }
    return {
        signal: {
            **values,
            **research_utility_breakdown(**values),
        }
        for signal, values in specifications.items()
    }


def mean_variance_certainty_equivalent(
    mean_return: float,
    variance: float,
    risk_aversion: float,
    cost: float = 0.0,
) -> float:
    """Compatibility helper for the mean-variance special case."""

    return research_utility_breakdown(
        gross_alpha=mean_return,
        trading_cost=cost,
        variance=variance,
        risk_aversion=risk_aversion,
    )["certainty_equivalent"]


def expected_log_growth(
    win_probability: float,
    win_return: float,
    loss_return: float,
    fraction: float,
) -> float:
    """Return expected log growth for a two-outcome payoff."""

    if not all(
        isfinite(value)
        for value in (win_probability, win_return, loss_return, fraction)
    ):
        raise ValueError("payoff inputs must be finite")
    if not 0.0 <= win_probability <= 1.0 or fraction < 0.0:
        raise ValueError("invalid probability or fraction")
    up_wealth = 1.0 + fraction * win_return
    down_wealth = 1.0 + fraction * loss_return
    if up_wealth <= 0.0 or down_wealth <= 0.0:
        return float("-inf")
    return (
        win_probability * log(up_wealth)
        + (1.0 - win_probability) * log(down_wealth)
    )
