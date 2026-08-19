"""Executable bindings for the governed Chapter 1 AI tool contracts.

The contracts are deterministic evidence interfaces.  They do not authorize a
deployment: a caller must still supply the human-approval bit and preserve the
returned hashes with the review record.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Any

import numpy as np

from quant_interviews.ch01.bayes import (
    beta_binomial_edge_update,
    gaussian_log_predictive_score,
    positive_predictive_value,
)
from quant_interviews.ch01.dependence import empirical_lower_tail_dependence
from quant_interviews.ch01.regression import newey_west_covariance
from quant_interviews.ch01.robust import huber_location, median_mad
from quant_interviews.ch01.sampling import (
    annualized_sharpe_standard_error,
    bootstrap_sharpe,
    normal_mle,
)
from quant_interviews.ch01.testing import (
    alpha_spending_schedule,
    benjamini_hochberg,
    minimum_detectable_mean,
    winner_curse_distribution,
)
from quant_interviews.ch01.utility import research_utility_breakdown


CODE_VERSION = "v0.12"


@dataclass(frozen=True)
class ToolBinding:
    """Connect one public contract name to one deterministic implementation."""

    implementation: str
    inputs: frozenset[str]
    adapter: Callable[[Mapping[str, Any], float], dict[str, Any]]


def _plain(value: Any) -> Any:
    """Convert NumPy/dataclass results to finite, canonical-JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("tool evidence must contain only finite numbers")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _ppv(arguments: Mapping[str, Any], _: float) -> dict[str, Any]:
    value = positive_predictive_value(
        prevalence=float(arguments["prevalence"]),
        sensitivity=float(arguments["sensitivity"]),
        specificity=float(arguments["specificity"]),
    )
    return {"posterior_probability": value}


def _tail_dependence(arguments: Mapping[str, Any], _: float) -> dict[str, Any]:
    return empirical_lower_tail_dependence(
        arguments["x"],
        arguments["y"],
        quantile=float(arguments["quantile"]),
    )


def _sharpe_uncertainty(
    arguments: Mapping[str, Any],
    _: float,
) -> dict[str, Any]:
    value = annualized_sharpe_standard_error(
        annualized_sharpe=float(arguments["annualized_sharpe"]),
        observations=int(arguments["observations"]),
        periods_per_year=float(arguments["periods_per_year"]),
        skewness=float(arguments["skewness"]),
        excess_kurtosis=float(arguments["excess_kurtosis"]),
    )
    return {"standard_error": value}


def _block_bootstrap(arguments: Mapping[str, Any], _: float) -> dict[str, Any]:
    confidence = float(arguments["confidence"])
    values = bootstrap_sharpe(
        arguments["returns"],
        expected_block=float(arguments["expected_block"]),
        draws=int(arguments["draws"]),
        seed=int(arguments["seed"]),
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap Sharpe draws must all be finite")
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return {
        "median_sharpe": float(np.median(values)),
        "lower_quantile": float(lower),
        "upper_quantile": float(upper),
        "draw_count": int(values.size),
    }


def _normal_mle(arguments: Mapping[str, Any], _: float) -> dict[str, Any]:
    return normal_mle(arguments["values"])


def _robust_location(arguments: Mapping[str, Any], tolerance: float) -> dict[str, Any]:
    summary = median_mad(arguments["values"])
    keywords: dict[str, Any] = {
        "tuning": float(arguments["tuning"]),
        "iterations": int(arguments["iterations"]),
    }
    # v0.12 exposes tolerance.  The compatibility branch keeps this binding
    # executable while an older checkout is being migrated atomically.
    try:
        estimate = huber_location(
            arguments["values"],
            tolerance=tolerance,
            **keywords,
        )
    except TypeError as error:
        if "tolerance" not in str(error):
            raise
        estimate = huber_location(arguments["values"], **keywords)
    if is_dataclass(estimate) and not isinstance(estimate, type):
        governed = asdict(estimate)
        location = float(governed.get("estimate", governed.get("location")))
        converged = bool(governed.get("converged", True))
        iterations_used = int(
            governed.get("iterations_used", governed.get("iterations", 0))
        )
        final_step = float(governed.get("final_step", 0.0))
    elif isinstance(estimate, Mapping):
        governed = estimate
        location = float(governed.get("estimate", governed.get("location")))
        converged = bool(governed.get("converged", True))
        iterations_used = int(
            governed.get("iterations_used", governed.get("iterations", 0))
        )
        final_step = float(governed.get("final_step", 0.0))
    else:
        location = float(estimate)
        converged = True
        iterations_used = int(arguments["iterations"])
        final_step = 0.0
    return {
        "median": summary["median"],
        "mad": summary["mad"],
        "robust_scale": summary["sigma_consistent"],
        "huber_location": location,
        "converged": converged,
        "iterations_used": iterations_used,
        "final_step": final_step,
    }


def _multiple_testing(arguments: Mapping[str, Any], _: float) -> dict[str, Any]:
    p_values = np.asarray(arguments["p_values"], dtype=float)
    rejected = benjamini_hochberg(
        p_values,
        alpha=float(arguments["alpha"]),
    )
    cutoff = float(np.max(p_values[rejected])) if np.any(rejected) else None
    return {
        "discoveries": int(rejected.sum()),
        "adaptive_cutoff": cutoff,
        "rejection_mask": rejected.tolist(),
    }


def _sequential_plan(arguments: Mapping[str, Any], _: float) -> dict[str, Any]:
    total_alpha = float(arguments["total_alpha"])
    schedule = alpha_spending_schedule(total_alpha, int(arguments["looks"]))
    spent = float(schedule.sum())
    return {
        "alpha_schedule": schedule.tolist(),
        "spent_alpha": spent,
        "remaining_alpha": total_alpha - spent,
    }


def _minimum_detectable(arguments: Mapping[str, Any], _: float) -> dict[str, Any]:
    value = minimum_detectable_mean(
        std=float(arguments["volatility"]),
        n=int(arguments["effective_sample_size"]),
        z_alpha=float(arguments["z_alpha"]),
        z_power=float(arguments["z_power"]),
    )
    return {"minimum_detectable_mean": value}


def _winner_curse(arguments: Mapping[str, Any], _: float) -> dict[str, Any]:
    confidence = float(arguments["confidence"])
    summary = winner_curse_distribution(
        models=int(arguments["models"]),
        observations=int(arguments["observations"]),
        experiments=int(arguments["experiments"]),
        true_mean=float(arguments["true_mean"]),
        seed=int(arguments["seed"]),
    )
    gaps = np.asarray(summary["selection_gaps"], dtype=float)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(gaps, [tail, 1.0 - tail])
    return {
        "mean_selection_gap": summary["mean_selection_gap"],
        "median_selection_gap": summary["median_selection_gap"],
        "expected_null_maximum": summary["expected_null_maximum"],
        "lower_selection_gap": float(lower),
        "upper_selection_gap": float(upper),
        "experiment_count": int(gaps.size),
    }


def _edge_posterior(arguments: Mapping[str, Any], _: float) -> dict[str, Any]:
    return beta_binomial_edge_update(
        prior_alpha=float(arguments["prior_alpha"]),
        prior_beta=float(arguments["prior_beta"]),
        wins=int(arguments["wins"]),
        trials=int(arguments["trials"]),
        lower_quantile=float(arguments["lower_quantile"]),
        kelly_fraction=float(arguments["kelly_fraction"]),
    )


def _predictive_score(arguments: Mapping[str, Any], _: float) -> dict[str, Any]:
    observations = arguments["observations"]
    means = arguments["means"]
    variances = arguments["variances"]
    total = gaussian_log_predictive_score(observations, means, variances)
    pointwise = [
        gaussian_log_predictive_score([value], [mean], [variance])
        for value, mean, variance in zip(
            observations,
            means,
            variances,
            strict=True,
        )
    ]
    return {
        "total_log_predictive_score": total,
        "pointwise_log_predictive_scores": pointwise,
    }


def _hac_covariance(arguments: Mapping[str, Any], _: float) -> dict[str, Any]:
    covariance = newey_west_covariance(
        arguments["design"],
        arguments["residuals"],
        int(arguments["max_lag"]),
    )
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2.0)
    return {
        "covariance": covariance.tolist(),
        "maximum_variance": float(np.max(np.diag(covariance))),
        "minimum_eigenvalue": float(np.min(eigenvalues)),
    }


def _expected_utility(arguments: Mapping[str, Any], _: float) -> dict[str, Any]:
    return research_utility_breakdown(
        gross_alpha=float(arguments["gross_alpha"]),
        trading_cost=float(arguments["trading_cost"]),
        variance=float(arguments["variance"]),
        risk_aversion=float(arguments["risk_aversion"]),
        tail_penalty=float(arguments["tail_penalty"]),
        model_penalty=float(arguments["model_penalty"]),
        capacity_scale=float(arguments["capacity_scale"]),
    )


def _binding(
    implementation: str,
    inputs: tuple[str, ...],
    adapter: Callable[[Mapping[str, Any], float], dict[str, Any]],
) -> ToolBinding:
    return ToolBinding(implementation, frozenset(inputs), adapter)


TOOL_BINDINGS: dict[str, ToolBinding] = {
    "compute_positive_predictive_value": _binding(
        "quant_interviews.ch01.bayes.positive_predictive_value",
        ("prevalence", "sensitivity", "specificity"),
        _ppv,
    ),
    "estimate_tail_dependence": _binding(
        "quant_interviews.ch01.dependence.empirical_lower_tail_dependence",
        ("x", "y", "quantile"),
        _tail_dependence,
    ),
    "estimate_sharpe_uncertainty": _binding(
        "quant_interviews.ch01.sampling.annualized_sharpe_standard_error",
        (
            "annualized_sharpe",
            "observations",
            "periods_per_year",
            "skewness",
            "excess_kurtosis",
        ),
        _sharpe_uncertainty,
    ),
    "run_block_bootstrap": _binding(
        "quant_interviews.ch01.sampling.bootstrap_sharpe",
        ("returns", "expected_block", "draws", "confidence", "seed"),
        _block_bootstrap,
    ),
    "compute_normal_mle": _binding(
        "quant_interviews.ch01.sampling.normal_mle",
        ("values",),
        _normal_mle,
    ),
    "estimate_robust_location": _binding(
        "quant_interviews.ch01.robust.huber_location",
        ("values", "tuning", "iterations"),
        _robust_location,
    ),
    "audit_multiple_testing": _binding(
        "quant_interviews.ch01.testing.benjamini_hochberg",
        ("p_values", "alpha"),
        _multiple_testing,
    ),
    "plan_sequential_test": _binding(
        "quant_interviews.ch01.testing.alpha_spending_schedule",
        ("total_alpha", "looks"),
        _sequential_plan,
    ),
    "compute_minimum_detectable_alpha": _binding(
        "quant_interviews.ch01.testing.minimum_detectable_mean",
        ("volatility", "effective_sample_size", "z_alpha", "z_power"),
        _minimum_detectable,
    ),
    "analyze_winner_curse": _binding(
        "quant_interviews.ch01.testing.winner_curse_distribution",
        (
            "models",
            "observations",
            "experiments",
            "true_mean",
            "confidence",
            "seed",
        ),
        _winner_curse,
    ),
    "update_trading_edge_posterior": _binding(
        "quant_interviews.ch01.bayes.beta_binomial_edge_update",
        (
            "prior_alpha",
            "prior_beta",
            "wins",
            "trials",
            "lower_quantile",
            "kelly_fraction",
        ),
        _edge_posterior,
    ),
    "score_gaussian_forecasts": _binding(
        "quant_interviews.ch01.bayes.gaussian_log_predictive_score",
        ("observations", "means", "variances"),
        _predictive_score,
    ),
    "compute_hac_covariance": _binding(
        "quant_interviews.ch01.regression.newey_west_covariance",
        ("design", "residuals", "max_lag"),
        _hac_covariance,
    ),
    "evaluate_expected_utility": _binding(
        "quant_interviews.ch01.utility.research_utility_breakdown",
        (
            "gross_alpha",
            "trading_cost",
            "variance",
            "risk_aversion",
            "tail_penalty",
            "model_penalty",
            "capacity_scale",
        ),
        _expected_utility,
    ),
}


def execute_tool(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one governed contract and return a schema-valid evidence envelope."""

    if name not in TOOL_BINDINGS:
        raise KeyError(f"unknown CH01 tool: {name}")
    binding = TOOL_BINDINGS[name]
    raw_arguments = dict(arguments)
    human_approved = raw_arguments.pop("human_approved", None)
    tolerance = raw_arguments.pop("tolerance", None)
    reported_tolerance = (
        float(tolerance)
        if isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and isfinite(float(tolerance))
        and float(tolerance) >= 0.0
        else 0.0
    )
    hash_payload = {"name": name, "arguments": arguments}
    try:
        input_hash = sha256(_canonical_bytes(hash_payload)).hexdigest()
    except (TypeError, ValueError):
        input_hash = sha256(repr(hash_payload).encode("utf-8")).hexdigest()
    seed = raw_arguments.get("seed")

    def envelope(
        *,
        status: str,
        reason: str,
        result: dict[str, Any] | None,
        error: dict[str, str] | None,
    ) -> dict[str, Any]:
        evidence_payload = {
            "name": name,
            "input_sha256": input_hash,
            "result": result,
            "status": status,
        }
        evidence_hash = sha256(_canonical_bytes(evidence_payload)).hexdigest()
        return {
            "status": status,
            "reason": reason,
            "result": result,
            "tolerance": reported_tolerance,
            "provenance": {
                "implementation": binding.implementation,
                "code_version": CODE_VERSION,
                "input_sha256": input_hash,
                "evidence_sha256": evidence_hash,
                "seed": int(seed) if seed is not None else None,
            },
            "human_approved": bool(human_approved),
            "error": error,
        }

    try:
        if type(human_approved) is not bool:
            raise ValueError("human_approved must be an explicit boolean")
        if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
            raise ValueError("tolerance must be an explicit finite number")
        if not isfinite(float(tolerance)) or float(tolerance) < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")
        supplied = frozenset(raw_arguments)
        if supplied != binding.inputs:
            missing = sorted(binding.inputs - supplied)
            unexpected = sorted(supplied - binding.inputs)
            raise ValueError(
                f"contract fields mismatch; missing={missing}, unexpected={unexpected}"
            )
        _canonical_bytes(raw_arguments)
        result = _plain(binding.adapter(raw_arguments, float(tolerance)))
        reason = (
            "deterministic evidence passed and human approval is recorded"
            if human_approved
            else "deterministic evidence passed; human approval remains required"
        )
        return envelope(
            status="PASS" if human_approved else "CONDITIONAL_PASS",
            reason=reason,
            result=result,
            error=None,
        )
    except (ArithmeticError, TypeError, ValueError) as error:
        return envelope(
            status="ERROR",
            reason="deterministic execution failed",
            result=None,
            error={"code": type(error).__name__, "message": str(error)},
        )
