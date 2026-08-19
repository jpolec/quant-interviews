"""Governed numerical decision record integrating the Chapter 3 contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import string

import numpy as np

from quant_interviews.ch03.autodiff import black_scholes_greek_crosscheck
from quant_interviews.ch03.barrier_pde import (
    barrier_far_boundary_study,
    barrier_pde_refinement,
)
from quant_interviews.ch03.calibration import CalibrationPolicy, calibrate
from quant_interviews.ch03.finite_difference import rannacher_comparison
from quant_interviews.ch03.implied_vol import call_price, implied_vol
from quant_interviews.ch03.lineage import (
    canonical_decimal,
    canonical_snapshot_hash,
    quote_payload_hash,
)


CASE_SNAPSHOT_GOLDEN_SHA256 = (
    "cf6d622f0dcdee9eebeacb29c5964d9514ed2656f0ff8d0b65ca6d6f737fb0a9"
)


@dataclass(frozen=True)
class ReleaseGatePolicy:
    """Versioned signed limits for the CH03 integrated decision."""

    version: str = "ch03-release-policy-v0.11"
    expected_rejected_quotes: int = 2
    maximum_implied_vol_failures: int = 0
    maximum_calibration_rmse: float = 1e-7
    expected_calibration_parameter_count: int = 3
    minimum_successful_restarts: int = 3
    maximum_failed_restarts: int = 0
    maximum_restart_dispersion: float = 1e-6
    maximum_heldout_rmse: float = 1e-6
    maximum_active_bounds: int = 0
    maximum_regularization_fraction: float = 0.25
    maximum_jacobian_condition: float = 100.0
    minimum_jacobian_singular_value: float = 20.0
    maximum_pde_price_error: float = 0.02
    maximum_pde_delta_error: float = 2e-4
    maximum_pde_gamma_error: float = 2e-5
    minimum_pde_observed_order: float = 1.8
    maximum_rannacher_oscillation_ratio: float = 0.10
    maximum_ad_bump_delta_error: float = 1e-8
    maximum_far_boundary_price_drift: float = 1e-8
    minimum_lineage_rows: int = 24
    required_monitoring: str = "continuous"

    def validate(self) -> None:
        """Reject non-finite, non-integer or unsigned policy controls."""

        limits = (
            self.maximum_calibration_rmse,
            self.maximum_restart_dispersion,
            self.maximum_heldout_rmse,
            self.maximum_jacobian_condition,
            self.minimum_jacobian_singular_value,
            self.maximum_pde_price_error,
            self.maximum_pde_delta_error,
            self.maximum_pde_gamma_error,
            self.minimum_pde_observed_order,
            self.maximum_rannacher_oscillation_ratio,
            self.maximum_ad_bump_delta_error,
            self.maximum_far_boundary_price_drift,
        )
        integer_controls = (
            self.expected_rejected_quotes,
            self.maximum_implied_vol_failures,
            self.minimum_lineage_rows,
            self.expected_calibration_parameter_count,
            self.minimum_successful_restarts,
            self.maximum_failed_restarts,
            self.maximum_active_bounds,
        )
        if (
            not self.version
            or not self.required_monitoring
            or not all(math.isfinite(value) and value > 0.0 for value in limits)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in integer_controls
            )
            or self.minimum_lineage_rows < 1
            or self.expected_calibration_parameter_count < 1
            or self.minimum_successful_restarts < 1
            or not math.isfinite(self.maximum_regularization_fraction)
            or not 0.0 <= self.maximum_regularization_fraction <= 1.0
        ):
            raise ValueError("finite signed release-gate policy required")


@dataclass(frozen=True)
class ReleaseGate:
    """One executable gate with publication-ready semantics."""

    name: str
    value: float | int | bool | str
    unit: str
    operator: str
    limit: float | int | bool | str
    owner: str
    required: bool
    passed: bool
    reason: str


@dataclass(frozen=True)
class ReleaseGateEvaluation:
    """Reduction of every required and advisory gate."""

    policy_version: str
    gates: tuple[ReleaseGate, ...]
    decision: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class NumericalCaseStudy:
    """Frozen evidence used by the chapter table and release decision."""

    fixture_type: str
    fixture_version: str
    snapshot_hash: str
    expected_snapshot_hash: str
    snapshot_hash_matches_artifact: bool
    snapshot_row_count: int
    quotes_received: int
    quotes_rejected: int
    implied_vol_failures: int
    weighted_calibration_rmse: float
    calibration_numerical_rank: int
    calibration_parameter_count: int
    calibration_nullity: int
    calibration_successful_restarts: int
    calibration_failed_restarts: int
    calibration_restart_dispersion: float
    calibration_heldout_rmse: float
    calibration_active_bounds: int
    calibration_regularization_fraction: float
    smallest_scaled_jacobian_singular_value: float
    scaled_jacobian_condition: float
    pde_grid_prices: tuple[float, ...]
    pde_grid_evidence: tuple[
        tuple[int, int, float, float, float, float],
        ...,
    ]
    finest_pde_reference_error: float
    finest_pde_delta_error: float
    finest_pde_gamma_error: float
    minimum_pde_price_observed_order: float
    minimum_pde_delta_observed_order: float
    minimum_pde_gamma_observed_order: float
    pde_errors_monotone: bool
    far_boundary_price_drift: float
    cn_curvature_oscillation: float
    rannacher_curvature_oscillation: float
    rannacher_oscillation_ratio: float
    greek_instrument: str
    ad_bump_delta_error: float
    release_policy_version: str
    release_gates: tuple[ReleaseGate, ...]
    decision_reasons: tuple[str, ...]
    decision: str


def _release_gate(
    name: str,
    value: float | int | bool | str,
    unit: str,
    operator: str,
    limit: float | int | bool | str,
    owner: str,
    required: bool,
    passed: bool,
    failure_reason: str,
) -> ReleaseGate:
    return ReleaseGate(
        name=name,
        value=value,
        unit=unit,
        operator=operator,
        limit=limit,
        owner=owner,
        required=required,
        passed=passed,
        reason="PASS" if passed else failure_reason,
    )


def _finite_number(evidence: Mapping[str, object], key: str) -> float:
    value = evidence[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite scalar")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{key} must be a finite scalar")
    return numeric


def _nonnegative_integer(evidence: Mapping[str, object], key: str) -> int:
    value = evidence[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return value


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
        and value == value.lower()
    )


def evaluate_release_gates(
    evidence: Mapping[str, object],
    policy: ReleaseGatePolicy | None = None,
) -> ReleaseGateEvaluation:
    """Evaluate every signed CH03 gate and reduce them to one decision."""

    governed_policy = ReleaseGatePolicy() if policy is None else policy
    governed_policy.validate()
    required_keys = {
        "quotes_rejected",
        "implied_vol_failures",
        "calibration_status",
        "weighted_calibration_rmse",
        "calibration_numerical_rank",
        "calibration_parameter_count",
        "calibration_nullity",
        "calibration_successful_restarts",
        "calibration_failed_restarts",
        "calibration_restart_dispersion",
        "calibration_heldout_rmse",
        "calibration_active_bounds",
        "calibration_regularization_fraction",
        "scaled_jacobian_condition",
        "smallest_scaled_jacobian_singular_value",
        "finest_pde_price_error",
        "finest_pde_delta_error",
        "finest_pde_gamma_error",
        "minimum_pde_price_observed_order",
        "minimum_pde_delta_observed_order",
        "minimum_pde_gamma_observed_order",
        "pde_errors_monotone",
        "rannacher_oscillation_ratio",
        "ad_bump_delta_error",
        "far_boundary_price_drift",
        "snapshot_row_count",
        "computed_snapshot_hash",
        "expected_snapshot_hash",
        "monitoring",
        "production_fixture_eligible",
    }
    missing = required_keys.difference(evidence)
    if missing:
        raise ValueError(f"release evidence is missing: {sorted(missing)}")

    quotes_rejected = _nonnegative_integer(evidence, "quotes_rejected")
    implied_vol_failures = _nonnegative_integer(evidence, "implied_vol_failures")
    snapshot_row_count = _nonnegative_integer(evidence, "snapshot_row_count")
    calibration_rank = _nonnegative_integer(
        evidence,
        "calibration_numerical_rank",
    )
    calibration_parameter_count = _nonnegative_integer(
        evidence,
        "calibration_parameter_count",
    )
    calibration_nullity = _nonnegative_integer(evidence, "calibration_nullity")
    successful_restarts = _nonnegative_integer(
        evidence,
        "calibration_successful_restarts",
    )
    failed_restarts = _nonnegative_integer(
        evidence,
        "calibration_failed_restarts",
    )
    active_bounds = _nonnegative_integer(evidence, "calibration_active_bounds")
    numeric = {
        key: _finite_number(evidence, key)
        for key in (
            "weighted_calibration_rmse",
            "calibration_restart_dispersion",
            "calibration_heldout_rmse",
            "calibration_regularization_fraction",
            "smallest_scaled_jacobian_singular_value",
            "finest_pde_price_error",
            "finest_pde_delta_error",
            "finest_pde_gamma_error",
            "minimum_pde_price_observed_order",
            "minimum_pde_delta_observed_order",
            "minimum_pde_gamma_observed_order",
            "rannacher_oscillation_ratio",
            "ad_bump_delta_error",
            "far_boundary_price_drift",
        )
    }
    condition_value = evidence["scaled_jacobian_condition"]
    if (
        isinstance(condition_value, bool)
        or not isinstance(condition_value, (int, float))
        or math.isnan(float(condition_value))
        or float(condition_value) < 0.0
    ):
        raise ValueError("scaled_jacobian_condition must be nonnegative or infinity")
    condition = float(condition_value)
    condition_display: float | str = (
        "Infinity" if math.isinf(condition) else condition
    )
    nonnegative_keys = (
        "weighted_calibration_rmse",
        "calibration_restart_dispersion",
        "calibration_heldout_rmse",
        "calibration_regularization_fraction",
        "smallest_scaled_jacobian_singular_value",
        "finest_pde_price_error",
        "finest_pde_delta_error",
        "finest_pde_gamma_error",
        "rannacher_oscillation_ratio",
        "ad_bump_delta_error",
        "far_boundary_price_drift",
    )
    if any(numeric[key] < 0.0 for key in nonnegative_keys):
        raise ValueError("release error and conditioning evidence must be nonnegative")
    calibration_status = evidence["calibration_status"]
    monitoring = evidence["monitoring"]
    pde_errors_monotone = evidence["pde_errors_monotone"]
    production_eligible = evidence["production_fixture_eligible"]
    if (
        not isinstance(calibration_status, str)
        or not isinstance(monitoring, str)
        or not isinstance(pde_errors_monotone, bool)
        or not isinstance(production_eligible, bool)
    ):
        raise ValueError("release status, convention and Boolean evidence are required")
    computed_hash = evidence["computed_snapshot_hash"]
    expected_hash = evidence["expected_snapshot_hash"]
    if not _valid_sha256(computed_hash) or not _valid_sha256(expected_hash):
        raise ValueError("lowercase 64-character SHA-256 evidence required")

    gates = (
        _release_gate(
            "QUOTE_REJECTIONS",
            quotes_rejected,
            "quotes",
            "==",
            governed_policy.expected_rejected_quotes,
            "Market Data Owner",
            True,
            quotes_rejected == governed_policy.expected_rejected_quotes,
            "QUOTE_REJECTION_COUNT_MISMATCH",
        ),
        _release_gate(
            "IMPLIED_VOL_ROOT_FAILURES",
            implied_vol_failures,
            "failures",
            "<=",
            governed_policy.maximum_implied_vol_failures,
            "Numerical Methods Owner",
            True,
            implied_vol_failures <= governed_policy.maximum_implied_vol_failures,
            "IMPLIED_VOL_FAILURE_LIMIT_EXCEEDED",
        ),
        _release_gate(
            "CALIBRATION_STATUS",
            calibration_status,
            "status",
            "==",
            "PASS",
            "Calibration Owner",
            True,
            calibration_status == "PASS",
            "CALIBRATION_STATUS_NOT_PASS",
        ),
        _release_gate(
            "CALIBRATION_WEIGHTED_RMSE",
            numeric["weighted_calibration_rmse"],
            "standardized residual",
            "<=",
            governed_policy.maximum_calibration_rmse,
            "Calibration Owner",
            True,
            numeric["weighted_calibration_rmse"]
            <= governed_policy.maximum_calibration_rmse,
            "CALIBRATION_RMSE_LIMIT_EXCEEDED",
        ),
        _release_gate(
            "CALIBRATION_PARAMETER_COUNT",
            calibration_parameter_count,
            "parameters",
            "==",
            governed_policy.expected_calibration_parameter_count,
            "Calibration Owner",
            True,
            (
                calibration_parameter_count
                == governed_policy.expected_calibration_parameter_count
            ),
            "CALIBRATION_PARAMETER_COUNT_MISMATCH",
        ),
        _release_gate(
            "CALIBRATION_NUMERICAL_RANK",
            calibration_rank,
            "parameter directions",
            "==",
            calibration_parameter_count,
            "Calibration Owner",
            True,
            calibration_rank == calibration_parameter_count,
            "CALIBRATION_NOT_IDENTIFIED",
        ),
        _release_gate(
            "CALIBRATION_NULLITY",
            calibration_nullity,
            "parameter directions",
            "==",
            0,
            "Calibration Owner",
            True,
            calibration_nullity == 0,
            "CALIBRATION_NULLITY_NONZERO",
        ),
        _release_gate(
            "CALIBRATION_SUCCESSFUL_RESTARTS",
            successful_restarts,
            "restarts",
            ">=",
            governed_policy.minimum_successful_restarts,
            "Calibration Owner",
            True,
            successful_restarts >= governed_policy.minimum_successful_restarts,
            "INSUFFICIENT_SUCCESSFUL_RESTARTS",
        ),
        _release_gate(
            "CALIBRATION_FAILED_RESTARTS",
            failed_restarts,
            "restarts",
            "<=",
            governed_policy.maximum_failed_restarts,
            "Calibration Owner",
            True,
            failed_restarts <= governed_policy.maximum_failed_restarts,
            "FAILED_CALIBRATION_RESTART_PRESENT",
        ),
        _release_gate(
            "CALIBRATION_RESTART_DISPERSION",
            numeric["calibration_restart_dispersion"],
            "scaled parameter norm",
            "<=",
            governed_policy.maximum_restart_dispersion,
            "Calibration Owner",
            True,
            numeric["calibration_restart_dispersion"]
            <= governed_policy.maximum_restart_dispersion,
            "RESTART_DISPERSION_LIMIT_EXCEEDED",
        ),
        _release_gate(
            "CALIBRATION_HELDOUT_RMSE",
            numeric["calibration_heldout_rmse"],
            "volatility",
            "<=",
            governed_policy.maximum_heldout_rmse,
            "Model Validation Owner",
            True,
            numeric["calibration_heldout_rmse"]
            <= governed_policy.maximum_heldout_rmse,
            "HELDOUT_RMSE_LIMIT_EXCEEDED",
        ),
        _release_gate(
            "CALIBRATION_ACTIVE_BOUNDS",
            active_bounds,
            "parameters",
            "<=",
            governed_policy.maximum_active_bounds,
            "Calibration Owner",
            True,
            active_bounds <= governed_policy.maximum_active_bounds,
            "ACTIVE_BOUND_LIMIT_EXCEEDED",
        ),
        _release_gate(
            "CALIBRATION_REGULARIZATION_FRACTION",
            numeric["calibration_regularization_fraction"],
            "objective fraction",
            "<=",
            governed_policy.maximum_regularization_fraction,
            "Model Validation Owner",
            True,
            numeric["calibration_regularization_fraction"]
            <= governed_policy.maximum_regularization_fraction,
            "REGULARIZATION_DOMINANCE_LIMIT_EXCEEDED",
        ),
        _release_gate(
            "SCALED_JACOBIAN_CONDITION",
            condition_display,
            "ratio",
            "<=",
            governed_policy.maximum_jacobian_condition,
            "Model Validation Owner",
            True,
            math.isfinite(condition)
            and condition <= governed_policy.maximum_jacobian_condition,
            "CONDITION_LIMIT_EXCEEDED",
        ),
        _release_gate(
            "SMALLEST_SCALED_JACOBIAN_SINGULAR_VALUE",
            numeric["smallest_scaled_jacobian_singular_value"],
            "scaled singular-value units",
            ">=",
            governed_policy.minimum_jacobian_singular_value,
            "Model Validation Owner",
            True,
            numeric["smallest_scaled_jacobian_singular_value"]
            >= governed_policy.minimum_jacobian_singular_value,
            "SINGULAR_VALUE_FLOOR_BREACHED",
        ),
        _release_gate(
            "PDE_PRICE_REFERENCE_ERROR",
            numeric["finest_pde_price_error"],
            "price units",
            "<=",
            governed_policy.maximum_pde_price_error,
            "Numerical Methods Owner",
            True,
            numeric["finest_pde_price_error"]
            <= governed_policy.maximum_pde_price_error,
            "PDE_PRICE_ERROR_LIMIT_EXCEEDED",
        ),
        _release_gate(
            "PDE_DELTA_REFERENCE_ERROR",
            numeric["finest_pde_delta_error"],
            "delta",
            "<=",
            governed_policy.maximum_pde_delta_error,
            "Numerical Methods Owner",
            True,
            numeric["finest_pde_delta_error"]
            <= governed_policy.maximum_pde_delta_error,
            "PDE_DELTA_ERROR_LIMIT_EXCEEDED",
        ),
        _release_gate(
            "PDE_GAMMA_REFERENCE_ERROR",
            numeric["finest_pde_gamma_error"],
            "gamma",
            "<=",
            governed_policy.maximum_pde_gamma_error,
            "Numerical Methods Owner",
            True,
            numeric["finest_pde_gamma_error"]
            <= governed_policy.maximum_pde_gamma_error,
            "PDE_GAMMA_ERROR_LIMIT_EXCEEDED",
        ),
        _release_gate(
            "PDE_PRICE_OBSERVED_ORDER",
            numeric["minimum_pde_price_observed_order"],
            "order",
            ">=",
            governed_policy.minimum_pde_observed_order,
            "Numerical Methods Owner",
            True,
            numeric["minimum_pde_price_observed_order"]
            >= governed_policy.minimum_pde_observed_order,
            "PDE_PRICE_ORDER_FLOOR_BREACHED",
        ),
        _release_gate(
            "PDE_DELTA_OBSERVED_ORDER",
            numeric["minimum_pde_delta_observed_order"],
            "order",
            ">=",
            governed_policy.minimum_pde_observed_order,
            "Numerical Methods Owner",
            True,
            numeric["minimum_pde_delta_observed_order"]
            >= governed_policy.minimum_pde_observed_order,
            "PDE_DELTA_ORDER_FLOOR_BREACHED",
        ),
        _release_gate(
            "PDE_GAMMA_OBSERVED_ORDER",
            numeric["minimum_pde_gamma_observed_order"],
            "order",
            ">=",
            governed_policy.minimum_pde_observed_order,
            "Numerical Methods Owner",
            True,
            numeric["minimum_pde_gamma_observed_order"]
            >= governed_policy.minimum_pde_observed_order,
            "PDE_GAMMA_ORDER_FLOOR_BREACHED",
        ),
        _release_gate(
            "PDE_ERROR_MONOTONICITY",
            pde_errors_monotone,
            "boolean",
            "==",
            True,
            "Numerical Methods Owner",
            True,
            pde_errors_monotone,
            "PDE_ERRORS_NOT_MONOTONE",
        ),
        _release_gate(
            "RANNACHER_OSCILLATION_RATIO",
            numeric["rannacher_oscillation_ratio"],
            "ratio",
            "<=",
            governed_policy.maximum_rannacher_oscillation_ratio,
            "Numerical Methods Owner",
            True,
            numeric["rannacher_oscillation_ratio"]
            <= governed_policy.maximum_rannacher_oscillation_ratio,
            "RANNACHER_EVIDENCE_LIMIT_EXCEEDED",
        ),
        _release_gate(
            "AD_BUMP_DELTA_ERROR",
            numeric["ad_bump_delta_error"],
            "delta",
            "<=",
            governed_policy.maximum_ad_bump_delta_error,
            "Model Validation Owner",
            True,
            numeric["ad_bump_delta_error"]
            <= governed_policy.maximum_ad_bump_delta_error,
            "AD_GREEK_ERROR_LIMIT_EXCEEDED",
        ),
        _release_gate(
            "FAR_BOUNDARY_PRICE_DRIFT",
            numeric["far_boundary_price_drift"],
            "price units",
            "<=",
            governed_policy.maximum_far_boundary_price_drift,
            "Numerical Methods Owner",
            True,
            numeric["far_boundary_price_drift"]
            <= governed_policy.maximum_far_boundary_price_drift,
            "FAR_BOUNDARY_DRIFT_LIMIT_EXCEEDED",
        ),
        _release_gate(
            "SNAPSHOT_LINEAGE_ROWS",
            snapshot_row_count,
            "rows",
            ">=",
            governed_policy.minimum_lineage_rows,
            "Data Governance Owner",
            True,
            snapshot_row_count >= governed_policy.minimum_lineage_rows,
            "INSUFFICIENT_LINEAGE_ROWS",
        ),
        _release_gate(
            "SNAPSHOT_HASH_MATCH",
            computed_hash == expected_hash,
            "boolean",
            "==",
            True,
            "Data Governance Owner",
            True,
            computed_hash == expected_hash,
            "SNAPSHOT_HASH_MISMATCH",
        ),
        _release_gate(
            "BARRIER_MONITORING_CONVENTION",
            monitoring,
            "convention",
            "==",
            governed_policy.required_monitoring,
            "Product Control Owner",
            True,
            monitoring == governed_policy.required_monitoring,
            "MONITORING_CONVENTION_MISMATCH",
        ),
        _release_gate(
            "PRODUCTION_FIXTURE_ELIGIBILITY",
            production_eligible,
            "boolean",
            "==",
            True,
            "Model Risk Owner",
            False,
            production_eligible,
            "SYNTHETIC_FIXTURE_ONLY",
        ),
    )
    failed_required = tuple(
        gate.reason for gate in gates if gate.required and not gate.passed
    )
    failed_advisory = tuple(
        gate.reason for gate in gates if not gate.required and not gate.passed
    )
    decision = (
        "REJECT"
        if failed_required
        else "CONDITIONAL PASS"
        if failed_advisory
        else "PASS"
    )
    return ReleaseGateEvaluation(
        policy_version=governed_policy.version,
        gates=gates,
        decision=decision,
        reasons=failed_required + failed_advisory,
    )


def _case_snapshot_rows(
    strikes: Sequence[float],
    maturities: Sequence[float],
    prices: Sequence[float],
) -> list[dict[str, object]]:
    expiries = {
        0.25: "2026-09-18",
        0.5: "2026-12-18",
        0.75: "2027-03-19",
        1.0: "2027-06-18",
        1.5: "2027-12-17",
        2.0: "2028-06-16",
    }
    rows: list[dict[str, object]] = []
    for index, (strike, maturity, price) in enumerate(
        zip(strikes, maturities, prices, strict=True)
    ):
        canonical_price = canonical_decimal(price)
        row: dict[str, object] = {
            "instrument_id": f"SYNTH-CH03-C-{index:02d}",
            "expiry": expiries[float(maturity)],
            "strike": canonical_decimal(strike),
            "option_type": "CALL",
            "quote_convention": "PRICE_USD",
            "bid": canonical_price,
            "ask": canonical_price,
            "valid_from": "2026-06-01T00:00:00.000000Z",
            "valid_to": "2026-07-01T00:00:00.000000Z",
            "known_at": "2026-06-01T08:00:00.000000Z",
            "superseded_at": None,
            "revision_id": 1,
            "quote_payload_hash": quote_payload_hash(
                canonical_price,
                canonical_price,
            ),
        }
        rows.append(row)
    return rows


def governed_numerical_case(
    *,
    snapshot_hash_from_query: str | None = None,
    release_policy: ReleaseGatePolicy | None = None,
) -> NumericalCaseStudy:
    """Execute a deterministic quote-to-pricing decision record."""

    spot = 100.0
    rate = 0.03
    dividend = 0.01
    strikes = np.tile(np.array([80.0, 90.0, 100.0, 110.0]), 6)
    maturities = np.repeat(np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0]), 4)
    coordinates = np.column_stack(
        (
            (strikes / spot) - 1.0,
            maturities - 1.0,
        )
    )
    true_parameters = np.array([0.22, -0.04, 0.015])
    true_volatilities = (
        true_parameters[0]
        + true_parameters[1] * coordinates[:, 0]
        + true_parameters[2] * coordinates[:, 1]
    )
    raw_prices = np.array(
        [
            call_price(
                spot,
                strike,
                rate,
                dividend,
                volatility,
                maturity,
            )
            for strike, volatility, maturity in zip(
                strikes,
                true_volatilities,
                maturities,
                strict=True,
            )
        ]
    )
    upper_bounds = spot * np.exp(-dividend * maturities)
    raw_prices[0] = -1.0
    raw_prices[-1] = upper_bounds[-1] + 1.0
    snapshot_rows = _case_snapshot_rows(strikes, maturities, raw_prices)
    prices = np.array([float(row["bid"]) for row in snapshot_rows])
    snapshot_hash = canonical_snapshot_hash(snapshot_rows)
    expected_snapshot_hash = (
        CASE_SNAPSHOT_GOLDEN_SHA256
        if snapshot_hash_from_query is None
        else snapshot_hash_from_query
    )

    accepted_coordinates: list[np.ndarray] = []
    recovered_volatilities: list[float] = []
    rejected = 0
    root_failures = 0
    for coordinate, price, strike, maturity in zip(
        coordinates,
        prices,
        strikes,
        maturities,
        strict=True,
    ):
        result = implied_vol(
            float(price),
            spot,
            float(strike),
            rate,
            dividend,
            float(maturity),
        )
        if result.status == "INVALID_QUOTE":
            rejected += 1
            continue
        if result.status != "CONVERGED" or result.volatility is None:
            root_failures += 1
            continue
        accepted_coordinates.append(coordinate)
        recovered_volatilities.append(result.volatility)

    design = np.column_stack(
        (
            np.ones(len(accepted_coordinates)),
            np.asarray(accepted_coordinates),
        )
    )
    observations = np.asarray(recovered_volatilities)
    heldout_coordinates = np.array(
        [
            [-0.05, 0.25],
            [0.05, 0.75],
        ]
    )
    heldout_design = np.column_stack(
        (np.ones(heldout_coordinates.shape[0]), heldout_coordinates)
    )
    heldout_observations = heldout_design @ true_parameters
    calibration = calibrate(
        [0.20, 0.0, 0.0],
        lambda parameters: design @ parameters,
        observations,
        np.full(observations.size, 1.0 / 0.002**2),
        bounds=([0.05, -0.50, -0.50], [1.00, 0.50, 0.50]),
        parameter_scale=[0.20, 0.10, 0.10],
        restart_points=([0.30, -0.10, 0.05], [0.15, 0.05, -0.02]),
        heldout_model=lambda parameters: heldout_design @ parameters,
        heldout_observations=heldout_observations,
        policy=CalibrationPolicy(
            weighted_rmse_limit=1e-6,
            heldout_rmse_limit=1e-6,
            minimum_successful_restarts=3,
            require_heldout=True,
        ),
    )

    pde_rows = barrier_pde_refinement(
        spot,
        100.0,
        85.0,
        rate,
        dividend,
        0.22,
        1.0,
        s_max=300.0,
        space_steps=[86, 172, 344],
    )
    far_boundary_rows = barrier_far_boundary_study(
        spot,
        100.0,
        85.0,
        rate,
        dividend,
        0.22,
        1.0,
        s_max_values=[250.0, 320.0, 400.0],
        spatial_spacing=1.0,
        time_steps=500,
    )
    far_boundary_prices = [float(row["price"]) for row in far_boundary_rows]
    far_boundary_drift = max(far_boundary_prices) - min(far_boundary_prices)
    grid = np.linspace(0.0, 1.0, 101)
    discontinuous = (grid >= 0.5).astype(float)
    discontinuous[0], discontinuous[-1] = 0.0, 1.0
    smoothing = rannacher_comparison(
        discontinuous,
        0.1,
        0.01,
        0.01,
        5,
    )
    greeks = black_scholes_greek_crosscheck(
        spot,
        105.0,
        rate,
        dividend,
        0.22,
        1.4,
    )

    error_keys = (
        "absolute_price_error",
        "absolute_delta_error",
        "absolute_gamma_error",
    )
    pde_errors_monotone = all(
        all(
            float(current[key]) < float(previous[key])
            for previous, current in zip(pde_rows, pde_rows[1:], strict=False)
        )
        for key in error_keys
    )
    minimum_orders = {
        quantity: min(
            float(row[f"{quantity}_observed_order"])
            for row in pde_rows[1:]
            if row[f"{quantity}_observed_order"] is not None
        )
        for quantity in ("price", "delta", "gamma")
    }
    cn_oscillation = float(smoothing["cn_curvature_oscillation"])
    rannacher_oscillation = float(
        smoothing["rannacher_curvature_oscillation"]
    )
    rannacher_ratio = rannacher_oscillation / cn_oscillation
    ad_bump_delta_error = abs(greeks["ad_delta"] - greeks["bump_delta"])
    finest = pde_rows[-1]
    gate_evidence: dict[str, object] = {
        "quotes_rejected": rejected,
        "implied_vol_failures": root_failures,
        "calibration_status": calibration.status,
        "weighted_calibration_rmse": calibration.weighted_rmse,
        "calibration_numerical_rank": calibration.numerical_rank,
        "calibration_parameter_count": calibration.parameter_count,
        "calibration_nullity": calibration.nullity,
        "calibration_successful_restarts": calibration.successful_restarts,
        "calibration_failed_restarts": (
            len(calibration.restart_statuses) - calibration.successful_restarts
        ),
        "calibration_restart_dispersion": calibration.restart_dispersion,
        "calibration_heldout_rmse": calibration.heldout_rmse,
        "calibration_active_bounds": calibration.active_bounds,
        "calibration_regularization_fraction": (
            calibration.regularization_fraction
        ),
        "scaled_jacobian_condition": calibration.scaled_jacobian_condition,
        "smallest_scaled_jacobian_singular_value": float(
            calibration.singular_values[-1]
        ),
        "finest_pde_price_error": float(finest["absolute_price_error"]),
        "finest_pde_delta_error": float(finest["absolute_delta_error"]),
        "finest_pde_gamma_error": float(finest["absolute_gamma_error"]),
        "minimum_pde_price_observed_order": minimum_orders["price"],
        "minimum_pde_delta_observed_order": minimum_orders["delta"],
        "minimum_pde_gamma_observed_order": minimum_orders["gamma"],
        "pde_errors_monotone": pde_errors_monotone,
        "rannacher_oscillation_ratio": rannacher_ratio,
        "ad_bump_delta_error": ad_bump_delta_error,
        "far_boundary_price_drift": far_boundary_drift,
        "snapshot_row_count": len(snapshot_rows),
        "computed_snapshot_hash": snapshot_hash,
        "expected_snapshot_hash": expected_snapshot_hash,
        "monitoring": "continuous",
        "production_fixture_eligible": False,
    }
    evaluation = evaluate_release_gates(gate_evidence, release_policy)

    return NumericalCaseStudy(
        fixture_type=(
            "Synthetic governed fixture designed to exercise "
            "the complete release path"
        ),
        fixture_version="ch03-case-v0.11",
        snapshot_hash=snapshot_hash,
        expected_snapshot_hash=expected_snapshot_hash,
        snapshot_hash_matches_artifact=snapshot_hash == expected_snapshot_hash,
        snapshot_row_count=len(snapshot_rows),
        quotes_received=prices.size,
        quotes_rejected=rejected,
        implied_vol_failures=root_failures,
        weighted_calibration_rmse=calibration.weighted_rmse,
        calibration_numerical_rank=calibration.numerical_rank,
        calibration_parameter_count=calibration.parameter_count,
        calibration_nullity=calibration.nullity,
        calibration_successful_restarts=calibration.successful_restarts,
        calibration_failed_restarts=(
            len(calibration.restart_statuses) - calibration.successful_restarts
        ),
        calibration_restart_dispersion=calibration.restart_dispersion,
        calibration_heldout_rmse=float(calibration.heldout_rmse),
        calibration_active_bounds=calibration.active_bounds,
        calibration_regularization_fraction=calibration.regularization_fraction,
        smallest_scaled_jacobian_singular_value=float(
            calibration.singular_values[-1]
        ),
        scaled_jacobian_condition=calibration.scaled_jacobian_condition,
        pde_grid_prices=tuple(float(row["price"]) for row in pde_rows),
        pde_grid_evidence=tuple(
            (
                int(row["space_steps"]),
                int(row["time_steps"]),
                float(row["price"]),
                float(row["absolute_price_error"]),
                float(row["absolute_delta_error"]),
                float(row["absolute_gamma_error"]),
            )
            for row in pde_rows
        ),
        finest_pde_reference_error=float(finest["absolute_price_error"]),
        finest_pde_delta_error=float(finest["absolute_delta_error"]),
        finest_pde_gamma_error=float(finest["absolute_gamma_error"]),
        minimum_pde_price_observed_order=minimum_orders["price"],
        minimum_pde_delta_observed_order=minimum_orders["delta"],
        minimum_pde_gamma_observed_order=minimum_orders["gamma"],
        pde_errors_monotone=pde_errors_monotone,
        far_boundary_price_drift=far_boundary_drift,
        cn_curvature_oscillation=cn_oscillation,
        rannacher_curvature_oscillation=rannacher_oscillation,
        rannacher_oscillation_ratio=rannacher_ratio,
        greek_instrument=(
            "Black-Scholes vanilla call; independent real central bump"
        ),
        ad_bump_delta_error=ad_bump_delta_error,
        release_policy_version=evaluation.policy_version,
        release_gates=evaluation.gates,
        decision_reasons=evaluation.reasons,
        decision=evaluation.decision,
    )
