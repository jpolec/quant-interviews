"""Regression and mutation tests for the CH03 audit findings."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import math

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import quant_interviews.ch03.barrier_pde as barrier_module
import quant_interviews.ch03.calibration as calibration_module
from quant_interviews.ch03.autodiff import (
    Dual,
    black_scholes_call_dual,
    black_scholes_greek_crosscheck,
    dual_exp,
)
from quant_interviews.ch03.barrier_pde import (
    barrier_terminal_grid,
    barrier_pde_refinement,
    down_and_out_call_pde,
    enforce_absorbing_boundary,
)
from quant_interviews.ch03.calibration import (
    CalibrationPolicy,
    calibrate,
    solve_calibration_restarts,
)
from quant_interviews.ch03.case_study import (
    CASE_SNAPSHOT_GOLDEN_SHA256,
    ReleaseGatePolicy,
    evaluate_release_gates,
    governed_numerical_case,
)
from quant_interviews.ch03.implied_vol import call_price, implied_vol
from quant_interviews.ch03.interpolation import call_price_arbitrage_audit
from quant_interviews.ch03.lineage import (
    canonical_decimal,
    canonical_snapshot_hash,
    canonical_snapshot_row,
    quote_payload_hash,
)
from quant_interviews.ch03.mean_reversion import (
    euler_mean_reversion,
    linear_mean_reversion,
    mean_reversion_convergence,
)
from quant_interviews.ch03.quadrature import (
    composite_trapezoid,
    quadrature_convergence_table,
    singularity_transform_integral,
)


def _passing_release_evidence() -> dict[str, object]:
    return {
        "quotes_rejected": 2,
        "implied_vol_failures": 0,
        "calibration_status": "PASS",
        "weighted_calibration_rmse": 5e-8,
        "calibration_numerical_rank": 3,
        "calibration_parameter_count": 3,
        "calibration_nullity": 0,
        "calibration_successful_restarts": 3,
        "calibration_failed_restarts": 0,
        "calibration_restart_dispersion": 1e-12,
        "calibration_heldout_rmse": 1e-10,
        "calibration_active_bounds": 0,
        "calibration_regularization_fraction": 0.0,
        "scaled_jacobian_condition": 20.0,
        "smallest_scaled_jacobian_singular_value": 25.0,
        "finest_pde_price_error": 1e-3,
        "finest_pde_delta_error": 1e-5,
        "finest_pde_gamma_error": 1e-6,
        "minimum_pde_price_observed_order": 2.0,
        "minimum_pde_delta_observed_order": 2.0,
        "minimum_pde_gamma_observed_order": 2.0,
        "pde_errors_monotone": True,
        "rannacher_oscillation_ratio": 0.01,
        "ad_bump_delta_error": 1e-11,
        "far_boundary_price_drift": 1e-12,
        "snapshot_row_count": 24,
        "computed_snapshot_hash": "1" * 64,
        "expected_snapshot_hash": "1" * 64,
        "monitoring": "continuous",
        "production_fixture_eligible": False,
    }


@pytest.mark.parametrize(
    ("field", "mutation", "reason"),
    [
        ("quotes_rejected", 3, "QUOTE_REJECTION_COUNT_MISMATCH"),
        ("implied_vol_failures", 1, "IMPLIED_VOL_FAILURE_LIMIT_EXCEEDED"),
        ("calibration_status", "NOT_IDENTIFIED", "CALIBRATION_STATUS_NOT_PASS"),
        ("weighted_calibration_rmse", 2e-7, "CALIBRATION_RMSE_LIMIT_EXCEEDED"),
        (
            "calibration_parameter_count",
            4,
            "CALIBRATION_PARAMETER_COUNT_MISMATCH",
        ),
        ("calibration_numerical_rank", 2, "CALIBRATION_NOT_IDENTIFIED"),
        ("calibration_nullity", 1, "CALIBRATION_NULLITY_NONZERO"),
        ("calibration_successful_restarts", 2, "INSUFFICIENT_SUCCESSFUL_RESTARTS"),
        ("calibration_failed_restarts", 1, "FAILED_CALIBRATION_RESTART_PRESENT"),
        ("calibration_restart_dispersion", 2e-6, "RESTART_DISPERSION_LIMIT_EXCEEDED"),
        ("calibration_heldout_rmse", 2e-6, "HELDOUT_RMSE_LIMIT_EXCEEDED"),
        ("calibration_active_bounds", 1, "ACTIVE_BOUND_LIMIT_EXCEEDED"),
        ("calibration_regularization_fraction", 0.5, "REGULARIZATION_DOMINANCE_LIMIT_EXCEEDED"),
        ("scaled_jacobian_condition", 101.0, "CONDITION_LIMIT_EXCEEDED"),
        (
            "smallest_scaled_jacobian_singular_value",
            19.0,
            "SINGULAR_VALUE_FLOOR_BREACHED",
        ),
        ("finest_pde_price_error", 0.03, "PDE_PRICE_ERROR_LIMIT_EXCEEDED"),
        ("finest_pde_delta_error", 3e-4, "PDE_DELTA_ERROR_LIMIT_EXCEEDED"),
        ("finest_pde_gamma_error", 3e-5, "PDE_GAMMA_ERROR_LIMIT_EXCEEDED"),
        (
            "minimum_pde_price_observed_order",
            1.7,
            "PDE_PRICE_ORDER_FLOOR_BREACHED",
        ),
        (
            "minimum_pde_delta_observed_order",
            1.7,
            "PDE_DELTA_ORDER_FLOOR_BREACHED",
        ),
        (
            "minimum_pde_gamma_observed_order",
            1.7,
            "PDE_GAMMA_ORDER_FLOOR_BREACHED",
        ),
        ("pde_errors_monotone", False, "PDE_ERRORS_NOT_MONOTONE"),
        (
            "rannacher_oscillation_ratio",
            0.2,
            "RANNACHER_EVIDENCE_LIMIT_EXCEEDED",
        ),
        ("ad_bump_delta_error", 2e-8, "AD_GREEK_ERROR_LIMIT_EXCEEDED"),
        (
            "far_boundary_price_drift",
            2e-8,
            "FAR_BOUNDARY_DRIFT_LIMIT_EXCEEDED",
        ),
        ("snapshot_row_count", 23, "INSUFFICIENT_LINEAGE_ROWS"),
        ("expected_snapshot_hash", "0" * 64, "SNAPSHOT_HASH_MISMATCH"),
        ("monitoring", "discrete", "MONITORING_CONVENTION_MISMATCH"),
    ],
)
def test_evaluate_release_gates_rejects_every_required_gate_mutation(
    field: str,
    mutation: object,
    reason: str,
) -> None:
    evidence = _passing_release_evidence()
    evidence[field] = mutation
    policy = ReleaseGatePolicy()

    evaluation = evaluate_release_gates(evidence, policy)

    assert evaluation.decision == "REJECT"
    assert reason in evaluation.reasons


def test_release_evaluator_normalizes_infinite_condition_to_reject() -> None:
    evidence = _passing_release_evidence()
    evidence["calibration_status"] = "NOT_IDENTIFIED"
    evidence["calibration_numerical_rank"] = 2
    evidence["calibration_nullity"] = 1
    evidence["scaled_jacobian_condition"] = float("inf")

    result = evaluate_release_gates(evidence)

    condition_gate = next(
        gate for gate in result.gates if gate.name == "SCALED_JACOBIAN_CONDITION"
    )
    assert result.decision == "REJECT"
    assert condition_gate.value == "Infinity"
    assert condition_gate.reason == "CONDITION_LIMIT_EXCEEDED"


def test_governed_case_hash_and_all_required_gates_are_executable() -> None:
    case = governed_numerical_case()

    assert case.snapshot_hash == CASE_SNAPSHOT_GOLDEN_SHA256
    assert case.snapshot_hash_matches_artifact
    assert case.snapshot_row_count == 24
    assert all(gate.passed for gate in case.release_gates if gate.required)
    assert case.decision == "CONDITIONAL PASS"
    assert case.decision_reasons == ("SYNTHETIC_FIXTURE_ONLY",)

    mismatch = governed_numerical_case(snapshot_hash_from_query="0" * 64)
    assert mismatch.decision == "REJECT"
    assert "SNAPSHOT_HASH_MISMATCH" in mismatch.decision_reasons


def test_calibration_rejects_underdetermined_model_as_not_identified() -> None:
    record = calibrate(
        [0.0, 0.0, 0.0],
        lambda parameters: parameters[:2],
        [1.0, 1.0],
        [1.0, 1.0],
    )

    assert record.observation_count == 2
    assert record.parameter_count == 3
    assert record.numerical_rank == 2
    assert record.nullity == 1
    assert not record.is_locally_identified
    assert math.isinf(record.scaled_jacobian_condition)
    assert record.status == "NOT_IDENTIFIED"
    assert "NOT_IDENTIFIED" in record.status_reasons


def test_calibration_validates_heldout_and_selects_successful_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="held-out"):
        calibrate(
            [0.0],
            lambda parameters: parameters,
            [1.0],
            [1.0],
            heldout_model=lambda _parameters: [float("nan")],
            heldout_observations=[1.0],
        )

    def fake_least_squares(
        _objective: object,
        start: np.ndarray,
        **_kwargs: object,
    ) -> OptimizeResult:
        successful = bool(start[0] > 0.0)
        return OptimizeResult(
            x=np.asarray(start, dtype=float),
            fun=np.array([0.0 if successful else 0.001]),
            cost=1.0 if successful else 0.0,
            success=successful,
            active_mask=np.zeros_like(start, dtype=int),
            jac=np.eye(start.size),
        )

    monkeypatch.setattr(calibration_module, "least_squares", fake_least_squares)
    best, results = solve_calibration_restarts(
        lambda parameters: parameters,
        [np.array([0.0]), np.array([1.0])],
        ([-2.0], [2.0]),
        np.array([1.0]),
    )

    assert len(results) == 2
    assert best.success
    assert best.x == pytest.approx([1.0])
    partial = calibrate(
        [0.0],
        lambda parameters: parameters,
        [1.0],
        [1.0],
        restart_points=([1.0],),
    )
    assert partial.restart_statuses == ("FAILED", "SUCCESS")
    assert partial.successful_restarts == 1
    assert partial.status == "REJECT"
    assert "FAILED_RESTART_PRESENT" in partial.status_reasons

    def all_successful(
        _objective: object,
        start: np.ndarray,
        **_kwargs: object,
    ) -> OptimizeResult:
        return OptimizeResult(
            x=np.asarray(start, dtype=float),
            fun=np.asarray(start, dtype=float),
            cost=0.5 * float(np.dot(start, start)),
            success=True,
            active_mask=np.zeros_like(start, dtype=int),
            jac=np.eye(start.size),
        )

    monkeypatch.setattr(calibration_module, "least_squares", all_successful)
    record = calibrate(
        [0.0],
        lambda parameters: parameters,
        [0.0],
        [1.0],
        parameter_scale=[2.0],
        restart_points=([1.0],),
        policy=CalibrationPolicy(restart_dispersion_limit=1.0),
    )
    assert record.restart_dispersion == pytest.approx(0.5)


@pytest.mark.parametrize(
    "arguments",
    [
        (100.0, 100.0, 0.03, 0.01, -0.2, 1.0),
        (100.0, 100.0, 0.03, 0.01, 0.0, 1.0),
        (100.0, 0.0, 0.03, 0.01, 0.2, 1.0),
        (100.0, 100.0, float("nan"), 0.01, 0.2, 1.0),
        (100.0, 100.0, 0.03, 0.01, 0.2, 0.0),
    ],
)
def test_black_scholes_crosscheck_rejects_invalid_economic_domain(
    arguments: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        black_scholes_greek_crosscheck(*arguments)

    with pytest.raises(ValueError, match="bump"):
        black_scholes_greek_crosscheck(
            1.0,
            1.0,
            0.0,
            0.0,
            0.2,
            1.0,
            spot_bump=1.0,
        )


def test_dual_exp_and_dual_black_scholes_are_directly_exercised() -> None:
    exponential = dual_exp(Dual(0.7, 2.0))
    call = black_scholes_call_dual(
        Dual(100.0, 1.0),
        100.0,
        0.03,
        0.01,
        Dual(0.2, 0.0),
        1.0,
    )

    assert exponential.value == pytest.approx(math.exp(0.7))
    assert exponential.derivative == pytest.approx(2.0 * math.exp(0.7))
    assert call.value > 0.0
    assert 0.0 < call.derivative < 1.0


@pytest.mark.parametrize(
    ("control", "bad_value"),
    [
        ("price_tolerance", float("nan")),
        ("volatility_tolerance", float("inf")),
        ("price_relative_tolerance", float("nan")),
        ("volatility_relative_tolerance", float("inf")),
        ("vega_floor", float("nan")),
        ("maximum_volatility", float("nan")),
    ],
)
def test_implied_vol_validates_all_controls_and_uses_finite_status_payloads(
    control: str,
    bad_value: float,
) -> None:
    price = call_price(100.0, 100.0, 0.0, 0.0, 0.2, 1.0)
    with pytest.raises(ValueError, match="solver controls"):
        implied_vol(
            price,
            100.0,
            100.0,
            0.0,
            0.0,
            1.0,
            **{control: bad_value},
        )

    invalid = implied_vol(-1.0, 100.0, 100.0, 0.0, 0.0, 1.0)
    high_price = call_price(100.0, 120.0, 0.0, 0.0, 2.0, 1.0)
    no_bracket = implied_vol(
        high_price,
        100.0,
        120.0,
        0.0,
        0.0,
        1.0,
        maximum_volatility=0.5,
    )
    upper = implied_vol(100.0, 100.0, 100.0, 0.0, 0.0, 1.0)
    for result in (invalid, no_bracket, upper):
        assert result.volatility is None
        assert math.isfinite(result.price_residual)
        assert result.volatility_bracket_width is None or math.isfinite(
            result.volatility_bracket_width
        )


@pytest.mark.parametrize("spot", [1.0, 1_000_000.0])
def test_implied_vol_scaled_stops_hold_at_two_price_scales(spot: float) -> None:
    market_price = call_price(spot, spot, 0.01, 0.0, 0.35, 1.0)
    result = implied_vol(
        market_price,
        spot,
        spot,
        0.01,
        0.0,
        1.0,
        price_tolerance=1e-11,
        price_relative_tolerance=1e-10,
        volatility_tolerance=1e-11,
        volatility_relative_tolerance=1e-10,
    )

    assert result.status == "CONVERGED"
    assert result.price_residual <= 1e-11 + 1e-10 * max(1.0, market_price)
    assert result.volatility is not None
    assert result.volatility_bracket_width is not None
    assert result.volatility_bracket_width <= (
        1e-11 + 1e-10 * max(1.0, result.volatility)
    )


def test_call_arbitrage_audit_requires_finite_three_strike_evidence() -> None:
    with pytest.raises(ValueError, match="three strikes"):
        call_price_arbitrage_audit(
            [90.0, 110.0],
            [12.0, 2.0],
            forward=100.0,
            discount=0.98,
        )
    for kwargs in (
        {"forward": float("nan"), "discount": 0.98},
        {"forward": 100.0, "discount": float("nan")},
        {
            "forward": 100.0,
            "discount": 0.98,
            "price_tolerance": float("nan"),
        },
        {
            "forward": 100.0,
            "discount": 0.98,
            "slope_tolerance": float("nan"),
        },
        {
            "forward": 100.0,
            "discount": 0.98,
            "convexity_tolerance": float("nan"),
        },
    ):
        with pytest.raises(ValueError, match="finite positive"):
            call_price_arbitrage_audit(
                [90.0, 100.0, 110.0],
                [12.0, 7.0, 3.0],
                **kwargs,
            )


def test_quadrature_rejects_bad_bounds_shape_and_exact_value() -> None:
    for lower, upper in ((1.0, 0.0), (1.0, 1.0), (float("nan"), 1.0)):
        with pytest.raises(ValueError, match="ordered interval"):
            singularity_transform_integral(
                np.sqrt,
                lower,
                upper,
                20,
                transformed_endpoint_limit=0.0,
            )
    with pytest.raises(ValueError, match="one value"):
        singularity_transform_integral(
            lambda _values: 1.0,
            0.0,
            1.0,
            20,
            transformed_endpoint_limit=0.0,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        quadrature_convergence_table(
            composite_trapezoid,
            np.exp,
            0.0,
            1.0,
            float("nan"),
            [20, 40],
        )


def test_mean_reversion_mesh_ratio_order_and_half_life_states() -> None:
    rows = mean_reversion_convergence(
        0.0,
        2.0,
        1.0,
        0.4,
        1.0,
        [20, 30, 50],
    )
    neutral = mean_reversion_convergence(0.0, 0.0, 1.0, 0.4, 1.0, [1, 2])
    annihilated = mean_reversion_convergence(0.0, 2.0, 1.0, 0.0, 0.5, [1, 2])
    unstable = mean_reversion_convergence(0.0, 4.0, 1.0, 0.0, 1.0, [1, 2])

    assert rows[-1]["observed_order"] == pytest.approx(1.0, rel=0.02)
    assert rows[-1]["physical_half_life"] == pytest.approx(math.log(2.0) / 2.0)
    assert rows[-1]["effective_half_life"] != rows[-1]["physical_half_life"]
    assert neutral[0]["physical_half_life"] == float("inf")
    assert neutral[0]["effective_half_life"] == float("inf")
    assert annihilated[0]["effective_half_life"] == 0.0
    assert unstable[0]["effective_half_life"] is None

    for bad_counts in ([20, 20], [20, 10], [20, 30.0], [True, 20]):
        with pytest.raises(ValueError, match="increasing integer"):
            mean_reversion_convergence(0.0, 2.0, 1.0, 0.4, 1.0, bad_counts)
    with pytest.raises(ValueError, match="positive step"):
        euler_mean_reversion(0.0, float("nan"), 1.0, 0.0, 0.1, 2)


def test_linear_mean_reversion_is_directly_exercised() -> None:
    path = linear_mean_reversion(0.0, 2.0, 1.0, [0.0, 0.5, 1.0], 0.4)

    assert path.shape == (3,)
    assert np.all(np.diff(path) > 0.0)
    with pytest.raises(ValueError, match="increasing times"):
        linear_mean_reversion(0.0, 2.0, 1.0, [0.0, float("nan")], 0.4)


def test_barrier_contract_validates_controls_and_uses_absorbing_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = enforce_absorbing_boundary

    def counting_boundary(values: object, rebate: float = 0.0) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(values, rebate)

    monkeypatch.setattr(barrier_module, "enforce_absorbing_boundary", counting_boundary)
    result = down_and_out_call_pde(
        100.0,
        100.0,
        85.0,
        0.03,
        0.01,
        0.22,
        1.0,
        s_max=250.0,
        space_steps=80,
        time_steps=160,
        rebate=2.0,
        rebate_timing="maturity",
        monitoring="continuous",
    )

    assert calls > 1
    assert result.values[0] == pytest.approx(2.0 * math.exp(-0.03))
    assert np.isfinite(result.values).all()

    with pytest.raises(ValueError, match="finite grid"):
        barrier_terminal_grid(
            [85.0, 90.0, 100.0],
            strike=float("nan"),
            barrier=85.0,
        )
    with pytest.raises(ValueError, match="finite"):
        enforce_absorbing_boundary([0.0, float("nan"), 1.0])

    base = dict(
        spot=100.0,
        strike=100.0,
        barrier=85.0,
        rate=0.03,
        dividend=0.01,
        volatility=0.22,
        maturity=1.0,
        s_max=250.0,
        space_steps=80,
        time_steps=160,
    )
    for mutation in (
        {"rate": float("nan")},
        {"rebate": float("nan")},
        {"monitoring": "discrete"},
        {"rebate_timing": "settlement"},
        {"space_steps": 80.0},
        {"time_steps": True},
    ):
        request = {**base, **mutation}
        with pytest.raises(ValueError, match="continuous barrier contract"):
            down_and_out_call_pde(**request)


def test_barrier_refinement_uses_none_for_unavailable_order() -> None:
    rows = barrier_pde_refinement(
        100.0,
        100.0,
        85.0,
        0.03,
        0.01,
        0.22,
        1.0,
        s_max=300.0,
        space_steps=[86, 172],
    )

    assert rows[0]["price_observed_order"] is None
    assert rows[0]["delta_observed_order"] is None
    assert rows[0]["gamma_observed_order"] is None


def _lineage_row(strike: object, revision: int) -> dict[str, object]:
    bid = canonical_decimal(4.2)
    ask = canonical_decimal(4.4)
    return {
        "instrument_id": "SPX-C",
        "expiry": "2026-09-18",
        "strike": strike,
        "option_type": "CALL",
        "quote_convention": "PRICE_USD",
        "bid": bid,
        "ask": ask,
        "valid_from": "2026-06-01T00:00:00.000000Z",
        "valid_to": "2026-07-01T00:00:00.000000Z",
        "known_at": datetime(2026, 6, 2, 8, tzinfo=timezone.utc),
        "superseded_at": None,
        "revision_id": revision,
        "quote_payload_hash": quote_payload_hash(bid, ask),
    }


def test_lineage_hash_uses_natural_numeric_key_order_and_rejects_huge_decimal() -> None:
    rows = [
        _lineage_row(100, 10),
        _lineage_row(100, 2),
        _lineage_row(20, 10),
        _lineage_row(20, 2),
    ]
    natural_order = [rows[3], rows[2], rows[1], rows[0]]
    manual_hash = hashlib.sha256(
        "".join(canonical_snapshot_row(row) for row in natural_order).encode(
            "utf-8"
        )
    ).hexdigest()

    assert canonical_snapshot_hash(rows) == manual_hash
    with pytest.raises(ValueError, match="outside"):
        canonical_decimal(Decimal("1e1000000"))
