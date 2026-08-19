"""Focused regression tests for the governed Chapter 2 allocation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import quant_interviews.ch02.risk as risk_module
from quant_interviews.ch02.case_study import (
    evaluate_case_decision,
    governed_risk_model_case,
)
from quant_interviews.ch02.optimization import (
    constrained_turnover_regularized,
    long_only_markowitz_audit,
    turnover_regularized,
)
from quant_interviews.ch02.risk import (
    risk_contributions,
    risk_parity,
)
from quant_interviews.ch02.sparse import long_only_l1_norm


def _four_asset_allocation() -> dict[str, float | int | str | np.ndarray]:
    covariance = np.array(
        [
            [0.05, 0.01, 0.00, 0.00],
            [0.01, 0.08, 0.01, 0.00],
            [0.00, 0.01, 0.06, 0.01],
            [0.00, 0.00, 0.01, 0.10],
        ]
    )
    constraints = np.column_stack(
        [np.ones(4), [-1.0, -0.4, 0.3, 1.0]]
    )
    return constrained_turnover_regularized(
        covariance,
        [0.03, 0.05, 0.08, 0.11],
        np.full(4, 0.25),
        constraints,
        [1.0, 0.0],
        risk_aversion=2.0,
        turnover_penalty=0.04,
    )


def test_constrained_turnover_uses_one_allocation_for_all_evidence() -> None:
    result = _four_asset_allocation()
    weights = np.asarray(result["weights"])
    previous = np.full(4, 0.25)
    constraints = np.column_stack(
        [np.ones(4), [-1.0, -0.4, 0.3, 1.0]]
    )

    assert constraints.T @ weights == pytest.approx([1.0, 0.0], abs=1e-12)
    assert result["constraint_residuals"] == pytest.approx(
        constraints.T @ weights - [1.0, 0.0]
    )
    assert result["turnover"] == pytest.approx(
        np.abs(weights - previous).sum()
    )
    assert result["turnover_cost"] == pytest.approx(
        0.04 * np.abs(weights - previous).sum()
    )
    assert result["factor_neutrality_residual"] < 1e-12
    assert result["stationarity_residual"] < 1e-8
    assert result["dual_feasibility_violation"] < 1e-12
    assert result["complementarity_residual"] < 1e-10
    assert result["maximum_kkt_violation"] < 1e-8
    assert result["status"] == "PASS"


def test_constrained_turnover_zero_cost_solves_same_equalities() -> None:
    covariance = np.diag([0.04, 0.06, 0.09])
    constraints = np.column_stack([np.ones(3), [-1.0, 0.0, 1.0]])

    result = constrained_turnover_regularized(
        covariance,
        [0.02, 0.05, 0.09],
        np.full(3, 1.0 / 3.0),
        constraints,
        [1.0, 0.0],
        risk_aversion=3.0,
        turnover_penalty=0.0,
    )

    assert constraints.T @ result["weights"] == pytest.approx(
        [1.0, 0.0],
        abs=1e-13,
    )
    assert result["turnover_cost"] == 0.0
    assert result["maximum_kkt_violation"] < 1e-13
    assert result["solver_iterations"] == 0


def test_constrained_turnover_rejects_invalid_covariance_and_equalities() -> None:
    common = (
        [0.03, 0.04],
        [0.5, 0.5],
        np.ones((2, 1)),
        [1.0],
    )
    with pytest.raises(ValueError, match="positive-definite"):
        constrained_turnover_regularized(
            [[1.0, 2.0], [2.0, 1.0]],
            *common,
            risk_aversion=2.0,
            turnover_penalty=0.1,
        )
    with pytest.raises(ValueError, match="full-rank"):
        constrained_turnover_regularized(
            np.eye(2),
            common[0],
            common[1],
            np.ones((2, 2)),
            [1.0, 0.0],
            risk_aversion=2.0,
            turnover_penalty=0.1,
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "control",
    ["risk_aversion", "turnover_penalty", "tolerance"],
)
def test_constrained_turnover_rejects_nonfinite_controls(
    invalid: float,
    control: str,
) -> None:
    controls = {
        "risk_aversion": 2.0,
        "turnover_penalty": 0.1,
        "tolerance": 1e-8,
    }
    controls[control] = invalid

    with pytest.raises(ValueError, match=control):
        constrained_turnover_regularized(
            np.eye(2),
            [0.03, 0.04],
            [0.5, 0.5],
            np.ones((2, 1)),
            [1.0],
            **controls,
        )


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_constrained_turnover_rejects_invalid_iteration_limit(
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        constrained_turnover_regularized(
            np.eye(2),
            [0.03, 0.04],
            [0.5, 0.5],
            np.ones((2, 1)),
            [1.0],
            risk_aversion=2.0,
            turnover_penalty=0.1,
            max_iterations=invalid,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_existing_optimizers_reject_nonfinite_controls(invalid: float) -> None:
    with pytest.raises(ValueError, match="risk_aversion"):
        turnover_regularized(
            np.eye(2),
            [0.03, 0.04],
            [0.5, 0.5],
            risk_aversion=invalid,
            turnover_penalty=0.1,
            penalty="l1",
        )
    with pytest.raises(ValueError, match="upper_bound"):
        long_only_markowitz_audit(
            np.eye(2),
            [0.03, 0.04],
            risk_aversion=2.0,
            upper_bound=invalid,
        )


def _passing_decision_inputs() -> dict[str, float | int | bool]:
    return {
        "condition_number": 150.0,
        "turnover": 0.20,
        "maximum_kkt_violation": 1e-8,
        "factor_neutrality_residual": 1e-8,
        "lineage_matches": True,
        "pit_windows": 3,
        "human_approved": True,
        "conditioning_limit": 150.0,
        "turnover_limit": 0.20,
        "kkt_residual_limit": 1e-8,
        "factor_neutrality_limit": 1e-8,
        "required_pit_windows": 3,
    }


def test_case_decision_boundaries_are_inclusive_and_pass() -> None:
    result = evaluate_case_decision(**_passing_decision_inputs())

    assert result == {"decision": "PASS", "decision_reasons": ()}


def test_case_decision_temporal_and_human_gaps_are_conditional() -> None:
    inputs = _passing_decision_inputs()
    inputs.update(pit_windows=2, human_approved=False)

    result = evaluate_case_decision(**inputs)

    assert result["decision"] == "CONDITIONAL_PASS"
    assert result["decision_reasons"] == (
        "INSUFFICIENT_PIT_WINDOWS",
        "HUMAN_APPROVAL_REQUIRED",
    )


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"condition_number": 150.0 + 1e-9}, "CONDITIONING_LIMIT_EXCEEDED"),
        ({"turnover": 0.20 + 1e-9}, "TURNOVER_LIMIT_EXCEEDED"),
        ({"maximum_kkt_violation": 1.1e-8}, "KKT_LIMIT_EXCEEDED"),
        (
            {"factor_neutrality_residual": 1.1e-8},
            "FACTOR_NEUTRALITY_BREACH",
        ),
        ({"lineage_matches": False}, "LINEAGE_MISMATCH"),
    ],
)
def test_case_decision_each_hard_gate_rejects(
    updates: dict[str, float | bool],
    reason: str,
) -> None:
    inputs = _passing_decision_inputs()
    inputs.update(updates)

    result = evaluate_case_decision(**inputs)

    assert result["decision"] == "REJECT"
    assert reason in result["decision_reasons"]


def test_case_decision_rejects_nonfinite_and_noninteger_evidence() -> None:
    inputs = _passing_decision_inputs()
    inputs["turnover"] = np.nan
    with pytest.raises(ValueError, match="turnover"):
        evaluate_case_decision(**inputs)

    inputs = _passing_decision_inputs()
    inputs["pit_windows"] = 2.5
    with pytest.raises(ValueError, match="pit_windows"):
        evaluate_case_decision(**inputs)


def _case_constraints(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    generator.normal(size=(45, 3))
    loadings = generator.normal(scale=0.5, size=(3, 25))
    return np.column_stack([np.ones(25), loadings[0], loadings[1]])


def test_seed7_case_rejects_one_neutral_portfolio_on_turnover() -> None:
    result = governed_risk_model_case(7)
    weights = np.asarray(result["weights"])
    previous = np.full(25, 1.0 / 25.0)

    assert _case_constraints(7).T @ weights == pytest.approx(
        [1.0, 0.0, 0.0],
        abs=5e-16,
    )
    assert result["turnover"] == pytest.approx(
        np.abs(weights - previous).sum(),
        abs=1e-15,
    )
    assert result["turnover"] == pytest.approx(0.279274107788, abs=2e-9)
    assert result["turnover_cost"] == pytest.approx(
        0.30 * result["turnover"]
    )
    assert result["raw_target_turnover"] == pytest.approx(
        2.034687233413,
        abs=2e-9,
    )
    assert result["no_trade_positions"] == 21
    assert result["factor_neutrality_residual"] < 2e-16
    assert result["turnover_kkt_violation"] < 1e-8
    assert result["decision"] == "REJECT"
    assert "TURNOVER_LIMIT_EXCEEDED" in result["decision_reasons"]


def test_seed8_case_rejects_conditioning_breach() -> None:
    result = governed_risk_model_case(8)

    assert result["shrunk_condition_number"] > 150.0
    assert result["decision"] == "REJECT"
    assert "CONDITIONING_LIMIT_EXCEEDED" in result["decision_reasons"]


def test_governed_case_rejects_invalid_seed_contract() -> None:
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError, match="seed"):
            governed_risk_model_case(invalid)  # type: ignore[arg-type]


def test_risk_contributions_accept_singular_psd_with_positive_variance() -> None:
    result = risk_contributions([1.0, 0.0], [[0.04, 0.0], [0.0, 0.0]])

    assert result == pytest.approx([0.2, 0.0])


@pytest.mark.parametrize(
    "covariance",
    [
        [[1.0, 0.2], [0.3, 1.0]],
        [[1.0, 2.0], [2.0, 1.0]],
        [[1.0, np.nan], [np.nan, 1.0]],
    ],
)
def test_risk_contributions_reject_invalid_covariance(
    covariance: list[list[float]],
) -> None:
    with pytest.raises(ValueError):
        risk_contributions([0.5, 0.5], covariance)


def test_risk_contributions_reject_zero_variance_explicitly() -> None:
    with pytest.raises(ValueError, match="positive portfolio variance"):
        risk_contributions([0.0, 1.0], [[0.04, 0.0], [0.0, 0.0]])


def test_risk_parity_exposes_solver_termination_evidence() -> None:
    result = risk_parity(np.diag([0.04, 0.09, 0.16]), [0.2, 0.3, 0.5])

    assert result["termination_status"] == "SOLVER_SUCCESS"
    assert isinstance(result["termination_message"], str)
    assert result["termination_message"]
    assert result["fallback_residual_limit"] == pytest.approx(1e-9)
    assert result["status"] == "PASS"


def test_risk_parity_fallback_is_tied_to_requested_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.sqrt(np.array([0.5, 0.5]))

    def failed_but_accurate(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            x=raw,
            nit=5,
            success=False,
            message="synthetic line-search termination",
        )

    monkeypatch.setattr(risk_module, "minimize", failed_but_accurate)
    result = risk_parity(np.eye(2), tolerance=1e-10)

    assert result["termination_status"] == "RESIDUAL_ACCEPTED"
    assert result["fallback_residual_limit"] == pytest.approx(1e-9)
    assert result["status"] == "CONDITIONAL_PASS"


def test_risk_parity_rejects_failed_solver_outside_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_and_inaccurate(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            x=np.ones(2),
            nit=5,
            success=False,
            message="synthetic inaccurate termination",
        )

    monkeypatch.setattr(risk_module, "minimize", failed_and_inaccurate)
    with pytest.raises(RuntimeError, match="inaccurate termination"):
        risk_parity(np.eye(2), tolerance=1e-10)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_risk_parity_rejects_nonfinite_tolerance(invalid: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        risk_parity(np.eye(2), tolerance=invalid)


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_risk_parity_rejects_noninteger_iteration_limit(invalid: object) -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        risk_parity(np.eye(2), max_iterations=invalid)  # type: ignore[arg-type]


def test_long_only_l1_norm_has_a_direct_simplex_contract() -> None:
    assert long_only_l1_norm([0.2, 0.3, 0.5]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="long-only"):
        long_only_l1_norm([0.6, -0.1, 0.5])
