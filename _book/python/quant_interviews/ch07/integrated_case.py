from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bond_risk import BondCashflows, bond_risk_certificate
from .callable_bond_tree import (
    CallableBond,
    calibrate_additive_short_rate_tree,
    callable_bond_oas_certificate,
    price_on_calibrated_tree,
)
from .cds_bootstrap import CDSQuote, bootstrap_cds_certificate
from .curve_bootstrap import bootstrap_curve_certificate
from .curve_representation import (
    DiscountCurve as RepresentationCurve,
    curve_representation_certificate,
)
from .credit_survival import FlatHazardCredit, credit_survival_certificate
from .evidence import DecisionStatus, stable_manifest_hash
from .hjm_drift import hjm_martingale_audit
from .hull_white_exact import hull_white_curve_fit_audit
from .key_rate_hedge import key_rate_hedge_certificate
from .multicurve_swap import example_swap, multicurve_risk_certificate
from .merton_structural import merton_credit_certificate
from .pnl_explain import PnLInputs, StateManifest, explain_certificate

ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class ReleaseGate:
    artifact_id: str
    required: bool
    status: DecisionStatus
    evidence_hash: str
    owner: str
    observed: float | str | bool
    operator: str
    limit: float | str | bool
    unit: str
    exception_owner: str | None = None
    exception_expires_at: str | None = None


REQUIRED_ARTIFACT_IDS = frozenset(
    {
        "curve_representation",
        "discount_bootstrap",
        "bond_risk",
        "key_rate_hedge",
        "multicurve_node_risk",
        "hull_white_curve_fit",
        "hjm_martingale",
        "credit_survival",
        "cds_bootstrap",
        "merton_parity",
        "callable_oas",
        "pnl_python",
        "asof_sql",
        "pnl_sql",
        "cpp_risk_kernel",
    }
)


def reduce_release_gates(
    gates: tuple[ReleaseGate, ...],
    *,
    required_artifact_ids: frozenset[str] = REQUIRED_ARTIFACT_IDS,
) -> DecisionStatus:
    by_id = {gate.artifact_id: gate for gate in gates}
    if len(by_id) != len(gates) or required_artifact_ids - set(by_id):
        return "REJECT"
    required = tuple(by_id[artifact_id] for artifact_id in required_artifact_ids)
    if any(not gate.required or gate.status == "REJECT" for gate in required):
        return "REJECT"
    if any(gate.status == "CONDITIONAL_PASS" for gate in required):
        return "CONDITIONAL_PASS"
    return "PASS"


def _validator_gate(relative_path: str) -> tuple[DecisionStatus, str, str]:
    path = ROOT / relative_path
    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    output = completed.stdout.strip()
    status: DecisionStatus = (
        "PASS"
        if completed.returncode == 0 and output.startswith("PASS ")
        else "REJECT"
    )
    evidence_hash = stable_manifest_hash(
        {
            "validator": relative_path,
            "validator_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "returncode": completed.returncode,
            "stdout": output,
            "stderr": completed.stderr.strip(),
        }
    )
    return status, evidence_hash, output or completed.stderr.strip()


def _certificate_hash(certificate: Any) -> str:
    """Content-address the result as well as the certificate input manifest."""

    return stable_manifest_hash(asdict(certificate))


def _source_hashes(relative_paths: tuple[str, ...]) -> dict[str, str]:
    return {
        relative_path: hashlib.sha256(
            (ROOT / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in sorted(relative_paths)
    }


def chapter_release_case(
    *,
    projection_first_node_shift: float = 0.0,
    policy_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not np.isfinite(projection_first_node_shift):
        raise ValueError("projection_first_node_shift must be finite")
    fixture_manifest = {
        "fixture_type": "synthetic governed fixed-income release fixture",
        "fixture_version": "ch07-release-v0.10",
        "valuation_timestamp": "2026-07-30T16:00:00+02:00",
        "knowledge_cutoff": "2026-07-30T15:55:00+02:00",
        "curve_policy": "FI-CURVE-v3",
        "credit_policy": "FI-CREDIT-v2",
        "oas_policy": "FI-OAS-v2",
        "pnl_policy": "FI-PNL-v4",
        "seed": 707,
    }
    if policy_overrides:
        unknown_policies = set(policy_overrides) - {
            "curve_policy",
            "credit_policy",
            "oas_policy",
            "pnl_policy",
        }
        if unknown_policies or any(
            not isinstance(value, str) or not value.strip()
            for value in policy_overrides.values()
        ):
            raise ValueError("policy overrides must name non-empty known policies")
        fixture_manifest.update(policy_overrides)

    deposits = {0.5: 0.0350, 1.0: 0.0370}
    swaps = {2: 0.0400, 3: 0.0420, 4: 0.0430, 5: 0.0440}
    curve = bootstrap_curve_certificate(deposits, swaps)
    representation = curve_representation_certificate(
        RepresentationCurve(
            curve.result.curve.times,
            curve.result.curve.discount_factors,
        )
    )

    portfolio = BondCashflows(
        times=np.arange(1.0, 6.0),
        amounts=np.array([5.0, 5.0, 5.0, 5.0, 105.0]),
    )
    bond_risk = bond_risk_certificate(
        portfolio,
        0.04,
        compounding="continuous",
    )
    hedge_bonds = tuple(
        BondCashflows(
            times=np.array([float(maturity)]),
            amounts=np.array([100.0]),
        )
        for maturity in range(1, 6)
    )
    hedge = key_rate_hedge_certificate(
        portfolio_bonds=(portfolio,),
        portfolio_notionals=np.array([1.0]),
        hedge_bonds=hedge_bonds,
        deposit_quotes={1.0: deposits[1.0]},
        swap_quotes=swaps,
        notional_rounding=1.0e-6,
    )

    swap, discount_curve, projection_curve = example_swap()
    shifted_projection_values = projection_curve.values.copy()
    shifted_projection_values[0] += projection_first_node_shift
    projection_curve = type(projection_curve)(
        projection_curve.times,
        shifted_projection_values,
    )
    multicurve = multicurve_risk_certificate(
        swap,
        discount_curve,
        projection_curve,
    )
    hull_white = hull_white_curve_fit_audit(
        curve_times=np.arange(1.0, 6.0),
        market_discount_factors=np.exp(-0.04 * np.arange(1.0, 6.0)),
        mean_reversion=0.60,
        volatility=0.012,
        paths=30_000,
        seed=709,
    )
    hjm = hjm_martingale_audit(
        maturities=np.array([1.0, 2.0, 4.0]),
        initial_forward_rate=0.035,
        sigma0=0.006,
        decay=0.30,
        paths=3_000,
        seed=fixture_manifest["seed"],
        z_limit=5.0,
        refinement_limit=1.0e-3,
    )
    credit_survival = credit_survival_certificate(
        FlatHazardCredit(0.03, 0.40, 0.025),
        5.0,
    )

    cds_quotes = (
        CDSQuote(1, 0.0100),
        CDSQuote(3, 0.0150),
        CDSQuote(5, 0.0100, upfront=0.0350),
    )
    cds_base = bootstrap_cds_certificate(
        cds_quotes,
        recovery=0.40,
        discount_curve=curve.result.curve,
        discount_curve_run_id=curve.input_manifest_hash,
    )
    cds_low_recovery = bootstrap_cds_certificate(
        cds_quotes,
        recovery=0.20,
        discount_curve=curve.result.curve,
        discount_curve_run_id=curve.input_manifest_hash,
    )
    cds_high_recovery = bootstrap_cds_certificate(
        cds_quotes,
        recovery=0.60,
        discount_curve=curve.result.curve,
        discount_curve_run_id=curve.input_manifest_hash,
    )
    merton = merton_credit_certificate(
        asset_value=150.0,
        debt_face=100.0,
        asset_volatility=0.25,
        risk_free_rate=0.03,
        maturity=1.0,
        physical_asset_drift=0.06,
    )

    callable_bond = CallableBond(100.0, 0.06, 5, 100.0, 2)
    market_dfs = np.exp(-0.04 * np.arange(1.0, 6.0))
    tree = calibrate_additive_short_rate_tree(market_dfs, 0.01)
    governed_market_price = price_on_calibrated_tree(
        callable_bond,
        tree,
        callable_=True,
        oas=0.0125,
    ).price
    callable_result = callable_bond_oas_certificate(
        bond=callable_bond,
        market_discount_factors=market_dfs,
        rate_step=0.01,
        market_price=governed_market_price,
        market_price_source="self_roundtrip",
    )

    market_snapshot = {
        "valuation_timestamp": fixture_manifest["valuation_timestamp"],
        "knowledge_cutoff": fixture_manifest["knowledge_cutoff"],
        "ordered_selected_sql_rows": (
            {
                "record_type": "QUOTE",
                "economic_key": "IRS5Y|OIS|5Y|PAR",
                "numeric_value": 0.0400,
                "row_hash": "a" * 64,
                "selection_status": "PASS",
            },
            {
                "record_type": "FIXING",
                "economic_key": "SOFR|2026-07-29",
                "numeric_value": 0.0435,
                "row_hash": "c" * 64,
                "selection_status": "PASS",
            },
        ),
        "discount_instruments": {
            "deposits": tuple(sorted(deposits.items())),
            "swaps": tuple(sorted(swaps.items())),
        },
        "discount_curve": {
            "times": curve.result.curve.times,
            "discount_factors": curve.result.curve.discount_factors,
        },
        "multicurve": {
            "payment_times": swap.payment_times,
            "accruals": swap.accruals,
            "notional": swap.notional,
            "discount_log_df_nodes": discount_curve.values,
            "projection_forward_nodes": projection_curve.values,
        },
        "portfolio": {
            "times": portfolio.times,
            "amounts": portfolio.amounts,
            "hedge_cashflows": tuple(
                {"times": bond.times, "amounts": bond.amounts}
                for bond in hedge_bonds
            ),
        },
        "cds_quotes": tuple(asdict(quote) for quote in cds_quotes),
        "callable_bond": asdict(callable_bond),
        "callable_market_discount_factors": market_dfs,
        "callable_market_price": governed_market_price,
        "pnl_market_moves_bp": np.array([2.0, -1.0, 3.0]),
    }
    market_snapshot_hash = stable_manifest_hash(market_snapshot)

    position_hash = stable_manifest_hash(
        {
            "artifact_type": "position",
            "market_snapshot_hash": market_snapshot_hash,
            "portfolio": market_snapshot["portfolio"],
        }
    )
    market_hash = stable_manifest_hash(
        {
            "artifact_type": "market",
            "market_snapshot_hash": market_snapshot_hash,
        }
    )
    risk_hash = stable_manifest_hash(
        {
            "artifact_type": "risk",
            "position_hash": position_hash,
            "market_hash": market_hash,
        }
    )
    pnl_inputs = PnLInputs(
        carry=12_000.0,
        roll_down=5_000.0,
        curve_dv01=np.array([800.0, 1_200.0, 500.0]),
        curve_move_bp=np.array([2.0, -1.0, 3.0]),
        curve_cross_gamma=np.diag([4.0, 6.0, 2.0]),
        basis_pnl=-250.0,
        cs01=900.0,
        credit_spread_move_bp=4.0,
        credit_convexity=3.0,
        volatility_pnl=-75.0,
        cash_and_fixings_pnl=400.0,
        exercise_pnl=0.0,
        model_change_pnl=0.0,
        data_correction_pnl=0.0,
        actual_pnl=13_000.0,
        position_manifest=StateManifest(
            "position", "ch07-position-run", position_hash, (), 1
        ),
        risk_manifest=StateManifest(
            "risk",
            "ch07-risk-run",
            risk_hash,
            (position_hash, market_hash),
            6,
        ),
        market_manifest=StateManifest(
            "market", "ch07-market-run", market_hash, (), 12
        ),
        component_evidence_ids={
            name: (f"ch07:{name}:evidence",)
            for name in (
                "carry",
                "roll_down",
                "curve_delta",
                "curve_cross_gamma",
                "basis",
                "credit_delta",
                "credit_convexity",
                "volatility",
                "cash_and_fixings",
                "exercise",
                "model_change",
                "data_correction",
            )
        },
    )
    pnl = explain_certificate(
        pnl_inputs,
        absolute_residual_limit=500.0,
        relative_residual_limit=0.05,
    )

    sql_status, sql_validator_hash, sql_output = _validator_gate(
        "_project/tests/sql/ch07/validate_sql.py"
    )
    cpp_status, cpp_validator_hash, cpp_output = _validator_gate(
        "_project/tests/cpp/ch07/validate_cpp.py"
    )

    certificate_by_artifact = {
        "curve_representation": representation,
        "discount_bootstrap": curve,
        "bond_risk": bond_risk,
        "key_rate_hedge": hedge,
        "multicurve_node_risk": multicurve,
        "hull_white_curve_fit": hull_white,
        "hjm_martingale": hjm,
        "credit_survival": credit_survival,
        "cds_bootstrap": cds_base,
        "merton_parity": merton,
        "callable_oas": callable_result,
        "pnl_python": pnl,
    }
    certificate_hashes = {
        artifact_id: _certificate_hash(certificate)
        for artifact_id, certificate in certificate_by_artifact.items()
    }
    gates = (
        ReleaseGate(
            "curve_representation", True, representation.status,
            certificate_hashes["curve_representation"], "Rates Model Owner",
            representation.result.maximum_reconstruction_error, "<=", 1.1e-10,
            "discount-factor units",
        ),
        ReleaseGate(
            "discount_bootstrap", True, curve.status,
            certificate_hashes["discount_bootstrap"], "Rates Curve Owner",
            curve.result.maximum_repricing_residual, "<=", 1.1e-10,
            "price units",
        ),
        ReleaseGate(
            "bond_risk", True, bond_risk.status,
            certificate_hashes["bond_risk"], "Rates Risk Owner",
            bond_risk.result.reconciliation_error, "<=", 1.1e-7,
            "currency per bp",
        ),
        ReleaseGate(
            "key_rate_hedge", True, hedge.status,
            certificate_hashes["key_rate_hedge"], "Hedging Owner",
            float(np.linalg.norm(hedge.result.residual_after_rounding)),
            "<=", 1.5e-6, "currency per bp",
        ),
        ReleaseGate(
            "multicurve_node_risk", True, multicurve.status,
            certificate_hashes["multicurve_node_risk"], "Rates Model Owner",
            multicurve.result.risk_reconciliation_error, "<=", 1.0e-6,
            "currency per bp",
        ),
        ReleaseGate(
            "hull_white_curve_fit", True, hull_white.status,
            certificate_hashes["hull_white_curve_fit"], "Rates Model Owner",
            hull_white.result.maximum_curve_fit_residual, "<=", 1.0e-5,
            "discount-factor units",
        ),
        ReleaseGate(
            "hjm_martingale", True, hjm.status,
            certificate_hashes["hjm_martingale"], "Rates Model Owner",
            hjm.result.maximum_absolute_z_score, "<=", 5.0, "z-score",
        ),
        ReleaseGate(
            "credit_survival", True, credit_survival.status,
            certificate_hashes["credit_survival"], "Credit Model Owner",
            credit_survival.result.survival_probability, "in", "[0,1]",
            "probability",
        ),
        ReleaseGate(
            "cds_bootstrap", True, cds_base.status,
            certificate_hashes["cds_bootstrap"], "Credit Curve Owner",
            cds_base.result.maximum_repricing_residual, "<=", 1.1e-10,
            "spread/upfront PV units",
        ),
        ReleaseGate(
            "merton_parity", True, merton.status,
            certificate_hashes["merton_parity"], "Credit Model Owner",
            merton.result.put_call_parity_residual, "<=", 1.6e-8,
            "firm-value currency",
        ),
        ReleaseGate(
            "callable_oas", True, callable_result.status,
            certificate_hashes["callable_oas"], "Rates Model Owner",
            callable_result.result.market_price_source, "==", "independent",
            "classification",
            exception_owner="Rates Model Owner",
            exception_expires_at="2026-08-31T23:59:59+02:00",
        ),
        ReleaseGate(
            "pnl_python", True, pnl.status,
            certificate_hashes["pnl_python"], "Fixed-Income Risk Owner",
            pnl.result.residual_fraction_of_actual, "<=", 0.05, "fraction",
        ),
        ReleaseGate(
            "asof_sql", True, sql_status,
            stable_manifest_hash(
                {"validator_hash": sql_validator_hash, "artifact": "asof_sql"}
            ),
            "Rates Data Owner", sql_status == "PASS", "==", True, "boolean",
        ),
        ReleaseGate(
            "pnl_sql", True, sql_status,
            stable_manifest_hash(
                {"validator_hash": sql_validator_hash, "artifact": "pnl_sql"}
            ),
            "Fixed-Income Risk Owner", sql_status == "PASS", "==", True,
            "boolean",
        ),
        ReleaseGate(
            "cpp_risk_kernel", True, cpp_status, cpp_validator_hash,
            "Rates Risk Owner", cpp_status == "PASS", "==", True, "boolean",
        ),
    )
    final_decision = reduce_release_gates(gates)
    component_statuses = {
        gate.artifact_id: gate.status for gate in gates
    }
    recovery_hazards = {
        "20%": cds_low_recovery.result.calibration_nodes[-1].hazard,
        "40%": cds_base.result.calibration_nodes[-1].hazard,
        "60%": cds_high_recovery.result.calibration_nodes[-1].hazard,
    }
    source_hashes = _source_hashes(
        (
            "_book/python/quant_interviews/ch07/bond_risk.py",
            "_book/python/quant_interviews/ch07/callable_bond_tree.py",
            "_book/python/quant_interviews/ch07/cds_bootstrap.py",
            "_book/python/quant_interviews/ch07/credit_survival.py",
            "_book/python/quant_interviews/ch07/curve_bootstrap.py",
            "_book/python/quant_interviews/ch07/curve_representation.py",
            "_book/python/quant_interviews/ch07/evidence.py",
            "_book/python/quant_interviews/ch07/hjm_drift.py",
            "_book/python/quant_interviews/ch07/hull_white_exact.py",
            "_book/python/quant_interviews/ch07/integrated_case.py",
            "_book/python/quant_interviews/ch07/key_rate_hedge.py",
            "_book/python/quant_interviews/ch07/merton_structural.py",
            "_book/python/quant_interviews/ch07/multicurve_swap.py",
            "_book/python/quant_interviews/ch07/pnl_explain.py",
            "_book/python/quant_interviews/ch07/tool_api.py",
            "_book/sql/ch07/fixed_income_asof_market_data.sql",
            "_book/sql/ch07/fixed_income_pnl_reconciliation.sql",
            "_book/cpp/ch07/fixed_income_risk_kernel.cpp",
            "_project/tests/cpp/ch07/fixed_income_risk_kernel_test.cpp",
            "_project/tests/sql/ch07/validate_sql.py",
            "_project/tests/cpp/ch07/validate_cpp.py",
            "_book/json/ch07/bootstrap_cds_hazard_curve.json",
            "_book/json/ch07/bootstrap_discount_curve.json",
            "_book/json/ch07/compute_key_rate_hedge.json",
            "_book/json/ch07/explain_fixed_income_pnl.json",
            "_book/json/ch07/price_callable_bond_oas.json",
            "_book/json/ch07/price_multicurve_swap.json",
            "_book/json/ch07/tool_contract_library.json",
            "_book/ai/prompts/ch07/ai-ch07_01.md",
            "_book/ai/prompts/ch07/ai-ch07_02.md",
            "_book/ai/prompts/ch07/ai-ch07_03.md",
            "_book/ai/prompts/ch07/ai-ch07_04.md",
            "_book/ai/prompts/ch07/ai-ch07_05.md",
            "_book/ai/prompts/ch07/ai-ch07_06.md",
            "_book/ai/prompts/ch07/ai-ch07_07.md",
            "_book/ai/prompts/ch07/ai-ch07_08.md",
            "_book/ai/prompts/ch07/ai-ch07_09.md",
            "_book/ai/prompts/ch07/prompt_library.md",
            "_book/figures/ch07/fig_07_06_survival_hazard.png",
            "_project/scripts/figures/build_ch07_v010_figures.py",
            "_project/scripts/revision/generate_ch07_prompt_library.py",
            "_project/scripts/revision/generate_ch07_tool_contracts.py",
        )
    )
    release_bundle = {
        "fixture_manifest": fixture_manifest,
        "market_snapshot": market_snapshot,
        "market_snapshot_hash": market_snapshot_hash,
        "policy_registry": {
            key: value
            for key, value in fixture_manifest.items()
            if key.endswith("_policy")
        },
        "source_revision": {
            "kind": "local_content_addressed_source_tree",
            "commit_required": False,
            "source_hashes": source_hashes,
        },
        "component_bindings": {
            artifact_id: {
                "market_snapshot_hash": market_snapshot_hash,
                "certificate_hash": certificate_hash,
                "implementation_version": (
                    certificate_by_artifact[artifact_id].implementation_version
                ),
            }
            for artifact_id, certificate_hash in certificate_hashes.items()
        },
        "gate_registry": tuple(asdict(gate) for gate in gates),
        "external_validator_outputs": {
            "sql": sql_output,
            "cpp": cpp_output,
        },
        "final_decision": final_decision,
    }
    release_bundle_hash = stable_manifest_hash(release_bundle)
    return {
        **fixture_manifest,
        "snapshot_hash": market_snapshot_hash,
        "market_snapshot_hash": market_snapshot_hash,
        "release_bundle_hash": release_bundle_hash,
        "release_bundle": release_bundle,
        "gate_registry": tuple(asdict(gate) for gate in gates),
        "component_statuses": component_statuses,
        "curve_maximum_repricing_residual": (
            curve.result.maximum_repricing_residual
        ),
        "curve_minimum_forward": curve.result.minimum_interval_forward,
        "multicurve_coordinate_condition_number": (
            multicurve.result.coordinate_transform_condition_number
        ),
        "discount_curve_dv01": multicurve.result.discount_zero_node_dv01,
        "projection_curve_dv01": (
            multicurve.result.projection_forward_node_dv01
        ),
        "portfolio_key_rate_dv01": (
            hedge.result.portfolio_key_rate_dv01
        ),
        "hedge_residual_after_rounding": (
            hedge.result.residual_after_rounding
        ),
        "hjm_maximum_absolute_z_score": (
            hjm.result.maximum_absolute_z_score
        ),
        "hull_white_curve_fit_residual": (
            hull_white.result.maximum_curve_fit_residual
        ),
        "cds_maximum_repricing_residual": (
            cds_base.result.maximum_repricing_residual
        ),
        "recovery_hazard_scenarios": recovery_hazards,
        "straight_bond_price": callable_result.result.straight_price,
        "callable_bond_price": callable_result.result.callable_price,
        "embedded_option_value": (
            callable_result.result.embedded_option_value
        ),
        "oas": callable_result.result.oas,
        "oas_repricing_residual": (
            callable_result.result.oas_repricing_residual
        ),
        "effective_duration": (
            callable_result.result.effective_duration
        ),
        "effective_convexity": (
            callable_result.result.effective_convexity
        ),
        "pnl_actual": pnl.result.actual_pnl,
        "pnl_explained": pnl.result.explained_pnl,
        "pnl_residual": pnl.result.residual,
        "pnl_residual_fraction": (
            pnl.result.residual_fraction_of_actual
        ),
        "final_decision": final_decision,
        "stop_conditions": tuple(
            {
                "artifact_id": gate.artifact_id,
                "observed": gate.observed,
                "operator": gate.operator,
                "limit": gate.limit,
                "unit": gate.unit,
                "status": gate.status,
            }
            for gate in gates
        ),
        "rollback_rule": (
            "restore the last signed state bundle whenever a snapshot hash, "
            "curve policy or lifecycle event differs from the release manifest"
        ),
        "owners": {
            "market_data": "Rates Data Owner",
            "model": "Rates and Credit Model Owner",
            "risk_limit": "Fixed-Income Risk Owner",
        },
    }
