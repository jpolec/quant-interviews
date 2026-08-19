from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

from .calibration import temporal_calibration
from .cointegration import engle_granger, fit_ecm
from .drift import DriftPolicy, DriftWindowRecord, drift_evidence
from .economic_alpha import evaluate_threshold
from .kalman import dynamic_beta_filter, innovation_diagnostics
from .stationarity import stationarity_report
from .validation import assert_no_interval_overlap, purged_embargo_split
from .volatility import evaluate_har_holdout, fit_garch11


JsonObject = dict[str, Any]
Adapter = Callable[[Mapping[str, Any]], JsonObject]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return _jsonable(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Publishable tool output cannot contain NaN or infinity.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported tool-output type: {type(value).__name__}")


def _digest(value: object) -> str:
    return sha256(
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _stationarity(request: Mapping[str, Any]) -> JsonObject:
    report = stationarity_report(
        request["series"],
        max_lag=int(request.get("max_lag", 5)),
        timestamps=request["timestamps"],
        calendar=str(request.get("calendar", request["sampling_convention"])),
        transformation=str(request.get("transformation", "identity")),
        deterministic_terms=str(request["deterministic_terms"]),  # type: ignore[arg-type]
        lag_rule=str(request.get("lag_rule", "fixed")),
        sampling_interval=str(request["sampling_convention"]),
    )
    return {
        "ar1_phi": report.ar1.phi,
        "half_life": report.ar1.half_life,
        "adf": _jsonable(report.adf),
        "kpss": _jsonable(report.kpss),
        "break_diagnostics": {
            "largest_mean_break": report.largest_mean_break,
            "mean_break_score": report.mean_break_score,
        },
        "lineage": {
            "sample_digest": report.sample_digest,
            "first_timestamp": report.first_timestamp,
            "last_timestamp": report.last_timestamp,
            "status": report.status,
            "spec": _jsonable(report.spec),
        },
    }


def _cointegration(request: Mapping[str, Any]) -> JsonObject:
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(request["timestamps"], utc=True, errors="raise")
    )
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("Cointegration timestamps must be unique and increasing.")
    if len(timestamps) != len(request["x"]) or len(timestamps) != len(request["y"]):
        raise ValueError("Cointegration timestamps must align with both series.")
    if request["regression_direction"] != "y_on_x":
        raise ValueError("Only the governed y_on_x direction is supported.")
    fit = engle_granger(
        request["x"],
        request["y"],
        adf_lag=int(request.get("adf_lag", 1)),
        require_i1=bool(request["require_i1"]),
        max_condition_number=float(request.get("max_condition_number", 1.0e10)),
    )
    ecm = fit_ecm(request["x"], request["y"], fit, lag_order=1)
    status = "PASS" if fit.cointegrated and ecm.scalar_stable else "REJECT"
    return {
        "status": status,
        "alpha": fit.alpha,
        "beta": fit.beta,
        "macKinnon_cointegration_test": {
            "statistic": fit.coint_stat,
            "pvalue": fit.coint_pvalue,
            "critical_values": fit.critical_values,
        },
        "bivariate_ecm": {
            "lambda_y": ecm.y_equation.adjustment,
            "lambda_x": ecm.x_equation.adjustment,
            "lambda_spread": ecm.spread_adjustment,
        },
        "stability": {
            "scalar_root": ecm.scalar_root,
            "stable": ecm.scalar_stable,
        },
        "lineage": {
            "x_digest": fit.x_digest,
            "y_digest": fit.y_digest,
            "selection_family_id": request["selection_family_id"],
            "timestamps_digest": _digest([value.isoformat() for value in timestamps]),
        },
    }


def _dynamic_hedge(request: Mapping[str, Any]) -> JsonObject:
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(request["decision_timestamps"], utc=True, errors="raise")
    )
    if len(timestamps) != len(request["x"]) or len(timestamps) != len(request["y"]):
        raise ValueError("Decision timestamps must align with x and y.")
    path = dynamic_beta_filter(
        request["x"],
        request["y"],
        process_variance=float(request["process_variance"]),
        process_intercept_variance=float(request.get("process_intercept_variance", 0.0)),
        observation_variance=float(request["observation_variance"]),
    )
    return {
        "predicted_beta": path.predicted_beta,
        "predicted_intercept": path.predicted_intercept,
        "filtered_beta": path.filtered_beta,
        "filtered_intercept": path.filtered_intercept,
        "predicted_covariance": path.predicted_covariance,
        "filtered_covariance": path.filtered_covariance,
        "innovations": path.innovations,
        "log_likelihood": path.log_likelihood,
        "lineage": {
            "decision_timestamps_digest": _digest(
                [value.isoformat() for value in timestamps]
            ),
            "state_timing": "predicted_before_observation; filtered_after_observation",
        },
    }


def _volatility(request: Mapping[str, Any]) -> JsonObject:
    model_type = request["model_type"]
    if model_type == "garch11":
        fit = fit_garch11(
            request["returns"],
            initial_variance=request.get("initial_variance"),
            persistence_margin=float(request.get("persistence_margin", 1.0e-4)),
        )
        next_forecast = float(
            fit.omega
            + fit.alpha * float(request["returns"][-1]) ** 2
            + fit.beta * fit.variance[-1]
        )
        return {
            "parameters": {
                "omega": fit.omega,
                "alpha": fit.alpha,
                "beta": fit.beta,
                "success": fit.success,
            },
            "forecasts": [next_forecast],
            "fold_losses": [],
            "target_diagnostics": {"status": "FORECAST_ONLY"},
            "initial_state_provenance": {
                "initial_variance": fit.initial_variance,
                "ownership": "training_sample",
            },
            "lineage": {"input_digest": _digest(request["returns"])},
        }
    if model_type != "har":
        raise ValueError("model_type must be garch11 or har.")
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(request["timestamps"], utc=True, errors="raise")
    )
    realized = pd.Series(request["realized_variance"], index=timestamps, dtype=float)
    evaluated = evaluate_har_holdout(
        realized,
        training_end=request["training_end"],
        evaluation_end=request.get("evaluation_end"),
        calendar=str(request["calendar"]),
    )
    return {
        "parameters": {
            "coefficients": evaluated.model.coefficients,
            "smearing_factor": evaluated.model.smearing_factor,
        },
        "forecasts": [_jsonable(row) for row in evaluated.records],
        "fold_losses": [{"loss": "qlike", "value": evaluated.qlike}],
        "target_diagnostics": {
            "strictly_positive": True,
            "calendar": evaluated.model.calendar,
        },
        "initial_state_provenance": {"status": "NOT_APPLICABLE_TO_HAR"},
        "lineage": {
            "model_training_end": evaluated.model.training_end,
            "target_digest": _digest(request["realized_variance"]),
        },
    }


def _purged_splits(request: Mapping[str, Any]) -> JsonObject:
    rows = request["sample_intervals"]
    intervals = pd.DataFrame(
        {
            "start": [row["start"] for row in rows],
            "end": [row["end"] for row in rows],
        },
        index=[row["id"] for row in rows],
    )
    embargo = pd.Timedelta(request["embargo_duration"]).to_pytimedelta()
    folds = []
    purged_counts = []
    embargoed_counts = []
    for block in request["test_blocks"]:
        fold = purged_embargo_split(
            intervals,
            block,
            embargo=embargo,
            mode=str(request["mode"]),  # type: ignore[arg-type]
        )
        assert_no_interval_overlap(intervals, fold)
        folds.append(_jsonable(fold))
        purged_counts.append(len(fold.purged_index))
        embargoed_counts.append(len(fold.embargo_index))
    return {
        "folds": folds,
        "purged_counts": purged_counts,
        "embargoed_counts": embargoed_counts,
        "invariant_results": {
            "half_open_intervals": True,
            "no_overlap": True,
        },
    }


def _calibration(request: Mapping[str, Any]) -> JsonObject:
    result = temporal_calibration(
        request["raw_scores"],
        request["labels"],
        score_timestamps=request["score_timestamps"],
        label_known_timestamps=request["label_known_timestamps"],
        model_fit_ids=request["model_fit_ids"],
        calibration_indices=request["calibration_indices"],
        evaluation_indices=request["evaluation_indices"],
        method=str(request["method"]),  # type: ignore[arg-type]
        epsilon=float(request["epsilon"]),
    )
    return {
        "probabilities": result.calibrated_probabilities,
        "brier": result.calibrated_metrics.brier,
        "log_loss": result.calibrated_metrics.log_loss,
        "reliability": result.calibrated_metrics.reliability_bins,
        "base_rate": result.calibrated_metrics.event_rate,
        "lineage": {
            "digest": result.lineage_digest,
            "calibration_start": result.calibration_start,
            "calibration_end": result.calibration_end,
            "evaluation_start": result.evaluation_start,
            "evaluation_end": result.evaluation_end,
        },
    }


def _monitor_drift(request: Mapping[str, Any]) -> JsonObject:
    policy_payload = request["threshold_policy"]
    policy = DriftPolicy(
        policy_id=str(policy_payload["policy_id"]),
        bandwidth=float(policy_payload["bandwidth"]),
        mmd_threshold=float(policy_payload["mmd_threshold"]),
        brier_threshold=float(policy_payload["brier_threshold"]),
        minimum_persistence=int(request["minimum_persistence"]),
    )
    history = tuple(DriftWindowRecord(**row) for row in request.get("history", []))
    reference = request["reference_window"]
    live = request["live_window"]
    result = drift_evidence(
        reference["scores"],
        live["scores"],
        policy=policy,
        window_id=str(live["window_id"]),
        window_start=live["start"],
        window_end=live["end"],
        model_hash=str(request["model_version"]),
        data_hash=str(live["data_hash"]),
        evidence_id=str(request["evidence_id"]),
        history=history,
        reference_labels=reference.get("labels"),
        live_labels=live.get("labels"),
    )
    return {
        "data_integrity": {
            "scale_status": result.scale_status,
            "record_id": result.current_record.record_id,
        },
        "distribution_drift": {
            "mmd2": result.mmd2,
            "estimator": result.mmd_estimator,
            "mean_shift": result.mean_shift,
            "std_ratio": result.std_ratio,
        },
        "calibration_drift": {"brier_change": result.brier_change},
        "economic_drift": {"status": "SEPARATE_SQL_LEDGER_REQUIRED"},
        "available_label_count": int(len(live.get("labels", []))),
        "recommended_action": "INVESTIGATE" if result.actionable else "CONTINUE_MONITORING",
        "actionable": result.actionable,
    }


def _economic(request: Mapping[str, Any]) -> JsonObject:
    position_rule = request["position_rule"]
    cost_model = request["cost_model"]
    risk_model = request["risk_model"]
    if position_rule.get("selection_context") != "inner_validation":
        raise ValueError("Economic threshold must carry inner_validation lineage.")
    result = evaluate_threshold(
        request["forecasts"],
        request["returns"],
        threshold=float(position_rule["threshold"]),
        cost_per_unit_turnover=float(cost_model["cost_per_unit_turnover"]),
        risk_aversion=float(risk_model["risk_aversion"]),
    )
    levels = request.get("capacity_levels", [1.0])
    impact = float(cost_model.get("impact_coefficient", 0.0))
    capacity = [
        {
            "assets_under_management": float(level),
            "net_pnl": float(
                level * result.gross_pnl
                - level * result.cost
                - impact * result.turnover * float(level) ** 1.5
            ),
        }
        for level in levels
    ]
    return {
        "gross_pnl": result.gross_pnl,
        "net_pnl": result.net_pnl,
        "turnover": result.turnover,
        "entry_turnover": result.entry_turnover,
        "terminal_exit_turnover": result.terminal_exit_turnover,
        "period_pnl": result.period_pnl,
        "long_run_variance": result.long_run_variance,
        "risk": {"volatility": result.volatility, "risk_aversion": risk_model["risk_aversion"]},
        "utility": result.utility,
        "attribution": {
            "gross": result.gross_pnl,
            "cost": result.cost,
            "residual": result.net_pnl - (result.gross_pnl - result.cost),
        },
        "capacity_curve": capacity,
    }


TOOL_ADAPTERS: dict[str, Adapter] = {
    "stationarity_diagnostic_bundle": _stationarity,
    "fit_cointegration_ecm": _cointegration,
    "filter_dynamic_hedge_ratio": _dynamic_hedge,
    "fit_volatility_forecast": _volatility,
    "generate_purged_cv_splits": _purged_splits,
    "calibrate_probabilities": _calibration,
    "monitor_model_drift": _monitor_drift,
    "evaluate_economic_alpha": _economic,
}


def _contracts() -> dict[str, JsonObject]:
    path = Path(__file__).resolve().parents[4] / "_book/json/ch08/tool_contract_library.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    return {str(value["name"]): value for value in values}


def execute_tool(name: str, request: Mapping[str, Any]) -> JsonObject:
    """Execute one exact Chapter 8 contract and return strict JSON data."""

    if name not in TOOL_ADAPTERS:
        raise KeyError(f"Unknown CH08 tool: {name}")
    if not isinstance(request, Mapping):
        raise TypeError("Tool request must be a JSON object.")
    contract = _contracts()[name]
    Draft202012Validator(contract["inputSchema"]).validate(dict(request))
    output = _jsonable(TOOL_ADAPTERS[name](request))
    Draft202012Validator(contract["outputSchema"]).validate(output)
    # A final serialization check rejects non-string keys and NaN/Infinity.
    json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return output
