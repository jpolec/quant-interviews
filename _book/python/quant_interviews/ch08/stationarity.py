from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import log, sqrt
from typing import Literal
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


DiagnosticStatus = Literal["OK", "BOUND", "ERROR", "UNAVAILABLE"]
CertificateStatus = Literal["PASS", "CONDITIONAL_PASS", "REJECT"]


@dataclass(frozen=True)
class AR1Estimate:
    intercept: float
    phi: float
    innovation_std: float
    half_life: float | None
    oscillatory: bool
    design_condition: float
    phi_standard_error: float
    phi_confidence_interval_95: tuple[float, float]


@dataclass(frozen=True)
class DiagnosticResult:
    statistic: float | None
    pvalue: float | None
    status: DiagnosticStatus
    message: str | None


@dataclass(frozen=True)
class StationaritySpec:
    calendar: str
    transformation: str
    deterministic_terms: Literal["c", "ct"]
    lag_rule: str
    max_lag: int
    sampling_interval: str
    significance: float


@dataclass(frozen=True)
class StationarityReport:
    n_obs: int
    ar1: AR1Estimate
    adf: DiagnosticResult
    kpss: DiagnosticResult
    largest_mean_break: int | None
    mean_break_score: float
    spec: StationaritySpec
    sample_digest: str
    first_timestamp: str | None
    last_timestamp: str | None
    status: CertificateStatus

    @property
    def adf_stat(self) -> float:
        if self.adf.statistic is None:
            raise RuntimeError("ADF statistic is unavailable.")
        return self.adf.statistic

    @property
    def adf_pvalue(self) -> float:
        if self.adf.pvalue is None:
            raise RuntimeError("ADF p-value is unavailable.")
        return self.adf.pvalue

    @property
    def kpss_stat(self) -> float | None:
        return self.kpss.statistic

    @property
    def kpss_pvalue(self) -> float | None:
        return self.kpss.pvalue


def _as_finite_1d(values: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("Expected a one-dimensional series.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Missing values require an explicit time-series policy.")
    if x.size < 20:
        raise ValueError("At least 20 finite observations are required.")
    if np.ptp(x) <= np.finfo(float).eps:
        raise ValueError("A nonconstant series is required.")
    return x


def _timestamps(
    timestamps: ArrayLike | None,
    *,
    observations: int,
) -> tuple[list[str], str | None, str | None]:
    if timestamps is None:
        return [], None, None
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
    if len(index) != observations:
        raise ValueError("timestamps must align one-to-one with values.")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("timestamps must be unique and strictly increasing.")
    rendered = [value.isoformat() for value in index]
    return rendered, rendered[0], rendered[-1]


def _digest(values: NDArray[np.float64], timestamps: list[str]) -> str:
    payload = {
        "timestamps": timestamps,
        "values": [float(value).hex() for value in values],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def fit_ar1(values: ArrayLike) -> AR1Estimate:
    x = _as_finite_1d(values)
    design = np.column_stack([np.ones(x.size - 1), x[:-1]])
    target = x[1:]
    condition = float(np.linalg.cond(design))
    if (
        np.linalg.matrix_rank(design) != design.shape[1]
        or not np.isfinite(condition)
        or condition > 1.0e10
    ):
        raise ValueError("The AR(1) regression is rank deficient or ill-conditioned.")
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    intercept, phi = map(float, coefficients)
    residual = target - design @ coefficients
    degrees_of_freedom = residual.size - design.shape[1]
    if degrees_of_freedom <= 0:
        raise ValueError("The AR(1) fit has no residual degrees of freedom.")
    residual_variance = float(np.dot(residual, residual) / degrees_of_freedom)
    innovation_std = sqrt(residual_variance)
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    phi_standard_error = sqrt(max(float(covariance[1, 1]), 0.0))
    z = 1.959963984540054
    interval = (phi - z * phi_standard_error, phi + z * phi_standard_error)
    half_life = None
    if phi == 0.0:
        half_life = 0.0
    elif 0.0 < abs(phi) < 1.0:
        half_life = float(log(0.5) / log(abs(phi)))
    return AR1Estimate(
        intercept=intercept,
        phi=phi,
        innovation_std=innovation_std,
        half_life=half_life,
        oscillatory=bool(phi < 0.0),
        design_condition=condition,
        phi_standard_error=phi_standard_error,
        phi_confidence_interval_95=interval,
    )


def largest_mean_break(
    values: ArrayLike,
    min_fraction: float = 0.2,
) -> tuple[int | None, float]:
    x = _as_finite_1d(values)
    if not 0.05 <= min_fraction <= 0.45:
        raise ValueError("min_fraction must be between 0.05 and 0.45.")
    lo = int(np.ceil(min_fraction * x.size))
    hi = int(np.floor((1.0 - min_fraction) * x.size))
    scale = float(np.std(x, ddof=1))
    if scale == 0.0:
        return None, 0.0
    scores = [
        abs(float(np.mean(x[:k]) - np.mean(x[k:]))) / scale
        for k in range(lo, hi)
    ]
    idx = int(np.argmax(scores))
    return lo + idx, float(scores[idx])


def stationarity_report(
    values: ArrayLike,
    max_lag: int = 5,
    *,
    timestamps: ArrayLike | None = None,
    calendar: str = "UNSPECIFIED",
    transformation: str = "identity",
    deterministic_terms: Literal["c", "ct"] = "c",
    lag_rule: str = "fixed",
    sampling_interval: str = "UNSPECIFIED",
    significance: float = 0.05,
) -> StationarityReport:
    """Build a provenance-bearing stationarity certificate.

    Diagnostic failure is represented as ``ERROR`` and makes the certificate
    reject.  It is never replaced by a synthetic reassuring p-value.
    """

    x = _as_finite_1d(values)
    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative.")
    if deterministic_terms not in {"c", "ct"}:
        raise ValueError("deterministic_terms must be 'c' or 'ct'.")
    if not 0.0 < significance < 1.0:
        raise ValueError("significance must lie strictly between zero and one.")
    for name, value in {
        "calendar": calendar,
        "transformation": transformation,
        "lag_rule": lag_rule,
        "sampling_interval": sampling_interval,
    }.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string.")
    rendered_timestamps, first_timestamp, last_timestamp = _timestamps(
        timestamps,
        observations=x.size,
    )
    spec = StationaritySpec(
        calendar=calendar.strip(),
        transformation=transformation.strip(),
        deterministic_terms=deterministic_terms,
        lag_rule=lag_rule.strip(),
        max_lag=max_lag,
        sampling_interval=sampling_interval.strip(),
        significance=float(significance),
    )
    adf_raw = adfuller(
        x,
        maxlag=max_lag,
        regression=deterministic_terms,
        autolag=None,
    )
    adf_result = DiagnosticResult(
        statistic=float(adf_raw[0]),
        pvalue=float(adf_raw[1]),
        status="OK",
        message=None,
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InterpolationWarning)
            kp = kpss(x, regression=deterministic_terms, nlags="auto")
        interpolation = next(
            (
                str(item.message)
                for item in caught
                if issubclass(item.category, InterpolationWarning)
            ),
            None,
        )
        kpss_result = DiagnosticResult(
            statistic=float(kp[0]),
            pvalue=float(kp[1]),
            status="BOUND" if interpolation is not None else "OK",
            message=interpolation,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        kpss_result = DiagnosticResult(
            statistic=None,
            pvalue=None,
            status="ERROR",
            message=f"{type(error).__name__}: {error}",
        )
    break_idx, break_score = largest_mean_break(x)
    lineage_complete = bool(rendered_timestamps) and all(
        value != "UNSPECIFIED" for value in (calendar, sampling_interval)
    )
    status: CertificateStatus
    if kpss_result.status == "ERROR" or not lineage_complete:
        status = "REJECT"
    elif kpss_result.status == "BOUND":
        status = "CONDITIONAL_PASS"
    else:
        status = "PASS"
    return StationarityReport(
        n_obs=int(x.size),
        ar1=fit_ar1(x),
        adf=adf_result,
        kpss=kpss_result,
        largest_mean_break=break_idx,
        mean_break_score=break_score,
        spec=spec,
        sample_digest=_digest(x, rendered_timestamps),
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        status=status,
    )
