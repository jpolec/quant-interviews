from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import ArrayLike, NDArray
from statsmodels.tsa.stattools import adfuller, coint


@dataclass(frozen=True)
class CointegrationFit:
    alpha: float
    beta: float
    residuals: NDArray[np.float64]
    coint_stat: float
    coint_pvalue: float
    critical_values: NDArray[np.float64]
    design_rank: int
    design_condition: float
    x_level_adf_pvalue: float
    x_difference_adf_pvalue: float
    y_level_adf_pvalue: float
    y_difference_adf_pvalue: float
    cointegrated: bool
    x_digest: str
    y_digest: str
    residual_digest: str
    regression_direction: str
    deterministic_terms: str
    adf_lag: int
    significance: float


@dataclass(frozen=True)
class ECMEquation:
    intercept: float
    adjustment: float
    lagged_dy: NDArray[np.float64]
    lagged_dx: NDArray[np.float64]
    residual_std: float
    design_rank: int
    design_condition: float


@dataclass(frozen=True)
class ECMFit:
    y_equation: ECMEquation
    x_equation: ECMEquation
    lag_order: int
    n_obs: int
    spread_adjustment: float
    scalar_root: float
    scalar_stable: bool
    certificate_digest: str

    @property
    def adjustment(self) -> float:
        """Backward-compatible alias for the Y-equation adjustment speed."""

        return self.y_equation.adjustment


def _pair(
    x: ArrayLike,
    y: ArrayLike,
    *,
    min_obs: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if (
        x_array.ndim != 1
        or y_array.ndim != 1
        or x_array.size != y_array.size
    ):
        raise ValueError("x and y must be aligned one-dimensional arrays.")
    if x_array.size < min_obs:
        raise ValueError(f"At least {min_obs} paired observations are required.")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("Missing or non-finite observations require an explicit policy.")
    if np.ptp(x_array) <= np.finfo(float).eps or np.ptp(y_array) <= np.finfo(float).eps:
        raise ValueError("Cointegration is undefined for a constant input series.")
    return x_array, y_array


def _adf_pvalue(values: NDArray[np.float64], *, maxlag: int) -> float:
    return float(
        adfuller(
            values,
            maxlag=maxlag,
            regression="c",
            autolag=None,
        )[1]
    )


def _array_digest(values: NDArray[np.float64]) -> str:
    """Hash shape and exact IEEE-754 values, not a rounded representation."""

    contiguous = np.ascontiguousarray(values, dtype="<f8")
    payload = (
        f"shape={contiguous.shape};dtype=<f8;".encode("ascii")
        + contiguous.tobytes(order="C")
    )
    return sha256(payload).hexdigest()


def engle_granger(
    x: ArrayLike,
    y: ArrayLike,
    adf_lag: int = 1,
    *,
    significance: float = 0.05,
    require_i1: bool = True,
    min_obs: int = 80,
    max_condition_number: float = 1.0e10,
) -> CointegrationFit:
    """Fit the long-run relation and apply the MacKinnon cointegration test."""

    if adf_lag < 0:
        raise ValueError("adf_lag must be nonnegative.")
    if not 0.0 < significance < 1.0:
        raise ValueError("significance must lie strictly between zero and one.")
    if max_condition_number <= 1.0 or not np.isfinite(max_condition_number):
        raise ValueError("max_condition_number must be finite and greater than one.")
    x_array, y_array = _pair(x, y, min_obs=min_obs)
    design = np.column_stack([np.ones(x_array.size), x_array])
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design))
    if rank != design.shape[1]:
        raise ValueError("The long-run regression is rank deficient.")
    if not np.isfinite(condition) or condition > max_condition_number:
        raise ValueError("The long-run regression is ill-conditioned.")

    alpha, beta = np.linalg.lstsq(design, y_array, rcond=None)[0]
    # Use the same canonical arithmetic downstream certificates recompute.
    residuals = y_array - float(alpha) - float(beta) * x_array
    coint_stat, coint_pvalue, critical_values = coint(
        y_array,
        x_array,
        trend="c",
        maxlag=adf_lag,
        autolag=None,
    )
    x_level = _adf_pvalue(x_array, maxlag=adf_lag)
    y_level = _adf_pvalue(y_array, maxlag=adf_lag)
    x_difference = _adf_pvalue(np.diff(x_array), maxlag=adf_lag)
    y_difference = _adf_pvalue(np.diff(y_array), maxlag=adf_lag)
    i1_compatible = (
        x_level > significance
        and y_level > significance
        and x_difference < significance
        and y_difference < significance
    )
    if require_i1 and not i1_compatible:
        raise ValueError("Both level series must be compatible with I(1).")
    accepted = bool(coint_pvalue < significance and i1_compatible)
    return CointegrationFit(
        alpha=float(alpha),
        beta=float(beta),
        residuals=np.asarray(residuals, dtype=float),
        coint_stat=float(coint_stat),
        coint_pvalue=float(coint_pvalue),
        critical_values=np.asarray(critical_values, dtype=float),
        design_rank=rank,
        design_condition=condition,
        x_level_adf_pvalue=x_level,
        x_difference_adf_pvalue=x_difference,
        y_level_adf_pvalue=y_level,
        y_difference_adf_pvalue=y_difference,
        cointegrated=accepted,
        x_digest=_array_digest(x_array),
        y_digest=_array_digest(y_array),
        residual_digest=_array_digest(np.asarray(residuals, dtype=float)),
        regression_direction="y_on_x",
        deterministic_terms="c",
        adf_lag=adf_lag,
        significance=float(significance),
    )


def _fit_equation(
    target: NDArray[np.float64],
    design: NDArray[np.float64],
    *,
    lag_order: int,
    max_condition_number: float,
) -> ECMEquation:
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design))
    if rank != design.shape[1]:
        raise ValueError("The ECM design is rank deficient.")
    if not np.isfinite(condition) or condition > max_condition_number:
        raise ValueError("The ECM design is ill-conditioned.")
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ coefficients
    degrees_of_freedom = residual.size - design.shape[1]
    if degrees_of_freedom <= 0:
        raise ValueError("The ECM has no residual degrees of freedom.")
    residual_std = float(
        np.sqrt(np.dot(residual, residual) / degrees_of_freedom)
    )
    lagged_dy = np.asarray(coefficients[2 : 2 + lag_order], dtype=float)
    lagged_dx = np.asarray(coefficients[2 + lag_order :], dtype=float)
    return ECMEquation(
        intercept=float(coefficients[0]),
        adjustment=float(coefficients[1]),
        lagged_dy=lagged_dy,
        lagged_dx=lagged_dx,
        residual_std=residual_std,
        design_rank=rank,
        design_condition=condition,
    )


def fit_ecm(
    x: ArrayLike,
    y: ArrayLike,
    coint_fit: CointegrationFit | None = None,
    *,
    lag_order: int = 1,
    max_condition_number: float = 1.0e10,
) -> ECMFit:
    """Estimate both equations of a bivariate ECM with lagged differences."""

    if lag_order < 1:
        raise ValueError("lag_order must be at least one.")
    x_array, y_array = _pair(
        x,
        y,
        min_obs=max(80, 5 * lag_order + 10),
    )
    if coint_fit is None:
        coint_fit = engle_granger(x_array, y_array)
    if not coint_fit.cointegrated:
        raise ValueError("The cointegration certificate is not accepted.")
    if coint_fit.residuals.size != x_array.size:
        raise ValueError("The cointegration fit does not align with the input pair.")
    if coint_fit.regression_direction != "y_on_x":
        raise ValueError("The cointegration regression direction is incompatible.")
    if (
        coint_fit.x_digest != _array_digest(x_array)
        or coint_fit.y_digest != _array_digest(y_array)
    ):
        raise ValueError("The cointegration certificate belongs to another input pair.")

    spread = y_array - coint_fit.alpha - coint_fit.beta * x_array
    if coint_fit.residual_digest != _array_digest(spread):
        raise ValueError("The cointegration residual digest does not reconcile.")
    dy = np.diff(y_array)
    dx = np.diff(x_array)
    target_y = dy[lag_order:]
    target_x = dx[lag_order:]
    columns: list[NDArray[np.float64]] = [
        np.ones(target_y.size),
        spread[:-1][lag_order:],
    ]
    for lag in range(1, lag_order + 1):
        columns.append(dy[lag_order - lag : -lag])
    for lag in range(1, lag_order + 1):
        columns.append(dx[lag_order - lag : -lag])
    design = np.column_stack(columns)
    y_equation = _fit_equation(
            target_y,
            design,
            lag_order=lag_order,
            max_condition_number=max_condition_number,
        )
    x_equation = _fit_equation(
            target_x,
            design,
            lag_order=lag_order,
            max_condition_number=max_condition_number,
        )
    spread_adjustment = float(
        y_equation.adjustment - coint_fit.beta * x_equation.adjustment
    )
    scalar_root = 1.0 + spread_adjustment
    return ECMFit(
        y_equation=y_equation,
        x_equation=x_equation,
        lag_order=lag_order,
        n_obs=int(target_y.size),
        spread_adjustment=spread_adjustment,
        scalar_root=scalar_root,
        scalar_stable=bool(abs(scalar_root) < 1.0),
        certificate_digest=sha256(
            (
                coint_fit.residual_digest
                + f":lag={lag_order}:n={target_y.size}"
            ).encode("ascii")
        ).hexdigest(),
    )
