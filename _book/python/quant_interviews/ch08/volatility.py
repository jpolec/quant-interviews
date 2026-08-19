from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize


@dataclass(frozen=True)
class GARCHFit:
    omega: float
    alpha: float
    beta: float
    initial_variance: float
    persistence_margin: float
    variance: NDArray[np.float64]
    log_likelihood: float
    success: bool
    iterations: int
    optimizer_message: str
    objective: float


@dataclass(frozen=True)
class HARFit:
    coefficients: NDArray[np.float64]
    log_forecast: pd.Series
    variance_forecast: pd.Series
    smearing_factor: float
    residual_std: float
    calendar: str
    training_start: pd.Timestamp
    training_end: pd.Timestamp


@dataclass(frozen=True)
class HARForecastRecord:
    forecast_origin: pd.Timestamp
    target_time: pd.Timestamp
    variance_forecast: float
    realized_variance: float
    model_training_end: pd.Timestamp


@dataclass(frozen=True)
class HARHoldoutEvaluation:
    records: tuple[HARForecastRecord, ...]
    qlike: float
    model: HARFit


def _returns(values: ArrayLike, *, minimum: int = 20) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 1
        or result.size < minimum
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(
            f"A finite one-dimensional series with at least {minimum} values is required."
        )
    return result


def garch11_filter(
    returns: ArrayLike,
    omega: float,
    alpha: float,
    beta: float,
    *,
    initial_variance: float | None = None,
    persistence_margin: float = 1.0e-4,
) -> NDArray[np.float64]:
    """Filter h_t without estimating its initial state from future observations."""

    result = _returns(returns)
    parameters = np.array(
        [omega, alpha, beta, persistence_margin],
        dtype=float,
    )
    if not np.all(np.isfinite(parameters)):
        raise ValueError("GARCH parameters must be finite.")
    if not 0.0 <= persistence_margin < 1.0:
        raise ValueError("persistence_margin must lie in [0, 1).")
    if (
        omega <= 0.0
        or alpha < 0.0
        or beta < 0.0
        or alpha + beta >= 1.0 - persistence_margin
    ):
        raise ValueError(
            "Require omega > 0, alpha,beta >= 0 and "
            "alpha + beta < 1 - persistence_margin."
        )
    unconditional = omega / (1.0 - alpha - beta)
    start = unconditional if initial_variance is None else float(initial_variance)
    if not np.isfinite(start) or start <= 0.0:
        raise ValueError("initial_variance must be finite and strictly positive.")

    variance = np.empty_like(result)
    variance[0] = start
    for index in range(1, result.size):
        variance[index] = (
            omega
            + alpha * result[index - 1] ** 2
            + beta * variance[index - 1]
        )
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise FloatingPointError("GARCH filtering produced an invalid variance.")
    return variance


def fit_garch11(
    returns: ArrayLike,
    *,
    initial_variance: float | None = None,
    persistence_margin: float = 1.0e-4,
) -> GARCHFit:
    """Estimate a Gaussian GARCH(1,1) on the supplied training sample."""

    result = _returns(returns, minimum=50)
    training_scale = float(np.var(result, ddof=1))
    if training_scale <= 0.0:
        raise ValueError("GARCH estimation requires nonzero training variance.")
    start_variance = (
        training_scale if initial_variance is None else float(initial_variance)
    )

    def objective(theta: NDArray[np.float64]) -> float:
        omega, alpha, beta = theta
        try:
            variance = garch11_filter(
                result,
                float(omega),
                float(alpha),
                float(beta),
                initial_variance=start_variance,
                persistence_margin=persistence_margin,
            )
        except (ValueError, FloatingPointError):
            return 1.0e100
        return float(
            0.5
            * np.sum(
                np.log(2.0 * np.pi)
                + np.log(variance)
                + result * result / variance
            )
        )

    fit = minimize(
        objective,
        x0=np.array(
            [max(training_scale * 0.03, 1.0e-12), 0.07, 0.88]
        ),
        method="Nelder-Mead",
        options={"maxiter": 4_000, "xatol": 1.0e-12, "fatol": 1.0e-9},
    )
    if not bool(fit.success):
        raise RuntimeError(
            "GARCH optimizer did not converge: "
            f"iterations={int(fit.nit)} message={fit.message}"
        )
    omega, alpha, beta = map(float, fit.x)
    variance = garch11_filter(
        result,
        omega,
        alpha,
        beta,
        initial_variance=start_variance,
        persistence_margin=persistence_margin,
    )
    return GARCHFit(
        omega=omega,
        alpha=alpha,
        beta=beta,
        initial_variance=start_variance,
        persistence_margin=persistence_margin,
        variance=variance,
        log_likelihood=-objective(fit.x),
        success=bool(fit.success),
        iterations=int(fit.nit),
        optimizer_message=str(fit.message),
        objective=float(fit.fun),
    )


def qlike(realized: ArrayLike, forecast: ArrayLike) -> float:
    realized_array = np.asarray(realized, dtype=float)
    forecast_array = np.asarray(forecast, dtype=float)
    if (
        realized_array.shape != forecast_array.shape
        or realized_array.ndim != 1
        or realized_array.size == 0
        or not np.all(np.isfinite(realized_array))
        or not np.all(np.isfinite(forecast_array))
        or np.any(realized_array <= 0.0)
        or np.any(forecast_array <= 0.0)
    ):
        raise ValueError(
            "QLIKE requires aligned, finite and strictly positive variance arrays."
        )
    ratio = realized_array / forecast_array
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _validated_realized_variance(
    realized_variance: pd.Series,
    *,
    calendar: str,
) -> pd.Series:
    if not isinstance(realized_variance, pd.Series):
        raise TypeError("realized_variance must be a pandas Series.")
    if not isinstance(realized_variance.index, pd.DatetimeIndex):
        raise ValueError("Realized variance requires a DatetimeIndex.")
    if realized_variance.index.tz is None:
        raise ValueError("Realized-variance timestamps must be timezone-aware.")
    if not realized_variance.index.is_monotonic_increasing:
        raise ValueError("Realized-variance timestamps must be increasing.")
    if realized_variance.index.has_duplicates:
        raise ValueError("Duplicate realized-variance timestamps are not allowed.")
    if not isinstance(calendar, str) or not calendar.strip():
        raise ValueError("calendar must name the governed trading calendar.")
    result = realized_variance.astype(float)
    values = result.to_numpy()
    if values.size < 30 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(
            "HAR requires at least 30 finite, strictly positive observations."
        )
    return result


def har_frame(
    realized_variance: pd.Series,
    *,
    calendar: str,
) -> pd.DataFrame:
    result = _validated_realized_variance(
        realized_variance,
        calendar=calendar,
    )
    log_variance = np.log(result)
    frame = pd.DataFrame(
        {
            "target": log_variance.shift(-1),
            "daily": log_variance,
            "weekly": log_variance.rolling(5, min_periods=5).mean(),
            "monthly": log_variance.rolling(22, min_periods=22).mean(),
        }
    )
    return frame.dropna()


def fit_har(
    realized_variance: pd.Series,
    *,
    calendar: str,
) -> HARFit:
    validated = _validated_realized_variance(
        realized_variance,
        calendar=calendar,
    )
    frame = har_frame(validated, calendar=calendar)
    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame[["daily", "weekly", "monthly"]].to_numpy(),
        ]
    )
    target = frame["target"].to_numpy()
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The HAR design is rank deficient.")
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted_log = design @ coefficients
    residual = target - fitted_log
    degrees = residual.size - design.shape[1]
    if degrees <= 0:
        raise ValueError("The HAR fit has no residual degrees of freedom.")
    residual_std = float(np.sqrt(np.dot(residual, residual) / degrees))
    smearing_factor = float(np.mean(np.exp(residual)))
    log_forecast = pd.Series(
        fitted_log,
        index=frame.index,
        name="log_forecast_variance",
    )
    variance_forecast = pd.Series(
        np.exp(fitted_log) * smearing_factor,
        index=frame.index,
        name="forecast_variance",
    )
    if not np.all(np.isfinite(variance_forecast)) or np.any(
        variance_forecast <= 0.0
    ):
        raise FloatingPointError("HAR inverse transformation is invalid.")
    return HARFit(
        coefficients=np.asarray(coefficients, dtype=float),
        log_forecast=log_forecast,
        variance_forecast=variance_forecast,
        smearing_factor=smearing_factor,
        residual_std=residual_std,
        calendar=calendar.strip(),
        training_start=validated.index[0],
        training_end=validated.index[-1],
    )


def _har_one_step(
    fit: HARFit,
    history: pd.Series,
) -> float:
    governed = _validated_realized_variance(history, calendar=fit.calendar)
    if len(governed) < 22:
        raise ValueError("At least 22 historical variances are required.")
    log_values = np.log(governed.to_numpy())
    design = np.array(
        [
            1.0,
            log_values[-1],
            float(np.mean(log_values[-5:])),
            float(np.mean(log_values[-22:])),
        ]
    )
    forecast = float(np.exp(design @ fit.coefficients) * fit.smearing_factor)
    if not np.isfinite(forecast) or forecast <= 0.0:
        raise FloatingPointError("HAR forecast is non-finite or nonpositive.")
    return forecast


def evaluate_har_holdout(
    realized_variance: pd.Series,
    *,
    training_end: pd.Timestamp | str,
    evaluation_end: pd.Timestamp | str | None = None,
    calendar: str,
) -> HARHoldoutEvaluation:
    """Fit once on training data and score only later one-step forecasts."""

    governed = _validated_realized_variance(
        realized_variance,
        calendar=calendar,
    )
    train_end = pd.Timestamp(training_end)
    if train_end.tzinfo is None:
        raise ValueError("training_end must be timezone-aware.")
    train_end = train_end.tz_convert("UTC")
    end = governed.index[-1]
    if evaluation_end is not None:
        end = pd.Timestamp(evaluation_end)
        if end.tzinfo is None:
            raise ValueError("evaluation_end must be timezone-aware.")
        end = end.tz_convert("UTC")
    training = governed.loc[governed.index <= train_end]
    evaluation = governed.loc[(governed.index > train_end) & (governed.index <= end)]
    if len(training) < 30 or evaluation.empty:
        raise ValueError("Training and nonempty later evaluation periods are required.")
    model = fit_har(training, calendar=calendar)
    records: list[HARForecastRecord] = []
    for target_time in evaluation.index:
        location = governed.index.get_loc(target_time)
        if not isinstance(location, (int, np.integer)) or location <= 0:
            raise ValueError("Target timestamps must resolve uniquely.")
        history = governed.iloc[: int(location)]
        origin = history.index[-1]
        records.append(
            HARForecastRecord(
                forecast_origin=origin,
                target_time=target_time,
                variance_forecast=_har_one_step(model, history),
                realized_variance=float(governed.loc[target_time]),
                model_training_end=model.training_end,
            )
        )
    loss = qlike(
        [record.realized_variance for record in records],
        [record.variance_forecast for record in records],
    )
    return HARHoldoutEvaluation(records=tuple(records), qlike=loss, model=model)
