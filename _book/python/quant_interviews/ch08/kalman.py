from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass(frozen=True)
class KalmanPath:
    predicted_state: NDArray[np.float64]
    filtered_state: NDArray[np.float64]
    predicted_covariance: NDArray[np.float64]
    filtered_covariance: NDArray[np.float64]
    innovations: NDArray[np.float64]
    standardized_innovations: NDArray[np.float64]
    innovation_variance: NDArray[np.float64]
    log_likelihood: float

    @property
    def predicted_intercept(self) -> NDArray[np.float64]:
        return self.predicted_state[:, 0]

    @property
    def predicted_beta(self) -> NDArray[np.float64]:
        return self.predicted_state[:, 1]

    @property
    def filtered_intercept(self) -> NDArray[np.float64]:
        return self.filtered_state[:, 0]

    @property
    def filtered_beta(self) -> NDArray[np.float64]:
        return self.filtered_state[:, 1]

    @property
    def filtered_variance(self) -> NDArray[np.float64]:
        """Backward-compatible beta-variance view."""

        return self.filtered_covariance[:, 1, 1]


@dataclass(frozen=True)
class InnovationDiagnostics:
    observations: int
    mean: float
    variance: float
    ljung_box_lag: int
    ljung_box_pvalue: float
    tail_exceedance_rate_3sigma: float


@dataclass(frozen=True)
class KalmanCalibration:
    process_variance: float
    process_intercept_variance: float
    observation_variance: float
    optimizer_iterations: int
    optimizer_message: str
    training_log_likelihood: float
    training_input_digest: str
    terminal_filtered_state: NDArray[np.float64]
    terminal_filtered_covariance: NDArray[np.float64]


@dataclass(frozen=True)
class KalmanHoldoutEvaluation:
    path: KalmanPath
    diagnostics: InnovationDiagnostics
    one_step_rmse: float
    one_step_log_likelihood: float


def dynamic_beta_filter(
    x: ArrayLike,
    y: ArrayLike,
    *,
    process_variance: float,
    observation_variance: float,
    process_intercept_variance: float = 0.0,
    initial_intercept: float = 0.0,
    initial_beta: float = 1.0,
    initial_intercept_variance: float = 1.0,
    initial_variance: float = 1.0,
    initial_covariance: ArrayLike | None = None,
) -> KalmanPath:
    """Filter a dynamic intercept and beta using the Joseph covariance update."""

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if (
        x_array.ndim != 1
        or y_array.ndim != 1
        or x_array.size != y_array.size
        or x_array.size < 2
    ):
        raise ValueError("x and y must be aligned one-dimensional arrays.")
    parameters = np.array(
        [
            process_variance,
            observation_variance,
            process_intercept_variance,
            initial_intercept,
            initial_beta,
            initial_intercept_variance,
            initial_variance,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(parameters)):
        raise ValueError("All filter parameters must be finite.")
    if (
        process_variance < 0.0
        or process_intercept_variance < 0.0
        or observation_variance <= 0.0
        or initial_intercept_variance <= 0.0
        or initial_variance <= 0.0
    ):
        raise ValueError("Q must be PSD, R positive and P0 positive definite.")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("Missing observations require an explicit filter policy.")

    count = x_array.size
    predicted_state = np.empty((count, 2))
    filtered_state = np.empty((count, 2))
    predicted_covariance = np.empty((count, 2, 2))
    filtered_covariance = np.empty((count, 2, 2))
    innovations = np.empty(count)
    standardized = np.empty(count)
    innovation_variance = np.empty(count)

    state = np.array([initial_intercept, initial_beta], dtype=float)
    if initial_covariance is None:
        covariance = np.diag([initial_intercept_variance, initial_variance])
    else:
        covariance = np.asarray(initial_covariance, dtype=float)
        if (
            covariance.shape != (2, 2)
            or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.T, atol=1.0e-12, rtol=1.0e-12)
            or np.min(np.linalg.eigvalsh(covariance)) <= 0.0
        ):
            raise ValueError("initial_covariance must be finite symmetric positive definite.")
    process_covariance = np.diag(
        [process_intercept_variance, process_variance]
    )
    identity = np.eye(2)
    log_likelihood = 0.0
    for index, (regressor, observation) in enumerate(zip(x_array, y_array)):
        state_prediction = state.copy()
        covariance_prediction = covariance + process_covariance
        observation_matrix = np.array([1.0, regressor])
        innovation = float(observation - observation_matrix @ state_prediction)
        innovation_scale = float(
            observation_matrix @ covariance_prediction @ observation_matrix
            + observation_variance
        )
        if not np.isfinite(innovation_scale) or innovation_scale <= 0.0:
            raise FloatingPointError("Invalid Kalman innovation variance.")
        gain = covariance_prediction @ observation_matrix / innovation_scale
        state = state_prediction + gain * innovation

        residual_operator = identity - np.outer(gain, observation_matrix)
        covariance = (
            residual_operator
            @ covariance_prediction
            @ residual_operator.T
            + observation_variance * np.outer(gain, gain)
        )
        covariance = 0.5 * (covariance + covariance.T)
        minimum_eigenvalue = float(np.linalg.eigvalsh(covariance)[0])
        if not np.all(np.isfinite(covariance)) or minimum_eigenvalue < -1.0e-12:
            raise FloatingPointError("Filtered covariance lost positive semidefiniteness.")

        predicted_state[index] = state_prediction
        filtered_state[index] = state
        predicted_covariance[index] = covariance_prediction
        filtered_covariance[index] = covariance
        innovations[index] = innovation
        standardized[index] = innovation / np.sqrt(innovation_scale)
        innovation_variance[index] = innovation_scale
        log_likelihood += -0.5 * (
            np.log(2.0 * np.pi)
            + np.log(innovation_scale)
            + innovation * innovation / innovation_scale
        )
    return KalmanPath(
        predicted_state=predicted_state,
        filtered_state=filtered_state,
        predicted_covariance=predicted_covariance,
        filtered_covariance=filtered_covariance,
        innovations=innovations,
        standardized_innovations=standardized,
        innovation_variance=innovation_variance,
        log_likelihood=float(log_likelihood),
    )


def innovation_diagnostics(
    standardized_innovations: ArrayLike,
    *,
    ljung_box_lag: int = 10,
) -> InnovationDiagnostics:
    values = np.asarray(standardized_innovations, dtype=float)
    if (
        values.ndim != 1
        or values.size < max(20, 3 * ljung_box_lag)
        or not np.all(np.isfinite(values))
        or ljung_box_lag <= 0
        or ljung_box_lag >= values.size
    ):
        raise ValueError("Invalid standardized-innovation diagnostic sample.")
    result = acorr_ljungbox(values, lags=[ljung_box_lag], return_df=True)
    return InnovationDiagnostics(
        observations=int(values.size),
        mean=float(np.mean(values)),
        variance=float(np.var(values, ddof=1)),
        ljung_box_lag=ljung_box_lag,
        ljung_box_pvalue=float(result["lb_pvalue"].iloc[0]),
        tail_exceedance_rate_3sigma=float(np.mean(np.abs(values) > 3.0)),
    )


def _input_digest(x: NDArray[np.float64], y: NDArray[np.float64]) -> str:
    payload = (
        np.ascontiguousarray(x, dtype="<f8").tobytes()
        + np.ascontiguousarray(y, dtype="<f8").tobytes()
    )
    return sha256(payload).hexdigest()


def fit_dynamic_beta_state_space(
    x_train: ArrayLike,
    y_train: ArrayLike,
    *,
    initial_intercept: float = 0.0,
    initial_beta: float = 1.0,
) -> KalmanCalibration:
    """Calibrate Q/R only on the supplied training sample and fail closed."""

    x_array = np.asarray(x_train, dtype=float)
    y_array = np.asarray(y_train, dtype=float)
    if (
        x_array.ndim != 1
        or y_array.shape != x_array.shape
        or x_array.size < 50
        or not np.all(np.isfinite(x_array))
        or not np.all(np.isfinite(y_array))
    ):
        raise ValueError("At least 50 aligned finite training observations are required.")
    residual_scale = max(float(np.var(y_array, ddof=1)) * 0.05, 1.0e-8)

    def objective(log_variances: NDArray[np.float64]) -> float:
        process_intercept, process_beta, observation = np.exp(log_variances)
        try:
            path = dynamic_beta_filter(
                x_array,
                y_array,
                process_variance=float(process_beta),
                process_intercept_variance=float(process_intercept),
                observation_variance=float(observation),
                initial_intercept=initial_intercept,
                initial_beta=initial_beta,
            )
        except (ValueError, FloatingPointError):
            return 1.0e100
        return -path.log_likelihood

    fit = minimize(
        objective,
        np.log([1.0e-5, 1.0e-5, residual_scale]),
        method="L-BFGS-B",
        bounds=[(-25.0, 2.0), (-25.0, 2.0), (-25.0, 10.0)],
    )
    if not bool(fit.success) or not np.isfinite(fit.fun):
        raise RuntimeError(f"Kalman Q/R calibration failed: {fit.message}")
    process_intercept, process_beta, observation = map(float, np.exp(fit.x))
    path = dynamic_beta_filter(
        x_array,
        y_array,
        process_variance=process_beta,
        process_intercept_variance=process_intercept,
        observation_variance=observation,
        initial_intercept=initial_intercept,
        initial_beta=initial_beta,
    )
    return KalmanCalibration(
        process_variance=process_beta,
        process_intercept_variance=process_intercept,
        observation_variance=observation,
        optimizer_iterations=int(fit.nit),
        optimizer_message=str(fit.message),
        training_log_likelihood=path.log_likelihood,
        training_input_digest=_input_digest(x_array, y_array),
        terminal_filtered_state=path.filtered_state[-1].copy(),
        terminal_filtered_covariance=path.filtered_covariance[-1].copy(),
    )


def evaluate_dynamic_beta_holdout(
    calibration: KalmanCalibration,
    x_test: ArrayLike,
    y_test: ArrayLike,
    *,
    ljung_box_lag: int = 5,
) -> KalmanHoldoutEvaluation:
    """Evaluate one-step predicted, never smoothed, states on later data."""

    x_array = np.asarray(x_test, dtype=float)
    y_array = np.asarray(y_test, dtype=float)
    path = dynamic_beta_filter(
        x_array,
        y_array,
        process_variance=calibration.process_variance,
        process_intercept_variance=calibration.process_intercept_variance,
        observation_variance=calibration.observation_variance,
        initial_intercept=float(calibration.terminal_filtered_state[0]),
        initial_beta=float(calibration.terminal_filtered_state[1]),
        initial_covariance=calibration.terminal_filtered_covariance,
    )
    predicted_y = (
        path.predicted_intercept + x_array * path.predicted_beta
    )
    return KalmanHoldoutEvaluation(
        path=path,
        diagnostics=innovation_diagnostics(
            path.standardized_innovations,
            ljung_box_lag=ljung_box_lag,
        ),
        one_step_rmse=float(np.sqrt(np.mean((y_array - predicted_y) ** 2))),
        one_step_log_likelihood=path.log_likelihood,
    )
