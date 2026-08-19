from __future__ import annotations

import numpy as np


def _strict_float64_array(value: np.ndarray, name: str, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array; implicit conversion is forbidden")
    if value.dtype != np.float64:
        raise ValueError(f"{name} must have dtype float64; implicit conversion is forbidden")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous; implicit layout copies are forbidden")
    if np.any(~np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def portfolio_returns(weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    w = _strict_float64_array(weights, "weights", 1)
    r = _strict_float64_array(asset_returns, "asset_returns", 2)
    if r.shape[1] != w.shape[0]:
        raise ValueError(f"asset dimension mismatch: {r.shape[1]} versus {w.shape[0]}")
    result = r @ w
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("portfolio return calculation produced non-finite output")
    return result


def scenario_pnl(exposures: np.ndarray, shocks: np.ndarray) -> np.ndarray:
    e = _strict_float64_array(exposures, "exposures", 2)
    s = _strict_float64_array(shocks, "shocks", 2)
    if e.shape[1] != s.shape[1]:
        raise ValueError("factor dimensions do not match")
    result = e @ s.T
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("scenario P&L calculation produced non-finite output")
    return result


def safe_standardize(values: np.ndarray, axis: int = 0, epsilon: float = 1e-12) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a NumPy array; implicit conversion is forbidden")
    if values.dtype != np.float64 or not values.flags.c_contiguous:
        raise ValueError("values must be a C-contiguous float64 array")
    x = values
    if x.ndim < 1:
        raise ValueError("values must have at least one dimension")
    if not 0 <= axis < x.ndim:
        raise ValueError("axis is outside the input dimensions")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    if x.shape[axis] < 2:
        raise ValueError("at least two observations are required")
    if np.any(~np.isfinite(x)):
        raise ValueError("values must be finite")
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, ddof=1, keepdims=True)
    if np.any(~np.isfinite(std)) or np.any(std <= epsilon):
        raise ValueError("cannot standardize a near-constant slice")
    result = (x - mean) / std
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("standardization produced non-finite output")
    return result
