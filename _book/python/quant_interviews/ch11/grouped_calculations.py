from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "asset", "value"}
ZSCORE_ZERO_TOLERANCE = 1.0e-12


def _utc_timestamps(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if getattr(parsed.dt, "tz", None) is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.dt.tz_convert("UTC")


def grouped_lagged_return(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    data = frame.loc[:, ["timestamp", "asset", "value"]].copy()
    data["timestamp"] = _utc_timestamps(data["timestamp"])
    if data["asset"].isna().any() or data["value"].isna().any():
        raise ValueError("asset and value must not be null")
    data["value"] = pd.to_numeric(data["value"], errors="raise").astype(np.float64)
    if not np.all(np.isfinite(data["value"].to_numpy(dtype=np.float64))):
        raise ValueError("values must be finite")
    data = data.sort_values(["asset", "timestamp"], kind="mergesort")
    if data.duplicated(["asset", "timestamp"]).any():
        raise ValueError("duplicate asset/timestamp rows")
    data["previous"] = data.groupby("asset", sort=False)["value"].shift(1)
    # Prices use an exact economic-domain gate, not an engine-dependent
    # ``isclose`` heuristic.  Every positive finite price is admissible;
    # zero and negative observations cannot define a price return.
    valid_previous = (
        data["previous"].notna()
        & (data["previous"] > 0.0)
        & (data["value"] > 0.0)
    )
    data["return"] = np.nan
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        calculated = (
            data.loc[valid_previous, "value"]
            / data.loc[valid_previous, "previous"]
            - 1.0
        )
    if not np.all(np.isfinite(calculated.to_numpy(dtype=np.float64))):
        raise FloatingPointError("price return calculation produced a non-finite value")
    data.loc[valid_previous, "return"] = calculated
    return data


def grouped_rolling_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    if window < 2:
        raise ValueError("window must be at least two")
    data = grouped_lagged_return(frame)
    grouped = data.groupby("asset", sort=False)["return"]
    rolling = grouped.rolling(window, min_periods=window)
    mean = rolling.mean().reset_index(level=0, drop=True).sort_index()
    std = rolling.std(ddof=1).reset_index(level=0, drop=True).sort_index()
    data["zscore"] = (data["return"] - mean) / std
    data.loc[
        (~np.isfinite(std)) | (std.abs() <= ZSCORE_ZERO_TOLERANCE),
        "zscore",
    ] = np.nan
    return data
