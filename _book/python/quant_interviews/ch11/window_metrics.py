from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_RETURN_DOMAIN = "finite_positive_v1"


def _percent_rank(values: pd.Series) -> pd.Series:
    valid = values.notna()
    count = int(valid.sum())
    output = pd.Series(np.nan, index=values.index, dtype=float)
    if count == 0:
        return output
    if count == 1:
        output.loc[valid] = 0.0
        return output
    ranks = values.loc[valid].rank(method="min")
    output.loc[valid] = (ranks - 1.0) / (count - 1.0)
    return output


def returns_and_cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Match SQL PERCENT_RANK with RANK-style ties and NULL preservation."""

    required = {"timestamp", "asset", "price"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    data = frame.loc[:, ["timestamp", "asset", "price"]].copy()
    parsed = pd.to_datetime(data["timestamp"], errors="raise")
    if getattr(parsed.dt, "tz", None) is None:
        raise ValueError("timestamps must be timezone-aware")
    data["timestamp"] = parsed.dt.tz_convert("UTC")
    if data["asset"].isna().any() or data["price"].isna().any():
        raise ValueError("asset and price must not be null")
    if not np.all(np.isfinite(data["price"].to_numpy(dtype=float))):
        raise ValueError("prices must be finite")
    data = data.sort_values(["asset", "timestamp"], kind="mergesort")
    if data.duplicated(["asset", "timestamp"]).any():
        raise ValueError("duplicate observations")
    previous = data.groupby("asset", sort=False)["price"].shift(1)
    eligible = previous.notna() & (previous > 0.0) & (data["price"] > 0.0)
    data["return_1"] = np.nan
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        calculated = data.loc[eligible, "price"] / previous.loc[eligible] - 1.0
    if not np.all(np.isfinite(calculated.to_numpy(dtype=np.float64))):
        raise FloatingPointError("price return calculation produced a non-finite value")
    data.loc[eligible, "return_1"] = calculated
    data["rank"] = data.groupby("timestamp", sort=False, group_keys=False)[
        "return_1"
    ].apply(_percent_rank)
    return data
