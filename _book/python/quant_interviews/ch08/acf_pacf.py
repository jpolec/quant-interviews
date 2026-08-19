from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf


@dataclass(frozen=True)
class Correlogram:
    acf: NDArray[np.float64]
    pacf: NDArray[np.float64]
    pointwise_band: float
    ljung_box_pvalue: float
    effective_observations: int
    lags: NDArray[np.int64]
    series_role: str


def correlogram(
    values: ArrayLike,
    nlags: int = 20,
    *,
    series_role: str = "raw_series",
) -> Correlogram:
    """Return ACF/PACF evidence without changing the time axis.

    Missing observations are deliberately rejected.  Dropping them here would
    make lag one mean "next observed row" rather than one sampling interval.
    A caller that wants imputation or calendar compression must perform and
    record that transformation before calling this function.
    """

    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be a one-dimensional time series.")
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Missing or non-finite values require an explicit calendar policy."
        )
    if x.size < max(40, 3 * nlags):
        raise ValueError("A longer finite one-dimensional series is required.")
    if nlags <= 0 or nlags >= x.size // 2:
        raise ValueError("nlags must be positive and materially smaller than the sample.")
    if not isinstance(series_role, str) or not series_role.strip():
        raise ValueError("series_role must identify raw data or fitted residuals.")
    acf_values = np.asarray(acf(x, nlags=nlags, fft=True), dtype=float)
    pacf_values = np.asarray(pacf(x, nlags=nlags, method="ywmle"), dtype=float)
    lb = acorr_ljungbox(x - np.mean(x), lags=[nlags], return_df=True)
    band = 1.96 / np.sqrt(x.size)
    return Correlogram(
        acf=acf_values,
        pacf=pacf_values,
        pointwise_band=float(band),
        ljung_box_pvalue=float(lb["lb_pvalue"].iloc[0]),
        effective_observations=int(x.size),
        lags=np.arange(nlags + 1, dtype=np.int64),
        series_role=series_role.strip(),
    )
