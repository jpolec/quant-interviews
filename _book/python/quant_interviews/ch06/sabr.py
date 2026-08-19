from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt

import numpy as np


@dataclass(frozen=True)
class SabrSmileAudit:
    strikes: np.ndarray
    implied_vols: np.ndarray
    atm_vol: float
    nearest_grid_strike: float
    nearest_grid_vol: float
    downside_skew: float
    curvature: float
    maximum_absolute_log_moneyness: float
    maximum_absolute_log_moneyness_limit: float
    nu_squared_horizon: float
    maximum_nu_squared_horizon: float
    invalid_volatility_count: int
    parameter_shock: str
    quote_convention: str
    status: str


def sabr_parameter_checks(
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    *,
    forward: float | None = None,
    shift: float = 0.0,
) -> None:
    values = (alpha, beta, rho, nu, shift)
    if not all(isfinite(value) for value in values):
        raise ValueError("SABR parameters must be finite")
    if alpha <= 0.0 or nu < 0.0 or not 0.0 <= beta <= 1.0 or abs(rho) >= 1.0:
        raise ValueError("invalid SABR parameters")
    if forward is not None:
        if not isfinite(forward) or forward + shift <= 0.0:
            raise ValueError("forward must be finite and forward plus shift positive")


def hagan_black_vol(
    *,
    forward: float,
    strike: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    horizon: float,
    shift: float = 0.0,
) -> float:
    sabr_parameter_checks(
        alpha,
        beta,
        rho,
        nu,
        forward=forward,
        shift=shift,
    )
    shifted_forward = forward + shift
    shifted_strike = strike + shift
    if shifted_strike <= 0.0 or horizon < 0.0:
        raise ValueError("shifted strike must be positive and horizon nonnegative")
    one_minus_beta = 1.0 - beta
    if abs(shifted_forward - shifted_strike) <= 1e-12 * shifted_forward:
        base = alpha / shifted_forward**one_minus_beta
        correction = (
            one_minus_beta**2
            * alpha**2
            / (24.0 * shifted_forward ** (2.0 * one_minus_beta))
            + rho
            * beta
            * nu
            * alpha
            / (4.0 * shifted_forward**one_minus_beta)
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
        ) * horizon
        result = float(base * (1.0 + correction))
        if not isfinite(result) or result <= 0.0:
            raise ValueError("Hagan approximation produced invalid volatility")
        return result

    log_fk = log(shifted_forward / shifted_strike)
    fk_beta = (shifted_forward * shifted_strike) ** (0.5 * one_minus_beta)
    z = (nu / alpha) * fk_beta * log_fk if nu > 0.0 else 0.0
    if abs(z) < 1e-8:
        z_over_x = 1.0 - 0.5 * rho * z
    else:
        x_z = log(
            (sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho)
            / (1.0 - rho)
        )
        z_over_x = z / x_z
    denominator = fk_beta * (
        1.0
        + one_minus_beta**2 * log_fk**2 / 24.0
        + one_minus_beta**4 * log_fk**4 / 1920.0
    )
    time_correction = (
        one_minus_beta**2 * alpha**2 / (24.0 * fk_beta * fk_beta)
        + rho * beta * nu * alpha / (4.0 * fk_beta)
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    ) * horizon
    result = float(alpha * z_over_x / denominator * (1.0 + time_correction))
    if not isfinite(result) or result <= 0.0:
        raise ValueError("Hagan approximation produced invalid volatility")
    return result


def hagan_atm_vol(
    forward: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    horizon: float,
    shift: float = 0.0,
) -> float:
    return hagan_black_vol(
        forward=forward,
        strike=forward,
        alpha=alpha,
        beta=beta,
        rho=rho,
        nu=nu,
        horizon=horizon,
        shift=shift,
    )


def sabr_smile_audit(
    *,
    forward: float,
    strikes: np.ndarray,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    horizon: float,
    shift: float = 0.0,
    parameter_shock: str = "base",
    maximum_absolute_log_moneyness_limit: float = 1.0,
    maximum_nu_squared_horizon: float = 2.0,
) -> SabrSmileAudit:
    strike_array = np.asarray(strikes, dtype=float)
    if (
        strike_array.ndim != 1
        or strike_array.size < 5
        or not np.isfinite(strike_array).all()
        or np.any(np.diff(strike_array) <= 0.0)
        or forward + shift <= 0.0
        or np.any(strike_array + shift <= 0.0)
        or horizon < 0.0
        or not isfinite(maximum_absolute_log_moneyness_limit)
        or maximum_absolute_log_moneyness_limit < 0.0
        or not isfinite(maximum_nu_squared_horizon)
        or maximum_nu_squared_horizon < 0.0
    ):
        raise ValueError("strikes must be finite, unique and increasing")
    computed_vols: list[float] = []
    invalid_count = 0
    for strike in strike_array:
        try:
            computed_vols.append(
                hagan_black_vol(
                forward=forward,
                strike=float(strike),
                alpha=alpha,
                beta=beta,
                rho=rho,
                nu=nu,
                horizon=horizon,
                shift=shift,
            )
            )
        except ValueError:
            computed_vols.append(float("nan"))
            invalid_count += 1
    vols = np.asarray(computed_vols)
    atm_index = int(np.argmin(np.abs(strike_array - forward)))
    if not strike_array[0] < forward < strike_array[-1]:
        raise ValueError("strike grid must bracket the forward")
    if not 0 < atm_index < strike_array.size - 1:
        raise ValueError("nearest grid strike must have neighbours")
    if invalid_count:
        left_slope = float("nan")
        curvature = float("nan")
    else:
        left_slope = (
            vols[atm_index] - vols[atm_index - 1]
        ) / (strike_array[atm_index] - strike_array[atm_index - 1])
        right_slope = (
            vols[atm_index + 1] - vols[atm_index]
        ) / (strike_array[atm_index + 1] - strike_array[atm_index])
        curvature = right_slope - left_slope
    domain_ratio = float(
        np.max(np.abs(np.log((strike_array + shift) / (forward + shift))))
    )
    nu_squared_horizon = nu * nu * horizon
    exact_atm_vol = hagan_atm_vol(
        forward,
        alpha,
        beta,
        rho,
        nu,
        horizon,
        shift,
    )
    if invalid_count:
        status = "REJECT"
    elif (
        domain_ratio <= maximum_absolute_log_moneyness_limit
        and nu_squared_horizon <= maximum_nu_squared_horizon
    ):
        status = "PASS"
    else:
        status = "CONDITIONAL_PASS"
    return SabrSmileAudit(
        strikes=strike_array,
        implied_vols=vols,
        atm_vol=exact_atm_vol,
        nearest_grid_strike=float(strike_array[atm_index]),
        nearest_grid_vol=float(vols[atm_index]),
        downside_skew=float(left_slope),
        curvature=float(curvature),
        maximum_absolute_log_moneyness=domain_ratio,
        maximum_absolute_log_moneyness_limit=maximum_absolute_log_moneyness_limit,
        nu_squared_horizon=nu_squared_horizon,
        maximum_nu_squared_horizon=maximum_nu_squared_horizon,
        invalid_volatility_count=invalid_count,
        parameter_shock=parameter_shock,
        quote_convention="shifted Black volatility",
        status=status,
    )
