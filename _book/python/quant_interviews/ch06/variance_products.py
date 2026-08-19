from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VarianceSwapAudit:
    fair_variance: float
    integral_component: float
    forward_correction: float
    k0: float
    included_strikes: int
    lower_tail_fraction: float
    upper_tail_fraction: float
    maximum_tail_fraction: float
    maximum_put_call_parity_residual: float
    parity_tolerance: float
    included_mask: tuple[bool, ...]
    inclusion_reasons: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class CorridorVarianceAudit:
    corridor_variance: float
    full_variance: float
    occupancy_fraction: float
    observation_convention: str
    boundary_convention: str
    return_convention: str
    return_consistency_error: float
    status: str


def _strike_widths(strikes: np.ndarray) -> np.ndarray:
    widths = np.empty_like(strikes)
    widths[1:-1] = 0.5 * (strikes[2:] - strikes[:-2])
    widths[0] = strikes[1] - strikes[0]
    widths[-1] = strikes[-1] - strikes[-2]
    return widths


def _zero_bid_inclusion_mask(
    *,
    k0_index: int,
    put_bids: np.ndarray,
    call_bids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    include = np.zeros(put_bids.size, dtype=bool)
    reasons = np.full(
        put_bids.size, "OUTSIDE_ZERO_BID_TRUNCATION", dtype=object
    )
    include[k0_index] = True
    reasons[k0_index] = "K0_AVERAGE"

    consecutive_zero_bids = 0
    for index in range(k0_index - 1, -1, -1):
        if put_bids[index] > 0.0:
            include[index] = True
            reasons[index] = "POSITIVE_PUT_BID"
            consecutive_zero_bids = 0
        else:
            reasons[index] = "ZERO_PUT_BID"
            consecutive_zero_bids += 1
            if consecutive_zero_bids >= 2:
                break

    consecutive_zero_bids = 0
    for index in range(k0_index + 1, call_bids.size):
        if call_bids[index] > 0.0:
            include[index] = True
            reasons[index] = "POSITIVE_CALL_BID"
            consecutive_zero_bids = 0
        else:
            reasons[index] = "ZERO_CALL_BID"
            consecutive_zero_bids += 1
            if consecutive_zero_bids >= 2:
                break
    return include, reasons


def variance_swap_audit(
    *,
    forward: float,
    strikes: np.ndarray,
    puts: np.ndarray,
    calls: np.ndarray,
    discount: float,
    horizon: float,
    put_bids: np.ndarray | None = None,
    call_bids: np.ndarray | None = None,
    lower_tail_cutoff: float = 0.70,
    upper_tail_cutoff: float = 1.30,
    parity_tolerance: float = 1e-8,
    maximum_tail_fraction: float = 0.25,
) -> VarianceSwapAudit:
    k = np.asarray(strikes, dtype=float)
    p = np.asarray(puts, dtype=float)
    c = np.asarray(calls, dtype=float)
    if (
        k.ndim != 1
        or p.shape != k.shape
        or c.shape != k.shape
        or k.size < 5
        or not np.isfinite(k).all()
        or not np.isfinite(p).all()
        or not np.isfinite(c).all()
        or np.any(k <= 0.0)
    ):
        raise ValueError("invalid variance-strip arrays")
    if np.any(np.diff(k) <= 0.0):
        raise ValueError("strikes must be unique and strictly increasing")
    scalars = np.asarray(
        (
            forward,
            discount,
            horizon,
            lower_tail_cutoff,
            upper_tail_cutoff,
            parity_tolerance,
            maximum_tail_fraction,
        ),
        dtype=float,
    )
    if not np.isfinite(scalars).all():
        raise ValueError("variance-strip scalar inputs must be finite")
    if forward <= 0.0 or discount <= 0.0 or horizon <= 0.0:
        raise ValueError("forward, discount and horizon must be positive")
    if not k[0] < forward < k[-1]:
        raise ValueError("variance strip must contain both put and call wings")
    if not 0.0 < lower_tail_cutoff < 1.0 < upper_tail_cutoff:
        raise ValueError("tail cutoffs must straddle the forward")
    if parity_tolerance < 0.0 or not 0.0 <= maximum_tail_fraction <= 1.0:
        raise ValueError("parity and tail limits are invalid")

    price_floor = 1e-12 * max(
        1.0,
        float(np.max(np.abs(p))),
        float(np.max(np.abs(c))),
    )
    if np.any(p < -price_floor) or np.any(c < -price_floor):
        raise ValueError("option prices cannot be negative")
    p = np.maximum(p, 0.0)
    c = np.maximum(c, 0.0)
    parity_residuals = c - p - discount * (forward - k)
    maximum_parity_residual = float(np.max(np.abs(parity_residuals)))
    parity_scale = max(1.0, forward * discount)
    if maximum_parity_residual > parity_tolerance * parity_scale:
        raise ValueError("put-call parity residual exceeds tolerance")

    k0_candidates = np.flatnonzero(k <= forward)
    k0_index = int(k0_candidates[-1])
    k0 = float(k[k0_index])
    quote = np.where(k < k0, p, c)
    quote[k0_index] = 0.5 * (p[k0_index] + c[k0_index])

    include = np.ones(k.size, dtype=bool)
    inclusion_reasons = np.full(k.size, "NO_BID_GATE", dtype=object)
    if put_bids is not None or call_bids is not None:
        if put_bids is None or call_bids is None:
            raise ValueError("put and call bids must be supplied together")
        put_bid = np.asarray(put_bids, dtype=float)
        call_bid = np.asarray(call_bids, dtype=float)
        if (
            put_bid.shape != k.shape
            or call_bid.shape != k.shape
            or not np.isfinite(put_bid).all()
            or not np.isfinite(call_bid).all()
            or np.any(put_bid < 0.0)
            or np.any(call_bid < 0.0)
        ):
            raise ValueError("bid grids must be finite, non-negative and match strikes")
        include, inclusion_reasons = _zero_bid_inclusion_mask(
            k0_index=k0_index,
            put_bids=put_bid,
            call_bids=call_bid,
        )
    if int(np.sum(include)) < 5:
        raise ValueError("too few quoted strikes after the zero-bid gate")

    widths = _strike_widths(k)
    weighted_terms = widths * quote / (k * k)
    integral_component = float(
        2.0 / (horizon * discount) * np.sum(weighted_terms[include])
    )
    correction = float((forward / k0 - 1.0) ** 2 / horizon)
    fair_variance = integral_component - correction
    if not np.isfinite((integral_component, correction, fair_variance)).all():
        raise ValueError("variance-strip calculation produced a non-finite result")

    total_weight = float(np.sum(weighted_terms[include]))
    lower_weight = float(
        np.sum(weighted_terms[include & (k < lower_tail_cutoff * forward)])
    )
    upper_weight = float(
        np.sum(weighted_terms[include & (k > upper_tail_cutoff * forward)])
    )
    lower_tail_fraction = lower_weight / max(total_weight, 1e-16)
    upper_tail_fraction = upper_weight / max(total_weight, 1e-16)
    status = (
        "PASS"
        if fair_variance >= 0.0
        and max(lower_tail_fraction, upper_tail_fraction)
        <= maximum_tail_fraction
        else "REJECT"
    )
    return VarianceSwapAudit(
        fair_variance=fair_variance,
        integral_component=integral_component,
        forward_correction=correction,
        k0=k0,
        included_strikes=int(np.sum(include)),
        lower_tail_fraction=lower_tail_fraction,
        upper_tail_fraction=upper_tail_fraction,
        maximum_tail_fraction=maximum_tail_fraction,
        maximum_put_call_parity_residual=maximum_parity_residual,
        parity_tolerance=parity_tolerance,
        included_mask=tuple(bool(value) for value in include),
        inclusion_reasons=tuple(str(value) for value in inclusion_reasons),
        status=status,
    )


def variance_swap_integral(
    forward: float,
    strikes: np.ndarray,
    puts: np.ndarray,
    calls: np.ndarray,
    discount: float,
    horizon: float,
) -> float:
    return variance_swap_audit(
        forward=forward,
        strikes=strikes,
        puts=puts,
        calls=calls,
        discount=discount,
        horizon=horizon,
    ).fair_variance


def corridor_realized_variance_audit(
    *,
    spots: np.ndarray,
    lower: float,
    upper: float,
    returns: np.ndarray | None = None,
    annualization: float = 252.0,
    observation_convention: str = "start",
    boundary_convention: str = "strict",
    return_consistency_tolerance: float = 1e-12,
) -> CorridorVarianceAudit:
    s = np.asarray(spots, dtype=float)
    if (
        s.ndim != 1
        or s.size < 2
        or not np.isfinite(s).all()
        or np.any(s <= 0.0)
    ):
        raise ValueError("spots must define one finite strictly positive path")
    scalars = np.asarray(
        (lower, upper, annualization, return_consistency_tolerance), dtype=float
    )
    if (
        not np.isfinite(scalars).all()
        or lower <= 0.0
        or lower >= upper
        or annualization <= 0.0
        or return_consistency_tolerance < 0.0
    ):
        raise ValueError("invalid corridor, annualization or tolerance")

    derived_returns = np.diff(np.log(s))
    consistency_error = 0.0
    if returns is not None:
        supplied_returns = np.asarray(returns, dtype=float)
        if (
            supplied_returns.ndim != 1
            or supplied_returns.shape != derived_returns.shape
            or not np.isfinite(supplied_returns).all()
        ):
            raise ValueError("supplied returns must align with the spot path")
        consistency_error = float(
            np.max(np.abs(supplied_returns - derived_returns))
        )
        if consistency_error > return_consistency_tolerance:
            raise ValueError("supplied returns are inconsistent with the spot path")

    if observation_convention == "start":
        observed_spot = s[:-1]
    elif observation_convention == "end":
        observed_spot = s[1:]
    elif observation_convention == "midpoint":
        observed_spot = np.sqrt(s[:-1] * s[1:])
    else:
        raise ValueError("unsupported observation convention")
    if boundary_convention == "strict":
        mask = (observed_spot > lower) & (observed_spot < upper)
    elif boundary_convention == "inclusive":
        mask = (observed_spot >= lower) & (observed_spot <= upper)
    else:
        raise ValueError("unsupported boundary convention")
    corridor = float(
        annualization * np.sum(mask * derived_returns * derived_returns)
        / derived_returns.size
    )
    full = float(
        annualization * np.sum(derived_returns * derived_returns)
        / derived_returns.size
    )
    return CorridorVarianceAudit(
        corridor_variance=corridor,
        full_variance=full,
        occupancy_fraction=float(np.mean(mask)),
        observation_convention=observation_convention,
        boundary_convention=boundary_convention,
        return_convention="log returns derived from consecutive positive spots",
        return_consistency_error=consistency_error,
        status="PASS",
    )


def corridor_realized_variance(
    returns: np.ndarray,
    spots: np.ndarray,
    lower: float,
    upper: float,
    annualization: float = 252.0,
) -> float:
    return corridor_realized_variance_audit(
        returns=returns,
        spots=spots,
        lower=lower,
        upper=upper,
        annualization=annualization,
    ).corridor_variance
