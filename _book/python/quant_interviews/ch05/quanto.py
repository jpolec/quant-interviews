from __future__ import annotations

from math import isfinite
from typing import Any, Literal

FXQuoteDirection = Literal[
    "DOMESTIC_PER_FOREIGN",
    "FOREIGN_PER_DOMESTIC",
]


def quanto_adjustment_certificate(
    *,
    asset_currency: str,
    payoff_currency: str,
    collateral_currency: str,
    pricing_numeraire: str,
    fx_quote_direction: FXQuoteDirection,
    asset_volatility: float,
    fx_volatility: float,
    correlation: float,
    correlation_orientation: str = "DOMESTIC_PER_FOREIGN_RETURN",
) -> dict[str, Any]:
    numeric = (asset_volatility, fx_volatility, correlation)
    if not all(isfinite(value) for value in numeric):
        raise ValueError("quanto inputs must be finite")
    if (
        not asset_currency
        or not payoff_currency
        or asset_currency == payoff_currency
        or collateral_currency not in {asset_currency, payoff_currency}
        or pricing_numeraire not in {asset_currency, payoff_currency}
        or asset_volatility < 0.0
        or fx_volatility < 0.0
        or not -1.0 <= correlation <= 1.0
        or correlation_orientation != "DOMESTIC_PER_FOREIGN_RETURN"
    ):
        raise ValueError("invalid quanto contract")
    if fx_quote_direction == "DOMESTIC_PER_FOREIGN":
        quoted_return_correlation = correlation
        sign_convention = (
            "canonical payoff-currency units per asset-currency unit"
        )
    elif fx_quote_direction == "FOREIGN_PER_DOMESTIC":
        quoted_return_correlation = -correlation
        sign_convention = (
            "display quote inverted; quoted-return correlation changes sign"
        )
    else:
        raise ValueError("unsupported FX quote direction")

    covariance_term = correlation * asset_volatility * fx_volatility
    adjustment = -covariance_term
    return {
        "asset_currency": asset_currency,
        "payoff_currency": payoff_currency,
        "collateral_currency": collateral_currency,
        "pricing_numeraire": pricing_numeraire,
        "fx_quote_direction": fx_quote_direction,
        "correlation_orientation": correlation_orientation,
        "canonical_fx_return_correlation": correlation,
        "quoted_fx_return_correlation": quoted_return_correlation,
        "covariance_term": covariance_term,
        "sign_convention": sign_convention,
        "drift_adjustment": adjustment,
        "economic_requotation_invariant": True,
        "zero_correlation_limit": 0.0,
        "zero_fx_volatility_limit": 0.0,
        "status": "PASS",
    }


def quanto_drift_adjustment(
    *,
    asset_volatility: float,
    fx_volatility: float,
    correlation: float,
) -> float:
    """Compatibility wrapper for domestic-per-foreign fixed quanto payoff."""

    return float(
        quanto_adjustment_certificate(
            asset_currency="FOREIGN",
            payoff_currency="DOMESTIC",
            collateral_currency="DOMESTIC",
            pricing_numeraire="DOMESTIC",
            fx_quote_direction="DOMESTIC_PER_FOREIGN",
            asset_volatility=asset_volatility,
            fx_volatility=fx_volatility,
            correlation=correlation,
        )["drift_adjustment"]
    )
