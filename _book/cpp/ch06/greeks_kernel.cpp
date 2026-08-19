#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

double normal_cdf(const double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double normal_pdf(const double x) {
    constexpr double two_pi = 6.28318530717958647692;
    return std::exp(-0.5 * x * x) / std::sqrt(two_pi);
}

double call_price(
    const double spot,
    const double strike,
    const double rate,
    const double dividend,
    const double volatility,
    const double horizon
) {
    if (!std::isfinite(spot) || !std::isfinite(strike)
        || !std::isfinite(rate) || !std::isfinite(dividend)
        || !std::isfinite(volatility) || !std::isfinite(horizon)
        || spot <= 0.0 || strike <= 0.0 || volatility <= 0.0
        || horizon <= 0.0) {
        throw std::invalid_argument("invalid Black-Scholes input");
    }
    const double root_time = std::sqrt(horizon);
    const double d1 = (
        std::log(spot / strike)
        + (rate - dividend + 0.5 * volatility * volatility) * horizon
    ) / (volatility * root_time);
    const double d2 = d1 - volatility * root_time;
    return (
        spot * std::exp(-dividend * horizon) * normal_cdf(d1)
        - strike * std::exp(-rate * horizon) * normal_cdf(d2)
    );
}

bool close(const double left, const double right, const double tolerance) {
    return std::abs(left - right)
        <= tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

}  // namespace

#ifndef QUANT_INTERVIEWS_CH06_GREEKS_KERNEL_NO_MAIN
int main() {
    constexpr double spot = 100.0;
    constexpr double strike = 100.0;
    constexpr double rate = 0.03;
    constexpr double dividend = 0.01;
    constexpr double volatility = 0.20;
    constexpr double horizon = 1.0;
    constexpr double spot_bump = 0.01;
    constexpr double vol_bump = 0.0001;
    constexpr double rate_bump = 0.000001;

    const double root_time = std::sqrt(horizon);
    const double d1 = (
        std::log(spot / strike)
        + (rate - dividend + 0.5 * volatility * volatility) * horizon
    ) / (volatility * root_time);
    const double d2 = d1 - volatility * root_time;
    const double delta = std::exp(-dividend * horizon) * normal_cdf(d1);
    const double gamma = (
        std::exp(-dividend * horizon) * normal_pdf(d1)
        / (spot * volatility * root_time)
    );
    const double vega = (
        spot * std::exp(-dividend * horizon) * normal_pdf(d1) * root_time
    );
    const double rho = (
        strike * horizon * std::exp(-rate * horizon) * normal_cdf(d2)
    );

    const double price_up = call_price(
        spot + spot_bump, strike, rate, dividend, volatility, horizon
    );
    const double price = call_price(
        spot, strike, rate, dividend, volatility, horizon
    );
    const double price_down = call_price(
        spot - spot_bump, strike, rate, dividend, volatility, horizon
    );
    const double bump_delta = (price_up - price_down) / (2.0 * spot_bump);
    const double bump_gamma = (
        price_up - 2.0 * price + price_down
    ) / (spot_bump * spot_bump);
    const double bump_vega = (
        call_price(
            spot, strike, rate, dividend, volatility + vol_bump, horizon
        )
        - call_price(
            spot, strike, rate, dividend, volatility - vol_bump, horizon
        )
    ) / (2.0 * vol_bump);
    const double bump_rho = (
        call_price(
            spot, strike, rate + rate_bump, dividend, volatility, horizon
        )
        - call_price(
            spot, strike, rate - rate_bump, dividend, volatility, horizon
        )
    ) / (2.0 * rate_bump);

    const bool passed = (
        close(delta, bump_delta, 1e-7)
        && close(gamma, bump_gamma, 5e-6)
        && close(vega, bump_vega, 1e-7)
        && close(rho, bump_rho, 1e-7)
    );
    std::cout << std::setprecision(12)
              << "delta=" << delta
              << " bump_delta=" << bump_delta
              << " gamma=" << gamma
              << " bump_gamma=" << bump_gamma
              << " vega_per_vol_point=" << vega * 0.01
              << " rho_per_basis_point=" << rho * 0.0001
              << " status=" << (passed ? "PASS" : "REJECT")
              << '\n';
    return passed ? 0 : 1;
}
#endif
