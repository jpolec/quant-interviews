#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

double normal_cdf(const double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
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

struct ImpliedVolCertificate {
    double implied_volatility;
    double repricing_residual;
    double lower_volatility;
    double upper_volatility;
    int iterations;
    bool converged;
};

ImpliedVolCertificate implied_volatility_call(
    const double target,
    const double spot,
    const double strike,
    const double rate,
    const double dividend,
    const double horizon,
    const double volatility_tolerance = 1.0e-12,
    const double price_tolerance = 1.0e-10,
    const int maximum_iterations = 200,
    const double maximum_volatility = 20.0
) {
    if (!std::isfinite(target) || target < 0.0
        || !std::isfinite(spot) || !std::isfinite(strike)
        || !std::isfinite(rate) || !std::isfinite(dividend)
        || !std::isfinite(horizon)
        || spot <= 0.0 || strike <= 0.0 || horizon <= 0.0
        || !std::isfinite(volatility_tolerance)
        || !std::isfinite(price_tolerance)
        || !std::isfinite(maximum_volatility)
        || volatility_tolerance <= 0.0 || price_tolerance <= 0.0
        || maximum_iterations <= 0 || maximum_volatility <= 0.0) {
        throw std::invalid_argument("invalid implied-volatility control");
    }
    const double discounted_spot = spot * std::exp(-dividend * horizon);
    const double discounted_strike = strike * std::exp(-rate * horizon);
    const double lower_price = std::max(discounted_spot - discounted_strike, 0.0);
    if (target <= lower_price || target >= discounted_spot) {
        throw std::domain_error("target is on or outside a price boundary");
    }

    double lower = 0.0;
    double upper = std::min(0.5, maximum_volatility);
    while (call_price(spot, strike, rate, dividend, upper, horizon) < target) {
        if (upper >= maximum_volatility) {
            throw std::runtime_error("NO_BRACKET");
        }
        upper = std::min(2.0 * upper, maximum_volatility);
    }

    double recovered = 0.5 * (lower + upper);
    double residual = call_price(
        spot, strike, rate, dividend, recovered, horizon
    ) - target;
    int iterations = 0;
    for (; iterations < maximum_iterations; ++iterations) {
        recovered = 0.5 * (lower + upper);
        residual = call_price(
            spot, strike, rate, dividend, recovered, horizon
        ) - target;
        if (residual < 0.0) {
            lower = recovered;
        } else {
            upper = recovered;
        }
        if (std::abs(residual) <= price_tolerance
            && upper - lower <= volatility_tolerance) {
            ++iterations;
            break;
        }
    }
    recovered = 0.5 * (lower + upper);
    residual = call_price(
        spot, strike, rate, dividend, recovered, horizon
    ) - target;
    const bool converged = (
        std::abs(residual) <= price_tolerance
        && upper - lower <= volatility_tolerance
    );
    return {
        recovered,
        residual,
        lower,
        upper,
        iterations,
        converged,
    };
}

}  // namespace

#ifndef QUANT_INTERVIEWS_CH06_IMPLIED_VOL_NO_MAIN
int main() {
    constexpr double spot = 100.0;
    constexpr double strike = 105.0;
    constexpr double rate = 0.03;
    constexpr double dividend = 0.01;
    constexpr double horizon = 1.2;
    constexpr double reference_volatility = 0.27;
    const double target = call_price(
        spot, strike, rate, dividend, reference_volatility, horizon
    );
    const auto result = implied_volatility_call(
        target, spot, strike, rate, dividend, horizon
    );
    const bool passed = (
        result.converged
        && std::abs(result.implied_volatility - reference_volatility) < 1e-10
        && std::abs(result.repricing_residual) < 1e-10
    );
    std::cout << std::setprecision(12)
              << "implied_vol=" << result.implied_volatility
              << " residual=" << result.repricing_residual
              << " bracket_width="
              << result.upper_volatility - result.lower_volatility
              << " iterations=" << result.iterations
              << " status=" << (passed ? "PASS" : "REJECT")
              << '\n';
    return passed ? 0 : 1;
}
#endif
