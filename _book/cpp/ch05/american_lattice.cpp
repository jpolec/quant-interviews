#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

double american_put_binomial(
    const double spot,
    const double strike,
    const double rate,
    const double dividend_yield,
    const double volatility,
    const double maturity,
    const int steps
) {
    if (!std::isfinite(spot) || !std::isfinite(strike) ||
        !std::isfinite(rate) || !std::isfinite(dividend_yield) ||
        !std::isfinite(volatility) || !std::isfinite(maturity) ||
        spot <= 0.0 || strike < 0.0 || volatility <= 0.0 ||
        maturity <= 0.0 || steps <= 0) {
        throw std::invalid_argument("invalid American lattice input");
    }
    const double dt = maturity / static_cast<double>(steps);
    const double up = std::exp(volatility * std::sqrt(dt));
    const double down = 1.0 / up;
    const double probability =
        (std::exp((rate - dividend_yield) * dt) - down) / (up - down);
    if (!(probability >= 0.0 && probability <= 1.0)) {
        throw std::invalid_argument("risk-neutral probability outside [0,1]");
    }
    const double discount = std::exp(-rate * dt);
    std::vector<double> values(static_cast<std::size_t>(steps + 1));
    for (int node = 0; node <= steps; ++node) {
        const double terminal_spot =
            spot * std::pow(up, node) * std::pow(down, steps - node);
        values[static_cast<std::size_t>(node)] =
            std::max(strike - terminal_spot, 0.0);
    }
    for (int step = steps - 1; step >= 0; --step) {
        for (int node = 0; node <= step; ++node) {
            const double state_spot =
                spot * std::pow(up, node) * std::pow(down, step - node);
            const double continuation = discount *
                (probability * values[static_cast<std::size_t>(node + 1)] +
                 (1.0 - probability) * values[static_cast<std::size_t>(node)]);
            values[static_cast<std::size_t>(node)] =
                std::max(strike - state_spot, continuation);
        }
    }
    if (!std::isfinite(values.front())) {
        throw std::runtime_error("non-finite American lattice value");
    }
    return values.front();
}

int main() {
    constexpr double spot = 100.0;
    constexpr double strike = 100.0;
    constexpr double rate = 0.03;
    constexpr double dividend_yield = 0.0;
    constexpr double volatility = 0.2;
    constexpr double maturity = 1.0;
    constexpr double frozen_reference = 6.743220479607686;
    constexpr double reference_tolerance = 1.0e-3;
    constexpr double pair_tolerance = 3.0e-3;
    try {
        const double even = american_put_binomial(
            spot, strike, rate, dividend_yield, volatility, maturity, 1000
        );
        const double odd = american_put_binomial(
            spot, strike, rate, dividend_yield, volatility, maturity, 1001
        );
        const double paired = 0.5 * (even + odd);
        const double reference_residual = paired - frozen_reference;
        const double pair_spread = std::abs(even - odd);
        const double intrinsic = std::max(strike - spot, 0.0);
        bool invalid_input_rejected = false;
        try {
            static_cast<void>(american_put_binomial(
                spot, strike, rate, dividend_yield, 0.0, maturity, 10
            ));
        } catch (const std::invalid_argument&) {
            invalid_input_rejected = true;
        }
        const bool passed =
            std::abs(reference_residual) <= reference_tolerance &&
            pair_spread <= pair_tolerance && paired >= intrinsic &&
            invalid_input_rejected;
        std::cout << std::setprecision(15)
                  << "even=" << even << " odd=" << odd
                  << " paired=" << paired
                  << " reference_residual=" << reference_residual
                  << " pair_spread=" << pair_spread
                  << " invalid_input_rejected="
                  << (invalid_input_rejected ? "true" : "false")
                  << " status=" << (passed ? "PASS" : "REJECT") << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "status=REJECT reason=" << error.what() << '\n';
        return 2;
    }
}
