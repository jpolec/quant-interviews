#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

long double hit_upper(int state, int capital, long double probability_up) {
    if (capital <= 0 || state < 0 || state > capital ||
        !std::isfinite(probability_up) || probability_up <= 0.0L ||
        probability_up >= 1.0L) {
        throw std::invalid_argument("invalid unit-step gambler's-ruin state");
    }
    if (state == 0) return 0.0L;
    if (state == capital) return 1.0L;
    const long double probability_down = 1.0L - probability_up;
    const long double log_ratio = std::log(probability_down / probability_up);
    if (probability_up == 0.5L) {
        return static_cast<long double>(state) / capital;
    }
    if (log_ratio > 0.0L) {
        const long double numerator = -std::expm1(-state * log_ratio);
        const long double denominator = -std::expm1(-capital * log_ratio);
        return std::exp((state - capital) * log_ratio) * numerator / denominator;
    }
    return std::expm1(state * log_ratio) /
           std::expm1(capital * log_ratio);
}

bool close(long double actual, long double expected, long double tolerance) {
    return std::abs(actual - expected) <= tolerance;
}

int main() {
    const long double biased = hit_upper(4, 10, 0.55L);
    const long double fair = hit_upper(4, 10, 0.5L);
    const long double near_fair = hit_upper(400, 1000, 0.500000000001L);
    const long double boundary_low = hit_upper(0, 10, 0.7L);
    const long double boundary_high = hit_upper(10, 10, 0.7L);
    const bool passed =
        close(biased, 0.637585955343937L, 1.0e-12L) &&
        close(fair, 0.4L, 1.0e-15L) &&
        close(near_fair, 0.40000000048L, 1.0e-12L) &&
        boundary_low == 0.0L && boundary_high == 1.0L;
    std::cout << std::setprecision(15)
              << "biased=" << static_cast<double>(biased) << '\n'
              << "fair=" << static_cast<double>(fair) << '\n'
              << "near_fair=" << static_cast<double>(near_fair) << '\n'
              << "status=" << (passed ? "PASS" : "REJECT") << '\n';
    return passed ? 0 : 1;
}
