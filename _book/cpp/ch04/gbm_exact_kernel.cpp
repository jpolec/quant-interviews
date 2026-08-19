#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

int main() {
    constexpr double spot = 100.0;
    constexpr double mu = 0.05;
    constexpr double sigma = 0.20;
    constexpr double horizon = 1.0;
    constexpr int paths = 200000;
    std::mt19937_64 rng(37);
    std::normal_distribution<double> normal(0.0, 1.0);
    double sum = 0.0;
    double sum_squared = 0.0;
    for (int i = 0; i < paths; ++i) {
        const double z = normal(rng);
        const double terminal = spot * std::exp((mu - 0.5 * sigma * sigma) * horizon + sigma * std::sqrt(horizon) * z);
        sum += terminal;
        sum_squared += terminal * terminal;
    }
    const double sample_mean = sum / static_cast<double>(paths);
    const double sample_variance =
        (sum_squared - static_cast<double>(paths) * sample_mean * sample_mean)
        / static_cast<double>(paths - 1);
    const double theoretical_mean = spot * std::exp(mu * horizon);
    const double standard_error =
        std::sqrt(sample_variance / static_cast<double>(paths));
    const bool passed =
        std::abs(sample_mean - theoretical_mean) <= 5.0 * standard_error;
    std::cout << std::setprecision(12)
              << "sample_mean=" << sample_mean
              << " theoretical_mean=" << theoretical_mean
              << " standard_error=" << standard_error
              << " status=" << (passed ? "PASS" : "REJECT") << "\n";
    return passed ? 0 : 1;
}
