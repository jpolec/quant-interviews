#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

double regression_slope(
    const std::array<double, 5>& x,
    const std::array<double, 5>& y
) {
    double mean_x = 0.0;
    double mean_y = 0.0;
    for (std::size_t index = 0; index < x.size(); ++index) {
        mean_x += x[index];
        mean_y += y[index];
    }
    mean_x /= static_cast<double>(x.size());
    mean_y /= static_cast<double>(y.size());
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t index = 0; index < x.size(); ++index) {
        numerator += (x[index] - mean_x) * (y[index] - mean_y);
        denominator += (x[index] - mean_x) * (x[index] - mean_x);
    }
    return numerator / denominator;
}

int main() {
    constexpr double x0 = 1.0;
    constexpr double mu = 0.03;
    constexpr double sigma = 0.20;
    constexpr int paths = 50000;
    constexpr int finest_steps = 128;
    constexpr std::array<int, 5> step_counts{8, 16, 32, 64, 128};
    std::array<double, 5> squared_euler_errors{};
    std::array<double, 5> squared_milstein_errors{};
    std::mt19937_64 rng(41);
    std::normal_distribution<double> normal(0.0, 1.0);

    for (int path = 0; path < paths; ++path) {
        std::array<double, finest_steps> fine_increments{};
        double terminal_brownian = 0.0;
        for (double& increment : fine_increments) {
            increment = normal(rng) / std::sqrt(finest_steps);
            terminal_brownian += increment;
        }
        const double exact =
            x0 * std::exp(mu - 0.5 * sigma * sigma + sigma * terminal_brownian);
        for (std::size_t level = 0; level < step_counts.size(); ++level) {
            const int steps = step_counts[level];
            const int block = finest_steps / steps;
            const double dt = 1.0 / static_cast<double>(steps);
            double euler = x0;
            double milstein = x0;
            for (int step = 0; step < steps; ++step) {
                double d_w = 0.0;
                for (int offset = 0; offset < block; ++offset) {
                    d_w += fine_increments[step * block + offset];
                }
                euler += mu * euler * dt + sigma * euler * d_w;
                milstein +=
                    mu * milstein * dt
                    + sigma * milstein * d_w
                    + 0.5 * sigma * sigma * milstein * (d_w * d_w - dt);
            }
            squared_euler_errors[level] += (euler - exact) * (euler - exact);
            squared_milstein_errors[level] +=
                (milstein - exact) * (milstein - exact);
        }
    }

    std::array<double, 5> log_step_size{};
    std::array<double, 5> log_euler_error{};
    std::array<double, 5> log_milstein_error{};
    for (std::size_t level = 0; level < step_counts.size(); ++level) {
        const double step_size =
            1.0 / static_cast<double>(step_counts[level]);
        const double euler_rmse =
            std::sqrt(squared_euler_errors[level] / paths);
        const double milstein_rmse =
            std::sqrt(squared_milstein_errors[level] / paths);
        log_step_size[level] = std::log(step_size);
        log_euler_error[level] = std::log(euler_rmse);
        log_milstein_error[level] = std::log(milstein_rmse);
    }
    const double euler_slope =
        regression_slope(log_step_size, log_euler_error);
    const double milstein_slope =
        regression_slope(log_step_size, log_milstein_error);
    const bool passed =
        euler_slope >= 0.4 && euler_slope <= 0.65
        && milstein_slope >= 0.85 && milstein_slope <= 1.15;
    std::cout << std::setprecision(10)
              << "euler_slope=" << euler_slope
              << " milstein_slope=" << milstein_slope
              << " status=" << (passed ? "PASS" : "REJECT") << "\n";
    return passed ? 0 : 1;
}
