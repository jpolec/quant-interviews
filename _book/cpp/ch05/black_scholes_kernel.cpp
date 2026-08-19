#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<double> solve_tridiagonal(
    std::vector<double> lower,
    std::vector<double> diagonal,
    std::vector<double> upper,
    std::vector<double> rhs
) {
    const std::size_t size = diagonal.size();
    for (std::size_t index = 1; index < size; ++index) {
        const double multiplier = lower[index - 1] / diagonal[index - 1];
        diagonal[index] -= multiplier * upper[index - 1];
        rhs[index] -= multiplier * rhs[index - 1];
    }
    std::vector<double> solution(size);
    solution.back() = rhs.back() / diagonal.back();
    for (std::size_t index = size - 1; index-- > 0;) {
        solution[index] =
            (rhs[index] - upper[index] * solution[index + 1])
            / diagonal[index];
    }
    return solution;
}

double normal_cdf(const double value) {
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

double analytic_call(
    const double spot,
    const double strike,
    const double rate,
    const double dividend,
    const double volatility,
    const double maturity
) {
    const double root_time = std::sqrt(maturity);
    const double d1 = (
        std::log(spot / strike)
        + (rate - dividend + 0.5 * volatility * volatility) * maturity
    ) / (volatility * root_time);
    const double d2 = d1 - volatility * root_time;
    return (
        spot * std::exp(-dividend * maturity) * normal_cdf(d1)
        - strike * std::exp(-rate * maturity) * normal_cdf(d2)
    );
}

}  // namespace

int main() {
    constexpr double spot = 100.0;
    constexpr double strike = 100.0;
    constexpr double rate = 0.03;
    constexpr double dividend = 0.01;
    constexpr double volatility = 0.20;
    constexpr double maturity = 1.0;
    constexpr double maximum_spot = 400.0;
    constexpr int spot_steps = 1000;
    constexpr int time_steps = 2000;

    const double spacing = maximum_spot / spot_steps;
    const double time_step = maturity / time_steps;
    std::vector<double> values(spot_steps + 1);
    for (int index = 0; index <= spot_steps; ++index) {
        values[index] = std::max(index * spacing - strike, 0.0);
    }

    for (int step = 0; step < time_steps; ++step) {
        const double old_time = step * time_step;
        const double new_time = (step + 1) * time_step;
        const int interior_size = spot_steps - 1;
        std::vector<double> lower(interior_size - 1);
        std::vector<double> diagonal(interior_size);
        std::vector<double> upper(interior_size - 1);
        std::vector<double> rhs(interior_size);

        for (int interior = 0; interior < interior_size; ++interior) {
            const double index = interior + 1.0;
            const double diffusion =
                0.5 * volatility * volatility * index * index;
            const double convection = 0.5 * (rate - dividend) * index;
            const double a = diffusion - convection;
            const double b = -2.0 * diffusion - rate;
            const double c = diffusion + convection;

            diagonal[interior] = 1.0 - 0.5 * time_step * b;
            rhs[interior] = (1.0 + 0.5 * time_step * b) * values[interior + 1];
            if (interior > 0) {
                lower[interior - 1] = -0.5 * time_step * a;
                rhs[interior] += 0.5 * time_step * a * values[interior];
            }
            if (interior + 1 < interior_size) {
                upper[interior] = -0.5 * time_step * c;
                rhs[interior] +=
                    0.5 * time_step * c * values[interior + 2];
            } else {
                const double old_boundary =
                    maximum_spot * std::exp(-dividend * old_time)
                    - strike * std::exp(-rate * old_time);
                const double new_boundary =
                    maximum_spot * std::exp(-dividend * new_time)
                    - strike * std::exp(-rate * new_time);
                rhs[interior] +=
                    0.5 * time_step * c * (old_boundary + new_boundary);
            }
        }

        const auto interior = solve_tridiagonal(
            lower,
            diagonal,
            upper,
            rhs
        );
        values.front() = 0.0;
        values.back() =
            maximum_spot * std::exp(-dividend * new_time)
            - strike * std::exp(-rate * new_time);
        std::copy(interior.begin(), interior.end(), values.begin() + 1);
    }

    const int left_index = static_cast<int>(spot / spacing);
    const double interpolation_weight =
        (spot - left_index * spacing) / spacing;
    const double pde_price =
        (1.0 - interpolation_weight) * values[left_index]
        + interpolation_weight * values[left_index + 1];
    const double reference = analytic_call(
        spot,
        strike,
        rate,
        dividend,
        volatility,
        maturity
    );
    const double error = std::abs(pde_price - reference);
    std::cout << std::setprecision(12)
              << "pde=" << pde_price
              << " analytic=" << reference
              << " error=" << error
              << " status=" << (error < 5e-4 ? "PASS" : "REJECT")
              << '\n';
    return error < 5e-4 ? 0 : 1;
}
