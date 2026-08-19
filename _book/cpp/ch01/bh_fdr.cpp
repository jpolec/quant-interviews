#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

std::size_t benjamini_hochberg_discoveries(
    std::vector<double> p_values,
    const double alpha
) {
    if (!std::isfinite(alpha) || alpha <= 0.0 || alpha > 1.0) {
        throw std::invalid_argument("alpha must be finite and in (0, 1]");
    }
    if (p_values.empty()) {
        throw std::invalid_argument("p_values must not be empty");
    }
    for (const double p_value : p_values) {
        if (!std::isfinite(p_value)
            || p_value < 0.0
            || p_value > 1.0) {
            throw std::invalid_argument(
                "every p-value must be finite and in [0, 1]"
            );
        }
    }

    std::sort(p_values.begin(), p_values.end());
    std::size_t discoveries = 0;
    for (std::size_t index = 0; index < p_values.size(); ++index) {
        const double rank = static_cast<double>(index + 1);
        const double threshold =
            alpha * rank / static_cast<double>(p_values.size());
        if (p_values[index] <= threshold) {
            discoveries = index + 1;
        }
    }
    return discoveries;
}

#ifndef QUANT_INTERVIEWS_CH01_NO_MAIN
int main() {
    const std::vector<double> p_values{0.001, 0.01, 0.20, 0.50};
    const std::size_t discoveries =
        benjamini_hochberg_discoveries(p_values, 0.05);

    bool invalid_alpha_rejected = false;
    try {
        static_cast<void>(benjamini_hochberg_discoveries(p_values, 0.0));
    } catch (const std::invalid_argument&) {
        invalid_alpha_rejected = true;
    }

    bool invalid_p_value_rejected = false;
    try {
        static_cast<void>(benjamini_hochberg_discoveries(
            {0.01, std::numeric_limits<double>::quiet_NaN()},
            0.05
        ));
    } catch (const std::invalid_argument&) {
        invalid_p_value_rejected = true;
    }

    const bool passed = discoveries == 2
        && invalid_alpha_rejected
        && invalid_p_value_rejected;
    std::cout << "discoveries=" << discoveries
              << " invalid_alpha_rejected="
              << (invalid_alpha_rejected ? "PASS" : "FAIL")
              << " invalid_p_value_rejected="
              << (invalid_p_value_rejected ? "PASS" : "FAIL")
              << " status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
#endif
