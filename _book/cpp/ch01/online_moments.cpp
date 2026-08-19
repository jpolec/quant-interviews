#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

struct OnlineMoments {
    std::size_t count{0};
    double mean{0.0};
    double centered_sum_of_squares{0.0};

    void update(const double value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("observation must be finite");
        }
        if (count == std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("observation count overflow");
        }

        const std::size_t next_count = count + 1;
        const double delta = value - mean;
        const double next_mean =
            mean + delta / static_cast<double>(next_count);
        const double updated_delta = value - next_mean;
        const double next_centered_sum =
            centered_sum_of_squares + delta * updated_delta;
        if (!std::isfinite(next_mean)
            || !std::isfinite(next_centered_sum)) {
            throw std::overflow_error(
                "online moment update produced a non-finite state"
            );
        }

        count = next_count;
        mean = next_mean;
        centered_sum_of_squares = next_centered_sum;
    }

    [[nodiscard]] double sample_variance() const {
        if (count < 2) {
            throw std::domain_error(
                "sample variance requires at least two observations"
            );
        }
        const double result = centered_sum_of_squares
            / static_cast<double>(count - 1);
        if (!std::isfinite(result)) {
            throw std::overflow_error("sample variance is non-finite");
        }
        return result;
    }
};

#ifndef QUANT_INTERVIEWS_CH01_NO_MAIN
int main() {
    const std::vector<double> observations{1.0, 2.0, 3.0, 4.0};
    OnlineMoments moments;
    for (const double observation : observations) {
        moments.update(observation);
    }

    bool nonfinite_rejected = false;
    try {
        moments.update(std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument&) {
        nonfinite_rejected = true;
    }

    bool insufficient_count_rejected = false;
    try {
        OnlineMoments empty;
        static_cast<void>(empty.sample_variance());
    } catch (const std::domain_error&) {
        insufficient_count_rejected = true;
    }

    const bool passed = nonfinite_rejected
        && insufficient_count_rejected
        && moments.count == observations.size()
        && std::abs(moments.mean - 2.5) < 1e-15
        && std::abs(moments.sample_variance() - (5.0 / 3.0)) < 1e-12;
    std::cout << std::setprecision(17)
              << "count=" << moments.count
              << " mean=" << moments.mean
              << " sample_variance=" << moments.sample_variance()
              << " nonfinite_rejected="
              << (nonfinite_rejected ? "PASS" : "FAIL")
              << " insufficient_count_rejected="
              << (insufficient_count_rejected ? "PASS" : "FAIL")
              << " status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
#endif
