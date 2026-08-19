#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

class OnlineStats {
public:
    void update(double x) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument("observation must be finite");
        }
        const std::size_t new_count = count_ + 1U;
        const double delta = x - mean_;
        const double new_mean = mean_ + delta / static_cast<double>(new_count);
        const double new_m2 = m2_ + delta * (x - new_mean);
        if (!std::isfinite(new_mean) || !std::isfinite(new_m2) || new_m2 < 0.0) {
            throw std::overflow_error("online update would create invalid state");
        }
        count_ = new_count;
        mean_ = new_mean;
        m2_ = new_m2;
    }

    [[nodiscard]] double mean() const noexcept { return mean_; }
    [[nodiscard]] std::size_t count() const noexcept { return count_; }
    [[nodiscard]] double variance() const {
        if (count_ < 2U) {
            throw std::domain_error(
                "sample variance requires at least two observations"
            );
        }
        const double result = m2_ / static_cast<double>(count_ - 1U);
        if (!std::isfinite(result) || result < 0.0) {
            throw std::overflow_error("online variance is invalid");
        }
        return result;
    }

private:
    std::size_t count_{0};
    double mean_{0.0};
    double m2_{0.0};
};

int main() {
    OnlineStats transactional;
    transactional.update(1.0e308);
    try {
        transactional.update(-1.0e308);
        return 2;
    } catch (const std::overflow_error&) {
        if (transactional.count() != 1U || transactional.mean() != 1.0e308) {
            return 3;
        }
    }
    OnlineStats stats;
    for (double x : std::vector<double>{1.0, 2.0, 3.0, 4.0}) {
        stats.update(x);
    }
    std::cout << std::fixed << std::setprecision(6)
              << stats.count() << " " << stats.mean() << " "
              << stats.variance() << "\n";
}
