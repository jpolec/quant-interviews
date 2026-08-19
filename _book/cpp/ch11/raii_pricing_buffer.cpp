#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

class PricingBuffer {
public:
    explicit PricingBuffer(std::size_t size)
        : size_(size), values_(std::make_unique<double[]>(size)) {
        if (size == 0) {
            throw std::invalid_argument("size must be positive");
        }
    }

    PricingBuffer(const PricingBuffer&) = delete;
    PricingBuffer& operator=(const PricingBuffer&) = delete;
    PricingBuffer(PricingBuffer&& other) noexcept
        : size_(std::exchange(other.size_, 0)), values_(std::move(other.values_)) {}
    PricingBuffer& operator=(PricingBuffer&& other) noexcept {
        if (this != &other) {
            values_ = std::move(other.values_);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }
    ~PricingBuffer() = default;

    std::span<double> span() noexcept { return {values_.get(), size_}; }
    std::span<const double> span() const noexcept { return {values_.get(), size_}; }

    void replace(std::span<const double> replacement) {
        if (replacement.size() != size_) {
            throw std::invalid_argument("replacement dimension mismatch");
        }
        auto candidate = std::make_unique<double[]>(size_);
        for (std::size_t i = 0; i < size_; ++i) {
            if (!std::isfinite(replacement[i])) {
                throw std::invalid_argument("replacement must be finite");
            }
            candidate[i] = replacement[i];
        }
        values_.swap(candidate);  // commit after complete validation
    }

private:
    std::size_t size_;
    std::unique_ptr<double[]> values_;
};

double discounted_value(std::span<const double> cashflows,
                        std::span<const double> discounts) {
    if (cashflows.size() != discounts.size()) {
        throw std::invalid_argument("dimension mismatch");
    }
    double value = 0.0;
    for (std::size_t i = 0; i < cashflows.size(); ++i) {
        if (!std::isfinite(cashflows[i]) || !std::isfinite(discounts[i])) {
            throw std::invalid_argument("inputs must be finite");
        }
        value += cashflows[i] * discounts[i];
        if (!std::isfinite(value)) {
            throw std::overflow_error("discounted value is non-finite");
        }
    }
    return value;
}

int main() {
    PricingBuffer cashflows(3);
    PricingBuffer discounts(3);
    cashflows.span()[0] = 2.0;
    cashflows.span()[1] = 2.0;
    cashflows.span()[2] = 102.0;
    discounts.span()[0] = 0.98;
    discounts.span()[1] = 0.95;
    discounts.span()[2] = 0.91;
    const std::vector<double> revised{2.0, 2.0, 102.0};
    cashflows.replace(revised);
    std::cout << std::fixed << discounted_value(cashflows.span(), discounts.span()) << '\n';
    return 0;
}
