#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

struct VirtualPricer {
    virtual ~VirtualPricer() = default;
    virtual double price(double spot) const noexcept = 0;
};

struct VirtualLinear final : VirtualPricer {
    explicit VirtualLinear(double multiplier) : multiplier_(multiplier) {}
    double price(double spot) const noexcept override { return multiplier_ * spot; }
    double multiplier_;
};

template <typename Derived>
struct StaticPricer {
    double price(double spot) const noexcept {
        return static_cast<const Derived*>(this)->price_impl(spot);
    }
};

struct StaticLinear final : StaticPricer<StaticLinear> {
    explicit StaticLinear(double multiplier) : multiplier_(multiplier) {}
    double price_impl(double spot) const noexcept { return multiplier_ * spot; }
    double multiplier_;
};

int main() {
    constexpr int iterations = 100'000;
    auto dynamic = std::make_unique<VirtualLinear>(1.25);
    StaticLinear statik(1.25);
    double dynamic_total = 0.0;
    double static_total = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const double spot = 100.0 + i * 1e-5;
        dynamic_total += dynamic->price(spot);
        static_total += statik.price(spot);
    }
    if (std::abs(dynamic_total - static_total) > 1e-9) return 1;
    std::cout << dynamic_total << ' ' << static_total << '\n';
}
