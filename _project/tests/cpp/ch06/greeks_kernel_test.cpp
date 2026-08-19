#define QUANT_INTERVIEWS_CH06_GREEKS_KERNEL_NO_MAIN
#include "../../../../_book/cpp/ch06/greeks_kernel.cpp"

#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(const bool condition, const char* const message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void require_invalid(Callable&& callable, const char* const message) {
    bool rejected = false;
    try {
        callable();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

}  // namespace

int main() {
    const double price = call_price(100.0, 100.0, 0.02, 0.0, 0.2, 1.0);
    require(price > 0.0 && price < 100.0, "regular price failed");
    require(normal_pdf(0.0) > 0.0, "normal density failed");
    require(close(price, price, 1.0e-12), "relative comparison failed");
    require_invalid(
        [] { static_cast<void>(call_price(0.0, 100.0, 0.02, 0.0, 0.2, 1.0)); },
        "zero spot was accepted");
    require_invalid(
        [] { static_cast<void>(call_price(100.0, 100.0, 0.02, 0.0, 0.0, 1.0)); },
        "zero volatility was accepted");
    require_invalid(
        [] {
            static_cast<void>(call_price(
                100.0,
                100.0,
                0.02,
                0.0,
                std::numeric_limits<double>::quiet_NaN(),
                1.0));
        },
        "NaN volatility was accepted");
    std::cout << "status=PASS regular=PASS invalid_domain=REJECT\n";
    return 0;
}
