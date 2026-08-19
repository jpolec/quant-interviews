#define main ch11_risk_kernel_example_main
#include "../../../../_book/cpp/ch11/risk_kernel.cpp"
#undef main

#include <cassert>

int main() {
    const std::vector<double> indefinite{1.0, 2.0, 2.0, 1.0};
    bool indefinite_rejected = false;
    try {
        (void)portfolio_variance({1.0, 0.0}, {indefinite.data(), 2, 2, 2});
    } catch (const std::invalid_argument&) {
        indefinite_rejected = true;
    }
    assert(indefinite_rejected);

    const std::vector<double> singular_psd{1.0, 1.0, 1.0, 1.0};
    assert(portfolio_variance({1.0, 1.0}, {singular_psd.data(), 2, 2, 2}) == 4.0);

    const double maximum = std::numeric_limits<double>::max();
    const std::vector<double> overflowing{maximum};
    bool overflow_rejected = false;
    try {
        (void)portfolio_variance({maximum}, {overflowing.data(), 1, 1, 1});
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }
    assert(overflow_rejected);
}
