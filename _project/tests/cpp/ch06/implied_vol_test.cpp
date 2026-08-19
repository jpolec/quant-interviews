#define QUANT_INTERVIEWS_CH06_IMPLIED_VOL_NO_MAIN
#include "../../../../_book/cpp/ch06/implied_vol.cpp"

#include <iostream>
#include <stdexcept>

namespace {

void require(const bool condition, const char* const message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, const char* const message) {
    bool rejected = false;
    try {
        callable();
    } catch (const Exception&) {
        rejected = true;
    }
    require(rejected, message);
}

}  // namespace

int main() {
    const double target = call_price(100.0, 105.0, 0.03, 0.01, 0.27, 1.2);
    const auto converged = implied_volatility_call(
        target, 100.0, 105.0, 0.03, 0.01, 1.2
    );
    require(converged.converged, "regular inversion did not converge");
    require(
        std::abs(converged.repricing_residual) <= 1.0e-10,
        "returned residual was stale");

    const auto exhausted = implied_volatility_call(
        target,
        100.0,
        105.0,
        0.03,
        0.01,
        1.2,
        1.0e-15,
        1.0e-15,
        1
    );
    require(!exhausted.converged, "one iteration falsely converged");
    require(
        exhausted.repricing_residual
            == call_price(
                100.0,
                105.0,
                0.03,
                0.01,
                exhausted.implied_volatility,
                1.2)
                - target,
        "exhausted certificate residual is stale");
    require_throws<std::domain_error>(
        [] {
            static_cast<void>(implied_volatility_call(
                100.0, 100.0, 105.0, 0.03, 0.01, 1.2));
        },
        "upper-bound quote was accepted");
    require_throws<std::invalid_argument>(
        [&target] {
            static_cast<void>(implied_volatility_call(
                target, 100.0, 105.0, 0.03, 0.01, 1.2, -1.0));
        },
        "negative tolerance was accepted");
    require_throws<std::invalid_argument>(
        [&target] {
            static_cast<void>(implied_volatility_call(
                target, 0.0, 105.0, 0.03, 0.01, 1.2));
        },
        "zero spot was accepted");
    std::cout << "status=PASS convergence=PASS stale_certificate=REJECT"
              << " invalid_quote=REJECT\n";
    return 0;
}
