#define QUANT_INTERVIEWS_CH07_FIXED_INCOME_RISK_NO_MAIN
#include "../../../../_book/cpp/ch07/fixed_income_risk_kernel.cpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, const char* message) {
    bool rejected = false;
    try {
        callable();
    } catch (const Exception&) {
        rejected = true;
    }
    require(rejected, message);
}

long double independent_price(
    const std::vector<fixed_income::Cashflow>& cashflows,
    long double annual_yield) {
    long double value = 0.0L;
    for (const auto& cashflow : cashflows) {
        value += static_cast<long double>(cashflow.amount)
                 / std::pow(
                     1.0L + annual_yield,
                     static_cast<long double>(cashflow.time_years));
    }
    return value;
}

}  // namespace

int main() {
    using fixed_income::Bond;
    using fixed_income::Cashflow;
    const std::vector<Cashflow> cashflows = {
        {1.0, 4.0}, {2.0, 4.0}, {3.0, 4.0}, {4.0, 4.0}, {5.0, 104.0}};
    const Bond bond(cashflows);
    const long double reference_price = independent_price(cashflows, 0.05L);
    const long double bump = 1.0e-4L;
    const long double reference_dv01 =
        (independent_price(cashflows, 0.05L - bump)
         - independent_price(cashflows, 0.05L + bump))
        / 2.0L;
    require(
        std::abs(static_cast<long double>(bond.price(0.05)) - reference_price)
            <= 1.0e-12L,
        "price does not match the independent long-double reference");
    require(
        std::abs(static_cast<long double>(bond.dv01(0.05)) - reference_dv01)
            <= 1.0e-12L,
        "DV01 does not match the independent long-double reference");

    require_throws<std::invalid_argument>(
        [] { static_cast<void>(Bond({})); },
        "empty cashflows were accepted");
    require_throws<std::invalid_argument>(
        [] { static_cast<void>(Bond({{0.0, 1.0}})); },
        "non-positive cashflow time was accepted");
    require_throws<std::invalid_argument>(
        [] { static_cast<void>(Bond({{1.0, 1.0}, {1.0, 2.0}})); },
        "duplicate cashflow time was accepted");
    require_throws<std::invalid_argument>(
        [] { static_cast<void>(Bond({{2.0, 1.0}, {1.0, 2.0}})); },
        "descending cashflow times were accepted");
    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(Bond({{
                std::numeric_limits<double>::quiet_NaN(), 1.0}}));
        },
        "NaN cashflow time was accepted");
    require_throws<std::invalid_argument>(
        [] {
            static_cast<void>(Bond({{
                1.0, std::numeric_limits<double>::infinity()}}));
        },
        "infinite cashflow amount was accepted");
    require_throws<std::invalid_argument>(
        [&bond] {
            static_cast<void>(
                bond.price(std::numeric_limits<double>::quiet_NaN()));
        },
        "NaN yield was accepted");
    require_throws<std::invalid_argument>(
        [&bond] { static_cast<void>(bond.price(-1.0)); },
        "annual yield at the domain boundary was accepted");
    require_throws<std::invalid_argument>(
        [&bond] {
            static_cast<void>(
                bond.dv01(0.05, std::numeric_limits<double>::infinity()));
        },
        "infinite bump was accepted");
    require_throws<std::invalid_argument>(
        [&bond] { static_cast<void>(bond.dv01(0.05, 0.0)); },
        "zero bump was accepted");
    require_throws<std::invalid_argument>(
        [&bond] { static_cast<void>(bond.dv01(-0.99995, 1.0)); },
        "bump crossing the annual-yield domain was accepted");

    std::cout
        << "status=PASS reference=PASS finite_domain=PASS schedule_domain=PASS\n";
    return 0;
}
