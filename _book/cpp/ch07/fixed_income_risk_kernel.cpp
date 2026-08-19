#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fixed_income {
constexpr double kBasisPoint = 1.0e-4;

struct Cashflow {
    double time_years;
    double amount;
};

class Bond {
  public:
    explicit Bond(std::vector<Cashflow> cashflows) : cashflows_(std::move(cashflows)) {
        if (cashflows_.empty()) {
            throw std::invalid_argument("cashflows must not be empty");
        }
        double previous_time = 0.0;
        for (const auto& cashflow : cashflows_) {
            if (!std::isfinite(cashflow.time_years)
                || !std::isfinite(cashflow.amount)) {
                throw std::invalid_argument("cashflows must be finite");
            }
            if (cashflow.time_years <= previous_time) {
                throw std::invalid_argument(
                    "cashflow times must be positive and strictly increasing");
            }
            previous_time = cashflow.time_years;
        }
    }

    [[nodiscard]] double price(double annual_yield) const {
        if (!std::isfinite(annual_yield) || annual_yield <= -1.0) {
            throw std::invalid_argument("yield must be finite and exceed -100%");
        }
        double value = 0.0;
        for (const auto& cf : cashflows_) {
            const double present_value =
                cf.amount / std::pow(1.0 + annual_yield, cf.time_years);
            if (!std::isfinite(present_value)) {
                throw std::overflow_error("cashflow present value is non-finite");
            }
            value += present_value;
            if (!std::isfinite(value)) {
                throw std::overflow_error("bond price is non-finite");
            }
        }
        return value;
    }

    [[nodiscard]] double dv01(double annual_yield, double bump_bp = 1.0) const {
        if (!std::isfinite(bump_bp) || bump_bp <= 0.0) {
            throw std::invalid_argument("bump_bp must be finite and positive");
        }
        const double bump = bump_bp * kBasisPoint;
        if (!std::isfinite(annual_yield) || annual_yield - bump <= -1.0) {
            throw std::invalid_argument(
                "yield bump must remain inside the annual-compounding domain");
        }
        const double result =
            (price(annual_yield - bump) - price(annual_yield + bump))
            / (2.0 * bump_bp);
        if (!std::isfinite(result)) {
            throw std::overflow_error("DV01 is non-finite");
        }
        return result;
    }

  private:
    std::vector<Cashflow> cashflows_;
};
}  // namespace fixed_income

#ifndef QUANT_INTERVIEWS_CH07_FIXED_INCOME_RISK_NO_MAIN
int main() {
    using fixed_income::Bond;
    using fixed_income::Cashflow;
    const Bond bond({{1.0, 4.0}, {2.0, 4.0}, {3.0, 4.0}, {4.0, 4.0}, {5.0, 104.0}});
    const double price = bond.price(0.05);
    const double dv01 = bond.dv01(0.05);
    constexpr double expected_price = 95.67052332936916;
    constexpr double expected_dv01 = 0.04209796989763248;
    constexpr double absolute_tolerance = 5.0e-12;
    const double price_residual = std::abs(price - expected_price);
    const double dv01_residual = std::abs(dv01 - expected_dv01);
    const bool passed = price_residual <= absolute_tolerance
                        && dv01_residual <= absolute_tolerance;
    std::cout << std::fixed << std::setprecision(8)
              << "status=" << (passed ? "PASS" : "REJECT") << "\n"
              << "price=" << price << "\n"
              << "dv01=" << dv01 << "\n"
              << std::scientific << std::setprecision(12)
              << "price_residual=" << price_residual << "\n"
              << "dv01_residual=" << dv01_residual << "\n";
    return passed ? 0 : 1;
}
#endif
