#include <cmath>
#include <iostream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

struct Position {
    std::string currency;
    double quantity;
    double mark;
};

double nav(
    const std::map<std::string, double>& cash,
    std::span<const Position> positions,
    const std::map<std::string, double>& fx_to_base
) {
    std::map<std::string, double> amounts;
    for (const auto& [currency, amount] : cash) {
        if (currency.empty() || !std::isfinite(amount)) {
            throw std::invalid_argument("invalid cash balance");
        }
        amounts[currency] = amount;
    }
    for (const auto& position : positions) {
        if (
            position.currency.empty() ||
            !std::isfinite(position.quantity) ||
            !std::isfinite(position.mark) ||
            position.mark < 0.0
        ) {
            throw std::invalid_argument("invalid position or mark");
        }
        const double market_value = position.quantity * position.mark;
        if (!std::isfinite(market_value) ||
            !std::isfinite(amounts[position.currency] + market_value)) {
            throw std::overflow_error("position market value");
        }
        amounts[position.currency] += market_value;
    }
    double result = 0.0;
    for (const auto& [currency, amount] : amounts) {
        const auto found = fx_to_base.find(currency);
        if (
            found == fx_to_base.end() ||
            !std::isfinite(found->second) ||
            found->second <= 0.0
        ) {
            throw std::invalid_argument("missing or invalid FX rate");
        }
        const double converted = amount * found->second;
        if (!std::isfinite(converted) || !std::isfinite(result + converted)) {
            throw std::overflow_error("NAV accumulation");
        }
        result += converted;
    }
    return result;
}

#ifndef QJ_CH12_NO_MAIN
int main() {
    const std::map<std::string, double> cash{
        {"USD", 1000.0},
        {"EUR", 100.0},
    };
    const std::vector<Position> positions{
        {"USD", 10.0, 12.0},
        {"EUR", 5.0, 7.0},
    };
    const std::map<std::string, double> fx{
        {"USD", 1.0},
        {"EUR", 1.1},
    };
    const double value = nav(cash, positions, fx);
    const bool passed = std::abs(value - 1268.5) < 1e-12;
    std::cout << value << '\n';
    return passed ? 0 : 1;
}
#endif
