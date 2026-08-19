#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Position {
    std::string instrument;
    double quantity;
};

void validate_position(const Position& position) {
    if (position.instrument.empty() || !std::isfinite(position.quantity) ||
        position.quantity < 0.0) {
        throw std::invalid_argument("position");
    }
}

double checked_product(double left, double right, const char* field) {
    const double result = left * right;
    if (!std::isfinite(result)) {
        throw std::overflow_error(field);
    }
    return result;
}

struct ActionResult {
    Position parent;
    std::vector<Position> children;
    double cash_receivable;
};

ActionResult apply_split(const Position& parent, double ratio) {
    validate_position(parent);
    if (!std::isfinite(ratio) || ratio <= 0.0) {
        throw std::invalid_argument("ratio");
    }
    return {{parent.instrument, checked_product(parent.quantity, ratio, "split")}, {}, 0.0};
}

ActionResult apply_dividend(
    const Position& parent,
    double cash_per_share
) {
    validate_position(parent);
    if (!std::isfinite(cash_per_share) || cash_per_share < 0.0) {
        throw std::invalid_argument("cash_per_share");
    }
    // Raw ex-date market prices are not modified by this posting.
    return {parent, {}, checked_product(parent.quantity, cash_per_share, "dividend")};
}

ActionResult apply_spinoff(
    const Position& parent,
    const std::string& child,
    double child_ratio
) {
    validate_position(parent);
    if (child.empty() || !std::isfinite(child_ratio) || child_ratio <= 0.0) {
        throw std::invalid_argument("child action");
    }
    return {parent, {{child, checked_product(parent.quantity, child_ratio, "spinoff")}}, 0.0};
}

#ifndef QJ_CH12_NO_MAIN
int main() {
    const Position parent{"PARENT", 100.0};
    const auto split = apply_split(parent, 4.0);
    const auto dividend = apply_dividend(parent, 1.25);
    const auto spinoff = apply_spinoff(parent, "CHILD", 0.5);
    const double before_split = parent.quantity * 80.0;
    const double after_split = split.parent.quantity * 20.0;
    const bool passed =
        std::abs(before_split - after_split) < 1e-12 &&
        std::abs(dividend.cash_receivable - 125.0) < 1e-12 &&
        spinoff.children.size() == 1U &&
        std::abs(spinoff.children.front().quantity - 50.0) < 1e-12;
    std::cout << (passed ? "PASS" : "REJECT") << '\n';
    return passed ? 0 : 1;
}
#endif
