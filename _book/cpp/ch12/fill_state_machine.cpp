#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

enum class Side { buy, sell };
enum class State { acknowledged, partially_filled, filled, cancelled };

struct Order {
    Side side;
    State state;
    std::int64_t quantity_lots;
    std::int64_t filled_lots;
    std::int64_t closed_lots;
    double limit_price;
    double cumulative_value;
};

void validate_order(const Order& order) {
    if (order.quantity_lots <= 0 || order.filled_lots < 0 ||
        order.closed_lots < 0 ||
        order.filled_lots > order.quantity_lots ||
        order.closed_lots > order.quantity_lots - order.filled_lots ||
        !std::isfinite(order.limit_price) || order.limit_price <= 0.0 ||
        !std::isfinite(order.cumulative_value) ||
        order.cumulative_value < 0.0) {
        throw std::invalid_argument("invalid order state");
    }
}

void apply_fill(
    Order& order,
    std::int64_t quantity_lots,
    double price
) {
    validate_order(order);
    if (
        order.state != State::acknowledged &&
        order.state != State::partially_filled
    ) {
        throw std::invalid_argument("order is not fillable");
    }
    if (
        quantity_lots <= 0 ||
        quantity_lots > order.quantity_lots - order.filled_lots -
                            order.closed_lots ||
        !std::isfinite(price) ||
        price <= 0.0
    ) {
        throw std::invalid_argument("invalid fill");
    }
    if (
        (order.side == Side::buy && price > order.limit_price) ||
        (order.side == Side::sell && price < order.limit_price)
    ) {
        throw std::invalid_argument("limit violation");
    }
    order.filled_lots += quantity_lots;
    const double increment = static_cast<double>(quantity_lots) * price;
    if (!std::isfinite(increment) ||
        !std::isfinite(order.cumulative_value + increment)) {
        throw std::overflow_error("cumulative execution value");
    }
    order.cumulative_value += increment;
    order.state = order.filled_lots + order.closed_lots == order.quantity_lots
        ? State::filled
        : State::partially_filled;
}

#ifndef QJ_CH12_NO_MAIN
int main() {
    Order order{
        Side::buy,
        State::acknowledged,
        100,
        0,
        0,
        11.0,
        0.0,
    };
    apply_fill(order, 40, 10.0);
    apply_fill(order, 60, 11.0);
    const auto average = order.cumulative_value /
                         static_cast<double>(order.filled_lots);
    const bool conserved =
        order.filled_lots + order.closed_lots == order.quantity_lots;
    const bool passed =
        order.state == State::filled &&
        conserved &&
        std::abs(average - 10.6) < 1e-12;
    std::cout << (passed ? "PASS" : "REJECT") << '\n';
    return passed ? 0 : 1;
}
#endif
