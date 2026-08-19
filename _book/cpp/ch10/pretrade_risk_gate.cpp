#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Limits {
    std::int64_t max_order_quantity;
    double max_notional;
    double max_participation;
    double price_collar_bps;
    std::int64_t max_market_volume_age_ms;
    std::string limit_snapshot_version;
    double max_position_after_order;
    double max_credit_notional;
    bool kill_switch_active{false};
};

struct OrderIntent {
    std::int64_t quantity;
    double price;
    double reference_price;
    double forecast_market_volume;
    std::int64_t market_volume_age_ms;
    std::string limit_snapshot_version;
    double projected_position;
    double available_credit_notional;
    bool self_trade_risk;
};

bool valid_limits(const Limits& limits) {
    return limits.max_order_quantity > 0
        && std::isfinite(limits.max_notional) && limits.max_notional > 0.0
        && std::isfinite(limits.max_participation)
        && limits.max_participation > 0.0 && limits.max_participation <= 1.0
        && std::isfinite(limits.price_collar_bps) && limits.price_collar_bps >= 0.0
        && limits.max_market_volume_age_ms >= 0
        && !limits.limit_snapshot_version.empty()
        && std::isfinite(limits.max_position_after_order)
        && limits.max_position_after_order >= 0.0
        && std::isfinite(limits.max_credit_notional)
        && limits.max_credit_notional > 0.0;
}

std::vector<std::string> check(const OrderIntent& order, const Limits& limits) {
    std::vector<std::string> breaches;
    if (!valid_limits(limits)) {
        breaches.emplace_back("INVALID_LIMITS");
        return breaches;
    }
    if (order.quantity <= 0 || !std::isfinite(order.price) || order.price <= 0.0
        || !std::isfinite(order.reference_price) || order.reference_price <= 0.0) {
        breaches.emplace_back("INVALID_ORDER_FIELDS");
        return breaches;
    }
    if (limits.kill_switch_active) breaches.emplace_back("KILL_SWITCH_ACTIVE");
    if (order.limit_snapshot_version != limits.limit_snapshot_version) {
        breaches.emplace_back("STALE_LIMIT_SNAPSHOT");
    }
    if (order.self_trade_risk) breaches.emplace_back("SELF_TRADE_RISK");
    if (order.quantity > limits.max_order_quantity) {
        breaches.emplace_back("MAX_ORDER_QUANTITY");
    }
    const auto notional = static_cast<double>(order.quantity) * order.price;
    if (!std::isfinite(notional) || notional > limits.max_notional) {
        breaches.emplace_back("MAX_NOTIONAL");
    }
    if (!std::isfinite(order.forecast_market_volume)
        || order.forecast_market_volume <= 0.0
        || order.market_volume_age_ms < 0
        || order.market_volume_age_ms > limits.max_market_volume_age_ms) {
        breaches.emplace_back("INVALID_OR_STALE_MARKET_VOLUME");
    } else {
        const auto participation =
            static_cast<double>(order.quantity) / order.forecast_market_volume;
        if (participation > limits.max_participation) {
            breaches.emplace_back("MAX_PARTICIPATION");
        }
    }
    const auto deviation_bps =
        std::abs(order.price / order.reference_price - 1.0) * 10000.0;
    if (deviation_bps > limits.price_collar_bps) {
        breaches.emplace_back("PRICE_COLLAR");
    }
    if (!std::isfinite(order.projected_position)) {
        breaches.emplace_back("UNKNOWN_POSITION_STATE");
    } else if (std::abs(order.projected_position) > limits.max_position_after_order) {
        breaches.emplace_back("POSITION_LIMIT");
    }
    if (!std::isfinite(order.available_credit_notional)
        || order.available_credit_notional < 0.0) {
        breaches.emplace_back("UNKNOWN_CREDIT_STATE");
    } else if (notional > order.available_credit_notional
               || notional > limits.max_credit_notional) {
        breaches.emplace_back("CREDIT_LIMIT");
    }
    return breaches;
}

#ifndef QI_CH10_NO_MAIN
int main() {
    const Limits limits{
        100000, 5000000.0, 0.10, 50.0, 1000, "limits-v3",
        200000.0, 5000000.0, false
    };
    const OrderIntent order{
        75000, 101.0, 100.0, 500000.0, 200, "limits-v3",
        250000.0, 4000000.0, false
    };
    const auto breaches = check(order, limits);
    const std::vector<std::string> expected{
        "MAX_NOTIONAL", "MAX_PARTICIPATION", "PRICE_COLLAR",
        "POSITION_LIMIT", "CREDIT_LIMIT"
    };
    const auto pass = breaches == expected;
    std::cout << "breaches=" << breaches.size();
    for (const auto& breach : breaches) std::cout << " " << breach;
    std::cout << " status=" << (pass ? "PASS" : "REJECT") << '\n';
    return pass ? 0 : 1;
}
#endif
