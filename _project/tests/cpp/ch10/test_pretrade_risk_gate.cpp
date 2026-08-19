#define QI_CH10_NO_MAIN
#include "../../../../_book/cpp/ch10/pretrade_risk_gate.cpp"

#include <cassert>
#include <limits>

int main() {
    const Limits limits{
        100000, 5000000.0, 0.10, 50.0, 1000, "limits-v3",
        200000.0, 5000000.0, false
    };
    const OrderIntent accepted{
        100, 100.0, 100.0, 10000.0, 100, "limits-v3",
        100.0, 1000000.0, false
    };
    assert(check(accepted, limits).empty());

    const OrderIntent rejected{
        75000, 101.0, 100.0, 500000.0, 100, "limits-v2",
        250000.0, 4000000.0, true
    };
    const std::vector<std::string> expected{
        "STALE_LIMIT_SNAPSHOT", "SELF_TRADE_RISK", "MAX_NOTIONAL",
        "MAX_PARTICIPATION", "PRICE_COLLAR", "POSITION_LIMIT", "CREDIT_LIMIT"
    };
    assert(check(rejected, limits) == expected);

    auto unknown = accepted;
    unknown.projected_position = std::numeric_limits<double>::quiet_NaN();
    unknown.available_credit_notional = std::numeric_limits<double>::quiet_NaN();
    const std::vector<std::string> unknown_expected{
        "UNKNOWN_POSITION_STATE", "UNKNOWN_CREDIT_STATE"
    };
    assert(check(unknown, limits) == unknown_expected);
    return 0;
}
