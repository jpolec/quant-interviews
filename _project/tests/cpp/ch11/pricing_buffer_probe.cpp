#define main ch11_pricing_buffer_example_main
#include "../../../../_book/cpp/ch11/raii_pricing_buffer.cpp"
#undef main

#include <cassert>
#include <limits>

int main() {
    PricingBuffer source(2);
    source.span()[0] = 2.0;
    source.span()[1] = 3.0;
    PricingBuffer moved(std::move(source));
    assert(source.span().empty());
    assert(moved.span().size() == 2);

    PricingBuffer assigned(1);
    assigned = std::move(moved);
    assert(moved.span().empty());
    assert(assigned.span().size() == 2);

    const std::vector<double> valid{4.0, 5.0};
    assigned.replace(valid);
    assert(assigned.span()[0] == 4.0 && assigned.span()[1] == 5.0);

    const double maximum = std::numeric_limits<double>::max();
    const std::vector<double> cashflows{maximum, maximum};
    const std::vector<double> discounts{maximum, maximum};
    bool overflow_rejected = false;
    try {
        (void)discounted_value(cashflows, discounts);
    } catch (const std::overflow_error&) {
        overflow_rejected = true;
    }
    assert(overflow_rejected);
}
