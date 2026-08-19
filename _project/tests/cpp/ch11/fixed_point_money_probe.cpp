#define main ch11_fixed_point_example_main
#include "../../../../_book/cpp/ch11/fixed_point_money.cpp"
#undef main

#include <cassert>

int main() {
    for (const std::string invalid : {"usd", "US1", "EURO"}) {
        bool rejected = false;
        try {
            (void)Money(1, invalid, 2);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }

    bool positive_overflow_rejected = false;
    try {
        (void)(Money(std::numeric_limits<std::int64_t>::max(), "USD", 2) +
               Money(1, "USD", 2));
    } catch (const std::overflow_error&) {
        positive_overflow_rejected = true;
    }
    assert(positive_overflow_rejected);

    bool negative_overflow_rejected = false;
    try {
        (void)(Money(std::numeric_limits<std::int64_t>::min(), "USD", 2) +
               Money(-1, "USD", 2));
    } catch (const std::overflow_error&) {
        negative_overflow_rejected = true;
    }
    assert(negative_overflow_rejected);
}
