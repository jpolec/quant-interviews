#define QI_CH10_NO_MAIN
#include "../../../../_book/cpp/ch10/price_time_order_book.cpp"

#include <cassert>
#include <limits>

template <typename Exception, typename Function>
void assert_throws(Function&& function) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    assert(false);
}

int main() {
    PriceTimeBook book(0.01);
    book.add_bid(99.99, Order{1, 100});
    book.add_ask(100.01, Order{2, 100});
    assert_throws<std::invalid_argument>([&] { book.add_ask(100.02, Order{2, 50}); });
    assert_throws<std::invalid_argument>([&] { book.add_bid(100.01, Order{3, 50}); });
    assert_throws<std::invalid_argument>([&] { book.add_ask(99.99, Order{4, 50}); });
    assert_throws<std::out_of_range>([&] {
        static_cast<void>(price_to_ticks(std::numeric_limits<double>::max(), 0.01));
    });
    assert(book.execute_buy(100, 100.01) == 100);
    assert(!book.contains(2));
    book.add_ask(100.02, Order{2, 50});
    assert(book.contains(2));
    return 0;
}
