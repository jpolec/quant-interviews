#define main ch11_order_book_example_main
#include "../../../../_book/cpp/ch11/order_book.cpp"
#undef main

#include <cassert>

int main() {
    PriceTimeBook fresh;
    bool gap_rejected = false;
    try {
        (void)fresh.add_limit(1, Side::Buy, 10'000, 1, 2);
    } catch (const std::invalid_argument&) {
        gap_rejected = true;
    }
    assert(gap_rejected);

    PriceTimeBook book;
    (void)book.add_limit(101, Side::Sell, 10'100, 5, 1);
    const auto fill = book.execute_market(202, Side::Buy, 3, 2);
    assert(fill.fills.size() == 1 && fill.remaining == 0);
    assert(book.cancel(101, 3));
    assert(!book.best_ask());
    const std::string certificate = book.state_hash();
    assert(certificate.size() == 16);
    assert(certificate == "384dc9b637457002");
    assert(certificate != fresh.state_hash());

    bool reuse_rejected = false;
    try {
        (void)book.add_limit(101, Side::Sell, 10'100, 1, 4);
    } catch (const std::invalid_argument&) {
        reuse_rejected = true;
    }
    assert(reuse_rejected);
}
