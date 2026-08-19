#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_set>

struct Order {
    std::uint64_t id{};
    std::int64_t quantity{};
};

using PriceTicks = std::int64_t;

PriceTicks price_to_ticks(double price, double tick_size) {
    if (!std::isfinite(price) || !std::isfinite(tick_size)
        || price <= 0.0 || tick_size <= 0.0) {
        throw std::invalid_argument("price and tick size must be finite and positive");
    }
    const auto ratio = price / tick_size;
    const auto maximum = static_cast<double>(std::numeric_limits<PriceTicks>::max());
    if (!std::isfinite(ratio) || ratio > maximum - 0.5) {
        throw std::out_of_range("price-to-tick conversion overflows int64");
    }
    const auto ticks = std::llround(ratio);
    if (ticks <= 0
        || std::abs(price - static_cast<double>(ticks) * tick_size)
            > tick_size * 1e-8) {
        throw std::invalid_argument("price is not aligned to the tick grid");
    }
    return ticks;
}

class PriceTimeBook {
public:
    explicit PriceTimeBook(double tick_size) : tick_size_(tick_size) {
        if (!std::isfinite(tick_size_) || tick_size_ <= 0.0) {
            throw std::invalid_argument("tick size must be finite and positive");
        }
    }

    void add_bid(double price, Order order) {
        validate_new_order(order);
        const auto ticks = price_to_ticks(price, tick_size_);
        if (!asks_.empty() && ticks >= asks_.begin()->first) {
            throw std::invalid_argument("bid would lock or cross the book");
        }
        order_ids_.insert(order.id);
        bids_[ticks].push_back(order);
    }

    void add_ask(double price, Order order) {
        validate_new_order(order);
        const auto ticks = price_to_ticks(price, tick_size_);
        if (!bids_.empty() && ticks <= bids_.begin()->first) {
            throw std::invalid_argument("ask would lock or cross the book");
        }
        order_ids_.insert(order.id);
        asks_[ticks].push_back(order);
    }

    std::int64_t execute_buy(std::int64_t quantity, double limit_price) {
        if (quantity <= 0) throw std::invalid_argument("quantity must be positive");
        const auto limit_ticks = price_to_ticks(limit_price, tick_size_);
        std::int64_t filled = 0;
        while (quantity > 0 && !asks_.empty()) {
            auto level = asks_.begin();
            if (level->first > limit_ticks) break;
            auto& queue = level->second;
            auto& resting = queue.front();
            const auto take = std::min(quantity, resting.quantity);
            resting.quantity -= take;
            quantity -= take;
            filled += take;
            if (resting.quantity == 0) {
                order_ids_.erase(resting.id);
                queue.pop_front();
            }
            if (queue.empty()) asks_.erase(level);
        }
        return filled;
    }

    PriceTicks best_bid_ticks() const {
        if (bids_.empty()) throw std::runtime_error("no bid");
        return bids_.begin()->first;
    }

    PriceTicks best_ask_ticks() const {
        if (asks_.empty()) throw std::runtime_error("no ask");
        return asks_.begin()->first;
    }

    bool contains(std::uint64_t order_id) const {
        return order_ids_.contains(order_id);
    }

private:
    void validate_new_order(const Order& order) const {
        if (order.id == 0 || order.quantity <= 0) {
            throw std::invalid_argument("order id and quantity must be positive");
        }
        if (order_ids_.contains(order.id)) {
            throw std::invalid_argument("duplicate order id");
        }
    }

    double tick_size_;
    std::unordered_set<std::uint64_t> order_ids_;
    std::map<PriceTicks, std::deque<Order>, std::greater<PriceTicks>> bids_;
    std::map<PriceTicks, std::deque<Order>, std::less<PriceTicks>> asks_;
};

#ifndef QI_CH10_NO_MAIN
int main() {
    PriceTimeBook book(0.01);
    book.add_bid(99.99, Order{1, 500});
    book.add_ask(100.01, Order{2, 300});
    book.add_ask(100.01, Order{3, 400});
    const auto filled = book.execute_buy(450, 100.01);
    const auto pass = filled == 450 && book.best_bid_ticks() == 9999
        && book.best_ask_ticks() == 10001 && !book.contains(2) && book.contains(3);
    std::cout << "filled=" << filled << " best_bid_ticks=" << book.best_bid_ticks()
              << " best_ask_ticks=" << book.best_ask_ticks()
              << " status=" << (pass ? "PASS" : "REJECT") << '\n';
    return pass ? 0 : 1;
}
#endif
