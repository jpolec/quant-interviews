#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <list>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

enum class Side { Buy, Sell };

struct Order {
    std::uint64_t id;
    Side side;
    std::int64_t price_ticks;
    std::int64_t quantity;
    std::uint64_t sequence;
};

struct Fill {
    std::uint64_t maker_id;
    std::uint64_t taker_id;
    std::int64_t price_ticks;
    std::int64_t quantity;
};

struct ExecutionResult {
    std::vector<Fill> fills;
    std::int64_t requested;
    std::int64_t remaining;
};

class PriceTimeBook {
    using Queue = std::list<Order>;
    using Levels = std::map<std::int64_t, Queue>;
    struct Locator {
        Side side;
        std::int64_t price_ticks;
        Queue::iterator iterator;
    };
    struct CancelRecord {
        std::uint64_t id;
        std::uint64_t sequence;
        bool found;
    };

public:
    ExecutionResult add_limit(
        std::uint64_t id,
        Side side,
        std::int64_t price_ticks,
        std::int64_t quantity,
        std::uint64_t sequence
    ) {
        validate_new(id, price_ticks, quantity, sequence);
        seen_.emplace(id);
        last_sequence_ = sequence;
        auto result = match(id, side, quantity, price_ticks);
        if (result.remaining > 0) {
            rest({id, side, price_ticks, result.remaining, sequence});
        }
        if (best_bid().has_value() && best_ask().has_value() &&
            *best_bid() >= *best_ask()) {
            throw std::logic_error("resting book crossed");
        }
        return result;
    }

    ExecutionResult execute_market(
        std::uint64_t id,
        Side side,
        std::int64_t quantity,
        std::uint64_t sequence
    ) {
        validate_new(id, 1, quantity, sequence);
        seen_.emplace(id);
        last_sequence_ = sequence;
        return match(id, side, quantity, std::nullopt);
    }

    bool cancel(std::uint64_t id, std::uint64_t sequence) {
        validate_sequence(sequence);
        last_sequence_ = sequence;
        auto found = index_.find(id);
        if (found == index_.end()) {
            cancellations_.push_back({id, sequence, false});
            return false;
        }
        auto& levels = found->second.side == Side::Buy ? bids_ : asks_;
        auto level = levels.find(found->second.price_ticks);
        level->second.erase(found->second.iterator);
        if (level->second.empty()) levels.erase(level);
        index_.erase(found);
        cancellations_.push_back({id, sequence, true});
        return true;
    }

    std::optional<std::int64_t> best_bid() const {
        return bids_.empty() ? std::nullopt
                             : std::optional<std::int64_t>(bids_.rbegin()->first);
    }

    std::optional<std::int64_t> best_ask() const {
        return asks_.empty() ? std::nullopt
                             : std::optional<std::int64_t>(asks_.begin()->first);
    }

    std::string state_hash() const {
        std::vector<Order> orders;
        orders.reserve(index_.size());
        for (const auto& [price, queue] : bids_) {
            (void)price;
            orders.insert(orders.end(), queue.begin(), queue.end());
        }
        for (const auto& [price, queue] : asks_) {
            (void)price;
            orders.insert(orders.end(), queue.begin(), queue.end());
        }
        std::sort(
            orders.begin(),
            orders.end(),
            [](const Order& left, const Order& right) {
                return left.sequence < right.sequence;
            }
        );
        std::ostringstream canonical;
        std::vector<std::uint64_t> seen_ids(seen_.begin(), seen_.end());
        std::sort(seen_ids.begin(), seen_ids.end());
        canonical << "v1|last=" << last_sequence_ << "|seen=";
        for (std::size_t index = 0; index < seen_ids.size(); ++index) {
            if (index > 0) canonical << ',';
            canonical << seen_ids[index];
        }
        canonical << "|orders=";
        for (const auto& order : orders) {
            canonical << order.id << ','
                      << (order.side == Side::Buy ? "buy" : "sell") << ','
                      << order.price_ticks << ',' << order.quantity << ','
                      << order.sequence << ';';
        }
        canonical << "|fills=";
        for (const auto& fill : fill_history_) {
            canonical << fill.maker_id << ',' << fill.taker_id << ','
                      << fill.price_ticks << ',' << fill.quantity << ';';
        }
        canonical << "|cancels=";
        for (const auto& cancel_record : cancellations_) {
            canonical << cancel_record.id << ',' << cancel_record.sequence << ','
                      << (cancel_record.found ? 1 : 0) << ';';
        }
        std::uint64_t hash = 14695981039346656037ULL;
        for (const unsigned char byte : canonical.str()) {
            hash ^= byte;
            hash *= 1099511628211ULL;
        }
        std::ostringstream output;
        output << std::hex << std::setw(16) << std::setfill('0') << hash;
        return output.str();
    }

private:
    void validate_new(
        std::uint64_t id,
        std::int64_t price_ticks,
        std::int64_t quantity,
        std::uint64_t sequence
    ) const {
        if (seen_.contains(id)) throw std::invalid_argument("duplicate order id");
        if (price_ticks <= 0 || quantity <= 0) throw std::invalid_argument("invalid order");
        validate_sequence(sequence);
    }

    void validate_sequence(std::uint64_t sequence) const {
        if (sequence != last_sequence_ + 1) {
            throw std::invalid_argument("sequence must be contiguous");
        }
    }

    void rest(Order order) {
        auto& levels = order.side == Side::Buy ? bids_ : asks_;
        auto& queue = levels[order.price_ticks];
        queue.push_back(order);
        auto iterator = std::prev(queue.end());
        index_.emplace(order.id, Locator{order.side, order.price_ticks, iterator});
    }

    ExecutionResult match(
        std::uint64_t taker_id,
        Side side,
        std::int64_t quantity,
        std::optional<std::int64_t> limit
    ) {
        auto& opposite = side == Side::Buy ? asks_ : bids_;
        std::vector<Fill> fills;
        std::int64_t remaining = quantity;
        while (remaining > 0 && !opposite.empty()) {
            auto level = side == Side::Buy ? opposite.begin() : std::prev(opposite.end());
            const auto price = level->first;
            if (limit.has_value()) {
                const bool marketable = side == Side::Buy ? price <= *limit : price >= *limit;
                if (!marketable) break;
            }
            auto& maker = level->second.front();
            const auto traded = std::min(remaining, maker.quantity);
            fills.push_back({maker.id, taker_id, price, traded});
            remaining -= traded;
            maker.quantity -= traded;
            if (maker.quantity == 0) {
                index_.erase(maker.id);
                level->second.pop_front();
                if (level->second.empty()) opposite.erase(level);
            }
        }
        fill_history_.insert(fill_history_.end(), fills.begin(), fills.end());
        return {std::move(fills), quantity, remaining};
    }

    Levels bids_;
    Levels asks_;
    std::unordered_map<std::uint64_t, Locator> index_;
    std::unordered_set<std::uint64_t> seen_;
    std::vector<Fill> fill_history_;
    std::vector<CancelRecord> cancellations_;
    std::uint64_t last_sequence_{0};
};

int main() {
    PriceTimeBook book;
    book.add_limit(1, Side::Sell, 101, 10, 1);
    book.add_limit(2, Side::Sell, 101, 20, 2);
    const auto result = book.add_limit(3, Side::Buy, 102, 15, 3);
    if (result.fills.size() != 2 || result.remaining != 0) return 1;
    std::cout << "fills=" << result.fills.size()
              << " remaining=" << result.remaining
              << " best_ask=" << *book.best_ask()
              << " state_hash=" << book.state_hash() << '\n';
    return 0;
}
