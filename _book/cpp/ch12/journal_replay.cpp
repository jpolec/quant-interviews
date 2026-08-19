#include <iostream>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct Event {
    std::string event_id;
    std::string aggregate_id;
    std::int64_t sequence;
    std::string type;
    std::int64_t delta;
};

std::int64_t checked_add(std::int64_t left, std::int64_t right) {
    if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
        throw std::overflow_error("journal state");
    }
    return left + right;
}

std::int64_t replay(
    const std::string& aggregate_id,
    const std::vector<Event>& events
) {
    if (aggregate_id.empty()) {
        throw std::invalid_argument("aggregate_id");
    }
    std::int64_t state = 0;
    std::int64_t expected = 1;
    std::map<std::string, std::tuple<std::string, std::int64_t, std::string, std::int64_t>> seen;
    for (const auto& event : events) {
        if (event.event_id.empty() || event.aggregate_id != aggregate_id ||
            event.sequence <= 0 || event.delta < 0) {
            throw std::runtime_error("mixed aggregate stream");
        }
        const auto content = std::make_tuple(
            event.aggregate_id, event.sequence, event.type, event.delta
        );
        const auto duplicate = seen.find(event.event_id);
        if (duplicate != seen.end()) {
            if (duplicate->second != content) {
                throw std::runtime_error("event id reused with different content");
            }
            continue;  // Exact delivery replay is idempotent.
        }
        seen.emplace(event.event_id, content);
        if (event.sequence != expected++) {
            throw std::runtime_error("sequence gap");
        }
        if (event.type == "add") {
            state = checked_add(state, event.delta);
        } else if (event.type == "remove") {
            state = checked_add(state, -event.delta);
        } else {
            throw std::runtime_error("unknown event type");
        }
    }
    return state;
}

#ifndef QJ_CH12_NO_MAIN
int main() {
    const std::vector<Event> events{
        {"e1", "portfolio-1", 1, "add", 10},
        {"e2", "portfolio-1", 2, "remove", 3},
        {"e3", "portfolio-1", 3, "add", 2},
    };
    const std::int64_t state = replay("portfolio-1", events);
    std::cout << state << '\n';
    return state == 9 ? 0 : 1;
}
#endif
