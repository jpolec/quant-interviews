#include <cstdint>
#include <iostream>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct Event {
    std::int64_t available_at_ns;
    int priority;
    std::int64_t event_time_ns;
    std::uint64_t sequence;
    std::string event_id;
    std::string kind;
};

struct Later {
    bool operator()(const Event& left, const Event& right) const {
        return std::tie(
            left.available_at_ns,
            left.priority,
            left.event_time_ns,
            left.sequence,
            left.event_id
        ) > std::tie(
            right.available_at_ns,
            right.priority,
            right.event_time_ns,
            right.sequence,
            right.event_id
        );
    }
};

std::vector<std::string> ordered_kinds(const std::vector<Event>& events) {
    std::priority_queue<Event, std::vector<Event>, Later> queue;
    std::set<std::string> ids;
    std::set<std::uint64_t> sequences;
    for (const auto& event : events) {
        if (event.available_at_ns < 0 || event.event_time_ns < 0 ||
            event.priority < 0 || event.sequence == 0 || event.event_id.empty() ||
            event.kind.empty()) {
            throw std::invalid_argument("invalid event");
        }
        if (!ids.insert(event.event_id).second ||
            !sequences.insert(event.sequence).second) {
            throw std::invalid_argument("duplicate event identity");
        }
        queue.push(event);
    }
    std::vector<std::string> result;
    while (!queue.empty()) {
        result.push_back(queue.top().kind);
        queue.pop();
    }
    return result;
}

#ifndef QJ_CH12_NO_MAIN
int main() {
    const std::vector<Event> fixture{
        {10, 30, 5, 3, "order-1", "order"},
        {10, 10, 10, 1, "market-1", "market"},
        // Earlier event time, but it became available later.
        {20, 10, 1, 2, "late-1", "late_market"},
    };
    const auto ordered = ordered_kinds(fixture);
    std::string result;
    for (const auto& kind : ordered) {
        result += kind + ":";
    }
    std::cout << result << '\n';
    return result == "market:order:late_market:" ? 0 : 1;
}
#endif
