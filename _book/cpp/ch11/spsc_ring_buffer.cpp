#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>

template <typename T, std::size_t Capacity>
class SpscRingBuffer {
    static_assert(Capacity >= 2);
    static_assert(std::is_nothrow_move_constructible_v<T>);
    static_assert(std::is_nothrow_move_assignable_v<T>);

public:
    static constexpr std::size_t usable_capacity = Capacity - 1;

    bool push(T value) noexcept {
        const auto head = head_.load(std::memory_order_relaxed);
        const auto next = increment(head);
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        storage_[head].emplace(std::move(value));
        head_.store(next, std::memory_order_release);
        return true;
    }

    std::optional<T> pop() noexcept {
        const auto tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        T value = std::move(*storage_[tail]);
        storage_[tail].reset();
        tail_.store(increment(tail), std::memory_order_release);
        return value;
    }

private:
    static constexpr std::size_t increment(std::size_t value) noexcept {
        return (value + 1) % Capacity;
    }

    std::array<std::optional<T>, Capacity> storage_{};
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};

int main() {
    static_assert(std::atomic<std::size_t>::is_always_lock_free);
    constexpr int observations = 100'000;
    SpscRingBuffer<int, 1024> queue;
    std::uint64_t total = 0;
    bool fifo_valid = true;
    std::thread producer([&]() {
        for (int value = 1; value <= observations; ++value) {
            while (!queue.push(value)) std::this_thread::yield();
        }
    });
    std::thread consumer([&]() {
        for (int received = 0; received < observations;) {
            if (auto value = queue.pop()) {
                if (*value != received + 1) fifo_valid = false;
                total += static_cast<std::uint64_t>(*value);
                ++received;
            } else {
                std::this_thread::yield();
            }
        }
    });
    producer.join();
    consumer.join();
    const auto expected =
        static_cast<std::uint64_t>(observations) * (observations + 1) / 2;
    if (total != expected || !fifo_valid) return 1;
    std::cout << "usable_capacity=" << queue.usable_capacity
              << " fifo_sum=" << total << " fifo=PASS\n";
    return 0;
}
