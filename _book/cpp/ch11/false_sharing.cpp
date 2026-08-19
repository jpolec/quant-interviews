#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

struct PackedCounter {
    std::atomic<std::uint64_t> value{0};
};

struct alignas(64) PaddedCounter {
    std::atomic<std::uint64_t> value{0};
};

void validate_checksum(std::uint64_t observed, std::uint64_t expected) {
    if (observed != expected) {
        throw std::runtime_error("counter checksum mismatch");
    }
}

template <class Counter>
double run_once(std::size_t workers, std::uint64_t iterations) {
    std::vector<Counter> counters(workers);
    std::barrier synchronized_start(static_cast<std::ptrdiff_t>(workers + 1));
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker]() {
            synchronized_start.arrive_and_wait();
            for (std::uint64_t i = 0; i < iterations; ++i) {
                counters[worker].value.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    const auto start = std::chrono::steady_clock::now();
    synchronized_start.arrive_and_wait();
    for (auto& thread : threads) thread.join();
    const auto stop = std::chrono::steady_clock::now();
    std::uint64_t total = 0;
    for (const auto& counter : counters) {
        total += counter.value.load(std::memory_order_relaxed);
    }
    validate_checksum(total, workers * iterations);
    const double seconds = std::chrono::duration<double>(stop - start).count();
    if (!std::isfinite(seconds) || seconds <= 0.0) {
        throw std::runtime_error("invalid timing sample");
    }
    return seconds;
}

template <class Counter>
std::vector<double> benchmark_distribution(
    std::size_t workers,
    std::uint64_t iterations,
    int repeats
) {
    if (repeats < 3) throw std::invalid_argument("at least three repeats required");
    if (workers == 0 || iterations < 10) {
        throw std::invalid_argument("workers and iterations must be positive");
    }
    (void)run_once<Counter>(workers, iterations / 10);  // validated warm-up
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        samples.push_back(run_once<Counter>(workers, iterations));
    }
    std::sort(samples.begin(), samples.end());
    return samples;
}

double linear_percentile(const std::vector<double>& sorted, double probability) {
    if (sorted.empty() || probability < 0.0 || probability > 1.0) {
        throw std::invalid_argument("invalid percentile request");
    }
    const double position = probability * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

int main() {
    const std::size_t workers = 4;
    const std::uint64_t iterations = 2'000'000;
    constexpr int repeats = 31;
    const auto packed = benchmark_distribution<PackedCounter>(workers, iterations, repeats);
    const auto padded = benchmark_distribution<PaddedCounter>(workers, iterations, repeats);
    const double packed_median = linear_percentile(packed, 0.50);
    const double padded_median = linear_percentile(padded, 0.50);
    std::cout << "workers=" << workers << " iterations=" << iterations
              << " repeats=" << repeats << " percentile=linear_type7\n"
              << "packed_size=" << sizeof(PackedCounter)
              << " packed_alignment=" << alignof(PackedCounter)
              << " padded_size=" << sizeof(PaddedCounter)
              << " padded_alignment=" << alignof(PaddedCounter) << "\n"
              << "packed_median=" << packed_median << "\n"
              << "packed_sample_q99=" << linear_percentile(packed, 0.99) << "\n"
              << "padded_median=" << padded_median << "\n"
              << "padded_sample_q99=" << linear_percentile(padded, 0.99) << "\n"
              << "median_ratio=" << packed_median / padded_median << "\n";
    return 0;
}
