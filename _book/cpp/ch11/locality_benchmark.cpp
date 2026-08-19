#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct BenchmarkEvidence {
    std::vector<double> samples;
    double checksum;
    int repeats;
    int warmups;
};

template <class Fn>
BenchmarkEvidence benchmark(Fn&& fn, int repeats = 9, int warmups = 2) {
    if (repeats < 3 || warmups < 1) throw std::invalid_argument("invalid repetitions");
    volatile double sink = 0.0;
    const double expected = fn();
    if (!std::isfinite(expected)) throw std::runtime_error("non-finite checksum");
    sink += expected;
    for (int r = 1; r < warmups; ++r) {
        const double result = fn();
        if (result != expected) throw std::runtime_error("warm-up checksum mismatch");
        sink += result;
    }
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r) {
        const auto start = std::chrono::steady_clock::now();
        const double result = fn();
        if (result != expected) throw std::runtime_error("benchmark checksum mismatch");
        sink += result;
        const auto stop = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(stop - start).count();
        if (!std::isfinite(seconds) || seconds <= 0.0) {
            throw std::runtime_error("invalid timing sample");
        }
        samples.push_back(seconds);
    }
    if (sink == -1.0) std::cerr << "unreachable";
    std::sort(samples.begin(), samples.end());
    return {std::move(samples), expected, repeats, warmups};
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
    constexpr std::size_t rows = 2048;
    constexpr std::size_t cols = 2048;
    std::vector<double> values(rows * cols, 1.0);

    const auto row_major = [&]() {
        double total = 0.0;
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols; ++j)
                total += values[i * cols + j];
        return total;
    };

    const auto column_major = [&]() {
        double total = 0.0;
        for (std::size_t j = 0; j < cols; ++j)
            for (std::size_t i = 0; i < rows; ++i)
                total += values[i * cols + j];
        return total;
    };

    const auto row_samples = benchmark(row_major);
    const auto col_samples = benchmark(column_major);
    if (row_samples.checksum != col_samples.checksum) return 2;
    const double row_seconds = linear_percentile(row_samples.samples, 0.50);
    const double col_seconds = linear_percentile(col_samples.samples, 0.50);
    std::cout << std::fixed << std::setprecision(6)
              << "rows=" << rows << " cols=" << cols
              << " repeats=" << row_samples.repeats
              << " warmups=" << row_samples.warmups
              << " percentile=linear_type7\n"
              << "row_checksum=" << row_samples.checksum
              << " column_checksum=" << col_samples.checksum << "\n"
              << "row_major_median=" << row_seconds << "\n"
              << "row_major_sample_q99=" << linear_percentile(row_samples.samples, 0.99) << "\n"
              << "column_major_median=" << col_seconds << "\n"
              << "column_major_sample_q99=" << linear_percentile(col_samples.samples, 0.99) << "\n"
              << "ratio=" << col_seconds / row_seconds << "\n";
}
