#define main ch11_false_sharing_example_main
#include "../../../../_book/cpp/ch11/false_sharing.cpp"
#undef main

#include <cassert>

int main() {
    static_assert(alignof(PaddedCounter) >= 64);
    bool checksum_rejected = false;
    try {
        validate_checksum(9, 10);
    } catch (const std::runtime_error&) {
        checksum_rejected = true;
    }
    assert(checksum_rejected);
    bool repeats_rejected = false;
    try {
        (void)benchmark_distribution<PackedCounter>(2, 100, 2);
    } catch (const std::invalid_argument&) {
        repeats_rejected = true;
    }
    assert(repeats_rejected);
    const std::vector<double> samples{1.0, 2.0, 3.0};
    assert(linear_percentile(samples, 0.99) > 2.0);
}
