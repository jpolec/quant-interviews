#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

class VonNeumannExtractor {
public:
    std::vector<int> update(const std::vector<int>& bits) {
        // Validate the entire chunk before mutating the streaming state.
        for (const int bit : bits) {
            if (bit != 0 && bit != 1) {
                throw std::invalid_argument("binary input required");
            }
        }
        std::vector<int> output;
        for (const int bit : bits) {
            ++total_tosses_;
            if (!pending_.has_value()) {
                pending_ = bit;
                continue;
            }
            const int first = *pending_;
            pending_.reset();
            if (first == bit) {
                ++rejected_pairs_;
                continue;
            }
            // Explicit convention H=0, T=1: HT -> 0, TH -> 1.
            output.push_back(first);
            ++accepted_pairs_;
        }
        return output;
    }

    [[nodiscard]] bool has_pending() const { return pending_.has_value(); }
    [[nodiscard]] std::size_t total_tosses() const { return total_tosses_; }
    [[nodiscard]] std::size_t accepted_pairs() const { return accepted_pairs_; }
    [[nodiscard]] std::size_t rejected_pairs() const { return rejected_pairs_; }

private:
    std::optional<int> pending_;
    std::size_t total_tosses_{0};
    std::size_t accepted_pairs_{0};
    std::size_t rejected_pairs_{0};
};

int main() {
    VonNeumannExtractor extractor;
    const auto first = extractor.update({1, 1, 1});
    const auto second = extractor.update({0, 0, 1, 0, 0});
    const std::vector<int> expected{1, 0};
    const auto before_tosses = extractor.total_tosses();
    const auto before_accepted = extractor.accepted_pairs();
    const auto before_rejected = extractor.rejected_pairs();
    const auto before_pending = extractor.has_pending();
    bool invalid_rejected_atomically = false;
    try {
        static_cast<void>(extractor.update({0, 2}));
    } catch (const std::invalid_argument&) {
        invalid_rejected_atomically =
            extractor.total_tosses() == before_tosses &&
            extractor.accepted_pairs() == before_accepted &&
            extractor.rejected_pairs() == before_rejected &&
            extractor.has_pending() == before_pending;
    }
    const bool passed =
        first.empty() && second == expected && !extractor.has_pending() &&
        extractor.total_tosses() == 8 && extractor.accepted_pairs() == 2 &&
        extractor.rejected_pairs() == 2 && invalid_rejected_atomically;
    std::cout << "streaming=" << (passed ? "PASS" : "REJECT") << '\n'
              << "invalid_atomic="
              << (invalid_rejected_atomically ? "PASS" : "REJECT") << '\n'
              << "status=" << (passed ? "PASS" : "REJECT") << '\n';
    return passed ? 0 : 1;
}
