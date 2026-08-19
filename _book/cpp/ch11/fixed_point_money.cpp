#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

class Money {
public:
    Money(std::int64_t minor_units, std::string currency, std::uint8_t scale)
        : minor_units_(minor_units),
          currency_(std::move(currency)),
          scale_(scale) {
        if (currency_.size() != 3 ||
            !std::all_of(currency_.begin(), currency_.end(), [](unsigned char value) {
                return std::isupper(value) != 0;
            })) {
            throw std::invalid_argument("three-letter uppercase currency code required");
        }
        if (scale_ > 9) throw std::invalid_argument("unsupported money scale");
    }

    [[nodiscard]] std::int64_t minor_units() const noexcept { return minor_units_; }
    [[nodiscard]] const std::string& currency() const noexcept { return currency_; }
    [[nodiscard]] std::uint8_t scale() const noexcept { return scale_; }

    friend Money operator+(const Money& left, const Money& right) {
        if (left.currency_ != right.currency_ || left.scale_ != right.scale_) {
            throw std::invalid_argument("currency/scale mismatch");
        }
        if ((right.minor_units_ > 0 &&
             left.minor_units_ > std::numeric_limits<std::int64_t>::max() - right.minor_units_) ||
            (right.minor_units_ < 0 &&
             left.minor_units_ < std::numeric_limits<std::int64_t>::min() - right.minor_units_)) {
            throw std::overflow_error("money addition overflow");
        }
        return Money(left.minor_units_ + right.minor_units_, left.currency_, left.scale_);
    }

private:
    std::int64_t minor_units_;
    std::string currency_;
    std::uint8_t scale_;
};

int main() {
    const Money cash(10'005, "USD", 2);
    const Money fee(-5, "USD", 2);
    const Money settled = cash + fee;
    std::cout << settled.currency() << ' ' << settled.minor_units()
              << " scale=" << static_cast<int>(settled.scale()) << '\n';
    return 0;
}
