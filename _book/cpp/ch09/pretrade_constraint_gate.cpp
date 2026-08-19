#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

bool passes(
    const std::vector<double>& weights,
    double gross_limit,
    double single_name_limit
) {
    if (
        weights.empty()
        || !std::isfinite(gross_limit)
        || !std::isfinite(single_name_limit)
        || gross_limit < 0.0
        || single_name_limit < 0.0
    ) {
        throw std::invalid_argument("invalid constraint inputs");
    }
    double gross = 0.0;
    double maximum_absolute = 0.0;
    for (double weight : weights) {
        if (!std::isfinite(weight)) {
            throw std::invalid_argument("weights must be finite");
        }
        gross += std::abs(weight);
        maximum_absolute = std::max(
            maximum_absolute,
            std::abs(weight)
        );
    }
    const double scale = std::max(
        {gross, gross_limit, maximum_absolute, single_name_limit, 1.0}
    );
    const double tolerance = 128.0
        * std::numeric_limits<double>::epsilon() * scale;
    return gross <= gross_limit + tolerance
        && maximum_absolute <= single_name_limit + tolerance;
}

#ifndef CH09_NO_MAIN
int main() {
    const bool valid = passes({0.35, 0.25, 0.20, 0.20}, 1.0, 0.40);
    const bool gross_breach = passes({0.60, 0.50}, 1.0, 0.70);
    const bool name_breach = passes({0.50, 0.25, 0.25}, 1.0, 0.40);
    const bool boundary = passes({0.40, 0.30, 0.30}, 1.0, 0.40);
    bool invalid_limit_blocked = false;
    try {
        static_cast<void>(passes({0.5, 0.5}, -1.0, 0.6));
    } catch (const std::invalid_argument&) {
        invalid_limit_blocked = true;
    }
    const bool passed = valid
        && !gross_breach
        && !name_breach
        && boundary
        && invalid_limit_blocked;
    std::cout << "valid=" << (valid ? "PASS" : "BLOCK") << "\n";
    std::cout << "gross_breach=" << (gross_breach ? "PASS" : "BLOCK") << "\n";
    std::cout << "name_breach=" << (name_breach ? "PASS" : "BLOCK") << "\n";
    std::cout << "boundary=" << (boundary ? "PASS" : "BLOCK") << "\n";
    std::cout << "status=" << (passed ? "PASS" : "BLOCK") << "\n";
    return passed ? 0 : 1;
}
#endif
