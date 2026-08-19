#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

struct PositionColumns {
    std::vector<double> quantity;
    std::vector<double> price;
    std::vector<double> beta;
};

double portfolio_beta_notional(const PositionColumns& positions) {
    const auto n = positions.quantity.size();
    if (positions.price.size() != n || positions.beta.size() != n) {
        throw std::invalid_argument("column lengths differ");
    }
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(positions.quantity[i]) ||
            !std::isfinite(positions.price[i]) ||
            !std::isfinite(positions.beta[i])) {
            throw std::invalid_argument("position columns must be finite");
        }
        result += positions.quantity[i] * positions.price[i] * positions.beta[i];
    }
    return result;
}

int main() {
    PositionColumns p{{100.0, -50.0, 25.0}, {10.0, 20.0, 40.0}, {1.2, 0.7, -0.2}};
    std::cout << portfolio_beta_notional(p) << '\n';
}
