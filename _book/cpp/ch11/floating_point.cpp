#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

bool almost_equal(double a, double b, double abs_tol, double rel_tol) {
    if (!std::isfinite(a) || !std::isfinite(b) ||
        !std::isfinite(abs_tol) || !std::isfinite(rel_tol) ||
        abs_tol < 0.0 || rel_tol < 0.0) {
        return false;
    }
    const double scale = std::max(std::abs(a), std::abs(b));
    return std::abs(a - b) <= std::max(abs_tol, rel_tol * scale);
}

std::pair<double, double> stable_quadratic_roots(double a, double b, double c) {
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) {
        throw std::invalid_argument("coefficients must be finite");
    }
    if (a == 0.0) throw std::invalid_argument("a must be non-zero");
    const double scale = std::max({std::abs(a), std::abs(b), std::abs(c)});
    const double as = a / scale;
    const double bs = b / scale;
    const double cs = c / scale;
    const double discriminant = std::fma(bs, bs, -4.0 * as * cs);
    const double floor = 32.0 * std::numeric_limits<double>::epsilon() *
                         (bs * bs + std::abs(4.0 * as * cs));
    if (discriminant < -floor) throw std::invalid_argument("real roots do not exist");
    const double root = std::sqrt(std::max(discriminant, 0.0));
    const double q = -0.5 * (bs + std::copysign(root, bs));
    if (q == 0.0) {
        const double repeated = -bs / (2.0 * as);
        return {repeated, repeated};
    }
    double x1 = q / as;
    double x2 = cs / q;
    if (x2 < x1) std::swap(x1, x2);
    return {x1, x2};
}

int main() {
    const auto [x1, x2] = stable_quadratic_roots(1.0, 1e8, 1.0);
    std::cout << std::setprecision(17)
              << "x1=" << x1 << " x2=" << x2 << "\n"
              << "product=" << x1 * x2 << "\n"
              << "equal_check=" << almost_equal(x1 * x2, 1.0, 1e-12, 1e-12) << "\n";
}
