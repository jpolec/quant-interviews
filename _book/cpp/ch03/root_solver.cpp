#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {

double evaluate_finite(
    const std::function<double(double)>& function,
    double argument) {
  const double value = function(argument);
  if (!std::isfinite(value)) {
    throw std::domain_error("root function must return finite values");
  }
  return value;
}

bool has_opposite_sign(double left, double right) {
  return (left < 0.0 && right > 0.0)
      || (left > 0.0 && right < 0.0);
}

}  // namespace

double bisection(
    const std::function<double(double)>& function,
    double lower,
    double upper,
    double tolerance,
    std::size_t maximum_iterations = 256) {
  if (!function) {
    throw std::invalid_argument("a root function is required");
  }
  if (!std::isfinite(lower) || !std::isfinite(upper) || !(lower < upper)) {
    throw std::invalid_argument("a finite ordered bracket is required");
  }
  if (!std::isfinite(upper - lower)) {
    throw std::invalid_argument("bracket width must be finite");
  }
  if (!std::isfinite(tolerance) || !(tolerance > 0.0)) {
    throw std::invalid_argument("tolerance must be finite and positive");
  }
  if (maximum_iterations == 0U) {
    throw std::invalid_argument("maximum_iterations must be positive");
  }

  double lower_value = evaluate_finite(function, lower);
  const double upper_value = evaluate_finite(function, upper);
  if (lower_value == 0.0) {
    return lower;
  }
  if (upper_value == 0.0) {
    return upper;
  }
  if (!has_opposite_sign(lower_value, upper_value)) {
    throw std::invalid_argument("a sign-changing bracket is required");
  }

  for (std::size_t iteration = 0; iteration < maximum_iterations; ++iteration) {
    if (upper - lower <= tolerance) {
      return std::midpoint(lower, upper);
    }
    const double midpoint = std::midpoint(lower, upper);
    if (midpoint == lower || midpoint == upper) {
      throw std::runtime_error("bisection stagnated before reaching tolerance");
    }
    const double midpoint_value = evaluate_finite(function, midpoint);
    if (midpoint_value == 0.0) {
      return midpoint;
    }
    if (has_opposite_sign(lower_value, midpoint_value)) {
      upper = midpoint;
    } else {
      lower = midpoint;
      lower_value = midpoint_value;
    }
  }
  throw std::runtime_error("bisection iteration limit exceeded");
}

#ifndef QUANT_INTERVIEWS_CH03_ROOT_SOLVER_NO_MAIN
int main() {
  const double root = bisection(
      [](double value) { return value * value - 2.0; },
      0.0,
      2.0,
      1e-14);
  const double residual = std::abs(root * root - 2.0);
  std::cout << "root=" << root << " residual=" << residual << '\n';
  return residual <= 1e-13 ? 0 : 1;
}
#endif
