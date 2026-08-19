#define QUANT_INTERVIEWS_CH03_ROOT_SOLVER_NO_MAIN
#include "../../../../_book/cpp/ch03/root_solver.cpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, const char* message) {
  bool rejected = false;
  try {
    callable();
  } catch (const Exception&) {
    rejected = true;
  }
  require(rejected, message);
}

}  // namespace

int main() {
  const auto quadratic = [](double value) { return value * value - 2.0; };
  const double root = bisection(quadratic, 0.0, 2.0, 1.0e-14);
  require(std::abs(root * root - 2.0) <= 1.0e-13, "regular root failed");
  require(
      bisection([](double value) { return value; }, 0.0, 1.0, 1.0e-12)
          == 0.0,
      "lower endpoint root failed");
  require(
      bisection([](double value) { return value - 1.0; }, 0.0, 1.0, 1.0e-12)
          == 1.0,
      "upper endpoint root failed");

  require_throws<std::invalid_argument>(
      [] { static_cast<void>(bisection({}, 0.0, 1.0, 1.0e-8)); },
      "empty function was accepted");
  require_throws<std::invalid_argument>(
      [&quadratic] { static_cast<void>(bisection(quadratic, 2.0, 0.0, 1.0e-8)); },
      "reversed bracket was accepted");
  require_throws<std::invalid_argument>(
      [&quadratic] {
        static_cast<void>(bisection(
            quadratic,
            -std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max(),
            1.0e-8));
      },
      "non-finite bracket width was accepted");
  require_throws<std::invalid_argument>(
      [&quadratic] { static_cast<void>(bisection(quadratic, 0.0, 2.0, 0.0)); },
      "zero tolerance was accepted");
  require_throws<std::invalid_argument>(
      [&quadratic] { static_cast<void>(bisection(quadratic, 0.0, 2.0, -1.0)); },
      "negative tolerance was accepted");
  require_throws<std::invalid_argument>(
      [&quadratic] {
        static_cast<void>(bisection(
            quadratic,
            0.0,
            2.0,
            std::numeric_limits<double>::quiet_NaN()));
      },
      "NaN tolerance was accepted");
  require_throws<std::invalid_argument>(
      [&quadratic] { static_cast<void>(bisection(quadratic, 0.0, 2.0, 1.0e-8, 0)); },
      "zero iteration limit was accepted");
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(bisection(
            [](double value) { return value * value + 1.0; },
            -1.0,
            1.0,
            1.0e-8));
      },
      "non-sign-changing bracket was accepted");
  require_throws<std::domain_error>(
      [] {
        static_cast<void>(bisection(
            [](double value) {
              return value == 1.0
                  ? std::numeric_limits<double>::quiet_NaN()
                  : value - 0.5;
            },
            0.0,
            2.0,
            1.0e-8));
      },
      "non-finite midpoint value was accepted");
  require_throws<std::runtime_error>(
      [&quadratic] { static_cast<void>(bisection(quadratic, 0.0, 2.0, 1.0e-15, 1)); },
      "insufficient iteration budget was accepted");

  std::cout << "status=PASS regular=PASS endpoints=PASS"
            << " invalid_tolerance=REJECT nonfinite_function=REJECT"
            << " iteration_limit=REJECT\n";
  return 0;
}
