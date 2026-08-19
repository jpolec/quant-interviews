#define QUANT_INTERVIEWS_CH03_TRIDIAGONAL_NO_MAIN
#include "../../../../_book/cpp/ch03/tridiagonal.cpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

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
  const std::vector<double> lower{0.0, -1.0, -1.0};
  const std::vector<double> diagonal{2.0, 2.0, 2.0};
  const std::vector<double> upper{-1.0, -1.0, 0.0};
  const std::vector<double> rhs{1.0, 0.0, 1.0};
  const auto solution = solve_tridiagonal(lower, diagonal, upper, rhs);
  require(
      tridiagonal_normwise_backward_error(
          lower, diagonal, upper, solution, rhs) <= 1.0e-14,
      "regular solve has excessive backward error");

  const std::vector<double> scaled_diagonal{1.0e-200, 2.0e-200};
  const std::vector<double> scaled_zeros{0.0, 0.0};
  const std::vector<double> scaled_rhs{1.0e-200, 4.0e-200};
  const auto scaled_solution = solve_tridiagonal(
      scaled_zeros, scaled_diagonal, scaled_zeros, scaled_rhs);
  require(
      std::abs(scaled_solution[0] - 1.0) <= 1.0e-14
          && std::abs(scaled_solution[1] - 2.0) <= 1.0e-14,
      "scale-aware solve failed");

  require_throws<std::invalid_argument>(
      [] { static_cast<void>(solve_tridiagonal({}, {}, {}, {})); },
      "empty system was accepted");
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(solve_tridiagonal(
            {0.0}, {1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0}));
      },
      "misaligned system was accepted");
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(solve_tridiagonal(
            {1.0, 0.0}, {1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0}));
      },
      "nonzero structural endpoint was accepted");
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(solve_tridiagonal(
            {0.0},
            {std::numeric_limits<double>::quiet_NaN()},
            {0.0},
            {1.0}));
      },
      "non-finite coefficient was accepted");
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(solve_tridiagonal(
            {0.0},
            {1.0},
            {0.0},
            {std::numeric_limits<double>::infinity()}));
      },
      "non-finite rhs was accepted");
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(solve_tridiagonal(
            {0.0}, {1.0}, {0.0}, {1.0}, 0.0));
      },
      "zero pivot tolerance was accepted");
  require_throws<std::invalid_argument>(
      [] {
        static_cast<void>(solve_tridiagonal(
            {0.0},
            {1.0},
            {0.0},
            {1.0},
            std::numeric_limits<double>::quiet_NaN()));
      },
      "NaN pivot tolerance was accepted");
  require_throws<std::runtime_error>(
      [] {
        static_cast<void>(solve_tridiagonal(
            {0.0, 1.0},
            {1.0e-16, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}));
      },
      "unsafe near-singular pivot was accepted");
  require_throws<std::runtime_error>(
      [] {
        static_cast<void>(solve_tridiagonal(
            {0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}));
      },
      "zero first pivot was accepted");
  require_throws<std::runtime_error>(
      [] {
        static_cast<void>(solve_tridiagonal(
            {0.0, 1.0, 0.0},
            {1.0, 1.0, 1.0},
            {1.0, 1.0, 0.0},
            {1.0, 1.0, 1.0}));
      },
      "zero intermediate pivot was accepted");
  require_throws<std::runtime_error>(
      [] {
        static_cast<void>(solve_tridiagonal(
            {0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}));
      },
      "zero last pivot was accepted");

  const std::vector<double> bad_solution{0.0, 0.0, 0.0};
  require(
      tridiagonal_normwise_backward_error(
          lower, diagonal, upper, bad_solution, rhs) > 1.0e-2,
      "bad solution was not detected");

  std::cout << "status=PASS regular=PASS scaled=PASS empty=REJECT"
            << " misaligned=REJECT nonfinite=REJECT near_singular=REJECT"
            << " all_pivots=REJECT"
            << " bad_solution=REJECT\n";
  return 0;
}
