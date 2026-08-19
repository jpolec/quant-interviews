#define QUANT_INTERVIEWS_CH03_ADJOINT_KERNEL_NO_MAIN
#include "../../../../_book/cpp/ch03/adjoint_kernel.cpp"

#include <cmath>
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
  const Matrix2 matrix{{{{2.0, 1.0}}, {{1.0, 3.0}}}};
  const Vector2 rhs{{1.0, 0.0}};
  const Vector2 solution = solve_transpose_2x2(matrix, rhs);
  require(
      transpose_2x2_normwise_backward_error(matrix, solution, rhs)
          <= 1.0e-14,
      "regular adjoint solve has excessive backward error");

  const Matrix2 scaled_matrix{{
      {{2.0e150, 1.0e150}},
      {{1.0e150, 3.0e150}},
  }};
  const Vector2 scaled_rhs{{1.0e150, 0.0}};
  const Vector2 scaled_solution =
      solve_transpose_2x2(scaled_matrix, scaled_rhs);
  require(
      std::abs(scaled_solution[0] - 0.6) <= 1.0e-14
          && std::abs(scaled_solution[1] + 0.2) <= 1.0e-14,
      "normalized determinant solve failed");
  require(
      transpose_2x2_normwise_backward_error(
          scaled_matrix, scaled_solution, scaled_rhs) <= 1.0e-14,
      "scaled solve has excessive backward error");

  require_throws<std::runtime_error>(
      [] {
        const Matrix2 singular{{{{1.0, 2.0}}, {{2.0, 4.0}}}};
        static_cast<void>(solve_transpose_2x2(singular, {{1.0, 0.0}}));
      },
      "singular determinant was accepted");
  require_throws<std::runtime_error>(
      [] {
        const Matrix2 unsafe{{{{1.0, 0.0}}, {{0.0, 1.0e-10}}}};
        static_cast<void>(
            solve_transpose_2x2(unsafe, {{1.0, 0.0}}, 1.0e-8));
      },
      "near-singular determinant was accepted");
  require_throws<std::invalid_argument>(
      [] {
        const Matrix2 nonfinite{{
            {{std::numeric_limits<double>::quiet_NaN(), 0.0}},
            {{0.0, 1.0}},
        }};
        static_cast<void>(solve_transpose_2x2(nonfinite, {{1.0, 0.0}}));
      },
      "non-finite matrix was accepted");
  require_throws<std::invalid_argument>(
      [&matrix] {
        static_cast<void>(solve_transpose_2x2(
            matrix,
            {{std::numeric_limits<double>::infinity(), 0.0}}));
      },
      "non-finite rhs was accepted");
  require_throws<std::invalid_argument>(
      [&matrix, &rhs] {
        static_cast<void>(solve_transpose_2x2(matrix, rhs, 0.0));
      },
      "zero determinant tolerance was accepted");

  const Vector2 bad_solution{{0.0, 0.0}};
  require(
      transpose_2x2_normwise_backward_error(matrix, bad_solution, rhs)
          > 1.0e-2,
      "bad adjoint solution was not detected");

  std::cout << "status=PASS regular=PASS scaled=PASS singular=REJECT"
            << " near_singular=REJECT nonfinite=REJECT"
            << " bad_solution=REJECT\n";
  return 0;
}
