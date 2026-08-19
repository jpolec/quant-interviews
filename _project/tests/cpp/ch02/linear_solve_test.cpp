#define QUANT_INTERVIEWS_CH02_NO_MAIN
#include "../../../../_book/cpp/ch02/linear_solve.cpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

int main() {
  constexpr Matrix3 regular{{
      {{4.0, 1.0, 0.5}},
      {{1.0, 3.0, 0.25}},
      {{0.5, 0.25, 2.0}},
  }};

  constexpr Vector3 zero_rhs{{0.0, 0.0, 0.0}};
  const Vector3 zero_solution = cholesky_solve(regular, zero_rhs);
  require(
      zero_solution == Vector3{{0.0, 0.0, 0.0}},
      "zero rhs did not produce the zero solution");
  require(
      relative_residual(regular, zero_solution, zero_rhs) == 0.0,
      "zero rhs backward error must be finite and zero");

  constexpr Matrix3 nearly_singular{{
      {{1.0, 0.0, 0.0}},
      {{0.0, 1.0e-14, 0.0}},
      {{0.0, 0.0, 2.0}},
  }};
  constexpr Vector3 near_rhs{{1.0, 1.0e-14, 2.0}};
  const Vector3 near_solution = cholesky_solve(nearly_singular, near_rhs);
  const double near_error =
      relative_residual(nearly_singular, near_solution, near_rhs);
  require(std::isfinite(near_error), "near-singular error is not finite");
  require(near_error <= 1.0e-14, "near-singular solve has excessive error");

  bool non_positive_pivot_rejected = false;
  try {
    constexpr Matrix3 indefinite{{
        {{1.0, 0.0, 0.0}},
        {{0.0, -1.0, 0.0}},
        {{0.0, 0.0, 1.0}},
    }};
    static_cast<void>(cholesky_solve(indefinite, zero_rhs));
  } catch (const std::runtime_error&) {
    non_positive_pivot_rejected = true;
  }
  require(non_positive_pivot_rejected, "non-positive pivot was not rejected");

  constexpr Vector3 rhs{{1.0, 2.0, -1.0}};
  constexpr Vector3 intentionally_bad_solution{{0.0, 0.0, 0.0}};
  const double bad_error =
      relative_residual(regular, intentionally_bad_solution, rhs);
  require(bad_error > 1.0e-2, "bad solution was not detected by the residual");

  std::cout << "status=PASS"
            << " zero_rhs=PASS"
            << " near_singular=PASS"
            << " non_positive_pivot=REJECT"
            << " bad_solution=REJECT\n";
  return 0;
}
