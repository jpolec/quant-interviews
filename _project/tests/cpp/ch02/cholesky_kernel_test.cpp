#define QUANT_INTERVIEWS_CH02_NO_MAIN
#include "../../../../_book/cpp/ch02/cholesky_kernel.cpp"

#include <iostream>
#include <stdexcept>

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

int main() {
  constexpr Matrix2 regular{{
      {{1.0, 0.3}},
      {{0.3, 1.0}},
  }};
  const Cholesky2x2 regular_factor = cholesky_2x2(regular);
  require(
      reconstruction_error(regular, regular_factor) <= 1.0e-14,
      "regular reconstruction failed");

  constexpr Matrix2 nearly_singular{{
      {{1.0, 1.0 - 1.0e-12}},
      {{1.0 - 1.0e-12, 1.0}},
  }};
  const Cholesky2x2 near_factor = cholesky_2x2(nearly_singular);
  require(near_factor.second_pivot > 0.0, "near-singular pivot was rejected");
  require(
      reconstruction_error(nearly_singular, near_factor) <= 1.0e-14,
      "near-singular reconstruction failed");

  bool non_positive_pivot_rejected = false;
  try {
    constexpr Matrix2 indefinite{{
        {{1.0, 2.0}},
        {{2.0, 1.0}},
    }};
    static_cast<void>(cholesky_2x2(indefinite));
  } catch (const std::runtime_error&) {
    non_positive_pivot_rejected = true;
  }
  require(non_positive_pivot_rejected, "non-positive pivot was not rejected");

  bool asymmetric_rejected = false;
  try {
    constexpr Matrix2 asymmetric{{
        {{1.0, 0.2}},
        {{0.3, 1.0}},
    }};
    static_cast<void>(cholesky_2x2(asymmetric));
  } catch (const std::invalid_argument&) {
    asymmetric_rejected = true;
  }
  require(asymmetric_rejected, "asymmetric matrix was not rejected");

  std::cout << "status=PASS"
            << " near_singular=PASS"
            << " non_positive_pivot=REJECT"
            << " asymmetric=REJECT\n";
  return 0;
}
