#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using Matrix2 = std::array<std::array<double, 2>, 2>;

struct Cholesky2x2 {
  double l00;
  double l10;
  double l11;
  double first_pivot;
  double second_pivot;
};

Cholesky2x2 cholesky_2x2(const Matrix2& matrix) {
  double scale = 0.0;
  for (const auto& row : matrix) {
    for (const double value : row) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("matrix entries must be finite");
      }
      scale = std::max(scale, std::abs(value));
    }
  }
  const double symmetry_tolerance =
      64.0 * std::numeric_limits<double>::epsilon()
      * std::max(1.0, scale);
  if (std::abs(matrix[0][1] - matrix[1][0]) > symmetry_tolerance) {
    throw std::invalid_argument("matrix must be symmetric");
  }

  const double first_pivot = matrix[0][0];
  if (!(first_pivot > 0.0)) {
    throw std::runtime_error("first Cholesky pivot is not positive");
  }
  const double l00 = std::sqrt(first_pivot);
  const double l10 = matrix[1][0] / l00;
  const double second_pivot = matrix[1][1] - l10 * l10;
  if (!(second_pivot > 0.0)) {
    throw std::runtime_error("second Cholesky pivot is not positive");
  }
  const double l11 = std::sqrt(second_pivot);
  return {l00, l10, l11, first_pivot, second_pivot};
}

double reconstruction_error(
    const Matrix2& matrix,
    const Cholesky2x2& factor) {
  return std::max({
      std::abs(matrix[0][0] - factor.l00 * factor.l00),
      std::abs(matrix[0][1] - factor.l00 * factor.l10),
      std::abs(matrix[1][0] - factor.l10 * factor.l00),
      std::abs(
          matrix[1][1]
          - (factor.l10 * factor.l10 + factor.l11 * factor.l11)),
  });
}

#ifndef QUANT_INTERVIEWS_CH02_NO_MAIN
int main() {
  constexpr Matrix2 correlation{{
      {{1.0, 0.3}},
      {{0.3, 1.0}},
  }};

  const Cholesky2x2 factor = cholesky_2x2(correlation);
  const double error = reconstruction_error(correlation, factor);

  std::cout << "method=cholesky"
            << " min_pivot="
            << std::min(factor.first_pivot, factor.second_pivot)
            << " reconstruction_error=" << error << '\n';
  return error <= 1e-14 ? 0 : 1;
}
#endif
