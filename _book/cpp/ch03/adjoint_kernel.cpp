#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using Matrix2 = std::array<std::array<double, 2>, 2>;
using Vector2 = std::array<double, 2>;

namespace {

void require_finite_contract(const Matrix2& matrix, const Vector2& rhs) {
  for (const auto& row : matrix) {
    for (const double value : row) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("adjoint matrix must be finite");
      }
    }
  }
  for (const double value : rhs) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("adjoint right-hand side must be finite");
    }
  }
}

}  // namespace

Vector2 solve_transpose_2x2(
    const Matrix2& matrix,
    const Vector2& rhs,
    double relative_determinant_tolerance =
        64.0 * std::numeric_limits<double>::epsilon()) {
  require_finite_contract(matrix, rhs);
  if (!std::isfinite(relative_determinant_tolerance)
      || !(relative_determinant_tolerance > 0.0)
      || !(relative_determinant_tolerance < 1.0)) {
    throw std::invalid_argument(
        "relative determinant tolerance must be finite and in (0, 1)");
  }

  const double scale = std::max({
      std::abs(matrix[0][0]),
      std::abs(matrix[0][1]),
      std::abs(matrix[1][0]),
      std::abs(matrix[1][1]),
  });
  if (!(scale > 0.0) || !std::isfinite(scale)) {
    throw std::runtime_error("singular adjoint matrix");
  }
  const Matrix2 normalized{{
      {{matrix[0][0] / scale, matrix[0][1] / scale}},
      {{matrix[1][0] / scale, matrix[1][1] / scale}},
  }};
  const Vector2 scaled_rhs{{rhs[0] / scale, rhs[1] / scale}};
  const double normalized_determinant =
      normalized[0][0] * normalized[1][1]
      - normalized[0][1] * normalized[1][0];
  if (!std::isfinite(normalized_determinant)
      || normalized_determinant == 0.0
      || std::abs(normalized_determinant)
          <= relative_determinant_tolerance) {
    throw std::runtime_error("singular or unsafe adjoint determinant");
  }
  if (!std::isfinite(scaled_rhs[0]) || !std::isfinite(scaled_rhs[1])) {
    throw std::runtime_error("non-finite scaled adjoint right-hand side");
  }

  const Vector2 solution{{
      (normalized[1][1] * scaled_rhs[0]
       - normalized[1][0] * scaled_rhs[1]) / normalized_determinant,
      (-normalized[0][1] * scaled_rhs[0]
       + normalized[0][0] * scaled_rhs[1]) / normalized_determinant,
  }};
  if (!std::isfinite(solution[0]) || !std::isfinite(solution[1])) {
    throw std::runtime_error("non-finite adjoint solution");
  }
  return solution;
}

double transpose_2x2_normwise_backward_error(
    const Matrix2& matrix,
    const Vector2& solution,
    const Vector2& rhs) {
  require_finite_contract(matrix, rhs);
  if (!std::isfinite(solution[0]) || !std::isfinite(solution[1])) {
    throw std::invalid_argument("adjoint solution must be finite");
  }
  const Vector2 residual{{
      matrix[0][0] * solution[0] + matrix[1][0] * solution[1] - rhs[0],
      matrix[0][1] * solution[0] + matrix[1][1] * solution[1] - rhs[1],
  }};
  const double matrix_norm = std::max(
      std::abs(matrix[0][0]) + std::abs(matrix[1][0]),
      std::abs(matrix[0][1]) + std::abs(matrix[1][1]));
  const double solution_norm =
      std::max(std::abs(solution[0]), std::abs(solution[1]));
  const double rhs_norm = std::max(std::abs(rhs[0]), std::abs(rhs[1]));
  const double residual_norm =
      std::max(std::abs(residual[0]), std::abs(residual[1]));
  const double denominator = matrix_norm * solution_norm + rhs_norm;
  if (!std::isfinite(residual_norm) || !std::isfinite(denominator)) {
    throw std::runtime_error("non-finite adjoint residual scale");
  }
  if (denominator == 0.0) {
    return residual_norm == 0.0
        ? 0.0
        : std::numeric_limits<double>::infinity();
  }
  return residual_norm / denominator;
}

#ifndef QUANT_INTERVIEWS_CH03_ADJOINT_KERNEL_NO_MAIN
int main() {
  const Matrix2 matrix{{
      {{2.0, 1.0}},
      {{1.0, 3.0}},
  }};
  const Vector2 objective_gradient{{1.0, 0.0}};
  const auto adjoint = solve_transpose_2x2(matrix, objective_gradient);
  const double residual = transpose_2x2_normwise_backward_error(
      matrix, adjoint, objective_gradient);
  std::cout << "adjoint=" << adjoint[0] << ',' << adjoint[1]
            << " backward_error=" << residual << '\n';
  return residual <= 1e-14 ? 0 : 1;
}
#endif
