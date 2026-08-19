#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using Matrix3 = std::array<std::array<double, 3>, 3>;
using Vector3 = std::array<double, 3>;

double relative_residual(
    const std::array<std::array<double, 3>, 3>& matrix,
    const std::array<double, 3>& solution,
    const std::array<double, 3>& rhs) {
  double residual_norm = 0.0;
  double matrix_norm = 0.0;
  double solution_norm = 0.0;
  double rhs_norm = 0.0;
  for (std::size_t row = 0; row < matrix.size(); ++row) {
    double fitted = 0.0;
    double absolute_row_sum = 0.0;
    for (std::size_t column = 0; column < matrix.size(); ++column) {
      if (!std::isfinite(matrix[row][column])) {
        throw std::invalid_argument("matrix entries must be finite");
      }
      fitted += matrix[row][column] * solution[column];
      absolute_row_sum += std::abs(matrix[row][column]);
    }
    if (!std::isfinite(solution[row]) || !std::isfinite(rhs[row])) {
      throw std::invalid_argument("solution and rhs entries must be finite");
    }
    residual_norm = std::max(residual_norm, std::abs(fitted - rhs[row]));
    matrix_norm = std::max(matrix_norm, absolute_row_sum);
    solution_norm = std::max(solution_norm, std::abs(solution[row]));
    rhs_norm = std::max(rhs_norm, std::abs(rhs[row]));
  }

  const double scale = matrix_norm * solution_norm + rhs_norm;
  if (scale == 0.0) {
    return residual_norm == 0.0
        ? 0.0
        : std::numeric_limits<double>::infinity();
  }
  return residual_norm / scale;
}

Vector3 cholesky_solve(const Matrix3& matrix, const Vector3& rhs) {
  double matrix_scale = 0.0;
  for (const auto& row : matrix) {
    for (const double value : row) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("matrix entries must be finite");
      }
      matrix_scale = std::max(matrix_scale, std::abs(value));
    }
  }
  for (const double value : rhs) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("rhs entries must be finite");
    }
  }
  const double symmetry_tolerance =
      64.0 * std::numeric_limits<double>::epsilon()
      * std::max(1.0, matrix_scale);
  for (std::size_t row = 0; row < matrix.size(); ++row) {
    for (std::size_t column = 0; column < row; ++column) {
      if (std::abs(matrix[row][column] - matrix[column][row])
          > symmetry_tolerance) {
        throw std::invalid_argument("matrix must be symmetric");
      }
    }
  }

  Matrix3 lower{};
  for (std::size_t row = 0; row < matrix.size(); ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      double value = matrix[row][column];
      for (std::size_t index = 0; index < column; ++index) {
        value -= lower[row][index] * lower[column][index];
      }
      if (row == column) {
        if (!(value > 0.0)) {
          throw std::runtime_error("matrix is not positive definite");
        }
        lower[row][column] = std::sqrt(value);
      } else {
        lower[row][column] = value / lower[column][column];
      }
    }
  }

  Vector3 intermediate{};
  for (std::size_t row = 0; row < matrix.size(); ++row) {
    double value = rhs[row];
    for (std::size_t column = 0; column < row; ++column) {
      value -= lower[row][column] * intermediate[column];
    }
    intermediate[row] = value / lower[row][row];
  }

  Vector3 solution{};
  for (std::size_t reverse = 0; reverse < matrix.size(); ++reverse) {
    const std::size_t row = matrix.size() - 1 - reverse;
    double value = intermediate[row];
    for (std::size_t column = row + 1; column < matrix.size(); ++column) {
      value -= lower[column][row] * solution[column];
    }
    solution[row] = value / lower[row][row];
  }
  return solution;
}

#ifndef QUANT_INTERVIEWS_CH02_NO_MAIN
int main() {
  constexpr Matrix3 matrix{{
      {{4.0, 1.0, 0.5}},
      {{1.0, 3.0, 0.25}},
      {{0.5, 0.25, 2.0}},
  }};
  constexpr Vector3 rhs{{1.0, 2.0, -1.0}};

  const Vector3 solution = cholesky_solve(matrix, rhs);
  const double backward_error = relative_residual(matrix, solution, rhs);

  std::cout << "method=cholesky_solve"
            << " normwise_backward_error=" << backward_error << '\n';
  return backward_error <= 1e-14 ? 0 : 1;
}
#endif
