#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_finite_vector(
    const std::vector<double>& values,
    const char* name) {
  if (!std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value);
      })) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

void require_safe_pivot(
    double pivot,
    double scale,
    double relative_pivot_tolerance) {
  if (!std::isfinite(pivot) || !std::isfinite(scale) || !(scale > 0.0)
      || std::abs(pivot) <= relative_pivot_tolerance * scale) {
    throw std::runtime_error("singular or unsafe tridiagonal pivot");
  }
}

double infinity_norm(const std::vector<double>& values) {
  double result = 0.0;
  for (const double value : values) {
    result = std::max(result, std::abs(value));
  }
  return result;
}

}  // namespace

std::vector<double> solve_tridiagonal(
    std::vector<double> lower,
    std::vector<double> diagonal,
    std::vector<double> upper,
    std::vector<double> rhs,
    double relative_pivot_tolerance =
        64.0 * std::numeric_limits<double>::epsilon()) {
  const std::size_t size = diagonal.size();
  if (size == 0U) {
    throw std::invalid_argument("tridiagonal system must be non-empty");
  }
  if (lower.size() != size || upper.size() != size || rhs.size() != size) {
    throw std::invalid_argument("aligned tridiagonal system required");
  }
  if (!std::isfinite(relative_pivot_tolerance)
      || !(relative_pivot_tolerance > 0.0)
      || !(relative_pivot_tolerance < 1.0)) {
    throw std::invalid_argument(
        "relative pivot tolerance must be finite and in (0, 1)");
  }
  require_finite_vector(lower, "lower diagonal");
  require_finite_vector(diagonal, "diagonal");
  require_finite_vector(upper, "upper diagonal");
  require_finite_vector(rhs, "right-hand side");
  if (lower.front() != 0.0 || upper.back() != 0.0) {
    throw std::invalid_argument(
        "unused tridiagonal endpoint coefficients must be zero");
  }

  for (std::size_t row = 1; row < size; ++row) {
    const std::size_t pivot_row = row - 1U;
    const double pivot_scale = std::max({
        std::abs(diagonal[pivot_row]),
        std::abs(upper[pivot_row]),
        std::abs(lower[row]),
    });
    require_safe_pivot(
        diagonal[pivot_row], pivot_scale, relative_pivot_tolerance);
    const double multiplier = lower[row] / diagonal[pivot_row];
    diagonal[row] -= multiplier * upper[pivot_row];
    rhs[row] -= multiplier * rhs[pivot_row];
    if (!std::isfinite(diagonal[row]) || !std::isfinite(rhs[row])) {
      throw std::runtime_error("non-finite value during tridiagonal elimination");
    }
  }

  const double last_scale = std::max({
      std::abs(diagonal.back()),
      std::abs(lower.back()),
      std::abs(upper.back()),
  });
  require_safe_pivot(
      diagonal.back(), last_scale, relative_pivot_tolerance);

  std::vector<double> solution(size);
  solution.back() = rhs.back() / diagonal.back();
  if (!std::isfinite(solution.back())) {
    throw std::runtime_error("non-finite tridiagonal solution");
  }
  for (std::size_t reverse = 1; reverse < size; ++reverse) {
    const std::size_t row = size - 1U - reverse;
    const double row_scale = std::max({
        std::abs(diagonal[row]),
        std::abs(lower[row]),
        std::abs(upper[row]),
    });
    require_safe_pivot(
        diagonal[row], row_scale, relative_pivot_tolerance);
    solution[row] = (
        rhs[row] - upper[row] * solution[row + 1U]
    ) / diagonal[row];
    if (!std::isfinite(solution[row])) {
      throw std::runtime_error("non-finite tridiagonal solution");
    }
  }
  return solution;
}

double tridiagonal_normwise_backward_error(
    const std::vector<double>& lower,
    const std::vector<double>& diagonal,
    const std::vector<double>& upper,
    const std::vector<double>& solution,
    const std::vector<double>& rhs) {
  const std::size_t size = diagonal.size();
  if (size == 0U || lower.size() != size || upper.size() != size
      || solution.size() != size || rhs.size() != size) {
    throw std::invalid_argument("aligned non-empty residual inputs required");
  }
  require_finite_vector(lower, "lower diagonal");
  require_finite_vector(diagonal, "diagonal");
  require_finite_vector(upper, "upper diagonal");
  require_finite_vector(solution, "solution");
  require_finite_vector(rhs, "right-hand side");
  if (lower.front() != 0.0 || upper.back() != 0.0) {
    throw std::invalid_argument(
        "unused tridiagonal endpoint coefficients must be zero");
  }

  double matrix_norm = 0.0;
  double residual_norm = 0.0;
  for (std::size_t row = 0; row < size; ++row) {
    double product = diagonal[row] * solution[row];
    double row_sum = std::abs(diagonal[row]);
    if (row > 0U) {
      product += lower[row] * solution[row - 1U];
      row_sum += std::abs(lower[row]);
    }
    if (row + 1U < size) {
      product += upper[row] * solution[row + 1U];
      row_sum += std::abs(upper[row]);
    }
    if (!std::isfinite(product) || !std::isfinite(row_sum)) {
      throw std::runtime_error("non-finite tridiagonal residual");
    }
    matrix_norm = std::max(matrix_norm, row_sum);
    residual_norm = std::max(residual_norm, std::abs(product - rhs[row]));
  }
  const double denominator =
      matrix_norm * infinity_norm(solution) + infinity_norm(rhs);
  if (!std::isfinite(denominator)) {
    throw std::runtime_error("non-finite residual scale");
  }
  if (denominator == 0.0) {
    return residual_norm == 0.0
        ? 0.0
        : std::numeric_limits<double>::infinity();
  }
  return residual_norm / denominator;
}

#ifndef QUANT_INTERVIEWS_CH03_TRIDIAGONAL_NO_MAIN
int main() {
  const std::vector<double> lower{0.0, -1.0, -1.0};
  const std::vector<double> diagonal{2.0, 2.0, 2.0};
  const std::vector<double> upper{-1.0, -1.0, 0.0};
  const std::vector<double> rhs{1.0, 0.0, 1.0};
  const auto solution = solve_tridiagonal(lower, diagonal, upper, rhs);
  const double error = tridiagonal_normwise_backward_error(
      lower, diagonal, upper, solution, rhs);
  std::cout << "solution=" << solution[0] << ',' << solution[1]
            << ',' << solution[2] << " backward_error=" << error << '\n';
  return error <= 1e-14 ? 0 : 1;
}
#endif
