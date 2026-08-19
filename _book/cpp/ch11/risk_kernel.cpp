#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

struct MatrixView {
    const double* data;
    std::size_t rows;
    std::size_t cols;
    std::size_t row_stride;
    double operator()(std::size_t i, std::size_t j) const { return data[i * row_stride + j]; }
};

double portfolio_variance(const std::vector<double>& weights, MatrixView covariance) {
    if (weights.empty()) throw std::invalid_argument("portfolio must not be empty");
    if (covariance.rows != covariance.cols || covariance.rows != weights.size()) {
        throw std::invalid_argument("shape mismatch");
    }
    if (covariance.data == nullptr || covariance.row_stride < covariance.cols) {
        throw std::invalid_argument("invalid matrix view");
    }
    for (double weight : weights) {
        if (!std::isfinite(weight)) throw std::invalid_argument("non-finite weight");
    }
    double matrix_scale = 1.0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        for (std::size_t j = 0; j < weights.size(); ++j) {
            const double value = covariance(i, j);
            const double mirror = covariance(j, i);
            if (!std::isfinite(value)) {
                throw std::invalid_argument("covariance must be finite");
            }
            matrix_scale = std::max(matrix_scale, std::abs(value));
            if (std::abs(value - mirror) > 1e-12 * matrix_scale) {
                throw std::invalid_argument("covariance must be symmetric");
            }
        }
    }

    // Pivoted-by-order Cholesky certificate with an explicit semidefinite
    // branch.  A zero pivot may only have a zero residual column.
    const std::size_t dimension = weights.size();
    const long double tolerance = 1e-12L * matrix_scale;
    std::vector<long double> lower(dimension * dimension, 0.0L);
    for (std::size_t i = 0; i < dimension; ++i) {
        long double diagonal = covariance(i, i);
        for (std::size_t k = 0; k < i; ++k) {
            const long double value = lower[i * dimension + k];
            diagonal -= value * value;
        }
        if (diagonal < -tolerance) {
            throw std::invalid_argument("covariance must be positive semidefinite");
        }
        if (diagonal <= tolerance) {
            for (std::size_t j = i + 1; j < dimension; ++j) {
                long double residual = covariance(j, i);
                for (std::size_t k = 0; k < i; ++k) {
                    residual -= lower[j * dimension + k] * lower[i * dimension + k];
                }
                if (std::abs(residual) > tolerance) {
                    throw std::invalid_argument("covariance must be positive semidefinite");
                }
            }
            continue;
        }
        lower[i * dimension + i] = std::sqrt(diagonal);
        for (std::size_t j = i + 1; j < dimension; ++j) {
            long double residual = covariance(j, i);
            for (std::size_t k = 0; k < i; ++k) {
                residual -= lower[j * dimension + k] * lower[i * dimension + k];
            }
            lower[j * dimension + i] = residual / lower[i * dimension + i];
        }
    }

    long double total = 0.0L;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        long double row_dot = 0.0L;
        for (std::size_t j = 0; j < weights.size(); ++j) {
            const double value = covariance(i, j);
            row_dot += value * weights[j];
        }
        total += weights[i] * row_dot;
    }
    if (!std::isfinite(total)) throw std::overflow_error("portfolio variance is non-finite");
    if (total < -tolerance) throw std::invalid_argument("negative portfolio variance");
    const double result = static_cast<double>(std::max(total, 0.0L));
    if (!std::isfinite(result)) throw std::overflow_error("portfolio variance overflow");
    return result;
}

int main() {
    const std::vector<double> weights{0.5, 0.3, 0.2};
    const std::vector<double> covariance{
        0.04, 0.01, 0.00,
        0.01, 0.09, 0.02,
        0.00, 0.02, 0.16,
    };
    const double variance = portfolio_variance(weights, {covariance.data(), 3, 3, 3});
    const std::vector<double> indefinite{1.0, 2.0, 2.0, 1.0};
    try {
        (void)portfolio_variance({1.0, 0.0}, {indefinite.data(), 2, 2, 2});
        return 2;
    } catch (const std::invalid_argument&) {
        // Expected adversarial rejection.
    }
    std::cout << "variance=" << variance << " volatility=" << std::sqrt(variance) << "\n";
    return 0;
}
