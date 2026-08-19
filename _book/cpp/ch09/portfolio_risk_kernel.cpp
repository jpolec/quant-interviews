#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

double minimum_symmetric_eigenvalue(
    std::vector<double> matrix,
    std::size_t dimension,
    double tolerance
) {
    const std::size_t maximum_sweeps = 64 * dimension * dimension;
    for (std::size_t sweep = 0; sweep < maximum_sweeps; ++sweep) {
        std::size_t pivot_row = 0;
        std::size_t pivot_column = 0;
        double maximum_off_diagonal = 0.0;
        for (std::size_t row = 0; row < dimension; ++row) {
            for (std::size_t column = row + 1; column < dimension; ++column) {
                const double candidate = std::abs(
                    matrix[row * dimension + column]
                );
                if (candidate > maximum_off_diagonal) {
                    maximum_off_diagonal = candidate;
                    pivot_row = row;
                    pivot_column = column;
                }
            }
        }
        if (maximum_off_diagonal <= tolerance) {
            break;
        }
        const double a = matrix[pivot_row * dimension + pivot_row];
        const double b = matrix[pivot_column * dimension + pivot_column];
        const double off = matrix[pivot_row * dimension + pivot_column];
        const double angle = 0.5 * std::atan2(2.0 * off, b - a);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (std::size_t index = 0; index < dimension; ++index) {
            if (index == pivot_row || index == pivot_column) {
                continue;
            }
            const double first = matrix[index * dimension + pivot_row];
            const double second = matrix[index * dimension + pivot_column];
            const double rotated_first = cosine * first - sine * second;
            const double rotated_second = sine * first + cosine * second;
            matrix[index * dimension + pivot_row] = rotated_first;
            matrix[pivot_row * dimension + index] = rotated_first;
            matrix[index * dimension + pivot_column] = rotated_second;
            matrix[pivot_column * dimension + index] = rotated_second;
        }
        matrix[pivot_row * dimension + pivot_row] =
            cosine * cosine * a - 2.0 * sine * cosine * off
            + sine * sine * b;
        matrix[pivot_column * dimension + pivot_column] =
            sine * sine * a + 2.0 * sine * cosine * off
            + cosine * cosine * b;
        matrix[pivot_row * dimension + pivot_column] = 0.0;
        matrix[pivot_column * dimension + pivot_row] = 0.0;
    }
    double minimum = matrix[0];
    for (std::size_t index = 1; index < dimension; ++index) {
        minimum = std::min(minimum, matrix[index * dimension + index]);
    }
    return minimum;
}

}  // namespace

double variance(
    const std::vector<double>& weights,
    const std::vector<double>& covariance,
    std::size_t dimension
) {
    if (
        dimension == 0
        || weights.size() != dimension
        || covariance.size() != dimension * dimension
    ) {
        throw std::invalid_argument("shape mismatch");
    }
    double covariance_scale = 0.0;
    double weight_norm_squared = 0.0;
    for (double weight : weights) {
        if (!std::isfinite(weight)) {
            throw std::invalid_argument("weights must be finite");
        }
        weight_norm_squared += weight * weight;
    }
    for (double entry : covariance) {
        if (!std::isfinite(entry)) {
            throw std::invalid_argument("covariance must be finite");
        }
        covariance_scale = std::max(covariance_scale, std::abs(entry));
    }
    const double symmetry_tolerance = 128.0
        * std::numeric_limits<double>::epsilon()
        * std::max(covariance_scale, std::numeric_limits<double>::min());
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = row + 1; column < dimension; ++column) {
            const double entry = covariance[row * dimension + column];
            const double mirror = covariance[column * dimension + row];
            if (std::abs(entry - mirror) > symmetry_tolerance) {
                throw std::invalid_argument("covariance must be symmetric");
            }
        }
    }
    const double eigen_tolerance = 512.0
        * std::numeric_limits<double>::epsilon()
        * std::max(covariance_scale, std::numeric_limits<double>::min());
    if (
        minimum_symmetric_eigenvalue(
            covariance,
            dimension,
            eigen_tolerance
        ) < -eigen_tolerance
    ) {
        throw std::invalid_argument("covariance must be positive semidefinite");
    }
    double value = 0.0;
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            value += weights[row]
                * covariance[row * dimension + column]
                * weights[column];
        }
    }
    const double value_tolerance = 512.0
        * std::numeric_limits<double>::epsilon()
        * std::max(
            covariance_scale * weight_norm_squared,
            std::numeric_limits<double>::min()
        );
    if (value < -value_tolerance) {
        throw std::invalid_argument("negative portfolio variance");
    }
    return value < 0.0 ? 0.0 : value;
}

#ifndef CH09_NO_MAIN
int main() {
    const std::vector<double> weights{0.6, 0.4};
    const std::vector<double> covariance{0.04, 0.006, 0.006, 0.09};
    const double volatility = std::sqrt(variance(weights, covariance, 2));
    const double independently_frozen_reference = 0.17798876369;
    const double residual = std::abs(
        volatility - independently_frozen_reference
    );
    bool indefinite_rejected = false;
    try {
        static_cast<void>(variance({0.0, 1.0}, {-1.0, 0.0, 0.0, 1.0}, 2));
    } catch (const std::invalid_argument&) {
        indefinite_rejected = true;
    }
    const bool passed = residual <= 1e-10 && indefinite_rejected;
    std::cout << "volatility=" << volatility << "\n";
    std::cout << "reference_residual=" << residual << "\n";
    std::cout << "indefinite_rejected="
              << (indefinite_rejected ? "true" : "false") << "\n";
    std::cout << "status=" << (passed ? "PASS" : "BLOCK") << "\n";
    return passed ? 0 : 1;
}
#endif
