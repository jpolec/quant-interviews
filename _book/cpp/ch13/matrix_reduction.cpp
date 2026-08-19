#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using Matrix = std::vector<std::vector<int>>;
using Operation = std::array<int, 4>;

void validate(const Matrix& matrix) {
    if (matrix.size() < 2 || matrix.front().size() < 2) {
        throw std::invalid_argument("matrix must be at least 2x2");
    }
    const std::size_t columns = matrix.front().size();
    for (const auto& row : matrix) {
        if (row.size() != columns) {
            throw std::invalid_argument("matrix must be rectangular");
        }
        for (const int value : row) {
            if (value != 0 && value != 1) {
                throw std::invalid_argument("matrix must be binary");
            }
        }
    }
}

void toggle(Matrix& matrix, int r1, int r2, int c1, int c2) {
    validate(matrix);
    if (!(0 <= r1 && r1 < r2 && r2 < static_cast<int>(matrix.size()) &&
          0 <= c1 && c1 < c2 &&
          c2 < static_cast<int>(matrix.front().size()))) {
        throw std::invalid_argument("invalid rectangle");
    }
    for (const auto [row, column] :
         {std::pair{r1, c1}, {r1, c2}, {r2, c1}, {r2, c2}}) {
        matrix[static_cast<std::size_t>(row)]
              [static_cast<std::size_t>(column)] ^= 1;
    }
}

bool reducible(const Matrix& matrix) {
    validate(matrix);
    for (const auto& row : matrix) {
        int parity = 0;
        for (const int value : row) parity ^= value;
        if (parity != 0) return false;
    }
    for (std::size_t column = 0; column < matrix.front().size(); ++column) {
        int parity = 0;
        for (const auto& row : matrix) parity ^= row[column];
        if (parity != 0) return false;
    }
    return true;
}

bool is_zero(const Matrix& matrix) {
    validate(matrix);
    for (const auto& row : matrix) {
        for (const int value : row) {
            if (value != 0) return false;
        }
    }
    return true;
}

std::vector<Operation> reduction_operations(Matrix matrix) {
    validate(matrix);
    if (!reducible(matrix)) {
        throw std::invalid_argument("parity invariant blocks reduction");
    }
    const int last_row = static_cast<int>(matrix.size()) - 1;
    const int last_column = static_cast<int>(matrix.front().size()) - 1;
    std::vector<Operation> operations;
    for (int row = 0; row < last_row; ++row) {
        for (int column = 0; column < last_column; ++column) {
            if (matrix[static_cast<std::size_t>(row)]
                      [static_cast<std::size_t>(column)] == 1) {
                const Operation operation{row, last_row, column, last_column};
                toggle(matrix, operation[0], operation[1], operation[2], operation[3]);
                operations.push_back(operation);
            }
        }
    }
    if (!is_zero(matrix)) {
        throw std::runtime_error("constructive reduction failed");
    }
    return operations;
}

Matrix replay(Matrix matrix, const std::vector<Operation>& operations) {
    validate(matrix);
    for (const Operation& operation : operations) {
        toggle(matrix, operation[0], operation[1], operation[2], operation[3]);
    }
    return matrix;
}

int main() {
    const Matrix first{{1, 1, 0}, {1, 1, 0}, {0, 0, 0}};
    const Matrix second{{1, 0, 1, 0}, {1, 0, 1, 0}, {0, 0, 0, 0}};
    const auto first_operations = reduction_operations(first);
    const auto second_operations = reduction_operations(second);
    const bool replay_passed =
        is_zero(replay(first, first_operations)) &&
        is_zero(replay(second, second_operations));

    bool ragged_rejected = false;
    bool nonbinary_rejected = false;
    bool irreducible_rejected = false;
    try {
        const Matrix ragged{{1, 0}, {1}};
        static_cast<void>(reducible(ragged));
    } catch (const std::invalid_argument&) {
        ragged_rejected = true;
    }
    try {
        const Matrix nonbinary{{0, 2}, {0, 0}};
        static_cast<void>(reduction_operations(nonbinary));
    } catch (const std::invalid_argument&) {
        nonbinary_rejected = true;
    }
    try {
        const Matrix blocked{{1, 0}, {0, 0}};
        static_cast<void>(reduction_operations(blocked));
    } catch (const std::invalid_argument&) {
        irreducible_rejected = true;
    }
    const bool passed = replay_passed && ragged_rejected &&
                        nonbinary_rejected && irreducible_rejected;
    std::cout << "constructive_replay="
              << (replay_passed ? "PASS" : "REJECT") << '\n'
              << "invalid_rejected="
              << ((ragged_rejected && nonbinary_rejected) ? "PASS" : "REJECT")
              << '\n'
              << "irreducible_rejected="
              << (irreducible_rejected ? "PASS" : "REJECT") << '\n'
              << "status=" << (passed ? "PASS" : "REJECT") << '\n';
    return passed ? 0 : 3;
}
