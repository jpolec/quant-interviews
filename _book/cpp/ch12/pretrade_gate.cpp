#include <cmath>
#include <iostream>
#include <string>
#include <vector>

enum class Decision { pass, requires_approval, block, invalid_input };

struct Risk {
    double gross;
    double net;
    double single_name_max;
    double var99;
    double liquidity_days;
};

struct Limits {
    double gross;
    double abs_net;
    double single_name_max;
    double var99;
    double liquidity_days;
};

bool finite(const Risk& risk) {
    return std::isfinite(risk.gross) &&
           std::isfinite(risk.net) &&
           std::isfinite(risk.single_name_max) &&
           std::isfinite(risk.var99) &&
           std::isfinite(risk.liquidity_days);
}

bool finite(const Limits& limits) {
    return std::isfinite(limits.gross) && limits.gross >= 0.0 &&
           std::isfinite(limits.abs_net) && limits.abs_net >= 0.0 &&
           std::isfinite(limits.single_name_max) && limits.single_name_max >= 0.0 &&
           std::isfinite(limits.var99) && limits.var99 >= 0.0 &&
           std::isfinite(limits.liquidity_days) && limits.liquidity_days >= 0.0;
}

std::vector<std::string> hard_breaches(
    const Risk& risk,
    const Limits& limits
) {
    std::vector<std::string> result;
    if (risk.gross > limits.gross) {
        result.push_back("gross");
    }
    if (std::abs(risk.net) > limits.abs_net) {
        result.push_back("net");
    }
    if (risk.single_name_max > limits.single_name_max) {
        result.push_back("single_name_max");
    }
    if (risk.var99 > limits.var99) {
        result.push_back("var99");
    }
    return result;
}

Decision decide(
    const Risk& before,
    const Risk& after,
    const Limits& limits
) {
    if (!finite(before) || !finite(after) || !finite(limits)) {
        return Decision::invalid_input;
    }
    constexpr double tolerance = 1e-12;
    const bool new_or_worse_hard =
        (after.gross > limits.gross &&
         (before.gross <= limits.gross || after.gross > before.gross + tolerance)) ||
        (std::abs(after.net) > limits.abs_net &&
         (std::abs(before.net) <= limits.abs_net ||
          std::abs(after.net) > std::abs(before.net) + tolerance)) ||
        (after.single_name_max > limits.single_name_max &&
         (before.single_name_max <= limits.single_name_max ||
          after.single_name_max > before.single_name_max + tolerance)) ||
        (after.var99 > limits.var99 &&
         (before.var99 <= limits.var99 || after.var99 > before.var99 + tolerance));
    if (new_or_worse_hard) {
        return Decision::block;
    }
    if (after.liquidity_days > limits.liquidity_days &&
        before.liquidity_days <= limits.liquidity_days) {
        return Decision::requires_approval;
    }
    return Decision::pass;
}

#ifndef QJ_CH12_NO_MAIN
int main() {
    const Limits limits{1.5, 0.25, 0.20, 0.05, 3.0};
    const auto decision = decide(
        {1.2, 0.1, 0.10, 0.03, 2.0},
        {1.3, 0.1, 0.10, 0.03, 4.0},
        limits
    );
    const bool passed = decision == Decision::requires_approval;
    std::cout << (passed ? "REQUIRES_APPROVAL" : "REJECT") << '\n';
    return passed ? 0 : 1;
}
#endif
