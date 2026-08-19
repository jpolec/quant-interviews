#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

struct NamedFeature {
    std::string_view name;
    double value;
};

class OnlineScorer {
public:
    static constexpr std::size_t feature_count = 4;
    static constexpr std::array<std::string_view, feature_count> feature_names{
        "stationarity_score",
        "regime_probability",
        "volatility_ratio",
        "calibration_residual"
    };
    // SHA-256 of the canonical names/order/types/units/transformations printed
    // in CH08 v0.10 and recomputed independently in the Python parity test.
    static constexpr std::string_view feature_schema_hash =
        "sha256:8afd31a62f52490379822fd497e6e00fec579eb9d5dab008d284827fa0616c6c";

    OnlineScorer(
        std::array<double, feature_count> coefficients,
        double intercept,
        std::string model_version
    )
        : coefficients_(coefficients),
          intercept_(intercept),
          model_version_(std::move(model_version)) {
        if (!std::isfinite(intercept_) || model_version_.empty()) {
            throw std::invalid_argument("invalid scorer metadata");
        }
        for (const double coefficient : coefficients_) {
            if (!std::isfinite(coefficient)) {
                throw std::invalid_argument("non-finite coefficient");
            }
        }
    }

    [[nodiscard]] double probability(
        std::string_view request_schema_hash,
        std::span<const NamedFeature, feature_count> features
    ) const {
        if (request_schema_hash != feature_schema_hash) {
            throw std::invalid_argument("feature-schema hash mismatch");
        }
        double score = intercept_;
        for (std::size_t index = 0; index < feature_count; ++index) {
            if (features[index].name != feature_names[index]) {
                throw std::invalid_argument("feature name/order mismatch");
            }
            if (!std::isfinite(features[index].value)) {
                throw std::invalid_argument("non-finite feature value");
            }
            score += coefficients_[index] * features[index].value;
        }
        if (!std::isfinite(score)) {
            throw std::overflow_error("non-finite logistic score");
        }
        if (score >= 0.0) {
            const double exponential = std::exp(-score);
            return 1.0 / (1.0 + exponential);
        }
        const double exponential = std::exp(score);
        return exponential / (1.0 + exponential);
    }

    [[nodiscard]] std::string_view model_version() const noexcept {
        return model_version_;
    }

private:
    std::array<double, feature_count> coefficients_;
    double intercept_;
    std::string model_version_;
};

double parse_feature_argument(
    std::string_view text,
    std::string_view expected_name
) {
    const std::size_t separator = text.find('=');
    if (separator == std::string_view::npos || text.substr(0, separator) != expected_name) {
        throw std::invalid_argument("feature name/order mismatch");
    }
    const std::string_view value = text.substr(separator + 1);
    if (value.empty()) {
        throw std::invalid_argument("missing feature value");
    }
    double parsed = 0.0;
    const char* const first = value.data();
    const char* const last = value.data() + value.size();
    const auto result = std::from_chars(first, last, parsed, std::chars_format::general);
    if (result.ec != std::errc{} || result.ptr != last || !std::isfinite(parsed)) {
        throw std::invalid_argument("invalid finite feature value");
    }
    return parsed;
}

int main(int argc, char** argv) {
    try {
        const OnlineScorer scorer(
            {0.35, -0.20, 0.10, 0.05},
            -0.15,
            "ch08-demo-v3"
        );
        std::array<NamedFeature, OnlineScorer::feature_count> features{{
            {OnlineScorer::feature_names[0], 1.2},
            {OnlineScorer::feature_names[1], -0.4},
            {OnlineScorer::feature_names[2], 0.8},
            {OnlineScorer::feature_names[3], 0.1}
        }};
        std::string_view request_schema = OnlineScorer::feature_schema_hash;
        if (argc != 1 && argc != 6) {
            throw std::invalid_argument(
                "usage: online_scorer schema_hash name=value name=value name=value name=value"
            );
        }
        if (argc == 6) {
            request_schema = argv[1];
            for (std::size_t index = 0; index < features.size(); ++index) {
                features[index].value = parse_feature_argument(
                    argv[index + 2],
                    OnlineScorer::feature_names[index]
                );
            }
        }
        const double value = scorer.probability(request_schema, features);
        std::cout
            << scorer.model_version() << ' '
            << OnlineScorer::feature_schema_hash << ' '
            << std::setprecision(17) << value << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "{\"status\":\"REJECT\",\"reason\":\""
            << error.what()
            << "\"}\n";
        return 2;
    }
}
