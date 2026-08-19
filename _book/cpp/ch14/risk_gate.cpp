#include <cmath>
#include <iostream>
#include <string_view>

struct RiskEvidence {
    double proposed_risk;
    double limit;
    double coverage;
    double minimum_coverage;
    bool approved;
    bool approval_current;
    bool evidence_complete;
    bool tail_model_pass;
};

[[nodiscard]] std::string_view risk_gate(const RiskEvidence& evidence) noexcept {
    if (!std::isfinite(evidence.proposed_risk)
        || !std::isfinite(evidence.limit)
        || !std::isfinite(evidence.coverage)
        || !std::isfinite(evidence.minimum_coverage)
        || evidence.proposed_risk < 0.0
        || evidence.limit <= 0.0
        || evidence.coverage < 0.0
        || evidence.coverage > 1.0
        || evidence.minimum_coverage < 0.0
        || evidence.minimum_coverage > 1.0) {
        return "INVALID_INPUT";
    }
    if (!evidence.approved) return "BLOCK_NO_APPROVAL";
    if (!evidence.approval_current) return "BLOCK_STALE_APPROVAL";
    if (!evidence.evidence_complete) return "BLOCK_INCOMPLETE_EVIDENCE";
    if (evidence.coverage < evidence.minimum_coverage) return "BLOCK_COVERAGE";
    if (!evidence.tail_model_pass) return "BLOCK_TAIL_MODEL";
    if (evidence.proposed_risk > evidence.limit) return "BLOCK_LIMIT";
    return "PASS";
}

int main() {
    const RiskEvidence evidence{
        7.5,
        10.0,
        0.99,
        0.98,
        true,
        true,
        true,
        true,
    };
    const RiskEvidence impossible_coverage{
        1.0, 10.0, -2.0, -3.0, true, true, true, true,
    };
    const RiskEvidence stale{
        1.0, 10.0, 0.99, 0.98, true, false, true, true,
    };
    if (risk_gate(impossible_coverage) != "INVALID_INPUT"
        || risk_gate(stale) != "BLOCK_STALE_APPROVAL"
        || risk_gate(evidence) != "PASS") {
        return 2;
    }
    std::cout << risk_gate(evidence) << "\n";
}
