from __future__ import annotations

from quant_interviews.ch12.integrated_case import run_governed_promotion_case


def test_integrated_case_rejects_hard_pretrade_breach() -> None:
    report = run_governed_promotion_case()
    assert report.fixture_id == "ch12_governed_promotion_v1"
    assert len(report.gates) == 13
    assert report.gate("pretrade_risk").status == "REJECT"
    assert report.gate("data_quality").status == "CONDITIONAL_PASS"
    assert report.gate("agent_governance").status == "PASS"
    assert report.decision == "REJECT"
    assert len(report.report_hash) == 64


def test_integrated_case_is_deterministic_and_gate_sensitive() -> None:
    first = run_governed_promotion_case()
    second = run_governed_promotion_case()
    assert first == second
    repaired = run_governed_promotion_case(
        proposed_gross=1.5,
        data_quality_issue_count=0,
    )
    assert repaired.gate("pretrade_risk").status == "PASS"
    assert repaired.gate("data_quality").status == "PASS"
    assert repaired.decision == "PASS"
    assert repaired.report_hash != first.report_hash
