from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from quant_interviews.ch08.evidence import (
    GatePolicy,
    evaluate_gate,
    reduce_gate_records,
    seal_evidence_bundle,
    verify_evidence_bundle,
)


def payload() -> dict[str, object]:
    return {
        "raw_data": {"timestamps": ["2025-01-01T00:00:00Z"], "value": [1.0]},
        "splits": {"outer": [{"train": [0], "test": [1]}]},
        "parameters": {"threshold": 0.62, "cost": 0.0005, "seed": 808},
        "policies": {"decision": "reject_any_hard_failure-v1"},
        "results": {"loss": 0.2, "state": "PASS"},
        "source_hashes": {"module.py": "a" * 64},
        "environment": {"python": "3.x", "lock_hash": "b" * 64},
    }


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("raw_data", "value", [2.0]),
        ("splits", "outer", [{"train": [1], "test": [0]}]),
        ("parameters", "threshold", 0.63),
        ("parameters", "cost", 0.0006),
        ("parameters", "seed", 809),
        ("policies", "decision", "other-policy"),
        ("results", "loss", 0.3),
        ("source_hashes", "module.py", "c" * 64),
        ("environment", "lock_hash", "d" * 64),
    ],
)
def test_complete_bundle_digest_changes_for_every_governed_mutation(
    section: str,
    key: str,
    replacement: object,
) -> None:
    baseline = seal_evidence_bundle(payload())
    changed = payload()
    changed_section = changed[section]
    assert isinstance(changed_section, dict)
    changed_section[key] = replacement
    mutated = seal_evidence_bundle(changed)
    assert mutated["bundle_id"] != baseline["bundle_id"]
    verify_evidence_bundle(mutated)


def test_key_order_does_not_change_bundle_digest_and_tampering_rejects() -> None:
    original = payload()
    reordered = dict(reversed(list(original.items())))
    first = seal_evidence_bundle(original)
    second = seal_evidence_bundle(reordered)
    assert first["bundle_id"] == second["bundle_id"]
    tampered = deepcopy(first)
    tampered["payload"]["results"]["loss"] = {"$float64_hex": (0.4).hex()}
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_evidence_bundle(tampered)


def test_gate_state_and_final_decision_are_computed_not_literals() -> None:
    policy = GatePolicy("adf", "pvalue", "<", 0.05, "probability", True, "v1")
    passing = evaluate_gate(policy, 0.01, evidence={"pvalue": 0.01})
    failing = evaluate_gate(policy, 0.99, evidence={"pvalue": 0.99})
    assert passing.state == "PASS" and failing.state == "REJECT"
    assert reduce_gate_records(["adf"], [passing])[0] == "PASS"
    assert reduce_gate_records(["adf"], [failing])[0] == "REJECT"
    assert reduce_gate_records(["adf", "other"], [passing])[0] == "REJECT"
    with pytest.raises(ValueError, match="finite"):
        evaluate_gate(policy, np.inf, evidence={"value": 1})
