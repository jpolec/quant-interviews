from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json
import platform
import sqlite3
from typing import Callable

import numpy as np

from .dependency_graph import impacted_nodes, lineage_hash
from .lru_cache import CacheKey, LRUCache
from .order_book import PriceTimeOrderBook
from .profiling import portfolio_variance_vectorized
from .rolling_quantile import HistogramQuantile
from .stable_api import (
    MarketSnapshot,
    PriceRequest,
    Quote,
    canonical_json,
    price_cash_instrument,
)
from .temporal_sql import VersionedValue, point_in_time_lookup
from .topk_stream import StreamingTopK
from .vectorization import portfolio_returns


SCHEMA_VERSION = 1
CONFIG = {
    "chapter": "CH11",
    "contract_version": "0.10.0",
    "deployment_latency_policy": "NOT_EVALUATED_WITHOUT_TARGET_TRACE",
    "event_sequence_origin": 1,
    "histogram_range": ["-1", "1"],
    "histogram_bins": 20,
}
GATE_NAMES = (
    "stateful_structures",
    "numerical_boundary",
    "temporal_sql_boundary",
    "event_replay",
    "api_evidence_boundary",
)


def _digest(payload: object) -> str:
    return sha256(canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class GateEvidence:
    name: str
    status: str
    evidence_hash: str
    error_code: str | None = None

    def as_payload(self) -> dict[str, object]:
        return {
            "error_code": self.error_code,
            "evidence_hash": self.evidence_hash,
            "name": self.name,
            "status": self.status,
        }


@dataclass(frozen=True)
class IntegratedDecision:
    schema_version: int
    fixture_hash: str
    config_hash: str
    tool_versions: tuple[tuple[str, str], ...]
    gates: tuple[GateEvidence, ...]
    final_decision: str
    deployment_latency_decision: str
    decision_hash: str

    def as_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "config_hash": self.config_hash,
            "deployment_latency_decision": self.deployment_latency_decision,
            "final_decision": self.final_decision,
            "fixture_hash": self.fixture_hash,
            "gates": [gate.as_payload() for gate in self.gates],
            "schema_version": self.schema_version,
            "tool_versions": [list(item) for item in self.tool_versions],
        }
        if include_hash:
            payload["decision_hash"] = self.decision_hash
        return payload


def _stateful_gate(fixture_hash: str) -> dict[str, object]:
    stream = StreamingTopK[str](2)
    for score, value, event_id in (
        (1.0, "alpha", "event-1"),
        (3.0, "gamma", "event-2"),
        (2.0, "beta", "event-3"),
    ):
        stream.push(score, value, event_id=event_id)
    checkpoint = stream.checkpoint()
    restored = StreamingTopK.from_checkpoint(checkpoint)
    if restored.snapshot() != ((3.0, "gamma"), (2.0, "beta")):
        raise AssertionError("top-k checkpoint did not replay exactly")

    histogram = HistogramQuantile(
        -1.0,
        1.0,
        20,
        0.5,
        window=4,
        min_periods=2,
    )
    histogram.update(np.array([-0.2, 0.2], dtype=np.float64))
    before = histogram.diagnostics().observations
    try:
        histogram.update(np.array([0.1, 7.0], dtype=np.float64))
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range histogram batch was accepted")
    if histogram.diagnostics().observations != before:
        raise AssertionError("rejected histogram batch partially mutated state")

    key = CacheKey(
        artifact="integrated-fixture",
        input_hash=fixture_hash,
        knowledge_cutoff="2026-01-03T09:00:00Z",
        code_version="ch11-integrated-v0.10.0",
        config_hash=_digest(CONFIG),
    )
    cache = LRUCache[CacheKey, str](1)
    cached = cache.get_or_compute(key, lambda: fixture_hash)
    invalidated = impacted_nodes(
        (("raw", "features"), ("features", "decision")),
        ("features",),
    )
    lineage = lineage_hash(
        "decision",
        (("features", fixture_hash),),
        artifact_hash=fixture_hash,
        code_hash=_digest("integrated-case-code-v0.10.0"),
        config_hash=_digest(CONFIG),
        schema_hash=_digest("integrated-case-schema-v1"),
        parameters_hash=_digest("integrated-case-parameters-v1"),
    )
    return {
        "cache_replayed": cached == fixture_hash,
        "histogram_observations": before,
        "impacted": list(invalidated),
        "lineage_hash": lineage,
        "seen_event_count": checkpoint.next_sequence,
        "top_values": [item[1] for item in restored.snapshot()],
    }


def _numerical_gate() -> dict[str, object]:
    weights = np.array([0.4, 0.6], dtype=np.float64)
    returns = np.array([[0.01, -0.02], [0.03, 0.01]], dtype=np.float64)
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=np.float64)
    portfolio = portfolio_returns(weights, returns)
    variance = portfolio_variance_vectorized(weights, covariance)
    if not np.all(np.isfinite(portfolio)) or not np.isfinite(variance):
        raise FloatingPointError("integrated numerical gate produced non-finite output")
    return {
        "portfolio_returns_hex": [float(value).hex() for value in portfolio],
        "variance_hex": variance.hex(),
    }


def _temporal_sql_gate() -> dict[str, object]:
    utc = timezone.utc
    valid = datetime(2026, 1, 2, tzinfo=utc)
    early_knowledge = datetime(2026, 1, 2, 8, tzinfo=utc)
    late_knowledge = datetime(2026, 1, 4, 8, tzinfo=utc)
    rows = tuple(
        VersionedValue(
            natural_key="X",
            valid_time=valid,
            knowledge_time=known,
            value=value,
            version=version,
            source_id="feed-a",
            row_hash=VersionedValue.content_hash_for(
                natural_key="X",
                valid_time=valid,
                knowledge_time=known,
                value=value,
                version=version,
                source_id="feed-a",
            ),
        )
        for known, value, version in (
            (early_knowledge, 100.0, 1),
            (late_knowledge, 101.0, 2),
        )
    )
    selected = point_in_time_lookup(
        rows,
        "X",
        datetime(2026, 1, 3, tzinfo=utc),
        datetime(2026, 1, 3, 9, tzinfo=utc),
    )
    if selected is None or selected.version != 1:
        raise AssertionError("Python PIT lookup selected future knowledge")

    root = Path(__file__).resolve().parents[4]
    sql_path = root / "_book/sql/ch11/point_in_time_asof.sql"
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(
            """
            CREATE TABLE decisions(
                decision_row_id TEXT, decision_id TEXT, instrument_id TEXT,
                effective_at TEXT, knowledge_at TEXT
            );
            CREATE TABLE market_data(
                instrument_id TEXT, effective_at TEXT, known_at TEXT,
                version INTEGER, value REAL, source_id TEXT, row_hash TEXT
            );
            INSERT INTO decisions VALUES
              ('decision-row-1','decision-1','X',
               '2026-01-03T00:00:00Z','2026-01-03T09:00:00Z');
            INSERT INTO market_data VALUES
              ('X','2026-01-02T00:00:00Z','2026-01-02T08:00:00Z',
               1,100.0,'feed-a','row-hash-1'),
              ('X','2026-01-02T00:00:00Z','2026-01-04T08:00:00Z',
               2,101.0,'feed-a','row-hash-2');
            """
        )
        sql_rows = connection.execute(sql_path.read_text(encoding="utf-8")).fetchall()
    finally:
        connection.close()
    if len(sql_rows) != 1 or sql_rows[0][-1] != "PASS" or sql_rows[0][5] != 100.0:
        raise AssertionError("SQL PIT gate disagreed with the governed cutoff")
    return {
        "python_row_hash": selected.row_hash,
        "python_version": selected.version,
        "sql_row_count": len(sql_rows),
        "sql_status": sql_rows[0][-1],
        "sql_value_hex": float(sql_rows[0][5]).hex(),
    }


def _event_replay_gate() -> dict[str, object]:
    book = PriceTimeOrderBook()
    first = book.add_limit(101, "sell", 10_100, 5, sequence=1)
    second = book.execute_market(202, "buy", 3, sequence=2)
    cancelled = book.cancel(101, sequence=3)
    if first.status != "RESTING" or second.status != "FILLED" or not cancelled:
        raise AssertionError("order event replay did not reach the expected state")
    if second.filled_quantity != 3 or book.best_ask is not None:
        raise AssertionError("order replay failed quantity/state reconciliation")
    return {
        "filled_quantity": second.filled_quantity,
        "last_event_sequence": 3,
        "state_hash": book.state_hash(),
    }


def _api_gate() -> dict[str, object]:
    valuation_time = datetime(2026, 1, 3, 9, tzinfo=timezone.utc)
    snapshot = MarketSnapshot(
        snapshot_id="snapshot-1",
        valid_time=datetime(2026, 1, 3, 8, tzinfo=timezone.utc),
        knowledge_time=datetime(2026, 1, 3, 8, 30, tzinfo=timezone.utc),
        quotes=(Quote("CASH", Decimal("101.25"), "USD"),),
    )
    request = PriceRequest(
        request_id="request-1",
        idempotency_key="idem-1",
        instrument_id="CASH",
        quantity=2.0,
        currency="USD",
        side="long",
        valuation_time=valuation_time,
        knowledge_cutoff=valuation_time,
    )
    result = price_cash_instrument(request, snapshot, model_version="cash-v1")
    if result.value != Decimal("202.50") or result.status != "PASS":
        raise AssertionError("stable API produced an unexpected economic result")
    return {
        "output_hash": result.output_hash,
        "request_hash": result.request_hash,
        "snapshot_artifact_hash": result.snapshot_artifact_hash,
        "snapshot_content_hash": result.snapshot_content_hash,
        "status": result.status,
        "value": str(result.value),
    }


def _run_gate(
    name: str,
    action: Callable[[], dict[str, object]],
    fault_gate: str | None,
) -> GateEvidence:
    try:
        if fault_gate == name:
            raise RuntimeError("INJECTED_FAULT")
        evidence = action()
    except Exception as error:  # the evidence reducer must fail closed
        rejected = {
            "error_code": f"{type(error).__name__}:{error}",
            "name": name,
            "status": "REJECT",
        }
        return GateEvidence(
            name=name,
            status="REJECT",
            evidence_hash=_digest(rejected),
            error_code=str(rejected["error_code"]),
        )
    accepted = {"evidence": evidence, "name": name, "status": "PASS"}
    return GateEvidence(name=name, status="PASS", evidence_hash=_digest(accepted))


def run_integrated_case(*, fault_gate: str | None = None) -> IntegratedDecision:
    """Execute one immutable CH11 evidence fixture and reduce its closed gates."""

    if fault_gate is not None and fault_gate not in GATE_NAMES:
        raise ValueError(f"unknown fault gate: {fault_gate}")
    fixture = {
        "config": CONFIG,
        "economic_fixture": {
            "asset": "X",
            "currency": "USD",
            "decision_time": "2026-01-03T09:00:00Z",
            "prices": ["100", "101"],
        },
        "schema_version": SCHEMA_VERSION,
    }
    fixture_hash = _digest(fixture)
    config_hash = _digest(CONFIG)
    actions: tuple[tuple[str, Callable[[], dict[str, object]]], ...] = (
        ("stateful_structures", lambda: _stateful_gate(fixture_hash)),
        ("numerical_boundary", _numerical_gate),
        ("temporal_sql_boundary", _temporal_sql_gate),
        ("event_replay", _event_replay_gate),
        ("api_evidence_boundary", _api_gate),
    )
    gates = tuple(_run_gate(name, action, fault_gate) for name, action in actions)
    final_decision = (
        "PORTABLE_EVIDENCE_PASS"
        if all(gate.status == "PASS" for gate in gates)
        else "REJECT"
    )
    tool_versions = (
        ("numpy", np.__version__),
        ("python", platform.python_version()),
        ("sqlite", sqlite3.sqlite_version),
    )
    unsigned = IntegratedDecision(
        schema_version=SCHEMA_VERSION,
        fixture_hash=fixture_hash,
        config_hash=config_hash,
        tool_versions=tool_versions,
        gates=gates,
        final_decision=final_decision,
        deployment_latency_decision="NOT_EVALUATED",
        decision_hash="",
    )
    return IntegratedDecision(
        **{
            **unsigned.__dict__,
            "decision_hash": _digest(unsigned.as_payload(include_hash=False)),
        }
    )


if __name__ == "__main__":
    print(json.dumps(run_integrated_case().as_payload(), indent=2, sort_keys=True))
