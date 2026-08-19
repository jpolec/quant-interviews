from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

from quant_interviews.ch11.dependency_graph import lineage_hash
from quant_interviews.ch11.grouped_calculations import grouped_lagged_return
from quant_interviews.ch11.integrated_case import run_integrated_case
from quant_interviews.ch11.complexity_locality import (
    BenchmarkStats,
    column_major_python_sum,
    row_major_python_sum,
    stride_sum,
    timed,
)
from quant_interviews.ch11.lru_cache import LRUCache
from quant_interviews.ch11.numerical_testing import (
    assert_translation_invariance,
    assert_metamorphic_relation,
    differential_test,
)
from quant_interviews.ch11.rolling_quantile import HistogramQuantile
from quant_interviews.ch11.order_book import PriceTimeOrderBook
from quant_interviews.ch11.profiling import profile, portfolio_variance_vectorized
from quant_interviews.ch11.topk_stream import StreamingTopK
from quant_interviews.ch11.vectorization import portfolio_returns, scenario_pnl
from quant_interviews.ch11.stable_api import PriceResult, canonical_json
from quant_interviews.ch11.temporal_sql import VersionedValue
from decimal import Decimal

FIXTURE_HASH = "f" * 64


def test_streaming_top_k_rejects_reused_evicted_and_zero_capacity_ids() -> None:
    top = StreamingTopK[str](1)
    top.push(1.0, "evicted", event_id="event-a")
    top.push(2.0, "retained", event_id="event-b")
    with pytest.raises(ValueError, match="duplicate event_id"):
        top.push(3.0, "reused", event_id="event-a")

    zero = StreamingTopK[str](0)
    zero.push(1.0, "discarded", event_id="event-zero")
    with pytest.raises(ValueError, match="duplicate event_id"):
        zero.push(2.0, "reused", event_id="event-zero")
    assert zero.seen_event_count == 1

    restarted = StreamingTopK.from_checkpoint(top.checkpoint())
    assert restarted.snapshot() == top.snapshot()
    with pytest.raises(ValueError, match="duplicate event_id"):
        restarted.push(9.0, "conflicting retry", event_id="event-a")


def test_lru_same_thread_same_key_recursion_fails_instead_of_deadlocking() -> None:
    cache = LRUCache[str, int](2)

    def recursive_factory() -> int:
        return cache.get_or_compute("same", lambda: 7)

    with pytest.raises(RuntimeError, match="recursive single-flight"):
        cache.get_or_compute("same", recursive_factory)

    # The failed in-flight state is removed and a later independent build works.
    assert cache.get_or_compute("same", lambda: 11) == 11


def test_histogram_bulk_update_is_atomic_on_out_of_range_value() -> None:
    histogram = HistogramQuantile(
        lower=0.0,
        upper=1.0,
        bins=10,
        quantile=0.5,
        window=10,
        min_periods=1,
    )
    with pytest.raises(ValueError, match="outside the governed range"):
        histogram.update(np.array([0.5, 7.0], dtype=np.float64))
    diagnostics = histogram.diagnostics()
    assert diagnostics.observations == 0
    assert diagnostics.above_range == 1


def test_differential_and_metamorphic_checks_detect_input_mutation() -> None:
    values = np.arange(5.0)

    def mutating_candidate(x: np.ndarray) -> np.ndarray:
        answer = np.square(x)
        x[:] = 0.0
        return answer

    with pytest.raises(AssertionError, match="candidate mutated"):
        differential_test(np.square, mutating_candidate, values)

    def mutating_scalar(x: np.ndarray) -> float:
        result = float(np.mean(x))
        x[:] = 0.0
        return result

    with pytest.raises(AssertionError, match="mutated the base fixture"):
        assert_metamorphic_relation(
            lambda x: x + 1.0,
            lambda result: result + 1.0,
            mutating_scalar,
            values,
        )


def test_python_zero_denominator_policy_is_explicit_not_default_isclose() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"], utc=True
            ),
            "asset": ["A", "A"],
            "value": [1.0e-9, 1.0],
        }
    )
    result = grouped_lagged_return(frame)
    assert result.iloc[1]["return"] == pytest.approx(
        999_999_999.0,
        rel=1.0e-15,
        abs=0.0,
    )


def test_vectorized_boundaries_reject_implicit_copy_and_nonfinite_output() -> None:
    with pytest.raises(TypeError, match="implicit conversion"):
        portfolio_returns(  # type: ignore[arg-type]
            [0.5, 0.5],
            np.ones((2, 2), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="dtype float64"):
        portfolio_returns(
            np.array([0.5, 0.5], dtype=np.float32),
            np.ones((2, 2), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        scenario_pnl(
            np.ones((2, 2), dtype=np.float64).T,
            np.ones((2, 2), dtype=np.float64),
        )
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="non-finite output"):
            portfolio_returns(
                np.array([1.0e200], dtype=np.float64),
                np.array([[1.0e200]], dtype=np.float64),
            )


def test_lineage_hash_requires_code_config_and_parameter_bindings() -> None:
    with pytest.raises(ValueError, match="code, config and parameter"):
        lineage_hash(
            "node",
            [("parent", "a" * 64)],
            artifact_hash="d" * 64,
            code_hash="",
            config_hash="b" * 64,
            schema_hash="e" * 64,
            parameters_hash="c" * 64,
        )


def test_stable_api_uses_utf16_key_order_and_rejects_forged_pass() -> None:
    encoded = canonical_json({"\U00010000": 1, "\uff61": 2}).decode("utf-8")
    assert encoded.index("\U00010000") < encoded.index("\uff61")
    with pytest.raises(ValueError, match="finite"):
        PriceResult(
            value=Decimal("NaN"),
            currency="USD",
            request_hash="a" * 64,
            snapshot_content_hash="b" * 64,
            snapshot_artifact_hash="c" * 64,
            model_version="1.0.0",
            schema_version=1,
            status="PASS",
        )


def test_benchmark_records_bind_environment_and_semantic_result() -> None:
    first = profile(
        "constant", lambda: 3.0, repeats=3, warmups=1, input_hash=FIXTURE_HASH
    )
    second = profile(
        "constant", lambda: 3.0, repeats=5, warmups=1, input_hash=FIXTURE_HASH
    )
    assert first.result_hash == second.result_hash
    assert first.warmups == 1 and first.repeats == 3
    assert dict(first.environment)["numpy_version"] == np.__version__
    with pytest.raises(ValueError, match="at least one"):
        timed(
            "invalid",
            lambda: 1.0,
            1,
            repeats=3,
            warmups=0,
            input_hash=FIXTURE_HASH,
        )


def test_portfolio_variance_rejects_nonfinite_output() -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="non-finite"):
            portfolio_variance_vectorized(
                np.array([1.0e200], dtype=np.float64),
                np.array([[1.0]], dtype=np.float64),
            )


def test_order_book_requires_contiguous_sequenced_cancellation_and_complete_state() -> None:
    fresh = PriceTimeOrderBook()
    with pytest.raises(ValueError, match="next contiguous"):
        fresh.add_limit(1, "buy", 100, 1, sequence=2)

    book = PriceTimeOrderBook()
    book.add_limit(1, "buy", 100, 1, sequence=1)
    with pytest.raises(ValueError, match="next contiguous"):
        book.add_limit(2, "buy", 99, 1, sequence=3)
    before_cancel = book.state_hash()
    assert book.cancel(1, sequence=2)
    assert book.state_hash() != before_cancel
    assert book.state_hash() != fresh.state_hash()


def test_temporal_row_rejects_empty_lineage_hash() -> None:
    with pytest.raises(ValueError, match="row_hash"):
        VersionedValue(
            natural_key="X",
            valid_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            knowledge_time=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
            value=1.0,
            version=1,
            source_id="feed",
            row_hash="",
        )


def test_integrated_case_is_deterministic_and_fails_closed() -> None:
    first = run_integrated_case()
    second = run_integrated_case()
    assert first.as_payload() == second.as_payload()
    assert first.final_decision == "PORTABLE_EVIDENCE_PASS"
    assert first.deployment_latency_decision == "NOT_EVALUATED"
    assert len(first.fixture_hash) == len(first.config_hash) == len(first.decision_hash) == 64
    assert all(gate.status == "PASS" for gate in first.gates)

    fault = run_integrated_case(fault_gate="event_replay")
    assert fault.final_decision == "REJECT"
    assert [gate.status for gate in fault.gates].count("REJECT") == 1
    rejected = next(gate for gate in fault.gates if gate.status == "REJECT")
    assert rejected.name == "event_replay"
    assert rejected.error_code == "RuntimeError:INJECTED_FAULT"

    with pytest.raises(ValueError, match="unknown fault gate"):
        run_integrated_case(fault_gate="not-a-gate")


def test_every_previously_uncovered_printed_symbol_has_direct_evidence() -> None:
    matrix = np.arange(1.0, 13.0, dtype=np.float64).reshape(3, 4)
    assert column_major_python_sum(matrix) == row_major_python_sum(matrix) == 78.0
    values = np.arange(1.0, 7.0, dtype=np.float64)
    assert stride_sum(values, 2) == 9.0
    with pytest.raises(ValueError, match="stride must be positive"):
        stride_sum(values, 0)

    assert_translation_invariance(np.mean, values, 7.0)
    with pytest.raises(AssertionError, match="metamorphic relation failed"):
        assert_translation_invariance(lambda x: float(np.mean(np.square(x))), values, 1.0)

    stats = BenchmarkStats(
        label="direct-property",
        samples_seconds=(0.5, 0.5, 0.5),
        median_seconds=0.5,
        p95_seconds=0.5,
        p99_seconds=0.5,
        elements=2,
        result=1.0,
        warmup_results=(1.0,),
        warmups=1,
        repeats=3,
        result_hash="a" * 64,
        measurement_hash="b" * 64,
        input_hash="c" * 64,
        clock="test-clock",
        percentile_method="test-method",
        environment=(("test", "fixture"),),
    )
    assert stats.median_nanoseconds_per_element == 250_000_000.0
