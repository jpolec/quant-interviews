from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import pytest

from quant_interviews.ch11.complexity_locality import (
    amdahl_speedup,
    row_major_python_sum,
    timed,
)
from quant_interviews.ch11.dependency_graph import (
    find_cycle,
    impacted_nodes,
    lineage_hash,
    topological_order,
)
from quant_interviews.ch11.floating_point import (
    Tolerance,
    almost_equal,
    kahan_sum,
    stable_quadratic_roots,
)
from quant_interviews.ch11.grouped_calculations import (
    grouped_lagged_return,
    grouped_rolling_zscore,
)
from quant_interviews.ch11.lru_cache import CacheKey, LRUCache
from quant_interviews.ch11.numerical_testing import (
    assert_metamorphic_relation,
    check_monte_carlo_mean,
    differential_test,
)
from quant_interviews.ch11.order_book import PriceTimeOrderBook
from quant_interviews.ch11.profiling import (
    amdahl_asymptotic_limit,
    amdahl_speedup_from_serial_fraction,
    portfolio_variance_loop,
    portfolio_variance_vectorized,
    profile,
)
from quant_interviews.ch11.rolling_quantile import ExactRollingQuantile, HistogramQuantile
from quant_interviews.ch11.stable_api import (
    MarketSnapshot,
    PriceRequest,
    PricingError,
    Quote,
    price_cash_instrument,
)
from quant_interviews.ch11.temporal_sql import VersionedValue, point_in_time_lookup
from quant_interviews.ch11.topk_stream import StreamingTopK, top_k_stream
from quant_interviews.ch11.vectorization import portfolio_returns, safe_standardize, scenario_pnl
from quant_interviews.ch11.window_metrics import returns_and_cross_sectional_rank

REPO_ROOT = Path(__file__).resolve().parents[4]
SQL = REPO_ROOT / "_book" / "sql" / "ch11"
CPP = REPO_ROOT / "_book" / "cpp" / "ch11"
FIXTURE_HASH = "f" * 64


def versioned_value(
    key: str,
    valid_time: datetime,
    knowledge_time: datetime,
    value: float,
    version: int,
    source_id: str,
) -> VersionedValue:
    row_hash = VersionedValue.content_hash_for(
        natural_key=key,
        valid_time=valid_time,
        knowledge_time=knowledge_time,
        value=value,
        version=version,
        source_id=source_id,
    )
    return VersionedValue(
        key,
        valid_time,
        knowledge_time,
        value,
        version,
        source_id,
        row_hash,
    )


def test_benchmark_reports_distribution_after_warmup() -> None:
    matrix = np.ones((8, 8), dtype=np.float64, order="C")
    result = timed(
        "row",
        lambda: row_major_python_sum(matrix),
        64,
        repeats=5,
        warmups=1,
        input_hash=FIXTURE_HASH,
    )
    assert len(result.samples_seconds) == 5
    assert result.median_seconds <= result.p95_seconds <= result.p99_seconds
    assert result.result == 64.0


def test_amdahl_speedup_limits() -> None:
    assert amdahl_speedup(0.0, 64) == 1.0
    assert amdahl_speedup(1.0, 8) == 8.0


def test_top_k_stream_matches_stable_full_sort() -> None:
    items = [(3.0, "a"), (5.0, "first"), (5.0, "second"), (4.0, "d")]
    assert top_k_stream((item for item in items), 2) == [(5.0, "first"), (5.0, "second")]


def test_streaming_top_k_rejects_nonfinite_and_duplicate_event() -> None:
    top = StreamingTopK[str](2)
    top.push(1.0, "first", event_id="a")
    with pytest.raises(ValueError):
        top.push(float("nan"), "bad")
    with pytest.raises(ValueError):
        top.push(2.0, "duplicate", event_id="a")


def test_exact_rolling_quantile_endpoints_and_expiry() -> None:
    minimum = ExactRollingQuantile(window=3, quantile=0.0, min_periods=1)
    maximum = ExactRollingQuantile(window=3, quantile=1.0, min_periods=1)
    observed_min = [minimum.push(value) for value in [3.0, 1.0, 2.0, 10.0]]
    observed_max = [maximum.push(value) for value in [3.0, 1.0, 2.0, 10.0]]
    assert observed_min == [3.0, 1.0, 1.0, 1.0]
    assert observed_max == [3.0, 3.0, 3.0, 10.0]


def test_histogram_quantile_is_rolling_and_range_governed() -> None:
    estimate = HistogramQuantile(
        lower=-5.0,
        upper=5.0,
        bins=100,
        quantile=0.5,
        window=101,
        min_periods=101,
    )
    estimate.update(np.linspace(-1.0, 1.0, 101))
    assert abs(estimate.value()) <= estimate.diagnostics().bin_width
    for value in np.full(101, 4.0):
        estimate.push(float(value))
    assert estimate.value() > 3.9
    with pytest.raises(ValueError):
        estimate.push(6.0)
    assert estimate.diagnostics().above_range == 1


def test_lru_supports_cached_none_without_miss_ambiguity() -> None:
    cache = LRUCache[str, int | None](2)
    cache.put("none", None)
    lookup = cache.lookup("none")
    assert lookup.hit and lookup.value is None
    assert not cache.lookup("missing").hit


def test_lru_uses_a_complete_semantic_cache_key() -> None:
    key = CacheKey(
        artifact="curve",
        input_hash="a" * 64,
        knowledge_cutoff="2026-01-01T00:00:00Z",
        code_version="curve-v1",
        config_hash="b" * 64,
    )
    cache = LRUCache[CacheKey, int](1)
    assert cache.get_or_compute(key, lambda: 7) == 7
    assert cache.get(key) == 7


def test_lru_single_flight_runs_one_builder() -> None:
    cache = LRUCache[str, int](2)
    calls = 0
    lock = threading.Lock()

    def factory() -> int:
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.02)
        return 7

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: cache.get_or_compute("x", factory), range(8)))
    assert values == [7] * 8
    assert calls == 1
    assert cache.stats().builds == 1
    assert cache.stats().waits == 7


def test_dependency_topological_order_is_generator_and_edge_order_safe() -> None:
    edges = [("raw", "clean"), ("clean", "features"), ("features", "model")]
    expected = ["isolated", "raw", "clean", "features", "model"]
    assert topological_order((edge for edge in edges), nodes=["isolated"]) == expected
    assert topological_order(reversed(edges), nodes=["isolated"]) == expected


def test_dependency_cycle_and_lineage_hash_are_canonical() -> None:
    cycle = find_cycle((edge for edge in [("a", "b"), ("b", "c"), ("c", "a")]))
    assert cycle[0] == cycle[-1]
    first = lineage_hash(
        "c",
        [("b", "2" * 64), ("a", "1" * 64)],
        artifact_hash="6" * 64,
        code_hash="3" * 64,
        config_hash="4" * 64,
        schema_hash="7" * 64,
        parameters_hash="5" * 64,
    )
    second = lineage_hash(
        "c",
        [("a", "1" * 64), ("b", "2" * 64)],
        artifact_hash="6" * 64,
        code_hash="3" * 64,
        config_hash="4" * 64,
        schema_hash="7" * 64,
        parameters_hash="5" * 64,
    )
    assert first == second
    assert impacted_nodes(
        [("raw", "clean"), ("clean", "features"), ("other", "features")],
        ["clean"],
    ) == ("clean", "features")


def test_almost_equal_mixed_tolerance_and_nonfinite_policy() -> None:
    tolerance = Tolerance(absolute=1e-12, relative=1e-8)
    assert almost_equal(1e8, 1e8 + 0.5, tolerance)
    assert not almost_equal(float("inf"), float("inf"), tolerance)


def test_kahan_sum_beats_naive_on_many_small_terms() -> None:
    values = [1.0] + [1e-16] * 100_000
    reference = float(np.longdouble(1.0) + np.longdouble("1e-16") * 100_000)
    naive = 0.0
    for value in values:
        naive += value
    assert abs(kahan_sum(values) - reference) < abs(naive - reference)


def test_stable_quadratic_roots_scale_and_vieta() -> None:
    x1, x2 = stable_quadratic_roots(1e200, 1e208, 1e200)
    assert x1 * x2 == pytest.approx(1.0, rel=1e-11, abs=0.0)
    assert x1 + x2 == pytest.approx(-1e8, rel=1e-15, abs=0.0)


def test_vectorization_validates_shapes_and_finiteness() -> None:
    weights = np.array([0.4, 0.6])
    returns = np.array([[0.01, -0.02], [0.03, 0.01]])
    np.testing.assert_allclose(
        portfolio_returns(weights, returns),
        np.array([-0.008, 0.018]),
        rtol=1e-15,
        atol=1e-15,
    )
    with pytest.raises(ValueError):
        scenario_pnl(np.array([[np.nan]]), np.array([[1.0]]))
    with pytest.raises(ValueError):
        safe_standardize(np.ones((1, 2)), axis=0)


def sample_prices() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=5, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": list(timestamps) * 2,
            "asset": ["A"] * 5 + ["B"] * 5,
            "value": [100.0, 101.0, 103.0, 102.0, 106.0, 50.0, 49.0, 51.0, 52.0, 51.0],
        }
    )


def test_grouped_returns_zero_policy_and_return_zscore() -> None:
    frame = sample_prices()
    frame.loc[0, "value"] = 0.0
    result = grouped_rolling_zscore(frame, window=3)
    a = result[result["asset"] == "A"]
    assert np.isnan(a.iloc[1]["return"])
    assert np.isfinite(a.iloc[-1]["zscore"])


def test_point_in_time_lookup_total_order_and_duplicate_gate() -> None:
    utc = timezone.utc
    rows = [
        versioned_value("X", datetime(2026, 1, 1, tzinfo=utc), datetime(2026, 1, 1, 9, tzinfo=utc), 100.0, 1, "a"),
        versioned_value("X", datetime(2026, 1, 1, tzinfo=utc), datetime(2026, 1, 1, 9, tzinfo=utc), 101.0, 1, "b"),
    ]
    selected = point_in_time_lookup(
        rows,
        "X",
        datetime(2026, 1, 1, 12, tzinfo=utc),
        datetime(2026, 1, 1, 12, tzinfo=utc),
    )
    assert selected is not None and selected.source_id == "b"
    with pytest.raises(ValueError):
        point_in_time_lookup(
            [rows[0], rows[0]],
            "X",
            datetime(2026, 1, 1, 12, tzinfo=utc),
            datetime(2026, 1, 1, 12, tzinfo=utc),
        )


def test_window_percent_rank_matches_sql_tie_semantics() -> None:
    timestamp = pd.Timestamp("2026-01-02", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01", tz="UTC"),
                timestamp,
                pd.Timestamp("2026-01-01", tz="UTC"),
                timestamp,
                pd.Timestamp("2026-01-01", tz="UTC"),
                timestamp,
            ],
            "asset": ["A", "A", "B", "B", "C", "C"],
            "price": [100.0, 101.0, 100.0, 101.0, 0.0, 1.0],
        }
    )
    result = returns_and_cross_sectional_rank(frame)
    final = result[result["timestamp"] == timestamp].sort_values("asset")
    assert final["rank"].tolist()[:2] == [0.0, 0.0]
    assert np.isnan(final["rank"].iloc[2])


def test_order_book_marketable_limit_matches_before_resting() -> None:
    book = PriceTimeOrderBook()
    book.add_limit(1, "sell", 101, 10, sequence=1)
    book.add_limit(2, "sell", 101, 20, sequence=2)
    result = book.add_limit(3, "buy", 102, 15, sequence=3)
    assert [(fill.maker_order_id, fill.quantity) for fill in result.fills] == [(1, 10), (2, 5)]
    assert result.remaining_quantity == 0
    assert book.best_ask == 101
    assert book.best_bid is None


def test_order_book_cancel_duplicate_side_and_quantity_contracts() -> None:
    book = PriceTimeOrderBook()
    book.add_limit(1, "buy", 100, 10, sequence=1)
    book.add_limit(2, "buy", 101, 10, sequence=2)
    assert book.cancel(2)
    assert book.best_bid == 100
    with pytest.raises(ValueError):
        book.add_limit(1, "sell", 110, 1, sequence=3)
    with pytest.raises(ValueError):
        book.execute_market(3, "invalid", 1, sequence=3)  # type: ignore[arg-type]


def test_monte_carlo_and_metamorphic_checks() -> None:
    def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(size=n)

    first = check_monte_carlo_mean(
        sampler, 0.0, 50_000, seed=7, iid_finite_variance_verified=True
    )
    second = check_monte_carlo_mean(
        sampler, 0.0, 50_000, seed=7, iid_finite_variance_verified=True
    )
    assert first == second and first.passed
    assert_metamorphic_relation(
        lambda x: x + 2.0,
        lambda result: result + 2.0,
        lambda x: float(np.mean(x)),
        np.arange(5.0),
    )


def test_differential_test_uses_independent_fixtures() -> None:
    values = np.arange(5.0)
    differential_test(lambda x: np.square(x), lambda x: x * x, values)
    with pytest.raises(AssertionError):
        differential_test(lambda x: np.square(x), lambda x: x * x + 0.1, values)


def stable_api_fixture() -> tuple[MarketSnapshot, PriceRequest]:
    utc = timezone.utc
    snapshot = MarketSnapshot(
        snapshot_id="snapshot-2026-01-01",
        valid_time=datetime(2026, 1, 1, 10, tzinfo=utc),
        knowledge_time=datetime(2026, 1, 1, 10, 1, tzinfo=utc),
        quotes=(Quote("ABC", Decimal("42.00"), "USD"),),
    )
    request = PriceRequest(
        request_id="request-1",
        idempotency_key="idem-1",
        instrument_id="ABC",
        quantity=10.0,
        currency="USD",
        side="long",
        valuation_time=datetime(2026, 1, 1, 10, tzinfo=utc),
        knowledge_cutoff=datetime(2026, 1, 1, 10, 2, tzinfo=utc),
    )
    return snapshot, request


def test_stable_api_exact_hash_price_and_immutability() -> None:
    snapshot, request = stable_api_fixture()
    result = price_cash_instrument(request, snapshot)
    assert result.value == Decimal("420")
    assert result.snapshot_content_hash == (
        "8feb92e21123ed1ccb6c73356b99721aa8d11855e74a8d666a89f4e7b7fc1ea6"
    )
    assert result.request_hash == (
        "5304864e1398a338950ea2714e141c2db6b726122eca1f63fe2fa564d533bc43"
    )
    assert len(result.implementation_hash) == 64
    assert len(result.output_hash) == 64
    assert snapshot.quotes == (Quote("ABC", Decimal("42"), "USD"),)


def test_stable_api_currency_and_temporal_gates() -> None:
    snapshot, request = stable_api_fixture()
    wrong_currency = PriceRequest(
        **{**request.__dict__, "currency": "EUR", "request_id": "request-2"}
    )
    with pytest.raises(PricingError, match="currency"):
        price_cash_instrument(wrong_currency, snapshot)


def test_profile_reports_tail_and_result_hash() -> None:
    stats = profile(
        "constant", lambda: 3.0, repeats=5, warmups=1, input_hash=FIXTURE_HASH
    )
    assert stats.median_seconds <= stats.p95_seconds <= stats.p99_seconds
    assert len(stats.result_hash) == 64


def test_portfolio_variance_loop_matches_vectorized_and_rejects_indefinite() -> None:
    weights = np.array([0.5, 0.3, 0.2])
    covariance = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
    expected = portfolio_variance_vectorized(weights, covariance)
    assert portfolio_variance_loop(weights, covariance) == pytest.approx(
        expected, rel=1e-14, abs=1e-16
    )
    with pytest.raises(ValueError):
        portfolio_variance_vectorized(np.array([1.0, 1.0]), np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert amdahl_speedup_from_serial_fraction(0.25, 1000) < 4.0
    assert amdahl_asymptotic_limit(0.25) == 4.0


def execute_sql(script_name: str, setup: str, params: dict | None = None) -> list[tuple]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(setup)
        script = (SQL / script_name).read_text(encoding="utf-8")
        return connection.execute(script, params or {}).fetchall()
    finally:
        connection.close()


def test_python_sql_grouped_calculation_parity() -> None:
    frame = sample_prices()
    python = grouped_rolling_zscore(frame, window=3)
    rows = ",".join(
        f"('{row.asset}','{row.timestamp.isoformat()}',{row.value})"
        for row in frame.itertuples(index=False)
    )
    sql_rows = execute_sql(
        "grouped_calculations.sql",
        "CREATE TABLE prices(asset TEXT, timestamp TEXT, price REAL);"
        f"INSERT INTO prices VALUES {rows};",
    )
    sql = pd.DataFrame(
        sql_rows,
        columns=["asset", "timestamp", "price", "previous", "return", "zscore", "status"],
    )
    sql["timestamp"] = pd.to_datetime(sql["timestamp"], utc=True)
    merged = python.merge(sql, on=["asset", "timestamp"], suffixes=("_py", "_sql"))
    assert merged["status"].eq("PASS").all()
    assert np.array_equal(merged["return_py"].isna(), merged["return_sql"].isna())
    assert np.allclose(
        merged["return_py"].fillna(0.0),
        merged["return_sql"].fillna(0.0),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.array_equal(merged["zscore_py"].isna(), merged["zscore_sql"].isna())
    assert np.allclose(
        merged["zscore_py"].fillna(0.0),
        merged["zscore_sql"].fillna(0.0),
        rtol=1e-10,
        atol=1e-12,
    )


def test_sql_point_in_time_preserves_lineage_and_missing_decision() -> None:
    setup = """
    CREATE TABLE decisions(
        decision_row_id TEXT, decision_id TEXT, instrument_id TEXT,
        effective_at TEXT, knowledge_at TEXT
    );
    CREATE TABLE market_data(
        instrument_id TEXT, effective_at TEXT, known_at TEXT, version INTEGER,
        value REAL, source_id TEXT, row_hash TEXT
    );
    INSERT INTO decisions VALUES
      ('dr1','d1','X','2026-01-01T12:00:00Z','2026-01-01T12:00:00Z'),
      ('dr2','d2','Y','2026-01-01T12:00:00Z','2026-01-01T12:00:00Z');
    INSERT INTO market_data VALUES
      ('X','2026-01-01T10:00:00Z','2026-01-01T10:01:00Z',1,100.0,'feed-a','hash-a'),
      ('X','2026-01-01T10:00:00Z','2026-01-02T09:00:00Z',2,101.0,'feed-a','hash-b');
    """
    rows = execute_sql("point_in_time_asof.sql", setup)
    assert len(rows) == 2
    assert rows[0][5:] == (100.0, "2026-01-01T10:00:00Z", "2026-01-01T10:01:00Z", 1, "feed-a", "hash-a", "PASS")
    assert rows[1][-1] == "REJECT_MISSING_ROW"


def test_python_sql_rank_parity_with_ties_nulls_and_singletons() -> None:
    timestamps = [
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-01-02", tz="UTC"),
    ]
    frame = pd.DataFrame(
        {
            "timestamp": [timestamps[0], timestamps[1]] * 3,
            "asset": ["A", "A", "B", "B", "C", "C"],
            "price": [100.0, 101.0, 100.0, 101.0, 0.0, 1.0],
        }
    )
    python = returns_and_cross_sectional_rank(frame)
    sequence_by_asset: dict[str, int] = {}
    sql_values: list[str] = []
    for row in frame.itertuples(index=False):
        sequence = sequence_by_asset.get(row.asset, 0) + 1
        sequence_by_asset[row.asset] = sequence
        row_id = f"{row.asset}-{sequence}"
        sql_values.append(
            f"('{row.timestamp.isoformat()}','{row.asset}',{sequence},1,"
            f"'{row_id}','hash-{row_id}',{row.price})"
        )
    rows = ",".join(sql_values)
    sql_rows = execute_sql(
        "window_returns_ranks.sql",
        "CREATE TABLE prices("
        "timestamp TEXT, asset TEXT, sequence_number INTEGER, version_id INTEGER, "
        "row_id TEXT, row_hash TEXT, price REAL);"
        f"INSERT INTO prices VALUES {rows};",
    )
    sql = pd.DataFrame(
        sql_rows,
        columns=[
            "timestamp", "asset", "sequence_number", "version_id", "row_id",
            "row_hash", "price", "return_1", "rank", "mean", "status",
        ],
    )
    sql["timestamp"] = pd.to_datetime(sql["timestamp"], utc=True)
    merged = python.merge(sql, on=["timestamp", "asset"], suffixes=("_py", "_sql"))
    assert np.array_equal(merged["rank_py"].isna(), merged["rank_sql"].isna())
    np.testing.assert_array_equal(
        merged["rank_py"].fillna(0.0).to_numpy(),
        merged["rank_sql"].fillna(0.0).to_numpy(),
    )


def test_sql_feature_join_preserves_selected_lineage() -> None:
    setup = """
    CREATE TABLE decisions(
        decision_row_id TEXT, decision_id TEXT, instrument_id TEXT,
        event_time TEXT, decision_time TEXT
    );
    CREATE TABLE feature_versions(
        instrument_id TEXT, feature_name TEXT, feature_value REAL,
        effective_from TEXT, knowledge_from TEXT, version_id INTEGER,
        source_id TEXT, row_hash TEXT
    );
    INSERT INTO decisions VALUES ('dr1','d1','X','2026-01-03T00:00:00Z','2026-01-03T09:00:00Z');
    INSERT INTO feature_versions VALUES
      ('X','momentum',1.5,'2026-01-02T00:00:00Z','2026-01-02T08:00:00Z',1,'calc','hash-1'),
      ('X','momentum',9.9,'2026-01-02T00:00:00Z','2026-01-04T08:00:00Z',2,'calc','hash-2');
    """
    rows = execute_sql("point_in_time_feature_join.sql", setup)
    assert rows[0][5:] == (
        "momentum",
        1.5,
        "2026-01-02T00:00:00Z",
        "2026-01-02T08:00:00Z",
        1,
        "calc",
        "hash-1",
        "PASS",
    )


def _compile_and_run(source: Path) -> str:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    with tempfile.TemporaryDirectory() as directory:
        executable = Path(directory) / "fixture"
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                str(source),
                "-o",
                str(executable),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return subprocess.run(
            [str(executable)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()


def test_cpp_order_book_matches_python_fixture() -> None:
    output = _compile_and_run(CPP / "order_book.cpp")
    book = PriceTimeOrderBook()
    book.add_limit(1, "sell", 101, 10, sequence=1)
    book.add_limit(2, "sell", 101, 20, sequence=2)
    result = book.add_limit(3, "buy", 102, 15, sequence=3)
    assert (
        f"fills={len(result.fills)} remaining={result.remaining_quantity} "
        f"best_ask={book.best_ask} state_hash={book.state_hash()}"
    ) == output


def test_cpp_variance_kernel_matches_python_fixture() -> None:
    output = _compile_and_run(CPP / "risk_kernel.cpp")
    variance_text = output.split()[0].split("=")[1]
    weights = np.array([0.5, 0.3, 0.2])
    covariance = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
    assert float(variance_text) == pytest.approx(
        portfolio_variance_vectorized(weights, covariance),
        rel=1e-6,
        abs=1e-12,
    )
