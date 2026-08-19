from __future__ import annotations

import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ENUMERATION_SHA256 = "01ed3c62448dd6e0fd999d85cb79a2585454557cc1b50540668332f2b88642a0"


def _query(connection: sqlite3.Connection, filename: str) -> list[dict[str, object]]:
    sql = (ROOT / f"_book/sql/ch13/{filename}").read_text(encoding="utf-8")
    return [dict(row) for row in connection.execute(sql).fetchall()]


def validate_auction() -> list[dict[str, object]]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE structurally_scored_auction_bids (
          observation_id TEXT, auction_id TEXT, bidder_id TEXT,
          value REAL, bid REAL, bidders, risk_aversion REAL,
          value_source TEXT, model_version TEXT, tie_rule TEXT,
          as_of_utc TEXT, candidate_utility REAL,
          best_deviation_utility REAL, regret_tolerance REAL
        )
        """
    )
    rows = [
        ("o1", "a1", "A", 0.8, 0.6, 4, 0.0, "SIMULATED", "v-good",
         "uniform_among_highest_bids", "2026-08-02T00:00:00Z", 0.1000, 0.1002, 0.001),
        ("o2", "a1", "B", 0.5, 0.375, 4, 0.0, "SIMULATED", "v-good",
         "uniform_among_highest_bids", "2026-08-02T00:00:00Z", 0.0800, 0.0801, 0.001),
        ("dup", "a2", "C", 0.7, 0.5, 4, 0.0, "SIMULATED", "v-dup",
         "uniform_among_highest_bids", "2026-08-02T00:00:00Z", 0.1, 0.1, 0.001),
        ("dup", "a2", "C", 0.7, 0.5, 4, 0.0, "SIMULATED", "v-dup",
         "uniform_among_highest_bids", "2026-08-02T00:00:00Z", 0.1, 0.1, 0.001),
        ("bad", "", "", 0.8, 0.7, 2.5, 0.0, "SIMULATED", "",
         "unknown", "", 0.1, 0.8, 0.001),
    ]
    connection.executemany(
        "INSERT INTO structurally_scored_auction_bids VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    output = _query(connection, "auction_regret.sql")
    by_version = {row["model_version"]: row for row in output}
    assert by_version["v-good"]["status"] == "PASS"
    assert by_version["v-good"]["maximum_utility_regret"] <= 0.001
    assert by_version["v-dup"]["status"] == "REJECT"
    assert by_version[""]["status"] == "REJECT"
    return output


def validate_selection() -> list[dict[str, object]]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE selection_events (
          trial_id TEXT, protocol TEXT, protocol_version TEXT,
          prior_definition TEXT, prior_program REAL, candidates,
          alpha REAL, power REAL, conditional_independence INTEGER,
          complete_trial INTEGER, reported INTEGER, signal_world INTEGER,
          minimum_reports INTEGER, reconciliation_tolerance REAL
        )
        """
    )
    good = []
    for index in range(1_000):
        reported = int(index < 100)
        signal_world = int(index < 3)
        good.append(
            (f"good-{index}", "screen", "v0.10",
             "program contains exactly one genuine model", 0.02, 20,
             0.05, 0.8, 1, 1, reported, signal_world, 30, 0.005)
        )
    connection.executemany(
        "INSERT INTO selection_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        good,
    )
    connection.execute(
        "INSERT INTO selection_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("bad-1", "bad", "", "", 0.5, -1, 2.0, -3.0, 0, 0, 1, 1, 30, 0.01),
    )
    duplicate = (
        "dup-1", "duplicate", "v0.10", "program contains exactly one genuine model",
        0.02, 20, 0.05, 0.8, 1, 1, 1, 0, 30, 1.0e-6,
    )
    connection.executemany(
        "INSERT INTO selection_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [duplicate, duplicate],
    )
    output = _query(connection, "selection_protocol.sql")
    by_protocol = {row["protocol"]: row for row in output}
    assert by_protocol["screen"]["status"] == "PASS"
    assert abs(float(by_protocol["screen"]["empirical_posterior"]) - 0.03) <= 1e-12
    assert by_protocol["bad"]["status"] == "REJECT"
    assert by_protocol["duplicate"]["status"] == "REJECT"
    return output


def validate_coordination() -> list[dict[str, object]]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE coordination_trials (
          strategy TEXT, strategy_version TEXT, enumeration_hash TEXT,
          manifest_enumeration_hash TEXT, trial_id TEXT,
          alice_forbidden INTEGER, bob_forbidden INTEGER,
          success REAL, expected_success REAL,
          regret REAL, regret_tolerance REAL
        )
        """
    )
    good = [
        ("public-randomness-maximin", "v0.10", ENUMERATION_SHA256,
         ENUMERATION_SHA256, f"state-{a}-{b}", a, b, 0.5, 0.5, 0.0, 1e-12)
        for a in range(3) for b in range(3)
    ]
    connection.executemany(
        "INSERT INTO coordination_trials VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        good,
    )
    bad = [
        ("bad-duplicate", "v0.10", "0" * 64, "0" * 64,
         f"bad-{a}-{b}", a, b, 7.0, 0.5, -100.0, 1e-12)
        for a in range(3) for b in range(3)
    ]
    connection.executemany(
        "INSERT INTO coordination_trials VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        bad + bad,
    )
    output = _query(connection, "protocol_regret.sql")
    by_strategy = {row["strategy"]: row for row in output}
    assert by_strategy["public-randomness-maximin"]["status"] == "PASS"
    assert by_strategy["public-randomness-maximin"]["represented_states"] == 9
    assert by_strategy["bad-duplicate"]["status"] == "REJECT"
    return output


def main() -> int:
    outputs = {
        "auction_regret.sql": validate_auction(),
        "selection_protocol.sql": validate_selection(),
        "protocol_regret.sql": validate_coordination(),
    }
    print(json.dumps({"status": "PASS", "named_cases": 9, "queries": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
