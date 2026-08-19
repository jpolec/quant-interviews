"""Strict compile-and-run verification for the CH01 C++20 examples."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
CPP_ROOT = REPO_ROOT / "_book" / "cpp" / "ch01"
CPP_FLAGS = ("-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic")


def compiler() -> str:
    executable = (
        shutil.which("c++")
        or shutil.which("clang++")
        or shutil.which("g++")
    )
    if executable is None:
        pytest.fail("a C++20 compiler is required for the CH01 verification")
    return executable


def compile_and_run(source: Path, executable: Path) -> subprocess.CompletedProcess[str]:
    compiled = subprocess.run(
        [compiler(), *CPP_FLAGS, str(source), "-o", str(executable)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    assert compiled.stderr == "", compiled.stderr
    result = subprocess.run(
        [str(executable)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stderr == "", result.stderr
    return result


@pytest.mark.parametrize(
    ("source_name", "expected_fragments"),
    [
        (
            "online_moments.cpp",
            (
                "count=4",
                "mean=2.5",
                "sample_variance=1.6666666666666667",
                "nonfinite_rejected=PASS",
                "insufficient_count_rejected=PASS",
                "status=PASS",
            ),
        ),
        (
            "bh_fdr.cpp",
            (
                "discoveries=2",
                "invalid_alpha_rejected=PASS",
                "invalid_p_value_rejected=PASS",
                "status=PASS",
            ),
        ),
    ],
)
def test_canonical_targets_compile_and_run(
    source_name: str,
    expected_fragments: tuple[str, ...],
) -> None:
    with tempfile.TemporaryDirectory(prefix="ch01-cpp-") as directory:
        result = compile_and_run(
            CPP_ROOT / source_name,
            Path(directory) / source_name.removesuffix(".cpp"),
        )
    assert all(fragment in result.stdout for fragment in expected_fragments)


def test_online_moments_semantics_and_failure_paths() -> None:
    source = CPP_ROOT / "online_moments.cpp"
    harness = f'''\
#define QUANT_INTERVIEWS_CH01_NO_MAIN
#include "{source}"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {{
    OnlineMoments moments;
    for (const double value : {{1.0, 2.0, 3.0, 4.0}}) {{
        moments.update(value);
    }}
    if (moments.count != 4 || std::abs(moments.mean - 2.5) > 1e-15
        || std::abs(moments.sample_variance() - 5.0 / 3.0) > 1e-12) {{
        return 1;
    }}

    const std::size_t frozen_count = moments.count;
    const double frozen_mean = moments.mean;
    bool nan_rejected = false;
    try {{
        moments.update(std::numeric_limits<double>::quiet_NaN());
    }} catch (const std::invalid_argument&) {{
        nan_rejected = true;
    }}
    if (!nan_rejected || moments.count != frozen_count
        || moments.mean != frozen_mean) {{
        return 2;
    }}

    bool infinity_rejected = false;
    try {{
        moments.update(std::numeric_limits<double>::infinity());
    }} catch (const std::invalid_argument&) {{
        infinity_rejected = true;
    }}
    if (!infinity_rejected) {{
        return 3;
    }}

    bool insufficient_count_rejected = false;
    try {{
        OnlineMoments one;
        one.update(1.0);
        static_cast<void>(one.sample_variance());
    }} catch (const std::domain_error&) {{
        insufficient_count_rejected = true;
    }}
    if (!insufficient_count_rejected) {{
        return 4;
    }}

    bool overflow_rejected = false;
    try {{
        OnlineMoments extreme;
        extreme.update(std::numeric_limits<double>::max());
        extreme.update(-std::numeric_limits<double>::max());
    }} catch (const std::overflow_error&) {{
        overflow_rejected = true;
    }}
    if (!overflow_rejected) {{
        return 5;
    }}

    std::cout << "online_moments_semantics=PASS\\n";
    return 0;
}}
'''
    with tempfile.TemporaryDirectory(prefix="ch01-online-harness-") as directory:
        directory_path = Path(directory)
        harness_path = directory_path / "online_moments_harness.cpp"
        harness_path.write_text(harness, encoding="utf-8")
        result = compile_and_run(harness_path, directory_path / "harness")
    assert result.stdout == "online_moments_semantics=PASS\n"


def test_benjamini_hochberg_semantics_and_failure_paths() -> None:
    source = CPP_ROOT / "bh_fdr.cpp"
    harness = f'''\
#define QUANT_INTERVIEWS_CH01_NO_MAIN
#include "{source}"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

bool rejects_alpha(const double alpha) {{
    try {{
        static_cast<void>(benjamini_hochberg_discoveries({{0.01}}, alpha));
    }} catch (const std::invalid_argument&) {{
        return true;
    }}
    return false;
}}

bool rejects_p_values(const std::vector<double>& p_values) {{
    try {{
        static_cast<void>(benjamini_hochberg_discoveries(p_values, 0.05));
    }} catch (const std::invalid_argument&) {{
        return true;
    }}
    return false;
}}

int main() {{
    if (benjamini_hochberg_discoveries(
            {{0.001, 0.01, 0.20, 0.50}}, 0.05
        ) != 2) {{
        return 1;
    }}
    if (benjamini_hochberg_discoveries({{0.025, 0.05}}, 0.05) != 2) {{
        return 2;
    }}
    if (benjamini_hochberg_discoveries({{0.5, 0.8}}, 0.05) != 0) {{
        return 3;
    }}

    const std::vector<double> invalid_alpha_values{{
        0.0,
        -0.1,
        1.1,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity()
    }};
    for (const double alpha : invalid_alpha_values) {{
        if (!rejects_alpha(alpha)) {{
            return 4;
        }}
    }}

    const std::vector<std::vector<double>> invalid_p_value_sets{{
        {{}},
        {{-0.1}},
        {{1.1}},
        {{std::numeric_limits<double>::quiet_NaN()}},
        {{std::numeric_limits<double>::infinity()}}
    }};
    for (const auto& p_values : invalid_p_value_sets) {{
        if (!rejects_p_values(p_values)) {{
            return 5;
        }}
    }}

    std::cout << "bh_fdr_semantics=PASS\\n";
    return 0;
}}
'''
    with tempfile.TemporaryDirectory(prefix="ch01-bh-harness-") as directory:
        directory_path = Path(directory)
        harness_path = directory_path / "bh_fdr_harness.cpp"
        harness_path.write_text(harness, encoding="utf-8")
        result = compile_and_run(harness_path, directory_path / "harness")
    assert result.stdout == "bh_fdr_semantics=PASS\n"
