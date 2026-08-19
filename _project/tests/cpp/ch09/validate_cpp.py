from __future__ import annotations

import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
CPP_DIR = REPO_ROOT / "_book" / "cpp" / "ch09"
FLAGS = ("-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic")


def _compile(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)


def _run(executable: Path) -> dict[str, str]:
    result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_ch09_cpp() -> None:
    with tempfile.TemporaryDirectory(prefix="qj-ch09-cpp-") as temporary:
        root = Path(temporary)
        risk_executable = root / "risk"
        gate_executable = root / "gate"
        _compile(
            [
                "c++",
                *FLAGS,
                str(CPP_DIR / "portfolio_risk_kernel.cpp"),
                "-o",
                str(risk_executable),
            ]
        )
        _compile(
            [
                "c++",
                *FLAGS,
                str(CPP_DIR / "pretrade_constraint_gate.cpp"),
                "-o",
                str(gate_executable),
            ]
        )
        assert _run(risk_executable)["status"] == "PASS"
        assert _run(gate_executable)["status"] == "PASS"

        risk_probe = root / "risk_probe.cpp"
        risk_probe.write_text(
            r'''#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
double variance(const std::vector<double>&, const std::vector<double>&, std::size_t);
int main() {
    std::cout << std::setprecision(17);
    std::cout << "base=" << variance({0.6, 0.4}, {0.04, 0.006, 0.006, 0.09}, 2) << "\n";
    std::cout << "small=" << variance({2.0, -1.0}, {1e-12, 0.0, 0.0, 2e-12}, 2) << "\n";
    std::cout << "large=" << variance({0.2, 0.8}, {4e8, 1e8, 1e8, 9e8}, 2) << "\n";
    bool rejected = false;
    try { static_cast<void>(variance({0.0, 1.0}, {-1.0, 0.0, 0.0, 1.0}, 2)); }
    catch (const std::invalid_argument&) { rejected = true; }
    std::cout << "indefinite_rejected=" << (rejected ? "true" : "false") << "\n";
}
''',
            encoding="utf-8",
        )
        risk_probe_executable = root / "risk_probe"
        _compile(
            [
                "c++",
                *FLAGS,
                "-DCH09_NO_MAIN",
                str(CPP_DIR / "portfolio_risk_kernel.cpp"),
                str(risk_probe),
                "-o",
                str(risk_probe_executable),
            ]
        )
        risk = _run(risk_probe_executable)
        cases = {
            "base": (
                np.array([0.6, 0.4]),
                np.array([[0.04, 0.006], [0.006, 0.09]]),
            ),
            "small": (
                np.array([2.0, -1.0]),
                np.diag([1.0e-12, 2.0e-12]),
            ),
            "large": (
                np.array([0.2, 0.8]),
                np.array([[4.0e8, 1.0e8], [1.0e8, 9.0e8]]),
            ),
        }
        for name, (weights, covariance) in cases.items():
            expected = float(weights @ covariance @ weights)
            actual = float(risk[name])
            assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-18)
        assert risk["indefinite_rejected"] == "true"

        gate_probe = root / "gate_probe.cpp"
        gate_probe.write_text(
            r'''#include <iostream>
#include <vector>
bool passes(const std::vector<double>&, double, double);
int main() {
    std::cout << "boundary=" << (passes({0.4, 0.3, 0.3}, 1.0, 0.4) ? "true" : "false") << "\n";
    std::cout << "gross=" << (passes({0.6, 0.5}, 1.0, 0.7) ? "true" : "false") << "\n";
}
''',
            encoding="utf-8",
        )
        gate_probe_executable = root / "gate_probe"
        _compile(
            [
                "c++",
                *FLAGS,
                "-DCH09_NO_MAIN",
                str(CPP_DIR / "pretrade_constraint_gate.cpp"),
                str(gate_probe),
                "-o",
                str(gate_probe_executable),
            ]
        )
        gate = _run(gate_probe_executable)
        assert gate == {"boundary": "true", "gross": "false"}


def main() -> None:
    test_ch09_cpp()
    print("PASS ch09 strict C++20 kernels and Python cross-check")


if __name__ == "__main__":
    main()
