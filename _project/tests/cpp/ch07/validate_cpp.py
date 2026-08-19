#!/usr/bin/env python3
"""Strict compile, run and independent-reference checks for CH07 C++20."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CPP_ROOT = REPO_ROOT / "_book" / "cpp" / "ch07"
TEST_ROOT = REPO_ROOT / "_project" / "tests" / "cpp" / "ch07"
CPP_FLAGS = (
    "-std=c++20",
    "-O2",
    "-Wall",
    "-Wextra",
    "-pedantic",
    "-Werror",
)


def compiler() -> str:
    executable = shutil.which("c++") or shutil.which("clang++") or shutil.which(
        "g++"
    )
    if executable is None:
        raise RuntimeError("a C++20 compiler is required for CH07 verification")
    return executable


def compile_and_run(source: Path, executable: Path) -> str:
    compiled = subprocess.run(
        [compiler(), *CPP_FLAGS, str(source), "-o", str(executable)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if compiled.returncode != 0:
        raise AssertionError(
            f"{source.relative_to(REPO_ROOT)} failed strict compilation:\n"
            f"{compiled.stderr}"
        )
    if compiled.stderr:
        raise AssertionError(
            f"{source.relative_to(REPO_ROOT)} emitted compiler diagnostics:\n"
            f"{compiled.stderr}"
        )
    completed = subprocess.run(
        [str(executable)],
        cwd=executable.parent,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{source.relative_to(REPO_ROOT)} exited {completed.returncode}:\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    if completed.stderr:
        raise AssertionError(
            f"{source.relative_to(REPO_ROOT)} emitted stderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def main() -> int:
    sources = (
        CPP_ROOT / "fixed_income_risk_kernel.cpp",
        TEST_ROOT / "fixed_income_risk_kernel_test.cpp",
    )
    with tempfile.TemporaryDirectory(prefix="ch07-cpp-") as temporary_directory:
        build_root = Path(temporary_directory)
        outputs = {
            source.name: compile_and_run(source, build_root / source.stem)
            for source in sources
        }
    canonical = outputs["fixed_income_risk_kernel.cpp"]
    harness = outputs["fixed_income_risk_kernel_test.cpp"]
    assert canonical.startswith("status=PASS\n")
    assert "price_residual=" in canonical and "dv01_residual=" in canonical
    assert harness.startswith("status=PASS reference=PASS")
    assert "finite_domain=PASS" in harness
    print(
        "PASS standard=C++20 strict_targets=2 independent_reference=PASS "
        "finite_domain=PASS schedule_domain=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
