#!/usr/bin/env python3
"""Strict compile-and-run verification for the CH02 C++20 examples."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CPP_ROOT = REPO_ROOT / "_book" / "cpp" / "ch02"
TEST_ROOT = REPO_ROOT / "_project" / "tests" / "cpp" / "ch02"
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
        raise RuntimeError("a C++20 compiler is required for CH02 verification")
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
        CPP_ROOT / "cholesky_kernel.cpp",
        CPP_ROOT / "linear_solve.cpp",
        TEST_ROOT / "cholesky_kernel_test.cpp",
        TEST_ROOT / "linear_solve_test.cpp",
    )
    with tempfile.TemporaryDirectory(prefix="ch02-cpp-") as temporary_directory:
        build_root = Path(temporary_directory)
        outputs = {
            source.name: compile_and_run(source, build_root / source.stem)
            for source in sources
        }

    assert outputs["cholesky_kernel.cpp"].startswith("method=cholesky ")
    assert "reconstruction_error=" in outputs["cholesky_kernel.cpp"]
    assert outputs["linear_solve.cpp"].startswith("method=cholesky_solve ")
    assert "normwise_backward_error=" in outputs["linear_solve.cpp"]
    assert outputs["cholesky_kernel_test.cpp"].startswith("status=PASS ")
    assert "non_positive_pivot=REJECT" in outputs["cholesky_kernel_test.cpp"]
    assert outputs["linear_solve_test.cpp"].startswith("status=PASS ")
    assert "zero_rhs=PASS" in outputs["linear_solve_test.cpp"]
    assert "bad_solution=REJECT" in outputs["linear_solve_test.cpp"]

    print(
        "PASS standard=C++20 strict_targets=4 "
        "zero_rhs=PASS near_singular=PASS non_positive_pivot=REJECT "
        "bad_solution=REJECT"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
