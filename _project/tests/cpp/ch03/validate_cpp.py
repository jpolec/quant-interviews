#!/usr/bin/env python3
"""Strict compile-and-run verification for the CH03 C++20 examples."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CPP_ROOT = REPO_ROOT / "_book" / "cpp" / "ch03"
TEST_ROOT = REPO_ROOT / "_project" / "tests" / "cpp" / "ch03"
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
        raise RuntimeError("a C++20 compiler is required for CH03 verification")
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
        CPP_ROOT / "root_solver.cpp",
        CPP_ROOT / "tridiagonal.cpp",
        CPP_ROOT / "adjoint_kernel.cpp",
        TEST_ROOT / "root_solver_test.cpp",
        TEST_ROOT / "tridiagonal_test.cpp",
        TEST_ROOT / "adjoint_kernel_test.cpp",
    )
    with tempfile.TemporaryDirectory(prefix="ch03-cpp-") as temporary_directory:
        build_root = Path(temporary_directory)
        outputs = {
            source.name: compile_and_run(source, build_root / source.stem)
            for source in sources
        }

    assert outputs["root_solver.cpp"].startswith("root=")
    assert "residual=" in outputs["root_solver.cpp"]
    assert outputs["tridiagonal.cpp"].startswith("solution=1,1,1 ")
    assert "backward_error=" in outputs["tridiagonal.cpp"]
    assert outputs["adjoint_kernel.cpp"].startswith("adjoint=0.6,-0.2 ")
    assert "backward_error=" in outputs["adjoint_kernel.cpp"]
    assert outputs["root_solver_test.cpp"].startswith("status=PASS ")
    assert "invalid_tolerance=REJECT" in outputs["root_solver_test.cpp"]
    assert outputs["tridiagonal_test.cpp"].startswith("status=PASS ")
    assert "near_singular=REJECT" in outputs["tridiagonal_test.cpp"]
    assert "all_pivots=REJECT" in outputs["tridiagonal_test.cpp"]
    assert outputs["adjoint_kernel_test.cpp"].startswith("status=PASS ")
    assert "singular=REJECT" in outputs["adjoint_kernel_test.cpp"]

    print(
        "PASS standard=C++20 strict_targets=6 "
        "bisection_edges=PASS tridiagonal_edges=PASS adjoint_edges=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
