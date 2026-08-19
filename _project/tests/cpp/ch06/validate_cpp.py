#!/usr/bin/env python3
"""Strict compile-and-run verification for the CH06 C++20 examples."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CPP_ROOT = REPO_ROOT / "_book" / "cpp" / "ch06"
TEST_ROOT = REPO_ROOT / "_project" / "tests" / "cpp" / "ch06"
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
        raise RuntimeError("a C++20 compiler is required for CH06 verification")
    return executable


def compile_and_run(source: Path, executable: Path) -> str:
    compiled = subprocess.run(
        [compiler(), *CPP_FLAGS, str(source), "-o", str(executable)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if compiled.returncode != 0 or compiled.stderr:
        raise AssertionError(
            f"{source.relative_to(REPO_ROOT)} failed strict compilation:\n"
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
    if completed.returncode != 0 or completed.stderr:
        raise AssertionError(
            f"{source.relative_to(REPO_ROOT)} failed execution:\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed.stdout.strip()


def main() -> int:
    sources = (
        CPP_ROOT / "greeks_kernel.cpp",
        CPP_ROOT / "implied_vol.cpp",
        TEST_ROOT / "greeks_kernel_test.cpp",
        TEST_ROOT / "implied_vol_test.cpp",
    )
    with tempfile.TemporaryDirectory(prefix="ch06-cpp-") as temporary_directory:
        build_root = Path(temporary_directory)
        outputs = {
            source.name: compile_and_run(source, build_root / source.stem)
            for source in sources
        }
    assert "status=PASS" in outputs["greeks_kernel.cpp"]
    assert "status=PASS" in outputs["implied_vol.cpp"]
    assert outputs["greeks_kernel_test.cpp"].startswith("status=PASS")
    assert outputs["implied_vol_test.cpp"].startswith("status=PASS")
    print(
        "PASS standard=C++20 strict_targets=4 "
        "greek_edges=PASS implied_vol_edges=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
