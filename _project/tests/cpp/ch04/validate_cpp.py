"""Compile and execute both Chapter 4 kernels under the publication flags."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[4]
FLAGS = ("-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic")
TARGETS = (
    (ROOT / "_book/cpp/ch04/gbm_exact_kernel.cpp", "status=PASS"),
    (ROOT / "_book/cpp/ch04/sde_scheme_kernel.cpp", "status=PASS"),
)


def main() -> None:
    compiler = os.environ.get("CXX", "c++")
    with tempfile.TemporaryDirectory(prefix="qj-ch04-cpp-") as directory:
        output_root = Path(directory)
        for source, marker in TARGETS:
            binary = output_root / source.stem
            compile_result = subprocess.run(
                (compiler, *FLAGS, str(source), "-o", str(binary)),
                check=False,
                capture_output=True,
                text=True,
            )
            if compile_result.returncode != 0 or compile_result.stderr:
                raise AssertionError(
                    f"strict compile failed for {source.name}: "
                    f"{compile_result.stdout}{compile_result.stderr}"
                )
            run_result = subprocess.run(
                (str(binary),), check=False, capture_output=True, text=True
            )
            if run_result.returncode != 0 or marker not in run_result.stdout:
                raise AssertionError(
                    f"runtime validation failed for {source.name}: "
                    f"{run_result.stdout}{run_result.stderr}"
                )
    print(
        "PASS standard=C++20 strict_targets=2 "
        "gbm_exact=PASS sde_schemes=PASS"
    )


if __name__ == "__main__":
    main()
