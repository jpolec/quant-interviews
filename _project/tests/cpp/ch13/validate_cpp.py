from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "_book/python"))

from quant_interviews.ch13.gambler_ruin import hit_upper_probability


SOURCES = (
    ROOT / "_book/cpp/ch13/gambler_ruin.cpp",
    ROOT / "_book/cpp/ch13/matrix_reduction.cpp",
    ROOT / "_book/cpp/ch13/unbiased_bits.cpp",
)
FLAGS = ("-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic", "-Werror")


def main() -> None:
    results: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="ch13-cpp-") as directory:
        build = Path(directory)
        for index, source in enumerate(SOURCES):
            executable = build / f"target-{index}"
            compile_result = subprocess.run(
                ["c++", *FLAGS, str(source), "-o", str(executable)],
                check=False,
                capture_output=True,
                text=True,
            )
            assert compile_result.returncode == 0, compile_result.stderr
            run_result = subprocess.run(
                [str(executable)], check=False, capture_output=True, text=True
            )
            assert run_result.returncode == 0, run_result.stdout + run_result.stderr
            assert "status=PASS" in run_result.stdout, run_result.stdout
            record = {
                "source": str(source.relative_to(ROOT)),
                "compile": "PASS",
                "run": "PASS",
                "stdout": run_result.stdout.strip(),
            }
            if source.name == "gambler_ruin.cpp":
                observed = dict(
                    line.split("=", maxsplit=1)
                    for line in run_result.stdout.splitlines()
                    if "=" in line and not line.startswith("status=")
                )
                cases = {
                    "biased": (4, 10, 0.55),
                    "fair": (4, 10, 0.5),
                    "near_fair": (400, 1000, 0.500000000001),
                }
                residuals = [
                    abs(float(observed[name]) - hit_upper_probability(*arguments))
                    for name, arguments in cases.items()
                ]
                maximum_residual = max(residuals)
                assert maximum_residual <= 1e-12, maximum_residual
                record["python_cpp_max_abs_residual"] = f"{maximum_residual:.17g}"
            results.append(record)
    output_dir = Path(
        os.environ.get("QI_CH13_CPP_OUTPUT", "/private/tmp/ch13-cpp-validation")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"standard": "C++20", "flags": list(FLAGS), "targets": results}
    (output_dir / "cpp_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
