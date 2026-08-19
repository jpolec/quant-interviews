from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCES = (
    ROOT / "_book/cpp/ch10/pretrade_risk_gate.cpp",
    ROOT / "_book/cpp/ch10/price_time_order_book.cpp",
    ROOT / "_project/tests/cpp/ch10/test_pretrade_risk_gate.cpp",
    ROOT / "_project/tests/cpp/ch10/test_price_time_order_book.cpp",
)
FLAGS = ("-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic", "-Werror")


def main() -> None:
    results: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="ch10-cpp-") as directory:
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
            results.append(
                {
                    "source": str(source.relative_to(ROOT)),
                    "compile": "PASS",
                    "run": "PASS",
                    "stdout": run_result.stdout.strip(),
                }
            )
    output_dir = Path(
        os.environ.get("QI_CH10_CPP_OUTPUT", "/private/tmp/ch10-cpp-validation")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"standard": "C++20", "flags": list(FLAGS), "targets": results}
    (output_dir / "cpp_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
