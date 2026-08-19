#!/usr/bin/env python3
"""Compile and execute every CH11 C++20 target plus adversarial probes."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[4]
CPP = ROOT / "_book/cpp/ch11"
PROBES = Path(__file__).resolve().parent
FLAGS = ("-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic", "-Werror", "-pthread")
CANONICAL = tuple(sorted(CPP.glob("*.cpp")))
ADVERSARIAL = tuple(sorted(PROBES.glob("*_probe.cpp")))


def _run(command: list[str], *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _compile_and_run(
    compiler: str,
    source: Path,
    binary: Path,
    extra_flags: tuple[str, ...] = (),
) -> dict[str, object]:
    completed = _run([compiler, *FLAGS, *extra_flags, str(source), "-o", str(binary)])
    if completed.stderr:
        raise AssertionError(f"compiler emitted diagnostics for {source}: {completed.stderr}")
    executed = _run([str(binary)])
    return {
        "binary_sha256": sha256(binary.read_bytes()).hexdigest(),
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "stdout_sha256": sha256(executed.stdout.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/ch11-cpp-validation.json"),
    )
    parser.add_argument("--asan-ubsan", action="store_true")
    arguments = parser.parse_args()
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise SystemExit("no C++20 compiler found")
    version = _run([compiler, "--version"]).stdout.splitlines()[0]
    records: list[dict[str, object]] = []
    sanitizer_records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="ch11-cpp-", dir="/private/tmp") as temporary:
        build = Path(temporary)
        for index, source in enumerate((*CANONICAL, *ADVERSARIAL), start=1):
            records.append(_compile_and_run(compiler, source, build / f"target-{index}"))
        if arguments.asan_ubsan:
            sanitizer_flags = (
                "-O1",
                "-g",
                "-fno-omit-frame-pointer",
                "-fsanitize=address,undefined",
            )
            for index, source in enumerate(ADVERSARIAL, start=1):
                sanitizer_records.append(
                    _compile_and_run(
                        compiler,
                        source,
                        build / f"sanitized-{index}",
                        sanitizer_flags,
                    )
                )
    payload = {
        "status": "PASS",
        "compiler": version,
        "flags": list(FLAGS),
        "canonical_targets": len(CANONICAL),
        "adversarial_probes": len(ADVERSARIAL),
        "asan_ubsan_probes": len(sanitizer_records),
        "records": records,
        "sanitizer_records": sanitizer_records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"PASS canonical={len(CANONICAL)} probes={len(ADVERSARIAL)} "
        f"asan_ubsan={len(sanitizer_records)} compiler={version}"
    )


if __name__ == "__main__":
    main()
