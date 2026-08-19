#!/usr/bin/env python3
"""Strictly compile and execute every canonical CH14 C++20 program."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[4]
SOURCES = ROOT / "_book/cpp/ch14"
FLAGS = ("-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic", "-Werror")


def main() -> int:
    compiler = shutil.which("c++") or shutil.which("clang++")
    if compiler is None:
        raise AssertionError("C++ compiler is required; skipping is not release evidence")
    expected = {
        "online_stats": "4 2.500000 1.666667",
        "risk_gate": "PASS",
        "top_k": "B:4\nC:3",
    }
    with tempfile.TemporaryDirectory(prefix="ch14-cpp-") as directory:
        build = Path(directory)
        binaries: dict[str, Path] = {}
        for source in sorted(SOURCES.glob("*.cpp")):
            binary = build / source.stem
            subprocess.run(
                [compiler, *FLAGS, str(source), "-o", str(binary)],
                check=True,
                capture_output=True,
                text=True,
            )
            binaries[source.stem] = binary
            output = subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True
            ).stdout.strip()
            if source.stem in expected and output != expected[source.stem]:
                raise AssertionError(f"unexpected {source.stem} output: {output!r}")
        canonical = '{"schema_version":"1.0","value":17}'
        digest = subprocess.run(
            [str(binaries["replay_hash"]), canonical],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if digest != sha256(canonical.encode()).hexdigest():
            raise AssertionError("Python/C++ SHA-256 parity failed")
    print(
        f"PASS compiler={Path(compiler).name} standard=c++20 flags={' '.join(FLAGS[1:])} "
        f"targets={len(binaries)} adversarial_self_tests=2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
