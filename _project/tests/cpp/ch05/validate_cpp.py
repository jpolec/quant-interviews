from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def _compile_and_run(source: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="ch05-cpp-") as temporary:
        binary = Path(temporary) / source.stem
        compile_run = subprocess.run(
            [
                "c++", "-std=c++20", "-O2", "-Wall", "-Wextra",
                "-pedantic", str(source), "-o", str(binary),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert compile_run.returncode == 0, compile_run.stderr
        assert compile_run.stderr == ""
        execution = subprocess.run(
            [str(binary)], check=False, capture_output=True, text=True
        )
        assert execution.returncode == 0, execution.stdout + execution.stderr
        assert "status=PASS" in execution.stdout
        return execution.stdout.strip()


def test_ch05_cpp_kernels() -> None:
    american = _compile_and_run(ROOT / "_book/cpp/ch05/american_lattice.cpp")
    black_scholes = _compile_and_run(
        ROOT / "_book/cpp/ch05/black_scholes_kernel.cpp"
    )
    assert "reference_residual=" in american
    assert "error=" in black_scholes


def main() -> int:
    test_ch05_cpp_kernels()
    print("PASS ch05 strict C++20 kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
