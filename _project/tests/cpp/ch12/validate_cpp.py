from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE_DIR = ROOT / "_book/cpp/ch12"
SOURCES = tuple(sorted(SOURCE_DIR.glob("*.cpp")))
FLAGS = ("-std=c++20", "-O2", "-Wall", "-Wextra", "-pedantic", "-Werror")

PROBES = {
    "adjustment_kernel.cpp": r'''
int main() {
    bool invalid = false;
    bool overflow = false;
    try { (void)apply_split({"P", -1.0}, 2.0); }
    catch (const std::invalid_argument&) { invalid = true; }
    try { (void)apply_split({"P", 1e308}, 1e308); }
    catch (const std::overflow_error&) { overflow = true; }
    return invalid && overflow ? 0 : 1;
}
''',
    "event_queue.cpp": r'''
int main() {
    try {
        (void)ordered_kinds({{1, 1, 1, 1, "same", "market"},
                             {2, 1, 2, 2, "same", "order"}});
    } catch (const std::invalid_argument&) { return 0; }
    return 1;
}
''',
    "fill_state_machine.cpp": r'''
int main() {
    Order order{Side::buy, State::acknowledged, 10, 0, 0, 10.0, 0.0};
    try { apply_fill(order, 1, 0.0); }
    catch (const std::invalid_argument&) { return 0; }
    return 1;
}
''',
    "journal_replay.cpp": r'''
int main() {
    const Event e{"e1", "p", 1, "add", 2};
    if (replay("p", {e, e}) != 2) { return 1; }
    try { (void)replay("p", {e, {"e1", "p", 1, "add", 3}}); }
    catch (const std::runtime_error&) { return 0; }
    return 1;
}
''',
    "nav_kernel.cpp": r'''
int main() {
    try { (void)nav({{"EUR", 1.0}}, {}, {{"USD", 1.0}}); }
    catch (const std::invalid_argument&) { return 0; }
    return 1;
}
''',
    "pretrade_gate.cpp": r'''
int main() {
    const Limits limits{1.0, 0.25, 0.20, 0.05, 3.0};
    const Risk before{1.1, 0.10, 0.10, 0.03, 1.0};
    const Risk worsened{1.2, 0.10, 0.10, 0.03, 1.0};
    const Risk improved{1.0, 0.10, 0.10, 0.03, 1.0};
    return decide(before, worsened, limits) == Decision::block &&
           decide(before, improved, limits) == Decision::pass ? 0 : 1;
}
''',
}


def compile_and_run(source: Path, executable: Path) -> str:
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
    return run_result.stdout.strip()


def main() -> None:
    results: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="ch12-cpp-") as directory:
        build = Path(directory)
        for index, source in enumerate(SOURCES):
            stdout = compile_and_run(source, build / f"main-{index}")
            harness = build / f"probe-{index}.cpp"
            harness.write_text(
                f'#define QJ_CH12_NO_MAIN\n#include "{source}"\n{PROBES[source.name]}',
                encoding="utf-8",
            )
            compile_and_run(harness, build / f"probe-{index}")
            results.append(
                {
                    "source": str(source.relative_to(ROOT)),
                    "compile": "PASS",
                    "main": "PASS",
                    "adversarial_probe": "PASS",
                    "stdout": stdout,
                }
            )
    output_dir = Path(
        os.environ.get("QI_CH12_CPP_OUTPUT", "/private/tmp/ch12-cpp-validation")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"standard": "C++20", "flags": list(FLAGS), "targets": results}
    (output_dir / "cpp_validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
