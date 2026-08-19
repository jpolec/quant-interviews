from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from jsonschema import Draft202012Validator, FormatChecker


CONTRACT_VERSION = "0.10.0"


@dataclass(frozen=True)
class ContractValidation:
    tool_name: str
    contract_version: str
    input_valid: bool
    output_valid: bool


def contract_root() -> Path:
    return Path(__file__).resolve().parents[3] / "json/ch12"


def load_contract(tool_name: str) -> Mapping[str, object]:
    if not tool_name or any(character not in "abcdefghijklmnopqrstuvwxyz_" for character in tool_name):
        raise ValueError("invalid tool name")
    path = contract_root() / f"{tool_name}.json"
    if not path.is_file():
        raise KeyError(f"unknown CH12 tool: {tool_name}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("name") != tool_name or contract.get("contractVersion") != CONTRACT_VERSION:
        raise ValueError("tool-contract identity/version mismatch")
    return contract


def _validate(instance: Mapping[str, object], schema: object, label: str) -> None:
    if not isinstance(schema, dict):
        raise TypeError("contract schema must be an object")
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(dict(instance)), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.absolute_path) or "$"
        raise ValueError(f"{label} validation failed at {path}: {first.message}")


def validate_contract_call(
    tool_name: str,
    tool_input: Mapping[str, object],
    tool_output: Mapping[str, object],
) -> ContractValidation:
    contract = load_contract(tool_name)
    _validate(tool_input, contract["inputSchema"], "input")
    _validate(tool_output, contract["outputSchema"], "output")
    return ContractValidation(tool_name, CONTRACT_VERSION, True, True)
