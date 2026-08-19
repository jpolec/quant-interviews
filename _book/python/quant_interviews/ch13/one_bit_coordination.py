from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog


VENUES = (0, 1, 2)
STATES = tuple(product(VENUES, VENUES))


@dataclass(frozen=True)
class Protocol:
    messages: tuple[int, int, int]
    alice_actions: tuple[int, int, int]
    bob_actions: tuple[int, int, int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.messages) != 3 or any(message not in (0, 1) for message in self.messages):
            raise ValueError("messages must contain one binary symbol per Alice state")
        if len(self.alice_actions) != 3 or len(self.bob_actions) != 6:
            raise ValueError("protocol action mappings have invalid lengths")
        for forbidden, action in enumerate(self.alice_actions):
            if action not in VENUES or action == forbidden:
                raise ValueError("Alice action must be locally legal")
        for forbidden in VENUES:
            for message in (0, 1):
                action = self.bob_actions[2 * forbidden + message]
                if action not in VENUES or action == forbidden:
                    raise ValueError("Bob action must be locally legal")

    def success(self, alice_forbidden: int, bob_forbidden: int) -> int:
        if (
            isinstance(alice_forbidden, bool)
            or isinstance(bob_forbidden, bool)
            or alice_forbidden not in VENUES
            or bob_forbidden not in VENUES
        ):
            raise ValueError("forbidden venues must lie in {0,1,2}")
        message = self.messages[alice_forbidden]
        alice_action = self.alice_actions[alice_forbidden]
        bob_action = self.bob_actions[2 * bob_forbidden + message]
        return int(
            alice_action == bob_action
            and alice_action != alice_forbidden
            and alice_action != bob_forbidden
        )


@dataclass(frozen=True)
class CoordinationCertificate:
    deterministic_protocol_count: int
    best_deterministic_average: float
    best_deterministic_worst_case: float
    randomized_maximin_value: float
    enumeration_sha256: str
    support_indices: tuple[int, ...]
    support_protocols: tuple[Protocol, ...]
    support_weights: tuple[float, ...]
    statewise_success: tuple[float, ...]
    primal_residual: float
    analytic_upper_bound: float
    analytic_upper_bound_gap: float
    shared_randomness_assumption: str
    adversary_assumption: str
    status: str

    @property
    def upper_bound(self) -> float:
        """Backward-compatible alias with an explicit analytic meaning."""
        return self.analytic_upper_bound

    @property
    def duality_gap(self) -> float:
        """Deprecated alias; this is not a solver-produced dual certificate."""
        return self.analytic_upper_bound_gap


def enumerate_protocols() -> list[Protocol]:
    protocols: list[Protocol] = []
    alice_choices = [
        tuple(venue for venue in VENUES if venue != forbidden)
        for forbidden in VENUES
    ]
    bob_choice_sets = []
    for forbidden in VENUES:
        allowed = tuple(venue for venue in VENUES if venue != forbidden)
        bob_choice_sets.extend([allowed, allowed])
    for messages in product((0, 1), repeat=3):
        for alice_actions in product(*alice_choices):
            for bob_actions in product(*bob_choice_sets):
                protocols.append(
                    Protocol(
                        tuple(messages),
                        tuple(alice_actions),
                        tuple(bob_actions),
                    )
                )
    return protocols


def _payoff_matrix(protocols: list[Protocol]) -> NDArray[np.float64]:
    return np.array(
        [
            [protocol.success(*state) for protocol in protocols]
            for state in STATES
        ],
        dtype=float,
    )


def protocol_enumeration_sha256(protocols: list[Protocol] | None = None) -> str:
    """Hash venue/state order, every mapping and the authoritative payoff matrix."""
    canonical_protocols = enumerate_protocols() if protocols is None else protocols
    payoff = _payoff_matrix(canonical_protocols).astype(int)
    payload = {
        "venues": list(VENUES),
        "states": [list(state) for state in STATES],
        "protocols": [
            {
                "messages": list(protocol.messages),
                "alice_actions": list(protocol.alice_actions),
                "bob_actions": list(protocol.bob_actions),
            }
            for protocol in canonical_protocols
        ],
        "state_by_protocol_payoff": payoff.tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def best_average_protocol() -> tuple[Protocol, float]:
    protocols = enumerate_protocols()
    payoff = _payoff_matrix(protocols)
    averages = np.mean(payoff, axis=0)
    index = int(np.argmax(averages))
    return protocols[index], float(averages[index])


def coordination_certificate() -> CoordinationCertificate:
    """Solve the public-randomness maximin LP against an oblivious adversary."""
    protocols = enumerate_protocols()
    payoff = _payoff_matrix(protocols)
    deterministic_averages = np.mean(payoff, axis=0)
    deterministic_worst = np.min(payoff, axis=0)

    # Variables are protocol probabilities and guaranteed success t.
    objective = np.zeros(len(protocols) + 1)
    objective[-1] = -1.0
    inequality_matrix = np.hstack(
        [-payoff, np.ones((len(STATES), 1))]
    )
    equality_matrix = np.zeros((1, len(protocols) + 1))
    equality_matrix[0, :-1] = 1.0
    result = linprog(
        objective,
        A_ub=inequality_matrix,
        b_ub=np.zeros(len(STATES)),
        A_eq=equality_matrix,
        b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * len(protocols) + [(0.0, 1.0)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = result.x[:-1]
    value = float(result.x[-1])
    statewise = payoff @ weights
    support = np.flatnonzero(weights > 1e-10)
    primal_residual = max(
        abs(float(np.sum(weights)) - 1.0),
        max(0.0, value - float(np.min(statewise))),
        max(0.0, -float(np.min(weights))),
    )

    # States (1,0) and (1,2) provide a direct 1/2 upper bound:
    # deterministic Alice sees the same private state but must choose
    # different venues to win.
    upper_bound = 0.5
    gap = upper_bound - value
    enumeration_hash = protocol_enumeration_sha256(protocols)
    passed = (
        len(protocols) == 4096
        and abs(float(np.max(deterministic_averages)) - 2.0 / 3.0) <= 1e-12
        and float(np.max(deterministic_worst)) == 0.0
        and abs(value - 0.5) <= 1e-10
        and primal_residual <= 1e-9
        and abs(gap) <= 1e-9
    )
    return CoordinationCertificate(
        deterministic_protocol_count=len(protocols),
        best_deterministic_average=float(np.max(deterministic_averages)),
        best_deterministic_worst_case=float(np.max(deterministic_worst)),
        randomized_maximin_value=value,
        enumeration_sha256=enumeration_hash,
        support_indices=tuple(int(index) for index in support),
        support_protocols=tuple(protocols[int(index)] for index in support),
        support_weights=tuple(float(weights[index]) for index in support),
        statewise_success=tuple(float(entry) for entry in statewise),
        primal_residual=primal_residual,
        analytic_upper_bound=upper_bound,
        analytic_upper_bound_gap=gap,
        shared_randomness_assumption=(
            "after adversary commits, a shared seed is sampled and observed by both players"
        ),
        adversary_assumption=(
            "adversary commits the state before the shared-seed realisation"
        ),
        status="PASS" if passed else "REJECT",
    )


def randomized_maximin() -> tuple[float, NDArray[np.float64]]:
    certificate = coordination_certificate()
    weights = np.zeros(certificate.deterministic_protocol_count)
    for index, weight in zip(
        certificate.support_indices,
        certificate.support_weights,
    ):
        weights[index] = weight
    return certificate.randomized_maximin_value, weights
