from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isfinite, log2


@dataclass(frozen=True)
class ExtractionBatch:
    bits: tuple[int, ...]
    input_tosses: int
    accepted_pairs: int
    rejected_pairs: int
    pending_tosses: int
    acceptance_rate: float
    status: str


@dataclass
class VonNeumannExtractor:
    """Streaming extractor using the explicit convention H=0 and T=1."""

    pending_bit: int | None = None
    total_tosses: int = 0
    accepted_pairs: int = 0
    rejected_pairs: int = 0
    _outputs: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        counters = (self.total_tosses, self.accepted_pairs, self.rejected_pairs)
        if self.pending_bit not in (None, 0, 1):
            raise ValueError("pending_bit must be None, 0 or 1")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters):
            raise ValueError("extractor counters must be non-negative integers")
        if any(bit not in (0, 1) for bit in self._outputs):
            raise ValueError("stored outputs must be binary")
        completed_pairs = self.accepted_pairs + self.rejected_pairs
        if (
            len(self._outputs) != self.accepted_pairs
            or self.total_tosses != 2 * completed_pairs + int(self.pending_bit is not None)
        ):
            raise ValueError("extractor counters and buffered state are inconsistent")

    def update(self, bits: Iterable[int]) -> ExtractionBatch:
        # Validate the complete chunk before committing any state mutation.
        chunk = list(bits)
        if any(isinstance(bit, bool) or not isinstance(bit, int) or bit not in (0, 1) for bit in chunk):
            raise ValueError("bits must be binary integers under H=0, T=1")
        before_tosses = self.total_tosses
        before_accepted = self.accepted_pairs
        before_rejected = self.rejected_pairs
        output_start = len(self._outputs)
        for bit in chunk:
            self.total_tosses += 1
            if self.pending_bit is None:
                self.pending_bit = int(bit)
                continue
            first = self.pending_bit
            second = int(bit)
            self.pending_bit = None
            if first == second:
                self.rejected_pairs += 1
            else:
                # HT=(0,1) -> 0; TH=(1,0) -> 1.
                self._outputs.append(first)
                self.accepted_pairs += 1
        batch_tosses = self.total_tosses - before_tosses
        batch_accepted = self.accepted_pairs - before_accepted
        completed_pairs = batch_accepted + self.rejected_pairs - before_rejected
        acceptance_rate_value = (
            batch_accepted / completed_pairs if completed_pairs else 0.0
        )
        return ExtractionBatch(
            bits=tuple(self._outputs[output_start:]),
            input_tosses=batch_tosses,
            accepted_pairs=batch_accepted,
            rejected_pairs=self.rejected_pairs - before_rejected,
            pending_tosses=int(self.pending_bit is not None),
            acceptance_rate=acceptance_rate_value,
            status="PASS" if completed_pairs or self.pending_bit is not None else "NO_INPUT",
        )


def von_neumann_extract(bits: Iterable[int]) -> list[int]:
    return list(VonNeumannExtractor().update(bits).bits)


def acceptance_probability(p: float) -> float:
    if not isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0,1]")
    return 2.0 * p * (1.0 - p)


def expected_tosses_per_output(p: float) -> float:
    probability = acceptance_probability(p)
    return float("inf") if probability == 0.0 else 2.0 / probability


def binary_entropy(p: float) -> float:
    if not isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0,1]")
    if p in (0.0, 1.0):
        return 0.0
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p)


def iid_efficiency(p: float) -> float:
    """IID output rate divided by IID Bernoulli marginal entropy."""
    rate = p * (1.0 - p)
    entropy = binary_entropy(p)
    return 0.0 if entropy == 0.0 else rate / entropy


def efficiency(p: float) -> float:
    """Backward-compatible IID efficiency alias."""
    return iid_efficiency(p)
