from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IcebergEvidence:
    executed_quantity: float
    maximum_displayed_quantity: float
    replenishment_count: int
    excess_execution_over_peak_display: float
    execution_to_peak_display_ratio: float | None
    hidden_quantity_lower_bound: float
    status: str


def infer_iceberg(
    displayed_after_events: FloatArray,
    executed_at_level: FloatArray,
    tolerance: float = 1e-9,
) -> IcebergEvidence:
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    displayed = np.asarray(displayed_after_events, dtype=float)
    executed = np.asarray(executed_at_level, dtype=float)
    if displayed.ndim != 1 or executed.ndim != 1 or len(displayed) != len(executed):
        raise ValueError("displayed and executed arrays must align")
    if len(displayed) == 0:
        raise ValueError("at least one event is required")
    if not np.all(np.isfinite(displayed)) or not np.all(np.isfinite(executed)):
        raise ValueError("quantities must be finite")
    if np.any(displayed < 0.0) or np.any(executed < 0.0):
        raise ValueError("quantities cannot be negative")
    max_displayed = float(displayed.max(initial=0.0))
    total_executed = float(executed.sum())
    replenishments = int(np.sum(np.diff(displayed) > tolerance))
    excess = max(0.0, total_executed - max_displayed)
    ratio = total_executed / max_displayed if max_displayed > 0.0 else None
    # Without an order-level feed, independent visible arrivals can explain all
    # executions. The conservative hidden-size lower bound is therefore zero.
    status = "EVIDENCE_OF_REPLENISHMENT" if excess > tolerance else "NO_EVIDENCE"
    return IcebergEvidence(
        total_executed,
        max_displayed,
        replenishments,
        excess,
        ratio,
        0.0,
        status,
    )
