from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Topic:
    name: str
    current: float
    target: float
    importance: float
    hours_per_point: float

    def __post_init__(self) -> None:
        values = (self.current, self.target, self.importance, self.hours_per_point)
        if not self.name or not all(isfinite(value) for value in values):
            raise ValueError("topic name is required and values must be finite")
        if not 0.0 <= self.current <= 5.0 or not 0.0 <= self.target <= 5.0:
            raise ValueError("current and target must lie in [0, 5]")
        if self.importance < 0.0 or self.hours_per_point <= 0.0:
            raise ValueError(
                "importance must be non-negative and hours_per_point positive"
            )


def allocate_hours(topics: list[Topic], total_hours: float) -> dict[str, float]:
    if not isfinite(total_hours) or total_hours < 0.0:
        raise ValueError("total_hours must be finite and non-negative")
    if len({topic.name for topic in topics}) != len(topics):
        raise ValueError("topic names must be unique")
    actual_need = {
        topic.name: max(0.0, topic.target - topic.current) * topic.hours_per_point
        for topic in topics
    }
    priority = {
        topic.name: max(0.0, topic.target - topic.current) * topic.importance
        for topic in topics
    }
    allocation = {topic.name: 0.0 for topic in topics}
    active = {
        topic.name
        for topic in topics
        if actual_need[topic.name] > 0.0 and priority[topic.name] > 0.0
    }
    remaining = min(total_hours, sum(actual_need[name] for name in active))
    while active and remaining > 1.0e-12:
        total_priority = sum(priority[name] for name in active)
        tentative = {
            name: remaining * priority[name] / total_priority for name in active
        }
        capped = {
            name
            for name in active
            if tentative[name] >= actual_need[name] - allocation[name]
        }
        if not capped:
            for name, hours in tentative.items():
                allocation[name] += hours
            remaining = 0.0
            break
        for name in capped:
            hours = actual_need[name] - allocation[name]
            allocation[name] += hours
            remaining -= hours
        active -= capped
    if any(allocation[name] > actual_need[name] + 1.0e-10 for name in allocation):
        raise AssertionError("preparation allocation exceeded actual need")
    return allocation


def weekly_plan(allocation: dict[str, float], weeks: int) -> list[dict[str, float]]:
    if weeks <= 0:
        raise ValueError("weeks must be positive")
    if any(not name or not isfinite(hours) or hours < 0.0 for name, hours in allocation.items()):
        raise ValueError("allocation must contain named non-negative finite hours")
    # Front-load foundations while retaining spaced repetition in later weeks.
    raw_weights = [1.0 + 0.25 * (weeks - week - 1) for week in range(weeks)]
    total_weight = sum(raw_weights)
    return [
        {name: hours * raw_weights[week] / total_weight for name, hours in allocation.items()}
        for week in range(weeks)
    ]
