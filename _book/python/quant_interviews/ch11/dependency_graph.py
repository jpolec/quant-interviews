from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import heapq
import json
import re
from typing import Callable, Hashable, Iterable, TypeVar

T = TypeVar("T", bound=Hashable)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _default_id(node: T) -> str:
    if isinstance(node, (str, int)):
        return f"{type(node).__name__}:{node}"
    raise TypeError("non-scalar vertices require an explicit stable_id function")


def _normalize(
    edges: Iterable[tuple[T, T]],
    nodes: Iterable[T] | None,
    stable_id: Callable[[T], str],
) -> tuple[list[tuple[T, T]], set[T]]:
    edge_list = list(dict.fromkeys(edges))
    node_set = set(nodes or ())
    for parent, child in edge_list:
        if parent == child:
            node_set.add(parent)
        node_set.update((parent, child))
    identifiers = [stable_id(node) for node in node_set]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("stable vertex identifiers must be unique")
    return edge_list, node_set


def topological_order(
    edges: Iterable[tuple[T, T]],
    *,
    nodes: Iterable[T] | None = None,
    stable_id: Callable[[T], str] = _default_id,
) -> list[T]:
    edge_list, node_set = _normalize(edges, nodes, stable_id)
    graph: dict[T, list[T]] = defaultdict(list)
    indegree: dict[T, int] = {node: 0 for node in node_set}
    for parent, child in edge_list:
        graph[parent].append(child)
        indegree[child] += 1
    for parent in graph:
        graph[parent].sort(key=stable_id)

    queue = [(stable_id(node), node) for node in node_set if indegree[node] == 0]
    heapq.heapify(queue)
    order: list[T] = []
    while queue:
        _, node = heapq.heappop(queue)
        order.append(node)
        for child in graph[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(queue, (stable_id(child), child))

    if len(order) != len(node_set):
        cycle = find_cycle(edge_list, nodes=node_set, stable_id=stable_id)
        raise ValueError(f"dependency cycle detected: {cycle}")
    return order


def find_cycle(
    edges: Iterable[tuple[T, T]],
    *,
    nodes: Iterable[T] | None = None,
    stable_id: Callable[[T], str] = _default_id,
) -> list[T]:
    edge_list, node_set = _normalize(edges, nodes, stable_id)
    graph: dict[T, list[T]] = defaultdict(list)
    for parent, child in edge_list:
        graph[parent].append(child)
    for parent in graph:
        graph[parent].sort(key=stable_id)

    color: dict[T, int] = {node: 0 for node in node_set}
    parent_map: dict[T, T] = {}
    for start in sorted(node_set, key=stable_id):
        if color[start] != 0:
            continue
        color[start] = 1
        stack: list[tuple[T, int]] = [(start, 0)]
        while stack:
            node, child_index = stack[-1]
            children = graph[node]
            if child_index >= len(children):
                color[node] = 2
                stack.pop()
                continue
            child = children[child_index]
            stack[-1] = (node, child_index + 1)
            if color[child] == 0:
                parent_map[child] = node
                color[child] = 1
                stack.append((child, 0))
            elif color[child] == 1:
                path = [child]
                current = node
                while current != child:
                    path.append(current)
                    current = parent_map[current]
                path.append(child)
                path.reverse()
                return path
    return []


def lineage_hash(
    node: T,
    parent_hashes: Iterable[tuple[T, str]],
    *,
    artifact_hash: str,
    code_hash: str,
    config_hash: str,
    schema_hash: str,
    parameters_hash: str,
    stable_id: Callable[[T], str] = _default_id,
) -> str:
    controls = {
        "artifact_hash": artifact_hash,
        "code_hash": code_hash,
        "config_hash": config_hash,
        "schema_hash": schema_hash,
        "parameters_hash": parameters_hash,
    }
    if any(SHA256_PATTERN.fullmatch(value) is None for value in controls.values()):
        raise ValueError("code, config and parameter hashes must be lowercase SHA-256")
    normalized_parents = tuple(
        (stable_id(parent), digest) for parent, digest in parent_hashes
    )
    node_id = stable_id(node)
    identifiers = [node_id, *(identifier for identifier, _ in normalized_parents)]
    if any(not identifier or not identifier.isascii() for identifier in identifiers):
        raise ValueError("lineage stable identifiers must be non-empty ASCII")
    parent_ids = [identifier for identifier, _ in normalized_parents]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("duplicate parent identity")
    if any(SHA256_PATTERN.fullmatch(digest) is None for _, digest in normalized_parents):
        raise ValueError("parent lineage hashes must be lowercase SHA-256")
    payload = {
        "schema_version": 1,
        "node": node_id,
        **controls,
        "parents": sorted(
            normalized_parents,
            key=lambda item: item[0],
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def impacted_nodes(
    edges: Iterable[tuple[T, T]],
    changed: Iterable[T],
    *,
    nodes: Iterable[T] | None = None,
    include_changed: bool = True,
    stable_id: Callable[[T], str] = _default_id,
) -> tuple[T, ...]:
    """Return the deterministic downstream invalidation closure.

    A zero-length path is included exactly when ``include_changed`` is true.
    Unknown changed nodes are rejected rather than silently ignored.
    """

    edge_list, graph_nodes = _normalize(edges, nodes, stable_id)
    changed_set = set(changed)
    unknown = changed_set - graph_nodes
    if unknown:
        raise ValueError(
            "changed nodes are absent from the dependency graph: "
            f"{sorted((stable_id(node) for node in unknown))}"
        )
    children: dict[T, list[T]] = defaultdict(list)
    for parent, child in edge_list:
        children[parent].append(child)
    for parent in children:
        children[parent].sort(key=stable_id)
    seen = set(changed_set) if include_changed else set()
    frontier = sorted(changed_set, key=stable_id)
    while frontier:
        node = frontier.pop(0)
        for child in children[node]:
            if child not in seen:
                seen.add(child)
                frontier.append(child)
        frontier.sort(key=stable_id)
    if not include_changed:
        seen.difference_update(changed_set)
    return tuple(sorted(seen, key=stable_id))
