from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
import math
import re
from threading import RLock, get_ident
from typing import Callable, Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CacheKey:
    artifact: str
    input_hash: str
    knowledge_cutoff: str
    code_version: str
    config_hash: str

    def __post_init__(self) -> None:
        if not self.artifact or not self.code_version:
            raise ValueError("artifact and code_version are required")
        if not self.knowledge_cutoff.endswith("Z"):
            raise ValueError("knowledge_cutoff must be a canonical UTC timestamp ending in Z")
        if SHA256_PATTERN.fullmatch(self.input_hash) is None:
            raise ValueError("input_hash must be a lowercase SHA-256 digest")
        if SHA256_PATTERN.fullmatch(self.config_hash) is None:
            raise ValueError("config_hash must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CacheLookup(Generic[V]):
    hit: bool
    value: V | None


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    waits: int
    builds: int
    evictions: int


class LRUCache(Generic[K, V]):
    """Correctness reference with single-flight construction per semantic key."""

    def __init__(
        self,
        capacity: int,
        *,
        copier: Callable[[V], V] | None = None,
        wait_timeout_seconds: float = 30.0,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not math.isfinite(wait_timeout_seconds) or wait_timeout_seconds <= 0.0:
            raise ValueError("wait_timeout_seconds must be finite and positive")
        self.capacity = capacity
        self.wait_timeout_seconds = wait_timeout_seconds
        self._data: OrderedDict[K, V] = OrderedDict()
        self._inflight: dict[K, Future[V]] = {}
        self._builder_threads: dict[K, int] = {}
        self._copier = copier or deepcopy
        self._hits = 0
        self._misses = 0
        self._waits = 0
        self._builds = 0
        self._evictions = 0
        self._lock = RLock()

    def lookup(self, key: K) -> CacheLookup[V]:
        with self._lock:
            if key not in self._data:
                self._misses += 1
                return CacheLookup(hit=False, value=None)
            self._hits += 1
            self._data.move_to_end(key)
            return CacheLookup(hit=True, value=self._copier(self._data[key]))

    def get(self, key: K) -> V:
        lookup = self.lookup(key)
        if not lookup.hit:
            raise KeyError(key)
        return lookup.value  # type: ignore[return-value]

    def put(self, key: K, value: V) -> None:
        published = self._copier(value)
        with self._lock:
            if key in self._data:
                self._data[key] = published
                self._data.move_to_end(key)
                return
            self._data[key] = published
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)
                self._evictions += 1

    def get_or_compute(self, key: K, factory: Callable[[], V]) -> V:
        with self._lock:
            if key in self._data:
                self._hits += 1
                self._data.move_to_end(key)
                return self._copier(self._data[key])
            future = self._inflight.get(key)
            if future is None:
                future = Future()
                self._inflight[key] = future
                self._builder_threads[key] = get_ident()
                self._misses += 1
                self._builds += 1
                builder = True
            else:
                if self._builder_threads.get(key) == get_ident():
                    raise RuntimeError(
                        "recursive single-flight construction for the same key"
                    )
                self._waits += 1
                builder = False

        if not builder:
            return self._copier(future.result(timeout=self.wait_timeout_seconds))

        try:
            value = factory()
            self.put(key, value)
        except BaseException as error:
            future.set_exception(error)
            with self._lock:
                self._inflight.pop(key, None)
                self._builder_threads.pop(key, None)
            raise
        future.set_result(self._copier(value))
        with self._lock:
            self._inflight.pop(key, None)
            self._builder_threads.pop(key, None)
        return self._copier(value)

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                waits=self._waits,
                builds=self._builds,
                evictions=self._evictions,
            )

    def keys_lru_to_mru(self) -> tuple[K, ...]:
        with self._lock:
            return tuple(self._data.keys())
