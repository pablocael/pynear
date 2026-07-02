"""Sharded HNSW index — a Python wrapper around multiple pynear HNSW indices.

Designed for the "many small partitions" use case (multi-tenant SaaS,
per-category catalogues, time-bucketed datasets) where you'd otherwise
manage N indices manually. Each shard is an independent
HNSWL2Index / HNSWCosineIndex / HNSWL2IndexSQ8 / HNSWBinaryIndex —
pickled to its own file in a single directory.

What sharding buys you
----------------------
- **Tenant isolation.** Query a single shard at production-query
  speed; cross-shard queries also work but scan all shards.
- **Faster incremental rebuilds.** Only the affected shard rebuilds
  on a big add/remove, not the whole index.
- **Parallel build for free.** Shard builds are dispatched across a
  shared thread pool, so independent shards build concurrently
  (bounded so per-index OpenMP threads don't oversubscribe cores).
- **Manageable pickle files.** Each shard is a normal pickle the
  user can copy, version, gzip, or ship to S3 individually.

What sharding does NOT do
-------------------------
- **Reduce memory.** Sharded indices still live in RAM. For
  out-of-core storage you'd need mmap (a future v2.5+ feature).
- **Improve recall.** A single 1M-vector HNSW gives slightly
  better recall than 10×100k shards (each shard's graph has
  fewer long-range links). Typically < 2% difference.
- **Help very-selective filtering.** If your "filter" is really
  "tenant_id == X", route as `shard=X` instead of using `filter=`.

Quick example
-------------
    import pynear

    # Build sharded index with per-tenant partitioning
    shards = pynear.ShardedHNSWIndex(
        index_cls=pynear.HNSWCosineIndex,
        M=16, ef_construction=200, ef_search=64,
    )
    shards.set(vectors, shard_keys=tenant_ids)

    # Save to a directory (one .pkl per shard + manifest.json)
    shards.save("./tenants/")

    # ...later, in another process...
    shards = pynear.ShardedHNSWIndex.load("./tenants/")

    # Single-tenant query (fast — only one shard scanned)
    hits, dists = shards.searchKNN(query, k=10, shard="tenant_42")

    # Cross-tenant query (parallel — all shards scanned, top-k merged)
    hits, dists = shards.searchKNN(query, k=10)
"""

from __future__ import annotations

import json
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


ShardKey = Union[str, int]


class ShardedHNSWIndex:
    """Directory-backed shard manager for pynear HNSW indices.

    The wrapped index class can be any of `HNSWL2Index`,
    `HNSWCosineIndex`, `HNSWL2IndexSQ8`, `HNSWBinaryIndex`. All
    shards share the same constructor kwargs.

    Per-shard mutation (`add(vectors, shard_keys=...)`,
    `remove(node_id, shard=...)`, `rebuild(shard=...)`) works
    naturally — each shard is a normal pynear HNSW under the hood.

    Concurrency: like the underlying HNSW indices, `add` / `remove`
    / `rebuild` are single-mutator per shard. `searchKNN` is
    safe to call concurrently across threads as long as no mutator
    is in flight.
    """

    _MANIFEST_NAME = "manifest.json"

    def __init__(
        self,
        index_cls: type,
        **index_ctor_kwargs: Any,
    ):
        """Create an empty sharded index.

        index_cls: the pynear HNSW class to instantiate per shard.
        **index_ctor_kwargs: forwarded to every per-shard constructor
            (M, ef_construction, ef_search, seed, n_threads, ...).
        """
        if not hasattr(index_cls, "set") or not hasattr(index_cls, "searchKNN"):
            raise TypeError(
                f"index_cls={index_cls!r} doesn't look like a pynear index "
                "(missing set / searchKNN)"
            )
        self._index_cls = index_cls
        self._ctor_kwargs = dict(index_ctor_kwargs)
        # shard_key (str/int) → pynear index instance
        self._shards: Dict[ShardKey, Any] = {}
        # Lazily created, cached thread pool (not picklable — see __getstate__).
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_size: int = 0

    # ─── Build / mutate ─────────────────────────────────────────────────

    def set(
        self,
        vectors: np.ndarray,
        shard_keys: Sequence[ShardKey],
    ) -> None:
        """Replace all shards by partitioning `vectors` according to `shard_keys`.

        vectors: 2D array, shape (N, D)
        shard_keys: per-row label, length N. Each unique value becomes
            a shard. Strings or ints both work.

        Shards build in parallel on a thread pool (the underlying
        index releases the GIL during `set`). Concurrency is capped
        so that per-index OpenMP threads (`n_threads` ctor kwarg)
        don't oversubscribe the machine.
        """
        if len(shard_keys) != len(vectors):
            raise ValueError(
                f"shard_keys length {len(shard_keys)} != vectors length {len(vectors)}"
            )
        # Group row indices by shard key, then build each shard
        groups = self._group_by_key(shard_keys)
        items = list(groups.items())

        def _build(item: Tuple[ShardKey, List[int]]) -> Tuple[ShardKey, Any]:
            key, row_ids = item
            idx = self._index_cls(**self._ctor_kwargs)
            idx.set(vectors[row_ids])
            return key, idx

        self._shards.clear()
        # Results arrive in submission order, so shard insertion order
        # (and thus cross-shard tie-breaking) matches the serial build.
        for key, idx in self._map_parallel(_build, items, self._build_concurrency(len(items))):
            self._shards[key] = idx

    def add(
        self,
        vectors: np.ndarray,
        shard_keys: Optional[Sequence[ShardKey]] = None,
        shard: Optional[ShardKey] = None,
    ) -> Dict[ShardKey, List[int]]:
        """Append vectors to one or more shards.

        Two calling modes:
          * `add(vectors, shard="t42")` — all vectors go to shard "t42".
          * `add(vectors, shard_keys=[k_per_row, ...])` — vectors split
             across shards by the per-row key. New shards are created
             on demand.

        Returns: a dict mapping shard_key → list of new node ids in
        that shard (per-shard id space, not global).
        """
        if (shard is None) == (shard_keys is None):
            raise ValueError("provide exactly one of `shard` or `shard_keys`")

        new_ids: Dict[ShardKey, List[int]] = {}
        if shard is not None:
            shard_idx = self._get_or_create(shard)
            new_ids[shard] = list(shard_idx.add(vectors))
            return new_ids

        # Per-row routing: fan the per-shard appends out on the thread
        # pool (bounded like `set` to avoid OpenMP oversubscription).
        assert shard_keys is not None  # guaranteed by the exactly-one check above
        if len(shard_keys) != len(vectors):
            raise ValueError("shard_keys length must match vectors length")
        groups = self._group_by_key(shard_keys)
        # Create missing shards up-front on this thread so shard dict
        # insertion order stays deterministic (first-appearance order).
        items = [(key, self._get_or_create(key), row_ids) for key, row_ids in groups.items()]

        def _add(item: Tuple[ShardKey, Any, List[int]]) -> Tuple[ShardKey, List[int]]:
            key, shard_idx, row_ids = item
            return key, list(shard_idx.add(vectors[row_ids]))

        for key, ids in self._map_parallel(_add, items, self._build_concurrency(len(items))):
            new_ids[key] = ids
        return new_ids

    def remove(self, node_id: int, shard: ShardKey) -> None:
        """Tombstone a node in a specific shard. (node_id is the shard-local id.)"""
        self._require_shard(shard).remove(node_id)

    def rebuild(self, shard: Optional[ShardKey] = None) -> Dict[ShardKey, List[int]]:
        """Compact tombstones away.

        shard=None: rebuild every shard, return dict of remappings.
        shard="t42": rebuild just that one, return single-entry dict.
        """
        mappings: Dict[ShardKey, List[int]] = {}
        targets = [shard] if shard is not None else list(self._shards.keys())
        for key in targets:
            mappings[key] = list(self._require_shard(key).rebuild())
        return mappings

    # ─── Query ──────────────────────────────────────────────────────────

    def searchKNN(
        self,
        queries: np.ndarray,
        k: int,
        shard: Optional[ShardKey] = None,
        n_workers: Optional[int] = None,
    ) -> Tuple[List[List[Tuple[ShardKey, int]]], List[List[float]]]:
        """Top-k nearest neighbours across one or all shards.

        shard: if provided, query only that shard (fast — one HNSW
            call). If None, query every shard in parallel on a cached
            per-instance thread pool and merge results.
        n_workers: parallelism for cross-shard queries (default:
            number of shards, capped at os.cpu_count()).

        Returns:
            indices:   list of n_queries entries; each is a list of
                       up to k (shard_key, shard_local_id) tuples.
                       Order matches the wrapped index's convention:
                       *farthest-first within the top-k*.
            distances: matching per-query distance lists.
        """
        if not self._shards:
            return [[] for _ in range(len(queries))], [[] for _ in range(len(queries))]

        if shard is not None:
            local_idx, local_dist = self._require_shard(shard).searchKNN(queries, k)
            indices = [[(shard, int(i)) for i in row] for row in local_idx]
            distances = [list(map(float, row)) for row in local_dist]
            return indices, distances

        # Cross-shard: query each in parallel, then merge
        n_workers = n_workers or min(len(self._shards), os.cpu_count() or 4)
        shard_items = list(self._shards.items())

        def _run(item):
            key, idx = item
            i, d = idx.searchKNN(queries, k)
            return key, i, d

        per_shard_results: List[Tuple[ShardKey, List, List]] = self._map_parallel(
            _run, shard_items, n_workers
        )

        # Merge per-query: combine all shards' top-k, re-sort, take top-k.
        # Each shard returns farthest-first within top-k; we materialise
        # nearest-first for merging then re-flip on the way out to keep
        # the established pynear convention.
        #
        # Vectorised: pad each shard's (possibly ragged) rows into a
        # (n_queries, n_shards*k) distance matrix — columns laid out
        # shard-major, nearest-first within each shard — then a single
        # stable argsort per query reproduces the historical Python
        # stable sort exactly (ties keep shard-then-position order).
        n_queries = len(queries)
        n_shards = len(per_shard_results)
        shard_key_list: List[ShardKey] = [key for key, _, _ in per_shard_results]

        total_cols = n_shards * k
        dist_mat = np.full((n_queries, total_cols), np.inf, dtype=np.float64)
        id_mat = np.full((n_queries, total_cols), -1, dtype=np.int64)
        counts = np.zeros(n_queries, dtype=np.int64)
        for s, (_key, idx_per_q, dist_per_q) in enumerate(per_shard_results):
            base = s * k
            for qi in range(min(n_queries, len(idx_per_q))):
                m = len(idx_per_q[qi])
                if m == 0:
                    continue
                # Shard rows are farthest-first → reverse to nearest-first
                dist_mat[qi, base:base + m] = np.asarray(dist_per_q[qi], dtype=np.float64)[::-1]
                id_mat[qi, base:base + m] = np.asarray(idx_per_q[qi], dtype=np.int64)[::-1]
                counts[qi] += m

        order = np.argsort(dist_mat, axis=1, kind="stable")[:, :k]  # nearest first
        top_dist = np.take_along_axis(dist_mat, order, axis=1)
        top_ids = np.take_along_axis(id_mat, order, axis=1)
        top_shard = order // k  # column → shard ordinal

        out_idx: List[List[Tuple[ShardKey, int]]] = [[] for _ in range(n_queries)]
        out_dist: List[List[float]] = [[] for _ in range(n_queries)]
        for qi in range(n_queries):
            m = int(min(k, counts[qi]))  # drop +inf padding
            if m == 0:
                continue
            # Flip back to farthest-first to match pynear convention
            out_idx[qi] = [
                (shard_key_list[s], int(sid))
                for s, sid in zip(top_shard[qi, m - 1::-1], top_ids[qi, m - 1::-1])
            ]
            out_dist[qi] = [float(d) for d in top_dist[qi, m - 1::-1]]

        return out_idx, out_dist

    def search1NN(
        self,
        queries: np.ndarray,
        shard: Optional[ShardKey] = None,
    ) -> Tuple[List[Tuple[Optional[ShardKey], int]], List[float]]:
        """Top-1 nearest neighbour. Returns parallel lists of
        (shard_key, shard_local_id) tuples and distances; rows with no
        match yield ``(None, -1)`` with distance ``inf``."""
        idx, dist = self.searchKNN(queries, k=1, shard=shard)
        out_idx: List[Tuple[Optional[ShardKey], int]] = []
        out_dist: List[float] = []
        for row_i, row_d in zip(idx, dist):
            if row_i:
                # Farthest-first ordering with k=1 → the single entry IS
                # the nearest. (No reverse needed for size-1 lists.)
                out_idx.append(row_i[0])
                out_dist.append(row_d[0])
            else:
                out_idx.append((None, -1))
                out_dist.append(float("inf"))
        return out_idx, out_dist

    # ─── Introspection ──────────────────────────────────────────────────

    @property
    def shard_keys(self) -> List[ShardKey]:
        return list(self._shards.keys())

    @property
    def n_shards(self) -> int:
        return len(self._shards)

    @property
    def size(self) -> int:
        """Total live vectors across all shards (excludes tombstones)."""
        return sum(s.size - s.num_deleted for s in self._shards.values())

    def shard_sizes(self) -> Dict[ShardKey, int]:
        return {k: s.size - s.num_deleted for k, s in self._shards.items()}

    def get_shard(self, shard: ShardKey) -> Any:
        """Direct access to the underlying pynear index for one shard."""
        return self._require_shard(shard)

    # ─── Persistence ────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Write each shard to `directory/shard_<key>.pkl` plus a manifest.

        Creates the directory if missing. Existing files for shards that
        don't exist in `self` are NOT removed (manual cleanup if you
        deleted shards in-process). The manifest is overwritten.
        """
        os.makedirs(directory, exist_ok=True)
        manifest = {
            "format_version": 1,
            "index_class": self._index_cls.__name__,
            "ctor_kwargs": self._ctor_kwargs,
            "shards": [],
        }
        for key, idx in self._shards.items():
            fname = self._shard_filename(key)
            with open(os.path.join(directory, fname), "wb") as f:
                pickle.dump(idx, f)
            manifest["shards"].append({"key": key, "file": fname})
        with open(os.path.join(directory, self._MANIFEST_NAME), "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(
        cls,
        directory: str,
        index_cls: Optional[type] = None,
    ) -> "ShardedHNSWIndex":
        """Reload from a directory previously written by `save()`.

        index_cls: optional override for the index class. By default we
            resolve the class name stored in the manifest by name from
            the `pynear` module. Pass this if your wrapped class lives
            elsewhere or under a different name.
        """
        manifest_path = os.path.join(directory, cls._MANIFEST_NAME)
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("format_version") != 1:
            raise ValueError(
                f"Unsupported sharded-index format_version: {manifest.get('format_version')}"
            )

        if index_cls is None:
            # Resolve by name from pynear's top-level namespace.
            import pynear as _pynear
            cls_name = manifest["index_class"]
            index_cls = getattr(_pynear, cls_name, None)
            if index_cls is None:
                raise ValueError(
                    f"Could not resolve index_class={cls_name!r}; pass it "
                    "explicitly via the index_cls kwarg."
                )

        inst = cls(index_cls, **manifest["ctor_kwargs"])
        for shard_entry in manifest["shards"]:
            key = shard_entry["key"]
            fname = shard_entry["file"]
            with open(os.path.join(directory, fname), "rb") as f:
                inst._shards[key] = pickle.load(f)
        return inst

    # ─── Internal helpers ───────────────────────────────────────────────

    def __getstate__(self) -> Dict[str, Any]:
        # ThreadPoolExecutors aren't picklable; drop the cached pool.
        state = self.__dict__.copy()
        state["_executor"] = None
        state["_executor_size"] = 0
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._executor = None
        self._executor_size = 0

    def _get_executor(self, min_workers: int) -> ThreadPoolExecutor:
        """Return the cached per-instance thread pool, lazily created.

        The pool is sized `min(32, cpu_count)` by default and only
        recreated (grown) if a caller explicitly needs more workers.
        """
        # Backwards-compat with instances unpickled from older versions
        # that never had the attributes.
        ex = getattr(self, "_executor", None)
        size = getattr(self, "_executor_size", 0)
        want = max(min(32, (os.cpu_count() or 4)), min_workers)
        if ex is None or size < min_workers:
            if ex is not None:
                ex.shutdown(wait=False)
            ex = ThreadPoolExecutor(max_workers=want, thread_name_prefix="pynear-shard")
            self._executor = ex
            self._executor_size = want
        return ex

    def _map_parallel(
        self,
        fn: Callable[[Any], Any],
        items: Sequence[Any],
        max_concurrency: int,
    ) -> List[Any]:
        """Run `fn` over `items` on the cached pool, preserving order.

        Worker exceptions propagate to the caller (re-raised while the
        result list is consumed). `max_concurrency` bounds how many
        `fn` calls run simultaneously; <=1 (or a single item) runs
        serially on the calling thread.
        """
        if max_concurrency <= 1 or len(items) <= 1:
            return [fn(it) for it in items]
        ex = self._get_executor(min(max_concurrency, len(items)))
        if max_concurrency < min(self._executor_size, len(items)):
            # Pool is wider than the requested concurrency — bound the
            # number of in-flight fn calls without shrinking the pool.
            sem = threading.Semaphore(max_concurrency)

            def bounded(it: Any) -> Any:
                with sem:
                    return fn(it)

            return list(ex.map(bounded, items))
        return list(ex.map(fn, items))

    def _build_concurrency(self, n_tasks: int) -> int:
        """How many shard builds/adds may run at once.

        Each underlying index may itself run `n_threads` OpenMP threads
        during set/add, so concurrent builds are capped at
        cpu_count // n_threads to avoid oversubscription.
        """
        cpu = os.cpu_count() or 4
        n_threads = self._ctor_kwargs.get("n_threads", 1)
        try:
            n_threads = int(n_threads)
        except (TypeError, ValueError):
            n_threads = 1
        cap = max(1, cpu // n_threads) if n_threads > 1 else cpu
        return max(1, min(n_tasks, cap))

    @staticmethod
    def _group_by_key(shard_keys: Sequence[ShardKey]) -> Dict[ShardKey, List[int]]:
        """Group row indices by their per-row shard key."""
        groups: Dict[ShardKey, List[int]] = {}
        for i, k in enumerate(shard_keys):
            groups.setdefault(k, []).append(i)
        return groups

    def _get_or_create(self, key: ShardKey) -> Any:
        if key not in self._shards:
            self._shards[key] = self._index_cls(**self._ctor_kwargs)
        return self._shards[key]

    def _require_shard(self, key: ShardKey) -> Any:
        if key not in self._shards:
            raise KeyError(f"no shard with key {key!r}; known shards: {list(self._shards)}")
        return self._shards[key]

    @staticmethod
    def _shard_filename(key: ShardKey) -> str:
        # Sanitise to avoid path traversal / awkward chars
        safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in str(key))
        return f"shard_{safe}.pkl"
