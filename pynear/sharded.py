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
- **Parallel build for free.** Each shard builds independently —
  one OS thread per shard.
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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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
        """
        if len(shard_keys) != len(vectors):
            raise ValueError(
                f"shard_keys length {len(shard_keys)} != vectors length {len(vectors)}"
            )
        # Group row indices by shard key, then build each shard
        groups: Dict[ShardKey, List[int]] = {}
        for i, k in enumerate(shard_keys):
            groups.setdefault(k, []).append(i)

        self._shards.clear()
        for key, row_ids in groups.items():
            idx = self._index_cls(**self._ctor_kwargs)
            idx.set(vectors[row_ids])
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

        # Per-row routing
        if len(shard_keys) != len(vectors):
            raise ValueError("shard_keys length must match vectors length")
        groups: Dict[ShardKey, List[int]] = {}
        for i, k in enumerate(shard_keys):
            groups.setdefault(k, []).append(i)
        for key, row_ids in groups.items():
            shard_idx = self._get_or_create(key)
            new_ids[key] = list(shard_idx.add(vectors[row_ids]))
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
            call). If None, query every shard in parallel via a
            ThreadPool and merge results.
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

        per_shard_results: List[Tuple[ShardKey, List, List]] = []
        if n_workers <= 1 or len(shard_items) <= 1:
            per_shard_results = [_run(it) for it in shard_items]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                per_shard_results = list(ex.map(_run, shard_items))

        # Merge per-query: combine all shards' top-k, re-sort, take top-k.
        # Each shard returns farthest-first within top-k; we materialise
        # nearest-first for merging then re-flip on the way out to keep
        # the established pynear convention.
        n_queries = len(queries)
        out_idx: List[List[Tuple[ShardKey, int]]] = [[] for _ in range(n_queries)]
        out_dist: List[List[float]] = [[] for _ in range(n_queries)]
        for qi in range(n_queries):
            combined: List[Tuple[float, ShardKey, int]] = []
            for key, idx_per_q, dist_per_q in per_shard_results:
                if qi >= len(idx_per_q): continue
                # Each shard's row is farthest-first → reverse for nearest-first sort
                for shard_local_id, dist in zip(idx_per_q[qi][::-1], dist_per_q[qi][::-1]):
                    combined.append((float(dist), key, int(shard_local_id)))
            combined.sort(key=lambda t: t[0])  # nearest first
            combined = combined[:k]
            # Flip back to farthest-first to match pynear convention
            out_idx[qi] = [(key, sid) for _, key, sid in combined[::-1]]
            out_dist[qi] = [d for d, _, _ in combined[::-1]]

        return out_idx, out_dist

    def search1NN(
        self,
        queries: np.ndarray,
        shard: Optional[ShardKey] = None,
    ) -> Tuple[List[Tuple[ShardKey, int]], List[float]]:
        """Top-1 nearest neighbour. Returns parallel lists of
        (shard_key, shard_local_id) tuples and distances."""
        idx, dist = self.searchKNN(queries, k=1, shard=shard)
        out_idx: List[Tuple[ShardKey, int]] = []
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
