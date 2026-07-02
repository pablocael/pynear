"""ARM64 NEON micro-benchmark for pynear's HNSW kernels.

Designed to be run on an Apple Silicon Mac (M1/M2/M3/M4) or AWS Graviton
host once v2.4 is installed, to report the speedup from the NEON paths
landed in feat/hnsw-index. The numbers go into the CHANGELOG / release
post.

Usage (on an ARM64 box with pynear installed):

    python -m pynear.benchmark.arm64_neon_benchmark

What it measures
----------------
For each of HNSWL2Index, HNSWCosineIndex, HNSWL2IndexSQ8, and
HNSWBinaryIndex:
  * Build time at N = {20 000, 100 000}, d = 128, single-threaded.
  * Mean query latency at ef_search = 64 and ef_search = 256.

There is no automated comparison to a "before" baseline here because the
scalar fallback isn't compiled in once the NEON path is active. The way
to compare is:

  1. On ARM64, rebuild pynear with the PYNEAR_FORCE_SCALAR compile flag
     (e.g. `CFLAGS="-DPYNEAR_FORCE_SCALAR" pip install -e .`), which makes
     DistanceFunctions.hpp skip the SIMD branches and use the scalar
     fallbacks, and run this script — record the "scalar" numbers.
  2. Restore the default build (NEON path active) and run this script
     again — record the "NEON" numbers.
  3. Ratio is the NEON speedup. Expected: ~2-4x on the float kernels,
     ~3-5x on the SQ8 kernel.

A simpler one-off comparison: `taskset -c 0 python -m timeit -s 'import
pynear, numpy as np; idx = pynear.HNSWL2Index(M=16, ef_construction=200,
ef_search=128); idx.set(np.random.randn(10000, 128).astype("float32"))'
'idx.searchKNN(np.random.randn(100, 128).astype("float32"), k=10)'`
"""

from __future__ import annotations

import platform
import time
from typing import Callable

import numpy as np

import pynear


def _time(fn: Callable[[], None], iters: int = 5) -> float:
    """Median wall-clock seconds over `iters` runs (drops warm-up effect)."""
    # One warm-up call (not measured) — first call may include cache misses
    # on the underlying data that the steady-state queries don't see.
    fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def _bench_float(cls, name: str, db: np.ndarray, q: np.ndarray, ef_list):
    print(f"\n  {name}")
    print(f"  {'-' * len(name)}")
    idx = cls(M=16, ef_construction=200, ef_search=ef_list[0])
    build_s = _time(lambda: idx.set(db), iters=3)
    print(f"    build       : {build_s * 1000:7.1f} ms   (N={len(db)}, d={db.shape[1]})")
    for ef in ef_list:
        idx.set_ef(ef)
        q_us = _time(lambda: idx.searchKNN(q, k=10), iters=5) / len(q) * 1e6
        print(f"    query ef={ef:3d} : {q_us:7.2f} us/query")


def _bench_binary(cls, name: str, db: np.ndarray, q: np.ndarray, ef_list, **ctor_kw):
    print(f"\n  {name}")
    print(f"  {'-' * len(name)}")
    idx = cls(M=16, ef_construction=200, ef_search=ef_list[0], **ctor_kw)
    build_s = _time(lambda: idx.set(db), iters=3)
    print(f"    build       : {build_s * 1000:7.1f} ms   (N={len(db)}, d={db.shape[1] * 8} bits)")
    for ef in ef_list:
        idx.set_ef(ef)
        q_us = _time(lambda: idx.searchKNN(q, k=10), iters=5) / len(q) * 1e6
        print(f"    query ef={ef:3d} : {q_us:7.2f} us/query")


def main():
    print("pynear ARM64 / NEON micro-benchmark")
    print(f"  pynear  : {pynear.__version__}")
    print(f"  python  : {platform.python_version()}")
    print(f"  machine : {platform.machine()}  ({platform.processor() or 'unknown CPU'})")
    print(f"  system  : {platform.system()}  {platform.release()}")

    rng = np.random.default_rng(42)
    EFS = [64, 256]

    # Two N sizes — small fits in L2, larger spills to L3 (more realistic
    # for vector workloads).
    for n in (20_000, 100_000):
        print("\n" + "=" * 64)
        print(f"N = {n:,}, d = 128, single-threaded")
        print("=" * 64)

        db_f = rng.standard_normal((n, 128)).astype(np.float32)
        q_f = rng.standard_normal((200, 128)).astype(np.float32)
        db_b = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
        q_b = rng.integers(0, 256, size=(200, 16), dtype=np.uint8)

        _bench_float(pynear.HNSWL2Index,    "HNSWL2Index",      db_f, q_f, EFS)
        _bench_float(pynear.HNSWCosineIndex,"HNSWCosineIndex",  db_f, q_f, EFS)
        _bench_float(pynear.HNSWL2IndexSQ8, "HNSWL2IndexSQ8",   db_f, q_f, EFS)
        _bench_binary(pynear.HNSWBinaryIndex,"HNSWBinaryIndex", db_b, q_b, EFS)

    print("\nDone. Paste these numbers into the v2.4 release notes / ARM64 column.")


if __name__ == "__main__":
    main()
