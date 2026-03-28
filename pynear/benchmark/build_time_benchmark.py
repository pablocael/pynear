"""
Build-time scalability benchmark.

Measures index build time vs dataset size N for:
  Exact:       VPTreeL2Index (PyNear)   vs  Faiss IndexFlatL2
  Approximate: IVFFlatL2Index (PyNear)  vs  Faiss IndexIVFFlat

Fixed dimensionality:
  Exact:       d = 32   (VP-Tree sweet spot)
  Approximate: d = 128  (IVFFlat typical use case)

Data sizes: 10k, 50k, 100k, 250k, 500k, 1 000 000

Output: results printed to stdout + saved to ./results/build_time/
"""

import os
import time

import faiss
import numpy as np

import pynear
from pynear.benchmark.dataset import generate_gaussian_dataset

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_SIZES    = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
EXACT_DIM     = 32
APPROX_DIM    = 128
DATASET_CLUSTERS = 50
OUTPUT_DIR    = "./results/build_time"


def n_runs_for(n: int) -> int:
    """Fewer repetitions for large N to keep total wall time manageable."""
    if n >= 500_000:
        return 1
    if n >= 100_000:
        return 2
    return 3


def timed_build(build_fn: callable, n_runs: int) -> float:
    """Return mean build time in seconds."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        build_fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# ── Exact benchmark ────────────────────────────────────────────────────────────

def benchmark_exact(data: np.ndarray) -> dict:
    n, d = data.shape
    runs = n_runs_for(n)

    # PyNear VPTreeL2Index
    def build_vptree():
        idx = pynear.VPTreeL2Index()
        idx.set(data)

    # Faiss IndexFlatL2 (trivial: just copies data)
    def build_faiss_flat():
        idx = faiss.IndexFlatL2(d)
        idx.add(data)

    t_vptree = timed_build(build_vptree, runs)
    t_faiss  = timed_build(build_faiss_flat, runs)
    return {"vptree": t_vptree, "faiss_flat": t_faiss}


# ── Approximate benchmark ──────────────────────────────────────────────────────

def benchmark_approx(data: np.ndarray) -> dict:
    n, d = data.shape
    runs   = n_runs_for(n)
    n_clusters = max(10, int(np.sqrt(n)))

    # PyNear IVFFlatL2Index
    def build_pynear():
        idx = pynear.IVFFlatL2Index(n_clusters=n_clusters, n_probe=1)
        idx.set(data)

    # Faiss IndexIVFFlat
    def build_faiss_ivf():
        q   = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFFlat(q, d, n_clusters, faiss.METRIC_L2)
        idx.train(data)
        idx.add(data)

    t_pynear    = timed_build(build_pynear, runs)
    t_faiss_ivf = timed_build(build_faiss_ivf, runs)
    return {"ivfflat": t_pynear, "faiss_ivf": t_faiss_ivf}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    exact_results  = {}
    approx_results = {}

    print(f"{'N':>10}  {'VPTree(ms)':>12}  {'FaissFlat(ms)':>14}  "
          f"{'IVFFlatL2(ms)':>14}  {'FaissIVF(ms)':>13}")
    print("-" * 75)

    for n in DATA_SIZES:
        print(f"N={n:,} ...", flush=True)

        # Exact
        data_exact = generate_gaussian_dataset(
            n, DATASET_CLUSTERS, EXACT_DIM, data_type=np.float32
        )
        exact_results[n] = benchmark_exact(data_exact)

        # Approximate
        data_approx = generate_gaussian_dataset(
            n, DATASET_CLUSTERS, APPROX_DIM, data_type=np.float32
        )
        approx_results[n] = benchmark_approx(data_approx)

        er = exact_results[n]
        ar = approx_results[n]
        print(f"  N={n:>9,}  "
              f"VPTree={er['vptree']*1000:>8.0f}ms  "
              f"FaissFlat={er['faiss_flat']*1000:>8.1f}ms  "
              f"IVFFlatL2={ar['ivfflat']*1000:>8.0f}ms  "
              f"FaissIVF={ar['faiss_ivf']*1000:>8.0f}ms")

    # ── Save results ──────────────────────────────────────────────────────────
    header = "N,vptree_ms,faiss_flat_ms,ivfflat_ms,faiss_ivf_ms"
    rows = []
    for n in DATA_SIZES:
        er = exact_results[n]
        ar = approx_results[n]
        rows.append(
            f"{n},"
            f"{er['vptree']*1000:.1f},"
            f"{er['faiss_flat']*1000:.1f},"
            f"{ar['ivfflat']*1000:.1f},"
            f"{ar['faiss_ivf']*1000:.1f}"
        )
    csv_path = os.path.join(OUTPUT_DIR, "build_times.csv")
    with open(csv_path, "w") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n")
    print(f"\nSaved {csv_path}")

    # ── Print LaTeX-ready table rows ──────────────────────────────────────────
    print("\n--- LaTeX table rows ---")
    print("N  &  VPTreeL2  &  Faiss Flat  &  IVFFlatL2  &  Faiss IVF  \\\\")
    for n in DATA_SIZES:
        er = exact_results[n]
        ar = approx_results[n]
        n_fmt = f"{n//1000}k" if n < 1_000_000 else "1M"
        print(f"{n_fmt} & {er['vptree']*1000:.0f} & {er['faiss_flat']*1000:.1f} "
              f"& {ar['ivfflat']*1000:.0f} & {ar['faiss_ivf']*1000:.0f} \\\\")

    # ── Print pgfplots-ready coordinates ────────────────────────────────────
    print("\n--- pgfplots coordinates ---")
    print("VPTreeL2:")
    print("  " + " ".join(f"({n},{exact_results[n]['vptree']*1000:.0f})" for n in DATA_SIZES))
    print("Faiss FlatL2:")
    print("  " + " ".join(f"({n},{exact_results[n]['faiss_flat']*1000:.1f})" for n in DATA_SIZES))
    print("IVFFlatL2Index:")
    print("  " + " ".join(f"({n},{approx_results[n]['ivfflat']*1000:.0f})" for n in DATA_SIZES))
    print("Faiss IVFFlat:")
    print("  " + " ".join(f"({n},{approx_results[n]['faiss_ivf']*1000:.0f})" for n in DATA_SIZES))


if __name__ == "__main__":
    main()
