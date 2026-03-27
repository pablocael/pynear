"""
Approximate search benchmark: IVFFlatL2Index vs Faiss IndexIVFFlat.

Measures both query latency and recall@k across:
  - multiple dimensionalities (128, 256, 512, 1024)
  - multiple n_probe values (to build a recall vs speed Pareto curve)

Output images are written to ./results/approximate-l2-high-dimensionality/
"""

import os
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np

import pynear
from pynear.benchmark.dataset import generate_gaussian_dataset

# ── Config ─────────────────────────────────────────────────────────────────────

DATASET_SIZE = 50_000
DATASET_CLUSTERS = 50
DIMENSIONS = [128, 256, 512, 1024]
N_QUERIES = 32
K = 10
N_PROBE_VALUES = [1, 5, 10, 20, 40, 80]
NUM_AVG_RUNS = 5
OUTPUT_DIR = "./results/approximate-l2-high-dimensionality"

# ── Helpers ────────────────────────────────────────────────────────────────────


def compute_recall(approx_indices, exact_indices, k):
    """Mean recall@k over all queries."""
    return float(np.mean([
        len(set(a[:k]) & set(e[:k])) / k
        for a, e in zip(approx_indices, exact_indices)
    ]))


def timed_search(search_fn, query, k, n_runs):
    """Return (mean_seconds, results) averaged over n_runs, rejecting outliers."""
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = search_fn(query, k)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    # reject outliers beyond 1.5 std
    mask = np.abs(times - times.mean()) < 1.5 * (times.std() + 1e-12)
    return float(times[mask].mean()), result


def build_exact_ground_truth(data, queries, k):
    """Exact brute-force L2 search via Faiss FlatL2 (fastest exact baseline)."""
    d = data.shape[1]
    idx = faiss.IndexFlatL2(d)
    idx.add(data)
    _, indices = idx.search(queries, k)
    return indices.tolist()


# ── Per-dimension benchmark ────────────────────────────────────────────────────


def benchmark_dimension(dim, data, queries, k):
    """
    Returns a dict with keys 'vpforest' and 'faiss_ivf', each a list of
    {n_probe, time, recall} dicts.
    """
    print(f"  [dim={dim}] computing ground truth ... ", end="", flush=True)
    exact = build_exact_ground_truth(data, queries, k)
    print("done")

    n_clusters = max(10, int(np.sqrt(len(data))))
    results = {"vpforest": [], "faiss_ivf": []}

    for n_probe in N_PROBE_VALUES:
        n_probe_clamped = min(n_probe, n_clusters)

        # ── IVFFlatL2 ─────────────────────────────────────────────────────────
        forest = pynear.IVFFlatL2Index(n_clusters=n_clusters, n_probe=n_probe_clamped)
        forest.set(data)

        def vpf_search(q, k_):
            return forest.searchKNN(q, k_)

        t, res = timed_search(vpf_search, queries, k, NUM_AVG_RUNS)
        recall = compute_recall(res[0], exact, k)
        results["vpforest"].append({"n_probe": n_probe_clamped, "time": t, "recall": recall})
        print(f"  [dim={dim}] IVFFlatL2  n_probe={n_probe_clamped:3d}  "
              f"time={t*1000:.1f}ms  recall={recall:.3f}")

        # ── Faiss IVF ─────────────────────────────────────────────────────────
        quantizer = faiss.IndexFlatL2(dim)
        ivf = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_L2)
        ivf.train(data)
        ivf.add(data)
        ivf.nprobe = n_probe_clamped

        def faiss_search(q, k_):
            _, I = ivf.search(q, k_)
            return I.tolist()

        t, res = timed_search(faiss_search, queries, k, NUM_AVG_RUNS)
        recall = compute_recall(res, exact, k)
        results["faiss_ivf"].append({"n_probe": n_probe_clamped, "time": t, "recall": recall})
        print(f"  [dim={dim}] FaissIVF  n_probe={n_probe_clamped:3d}  "
              f"time={t*1000:.1f}ms  recall={recall:.3f}")

    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

COLORS = {"vpforest": "#5b8ff9", "faiss_ivf": "#f6bd16"}
LABELS = {"vpforest": "IVFFlatL2 (PyNear)", "faiss_ivf": "Faiss IndexIVFFlat"}


def plot_recall_vs_time(all_results, output_dir):
    """Recall@k vs query latency (ms) — one line per library, one subplot per dim."""
    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(5 * len(DIMENSIONS), 4), sharey=True)
    fig.suptitle(
        f"Recall@{K} vs Query Latency  |  N={DATASET_SIZE:,}  |  n_probe sweep",
        fontsize=13,
    )

    for ax, dim in zip(axes, DIMENSIONS):
        res = all_results[dim]
        for key in ("vpforest", "faiss_ivf"):
            pts = res[key]
            times_ms = [p["time"] * 1000 for p in pts]
            recalls = [p["recall"] for p in pts]
            ax.plot(times_ms, recalls, "o-", color=COLORS[key], label=LABELS[key], linewidth=2)
            for p, t, r in zip(pts, times_ms, recalls):
                ax.annotate(
                    str(p["n_probe"]),
                    (t, r),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=COLORS[key],
                )
        ax.set_title(f"{dim}-D")
        ax.set_xlabel("Query time (ms)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f"Recall@{K}")
    axes[-1].legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "recall_vs_time.png")
    plt.savefig(path, dpi=150)
    plt.clf()
    print(f"  saved {path}")


def plot_time_vs_dim(all_results, output_dir):
    """Query latency vs dimensionality at a fixed n_probe."""
    target_n_probe = 20

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(
        f"Query Latency vs Dimensionality  |  N={DATASET_SIZE:,}  |  n_probe≈{target_n_probe}  |  k={K}",
        fontsize=11,
    )

    for key in ("vpforest", "faiss_ivf"):
        times_ms = []
        dims_used = []
        for dim in DIMENSIONS:
            pts = all_results[dim][key]
            # pick the entry closest to target_n_probe
            closest = min(pts, key=lambda p: abs(p["n_probe"] - target_n_probe))
            times_ms.append(closest["time"] * 1000)
            dims_used.append(dim)
        ax.plot(dims_used, times_ms, "o-", color=COLORS[key], label=LABELS[key], linewidth=2)

    ax.set_xlabel("Dimensionality")
    ax.set_ylabel("Query time (ms)")
    ax.set_xticks(DIMENSIONS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "time_vs_dim.png")
    plt.savefig(path, dpi=150)
    plt.clf()
    print(f"  saved {path}")


def plot_recall_vs_dim(all_results, output_dir):
    """Recall@k vs dimensionality at a fixed n_probe."""
    target_n_probe = 20

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(
        f"Recall@{K} vs Dimensionality  |  N={DATASET_SIZE:,}  |  n_probe≈{target_n_probe}",
        fontsize=11,
    )

    for key in ("vpforest", "faiss_ivf"):
        recalls = []
        dims_used = []
        for dim in DIMENSIONS:
            pts = all_results[dim][key]
            closest = min(pts, key=lambda p: abs(p["n_probe"] - target_n_probe))
            recalls.append(closest["recall"])
            dims_used.append(dim)
        ax.plot(dims_used, recalls, "o-", color=COLORS[key], label=LABELS[key], linewidth=2)

    ax.set_xlabel("Dimensionality")
    ax.set_ylabel(f"Recall@{K}")
    ax.set_xticks(DIMENSIONS)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "recall_vs_dim.png")
    plt.savefig(path, dpi=150)
    plt.clf()
    print(f"  saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    for dim in DIMENSIONS:
        print(f"\n=== dim={dim} ===")
        data = generate_gaussian_dataset(
            DATASET_SIZE, DATASET_CLUSTERS, dim, data_type=np.float32
        )
        queries = generate_gaussian_dataset(
            N_QUERIES, 1, dim, data_type=np.float32
        )
        all_results[dim] = benchmark_dimension(dim, data, queries, K)

    print("\nGenerating plots ...")
    plot_recall_vs_time(all_results, OUTPUT_DIR)
    plot_time_vs_dim(all_results, OUTPUT_DIR)
    plot_recall_vs_dim(all_results, OUTPUT_DIR)
    print(f"\nDone. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
