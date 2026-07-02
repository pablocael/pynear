"""pynear HNSW vs Faiss HNSW — thread-matched, subprocess-isolated comparison.

Compares HNSWL2Index / HNSWL2IndexSQ8 against faiss.IndexHNSWFlat /
faiss.IndexHNSWSQ (8-bit) on the same data, sweeping ef_search to trace the
recall-vs-throughput Pareto curve. Faiss runs in a separate faiss-only
subprocess: pynear links libgomp and faiss-cpu links libomp, and the two
OpenMP runtimes contend when loaded together (see results/faiss_comparison.md),
which would depress Faiss's parallel batch search.

Usage:  python -m pynear.benchmark.hnsw_faiss_benchmark
Output: results/hnsw_faiss_comparison.md + .json (relative to CWD)
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

N = 100_000
DIM = 128
NQ = 1_000
K = 10
M = 16
EF_CONSTRUCTION = 200
EF_SWEEP = [16, 32, 64, 128, 256, 512, 1024]
THREADS = 24
SEED = 42
REPEATS = 5

FAISS_WORKER = r"""
import json, sys, time
import numpy as np
import faiss

cfg = json.load(open(sys.argv[1]))
db = np.load(cfg["db"]); q = np.load(cfg["q"])
faiss.omp_set_num_threads(cfg["threads"])

out = {}
for kind in ["flat", "sq8"]:
    if kind == "flat":
        idx = faiss.IndexHNSWFlat(db.shape[1], cfg["M"])
    else:
        idx = faiss.IndexHNSWSQ(db.shape[1], faiss.ScalarQuantizer.QT_8bit, cfg["M"])
        idx.train(db)
    idx.hnsw.efConstruction = cfg["ef_construction"]
    t0 = time.perf_counter(); idx.add(db); build_s = time.perf_counter() - t0
    sweep = []
    for ef in cfg["ef_sweep"]:
        idx.hnsw.efSearch = ef
        times = []
        ids = None
        for _ in range(cfg["repeats"]):
            t0 = time.perf_counter(); _, ids = idx.search(q, cfg["k"]); times.append(time.perf_counter() - t0)
        sweep.append({"ef": ef, "best_s": min(times), "ids": ids.tolist()})
    out[kind] = {"build_s": build_s, "sweep": sweep}
json.dump(out, open(cfg["out"], "w"))
"""


def clustered_dataset(n, dim, n_clusters, rng):
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32) * 5.0
    labels = rng.integers(0, n_clusters, n)
    return (centers[labels] + rng.standard_normal((n, dim)).astype(np.float32)).astype(np.float32)


def brute_gt(db, q, k, chunk=100):
    n2 = (db.astype(np.float64) ** 2).sum(1)
    out = np.empty((len(q), k), dtype=np.int64)
    for s in range(0, len(q), chunk):
        qq = q[s : s + chunk].astype(np.float64)
        d = (qq ** 2).sum(1)[:, None] + n2[None, :] - 2.0 * qq @ db.astype(np.float64).T
        idx = np.argpartition(d, k, axis=1)[:, :k]
        dsel = np.take_along_axis(d, idx, axis=1)
        out[s : s + chunk] = np.take_along_axis(idx, np.argsort(dsel, axis=1), axis=1)
    return out


def recall(ids, gt):
    hits = sum(len(set(map(int, a)) & set(map(int, b))) for a, b in zip(ids, gt))
    return hits / float(gt.size)


def main():
    import pynear

    rng = np.random.default_rng(SEED)
    db = clustered_dataset(N, DIM, 50, rng)
    q = clustered_dataset(NQ, DIM, 50, np.random.default_rng(SEED + 1))
    print(f"Computing ground truth ({NQ} queries, k={K}) ...", flush=True)
    gt = brute_gt(db, q, K)

    results = {"config": {"N": N, "dim": DIM, "nq": NQ, "k": K, "M": M,
                          "ef_construction": EF_CONSTRUCTION, "threads": THREADS}}

    # --- pynear (this process) ---
    for name, cls in [("pynear_hnsw", pynear.HNSWL2Index), ("pynear_sq8", pynear.HNSWL2IndexSQ8)]:
        idx = cls(M=M, ef_construction=EF_CONSTRUCTION, ef_search=EF_SWEEP[0],
                  seed=SEED, n_threads=THREADS)
        t0 = time.perf_counter(); idx.set(db); build_s = time.perf_counter() - t0
        sweep = []
        for ef in EF_SWEEP:
            idx.set_ef(ef)
            times, ids = [], None
            for _ in range(REPEATS):
                t0 = time.perf_counter(); ids, _ = idx.searchKNN_arrays(q, K); times.append(time.perf_counter() - t0)
            r = recall(ids, gt)
            sweep.append({"ef": ef, "best_s": min(times), "qps": NQ / min(times), "recall": r})
            print(f"{name:14s} ef={ef:<4d} qps={NQ/min(times):>9.0f} recall={r:.3f}", flush=True)
        results[name] = {"build_s": build_s, "sweep": sweep}
        print(f"{name:14s} build {build_s:.2f}s", flush=True)

    # --- faiss (isolated subprocess) ---
    with tempfile.TemporaryDirectory() as td:
        np.save(f"{td}/db.npy", db); np.save(f"{td}/q.npy", q)
        cfg = {"db": f"{td}/db.npy", "q": f"{td}/q.npy", "out": f"{td}/out.json",
               "threads": THREADS, "M": M, "ef_construction": EF_CONSTRUCTION,
               "ef_sweep": EF_SWEEP, "k": K, "repeats": REPEATS}
        json.dump(cfg, open(f"{td}/cfg.json", "w"))
        worker = f"{td}/worker.py"
        open(worker, "w").write(FAISS_WORKER)
        print("Running Faiss in isolated subprocess ...", flush=True)
        subprocess.run([sys.executable, worker, f"{td}/cfg.json"], check=True)
        fx = json.load(open(f"{td}/out.json"))

    for kind, name in [("flat", "faiss_hnsw"), ("sq8", "faiss_hnsw_sq8")]:
        sweep = []
        for p in fx[kind]["sweep"]:
            r = recall(np.array(p["ids"]), gt)
            sweep.append({"ef": p["ef"], "best_s": p["best_s"], "qps": NQ / p["best_s"], "recall": r})
            print(f"{name:14s} ef={p['ef']:<4d} qps={NQ/p['best_s']:>9.0f} recall={r:.3f}", flush=True)
        results[name] = {"build_s": fx[kind]["build_s"], "sweep": sweep}
        print(f"{name:14s} build {fx[kind]['build_s']:.2f}s", flush=True)

    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/hnsw_faiss_comparison.json", "w"), indent=1)

    with open("results/hnsw_faiss_comparison.md", "w") as f:
        c = results["config"]
        f.write("# pynear HNSW vs Faiss HNSW\n\n")
        f.write(f"N={c['N']:,} x {c['dim']}-D float32 (clustered), {c['nq']} queries, "
                f"k={c['k']}, M={c['M']}, ef_construction={c['ef_construction']}, "
                f"{c['threads']} threads, best of {REPEATS} batch runs. Faiss measured in a "
                f"faiss-only subprocess (OpenMP-runtime isolation; see faiss_comparison.md).\n\n")
        f.write("| Index | Build (s) | " + " | ".join(f"ef={e}" for e in EF_SWEEP) + " |\n")
        f.write("|---|---|" + "---|" * len(EF_SWEEP) + "\n")
        for name, label in [("pynear_hnsw", "pynear HNSWL2Index"),
                            ("pynear_sq8", "pynear HNSWL2IndexSQ8"),
                            ("faiss_hnsw", "faiss IndexHNSWFlat"),
                            ("faiss_hnsw_sq8", "faiss IndexHNSWSQ(8bit)")]:
            row = results[name]
            cells = " | ".join(f"{p['qps']:,.0f} QPS @ {p['recall']:.3f}" for p in row["sweep"])
            f.write(f"| {label} | {row['build_s']:.2f} | {cells} |\n")
    print("wrote results/hnsw_faiss_comparison.{md,json}", flush=True)


if __name__ == "__main__":
    main()
