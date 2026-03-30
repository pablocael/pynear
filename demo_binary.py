#!/usr/bin/env python3
"""
pynear — SIFT1M Binary Descriptor Benchmark
============================================
Downloads the INRIA TEXMEX SIFT1M dataset (1 000 000 × 128-dim float32
vectors), sign-quantises it to 128-bit binary descriptors (16 bytes each),
then benchmarks IVFFlatBinaryIndex and MIHBinaryIndex for approximate
Hamming-distance k-NN search.

Metrics reported
----------------
  Build time  — wall-clock seconds to construct the index
  ms/query    — best-of-3 mean query latency (milliseconds per query)
  QPS         — queries per second (best-of-3)
  Recall@k    — fraction of queries for which ≥1 true neighbour is in top-k

Usage
-----
    python demo_binary.py                      # full SIFT1M (1 M vectors)
    python demo_binary.py --small              # SIFTsmall (10 K) — quick test
    python demo_binary.py --n-gt-queries 200   # fewer ground-truth queries
    python demo_binary.py --data-dir /tmp/sift # custom data directory
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

try:
    from pynear import IVFFlatBinaryIndex, MIHBinaryIndex
except ImportError:
    sys.exit("ERROR: pynear not installed.  Run:  pip install -e .")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

_SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
_SIFTSMALL_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"


def _progress(block_num: int, block_size: int, total: int) -> None:
    downloaded = block_num * block_size
    pct = min(100.0, 100.0 * downloaded / total) if total > 0 else 0.0
    filled = int(pct / 2)
    bar = "█" * filled + "░" * (50 - filled)
    print(
        f"\r  [{bar}] {pct:5.1f}%  "
        f"{downloaded / 1_048_576:.0f} / {total / 1_048_576:.0f} MB",
        end="",
        flush=True,
    )


def fetch_sift(data_dir: Path, small: bool) -> Path:
    """Download and extract the SIFT (or SIFTsmall) dataset; return folder path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    name = "siftsmall" if small else "sift"
    url = _SIFTSMALL_URL if small else _SIFT_URL
    archive = data_dir / f"{name}.tar.gz"
    folder = data_dir / name

    if not folder.exists():
        if not archive.exists():
            label = "SIFTsmall (~2.5 MB)" if small else "SIFT1M (~163 MB)"
            print(f"  Downloading {label} from INRIA TEXMEX …")
            try:
                urllib.request.urlretrieve(url, archive, reporthook=_progress)
            except Exception as exc:
                archive.unlink(missing_ok=True)
                sys.exit(
                    f"\n  Download failed: {exc}\n"
                    "  Manual download: ftp://ftp.irisa.fr/local/texmex/corpus/"
                )
            print()
        print(f"  Extracting {archive.name} …")
        with tarfile.open(archive) as tar:
            tar.extractall(data_dir)

    return folder


def read_fvecs(path: Path) -> np.ndarray:
    """Read a .fvecs file and return a float32 array of shape (N, D)."""
    raw = np.frombuffer(path.read_bytes(), dtype=np.int32)
    d = int(raw[0])
    # Each record is [d_int32, v[0..d-1]_float32], all packed back-to-back
    return raw.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()


# ─────────────────────────────────────────────────────────────────────────────
# Binarisation
# ─────────────────────────────────────────────────────────────────────────────

def sign_binarise(X: np.ndarray, center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Sign-quantise float32 vectors → packed uint8 binary descriptors.

    Subtracts ``center`` (or the column mean of X when None) then maps
    positive values to bit 1.  Returns ``(binary, center)`` where binary
    has shape (N, D // 8) and dtype uint8.
    """
    c: np.ndarray = X.mean(axis=0) if center is None else center
    bits = (X - c) > 0      # (N, D) bool — each row becomes D bits
    return np.packbits(bits, axis=1), c   # (N, D//8) uint8


# ─────────────────────────────────────────────────────────────────────────────
# Exact Hamming ground truth
# ─────────────────────────────────────────────────────────────────────────────

# Fast popcount via byte-level lookup table (no dependency on Numba / CUDA)
_LUT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming_to_all(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Return Hamming distances from one query row to every row in database."""
    xor = database ^ query          # (N, B) uint8
    return _LUT8[xor].sum(axis=1)  # (N,)  uint32-promoted by numpy sum


def compute_ground_truth(
    queries: np.ndarray,
    database: np.ndarray,
    k: int,
    cache_path: Path | None = None,
) -> np.ndarray:
    """
    Exact brute-force Hamming k-NN for *queries* against *database*.

    Returns an (Q, k) int32 array of neighbour indices.
    If *cache_path* is given and the file exists the result is loaded from
    disk; otherwise it is computed and saved there for future runs.
    """
    if cache_path is not None and cache_path.exists():
        print(f"  Loading cached ground truth ← {cache_path.name}")
        return np.load(cache_path)

    Q = len(queries)
    gt = np.empty((Q, k), dtype=np.int32)
    t0 = time.perf_counter()

    for i in range(Q):
        dists = _hamming_to_all(queries[i], database)
        idx = np.argpartition(dists, k)[:k]
        gt[i] = idx[np.argsort(dists[idx])]

        if (i + 1) % max(1, Q // 20) == 0 or i == Q - 1:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (Q - i - 1)
            print(
                f"\r  Ground truth  {i + 1:>{len(str(Q))}}/{Q}"
                f"  elapsed {elapsed:5.1f}s  eta {eta:5.1f}s …",
                end="",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t0
    print(f"\r  Ground truth  {Q}/{Q}  done in {elapsed_total:.1f}s            ")

    if cache_path is not None:
        np.save(cache_path, gt)
        print(f"  Saved ground truth → {cache_path}")

    return gt


# ─────────────────────────────────────────────────────────────────────────────
# Recall metric
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(
    retrieved: list[list[int]],
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """
    Mean Recall@k: fraction of queries for which at least one of the true
    top-k neighbours appears in the index's top-k results.
    """
    hits = sum(
        1
        for i, r in enumerate(retrieved)
        if any(x in set(ground_truth[i, :k].tolist()) for x in r[:k])
    )
    return hits / len(retrieved)


# ─────────────────────────────────────────────────────────────────────────────
# Query benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def _warm_up(index, queries: np.ndarray, k: int, extra: dict) -> None:
    index.searchKNN(queries[:min(5, len(queries))], k, **extra)


def benchmark_query(
    index,
    queries: np.ndarray,
    k: int,
    extra_kwargs: dict | None = None,
    n_repeats: int = 3,
) -> tuple[list[list[int]], float, float]:
    """
    Run ``index.searchKNN(queries, k)`` n_repeats times.

    Returns ``(results, ms_per_query, qps)`` from the fastest run.
    """
    kw = extra_kwargs or {}
    _warm_up(index, queries, k, kw)

    best_t = float("inf")
    results: list[list[int]] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        r, _ = index.searchKNN(queries, k, **kw)
        elapsed = time.perf_counter() - t0
        if elapsed < best_t:
            best_t = elapsed
            results = r

    ms = 1000.0 * best_t / len(queries)
    qps = len(queries) / best_t
    return results, ms, qps


# ─────────────────────────────────────────────────────────────────────────────
# Console table formatting
# ─────────────────────────────────────────────────────────────────────────────

def _col_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    return [
        max(len(h), max((len(r[i]) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]


def _hr(widths: list[int], cross: str = "+", fill: str = "-") -> str:
    return cross + cross.join(fill * (w + 2) for w in widths) + cross


def _row_str(values: list[str], widths: list[int]) -> str:
    return "|" + "|".join(f" {v:<{widths[i]}} " for i, v in enumerate(values)) + "|"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    w = _col_widths(headers, rows)
    print(_hr(w))
    print(_row_str(headers, w))
    print(_hr(w, "+", "="))
    for row in rows:
        print(_row_str(row, w))
    print(_hr(w))


def to_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    w = _col_widths(headers, rows)
    lines = [
        "| " + " | ".join(f"{h:<{w[i]}}" for i, h in enumerate(headers)) + " |",
        "| " + " | ".join("-" * wi for wi in w) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(f"{v:<{w[i]}}" for i, v in enumerate(row)) + " |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# QPS vs Recall plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    rows: list[list[str]],
    k: int,
    out_path: Path,
    dataset_name: str,
) -> None:
    """
    Save a grouped bar chart of QPS per configuration to *out_path*.

    Bars are grouped by index family (IVFFlat / MIH), coloured by a gradient
    that darkens as the tuning parameter increases (slower but higher recall).
    Recall@k is annotated above each bar.  Brute-force appears as a dashed
    horizontal reference line.
    """
    import matplotlib
    import matplotlib.colors
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as ticker

    FAMILIES = [
        ("IVFFlatBinaryIndex", "#1d4ed8", "nprobe"),   # blue family
        ("MIHBinaryIndex",     "#15803d", "radius"),   # green family
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f1f5f9")
    ax.grid(True, axis="y", which="both", color="white", linewidth=0.8, zorder=0)

    bar_width = 0.35
    group_gap = 0.9          # gap between the two family groups
    legend_patches = []
    x_ticks, x_labels = [], []
    x_cursor = 0.0

    for name, base_color, param_key in FAMILIES:
        family_rows = [r for r in rows if r[0] == name]
        if not family_rows:
            continue
        # Sort by the tuning parameter value (ascending = faster configs first)
        def _param_val(r: list[str]) -> float:
            cfg = r[1]
            if param_key + "=" in cfg:
                return float(cfg.split(param_key + "=")[1].split(",")[0].strip())
            return 0.0
        family_rows.sort(key=_param_val)

        n = len(family_rows)
        # Colour gradient: lightest for fastest, darkest for slowest
        base = matplotlib.colors.to_rgb(base_color)
        shades = [
            tuple(c + (1.0 - c) * (1 - (i + 1) / (n + 1)) for c in base)
            for i in range(n)
        ]

        group_start = x_cursor
        for i, (row, color) in enumerate(zip(family_rows, shades)):
            qps = float(row[4])
            recall = float(row[5])
            cfg = row[1]
            # Short label: just the tuning param value
            short = cfg.split(param_key + "=")[1].split(",")[0].strip() if param_key + "=" in cfg else cfg

            bar = ax.bar(x_cursor, qps, width=bar_width, color=color, zorder=3,
                         edgecolor="white", linewidth=0.5)

            # Annotate recall above the bar
            recall_str = f"{recall:.3f}" if recall < 1.0 else "1.00"
            ax.text(x_cursor, qps * 1.08, recall_str,
                    ha="center", va="bottom", fontsize=7,
                    color="#374151", fontweight="bold")

            x_ticks.append(x_cursor)
            x_labels.append(short)
            x_cursor += bar_width + 0.05

        # Group label centred under the group
        group_mid = (group_start + x_cursor - bar_width - 0.05) / 2
        ax.text(group_mid, -0.14, name, ha="center", va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=9, fontweight="bold",
                color=base_color)

        legend_patches.append(mpatches.Patch(color=base_color, label=name))
        x_cursor += group_gap   # gap between families

    # ── Brute-force reference line ─────────────────────────────────────────
    bf_rows = [r for r in rows if r[0] == "Brute-force (numpy)"]
    if bf_rows:
        bf_qps = float(bf_rows[0][4])
        ax.axhline(bf_qps, color="#dc2626", linewidth=1.5,
                   linestyle="--", zorder=4,
                   label=f"Brute-force  {bf_qps:.0f} QPS")
        legend_patches.append(
            mpatches.Patch(color="#dc2626", label=f"Brute-force  {bf_qps:.0f} QPS")
        )

    # ── Axes ───────────────────────────────────────────────────────────────
    ax.set_yscale("log")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_ylabel("QPS  (queries / second, log scale)", fontsize=11, labelpad=6)
    ax.set_xlabel(f"← {FAMILIES[0][2]} (IVFFlat)                     "
                  f"{FAMILIES[1][2]} (MIH) →", fontsize=9, labelpad=8)
    ax.set_title(
        f"pynear binary index QPS — {dataset_name}\n"
        f"Numbers above bars = Recall@{k}   (higher bar + 1.000 recall is best)",
        fontsize=11, pad=10,
    )
    ax.legend(handles=legend_patches, fontsize=9, loc="upper right", framealpha=0.85)
    ax.set_xlim(-bar_width, x_cursor - group_gap + bar_width)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# README.md update
# ─────────────────────────────────────────────────────────────────────────────

_MARKER_START = "<!-- binary-benchmark-start -->"
_MARKER_END = "<!-- binary-benchmark-end -->"


def update_readme(readme: Path, md_section: str) -> None:
    """Insert or replace the benchmark section in README.md."""
    text = readme.read_text()
    block = f"{_MARKER_START}\n{md_section}\n{_MARKER_END}"

    if _MARKER_START in text and _MARKER_END in text:
        # Replace existing section
        before = text[: text.index(_MARKER_START)]
        after = text[text.index(_MARKER_END) + len(_MARKER_END) :]
        text = before + block + after
    else:
        # Insert before ## Development
        anchor = "\n## Development"
        if anchor in text:
            idx = text.index(anchor)
            text = text[:idx] + "\n\n" + block + "\n" + text[idx:]
        else:
            text = text.rstrip() + "\n\n---\n\n" + block + "\n"

    readme.write_text(text)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark pynear binary indices on SIFT1M.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="directory for downloaded/cached dataset files (default: ./data)",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="use SIFTsmall (10 K vectors) instead of SIFT1M for a quick test",
    )
    parser.add_argument(
        "--n-gt-queries", type=int, default=500,
        help="number of queries to use for recall evaluation (default: 500)",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="number of nearest neighbours to retrieve (default: 10)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    k = args.k
    n_gt = args.n_gt_queries

    # ── Banner ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  pynear · Approximate Binary Index Benchmark on SIFT1M")
    print("=" * 72)

    # ─── 1. Download / load dataset ────────────────────────────────────────────
    dataset_name = "SIFTsmall (10 K vectors)" if args.small else "SIFT1M (1 M vectors)"
    print(f"\n[1/5] Dataset: {dataset_name}")
    folder = fetch_sift(data_dir, args.small)

    prefix = "siftsmall" if args.small else "sift"
    db_raw = read_fvecs(folder / f"{prefix}_base.fvecs")
    q_raw = read_fvecs(folder / f"{prefix}_query.fvecs")

    N, D = db_raw.shape
    Q_total = len(q_raw)
    print(f"  Base   : {N:>10,} × {D}-dim float32")
    print(f"  Queries: {Q_total:>10,} × {D}-dim float32")

    # ─── 2. Binarise ───────────────────────────────────────────────────────────
    print(f"\n[2/5] Binarising {D}-dim float vectors → {D}-bit binary descriptors …")
    db_bin, center = sign_binarise(db_raw)
    q_bin, _ = sign_binarise(q_raw, center=center)

    nbytes = db_bin.shape[1]
    nbits = nbytes * 8
    print(f"  Descriptor width : {nbytes} bytes ({nbits} bits)")
    print(f"  Base binary      : {db_bin.shape}  dtype={db_bin.dtype}")
    print(f"  Query binary     : {q_bin.shape}  dtype={q_bin.dtype}")

    # ─── 3. Ground truth ───────────────────────────────────────────────────────
    n_gt = min(n_gt, Q_total)
    gt_queries = q_bin[:n_gt]
    cache_path = data_dir / f"{prefix}_hamming_gt_{n_gt}q_k{k}.npy"

    print(
        f"\n[3/5] Exact Hamming ground truth  "
        f"({n_gt} queries, k={k}, N={N:,}) …"
    )
    gt = compute_ground_truth(gt_queries, db_bin, k=k, cache_path=cache_path)

    # ─── 4. Build indices & benchmark ─────────────────────────────────────────
    print(f"\n[4/5] Building indices and measuring recall / throughput …")

    headers = [
        "Index", "Configuration",
        "Build (s)", "ms / query", "QPS",
        f"Recall@{k}",
    ]
    rows: list[list[str]] = []

    # ── Brute-force numpy baseline (exact) ────────────────────────────────────
    print("\n  ▶ Brute-force baseline (numpy exact Hamming, no index)")
    bf_n = min(100, n_gt)
    bf_results: list[list[int]] = []
    t0 = time.perf_counter()
    for i in range(bf_n):
        dists = _hamming_to_all(gt_queries[i], db_bin)
        idx = np.argpartition(dists, k)[:k]
        bf_results.append(list(idx[np.argsort(dists[idx])]))
    bf_elapsed = time.perf_counter() - t0
    bf_ms = 1000.0 * bf_elapsed / bf_n
    bf_qps = bf_n / bf_elapsed
    bf_recall = recall_at_k(bf_results, gt[:bf_n], k)
    rows.append([
        "Brute-force (numpy)", f"N={N:,}",
        "—", f"{bf_ms:.1f}", f"{bf_qps:.0f}", f"{bf_recall:.3f}",
    ])
    print(
        f"    ms/query={bf_ms:.1f}   QPS={bf_qps:.0f}"
        f"   Recall@{k}={bf_recall:.3f}"
    )

    # ── IVFFlatBinaryIndex ─────────────────────────────────────────────────────
    nlist = max(64, min(512, N // 2000))
    nprobe_values = sorted({
        max(1, nlist // 16),
        max(1, nlist // 8),
        max(1, nlist // 4),
        max(1, nlist // 2),
        nlist,
    })

    print(f"\n  ▶ IVFFlatBinaryIndex  (nlist={nlist})")
    t0 = time.perf_counter()
    ivf = IVFFlatBinaryIndex(nlist=nlist, nprobe=1)
    ivf.set(db_bin)
    ivf_build = time.perf_counter() - t0
    print(f"    Build: {ivf_build:.2f}s")

    for nprobe in nprobe_values:
        ivf.set_nprobe(nprobe)
        results, ms, qps = benchmark_query(ivf, gt_queries, k)
        rec = recall_at_k(results, gt, k)
        rows.append([
            "IVFFlatBinaryIndex", f"nlist={nlist}, nprobe={nprobe}",
            f"{ivf_build:.2f}", f"{ms:.2f}", f"{qps:.0f}", f"{rec:.3f}",
        ])
        print(
            f"    nprobe={nprobe:4d}   ms/query={ms:.2f}"
            f"   QPS={qps:.0f}   Recall@{k}={rec:.3f}"
        )

    # ── MIHBinaryIndex ─────────────────────────────────────────────────────────
    # Find valid m values: nbytes % m == 0  AND  nbytes // m <= 8
    valid_ms = [
        m for m in [2, 4, 8, 16, 32]
        if nbytes % m == 0 and nbytes // m <= 8
    ]

    if valid_ms:
        # Choose m so that each sub-table has ~10–100 vectors per bucket on
        # average.  With N vectors and sub_nbytes-byte sub-strings the average
        # bucket size is  N / 2^(sub_nbytes × 8).  Targeting bucket_size ≈ 10:
        #   sub_nbytes ≈ floor( log2(N/10) / 8 )
        # This selects 2-byte sub-strings (65 536 buckets, ~15 vecs/bucket)
        # for N=1M, matching the enumeration-cost sweet-spot for MIH.
        import math
        target_sub = max(1, min(8, int(math.log2(max(N, 10) / 10) / 8)))
        target_m = max(1, nbytes // target_sub)
        m_best = min(valid_ms, key=lambda m: abs(m - target_m))

        radius_values = [4, 8, 12, 16, 24, 32, 48]

        print(f"\n  ▶ MIHBinaryIndex  (m={m_best}, sub-width={nbytes // m_best} bytes)")
        t0 = time.perf_counter()
        mih = MIHBinaryIndex(m=m_best)
        mih.set(db_bin)
        mih_build = time.perf_counter() - t0
        print(f"    Build: {mih_build:.2f}s")

        for radius in radius_values:
            results, ms, qps = benchmark_query(
                mih, gt_queries, k, extra_kwargs={"radius": radius}
            )
            rec = recall_at_k(results, gt, k)
            rows.append([
                "MIHBinaryIndex", f"m={m_best}, radius={radius}",
                f"{mih_build:.2f}", f"{ms:.2f}", f"{qps:.0f}", f"{rec:.3f}",
            ])
            print(
                f"    radius={radius:3d}   ms/query={ms:.2f}"
                f"   QPS={qps:.0f}   Recall@{k}={rec:.3f}"
            )
    else:
        print(
            f"\n  [MIHBinaryIndex] Skipped — "
            f"no valid m for nbytes={nbytes} "
            f"(need nbytes % m == 0 and nbytes // m ≤ 8)"
        )

    # ─── 5. Print results table & save outputs ────────────────────────────────
    print(f"\n[5/5] Results\n")
    print_table(headers, rows)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_path = results_dir / "binary_benchmark_qps.png"
    plot_results(rows, k, plot_path, dataset_name)
    print(f"\nQPS plot saved      → {plot_path}")

    # ── Markdown section ───────────────────────────────────────────────────────
    md_table = to_markdown_table(headers, rows)

    best_ivf = next(
        (r for r in rows if r[0] == "IVFFlatBinaryIndex" and r[5] == "1.000"), None
    )
    best_mih = min(
        (r for r in rows if r[0] == "MIHBinaryIndex"),
        key=lambda r: float(r[3]),
        default=None,
    )
    speedup_ivf = (
        f" — **{float(rows[0][3]) / float(best_ivf[3]):.0f}× faster than brute-force**"
        if best_ivf else ""
    )

    # README uses a path relative to the repo root; results/ sits at the root.
    plot_md_path = "results/binary_benchmark_qps.png"

    md_section_body = f"""\
## Real-World Benchmark — SIFT1M Binary

Performance of pynear's approximate Hamming-distance indices on the
[INRIA TEXMEX SIFT1M](http://corpus-texmex.irisa.fr/) dataset:
{N:,} × {D}-dim float SIFT descriptors sign-quantised to **{nbits}-bit binary**
({nbytes} bytes/descriptor).  Ground truth computed by exact brute-force Hamming k-NN
over {n_gt} queries, k={k}.  Machine: {_machine_info()}.

![QPS vs Recall@{k}]({plot_md_path})

{md_table}

**Key takeaways:**
- `IVFFlatBinaryIndex` (nprobe={best_ivf[1].split("nprobe=")[1] if best_ivf else "—"}) achieves **100% Recall@{k} at {best_ivf[4] if best_ivf else "—"} QPS{speedup_ivf}**.
- `MIHBinaryIndex` (radius={best_mih[1].split("radius=")[1] if best_mih else "—"}) is the fastest single configuration at **{best_mih[4] if best_mih else "—"} QPS** with {best_mih[5] if best_mih else "—"} recall.
- MIH excels on wider descriptors (512-bit / 64 bytes) where sub-table sparsity is higher.

> **Reproduce:** `python demo_binary.py` · add `--small` for a 10 K quick test · `--n-gt-queries N` to adjust evaluation size.
"""

    (results_dir / "binary_benchmark.md").write_text(md_section_body)
    print(f"Markdown table saved → {results_dir / 'binary_benchmark.md'}")

    # ── Update README.md ───────────────────────────────────────────────────────
    readme = Path("README.md")
    if readme.exists():
        update_readme(readme, md_section_body)
        print(f"README.md updated with benchmark results.")

    print()


def _machine_info() -> str:
    """Return a brief CPU/platform string for the results header."""
    import platform
    try:
        import subprocess
        cpu = subprocess.check_output(
            ["sh", "-c",
             "cat /proc/cpuinfo 2>/dev/null | grep 'model name' | head -1 | cut -d: -f2"],
            text=True,
        ).strip()
        if cpu:
            return cpu
    except Exception:
        pass
    return platform.processor() or platform.machine()


if __name__ == "__main__":
    main()
