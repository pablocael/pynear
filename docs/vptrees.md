## How Vantage Point Trees Work

VP-Trees are binary trees that successively partition a metric space to enable efficient nearest-neighbor search.
They differ from kd-trees in that they always partition the **whole space** using a metric distance function and a selected "vantage point" as reference, rather than splitting individual dimensional axes.

For more details:
- [Wikipedia: Vantage-point tree](https://en.wikipedia.org/wiki/Vantage-point_tree)
- [VP-Tree writeup by fribbels](https://fribbels.github.io/vptree/writeup)
- [Probabilistic analysis of vantage point trees (VMSTA journal)](https://www.vmsta.org/journal/VMSTA/article/219/file/pdf)

### Theoretical advantages over kd-Trees

- **Intrinsic dimensionality**: VP-trees perform well when the effective dimensionality of the data is high. kd-trees degrade quickly as the number of dimensions increases — the so-called "curse of dimensionality".

- **No axis-alignment assumption**: kd-trees split along coordinate axes, which can be suboptimal when there is no natural alignment between the axes and the data distribution. VP-trees make no such assumption.

- **General metric spaces**: VP-trees work with any distance function that satisfies the triangle inequality, not just Euclidean distance. This makes them adaptable to Hamming distance, Manhattan distance, or custom application-specific metrics.

- **Categorical and binary data**: Because VP-trees use arbitrary distance functions, they handle binary and categorical data naturally. For binary data with low intrinsic dimensionality, a [BK-Tree](https://en.wikipedia.org/wiki/BK-tree) is another efficient alternative for range queries.

- **Balanced structure**: VP-trees partition the dataset at each level using the median distance, which tends to produce a balanced tree and consistent search performance even with non-uniform data.

### Implementation and performance

Different implementation choices significantly affect performance at higher dimensionalities.
PyNear uses SIMD intrinsics (AVX2 on x86-64) to accelerate the hot distance computation paths for L2, L1, Chebyshev, and Hamming distances.
On arm64 (Apple Silicon and similar), portable scalar fallbacks are used automatically — no source changes required.

For very high-dimensional spaces where search becomes nearly exhaustive, libraries like [Faiss](https://github.com/facebookresearch/faiss) with highly optimized BLAS kernels may outperform tree-based approaches.
PyNear targets the exact-search regime where tree pruning still provides a significant speedup.
