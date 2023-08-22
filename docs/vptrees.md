## How Vantage Point Trees work

VP-Trees are binary trees that successively divide spaces in order to perform different types of tasks, such as Nearest Neighbor Search. It differs from kd-Trees in the sense that they always partition the whole space, instead of individual dimensional axes, using a specific metric function and a selected "Vantage Point" that will be used as reference to allow splitting the dataset. For more details on how it works please access the following references:

- https://en.wikipedia.org/wiki/Vantage-point_tree
- https://fribbels.github.io/vptree/writeup
- [Probabilistic analysis of vantage point trees](https://www.vmsta.org/journal/VMSTA/article/219/file/pdf)

### Theoretical advantage of Vantage Points Trees compared to Kd-Trees

- Intrinsic Dimensionality: VP-trees perform well in data with high intrinsic dimensionality, where the effective dimensionality is high. In contrast, kd-trees are efficient in low-dimensional spaces but their performance degrades quickly as the number of dimensions increases. This is often referred to as the "curse of dimensionality".

- No Assumption about Axis-Alignment: Unlike kd-trees, which make a specific assumption about axis-alignment, VP-trees do not make such assumptions. This makes VP-trees potentially more robust to different kinds of data, especially when there is no natural way to align the axes with respect to the data.

- Metric Spaces: VP-trees can handle general metric spaces (any space where a distance function is defined that satisfies the triangle inequality), not just Euclidean spaces. This makes them more adaptable to different problem settings where the distance metric might not be the standard Euclidean distance.

- Handling Categorical Data: VP-trees, due to their use of arbitrary distance functions, can handle categorical data more naturally than kd-trees. While there are workarounds to use kd-trees with categorical data, they often require substantial tweaking and may not perform optimally. For categorical data, another reference structure is the [BK-Tree](https://en.wikipedia.org/wiki/BK-tree), which is very efficient when comes to low dimensional data.

- Balanced Tree Structure: VP-trees inherently try to create a balanced tree structure, which is beneficial for efficient searching. kd-trees can become unbalanced in certain situations, particularly with non-uniform data, leading to inefficient search operations.

Different mplementation approaches such as using [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) and [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) instructions affect final performance in high dimensions. For instance, [Faiss Library](https://github.com/facebookresearch/faiss) performs very efficiently in very high dimensions, where searches become near linear and exhaustive, due to highly optimized code.
 
PyNear adopts AVX2 instructions to optimize some of the indices such as VPTreeL2Index and VPTreeBinaryIndex.

