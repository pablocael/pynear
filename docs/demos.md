# Interactive Demos

PyNear ships two Qt/QML desktop demos that let you explore nearest-neighbour
search visually and in real time.  Both are in the `demo/` folder and require
only PySide6 on top of pynear itself.

```console
pip install pynear PySide6
```

---

## KNN Explorer

```console
python demo/point_cloud.py
```

<video src="video/demo1.mp4" controls muted loop width="100%"></video>

### What it shows

A canvas filled with randomly generated 2-D points.  Move the mouse over the
canvas and the **k nearest neighbours** of the cursor are highlighted
instantly, with lines drawn from the cursor to each neighbour.  Neighbours are
colour-coded by rank: the closest is bright red-orange, the farthest is a dark
warm tone.  The sidebar shows the index build time and the per-query search
latency in milliseconds.

### Controls

| Control | Action |
|---|---|
| **Hover** | Run a KNN search at the cursor position and highlight results |
| **Left-drag** | Pan the view |
| **Scroll wheel** | Zoom in / out, anchored to the cursor |
| **Reset view** button | Snap back to the full 1× view |
| **Regenerate** button | Generate a new random point cloud and rebuild the index |

### Configurable parameters

| Parameter | Range | Description |
|---|---|---|
| **Points** | 1 K – 1 M | Number of data points (exponential slider: 1K, 5K, 10K, 50K, 100K, 500K, 1M) |
| **k neighbors** | 1 – 50 | How many nearest neighbours to retrieve per query |
| **Point size** | 1 – 8 px | Rendered diameter of each point dot |

### What to look for

- At **low point counts** (1 K – 10 K) with a large point size you can clearly
  see individual points and appreciate the geometric meaning of "nearest
  neighbour".
- At **high point counts** (500 K – 1 M) the search latency stays in the
  single-digit millisecond range even though the canvas is dense, demonstrating
  the sub-linear scaling of the VPTree index.
- **Zooming in** reveals structure that is invisible at full scale: you can
  watch pynear find tight local clusters in real time.
- Increasing **k** shows how the neighbourhood expands outward, illustrating
  the difference between a tight local cluster and a sparse region.

---

## Voronoi Diagram

```console
python demo/voronoi.py
```

<video src="video/demo2.mp4" controls muted loop width="100%"></video>

### What it shows

An interactive [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram)
built entirely from pynear's 1-NN search.  Every pixel in the canvas is
coloured according to which seed point is closest to it — which is exactly
what a 1-nearest-neighbour query answers.  The diagram updates live as you
drag seed points around the canvas.

Under the hood, each frame issues a single batch `search1NN` call with all
canvas pixels (~480 K queries at 800 × 600) against the current set of seeds.
A typical frame with 20 seeds completes in around 30 ms on a modern desktop.

### Controls

| Control | Action |
|---|---|
| **Left-click** on empty area | Add a new seed point at that location |
| **Left-drag** a seed | Move the seed; Voronoi updates live |
| **Right-click** a seed | Remove that seed |
| **Randomize** button | Replace all seeds with N randomly placed ones |
| **Clear** button | Remove all seeds |

### Configurable parameters

| Parameter | Range | Description |
|---|---|---|
| **Seed count** | 2 – 64 | Number of seeds used by the Randomize button |

### What to look for

- **Dragging a seed** shows how the Voronoi cells grow and shrink in real time.
  Notice how moving one seed only affects its own cell and its immediate
  neighbours — cells that don't share a border are completely unaffected.
- Placing seeds in a **regular grid** produces uniform square-ish cells;
  placing them in clusters produces very irregular cells, with large empty
  regions between clusters.
- With many seeds (40+) the cells become small and the diagram starts to
  resemble a texture — a good intuition for how dense KNN graphs partition
  space at scale.
- The **latency readout** in the sidebar demonstrates that pynear's VPTree
  handles hundreds of thousands of simultaneous queries efficiently, even for
  this trivially small dataset (the seeds), because the bottleneck is the
  number of *queries*, not the index size.
