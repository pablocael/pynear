"""
PyNear interactive Voronoi diagram demo.

Each pixel is colored by its nearest seed point (1-NN over the full pixel grid).
Click to add seeds, drag to move them, right-click to remove.

Usage:
    pip install PySide6
    python demo/voronoi.py
"""

import sys
import os
import time
import colorsys

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterType
from PySide6.QtCore import Qt, Signal, Slot, Property, QPointF
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QBrush, QFont
from PySide6.QtQuick import QQuickPaintedItem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pynear

MAX_SEEDS = 64
SEED_HIT_RADIUS = 10    # pixel radius for drag/remove hit-testing


def _make_palette(n: int) -> np.ndarray:
    """n visually distinct RGBA colors using golden-angle hue spacing."""
    out = []
    for i in range(n):
        h = (i * 137.508) % 360
        r, g, b = colorsys.hsv_to_rgb(h / 360, 0.50, 0.88)
        out.append([int(r * 255), int(g * 255), int(b * 255), 255])
    return np.array(out, dtype=np.uint8)


_PALETTE = _make_palette(MAX_SEEDS)


class VoronoiView(QQuickPaintedItem):
    statusChanged = Signal()
    seedCountChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._seeds: list[list[float]] = []   # [[x, y], …] in data space [0, 1]²
        self._drag_idx = -1
        self._voronoi_img: QImage | None = None
        self._grid: np.ndarray | None = None   # cached pixel grid
        self._grid_size = (0, 0)
        self._status = "Click anywhere to add seed points"
        self._compute_ms = 0.0

        self.widthChanged.connect(self._on_resize)
        self.heightChanged.connect(self._on_resize)

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _px_to_data(self, px: float, py: float):
        return px / self.width(), py / self.height()

    def _data_to_px(self, dx: float, dy: float):
        return dx * self.width(), dy * self.height()

    def _hit_seed(self, px: float, py: float) -> int:
        best, best_d2 = -1, SEED_HIT_RADIUS ** 2
        for i, (sx, sy) in enumerate(self._seeds):
            spx, spy = self._data_to_px(sx, sy)
            d2 = (px - spx) ** 2 + (py - spy) ** 2
            if d2 < best_d2:
                best, best_d2 = i, d2
        return best

    def _on_resize(self):
        self._grid = None       # invalidate cached pixel grid
        self._recompute()

    # ── Voronoi computation ───────────────────────────────────────────────────

    def _recompute(self):
        if not self._seeds:
            self._voronoi_img = None
            self.update()
            return

        w = max(1, int(self.width()))
        h = max(1, int(self.height()))

        # Rebuild pixel grid only when canvas size changes
        if self._grid is None or self._grid_size != (w, h):
            xs = np.linspace(0, 1, w, dtype=np.float32)
            ys = np.linspace(0, 1, h, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            self._grid = np.column_stack([gx.ravel(), gy.ravel()])
            self._grid_size = (w, h)

        seeds_arr = np.array(self._seeds, dtype=np.float32)

        t0 = time.perf_counter()
        index = pynear.VPTreeL2Index()
        index.set(seeds_arr)
        nn_indices, _ = index.search1NN(self._grid)
        self._compute_ms = (time.perf_counter() - t0) * 1000

        idx_img = np.array(nn_indices, dtype=np.int32).reshape(h, w)
        colored = _PALETTE[idx_img]    # (h, w, 4) via fancy indexing

        self._voronoi_img = QImage(
            colored.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888
        )
        self._update_status()
        self.update()

    # ── paint ─────────────────────────────────────────────────────────────────

    def paint(self, painter: QPainter):
        w, h = int(self.width()), int(self.height())

        if self._voronoi_img is None:
            painter.fillRect(0, 0, w, h, QColor(18, 18, 30))
            painter.setPen(QColor(70, 70, 100))
            painter.setFont(QFont("sans-serif", 14))
            painter.drawText(
                0, 0, w, h,
                Qt.AlignmentFlag.AlignCenter,
                "Click anywhere to add seed points",
            )
            return

        painter.drawImage(0, 0, self._voronoi_img)

        # Draw seed circles
        for i, (sx, sy) in enumerate(self._seeds):
            px, py = self._data_to_px(sx, sy)
            c = _PALETTE[i % MAX_SEEDS]
            fill = QColor(int(c[0]), int(c[1]), int(c[2]))

            # Shadow for readability
            painter.setPen(QPen(QColor(0, 0, 0, 80), 4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(px, py), SEED_HIT_RADIUS, SEED_HIT_RADIUS)

            # Filled circle
            painter.setPen(QPen(QColor(255, 255, 255, 230), 2))
            painter.setBrush(QBrush(fill))
            painter.drawEllipse(QPointF(px, py), SEED_HIT_RADIUS - 1, SEED_HIT_RADIUS - 1)

            # Index label
            painter.setPen(QPen(QColor(20, 20, 20), 1))
            painter.setFont(QFont("sans-serif", 7, QFont.Weight.Bold))
            painter.drawText(
                int(px) - 9, int(py) - 9, 18, 18,
                Qt.AlignmentFlag.AlignCenter,
                str(i + 1),
            )

    # ── mouse slots (called from QML) ─────────────────────────────────────────

    @Slot(float, float, bool)
    def mousePressed(self, px: float, py: float, right_button: bool):
        hit = self._hit_seed(px, py)

        if right_button:
            if hit >= 0:
                self._seeds.pop(hit)
                self._drag_idx = -1
                self.seedCountChanged.emit()
                self._recompute()
        else:
            if hit >= 0:
                self._drag_idx = hit
            else:
                if len(self._seeds) < MAX_SEEDS:
                    dx, dy = self._px_to_data(px, py)
                    self._seeds.append([float(dx), float(dy)])
                    self._drag_idx = len(self._seeds) - 1
                    self.seedCountChanged.emit()
                    self._recompute()

    @Slot(float, float)
    def mouseDragged(self, px: float, py: float):
        if self._drag_idx < 0:
            return
        dx = max(0.0, min(1.0, px / self.width()))
        dy = max(0.0, min(1.0, py / self.height()))
        self._seeds[self._drag_idx] = [dx, dy]
        self._recompute()

    @Slot()
    def mouseReleased(self):
        self._drag_idx = -1

    @Slot(int)
    def randomize(self, n: int):
        n = max(2, min(n, MAX_SEEDS))
        coords = np.random.rand(n, 2).tolist()
        self._seeds = [[float(x), float(y)] for x, y in coords]
        self._drag_idx = -1
        self.seedCountChanged.emit()
        self._recompute()

    @Slot()
    def clearSeeds(self):
        self._seeds.clear()
        self._drag_idx = -1
        self._voronoi_img = None
        self.seedCountChanged.emit()
        self._status = "Click anywhere to add seed points"
        self.statusChanged.emit()
        self.update()

    def _update_status(self):
        n = len(self._seeds)
        pixels = self._grid_size[0] * self._grid_size[1]
        self._status = (
            f"{n} seed{'s' if n != 1 else ''}  •  "
            f"{pixels:,} pixels  •  "
            f"{self._compute_ms:.1f} ms\n"
            "Click: add  •  drag: move  •  right-click: remove"
        )
        self.statusChanged.emit()

    # ── properties ────────────────────────────────────────────────────────────

    @Property(str, notify=statusChanged)
    def status(self):
        return self._status

    @Property(int, notify=seedCountChanged)
    def seedCount(self):
        return len(self._seeds)


def main():
    app = QApplication(sys.argv)
    app.setOrganizationName("PyNear")
    app.setApplicationName("Voronoi Demo")

    qmlRegisterType(VoronoiView, "PyNearDemo", 1, 0, "VoronoiView")

    engine = QQmlApplicationEngine()
    qml_path = os.path.join(os.path.dirname(__file__), "voronoi.qml")
    engine.load(qml_path)

    if not engine.rootObjects():
        sys.exit(1)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
