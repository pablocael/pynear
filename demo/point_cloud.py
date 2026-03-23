"""
PyNear KNN interactive demo.

Usage:
    pip install PySide6
    python demo/main.py
"""

import sys
import os
import time

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterType
from PySide6.QtCore import Qt, Signal, Slot, Property, QPointF
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QBrush
from PySide6.QtQuick import QQuickPaintedItem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pynear

# When visible point count is below this, use per-point QPainter calls so that
# point size > 1 looks crisp.  Above it we fall back to 1-pixel numpy blits.
_PAINTER_THRESHOLD = 30_000


class PointCloudView(QQuickPaintedItem):
    statusChanged = Signal()
    searchTimeChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points = np.empty((0, 2), dtype=np.float32)
        self._index = None
        self._neighbor_indices: list[int] = []
        self._hover_pos: QPointF | None = None

        # view state
        self._zoom = 1.0        # 1.0 = show full [0,1]x[0,1] data range
        self._pan_x = 0.5       # viewport center in data space
        self._pan_y = 0.5
        self._point_size = 2    # diameter in pixels

        self._base_image: QImage | None = None
        self._status = "Click  Regenerate  to start"

        self.widthChanged.connect(self._on_resize)
        self.heightChanged.connect(self._on_resize)

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _to_pixel(self, data_xy: np.ndarray):
        """Map Nx2 data coords [0,1] → pixel coords for current view."""
        w, h = self.width(), self.height()
        px = (data_xy[:, 0] - self._pan_x) * self._zoom * w + w / 2
        py = (data_xy[:, 1] - self._pan_y) * self._zoom * h + h / 2
        return px, py

    def _to_data(self, pixel_x: float, pixel_y: float):
        """Map a single pixel position → data coords."""
        w, h = self.width(), self.height()
        dx = (pixel_x - w / 2) / (self._zoom * w) + self._pan_x
        dy = (pixel_y - h / 2) / (self._zoom * h) + self._pan_y
        return dx, dy

    # ── resize / invalidate ───────────────────────────────────────────────────

    def _on_resize(self):
        self._base_image = None
        if len(self._points):
            self._render_base_image()
        self.update()

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render_base_image(self):
        w = max(1, int(self.width()))
        h = max(1, int(self.height()))

        buf = np.zeros((h, w, 4), dtype=np.uint8)
        buf[:, :, 3] = 255          # opaque black background

        if not len(self._points):
            self._base_image = QImage(
                buf.data, w, h, w * 4, QImage.Format.Format_RGBA8888
            ).copy()
            return

        px, py = self._to_pixel(self._points)
        mask = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        vx = px[mask].astype(np.int32)
        vy = py[mask].astype(np.int32)

        s = self._point_size
        color = np.array([160, 160, 190, 255], dtype=np.uint8)

        if s <= 1 or len(vx) > _PAINTER_THRESHOLD:
            # Fast path: 1 pixel per point via numpy
            buf[vy, vx] = color
        else:
            # Slower path: s×s square per point (only for few visible points)
            half = s // 2
            for dy in range(-half, s - half):
                for dx in range(-half, s - half):
                    buf[
                        np.clip(vy + dy, 0, h - 1),
                        np.clip(vx + dx, 0, w - 1)
                    ] = color

        self._base_image = QImage(
            buf.data, w, h, w * 4, QImage.Format.Format_RGBA8888
        ).copy()

    # ── paint ─────────────────────────────────────────────────────────────────

    def paint(self, painter: QPainter):
        w, h = self.width(), self.height()

        if self._base_image is not None:
            painter.drawImage(0, 0, self._base_image)
        else:
            painter.fillRect(0, 0, int(w), int(h), QColor(12, 12, 22))
            return

        pts = self._points
        if self._neighbor_indices and len(pts):
            n = len(self._neighbor_indices)
            px, py = self._to_pixel(pts)

            # Lines from cursor to neighbors
            if self._hover_pos is not None:
                line_pen = QPen(QColor(80, 140, 255, 60))
                line_pen.setWidth(1)
                painter.setPen(line_pen)
                for idx in self._neighbor_indices:
                    painter.drawLine(
                        self._hover_pos,
                        QPointF(float(px[idx]), float(py[idx]))
                    )

            # Neighbor dots — closest is brightest / largest
            painter.setPen(Qt.PenStyle.NoPen)
            for rank, idx in enumerate(self._neighbor_indices):
                t = rank / max(n - 1, 1)
                r = int(255 * (1 - t * 0.5))
                g = int(180 * (1 - t))
                b = int(60 * t)
                painter.setBrush(QBrush(QColor(r, g, b, 220)))
                radius = max(3.0, 8.0 * (1 - t * 0.6))
                painter.drawEllipse(QPointF(float(px[idx]), float(py[idx])), radius, radius)

        # Cursor crosshair
        if self._hover_pos is not None:
            painter.setPen(QPen(QColor(0, 210, 255), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self._hover_pos, 5, 5)

    # ── slots called from QML ─────────────────────────────────────────────────

    @Slot(int, int)
    def regenerate(self, n: int, _k: int):
        self._neighbor_indices = []
        self._hover_pos = None
        self._status = f"Building index for {n:,} points…"
        self.statusChanged.emit()
        self.update()
        QApplication.processEvents()

        t0 = time.perf_counter()
        self._points = np.random.rand(n, 2).astype(np.float32)
        self._index = pynear.VPTreeL2Index()
        self._index.set(self._points)
        build_ms = (time.perf_counter() - t0) * 1000

        self._render_base_image()
        self._status = (
            f"{n:,} points  •  index built in {build_ms:.0f} ms\n"
            "Hover to search  •  scroll to zoom  •  drag to pan"
        )
        self.statusChanged.emit()
        self.update()

    @Slot(float, float, int)
    def hoverAt(self, x: float, y: float, k: int):
        if self._index is None:
            return
        qx, qy = self._to_data(x, y)
        query = np.array([[qx, qy]], dtype=np.float32)

        t0 = time.perf_counter()
        indices, _ = self._index.searchKNN(query, k)
        search_ms = (time.perf_counter() - t0) * 1000

        self._neighbor_indices = list(indices[0])
        self._hover_pos = QPointF(x, y)
        self._status = (
            f"{len(self._points):,} points  •  k = {k}\n"
            f"Search: {search_ms:.3f} ms  •  zoom: {self._zoom:.1f}×"
        )
        self.statusChanged.emit()
        self.update()

    @Slot(float, float, float)
    def zoomAt(self, angle_delta: float, px: float, py: float):
        """Zoom in/out keeping the point under the cursor stationary."""
        factor = 1.15 if angle_delta > 0 else 1 / 1.15
        mx, my = self._to_data(px, py)
        self._zoom = max(0.5, min(200.0, self._zoom * factor))
        # Recompute pan so that (mx, my) stays under (px, py)
        w, h = self.width(), self.height()
        self._pan_x = mx - (px - w / 2) / (self._zoom * w)
        self._pan_y = my - (py - h / 2) / (self._zoom * h)
        self._render_base_image()
        self._neighbor_indices = []
        self._hover_pos = None
        self.update()

    @Slot(float, float)
    def panBy(self, dx: float, dy: float):
        """Pan by a screen-space delta (pixels)."""
        w, h = self.width(), self.height()
        self._pan_x -= dx / (self._zoom * w)
        self._pan_y -= dy / (self._zoom * h)
        self._render_base_image()
        self._neighbor_indices = []
        self._hover_pos = None
        self.update()

    @Slot()
    def resetView(self):
        self._zoom = 1.0
        self._pan_x = 0.5
        self._pan_y = 0.5
        self._neighbor_indices = []
        self._hover_pos = None
        self._render_base_image()
        self.update()

    @Slot(int)
    def setPointSize(self, size: int):
        self._point_size = max(1, min(10, size))
        if len(self._points):
            self._render_base_image()
        self.update()

    # ── properties ────────────────────────────────────────────────────────────

    @Property(str, notify=statusChanged)
    def status(self):
        return self._status


def main():
    app = QApplication(sys.argv)
    app.setOrganizationName("PyNear")
    app.setApplicationName("KNN Demo")

    qmlRegisterType(PointCloudView, "PyNearDemo", 1, 0, "PointCloudView")

    engine = QQmlApplicationEngine()
    qml_path = os.path.join(os.path.dirname(__file__), "point_cloud.qml")
    engine.load(qml_path)

    if not engine.rootObjects():
        sys.exit(1)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
