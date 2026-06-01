from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "ml_model_lab_mpl"))

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy

from ml_model_lab.training.trainer import HeatmapData


PREDICTION_CMAP = LinearSegmentedColormap.from_list(
    "playground_prediction",
    ["#f4a259", "#fff8e8", "#3b82f6"],
)
POINT_CMAP = ListedColormap(["#f4a259", "#3b82f6"])


class DatasetView(FigureCanvas):
    def __init__(self) -> None:
        self.fig = Figure(figsize=(6.2, 6.2))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax.set_aspect("equal", "box")

    def plot(self, X: np.ndarray, y: np.ndarray, heatmap: Optional[HeatmapData] = None) -> None:
        self.ax.clear()
        if heatmap is not None:
            self.ax.contourf(
                heatmap.xx,
                heatmap.yy,
                heatmap.zz,
                levels=50,
                cmap=PREDICTION_CMAP,
                alpha=0.6,
                vmin=0.0,
                vmax=1.0,
            )
        self.ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap=POINT_CMAP,
            vmin=0,
            vmax=1,
            s=22,
            edgecolor="#172033",
            linewidth=0.4,
        )
        if heatmap is not None:
            self.ax.set_xlim(heatmap.xx.min(), heatmap.xx.max())
            self.ax.set_ylim(heatmap.yy.min(), heatmap.yy.max())
        else:
            pad = 0.2
            self.ax.set_xlim(X[:, 0].min() - pad, X[:, 0].max() + pad)
            self.ax.set_ylim(X[:, 1].min() - pad, X[:, 1].max() + pad)
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.ax.set_title("Dataset")
        self.ax.grid(True, color="#e5e7eb", linewidth=0.6, alpha=0.7)
        self.fig.subplots_adjust(left=0.12, right=0.96, bottom=0.13, top=0.90)
        self.draw_idle()
