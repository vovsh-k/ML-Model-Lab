from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ml_model_lab.training.trainer import HeatmapData


class DatasetView(FigureCanvas):
    def __init__(self) -> None:
        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.ax.set_aspect("equal", "box")

    def plot(self, X: np.ndarray, y: np.ndarray, heatmap: Optional[HeatmapData] = None) -> None:
        self.ax.clear()
        if heatmap is not None:
            self.ax.contourf(
                heatmap.xx,
                heatmap.yy,
                heatmap.zz,
                levels=50,
                cmap="RdBu",
                alpha=0.6,
            )
        self.ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap="bwr",
            s=22,
            edgecolor="k",
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
        self.fig.tight_layout()
        self.draw_idle()
