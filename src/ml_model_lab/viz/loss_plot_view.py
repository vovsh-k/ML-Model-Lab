from __future__ import annotations

import os
import tempfile
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "ml_model_lab_mpl"))

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy


class LossPlotView(FigureCanvas):
    def __init__(self) -> None:
        self.fig = Figure(figsize=(6.2, 2.3))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.fig.patch.set_facecolor("#f8fafc")
        self.plot([])

    def plot(self, losses: Sequence[float]) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#f8fafc")
        self.ax.grid(True, color="#e5e7eb", linewidth=0.6, alpha=0.8)
        self.ax.set_title("Training loss", fontsize=10, color="#334155")
        self.ax.set_xlabel("")
        self.ax.set_ylabel("Loss", fontsize=9)

        if losses:
            xs = range(1, len(losses) + 1)
            self.ax.plot(xs, losses, color="#1d4ed8", linewidth=2.0)
            self.ax.scatter([len(losses)], [losses[-1]], color="#f4a259", s=34, zorder=3)
        else:
            self.ax.text(
                0.5,
                0.5,
                "No training history yet",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                color="#64748b",
                fontsize=9,
            )

        self.fig.subplots_adjust(left=0.10, right=0.98, bottom=0.24, top=0.78)
        self.draw_idle()
