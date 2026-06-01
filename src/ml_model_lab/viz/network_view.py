from __future__ import annotations

import os
import tempfile
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "ml_model_lab_mpl"))

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy


class NetworkView(FigureCanvas):
    def __init__(self) -> None:
        self.fig = Figure(figsize=(5.2, 5.2))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setMinimumSize(300, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.fig.patch.set_facecolor("#f8fafc")

    def plot(self, feature_names: Sequence[str], hidden_layers: Iterable[int]) -> None:
        layers = [len(feature_names), *[int(x) for x in hidden_layers], 1]

        self.ax.clear()
        self.ax.set_facecolor("#f8fafc")
        self.ax.axis("off")

        x_positions = list(range(len(layers)))
        layer_nodes = []
        for x, count in zip(x_positions, layers):
            count = max(count, 1)
            if count == 1:
                ys = [0.5]
            else:
                ys = [0.12 + i * (0.76 / (count - 1)) for i in range(count)]
            layer_nodes.append([(x, y) for y in ys])

        for left, right in zip(layer_nodes, layer_nodes[1:]):
            for x1, y1 in left:
                for x2, y2 in right:
                    self.ax.plot([x1, x2], [y1, y2], color="#94a3b8", linewidth=0.8, alpha=0.5, zorder=1)

        for layer_index, nodes in enumerate(layer_nodes):
            color = "#f4a259" if layer_index == 0 else "#3b82f6"
            if layer_index == len(layer_nodes) - 1:
                color = "#1d4ed8"
            for x, y in nodes:
                self.ax.scatter([x], [y], s=360, color=color, edgecolor="#0f172a", linewidth=1.0, zorder=3)
        for idx, name in enumerate(feature_names[: len(layer_nodes[0])]):
            x, y = layer_nodes[0][idx]
            self.ax.text(x - 0.18, y, name, ha="right", va="center", fontsize=8, color="#334155")

        self.ax.set_xlim(-0.7, max(x_positions) + 0.8)
        self.ax.set_ylim(0.0, 1.12)
        self.fig.tight_layout(pad=0.4)
        self.draw_idle()
