from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from ml_model_lab.models.base_model import BaseModel


@dataclass
class HeatmapData:
    xx: np.ndarray
    yy: np.ndarray
    zz: np.ndarray


@dataclass
class TrainResult:
    metrics: Dict[str, Optional[float]]
    heatmap: Optional[HeatmapData]


class Trainer:
    def __init__(self, model: BaseModel) -> None:
        self.model = model

    def train_full(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: Optional[int] = None,
    ) -> TrainResult:
        if epochs is not None:
            self.model.fit(X_train, y_train, epochs=epochs)
        else:
            self.model.fit(X_train, y_train)
        return self.evaluate(X_train, y_train, X_test, y_test)

    def train_step(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> TrainResult:
        self.model.step(X_train, y_train)
        return self.evaluate(X_train, y_train, X_test, y_test)

    def evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> TrainResult:
        y_pred = self.model.predict(X_test)
        y_scores = None
        try:
            proba = self.model.predict_proba(X_test)
            if proba is not None and proba.shape[1] >= 2:
                y_scores = proba[:, 1]
        except Exception:
            y_scores = None

        metrics = self._compute_metrics(y_test, y_pred, y_scores)
        if hasattr(self.model, "last_loss") and getattr(self.model, "last_loss") is not None:
            metrics["loss"] = float(getattr(self.model, "last_loss"))

        X_all = np.vstack([X_train, X_test])
        heatmap = self._compute_heatmap(X_all)
        return TrainResult(metrics=metrics, heatmap=heatmap)

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: Optional[np.ndarray]
    ) -> Dict[str, Optional[float]]:
        metrics: Dict[str, Optional[float]] = {}
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        roc_auc = None
        try:
            if y_scores is not None and len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            roc_auc = None
        metrics["roc_auc"] = roc_auc
        return metrics

    def _compute_heatmap(self, X: np.ndarray, grid_size: int = 200) -> HeatmapData:
        x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
        y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_size),
            np.linspace(y_min, y_max, grid_size),
        )
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        try:
            proba = self.model.predict_proba(grid)
            zz = proba[:, 1]
        except Exception:
            zz = self.model.predict(grid)
        zz = zz.reshape(xx.shape)
        return HeatmapData(xx=xx, yy=yy, zz=zz)
