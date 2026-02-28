from __future__ import annotations

import numpy as np

from ml_model_lab.models.base_model import BaseModel, ModelCapabilities


class SklearnModelAdapter(BaseModel):
    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y)

    def step(self, X, y) -> None:
        if hasattr(self.estimator, "partial_fit"):
            classes = np.unique(y)
            self.estimator.partial_fit(X, y, classes=classes)
            return
        raise NotImplementedError("This estimator does not support step training")

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        if hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(X)
            if scores.ndim > 1:
                scores = scores[:, 0]
            probs = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - probs, probs])
        preds = self.predict(X)
        return np.column_stack([1.0 - preds, preds])

    def capabilities(self) -> ModelCapabilities:
        supports_step = hasattr(self.estimator, "partial_fit")
        supports_proba = hasattr(self.estimator, "predict_proba") or hasattr(
            self.estimator, "decision_function"
        )
        return ModelCapabilities(supports_step=supports_step, supports_proba=supports_proba)
