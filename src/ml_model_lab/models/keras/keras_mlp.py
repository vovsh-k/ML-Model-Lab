from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from tensorflow import keras

from ml_model_lab.models.base_model import BaseModel, ModelCapabilities


class KerasMLPAdapter(BaseModel):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_layers: Iterable[int] = (16, 16),
        activation: str = "tanh",
        lr: float = 0.01,
        l2: float = 0.0,
        batch_size: int = 32,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_layers = tuple(int(x) for x in hidden_layers)
        self.activation = activation
        self.lr = lr
        self.l2 = l2
        self.batch_size = batch_size
        self.last_loss = None
        self._build_model()

    def _build_model(self) -> None:
        reg = keras.regularizers.l2(self.l2) if self.l2 > 0 else None
        inputs = keras.Input(shape=(self.input_dim,))
        x = inputs
        for units in self.hidden_layers:
            x = keras.layers.Dense(units, activation=self.activation, kernel_regularizer=reg)(x)
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss="binary_crossentropy")
        self.model = model

    def reset(self) -> None:
        self._build_model()

    def fit(self, X, y, epochs: int = 50, batch_size: int | None = None, **kwargs):
        if batch_size is None:
            batch_size = self.batch_size
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        loss = history.history.get("loss")
        if loss:
            self.last_loss = float(loss[-1])

    def step(self, X, y) -> None:
        batch_size = min(self.batch_size, len(X))
        idx = np.random.choice(len(X), size=batch_size, replace=False)
        loss = self.model.train_on_batch(X[idx], y[idx], return_dict=False)
        try:
            self.last_loss = float(loss)
        except Exception:
            self.last_loss = None

    def predict(self, X):
        probs = self.model.predict(X, verbose=0).ravel()
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        probs = self.model.predict(X, verbose=0).ravel()
        return np.column_stack([1.0 - probs, probs])

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(supports_step=True, supports_proba=True)
