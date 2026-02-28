from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class DatasetConfig:
    name: str = "Circle"
    samples: int = 400
    noise: float = 0.10
    seed: int = 42
    test_size: float = 0.30


@dataclass
class ModelConfig:
    name: str = "Logistic Regression"
    params: Dict[str, object] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    play_interval_ms: int = 200


class AppState:
    def __init__(self) -> None:
        self.dataset = DatasetConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self._listeners: List[Callable[[str], None]] = []

    def add_listener(self, callback: Callable[[str], None]) -> None:
        self._listeners.append(callback)

    def notify(self, topic: str) -> None:
        for cb in list(self._listeners):
            cb(topic)

    def update_dataset(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            if hasattr(self.dataset, key):
                setattr(self.dataset, key, value)
        self.notify("dataset")

    def update_model(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
        self.notify("model")
