from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelCapabilities:
    supports_step: bool = False
    supports_proba: bool = True


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def step(self, X, y) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        raise NotImplementedError
