from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from ml_model_lab.models.base_model import BaseModel
from ml_model_lab.models.sklearn.sklearn_base import SklearnModelAdapter


@dataclass
class ModelSpec:
    name: str
    builder: Callable[[Dict[str, Any]], BaseModel]
    defaults: Dict[str, Any]


def _build_logreg(params: Dict[str, Any]) -> BaseModel:
    C = float(params.get("C", 1.0))
    max_iter = int(params.get("max_iter", 200))
    estimator = LogisticRegression(C=C, max_iter=max_iter)
    return SklearnModelAdapter(estimator)


def _build_svm(params: Dict[str, Any]) -> BaseModel:
    C = float(params.get("C", 1.0))
    gamma = params.get("gamma", "scale")
    estimator = SVC(C=C, gamma=gamma, probability=True)
    return SklearnModelAdapter(estimator)


def _build_rf(params: Dict[str, Any]) -> BaseModel:
    n_estimators = int(params.get("n_estimators", 200))
    max_depth = params.get("max_depth")
    estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    return SklearnModelAdapter(estimator)


def _build_knn(params: Dict[str, Any]) -> BaseModel:
    n_neighbors = int(params.get("n_neighbors", 5))
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    return SklearnModelAdapter(estimator)


def _build_mlp(params: Dict[str, Any]) -> BaseModel:
    hidden_layers = params.get("hidden_layers", (16, 16))
    alpha = float(params.get("alpha", 0.0001))
    max_iter = int(params.get("max_iter", 300))
    estimator = MLPClassifier(hidden_layer_sizes=hidden_layers, alpha=alpha, max_iter=max_iter)
    return SklearnModelAdapter(estimator)


def _build_keras(params: Dict[str, Any]) -> BaseModel:
    from ml_model_lab.models.keras.keras_mlp import KerasMLPAdapter

    hidden_layers = params.get("hidden_layers", (16, 16))
    activation = params.get("activation", "tanh")
    lr = float(params.get("lr", 0.01))
    batch_size = int(params.get("batch_size", 32))
    return KerasMLPAdapter(hidden_layers=hidden_layers, activation=activation, lr=lr, batch_size=batch_size)


def _keras_available() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


def get_model_specs() -> List[ModelSpec]:
    specs = [
        ModelSpec(
            name="Logistic Regression",
            builder=_build_logreg,
            defaults={"C": 1.0, "max_iter": 200},
        ),
        ModelSpec(
            name="SVM (RBF)",
            builder=_build_svm,
            defaults={"C": 1.0, "gamma": "scale"},
        ),
        ModelSpec(
            name="Random Forest",
            builder=_build_rf,
            defaults={"n_estimators": 200, "max_depth": None},
        ),
        ModelSpec(
            name="KNN",
            builder=_build_knn,
            defaults={"n_neighbors": 7},
        ),
        ModelSpec(
            name="MLPClassifier",
            builder=_build_mlp,
            defaults={"hidden_layers": (16, 16), "alpha": 0.0001, "max_iter": 300},
        ),
    ]
    if _keras_available():
        specs.append(
            ModelSpec(
                name="Keras MLP",
                builder=_build_keras,
                defaults={
                    "hidden_layers": (16, 16),
                    "activation": "tanh",
                    "lr": 0.01,
                    "batch_size": 32,
                    "epochs": 50,
                },
            )
        )
    return specs


def create_model(name: str, params: Dict[str, Any]) -> BaseModel:
    for spec in get_model_specs():
        if spec.name == name:
            return spec.builder(params)
    raise ValueError(f"Unknown model: {name}")
