from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

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
    penalty = str(params.get("penalty", "l2"))
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    estimator = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver)
    return SklearnModelAdapter(estimator)


def _build_svm(params: Dict[str, Any]) -> BaseModel:
    C = float(params.get("C", 1.0))
    gamma = params.get("gamma", "scale")
    kernel = str(params.get("kernel", "rbf"))
    degree = int(params.get("degree", 3))
    estimator = SVC(C=C, gamma=gamma, kernel=kernel, degree=degree, probability=True)
    return SklearnModelAdapter(estimator)


def _build_rf(params: Dict[str, Any]) -> BaseModel:
    n_estimators = int(params.get("n_estimators", 200))
    max_depth = params.get("max_depth")
    min_samples_split = int(params.get("min_samples_split", 2))
    criterion = str(params.get("criterion", "gini"))
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=42,
    )
    return SklearnModelAdapter(estimator)


def _build_knn(params: Dict[str, Any]) -> BaseModel:
    n_neighbors = int(params.get("n_neighbors", 5))
    weights = str(params.get("weights", "uniform"))
    metric = str(params.get("metric", "minkowski"))
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
    return SklearnModelAdapter(estimator)


def _build_mlp(params: Dict[str, Any]) -> BaseModel:
    hidden_layers = params.get("hidden_layers", (16, 16))
    alpha = float(params.get("alpha", 0.0001))
    max_iter = int(params.get("max_iter", 300))
    activation = str(params.get("activation", "relu"))
    lr = float(params.get("lr", 0.001))
    estimator = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        alpha=alpha,
        learning_rate_init=lr,
        max_iter=max_iter,
        random_state=42,
    )
    return SklearnModelAdapter(estimator)


def _build_keras(params: Dict[str, Any]) -> BaseModel:
    from ml_model_lab.models.keras.keras_mlp import KerasMLPAdapter

    hidden_layers = params.get("hidden_layers", (16, 16))
    input_dim = int(params.get("input_dim", 2))
    activation = params.get("activation", "tanh")
    lr = float(params.get("lr", 0.01))
    l2 = float(params.get("l2", 0.0))
    batch_size = int(params.get("batch_size", 32))
    return KerasMLPAdapter(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        lr=lr,
        l2=l2,
        batch_size=batch_size,
    )


def get_model_specs() -> List[ModelSpec]:
    specs = [
        ModelSpec(
            name="Logistic Regression",
            builder=_build_logreg,
            defaults={"C": 1.0, "penalty": "l2", "max_iter": 200},
        ),
        ModelSpec(
            name="SVM (RBF)",
            builder=_build_svm,
            defaults={"C": 1.0, "kernel": "rbf", "gamma": "scale", "degree": 3},
        ),
        ModelSpec(
            name="Random Forest",
            builder=_build_rf,
            defaults={
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "criterion": "gini",
            },
        ),
        ModelSpec(
            name="KNN",
            builder=_build_knn,
            defaults={"n_neighbors": 7, "weights": "uniform", "metric": "minkowski"},
        ),
        ModelSpec(
            name="MLPClassifier",
            builder=_build_mlp,
            defaults={
                "hidden_layers": (16, 16),
                "activation": "relu",
                "alpha": 0.0001,
                "lr": 0.001,
                "max_iter": 300,
            },
        ),
        ModelSpec(
            name="Keras MLP",
            builder=_build_keras,
            defaults={
                "hidden_layers": (4, 2),
                "activation": "tanh",
                "lr": 0.03,
                "l2": 0.0,
                "batch_size": 10,
                "epochs": 50,
            },
        ),
    ]
    return specs


def create_model(name: str, params: Dict[str, Any]) -> BaseModel:
    for spec in get_model_specs():
        if spec.name == name:
            return spec.builder(params)
    raise ValueError(f"Unknown model: {name}")
