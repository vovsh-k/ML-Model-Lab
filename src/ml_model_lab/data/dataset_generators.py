from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


DatasetFn = Callable[[int, float, int], Tuple[np.ndarray, np.ndarray]]


def _shuffle(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_circle(n: int, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    angles1 = rng.uniform(0, 2 * np.pi, n1)
    angles2 = rng.uniform(0, 2 * np.pi, n2)
    r1 = 0.5 + rng.normal(0, noise, n1)
    r2 = 1.0 + rng.normal(0, noise, n2)
    X1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    X2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    X = np.vstack([X1, X2])
    X += rng.normal(0, noise, X.shape)
    y = np.array([0] * n1 + [1] * n2)
    return _shuffle(X, y, rng)


def make_xor(n: int, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X += rng.normal(0, noise, X.shape)
    return _shuffle(X, y, rng)


def make_gaussian(n: int, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    X1 = rng.normal(loc=[-1.0, -1.0], scale=0.35 + noise, size=(n1, 2))
    X2 = rng.normal(loc=[1.0, 1.0], scale=0.35 + noise, size=(n2, 2))
    X = np.vstack([X1, X2])
    y = np.array([0] * n1 + [1] * n2)
    return _shuffle(X, y, rng)


def make_linear(n: int, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X += rng.normal(0, noise, X.shape)
    return _shuffle(X, y, rng)


def make_spiral(n: int, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    theta = np.sqrt(rng.uniform(0.0, 1.0, n1)) * 4 * np.pi
    r = theta
    X1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    X2 = np.column_stack([-r * np.cos(theta), -r * np.sin(theta)])
    X = np.vstack([X1, X2])
    X = X / (4 * np.pi)
    X += rng.normal(0, noise, X.shape)
    y = np.array([0] * n1 + [1] * n2)
    return _shuffle(X, y, rng)


def make_checkerboard(n: int, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    grid_x = np.floor((X[:, 0] + 1.0) / 0.5).astype(int)
    grid_y = np.floor((X[:, 1] + 1.0) / 0.5).astype(int)
    y = ((grid_x + grid_y) % 2).astype(int)
    X += rng.normal(0, noise, X.shape)
    return _shuffle(X, y, rng)


DATASET_GENERATORS: Dict[str, DatasetFn] = {
    "Circle": make_circle,
    "XOR": make_xor,
    "Gaussian": make_gaussian,
    "Linear": make_linear,
    "Spiral": make_spiral,
    "Checkerboard": make_checkerboard,
}


def list_datasets() -> Tuple[str, ...]:
    return tuple(DATASET_GENERATORS.keys())


def generate_dataset(name: str, n: int, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if name not in DATASET_GENERATORS:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_GENERATORS[name](n, noise, seed)
