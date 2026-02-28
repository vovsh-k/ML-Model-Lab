import numpy as np

from ml_model_lab.data.dataset_generators import generate_dataset, list_datasets


def test_dataset_shapes():
    for name in list_datasets():
        X, y = generate_dataset(name, 200, 0.1, 42)
        assert X.shape == (200, 2)
        assert y.shape == (200,)
        assert set(np.unique(y)).issubset({0, 1})
