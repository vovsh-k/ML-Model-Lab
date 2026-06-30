# ML Model Lab

## Project Overview

ML Model Lab is a desktop application for interactive experimentation with machine learning models on synthetic two-dimensional datasets. The project is inspired by TensorFlow Playground: the user can generate data, select input features, train a neural network step by step, observe the decision boundary, and compare the behavior of classic machine learning models.

The application is designed as a course project for object-oriented programming. It demonstrates encapsulation, composition, inheritance through model adapters, and a factory-style model registry.

## Goals

- Build a clear visual playground for machine learning experiments.
- Compare neural networks and classic models on the same synthetic datasets.
- Provide a GUI that allows users to change model parameters without editing code.
- Visualize decision boundaries, training loss, metrics, and neural-network architecture.
- Keep the codebase modular enough to extend after the course project.

## Technology Stack

- Language: Python.
- GUI: PySide6.
- Visualization: matplotlib embedded into Qt widgets.
- Classic ML models: scikit-learn.
- Neural network: TensorFlow/Keras through `from tensorflow import keras`.
- Testing: pytest.
- Packaging: setuptools through `pyproject.toml`.

## Current Functionality

- Synthetic datasets:
  - Circle.
  - XOR.
  - Gaussian.
  - Linear.
  - Spiral.
  - Checkerboard.
- Dataset controls:
  - sample count.
  - noise.
  - random seed.
  - train/test split.
  - regeneration button.
- Feature controls:
  - `x1`.
  - `x2`.
  - `x1^2`.
  - `x2^2`.
  - `x1*x2`.
  - `sin(x1)`.
  - `sin(x2)`.
- Neural-network workspace:
  - Keras MLP.
  - Step training.
  - Play/Stop training.
  - Full Train button.
  - Reset button.
  - Playground-style default parameters.
  - Add/remove hidden layers.
  - Add/remove neurons in a selected hidden layer.
  - Decision boundary visualization.
  - Training loss curve.
  - Metrics panel.
- Classic-model workspace:
  - Logistic Regression.
  - SVM.
  - Random Forest.
  - KNN.
  - MLPClassifier.
  - Model-specific parameter panels.
  - Automatic parameter search through small GridSearchCV presets.
- Metrics:
  - Accuracy.
  - Precision.
  - Recall.
  - F1.
  - ROC-AUC when available.
  - Loss for Keras MLP.

## Architecture

The project is organized as a Python package under `src/ml_model_lab`.

### `app`

Contains the PySide6 application entry point and the main window.

- `__main__.py`: starts QApplication and opens MainWindow.
- `main_window.py`: builds the GUI, handles user actions, connects controls to model training and visualization.

### `data`

Contains synthetic dataset generation.

- `dataset_generators.py`: functions for creating 2D classification datasets.

### `models`

Contains model abstractions and adapters.

- `base_model.py`: common model interface and capabilities.
- `model_registry.py`: creates selected models from GUI parameters.
- `sklearn/sklearn_base.py`: adapter for scikit-learn estimators.
- `keras/keras_mlp.py`: Keras MLP adapter with step training and loss history.

### `training`

Contains training and evaluation logic.

- `trainer.py`: trains models, computes metrics, and creates decision-boundary heatmap data.

### `viz`

Contains matplotlib-based visualization widgets.

- `dataset_view.py`: scatter plot and decision boundary.
- `network_view.py`: neural-network architecture diagram.
- `loss_plot_view.py`: training-loss curve.

### `core`

Contains application state and event helpers prepared for future expansion.

- `app_state.py`.
- `events.py`.

## Object-Oriented Design

The project uses several object-oriented ideas:

- `BaseModel` defines a common model interface.
- `SklearnModelAdapter` adapts different scikit-learn estimators to the same interface.
- `KerasMLPAdapter` implements the same interface for neural networks.
- `Trainer` works with any model through the `BaseModel` interface.
- `ModelRegistry` centralizes model creation and default parameters.
- Visualization classes encapsulate matplotlib rendering inside Qt widgets.

## Completed Work

- Created the initial package structure and app entry point.
- Added synthetic dataset generation.
- Added model registry for classic models and Keras MLP.
- Implemented train/test split, metrics, and decision-boundary heatmap.
- Built the first PySide6 GUI.
- Split the interface into Neural Network and Classic Models workspaces.
- Moved Data, Features, and Network controls into a layout inspired by TensorFlow Playground.
- Added feature transformations and connected them to training and heatmap prediction.
- Added neural-network architecture editing through buttons.
- Added step/play/stop training flow for Keras MLP.
- Added loss history and training-loss visualization.
- Added automatic parameter search for classic models.
- Improved plotting colors and layout sizing.
- Added basic pytest coverage for dataset generators.

## Running the Project

```bash
cd "/Users/vvshk/ml/ML Model Lab"
source .venv/bin/activate
ml-model-lab
```

If the package is not installed:

```bash
pip install -e .
ml-model-lab
```

Keras support requires TensorFlow:

```bash
pip install -e ".[keras]"
```

## Running Tests

Use the Python interpreter from the active virtual environment:

```bash
python -m pytest -q
```

This is preferred over plain `pytest -q`, because it guarantees that pytest uses the same environment where project dependencies are installed.
