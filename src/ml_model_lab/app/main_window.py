from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from sklearn.model_selection import train_test_split

from ml_model_lab.core.app_state import AppState
from ml_model_lab.data.dataset_generators import generate_dataset, list_datasets
from ml_model_lab.models.model_registry import ModelSpec, create_model, get_model_specs
from ml_model_lab.training.trainer import TrainResult, Trainer
from ml_model_lab.viz.dataset_view import DatasetView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ML Model Lab")
        self.state = AppState()

        self.model = None
        self.trainer: Trainer | None = None
        self.model_specs: Dict[str, ModelSpec] = {}

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(self.state.training.play_interval_ms)
        self.play_timer.timeout.connect(self.on_step)

        self._build_ui()
        self._load_models()
        self._update_param_widgets()
        self.generate_and_plot()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)

        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        layout.addLayout(left_panel, 0)
        layout.addLayout(right_panel, 1)

        left_panel.addWidget(self._build_data_group())
        left_panel.addWidget(self._build_model_group())
        left_panel.addWidget(self._build_train_group())
        left_panel.addStretch(1)

        self.dataset_view = DatasetView()
        right_panel.addWidget(self.dataset_view, 1)
        right_panel.addWidget(self._build_metrics_group(), 0)

        self.setCentralWidget(central)

    def _build_data_group(self) -> QGroupBox:
        group = QGroupBox("Data")
        form = QFormLayout(group)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(list_datasets())

        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(50, 5000)
        self.samples_spin.setValue(self.state.dataset.samples)

        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 0.5)
        self.noise_spin.setSingleStep(0.01)
        self.noise_spin.setDecimals(3)
        self.noise_spin.setValue(self.state.dataset.noise)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(self.state.dataset.seed)

        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0.1, 0.5)
        self.test_split_spin.setSingleStep(0.05)
        self.test_split_spin.setDecimals(2)
        self.test_split_spin.setValue(self.state.dataset.test_size)

        self.regen_button = QPushButton("Regenerate")
        self.regen_button.clicked.connect(self.generate_and_plot)

        form.addRow("Dataset", self.dataset_combo)
        form.addRow("Samples", self.samples_spin)
        form.addRow("Noise", self.noise_spin)
        form.addRow("Seed", self.seed_spin)
        form.addRow("Test split", self.test_split_spin)
        form.addRow(self.regen_button)

        self.dataset_combo.currentIndexChanged.connect(self.generate_and_plot)
        self.samples_spin.valueChanged.connect(self.generate_and_plot)
        self.noise_spin.valueChanged.connect(self.generate_and_plot)
        self.seed_spin.valueChanged.connect(self.generate_and_plot)
        self.test_split_spin.valueChanged.connect(self._reset_training)

        return group

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("Model")
        form = QFormLayout(group)

        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(0.01, 100.0)
        self.c_spin.setSingleStep(0.1)
        self.c_spin.setValue(1.0)

        self.gamma_edit = QLineEdit("scale")

        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 500)
        self.n_estimators_spin.setValue(200)

        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(0, 50)
        self.max_depth_spin.setSpecialValueText("None")
        self.max_depth_spin.setValue(0)

        self.n_neighbors_spin = QSpinBox()
        self.n_neighbors_spin.setRange(1, 50)
        self.n_neighbors_spin.setValue(7)

        self.hidden_layers_edit = QLineEdit("16,16")

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.00001, 1.0)
        self.alpha_spin.setDecimals(6)
        self.alpha_spin.setValue(0.0001)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(50, 2000)
        self.max_iter_spin.setValue(300)

        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["relu", "tanh", "sigmoid"])

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.01)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(8, 512)
        self.batch_size_spin.setValue(32)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)

        form.addRow("Model", self.model_combo)
        form.addRow("C", self.c_spin)
        form.addRow("Gamma", self.gamma_edit)
        form.addRow("N estimators", self.n_estimators_spin)
        form.addRow("Max depth", self.max_depth_spin)
        form.addRow("K neighbors", self.n_neighbors_spin)
        form.addRow("Hidden layers", self.hidden_layers_edit)
        form.addRow("Alpha", self.alpha_spin)
        form.addRow("Max iter", self.max_iter_spin)
        form.addRow("Activation", self.activation_combo)
        form.addRow("Learning rate", self.lr_spin)
        form.addRow("Batch size", self.batch_size_spin)
        form.addRow("Epochs", self.epochs_spin)

        return group

    def _build_train_group(self) -> QGroupBox:
        group = QGroupBox("Training")
        layout = QHBoxLayout(group)

        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.on_train)

        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self.on_step)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.on_play)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset_training)

        layout.addWidget(self.train_button)
        layout.addWidget(self.step_button)
        layout.addWidget(self.play_button)
        layout.addWidget(self.reset_button)

        return group

    def _build_metrics_group(self) -> QGroupBox:
        group = QGroupBox("Metrics")
        layout = QVBoxLayout(group)
        self.metrics_text = QPlainTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumBlockCount(50)
        layout.addWidget(self.metrics_text)
        return group

    def _load_models(self) -> None:
        self.model_combo.clear()
        self.model_specs = {}
        for spec in get_model_specs():
            self.model_combo.addItem(spec.name)
            self.model_specs[spec.name] = spec

    def _update_param_widgets(self) -> None:
        name = self.model_combo.currentText()
        mapping = {
            "Logistic Regression": {"C", "max_iter"},
            "SVM (RBF)": {"C", "gamma"},
            "Random Forest": {"n_estimators", "max_depth"},
            "KNN": {"n_neighbors"},
            "MLPClassifier": {"hidden_layers", "alpha", "max_iter"},
            "Keras MLP": {"hidden_layers", "activation", "lr", "batch_size", "epochs"},
        }
        enabled = mapping.get(name, set())

        self._set_enabled(self.c_spin, "C" in enabled)
        self._set_enabled(self.gamma_edit, "gamma" in enabled)
        self._set_enabled(self.n_estimators_spin, "n_estimators" in enabled)
        self._set_enabled(self.max_depth_spin, "max_depth" in enabled)
        self._set_enabled(self.n_neighbors_spin, "n_neighbors" in enabled)
        self._set_enabled(self.hidden_layers_edit, "hidden_layers" in enabled)
        self._set_enabled(self.alpha_spin, "alpha" in enabled)
        self._set_enabled(self.max_iter_spin, "max_iter" in enabled)
        self._set_enabled(self.activation_combo, "activation" in enabled)
        self._set_enabled(self.lr_spin, "lr" in enabled)
        self._set_enabled(self.batch_size_spin, "batch_size" in enabled)
        self._set_enabled(self.epochs_spin, "epochs" in enabled)

        supports_step = name == "Keras MLP"
        self.step_button.setEnabled(supports_step)
        self.play_button.setEnabled(supports_step)

    @staticmethod
    def _set_enabled(widget, enabled: bool) -> None:
        widget.setEnabled(enabled)

    def _collect_params(self) -> Dict[str, object]:
        params: Dict[str, object] = {}
        params["C"] = self.c_spin.value()
        params["gamma"] = self._parse_gamma(self.gamma_edit.text())
        params["n_estimators"] = self.n_estimators_spin.value()
        params["max_depth"] = None if self.max_depth_spin.value() == 0 else self.max_depth_spin.value()
        params["n_neighbors"] = self.n_neighbors_spin.value()
        params["hidden_layers"] = self._parse_hidden_layers(self.hidden_layers_edit.text())
        params["alpha"] = self.alpha_spin.value()
        params["max_iter"] = self.max_iter_spin.value()
        params["activation"] = self.activation_combo.currentText()
        params["lr"] = self.lr_spin.value()
        params["batch_size"] = self.batch_size_spin.value()
        params["epochs"] = self.epochs_spin.value()
        return params

    def _parse_gamma(self, text: str):
        text = text.strip()
        if text in {"scale", "auto"}:
            return text
        try:
            return float(text)
        except ValueError:
            return "scale"

    def _parse_hidden_layers(self, text: str) -> Tuple[int, ...]:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            return (16, 16)
        layers = []
        for part in parts:
            try:
                layers.append(int(part))
            except ValueError:
                pass
        return tuple(layers) if layers else (16, 16)

    def on_model_changed(self) -> None:
        self._apply_model_defaults(self.model_combo.currentText())
        self._update_param_widgets()
        self._reset_training()

    def _apply_model_defaults(self, name: str) -> None:
        spec = self.model_specs.get(name)
        if spec is None:
            return
        defaults = spec.defaults
        if "C" in defaults:
            self.c_spin.setValue(float(defaults["C"]))
        if "gamma" in defaults:
            self.gamma_edit.setText(str(defaults["gamma"]))
        if "n_estimators" in defaults:
            self.n_estimators_spin.setValue(int(defaults["n_estimators"]))
        if "max_depth" in defaults:
            depth = defaults["max_depth"]
            self.max_depth_spin.setValue(0 if depth is None else int(depth))
        if "n_neighbors" in defaults:
            self.n_neighbors_spin.setValue(int(defaults["n_neighbors"]))
        if "hidden_layers" in defaults:
            layers = defaults["hidden_layers"]
            self.hidden_layers_edit.setText(",".join(str(x) for x in layers))
        if "alpha" in defaults:
            self.alpha_spin.setValue(float(defaults["alpha"]))
        if "max_iter" in defaults:
            self.max_iter_spin.setValue(int(defaults["max_iter"]))
        if "activation" in defaults:
            idx = self.activation_combo.findText(str(defaults["activation"]))
            if idx >= 0:
                self.activation_combo.setCurrentIndex(idx)
        if "lr" in defaults:
            self.lr_spin.setValue(float(defaults["lr"]))
        if "batch_size" in defaults:
            self.batch_size_spin.setValue(int(defaults["batch_size"]))
        if "epochs" in defaults:
            self.epochs_spin.setValue(int(defaults["epochs"]))

    def generate_and_plot(self) -> None:
        name = self.dataset_combo.currentText()
        samples = int(self.samples_spin.value())
        noise = float(self.noise_spin.value())
        seed = int(self.seed_spin.value())

        self.X, self.y = generate_dataset(name, samples, noise, seed)
        self.dataset_view.plot(self.X, self.y)
        self._reset_training(clear_plot=False)

    def _split_data(self) -> None:
        if self.X is None or self.y is None:
            return
        test_size = float(self.test_split_spin.value())
        stratify = self.y if len(np.unique(self.y)) > 1 else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=int(self.seed_spin.value()),
            stratify=stratify,
        )

    def _ensure_training_ready(self) -> None:
        if self.X is None or self.y is None:
            self.generate_and_plot()
        if self.X_train is None or self.y_train is None:
            self._split_data()
        if self.model is None:
            self._build_model()

    def _build_model(self) -> None:
        name = self.model_combo.currentText()
        params = self._collect_params()
        self.model = create_model(name, params)
        self.trainer = Trainer(self.model)

    def on_train(self) -> None:
        self._ensure_training_ready()
        if self.trainer is None or self.X_train is None or self.y_train is None:
            return
        epochs = None
        if self.model_combo.currentText() == "Keras MLP":
            epochs = int(self.epochs_spin.value())
        result = self.trainer.train_full(self.X_train, self.y_train, self.X_test, self.y_test, epochs=epochs)
        self._apply_result(result)

    def on_step(self) -> None:
        if self.model_combo.currentText() != "Keras MLP":
            return
        self._ensure_training_ready()
        if self.trainer is None or self.X_train is None or self.y_train is None:
            return
        result = self.trainer.train_step(self.X_train, self.y_train, self.X_test, self.y_test)
        self._apply_result(result)

    def on_play(self) -> None:
        if self.model_combo.currentText() != "Keras MLP":
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("Play")
        else:
            self._ensure_training_ready()
            self.play_timer.start()
            self.play_button.setText("Pause")

    def _apply_result(self, result: TrainResult) -> None:
        if self.X is None or self.y is None:
            return
        self.dataset_view.plot(self.X, self.y, heatmap=result.heatmap)
        self.metrics_text.setPlainText(self._format_metrics(result.metrics))

    def _format_metrics(self, metrics: Dict[str, object]) -> str:
        lines = []
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "loss"]:
            if key not in metrics:
                continue
            val = metrics.get(key)
            if val is None:
                lines.append(f"{key}: n/a")
            else:
                lines.append(f"{key}: {val:.4f}")
        return "
".join(lines)

    def _reset_training(self, clear_plot: bool = True) -> None:
        self.play_timer.stop()
        self.play_button.setText("Play")
        self.model = None
        self.trainer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics_text.setPlainText("")
        if clear_plot and self.X is not None and self.y is not None:
            self.dataset_view.plot(self.X, self.y)
