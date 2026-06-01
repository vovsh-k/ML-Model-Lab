from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from ml_model_lab.core.app_state import AppState
from ml_model_lab.data.dataset_generators import generate_dataset, list_datasets
from ml_model_lab.models.model_registry import ModelSpec, create_model, get_model_specs
from ml_model_lab.training.trainer import TrainResult, Trainer
from ml_model_lab.viz.dataset_view import DatasetView
from ml_model_lab.viz.loss_plot_view import LossPlotView
from ml_model_lab.viz.network_view import NetworkView


FeatureFn = Callable[[np.ndarray], np.ndarray]


FEATURES: Tuple[Tuple[str, FeatureFn, bool], ...] = (
    ("x1", lambda X: X[:, 0], True),
    ("x2", lambda X: X[:, 1], True),
    ("x1^2", lambda X: X[:, 0] ** 2, False),
    ("x2^2", lambda X: X[:, 1] ** 2, False),
    ("x1*x2", lambda X: X[:, 0] * X[:, 1], False),
    ("sin(x1)", lambda X: np.sin(X[:, 0]), False),
    ("sin(x2)", lambda X: np.sin(X[:, 1]), False),
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ML Model Lab")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 760)
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
        self.nn_epoch = 0
        self.nn_target_epoch = 0
        self.nn_hidden_layers = [4, 2]

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(self.state.training.play_interval_ms)
        self.play_timer.timeout.connect(self.on_nn_step)

        self._classic_rows: Dict[str, Tuple[QLabel, QWidget]] = {}
        self.feature_checks: Dict[str, QCheckBox] = {}

        self._build_ui()
        self._load_models()
        self._apply_styles()
        self.generate_and_plot()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QLabel("ML Model Lab")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Interactive playground for synthetic datasets, neural networks, and classic ML models.")
        subtitle.setObjectName("AppSubtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        workspace = QSplitter(Qt.Orientation.Horizontal)
        workspace.setChildrenCollapsible(False)

        left_rail = QWidget()
        left_layout = QVBoxLayout(left_rail)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left_layout.addWidget(self._build_data_group())
        left_layout.addWidget(self._build_features_group())
        self.nn_controls_group = self._build_nn_config_group()
        left_layout.addWidget(self.nn_controls_group)
        left_layout.addStretch(1)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.addTab(self._build_nn_page(), "Neural Network")
        self.tabs.addTab(self._build_classic_page(), "Classic Models")

        workspace.addWidget(left_rail)
        workspace.addWidget(self.tabs)
        workspace.setStretchFactor(0, 0)
        workspace.setStretchFactor(1, 1)
        workspace.setSizes([280, 1160])
        root.addWidget(workspace, 1)

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
        group.setMaximumHeight(245)
        return group

    def _build_features_group(self) -> QGroupBox:
        group = QGroupBox("Features")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        for idx, (name, _, checked) in enumerate(FEATURES):
            check = QCheckBox(name)
            check.setChecked(checked)
            check.stateChanged.connect(self._on_features_changed)
            self.feature_checks[name] = check
            layout.addWidget(check)
        group.setMaximumHeight(210)
        return group

    def _build_nn_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        toolbar = QFrame()
        toolbar.setObjectName("Toolbar")
        top = QHBoxLayout(toolbar)

        self.nn_epoch_label = QLabel("Epoch 0")
        self.nn_epoch_label.setObjectName("EpochLabel")

        self.nn_play_button = QPushButton("Play")
        self.nn_play_button.clicked.connect(self.on_nn_play)
        self.nn_stop_button = QPushButton("Stop")
        self.nn_stop_button.clicked.connect(self.on_nn_stop)
        self.nn_step_button = QPushButton("Step")
        self.nn_step_button.clicked.connect(self.on_nn_step)
        self.nn_train_button = QPushButton("Train")
        self.nn_train_button.clicked.connect(self.on_nn_train)
        self.nn_reset_button = QPushButton("Reset")
        self.nn_reset_button.clicked.connect(self._reset_training)
        self.nn_auto_button = QPushButton("Playground defaults")
        self.nn_auto_button.clicked.connect(self.apply_nn_defaults)

        top.addWidget(self.nn_epoch_label)
        top.addStretch(1)
        top.addWidget(self.nn_play_button)
        top.addWidget(self.nn_stop_button)
        top.addWidget(self.nn_step_button)
        top.addWidget(self.nn_train_button)
        top.addWidget(self.nn_reset_button)
        top.addWidget(self.nn_auto_button)
        layout.addWidget(toolbar)

        body = QHBoxLayout()
        body.setSpacing(12)

        network_panel = QGroupBox("Hidden Layers")
        network_layout = QVBoxLayout(network_panel)
        self.network_view = NetworkView()
        network_layout.addWidget(self.network_view, 1)
        network_panel.setMinimumWidth(300)
        body.addWidget(network_panel, 2)

        output_panel = QGroupBox("Output")
        output_layout = QVBoxLayout(output_panel)
        output_layout.setSpacing(10)
        self.nn_dataset_view = DatasetView()
        self.nn_loss_view = LossPlotView()
        self.nn_metrics_text = self._make_metrics_box()
        self.nn_dataset_view.setMinimumSize(500, 380)
        output_layout.addWidget(self.nn_dataset_view, 6)
        output_layout.addWidget(self.nn_loss_view, 1)
        output_layout.addWidget(self.nn_metrics_text, 0)
        output_panel.setMinimumWidth(560)
        body.addWidget(output_panel, 4)

        layout.addLayout(body, 1)
        return page

    def _build_nn_config_group(self) -> QGroupBox:
        group = QGroupBox("Network controls")
        group.setMaximumWidth(280)
        form = QFormLayout(group)

        self.nn_activation_combo = QComboBox()
        self.nn_activation_combo.addItems(["tanh", "relu", "sigmoid"])
        self.nn_lr_spin = QDoubleSpinBox()
        self.nn_lr_spin.setRange(0.0001, 1.0)
        self.nn_lr_spin.setDecimals(4)
        self.nn_lr_spin.setValue(0.03)
        self.nn_l2_spin = QDoubleSpinBox()
        self.nn_l2_spin.setRange(0.0, 1.0)
        self.nn_l2_spin.setDecimals(5)
        self.nn_l2_spin.setSingleStep(0.001)
        self.nn_batch_size_spin = QSpinBox()
        self.nn_batch_size_spin.setRange(1, 512)
        self.nn_batch_size_spin.setValue(10)
        self.nn_epochs_spin = QSpinBox()
        self.nn_epochs_spin.setRange(1, 500)
        self.nn_epochs_spin.setValue(50)

        form.addRow("Learning rate", self.nn_lr_spin)
        form.addRow("Activation", self.nn_activation_combo)
        form.addRow("L2 regularization", self.nn_l2_spin)
        form.addRow("Batch size", self.nn_batch_size_spin)
        form.addRow("Play / train epochs", self.nn_epochs_spin)

        self.nn_architecture_label = QLabel("")
        self.nn_layer_combo = QComboBox()
        self.nn_layer_combo.currentIndexChanged.connect(self._update_architecture_label)

        layer_buttons = QGridLayout()
        self.nn_add_layer_button = QPushButton("+ layer")
        self.nn_add_layer_button.clicked.connect(self.add_nn_layer)
        self.nn_remove_layer_button = QPushButton("- layer")
        self.nn_remove_layer_button.clicked.connect(self.remove_nn_layer)
        self.nn_add_neuron_button = QPushButton("+ neuron")
        self.nn_add_neuron_button.clicked.connect(self.add_nn_neuron)
        self.nn_remove_neuron_button = QPushButton("- neuron")
        self.nn_remove_neuron_button.clicked.connect(self.remove_nn_neuron)
        for button in (
            self.nn_add_layer_button,
            self.nn_remove_layer_button,
            self.nn_add_neuron_button,
            self.nn_remove_neuron_button,
        ):
            button.setMinimumHeight(30)
        layer_buttons.addWidget(self.nn_add_layer_button, 0, 0)
        layer_buttons.addWidget(self.nn_remove_layer_button, 0, 1)
        layer_buttons.addWidget(self.nn_add_neuron_button, 1, 0)
        layer_buttons.addWidget(self.nn_remove_neuron_button, 1, 1)

        layer_widget = QWidget()
        layer_widget.setLayout(layer_buttons)
        layer_widget.setMinimumHeight(76)
        form.addRow(layer_widget)
        form.addRow("Layer", self.nn_layer_combo)
        form.addRow("Architecture", self.nn_architecture_label)

        for widget in (
            self.nn_activation_combo,
            self.nn_lr_spin,
            self.nn_l2_spin,
            self.nn_batch_size_spin,
        ):
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._on_nn_config_changed)
            else:
                widget.valueChanged.connect(self._on_nn_config_changed)
        self._refresh_layer_controls()
        group.setMinimumHeight(330)
        return group

    def _build_classic_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setSpacing(12)

        controls = QVBoxLayout()
        controls.addWidget(self._build_classic_model_group())
        controls.addWidget(self._build_classic_train_group())
        controls.addStretch(1)
        layout.addLayout(controls, 0)

        output = QGroupBox("Output")
        output_layout = QVBoxLayout(output)
        self.classic_dataset_view = DatasetView()
        self.classic_metrics_text = self._make_metrics_box()
        output_layout.addWidget(self.classic_dataset_view, 1)
        output_layout.addWidget(self.classic_metrics_text, 0)
        layout.addWidget(output, 1)
        return page

    def _build_classic_model_group(self) -> QGroupBox:
        group = QGroupBox("Model parameters")
        form = QFormLayout(group)

        self.classic_model_combo = QComboBox()
        self.classic_model_combo.currentIndexChanged.connect(self.on_classic_model_changed)
        form.addRow("Model", self.classic_model_combo)

        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(0.01, 100.0)
        self.c_spin.setSingleStep(0.1)
        self.c_spin.setValue(1.0)
        self._add_classic_row(form, "C", self.c_spin, "C")

        self.penalty_combo = QComboBox()
        self.penalty_combo.addItems(["l2", "l1"])
        self._add_classic_row(form, "Penalty", self.penalty_combo, "penalty")

        self.gamma_edit = QLineEdit("scale")
        self._add_classic_row(form, "Gamma", self.gamma_edit, "gamma")

        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["rbf", "linear", "poly", "sigmoid"])
        self._add_classic_row(form, "Kernel", self.kernel_combo, "kernel")

        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(2, 6)
        self.degree_spin.setValue(3)
        self._add_classic_row(form, "Degree", self.degree_spin, "degree")

        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 500)
        self.n_estimators_spin.setValue(200)
        self._add_classic_row(form, "Trees", self.n_estimators_spin, "n_estimators")

        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(0, 50)
        self.max_depth_spin.setSpecialValueText("None")
        self.max_depth_spin.setValue(0)
        self._add_classic_row(form, "Max depth", self.max_depth_spin, "max_depth")

        self.min_samples_split_spin = QSpinBox()
        self.min_samples_split_spin.setRange(2, 20)
        self.min_samples_split_spin.setValue(2)
        self._add_classic_row(form, "Min split", self.min_samples_split_spin, "min_samples_split")

        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(["gini", "entropy", "log_loss"])
        self._add_classic_row(form, "Criterion", self.criterion_combo, "criterion")

        self.n_neighbors_spin = QSpinBox()
        self.n_neighbors_spin.setRange(1, 50)
        self.n_neighbors_spin.setValue(7)
        self._add_classic_row(form, "Neighbors", self.n_neighbors_spin, "n_neighbors")

        self.weights_combo = QComboBox()
        self.weights_combo.addItems(["uniform", "distance"])
        self._add_classic_row(form, "Weights", self.weights_combo, "weights")

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["minkowski", "euclidean", "manhattan"])
        self._add_classic_row(form, "Metric", self.metric_combo, "metric")

        self.classic_hidden_layers_edit = QLineEdit("16,16")
        self._add_classic_row(form, "Hidden layers", self.classic_hidden_layers_edit, "hidden_layers")

        self.classic_activation_combo = QComboBox()
        self.classic_activation_combo.addItems(["relu", "tanh", "logistic"])
        self._add_classic_row(form, "Activation", self.classic_activation_combo, "activation")

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.00001, 1.0)
        self.alpha_spin.setDecimals(6)
        self.alpha_spin.setValue(0.0001)
        self._add_classic_row(form, "Alpha", self.alpha_spin, "alpha")

        self.classic_lr_spin = QDoubleSpinBox()
        self.classic_lr_spin.setRange(0.0001, 1.0)
        self.classic_lr_spin.setDecimals(4)
        self.classic_lr_spin.setValue(0.001)
        self._add_classic_row(form, "Learning rate", self.classic_lr_spin, "lr")

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(50, 2000)
        self.max_iter_spin.setValue(300)
        self._add_classic_row(form, "Max iter", self.max_iter_spin, "max_iter")

        for widget in group.findChildren(QWidget):
            if widget is not self.classic_model_combo:
                if isinstance(widget, QComboBox):
                    widget.currentIndexChanged.connect(self._reset_training)
                elif isinstance(widget, QLineEdit):
                    widget.textChanged.connect(self._reset_training)
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.valueChanged.connect(self._reset_training)
        return group

    def _build_classic_train_group(self) -> QGroupBox:
        group = QGroupBox("Training")
        layout = QHBoxLayout(group)
        self.classic_train_button = QPushButton("Train")
        self.classic_train_button.clicked.connect(self.on_classic_train)
        self.classic_auto_button = QPushButton("Auto params")
        self.classic_auto_button.clicked.connect(self.auto_tune_classic)
        self.classic_reset_button = QPushButton("Reset")
        self.classic_reset_button.clicked.connect(self._reset_training)
        layout.addWidget(self.classic_train_button)
        layout.addWidget(self.classic_auto_button)
        layout.addWidget(self.classic_reset_button)
        return group

    def _add_classic_row(self, form: QFormLayout, label: str, widget: QWidget, key: str) -> None:
        label_widget = QLabel(label)
        form.addRow(label_widget, widget)
        self._classic_rows[key] = (label_widget, widget)

    @staticmethod
    def _make_metrics_box() -> QPlainTextEdit:
        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setMaximumHeight(86)
        box.setPlaceholderText("Metrics will appear after training.")
        return box

    def _load_models(self) -> None:
        self.model_specs = {spec.name: spec for spec in get_model_specs()}
        self.classic_model_combo.clear()
        for spec in get_model_specs():
            if spec.name != "Keras MLP":
                self.classic_model_combo.addItem(spec.name)
        self._apply_classic_defaults(self.classic_model_combo.currentText())
        self._update_classic_param_visibility()
        self._refresh_network_view()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-size: 13px;
                color: #172033;
            }
            QMainWindow, QWidget {
                background: #eef2f6;
            }
            #AppTitle {
                font-size: 26px;
                font-weight: 700;
                color: #0f172a;
            }
            #AppSubtitle {
                color: #64748b;
            }
            QGroupBox {
                background: #f8fafc;
                border: 1px solid #d7dee8;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            #Toolbar {
                background: #f8fafc;
                border: 1px solid #d7dee8;
                border-radius: 8px;
            }
            #EpochLabel {
                font-size: 17px;
                font-weight: 700;
                color: #1d4ed8;
            }
            QPushButton {
                background: #1d4ed8;
                color: white;
                border: 0;
                border-radius: 6px;
                padding: 7px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #2563eb;
            }
            QPushButton:disabled {
                background: #94a3b8;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit {
                background: white;
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                padding: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                background: #f8fafc;
            }
            QTabBar::tab {
                background: #dbe4ee;
                padding: 9px 16px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                margin-right: 3px;
            }
            QTabBar::tab:selected {
                background: #f8fafc;
                color: #1d4ed8;
                font-weight: 700;
            }
            """
        )

    def _on_features_changed(self) -> None:
        if not self._selected_feature_names():
            self.feature_checks["x1"].setChecked(True)
            self.feature_checks["x2"].setChecked(True)
        self._refresh_network_view()
        self._reset_training()

    def _on_nn_config_changed(self) -> None:
        self._refresh_network_view()
        self._reset_training()

    def _on_tab_changed(self, index: int) -> None:
        if hasattr(self, "nn_controls_group"):
            self.nn_controls_group.setVisible(index == 0)
        self._reset_training()

    def _refresh_layer_controls(self) -> None:
        if not hasattr(self, "nn_layer_combo"):
            return
        current = self.nn_layer_combo.currentIndex()
        self.nn_layer_combo.blockSignals(True)
        self.nn_layer_combo.clear()
        for idx, units in enumerate(self.nn_hidden_layers):
            self.nn_layer_combo.addItem(f"Hidden {idx + 1}: {units}")
        if self.nn_hidden_layers:
            self.nn_layer_combo.setCurrentIndex(min(max(current, 0), len(self.nn_hidden_layers) - 1))
        self.nn_layer_combo.blockSignals(False)
        self._update_architecture_label()

    def _update_architecture_label(self) -> None:
        if not hasattr(self, "nn_architecture_label"):
            return
        architecture = " - ".join(str(units) for units in self.nn_hidden_layers)
        self.nn_architecture_label.setText(architecture if architecture else "No hidden layers")

    def _change_nn_architecture(self) -> None:
        self._refresh_layer_controls()
        self._refresh_network_view()
        self._reset_training()

    def add_nn_layer(self) -> None:
        self.nn_hidden_layers.append(4)
        self._change_nn_architecture()

    def remove_nn_layer(self) -> None:
        if len(self.nn_hidden_layers) <= 1:
            return
        index = self.nn_layer_combo.currentIndex()
        if index < 0:
            index = len(self.nn_hidden_layers) - 1
        self.nn_hidden_layers.pop(index)
        self._change_nn_architecture()

    def add_nn_neuron(self) -> None:
        index = self.nn_layer_combo.currentIndex()
        if index < 0:
            return
        self.nn_hidden_layers[index] = min(self.nn_hidden_layers[index] + 1, 32)
        self._change_nn_architecture()

    def remove_nn_neuron(self) -> None:
        index = self.nn_layer_combo.currentIndex()
        if index < 0:
            return
        self.nn_hidden_layers[index] = max(self.nn_hidden_layers[index] - 1, 1)
        self._change_nn_architecture()

    def on_classic_model_changed(self) -> None:
        self._apply_classic_defaults(self.classic_model_combo.currentText())
        self._update_classic_param_visibility()
        self._reset_training()

    def _refresh_network_view(self) -> None:
        if hasattr(self, "network_view"):
            self.network_view.plot(self._selected_feature_names(), tuple(self.nn_hidden_layers))

    def _selected_feature_names(self) -> Tuple[str, ...]:
        return tuple(name for name, _, _ in FEATURES if self.feature_checks.get(name) and self.feature_checks[name].isChecked())

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        cols = []
        selected = set(self._selected_feature_names())
        for name, fn, _ in FEATURES:
            if name in selected:
                cols.append(fn(X))
        if not cols:
            cols = [X[:, 0], X[:, 1]]
        return np.column_stack(cols)

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
        layers = []
        for part in parts:
            try:
                value = int(part)
            except ValueError:
                continue
            if value > 0:
                layers.append(value)
        return tuple(layers) if layers else (4, 2)

    def _set_combo(self, combo: QComboBox, value: object) -> None:
        idx = combo.findText(str(value))
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _apply_classic_defaults(self, name: str) -> None:
        spec = self.model_specs.get(name)
        if spec is None:
            return
        defaults = spec.defaults
        if "C" in defaults:
            self.c_spin.setValue(float(defaults["C"]))
        if "penalty" in defaults:
            self._set_combo(self.penalty_combo, defaults["penalty"])
        if "gamma" in defaults:
            self.gamma_edit.setText(str(defaults["gamma"]))
        if "kernel" in defaults:
            self._set_combo(self.kernel_combo, defaults["kernel"])
        if "degree" in defaults:
            self.degree_spin.setValue(int(defaults["degree"]))
        if "n_estimators" in defaults:
            self.n_estimators_spin.setValue(int(defaults["n_estimators"]))
        if "max_depth" in defaults:
            depth = defaults["max_depth"]
            self.max_depth_spin.setValue(0 if depth is None else int(depth))
        if "min_samples_split" in defaults:
            self.min_samples_split_spin.setValue(int(defaults["min_samples_split"]))
        if "criterion" in defaults:
            self._set_combo(self.criterion_combo, defaults["criterion"])
        if "n_neighbors" in defaults:
            self.n_neighbors_spin.setValue(int(defaults["n_neighbors"]))
        if "weights" in defaults:
            self._set_combo(self.weights_combo, defaults["weights"])
        if "metric" in defaults:
            self._set_combo(self.metric_combo, defaults["metric"])
        if "hidden_layers" in defaults:
            self.classic_hidden_layers_edit.setText(",".join(str(x) for x in defaults["hidden_layers"]))
        if "activation" in defaults:
            self._set_combo(self.classic_activation_combo, defaults["activation"])
        if "alpha" in defaults:
            self.alpha_spin.setValue(float(defaults["alpha"]))
        if "lr" in defaults:
            self.classic_lr_spin.setValue(float(defaults["lr"]))
        if "max_iter" in defaults:
            self.max_iter_spin.setValue(int(defaults["max_iter"]))

    def _update_classic_param_visibility(self) -> None:
        name = self.classic_model_combo.currentText()
        mapping = {
            "Logistic Regression": {"C", "penalty", "max_iter"},
            "SVM (RBF)": {"C", "kernel", "gamma", "degree"},
            "Random Forest": {"n_estimators", "max_depth", "min_samples_split", "criterion"},
            "KNN": {"n_neighbors", "weights", "metric"},
            "MLPClassifier": {"hidden_layers", "activation", "alpha", "lr", "max_iter"},
        }
        visible = mapping.get(name, set())
        for key, (label, widget) in self._classic_rows.items():
            show = key in visible
            label.setVisible(show)
            widget.setVisible(show)

    def _collect_classic_params(self) -> Dict[str, object]:
        return {
            "C": self.c_spin.value(),
            "penalty": self.penalty_combo.currentText(),
            "gamma": self._parse_gamma(self.gamma_edit.text()),
            "kernel": self.kernel_combo.currentText(),
            "degree": self.degree_spin.value(),
            "n_estimators": self.n_estimators_spin.value(),
            "max_depth": None if self.max_depth_spin.value() == 0 else self.max_depth_spin.value(),
            "min_samples_split": self.min_samples_split_spin.value(),
            "criterion": self.criterion_combo.currentText(),
            "n_neighbors": self.n_neighbors_spin.value(),
            "weights": self.weights_combo.currentText(),
            "metric": self.metric_combo.currentText(),
            "hidden_layers": self._parse_hidden_layers(self.classic_hidden_layers_edit.text()),
            "activation": self.classic_activation_combo.currentText(),
            "alpha": self.alpha_spin.value(),
            "lr": self.classic_lr_spin.value(),
            "max_iter": self.max_iter_spin.value(),
        }

    def _collect_nn_params(self) -> Dict[str, object]:
        return {
            "input_dim": len(self._selected_feature_names()),
            "hidden_layers": tuple(self.nn_hidden_layers),
            "activation": self.nn_activation_combo.currentText(),
            "lr": self.nn_lr_spin.value(),
            "l2": self.nn_l2_spin.value(),
            "batch_size": self.nn_batch_size_spin.value(),
            "epochs": self.nn_epochs_spin.value(),
        }

    def generate_and_plot(self) -> None:
        name = self.dataset_combo.currentText()
        samples = int(self.samples_spin.value())
        noise = float(self.noise_spin.value())
        seed = int(self.seed_spin.value())

        self.X, self.y = generate_dataset(name, samples, noise, seed)
        self.nn_dataset_view.plot(self.X, self.y)
        self.classic_dataset_view.plot(self.X, self.y)
        self._reset_training(clear_plot=False)

    def _split_data(self) -> None:
        if self.X is None or self.y is None:
            return
        stratify = self.y if len(np.unique(self.y)) > 1 else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=float(self.test_split_spin.value()),
            random_state=int(self.seed_spin.value()),
            stratify=stratify,
        )

    def _ensure_training_ready(self) -> bool:
        if self.X is None or self.y is None:
            self.generate_and_plot()
        if self.X_train is None or self.y_train is None:
            self._split_data()
        return self.X_train is not None and self.X_test is not None and self.y_train is not None and self.y_test is not None

    def _build_classic_model(self) -> bool:
        try:
            self.model = create_model(self.classic_model_combo.currentText(), self._collect_classic_params())
        except Exception as exc:
            self._show_error("Model error", str(exc))
            return False
        self.trainer = Trainer(self.model, feature_transform=self._transform_features)
        return True

    def _build_nn_model(self) -> bool:
        try:
            self.model = create_model("Keras MLP", self._collect_nn_params())
        except ModuleNotFoundError:
            self._show_error(
                "TensorFlow is not installed",
                "Install the optional dependency first: pip install -e .[keras]",
            )
            return False
        except Exception as exc:
            self._show_error("Keras error", str(exc))
            return False
        self.trainer = Trainer(self.model, feature_transform=self._transform_features)
        return True

    def on_classic_train(self) -> None:
        self._reset_training(clear_plot=False)
        if not self._ensure_training_ready() or not self._build_classic_model():
            return
        if self.trainer is None or self.X_train is None or self.y_train is None:
            return
        result = self.trainer.train_full(self.X_train, self.y_train, self.X_test, self.y_test)
        self._apply_result(result, self.classic_dataset_view, self.classic_metrics_text)

    def on_nn_train(self) -> None:
        self.on_nn_stop()
        self._reset_training(clear_plot=False)
        if not self._ensure_training_ready() or not self._build_nn_model():
            return
        if self.trainer is None or self.X_train is None or self.y_train is None:
            return
        epochs = int(self.nn_epochs_spin.value())
        result = self.trainer.train_full(self.X_train, self.y_train, self.X_test, self.y_test, epochs=epochs)
        self.nn_epoch += epochs
        self._apply_result(result, self.nn_dataset_view, self.nn_metrics_text)
        self._sync_nn_loss_plot()
        self._update_epoch_label()

    def on_nn_step(self) -> None:
        if self.model is None:
            if not self._ensure_training_ready() or not self._build_nn_model():
                self.play_timer.stop()
                self.nn_play_button.setText("Play")
                return
        if self.trainer is None or self.X_train is None or self.y_train is None:
            return
        result = self.trainer.train_step(self.X_train, self.y_train, self.X_test, self.y_test)
        self.nn_epoch += 1
        self._apply_result(result, self.nn_dataset_view, self.nn_metrics_text)
        self._sync_nn_loss_plot()
        self._update_epoch_label()
        if self.play_timer.isActive() and self.nn_target_epoch and self.nn_epoch >= self.nn_target_epoch:
            self.on_nn_stop()

    def on_nn_play(self) -> None:
        if self.play_timer.isActive():
            self.on_nn_stop()
            return
        if not self._ensure_training_ready():
            return
        if self.model is None and not self._build_nn_model():
            return
        self.nn_target_epoch = self.nn_epoch + int(self.nn_epochs_spin.value())
        self.play_timer.start()
        self.nn_play_button.setText("Pause")

    def on_nn_stop(self) -> None:
        self.play_timer.stop()
        self.nn_target_epoch = 0
        if hasattr(self, "nn_play_button"):
            self.nn_play_button.setText("Play")

    def auto_tune_classic(self) -> None:
        if not self._ensure_training_ready():
            return
        if self.X_train is None or self.y_train is None:
            return

        name = self.classic_model_combo.currentText()
        X_train = self._transform_features(self.X_train)
        estimator, grid = self._grid_for_classic_model(name)
        if estimator is None:
            self._show_error("Auto params", "No grid is configured for this model.")
            return

        self.classic_auto_button.setEnabled(False)
        self.classic_auto_button.setText("Tuning...")
        try:
            search = GridSearchCV(estimator, grid, cv=3, scoring="accuracy", n_jobs=1)
            search.fit(X_train, self.y_train)
        except Exception as exc:
            self._show_error("GridSearch error", str(exc))
            return
        finally:
            self.classic_auto_button.setEnabled(True)
            self.classic_auto_button.setText("Auto params")

        self._apply_grid_params(search.best_params_)
        search_summary = f"Best CV accuracy: {search.best_score_:.4f}\n{search.best_params_}"
        self.on_classic_train()
        metrics = self.classic_metrics_text.toPlainText()
        self.classic_metrics_text.setPlainText(f"{search_summary}\n\nTest metrics\n{metrics}")

    def _grid_for_classic_model(self, name: str):
        if name == "Logistic Regression":
            return LogisticRegression(max_iter=500), {"C": [0.3, 1.0, 3.0], "penalty": ["l2"], "solver": ["lbfgs"]}
        if name == "SVM (RBF)":
            return SVC(probability=True), {"C": [0.5, 1.0, 3.0], "gamma": ["scale", 0.5, 1.0], "kernel": ["rbf"]}
        if name == "Random Forest":
            return RandomForestClassifier(random_state=42), {
                "n_estimators": [100, 200],
                "max_depth": [None, 4, 8],
                "min_samples_split": [2, 4],
            }
        if name == "KNN":
            return KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]}
        if name == "MLPClassifier":
            return MLPClassifier(max_iter=400, random_state=42), {
                "hidden_layer_sizes": [(8,), (16, 16)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001],
            }
        return None, None

    def _apply_grid_params(self, params: Dict[str, object]) -> None:
        if "C" in params:
            self.c_spin.setValue(float(params["C"]))
        if "penalty" in params:
            self._set_combo(self.penalty_combo, params["penalty"])
        if "gamma" in params:
            self.gamma_edit.setText(str(params["gamma"]))
        if "kernel" in params:
            self._set_combo(self.kernel_combo, params["kernel"])
        if "n_estimators" in params:
            self.n_estimators_spin.setValue(int(params["n_estimators"]))
        if "max_depth" in params:
            depth = params["max_depth"]
            self.max_depth_spin.setValue(0 if depth is None else int(depth))
        if "min_samples_split" in params:
            self.min_samples_split_spin.setValue(int(params["min_samples_split"]))
        if "n_neighbors" in params:
            self.n_neighbors_spin.setValue(int(params["n_neighbors"]))
        if "weights" in params:
            self._set_combo(self.weights_combo, params["weights"])
        if "hidden_layer_sizes" in params:
            layers = params["hidden_layer_sizes"]
            if isinstance(layers, Iterable):
                self.classic_hidden_layers_edit.setText(",".join(str(x) for x in layers))
        if "activation" in params:
            self._set_combo(self.classic_activation_combo, params["activation"])
        if "alpha" in params:
            self.alpha_spin.setValue(float(params["alpha"]))

    def apply_nn_defaults(self) -> None:
        self.nn_hidden_layers = [4, 2]
        self._set_combo(self.nn_activation_combo, "tanh")
        self.nn_lr_spin.setValue(0.03)
        self.nn_l2_spin.setValue(0.0)
        self.nn_batch_size_spin.setValue(10)
        self.nn_epochs_spin.setValue(50)
        for name, check in self.feature_checks.items():
            check.setChecked(name in {"x1", "x2"})
        self._refresh_layer_controls()
        self._refresh_network_view()
        self._reset_training()

    def _apply_result(self, result: TrainResult, view: DatasetView, metrics_box: QPlainTextEdit) -> None:
        if self.X is None or self.y is None:
            return
        view.plot(self.X, self.y, heatmap=result.heatmap)
        metrics_box.setPlainText(self._format_metrics(result.metrics))

    def _sync_nn_loss_plot(self) -> None:
        if not hasattr(self, "nn_loss_view"):
            return
        losses = getattr(self.model, "loss_history", []) if self.model is not None else []
        self.nn_loss_view.plot(list(losses))

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
        return "\n".join(lines)

    def _reset_training(self, clear_plot: bool = True) -> None:
        self.play_timer.stop()
        if hasattr(self, "nn_play_button"):
            self.nn_play_button.setText("Play")
        self.nn_target_epoch = 0
        self.model = None
        self.trainer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.nn_epoch = 0
        self._update_epoch_label()
        if hasattr(self, "nn_metrics_text"):
            self.nn_metrics_text.setPlainText("")
        if hasattr(self, "nn_loss_view"):
            self.nn_loss_view.plot([])
        if hasattr(self, "classic_metrics_text"):
            self.classic_metrics_text.setPlainText("")
        if clear_plot and self.X is not None and self.y is not None:
            self.nn_dataset_view.plot(self.X, self.y)
            self.classic_dataset_view.plot(self.X, self.y)

    def _update_epoch_label(self) -> None:
        if hasattr(self, "nn_epoch_label"):
            self.nn_epoch_label.setText(f"Epoch {self.nn_epoch}")

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)
