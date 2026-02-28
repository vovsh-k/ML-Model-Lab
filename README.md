# ML Model Lab (Playground)

Desktop ML playground with PySide6 + matplotlib. Generates 2D synthetic datasets and trains multiple models.

Features
- Datasets: Circle, XOR, Gaussian, Linear, Spiral, Checkerboard
- Models: Logistic Regression, SVM, Random Forest, KNN, MLPClassifier, Keras MLP (optional)
- Metrics: accuracy, precision, recall, F1, ROC-AUC
- Decision boundary heatmap
- Training controls: Train, Step, Play/Pause, Reset

Install and run
1. `cd "/Users/vvshk/ml/ML Model Lab"`
2. `python -m venv .venv`
3. `source .venv/bin/activate`
4. `pip install -U pip`
5. `pip install -e .`
6. `ml-model-lab`

Keras support
- `pip install -e .[keras]`

Notes
- Step and Play are supported only for Keras MLP.
- If TensorFlow is not installed, the Keras model will not appear in the model list.
