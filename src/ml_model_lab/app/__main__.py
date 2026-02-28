from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ml_model_lab.app.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
