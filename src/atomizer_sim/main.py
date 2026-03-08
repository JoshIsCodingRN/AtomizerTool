from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .ui import AtomizerWindow


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = AtomizerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())