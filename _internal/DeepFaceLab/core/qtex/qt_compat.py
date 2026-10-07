"""Project Qt contract: PySide6 with retained editor aliases; never mix Qt bindings."""
from PySide6.QtCore import *  # noqa: F401,F403
from PySide6.QtGui import *  # noqa: F401,F403
from PySide6.QtWidgets import *  # noqa: F401,F403
QT_BINDING = 'PySide6'
QT_VERSION_STR = qVersion()
PYQT_VERSION_STR = None
pyqtSignal = Signal
pyqtSlot = Slot
pyqtProperty = Property
