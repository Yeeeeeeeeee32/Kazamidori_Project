"""Text displays whose changing contents cannot widen a splitter section."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QLabel, QSizePolicy, QStyle, QStyleOption


class ElidedLabel(QLabel):
    """Keep the full text/API and tooltip; paint a single elided line."""

    def __init__(self, text='', parent=None):
        super().__init__(text, parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setToolTip(text)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def sizeHint(self):
        return QSize(120, super().sizeHint().height())

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, event):
        painter = QPainter(self)
        option = QStyleOption()
        option.initFrom(self)
        self.style().drawPrimitive(QStyle.PrimitiveElement.PE_Widget, option, painter, self)
        rect = self.contentsRect().adjusted(self.margin(), 0, -self.margin(), 0)
        text = self.fontMetrics().elidedText(' '.join(self.text().splitlines()),
                                           Qt.TextElideMode.ElideRight, max(0, rect.width()))
        self.style().drawItemText(painter, rect, self.alignment(), self.palette(),
                                  self.isEnabled(), text, self.foregroundRole())
