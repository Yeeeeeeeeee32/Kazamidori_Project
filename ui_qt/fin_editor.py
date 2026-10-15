"""Staged polygon editor for a single fin's exposed outline."""
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                              QDialogButtonBox, QTableWidget, QTableWidgetItem)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from core.fin_geometry import validate_fin_points


class FinOutlineDialog(QDialog):
    def __init__(self, points, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Edit Fin Outline')
        self.resize(620, 680)
        layout = QVBoxLayout(self)
        note = QLabel('単位: m。最初の点 (0,0) は付け根の前縁。xは尾部方向、yは胴体から外側。\n'
                      '輪郭を順に入力し、最後の点は付け根の後縁 (x>0,0)。始点への閉じ線は自動です。')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(len(points), 2)
        self.table.setHorizontalHeaderLabels(['x aft [m]', 'y outward [m]'])
        for row, point in enumerate(points):
            for col, value in enumerate(point):
                self.table.setItem(row, col, QTableWidgetItem(format(value, '.10g')))
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        add, remove = QPushButton('Add Vertex'), QPushButton('Remove Selected')
        buttons.addWidget(add)
        buttons.addWidget(remove)
        layout.addLayout(buttons)
        add.clicked.connect(self._add)
        remove.clicked.connect(self._remove)
        self.fig = Figure(figsize=(5, 2.5), facecolor='#1e1e1e')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.canvas)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet('color: #f38ba8;')
        layout.addWidget(self.error_label)
        controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        controls.accepted.connect(self.accept)
        controls.rejected.connect(self.reject)
        layout.addWidget(controls)
        self.table.cellChanged.connect(self._preview)
        self.points = None
        self._preview()

    def _read(self):
        try:
            raw = [[self.table.item(r,c).text() for c in range(2)] for r in range(self.table.rowCount())]
        except AttributeError:
            raise ValueError('すべての頂点にxとyを入力してください。') from None
        return validate_fin_points(raw)

    def _preview(self, *_):
        self.ax.clear()
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='#aaaaaa')
        self.ax.set_xlabel('x aft [m]', color='#aaaaaa')
        self.ax.set_ylabel('y outward [m]', color='#aaaaaa')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axhline(0, color='#aaaaaa', linestyle='--', linewidth=.8)
        try:
            points = self._read()
            closed = points + points[:1]
            self.ax.fill([p[0] for p in closed], [p[1] for p in closed], color='#89b4fa', alpha=.35)
            self.ax.plot([p[0] for p in closed], [p[1] for p in closed], 'o-', color='#89b4fa', markersize=4)
            self.error_label.clear()
        except ValueError as exc:
            self.error_label.setText(str(exc))
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _add(self):
        count = self.table.rowCount()
        row = count if count < 2 else count-1
        with QSignalBlocker(self.table):
            try:
                a, b = self._read()[row-1:row+1]
                point = [(a[i]+b[i])/2 for i in (0,1)]
            except (ValueError, IndexError):
                point = [0.,0.] if count == 0 else ([.04,0.] if count == 1 else [0.,.01])
            self.table.insertRow(row)
            for col,value in enumerate(point):
                self.table.setItem(row,col,QTableWidgetItem(format(value,'.10g')))
        self._preview()

    def _remove(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        with QSignalBlocker(self.table):
            for row in rows:
                self.table.removeRow(row)
        self._preview()

    def accept(self):
        try:
            self.points = self._read()
        except ValueError as exc:
            self.error_label.setText(str(exc))
            return
        super().accept()
