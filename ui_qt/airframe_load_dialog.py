"""Explicit selection of one OpenRocket design; no Airframe JSON input."""
from pathlib import Path

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                              QLineEdit, QPushButton, QLabel, QFileDialog,
                              QDialogButtonBox)


class AirframeLoadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Load Airframe')
        self.resize(660, 190)
        layout = QVBoxLayout(self)
        note = QLabel('機体の.orkファイルを選択してください。補完JSONは不要です。\n'
                      '取得できない機体項目はManual Configで入力してください。')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        layout.addLayout(form)
        self.ork_path = QLineEdit()
        self.ork_browse = QPushButton('Select .ork…')
        for label, field, button in (('OpenRocket:', self.ork_path, self.ork_browse),):
            field.setClearButtonEnabled(True)
            row = QHBoxLayout()
            row.addWidget(field, 1)
            row.addWidget(button)
            form.addRow(label, row)
        self.ork_browse.clicked.connect(lambda: self._browse(self.ork_path, 'OpenRocket (*.ork *.ork.gz)'))
        self.error_label = QLabel()
        self.error_label.setStyleSheet('color: #f38ba8;')
        layout.addWidget(self.error_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText('Load')
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _browse(self, field, file_filter):
        path, _ = QFileDialog.getOpenFileName(self, 'Select file', field.text() or str(Path.cwd()), file_filter)
        if path:
            field.setText(path)

    def accept(self):
        ork = self.ork_path.text().strip()
        if not ork:
            self.error_label.setText('読み込むファイルを選択してください。')
            return
        if ork and not ork.lower().endswith(('.ork', '.ork.gz')):
            self.error_label.setText('OpenRocket欄には.orkまたは.ork.gzを選択してください。')
            return
        super().accept()


def select_airframe_files(parent):
    dialog = AirframeLoadDialog(parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    # Preserve the internal tuple contract for legacy callers.
    return dialog.ork_path.text().strip(), ''
