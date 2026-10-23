"""Explicit, independent selection of an airframe design and its supplement."""
from pathlib import Path

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                              QLineEdit, QPushButton, QLabel, QFileDialog,
                              QDialogButtonBox)


class AirframeLoadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Load Airframe')
        self.resize(660, 230)
        layout = QVBoxLayout(self)
        note = QLabel('機体の.orkと補完JSONをそれぞれ選択してください。ファイル名・保存場所は自由です。\n'
                      '.ork単体、または全パラメータを持つ従来のJSON単体も読み込めます。')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        layout.addLayout(form)
        self.ork_path = QLineEdit()
        self.json_path = QLineEdit()
        self.ork_browse = QPushButton('Select .ork…')
        self.json_browse = QPushButton('Select JSON…')
        for label, field, button in (('OpenRocket:', self.ork_path, self.ork_browse),
                                     ('補完JSON:', self.json_path, self.json_browse)):
            field.setClearButtonEnabled(True)
            row = QHBoxLayout()
            row.addWidget(field, 1)
            row.addWidget(button)
            form.addRow(label, row)
        self.ork_browse.clicked.connect(lambda: self._browse(self.ork_path, 'OpenRocket (*.ork *.ork.gz)'))
        self.json_browse.clicked.connect(lambda: self._browse(self.json_path, 'Airframe JSON (*.json)'))
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
        ork, supplement = self.ork_path.text().strip(), self.json_path.text().strip()
        if not ork and not supplement:
            self.error_label.setText('読み込むファイルを選択してください。')
            return
        if ork and not ork.lower().endswith(('.ork', '.ork.gz')):
            self.error_label.setText('OpenRocket欄には.orkまたは.ork.gzを選択してください。')
            return
        if supplement and not supplement.lower().endswith('.json'):
            self.error_label.setText('補完JSON欄には.jsonを選択してください。')
            return
        super().accept()


def select_airframe_files(parent):
    dialog = AirframeLoadDialog(parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.ork_path.text().strip(), dialog.json_path.text().strip()
