"""Weather acquisition and CSV selection, with all file/network work off the UI thread."""
from pathlib import Path

from PySide6.QtCore import QDateTime, QThread, QTimeZone, Signal
from PySide6.QtWidgets import (QDialog, QFormLayout, QDateTimeEdit, QPushButton,
                              QDialogButtonBox, QFileDialog, QLabel, QMessageBox, QCheckBox)

from utils.wind_data import fetch_wind_csv, read_wind_snapshot
from utils.time_display import format_jst

JST_ZONE = QTimeZone(9 * 3600)


class WindDataWorker(QThread):
    ready = Signal(str)
    error = Signal(str)
    status = Signal(str)

    def __init__(self, path, options, fetch=False, parent=None):
        super().__init__(parent)
        self.path, self.options, self.fetch = path, dict(options), fetch

    def run(self):
        try:
            p = self.options
            if self.fetch:
                fetch_wind_csv(self.path, p['latitude'], p['longitude'], p['start'], p['end'],
                               cancelled=self.isInterruptionRequested, progress=self.status.emit)
            if not p.get('download_only', False):
                read_wind_snapshot(self.path, p['target'], p['pressure'], p['temperature'],
                                   p['latitude'], p['longitude'], cancelled=self.isInterruptionRequested)
            self.ready.emit(self.path)
        except Exception as exc:
            self.error.emit(str(exc))


class WindDataDialog(QDialog):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state, self.worker = state, None
        self.setWindowTitle('Wind Data — Open-Meteo / CSV')
        form = QFormLayout(self)
        note = QLabel('日時は JST（日本標準時）。風向は真北からの FROM 方向、風速は m/s。\n'
                      '気圧面の高度は実行時の現地気圧で補正します。\n'
                      '提供: Open-Meteo (CC BY 4.0) — https://open-meteo.com/')
        note.setWordWrap(True)
        form.addRow(note)
        now = QDateTime.currentDateTimeUtc().toTimeZone(JST_ZONE)
        now = now.addSecs(-now.time().minute() * 60 - now.time().second())
        self.start, self.end, self.target = QDateTimeEdit(now), QDateTimeEdit(now.addSecs(86400)), QDateTimeEdit(now)
        for edit in (self.start, self.end, self.target):
            edit.setTimeZone(JST_ZONE)
            edit.setDisplayFormat("yyyy-MM-dd HH:00 'JST'")
            edit.setCalendarPopup(True)
        form.addRow('Download Start (JST)', self.start)
        form.addRow('Download End (JST)', self.end)
        form.addRow('Simulation time (JST)', self.target)
        self.apply_download = QCheckBox('Use downloaded data for this simulation')
        self.apply_download.setChecked(False)
        form.addRow(self.apply_download)
        saved = state.phase4_options['wind_time_utc']
        if saved:
            from PySide6.QtCore import Qt
            dt = QDateTime.fromString(saved, Qt.DateFormat.ISODate)
            if dt.isValid():
                self.target.setDateTime(dt.toTimeZone(JST_ZONE))
        self.path_label = QLabel(state.phase4_options['wind_csv'] or 'Manual wind profile')
        self.path_label.setWordWrap(True)
        form.addRow(self.path_label)
        self.fetch_btn, self.load_btn = QPushButton('Download Specified Period…'), QPushButton('Load Wind CSV…')
        self.use_btn, self.clear_btn = QPushButton('Apply target time'), QPushButton('Use Manual Wind')
        for button in (self.fetch_btn, self.load_btn, self.use_btn, self.clear_btn):
            form.addRow(button)
        self.fetch_btn.clicked.connect(lambda: self._choose(True))
        self.load_btn.clicked.connect(lambda: self._choose(False))
        self.use_btn.clicked.connect(self._apply_time)
        self.clear_btn.clicked.connect(self._clear)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @staticmethod
    def _iso(edit):
        # Convert the explicitly selected JST instant to the API/cache convention.
        return edit.dateTime().toUTC().toString('yyyy-MM-ddTHH:00:00') + 'Z'

    def _choose(self, fetch):
        choose = QFileDialog.getSaveFileName if fetch else QFileDialog.getOpenFileName
        path, _ = choose(self, 'Wind CSV', 'wind_cache.csv' if fetch else '', 'CSV (*.csv)')
        if path:
            self._run(path, fetch)

    def _apply_time(self):
        path = self.state.phase4_options['wind_csv']
        if path:
            self._run(path, False)

    def _clear(self):
        self.state.phase4_options = dict(wind_csv='', wind_time_utc='')
        self.path_label.setText('Manual wind profile')

    def _run(self, path, fetch):
        if self.worker and self.worker.isRunning():
            return
        target = self._iso(self.target)
        p = dict(latitude=self.state.launch_lat, longitude=self.state.launch_lon,
                 pressure=self.state.env_pressure, temperature=self.state.env_temp,
                 start=self._iso(self.start), end=self._iso(self.end), target=target,
                 download_only=fetch and not self.apply_download.isChecked())
        self.worker = WindDataWorker(path, p, fetch, self)
        def ready(saved):
            if not p['download_only']:
                self.state.phase4_options = dict(wind_csv=str(Path(saved).resolve()), wind_time_utc=target)
                self.path_label.setText(f'{Path(saved).name} — {format_jst(target)}')
            else:
                self.path_label.setText(f"Downloaded: {Path(saved).name}\n{format_jst(p['start'])} — {format_jst(p['end'])}")
        self.worker.ready.connect(ready)
        self.worker.error.connect(lambda msg: QMessageBox.warning(self, 'Wind data error', msg))
        self.worker.status.connect(lambda stamp: self.path_label.setText('Downloading: ' + format_jst(stamp)))
        self.worker.finished.connect(lambda: self._busy(False))
        self._busy(True)
        self.worker.start()

    def _busy(self, busy):
        for widget in (self.fetch_btn, self.load_btn, self.use_btn, self.clear_btn, self.start, self.end, self.target, self.apply_download):
            widget.setEnabled(not busy)

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.path_label.setText('Cancelling weather operation…')
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
