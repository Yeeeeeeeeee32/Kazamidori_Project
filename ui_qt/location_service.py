"""One-shot OS location requests; no coordinates are invented on failure."""
import math

from PySide6.QtCore import QObject, QCoreApplication, QLocationPermission, QTimer, Qt, Signal
from PySide6.QtPositioning import QGeoPositionInfo, QGeoPositionInfoSource
from utils.map_downloader import validate_map_coordinates


class LocationService(QObject):
    position_ready = Signal(float, float, float)
    error = Signal(str)
    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = None
        self._busy = False
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(lambda: self._fail(
            "Location request timed out. Check Windows location services or enter coordinates manually."))

    def request(self):
        if self._busy:
            return
        self._busy = True
        self.busy_changed.emit(True)
        self._timeout.start(35000)
        permission = QLocationPermission()
        permission.setAccuracy(QLocationPermission.Accuracy.Precise)
        permission.setAvailability(QLocationPermission.Availability.WhenInUse)
        app = QCoreApplication.instance()
        status = app.checkPermission(permission)
        if status == Qt.PermissionStatus.Undetermined:
            app.requestPermission(permission, self, self._permission_result)
        elif status == Qt.PermissionStatus.Denied:
            self._permission_denied()
        else:
            self._start_request()

    def _permission_result(self, permission):
        if not self._busy:
            return
        if permission.status() == Qt.PermissionStatus.Granted:
            self._start_request()
        else:
            self._permission_denied()

    def _permission_denied(self):
        self._fail("Location access denied. Enable Location services and desktop app access in Windows Settings > Privacy & security > Location.")

    def _start_request(self):
        if not self._busy:
            return
        if self._source is None:
            self._source = QGeoPositionInfoSource.createDefaultSource(self)
            if self._source is None:
                self._fail("No location provider is available. Check Windows location services or enter coordinates manually.")
                return
            self._source.positionUpdated.connect(self._on_position)
            self._source.errorOccurred.connect(self._on_error)
            self._source.setPreferredPositioningMethods(
                QGeoPositionInfoSource.PositioningMethod.AllPositioningMethods)
        self._source.requestUpdate(30000)

    def _on_position(self, position):
        if not self._busy:
            return
        try:
            if not position.isValid():
                raise ValueError("The location provider returned an invalid position.")
            coordinate = position.coordinate()
            lat, lon = validate_map_coordinates(coordinate.latitude(), coordinate.longitude())
            accuracy = (float(position.attribute(QGeoPositionInfo.Attribute.HorizontalAccuracy))
                        if position.hasAttribute(QGeoPositionInfo.Attribute.HorizontalAccuracy) else -1.0)
            if not math.isfinite(accuracy) or accuracy < 0:
                accuracy = -1.0
        except (ValueError, TypeError) as exc:
            self._fail(str(exc))
            return
        self._finish()
        self.position_ready.emit(lat, lon, accuracy)

    def _on_error(self, error):
        if not self._busy or error == QGeoPositionInfoSource.Error.NoError:
            return
        if error == QGeoPositionInfoSource.Error.AccessError:
            self._permission_denied()
        elif error == QGeoPositionInfoSource.Error.UpdateTimeoutError:
            self._fail("Location request timed out. Check Windows location services or enter coordinates manually.")
        else:
            self._fail("Location services are unavailable. Check Windows location settings or enter coordinates manually.")

    def _finish(self):
        self._busy = False
        self._timeout.stop()
        if self._source is not None:
            source, self._source = self._source, None
            source.stopUpdates()
            source.deleteLater()
        self.busy_changed.emit(False)

    def _fail(self, message):
        if self._busy:
            self._finish()
            self.error.emit(message)
