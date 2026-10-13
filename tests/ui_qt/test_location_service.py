"""Real Qt signals around OS positioning, without requesting a user's location."""
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal, QDateTime, Qt
from PySide6.QtPositioning import QGeoCoordinate, QGeoPositionInfo, QGeoPositionInfoSource
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication
from ui_qt.location_service import LocationService


class FakeSource(QObject):
    positionUpdated = Signal(object)
    errorOccurred = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)
        self.requestUpdate = Mock()
        self.stopUpdates = Mock()
        self.setPreferredPositioningMethods = Mock()


class TestLocationService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.service = LocationService()
        self.positions = QSignalSpy(self.service.position_ready)
        self.errors = QSignalSpy(self.service.error)
        self.busy = QSignalSpy(self.service.busy_changed)
        self.permission_app = Mock()
        self.permission_app.checkPermission.return_value = Qt.PermissionStatus.Granted
        permission = patch("ui_qt.location_service.QCoreApplication.instance", return_value=self.permission_app)
        permission.start()
        self.addCleanup(permission.stop)
        self.sources = []

        def create(parent):
            source = FakeSource(parent)
            self.sources.append(source)
            return source

        factory = patch("ui_qt.location_service.QGeoPositionInfoSource.createDefaultSource", side_effect=create)
        self.factory = factory.start()
        self.addCleanup(factory.stop)
        self.addCleanup(self.service.deleteLater)
        self.addCleanup(self.service._timeout.stop)

    def test_position_is_requested_and_delivered_with_accuracy(self):
        self.service.request()
        self.sources[0].requestUpdate.assert_called_once_with(30000)
        position = QGeoPositionInfo(QGeoCoordinate(35.706, 139.707), QDateTime.currentDateTimeUtc())
        position.setAttribute(QGeoPositionInfo.Attribute.HorizontalAccuracy, 25)
        self.sources[0].positionUpdated.emit(position)
        self.assertEqual(self.positions.at(0), [35.706, 139.707, 25])
        self.assertEqual(self.busy.at(0), [True])
        self.assertEqual(self.busy.at(1), [False])
        self.assertFalse(self.service._timeout.isActive())

    def test_access_error_is_reported_and_retry_uses_new_source(self):
        self.service.request()
        self.sources[0].errorOccurred.emit(QGeoPositionInfoSource.Error.AccessError)
        self.assertEqual(self.positions.count(), 0)
        self.assertIn("Windows Settings", self.errors.at(0)[0])
        self.service.request()
        self.assertEqual(len(self.sources), 2)
        self.sources[1].positionUpdated.emit(QGeoPositionInfo(
            QGeoCoordinate(35.706, 139.707), QDateTime.currentDateTimeUtc()))
        self.assertEqual(self.positions.at(0)[2], -1)

    def test_denied_permission_does_not_request_a_position(self):
        self.permission_app.checkPermission.return_value = Qt.PermissionStatus.Denied
        self.service.request()
        self.factory.assert_not_called()
        self.assertEqual(self.errors.count(), 1)
        self.assertFalse(self.service._busy)

    def test_permission_is_requested_when_undetermined(self):
        self.permission_app.checkPermission.return_value = Qt.PermissionStatus.Undetermined
        self.service.request()
        self.factory.assert_not_called()
        permission = Mock()
        permission.status.return_value = Qt.PermissionStatus.Granted
        callback = self.permission_app.requestPermission.call_args.args[2]
        callback(permission)
        self.assertEqual(len(self.sources), 1)

    def test_no_provider_and_timeout_do_not_emit_placeholder_coordinates(self):
        self.factory.side_effect = None
        self.factory.return_value = None
        self.service.request()
        self.assertEqual(self.positions.count(), 0)
        self.assertIn("No location provider", self.errors.at(0)[0])
        self.factory.side_effect = lambda parent: FakeSource(parent)
        self.service.request()
        self.service._timeout.timeout.emit()
        self.assertEqual(self.positions.count(), 0)
        self.assertIn("timed out", self.errors.at(1)[0])
        self.assertFalse(self.service._busy)

    def test_invalid_position_and_duplicate_clicks_are_handled(self):
        self.service.request()
        self.service.request()
        self.assertEqual(len(self.sources), 1)
        self.sources[0].positionUpdated.emit(QGeoPositionInfo())
        self.assertEqual(self.positions.count(), 0)
        self.assertEqual(self.errors.count(), 1)


if __name__ == "__main__":
    unittest.main()
