"""Exercise saved maps through the actual Qt view, controls, and worker."""
import contextlib
import io
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kazamidori-mpl-test"))

from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest, QSignalSpy
from PySide6.QtWidgets import QApplication
from main_qt import DEFAULT_CONFIG
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.sim_controller import SimController
from utils import map_downloader as maps
from utils.geo_math import offset_to_latlon


class TestOfflineMapUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(maps, "OUT_DIR", self.root))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(warnings.catch_warnings())
        warnings.filterwarnings("ignore", message="libpyside: Failed to disconnect")
        # No test can accidentally fetch real tiles, including startup/edits.
        self.network = stack.enter_context(patch.object(maps.requests.Session, "get", side_effect=AssertionError("Network used")))
        self.meta = {
            "center_lat": 35.422158, "center_lon": 139.422688,
            "extent_meters": [-1000, 1000, -1000, 1000],
            "image_size_px": [16, 16], "magnetic_declination": -7.5,
            "declination_model": "WMM", "manual_offset": 0,
        }
        self.save_map("red")
        self.state = AppState(DEFAULT_CONFIG)
        self.window = AppWindow(self.state)
        self.controller = SimController(self.window, self.state)
        self.controller._wind_timer.stop()
        self.view = self.window.map_view

    def tearDown(self):
        if self.controller._worker is not None:
            self.controller._worker.wait(5000)
        self.controller._tray_icon.hide()
        self.window.close()
        self.controller.deleteLater()
        self.window.deleteLater()
        self.app.processEvents()

    def save_map(self, color):
        (self.root / "map_meta.json").write_text(json.dumps(self.meta), encoding="utf-8")
        Image.new("RGB", (16, 16), color).save(self.root / "background.png")

    def wait_for_worker(self):
        for _ in range(100):
            self.app.processEvents()
            if self.controller._worker is None:
                return
            QTest.qWait(20)
        self.fail("Map worker did not finish")

    def test_startup_and_cache_hit_work_without_network(self):
        self.assertEqual(len(self.view.ax.images), 1)
        self.view.ax.clear()
        with patch("ui_qt.sim_controller.MapDownloadWorker") as worker:
            self.window.btn_download_map.click()
            worker.assert_not_called()
        self.assertEqual(len(self.view.ax.images), 1)
        self.assertIn("Saved offline map loaded", self.window._status_label.text())
        self.network.assert_not_called()

    def test_replacement_at_same_path_is_visible_and_preserves_simulation(self):
        self.state.simulation_result = {"impact_x": 10, "impact_y": 20}
        before = self.state.simulation_result
        self.save_map("blue")
        self.controller._on_download_map_finished(self.meta)
        self.assertEqual(list(self.view.ax.images[0].get_array()[0, 0]), [0, 0, 255, 255])
        self.assertIs(self.state.simulation_result, before)
        self.assertEqual(self.view._last_result["impact_x"], 10)

    def test_coordinate_edit_moves_background_without_downloading(self):
        lat, lon = offset_to_latlon(self.meta["center_lat"], self.meta["center_lon"], 200, 0)
        with patch("ui_qt.sim_controller.MapDownloadWorker") as worker:
            self.window.lon_input.setValue(lon)
            worker.assert_not_called()
        extent = self.view.ax.images[0].get_extent()
        self.assertAlmostEqual((extent[0] + extent[1]) / 2, -200, delta=0.1)
        self.network.assert_not_called()

    def test_outside_saved_area_does_not_show_wrong_background(self):
        self.window.lat_input.setValue(36.5)
        self.assertEqual(len(self.view.ax.images), 0)
        self.assertTrue(any("OUTSIDE" in t.get_text() for t in self.view.ax.texts))
        self.network.assert_not_called()

    def test_missing_map_has_actionable_message(self):
        (self.root / "background.png").unlink()
        self.view.request_render(True)
        self.assertEqual(len(self.view.ax.images), 0)
        self.assertTrue(any("Download Offline Map" in t.get_text() for t in self.view.ax.texts))

    def test_redraw_does_not_reset_manual_declination(self):
        self.state.manual_declination_offset = 2.5
        self.view.request_render()
        self.view.request_render(True)
        self.assertEqual(self.state.manual_declination_offset, 2.5)

    def test_failed_worker_can_be_retried_and_retains_saved_map(self):
        self.window.lat_input.setValue(35.6)
        with patch.object(maps, "download_offline_map", side_effect=RuntimeError("offline")) as download:
            for _ in range(2):
                self.window.btn_download_map.click()
                self.wait_for_worker()
                self.assertTrue(self.window.btn_download_map.isEnabled())
                self.assertFalse(self.state.is_calculating)
                self.assertIn("Map download failed", self.window._status_label.text())
            self.assertEqual(download.call_count, 2)
        self.assertEqual(maps.read_offline_map()[1].getpixel((0, 0)), (255, 0, 0, 255))

    def test_corrupt_cache_triggers_download_and_zero_coordinates_are_valid(self):
        (self.root / "background.png").write_bytes(b"broken")
        with patch.object(maps, "download_offline_map", side_effect=RuntimeError("offline")) as download:
            self.window.btn_download_map.click()
            self.wait_for_worker()
            self.assertEqual(download.call_count, 1)
            self.window.lat_input.setValue(0)
            self.window.lon_input.setValue(0)
            self.window.btn_download_map.click()
            self.wait_for_worker()
            download.assert_called_with(0, 0)

    def canvas_point(self, east, north):
        self.window.show()
        self.app.processEvents()
        self.view.canvas.draw()
        x, y = self.view.ax.transData.transform((east, north))
        point = QPoint(round(x), round(self.view.canvas.height() - y))
        # Use the actual rounded screen pixel to account for display resolution.
        actual_east, actual_north = self.view.ax.transData.inverted().transform(
            (point.x(), self.view.canvas.height() - point.y()))
        expected = offset_to_latlon(self.state.launch_lat, self.state.launch_lon,
                                    actual_east, actual_north)
        return point, expected

    def assert_selected_coordinates(self, expected):
        self.assertAlmostEqual(self.state.launch_lat, expected[0], places=10)
        self.assertAlmostEqual(self.state.launch_lon, expected[1], places=10)
        self.assertAlmostEqual(self.window.lat_input.value(), expected[0], places=6)
        self.assertAlmostEqual(self.window.lon_input.value(), expected[1], places=6)
        self.assertIn(f"{expected[0]:.6f}", self.window._map_launch_lbl.text())
        self.assertIn(f"{expected[1]:.6f}", self.window._map_launch_lbl.text())
        self.network.assert_not_called()

    def test_qt_ctrl_click_selects_coordinates_with_mouse_modifiers(self):
        point, expected = self.canvas_point(200, 120)
        picked = QSignalSpy(self.view.coordinates_picked)
        with patch.object(self.controller, "_on_run_clicked") as run:
            QTest.mouseClick(self.view.canvas, Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.ControlModifier, point)
        self.assertEqual(picked.count(), 1)
        self.assert_selected_coordinates(expected)
        run.assert_not_called()
        self.assertIn("Launch coordinates selected", self.window._status_label.text())

    def test_qt_ctrl_key_then_click_and_repeated_picks_use_current_origin(self):
        picked = QSignalSpy(self.view.coordinates_picked)
        for east, north in ((100, 75), (-60, 40)):
            point, expected = self.canvas_point(east, north)
            self.view.canvas.setFocus()
            QTest.keyPress(self.view.canvas, Qt.Key.Key_Control)
            QTest.mouseClick(self.view.canvas, Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.ControlModifier, point)
            QTest.keyRelease(self.view.canvas, Qt.Key.Key_Control)
            self.assert_selected_coordinates(expected)
        self.assertEqual(picked.count(), 2)
        self.assertIsNone(self.view._ghost_marker)

    def test_plain_drag_pans_without_selecting_coordinates(self):
        point, _ = self.canvas_point(0, 0)
        original = (self.state.launch_lat, self.state.launch_lon)
        limits = self.view.ax.get_xlim()
        picked = QSignalSpy(self.view.coordinates_picked)
        QTest.mousePress(self.view.canvas, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, point)
        QTest.mouseMove(self.view.canvas, point + QPoint(40, 20))
        QTest.mouseRelease(self.view.canvas, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, point + QPoint(40, 20))
        self.assertEqual(picked.count(), 0)
        self.assertEqual(original, (self.state.launch_lat, self.state.launch_lon))
        self.assertNotEqual(limits, self.view.ax.get_xlim())

    def test_ctrl_release_outside_map_cancels_selection(self):
        point, _ = self.canvas_point(0, 0)
        original = (self.state.launch_lat, self.state.launch_lon)
        picked = QSignalSpy(self.view.coordinates_picked)
        QTest.mousePress(self.view.canvas, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.ControlModifier, point)
        QTest.mouseRelease(self.view.canvas, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.ControlModifier, QPoint(2, 2))
        self.assertEqual(picked.count(), 0)
        self.assertEqual(original, (self.state.launch_lat, self.state.launch_lon))
        self.assertFalse(self.view._is_dragging)

    def test_missing_map_cannot_select_coordinates_from_blank_grid(self):
        (self.root / "background.png").unlink()
        self.view.request_render(True)
        point, _ = self.canvas_point(0, 0)
        picked = QSignalSpy(self.view.coordinates_picked)
        QTest.mouseClick(self.view.canvas, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.ControlModifier, point)
        self.assertEqual(picked.count(), 0)

    def test_current_location_button_requests_system_location(self):
        before = (self.state.launch_lat, self.state.launch_lon)
        with patch.object(self.controller._location_service, "request") as request:
            self.window.btn_get_location.click()
        request.assert_called_once()
        self.assertEqual(before, (self.state.launch_lat, self.state.launch_lon))
        self.assertIn("Acquiring current location", self.window.location_label.text())

    def test_current_location_success_updates_coordinates_and_requests_correct_map(self):
        with patch.object(self.controller, "_on_download_map_clicked") as download:
            self.controller._location_service.position_ready.emit(35.706, 139.707, 25)
        self.assert_selected_coordinates((35.706, 139.707))
        self.assertIn("±25 m", self.window.location_label.text())
        self.assertIn("Current location:", self.window.location_label.text())
        self.assertEqual(len(self.view.ax.images), 0)
        download.assert_called_once()

    def test_current_location_failure_preserves_inputs_and_does_not_download(self):
        before = (self.state.launch_lat, self.state.launch_lon)
        self.controller._location_service.busy_changed.emit(True)
        self.assertFalse(self.window.btn_get_location.isEnabled())
        with patch.object(self.controller, "_on_download_map_clicked") as download:
            self.controller._location_service.busy_changed.emit(False)
            self.controller._location_service.error.emit("Location access denied")
        self.assertTrue(self.window.btn_get_location.isEnabled())
        self.assertEqual(before, (self.state.launch_lat, self.state.launch_lon))
        self.assertIn("failed", self.window.location_label.text())
        download.assert_not_called()

    def test_startup_restores_saved_center_and_identifies_it_as_saved(self):
        self.assertIn("Saved map center", self.window.location_label.text())
        self.assertIn("not been acquired", self.window.location_label.text())
        self.meta["center_lat"], self.meta["center_lon"] = 35.706, 139.707
        self.save_map("blue")
        restored = AppWindow(AppState(DEFAULT_CONFIG))
        self.addCleanup(restored.deleteLater)
        self.assertEqual(restored.lat_input.value(), 35.706)
        self.assertEqual(restored.lon_input.value(), 139.707)
        restored.close()


if __name__ == "__main__":
    unittest.main()
