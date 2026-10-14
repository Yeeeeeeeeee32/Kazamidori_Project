"""Normal/optimization workers must preserve causes and make them readable."""
import contextlib
import io
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kazamidori-mpl-test"))

from PySide6.QtWidgets import QApplication
from main_qt import DEFAULT_CONFIG
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.sim_controller import SimController
from ui_qt.workers import OptimizationWorker, SimulationWorker
from utils import map_downloader as maps


class TestSimulationErrors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.params = json.loads((Path(__file__).parents[1] / "fixtures" / "simulation_params.json").read_text(encoding="utf-8"))
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        stack.enter_context(patch("ui_qt.workers.open", mock_open()))
        stack.enter_context(warnings.catch_warnings())
        warnings.filterwarnings("ignore", message="libpyside: Failed to disconnect")

    def test_both_workers_report_stability_details_without_traceback(self):
        for worker_type in (SimulationWorker, OptimizationWorker):
            with self.subTest(worker=worker_type.__name__):
                worker = worker_type(self.params)
                errors, results = [], []
                worker.error.connect(errors.append)
                worker.sig_finished.connect(results.append)
                worker.run()
                self.assertEqual(len(errors), 1)
                self.assertIn("Unstable Rocket", errors[0])
                self.assertIn("CG with motor/propellant", errors[0])
                self.assertIn("Isp = 80 s", errors[0])
                self.assertNotIn("Traceback", errors[0])
                self.assertEqual(results, [])

    def test_unexpected_failure_keeps_traceback_for_diagnosis(self):
        for worker_type in (SimulationWorker, OptimizationWorker):
            with self.subTest(worker=worker_type.__name__):
                worker = worker_type(self.params)
                errors = []
                worker.error.connect(errors.append)
                with patch.object(SimulationWorker, "_build_wind_profiles", side_effect=RuntimeError("unexpected test failure")):
                    worker.run()
                self.assertIn("Traceback", errors[0])
                self.assertIn("unexpected test failure", errors[0])

    def test_error_dialog_keeps_details_and_reenables_controls(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(maps, "OUT_DIR", Path(directory)):
            state = AppState(DEFAULT_CONFIG)
            window = AppWindow(state)
            controller = SimController(window, state)
            try:
                state.is_calculating = True
                state.mc_running = True
                state.phase2_active = True
                controller._set_run_buttons_enabled(False)
                window._motor_thrust_data = self.params["thrust_data"]
                window.motor_label.setText("Estes_G40.csv")
                controller._tray_icon.showMessage = Mock()
                message = "Nominal simulation failed:\nUnstable Rocket: CG=0.3154 m; CP=0.3056 m."
                controller._on_error(message)
                self.assertIn("不適合", controller._error_dialog.text())
                self.assertIn("不安定", controller._error_dialog.text())
                self.assertIn("Estes_G40.csv", controller._error_dialog.informativeText())
                self.assertIn("機体に適合するモーター", controller._error_dialog.informativeText())
                self.assertIn("CG=0.3154", controller._error_dialog.detailedText())
                self.assertIn("NO-GO", window._go_nogo_label.text())
                self.assertIn("不適合", window._status_label.text())
                self.assertIn("不適合", controller._tray_icon.showMessage.call_args.args[0])
                self.assertIn("不安定", controller._tray_icon.showMessage.call_args.args[1])
                self.assertTrue(controller._error_dialog.isVisible())
                self.assertFalse(state.is_calculating)
                self.assertFalse(state.mc_running)
                self.assertFalse(state.phase2_active)
                self.assertTrue(window.action_run.isEnabled())
                self.assertFalse(window.action_stop.isEnabled())
                self.assertTrue(controller._wind_timer.isActive())
                self.assertNotIn("CG=", window._status_label.text())
                controller._on_error("Traceback (most recent call last):\nRuntimeError: test error")
                self.assertIn("実行できませんでした", controller._error_dialog.text())
                self.assertIn("test error", controller._error_dialog.informativeText())
                self.assertIn("Traceback", controller._error_dialog.detailedText())
                self.assertNotIn("不適合", controller._error_dialog.text())
                controller._on_error("Simulation failed for all candidates.\nBad wind data")
                self.assertIn("Bad wind data", controller._error_dialog.informativeText())
                self.assertNotIn("不安定", controller._error_dialog.text())
            finally:
                controller._wind_timer.stop()
                controller._error_dialog.close()
                controller._tray_icon.hide()
                window.close()
                controller.deleteLater()
                window.deleteLater()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
