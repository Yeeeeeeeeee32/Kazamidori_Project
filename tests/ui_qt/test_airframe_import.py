"""Flat Airframe.json loading and the real Qt input bindings."""
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

from PySide6.QtCore import QSignalBlocker, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QDialogButtonBox, QLabel, QHBoxLayout

from main_qt import DEFAULT_CONFIG
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.sim_controller import SimController
from utils.data_loader import AirframeConfigError, load_airframe_config


AIRFRAME = {
    "mass": 0.0872, "cg": 0.210123456789, "length": 0.383, "radius": 0.015,
    "nose_length": 0.08, "fin_root": 0.04, "fin_tip": 0.02,
    "fin_span": 0.03, "fin_pos": 0.35, "motor_pos": 0.38,
    "motor_dry_mass": 0.015, "backfire_delay": 4.0,
}


class AirframeFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "Airframe.json"

    def write(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8-sig")


class TestAirframeLoader(AirframeFixture):
    def test_flat_file_including_motor_values_and_utf8_bom(self):
        self.write(AIRFRAME)
        self.assertEqual(load_airframe_config(str(self.path)), AIRFRAME)

    def test_missing_motor_field_is_rejected(self):
        data = dict(AIRFRAME)
        del data["motor_pos"]
        self.write(data)
        with self.assertRaisesRegex(AirframeConfigError, "motor_pos"):
            load_airframe_config(str(self.path))

    def test_invalid_values_are_rejected(self):
        for value in (None, True, "bad", float("nan"), float("inf"), -1, 0):
            with self.subTest(value=value):
                self.write({**AIRFRAME, "mass": value})
                with self.assertRaisesRegex(AirframeConfigError, "mass"):
                    load_airframe_config(str(self.path))

    def test_zero_fin_tip_and_delay_are_valid(self):
        self.write({**AIRFRAME, "fin_tip": 0, "backfire_delay": 0})
        result = load_airframe_config(str(self.path))
        self.assertEqual(result["fin_tip"], 0)
        self.assertEqual(result["backfire_delay"], 0)

    def test_wrong_format_and_missing_file_are_rejected(self):
        for data in ([], {"airframe": AIRFRAME}):
            self.write(data)
            with self.assertRaises(AirframeConfigError):
                load_airframe_config(str(self.path))
        with self.assertRaises(AirframeConfigError):
            load_airframe_config(str(self.path.parent / "missing.json"))


class TestAirframeImport(AirframeFixture):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        self.write(AIRFRAME)
        self.state = AppState(DEFAULT_CONFIG)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="libpyside: Failed to disconnect")
            self.window = AppWindow(self.state)
            self.controller = SimController(self.window, self.state)
        self.controller._wind_timer.stop()

    def tearDown(self):
        self.controller._tray_icon.hide()
        self.window.close()
        self.controller.deleteLater()
        self.window.deleteLater()

    def load(self, path=None):
        with patch("ui_qt.sim_controller.QFileDialog.getOpenFileName",
                   return_value=(str(self.path) if path is None else path, "")) as dialog:
            self.window.sig_load_airframe_json_clicked.emit()
        return dialog

    def test_button_loads_all_twelve_fields(self):
        self.state.set_moi(1, 2, 3)
        buttons = [b for b in self.window.findChildren(QPushButton)
                   if b.text() in ("📂  Load Airframe", "Load airframe")]
        self.assertEqual(len(buttons), 2)
        with patch("ui_qt.sim_controller.QFileDialog.getOpenFileName",
                   return_value=(str(self.path), "")):
            buttons[-1].click()
        for key, widget in (
            ("mass", "af_mass_input"), ("cg", "af_cg_input"),
            ("length", "af_len_input"), ("radius", "af_radius_input"),
            ("nose_length", "af_nose_input"), ("fin_root", "af_finroot_input"),
            ("fin_tip", "af_fintip_input"), ("fin_span", "af_finspan_input"),
            ("fin_pos", "af_finpos_input"), ("motor_pos", "motor_cg_input"),
            ("motor_dry_mass", "motor_dry_mass_input"), ("backfire_delay", "af_backfire_input"),
        ):
            with self.subTest(field=key):
                self.assertAlmostEqual(float(getattr(self.window, widget).text()), AIRFRAME[key], places=5)
        self.assertEqual(self.state.rocket_cg, AIRFRAME["cg"])
        self.assertEqual(self.state.rocket_diameter, 2 * AIRFRAME["radius"])
        self.assertEqual(self.state.motor_cg_pos, AIRFRAME["motor_pos"])
        self.assertEqual(self.state.motor_dry_mass, AIRFRAME["motor_dry_mass"])
        self.assertEqual(self.state.backfire_delay, AIRFRAME["backfire_delay"])
        self.assertEqual(self.state.moi_roll, 0)
        self.assertIn("CG:", self.window.airframe_label.toolTip())
        self.assertNotIn("Modified", self.window.airframe_label.text())

    def test_reload_refreshes_unchanged_values_and_manual_edits_work(self):
        self.load()
        with QSignalBlocker(self.window.af_cg_input):
            self.window.af_cg_input.clear()
        self.load()
        self.assertNotEqual(self.window.af_cg_input.text(), "")
        self.window.af_cg_input.setText("0.25")
        self.assertEqual(self.state.rocket_cg, 0.25)
        self.assertIn("Modified", self.window.airframe_label.text())

    def test_parachute_is_preserved_and_reset_uses_loaded_geometry(self):
        self.state.parachute_cd = 0.8
        self.state.parachute_area = 0.126
        self.state.parachute_lag = 0.5
        self.load()
        self.assertEqual(self.state.parachute_cd, 0.8)
        self.assertEqual(self.state.parachute_area, 0.126)
        self.assertEqual(self.state.parachute_lag, 0.5)
        self.window.af_cg_input.setText("0.3")
        self.window._on_manual_config_reset()
        self.assertEqual(self.state.rocket_cg, AIRFRAME["cg"])

    def test_invalid_file_and_cancel_leave_loaded_values_unchanged(self):
        self.load()
        label = self.window.airframe_label.text()
        self.path.write_text("{broken json", encoding="utf-8")
        self.load()
        self.assertEqual(self.state.rocket_cg, AIRFRAME["cg"])
        self.assertEqual(self.window.airframe_label.text(), label)

        self.assertIn("failed", self.window._status_label.text())
        dialog = self.load(path="")
        self.assertTrue(dialog.call_args.args[2].endswith("Airframe.json"))
        self.assertEqual(self.window.airframe_label.text(), label)

    def run_manual(self, action):
        errors = []
        def perform():
            try:
                action()
            except BaseException as exc:
                errors.append(exc)
                self.window._manual_dialog.reject()
        QTimer.singleShot(0, perform)
        self.controller._on_manual_config_clicked()
        if errors:
            raise errors[0]

    def test_zero_typing_is_staged_and_apply_commits(self):
        self.load()
        def edit():
            self.window.af_fintip_input.setText("0")
            self.assertEqual(self.window.af_fintip_input.text(), "0")
            self.assertEqual(self.state.fin_tip_chord, AIRFRAME["fin_tip"])
            self.window._manual_dialog.sig_apply.emit()
        self.run_manual(edit)
        self.assertEqual(self.state.fin_tip_chord, 0)

    def test_cancel_discards_edits_and_staged_load(self):
        self.load()
        original_label = self.window.airframe_label.text()
        self.write({**AIRFRAME, "mass": 0.5, "motor_pos": 0.9})
        def edit():
            self.load()
            self.assertEqual(float(self.window.af_mass_input.text()), 0.5)
            self.assertEqual(self.state.rocket_dry_mass, AIRFRAME["mass"])
            self.window.af_cg_input.setText("0.3")
            self.window._manual_dialog.reject()
        self.run_manual(edit)
        self.assertEqual(self.state.rocket_cg, AIRFRAME["cg"])
        self.assertEqual(self.state.motor_cg_pos, AIRFRAME["motor_pos"])
        self.assertEqual(self.window.airframe_label.text(), original_label)

    def test_save_exports_draft_without_applying(self):
        self.load()
        saved = self.path.parent / "saved.json"
        def edit():
            self.window.af_cg_input.setText("0.3")
            with patch("ui_qt.sim_controller.QFileDialog.getSaveFileName", return_value=(str(saved), "")):
                self.window._manual_dialog.sig_save_json.emit()
            self.assertEqual(self.state.rocket_cg, AIRFRAME["cg"])
            self.window._manual_dialog.reject()
        self.run_manual(edit)
        values = load_airframe_config(str(saved))
        self.assertEqual(values["cg"], 0.3)
        self.assertEqual(values["motor_dry_mass"], AIRFRAME["motor_dry_mass"])

    def test_staged_load_apply_includes_motor_values(self):
        self.load()
        self.write({**AIRFRAME, "mass": 0.5, "motor_pos": 0.9, "backfire_delay": 2})
        def edit():
            self.load()
            self.assertEqual(self.state.motor_cg_pos, AIRFRAME["motor_pos"])
            self.window._manual_dialog.sig_apply.emit()
        self.run_manual(edit)
        self.assertEqual(self.state.rocket_dry_mass, 0.5)
        self.assertEqual(self.state.motor_cg_pos, 0.9)
        self.assertEqual(self.state.backfire_delay, 2)

    def test_invalid_apply_keeps_dialog_open_and_state_unchanged(self):
        self.load()
        def edit():
            self.window.af_mass_input.setText("0")
            self.window._manual_dialog.sig_apply.emit()
            self.assertTrue(self.window._manual_dialog.isVisible())
            self.assertIn("mass", self.window._manual_dialog.error_label.text())
            self.assertEqual(self.state.rocket_dry_mass, AIRFRAME["mass"])
            self.window._manual_dialog.reject()
        self.run_manual(edit)

    def test_manual_buttons_and_filename_rows(self):
        names = [button.text() for button in self.window._manual_dialog.findChildren(QPushButton)]
        self.assertCountEqual(names, ["Apply", "Cancel", "Load airframe", "Save airframe"])
        self.load()
        para = self.path.parent / "Parachute.json"
        para.write_text(json.dumps({"cd": 0.8, "area": 0.126, "lag": 0.5}), encoding="utf-8")
        with patch("ui_qt.sim_controller.QFileDialog.getOpenFileName", return_value=(str(para), "")):
            self.window.sig_load_para_json_clicked.emit()
        self.assertEqual(self.window.parachute_label.text(), "Parachute.json")
        self.assertNotIn("\n", self.window.airframe_label.text())
        for label in (self.window.parachute_label, self.window.airframe_label):
            rows = label.parentWidget().findChildren(QHBoxLayout)
            self.assertTrue(any(row.indexOf(label) == 1 and isinstance(row.itemAt(0).widget(), QPushButton) for row in rows))

    def test_wind_timer_and_readings_continue_during_optimization(self):
        self.load()
        self.state.flight_mode = "Altitude Competition"
        self.controller._wind_timer.start()
        before = len(self.state.wind_history_for_alt(3.0))
        with patch("ui_qt.sim_controller.OptimizationWorker"), \
             patch("ui_qt.sim_controller.SimulationWorker"), \
             patch.object(self.controller, "_validate_run_prerequisites", return_value=True), \
             patch.object(self.controller, "_collect_params", return_value={}):
            self.controller._on_phase1_clicked()
            self.assertTrue(self.controller._wind_timer.isActive())
            QTest.qWait(1250)
        self.assertGreater(len(self.state.wind_history_for_alt(3.0)), before)
        self.controller._wind_timer.stop()


if __name__ == "__main__":
    unittest.main()
