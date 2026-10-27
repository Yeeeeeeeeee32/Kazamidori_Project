"""Flat Airframe.json loading and the real Qt input bindings."""
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
import warnings
import numpy as np
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
from utils.data_loader import AirframeConfigError, load_airframe_config, save_airframe_config
from ui_qt.fin_editor import FinOutlineDialog
from ui_qt.airframe_load_dialog import AirframeLoadDialog


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
    def test_optional_cp_round_trip_and_legacy_automatic(self):
        save_airframe_config(str(self.path), {**AIRFRAME, 'cp': .32})
        self.assertEqual(load_airframe_config(str(self.path))['cp'], .32)
        save_airframe_config(str(self.path), {**AIRFRAME, 'cp': None})
        self.assertIsNone(load_airframe_config(str(self.path))['cp'])
        self.write(AIRFRAME)
        self.assertNotIn('cp', load_airframe_config(str(self.path)))

    def test_invalid_cp_rejected(self):
        for value in (-1, True, float('nan'), float('inf'), .5, 'bad'):
            with self.subTest(cp=value):
                self.write({**AIRFRAME, 'cp': value})
                with self.assertRaisesRegex(AirframeConfigError, 'cp'):
                    load_airframe_config(str(self.path))

    def test_flat_file_including_motor_values_and_utf8_bom(self):
        self.write(AIRFRAME)
        self.assertEqual(load_airframe_config(str(self.path)), AIRFRAME)

    def test_airframe_does_not_require_motor_fields(self):
        data = dict(AIRFRAME)
        del data["motor_pos"]
        self.write(data)
        self.assertNotIn('motor_pos', load_airframe_config(str(self.path)))

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

    def load(self, path=None, supplement=None):
        path = str(self.path) if path is None else str(path)
        selection = ((path, str(supplement or '')) if path.endswith(('.ork', '.ork.gz')) else ('', path)) if path else None
        with patch("ui_qt.sim_controller.select_airframe_files", return_value=selection) as dialog:
            self.window.sig_load_airframe_json_clicked.emit()
        return dialog

    def test_button_loads_all_twelve_fields(self):
        self.state.set_moi(1, 2, 3)
        buttons = [b for b in self.window.findChildren(QPushButton)
                   if b.text() in ("Load Airframe", "Load airframe")]
        self.assertEqual(len(buttons), 2)
        with patch("ui_qt.sim_controller.select_airframe_files",
                   return_value=("", str(self.path))):
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
        dialog.assert_called_once()
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

    def test_cp_load_apply_save_and_restore_auto(self):
        self.write({**AIRFRAME, 'cp': .32})
        self.load()
        self.assertEqual(self.state.rocket_cp, .32)
        self.assertEqual(float(self.window.af_cp_input.text()), .32)
        saved = self.path.parent / 'manual_cp.json'
        def edit():
            self.window.af_cp_input.setText('0.33')
            self.assertEqual(self.state.rocket_cp, .32)
            with patch('ui_qt.sim_controller.QFileDialog.getSaveFileName', return_value=(str(saved), '')):
                self.window._manual_dialog.sig_save_json.emit()
            self.window._manual_dialog.sig_apply.emit()
        self.run_manual(edit)
        self.assertEqual(self.state.rocket_cp, .33)
        self.assertEqual(load_airframe_config(str(saved))['cp'], .33)
        def clear():
            self.window.af_cp_input.clear()
            self.window._manual_dialog.sig_apply.emit()
        self.run_manual(clear)
        self.assertIsNone(self.state.rocket_cp)
        self.assertNotIn('red', self.window.af_cp_input.styleSheet())
        self.state.rocket_cp = .31
        self.write(AIRFRAME)
        self.load()
        self.assertIsNone(self.state.rocket_cp)
        self.assertEqual(self.window.af_cp_input.text(), '')

    def test_cp_cancel_and_invalid_apply_keep_state(self):
        self.write({**AIRFRAME, 'cp': .32})
        self.load()
        def edit():
            self.window.af_cp_input.setText('0.5')
            self.window._manual_dialog.sig_apply.emit()
            self.assertTrue(self.window._manual_dialog.isVisible())
            self.assertIn('cp', self.window._manual_dialog.error_label.text())
            self.assertEqual(self.state.rocket_cp, .32)
            self.window._manual_dialog.reject()
        self.run_manual(edit)
        self.assertEqual(self.state.rocket_cp, .32)
        self.assertEqual(float(self.window.af_cp_input.text()), .32)

    def test_freeform_edit_save_apply_and_legacy_reload(self):
        self.load()
        points = [[0,0],[.01,.03],[.04,.03],[.04,0]]
        saved = self.path.parent/'fins.json'
        def edit():
            md = self.window._manual_dialog
            md._use_outline(points)
            md.fin_shape_input.setCurrentIndex(md.fin_shape_input.findData('freeform'))
            self.assertEqual(self.state.fin_options['fin_shape'],'trapezoid')
            with patch('ui_qt.sim_controller.QFileDialog.getSaveFileName', return_value=(str(saved),'')):
                md.sig_save_json.emit()
            md.sig_apply.emit()
        self.run_manual(edit)
        self.assertEqual(self.state.fin_options['fin_points'],points)
        self.assertEqual(load_airframe_config(str(saved))['fin_points'],points)
        self.assertAlmostEqual(self.state.fin_tip_chord,.03)
        self.load()
        self.assertEqual(self.state.fin_options['fin_shape'],'trapezoid')
        self.assertIsNone(self.state.fin_options['fin_points'])

    def test_openrocket_fin_import_is_staged_and_keeps_mount_position(self):
        import zipfile
        self.load()
        path = self.path.parent/'design.ork'
        xml = ('<openrocket><rocket><subcomponents><trapezoidfinset>'
               '<name>Imported fins</name><fincount>3</fincount><rootchord>.05</rootchord>'
               '<tipchord>.01</tipchord><height>.04</height><sweeplength>.02</sweeplength>'
               '<cant>2</cant></trapezoidfinset></subcomponents></rocket></openrocket>')
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('rocket.ork',xml)
        def edit():
            md = self.window._manual_dialog
            with patch('ui_qt.app_window.QFileDialog.getOpenFileName', return_value=(str(path),'')):
                md.load_ork_fins_btn.click()
            self.assertEqual(float(md.af_finroot_input.text()),.05)
            self.assertEqual(float(md.fin_sweep_input.text()),.02)
            self.assertEqual(self.state.fin_root_chord,AIRFRAME['fin_root'])
            md.sig_apply.emit()
        self.run_manual(edit)
        self.assertEqual(self.state.fin_root_chord,.05)
        self.assertEqual(self.state.fin_options['fin_cant'],2)
        self.assertEqual(self.state.fin_options['fin_sweep'],.02)
        self.assertEqual(self.state.fin_position,AIRFRAME['fin_pos'])
        def cancel():
            md=self.window._manual_dialog
            md.fin_shape_input.setCurrentIndex(md.fin_shape_input.findData('freeform'))
            md.reject()
        self.run_manual(cancel)
        self.assertEqual(self.state.fin_options['fin_shape'],'trapezoid')

    def test_outline_editor_rejects_crossing_without_closing(self):
        points=[[0,0],[.01,.03],[.04,.03],[.04,0]]
        dialog=FinOutlineDialog(points,self.window)
        dialog.show()
        dialog.table.item(1,0).setText('.04')
        dialog.table.item(2,0).setText('.01')
        dialog.accept()
        self.assertTrue(dialog.isVisible())
        self.assertTrue(dialog.error_label.text())
        dialog.table.item(1,0).setText('.01')
        dialog.table.item(2,0).setText('.04')
        dialog.accept()
        self.assertEqual(dialog.points,points)

    def test_worker_snapshot_includes_cp_and_fin_outline(self):
        from ui_qt.workers import SimulationWorker
        self.load()
        self.state.rocket_cp=.32
        self.state.fin_options=dict(fin_shape='freeform',fin_points=[[0,0],[.01,.03],[.04,.03],[.04,0]])
        self.window._motor_thrust_data=[[0,0],[.1,10],[1,0]]
        self.window._motor_burn_time=1
        self.window.lat_input.setValue(35)
        self.window.lon_input.setValue(139)
        with patch('utils.magnetic.declination_for_site',return_value=-7):
            p=self.controller._collect_params()
        sim=SimulationWorker._build_sim_params(p,[(0,0),(1000,0)],[(0,0),(1000,0)])
        self.assertEqual(sim['cp_override'],.32)
        self.assertEqual(sim['fin_points'],self.state.fin_options['fin_points'])

    def test_outline_editor_can_recover_after_deleting_all_vertices(self):
        dialog=FinOutlineDialog([[0,0],[.02,.03],[.04,0]],self.window)
        dialog.table.selectAll()
        dialog._remove()
        self.assertEqual(dialog.table.rowCount(),0)
        for _ in range(3):
            dialog._add()
        dialog.accept()
        self.assertIsNotNone(dialog.points)
        self.assertEqual(len(dialog.points),3)

    def test_whole_openrocket_load_sources_and_missing_motor(self):
        self.load()
        self.window._motor_thrust_data = [[0,0],[1,10]]
        fixture = Path(__file__).parents[1] / 'fixtures' / 'single_stage.ork'
        self.load(str(fixture))
        self.assertEqual(self.state.rocket_dry_mass,.25)
        self.assertAlmostEqual(self.state.fin_position,.33)
        self.assertAlmostEqual(self.state.motor_cg_pos,.405)
        self.assertIsNone(self.state.motor_dry_mass)
        self.assertEqual(self.window.motor_dry_mass_input.text(),'')
        self.assertFalse(self.window.motor_dry_mass_input.isReadOnly())
        self.assertTrue(self.window.motor_cg_input.isReadOnly())
        self.assertTrue(self.window.af_mass_input.isReadOnly())
        self.assertTrue(self.window.af_cp_input.isReadOnly())
        self.assertIsNone(self.window._motor_thrust_data)
        self.assertIsNone(self.state.parachute_lag)
        self.assertAlmostEqual(self.state.parachute_area,np.pi*.25**2)
        self.assertEqual(self.state.parachute_cd,1)
        self.assertIn('Estes C5',self.window.motor_label.text())
        self.assertGreater(self.window._manual_dialog.airframe_sources_table.rowCount(),10)
        # Manual fallback remains editable; source values beat programmatic edits
        # too, and the actual simulation keeps dynamic CP from geometry.
        self.window.motor_dry_mass_input.setText('.016')
        self.state.rocket_cg=.39
        self.state.fin_position=.1
        self.window.lat_input.setValue(35)
        self.window.lon_input.setValue(139)
        with patch('utils.magnetic.declination_for_site',return_value=-7):
            params=self.controller._collect_params()
        self.assertEqual(params['airframe_cg'],.2)
        self.assertAlmostEqual(params['fin_pos'],.33)
        self.assertEqual(params['motor_dry_mass'],.016)
        self.assertIsNone(params['cp_override'])
        # Legacy JSON switches back to editable configuration.
        self.load()
        self.assertFalse(self.window.af_mass_input.isReadOnly())
        self.assertFalse(self.window.af_cp_input.isReadOnly())
        self.assertFalse(self.window.motor_cg_input.isReadOnly())
        self.assertFalse(self.state.airframe_provenance)

    def test_two_file_load_displays_both_names_and_editable_supplement(self):
        ork = self.path.parent / 'design.ork'
        shutil.copyfile(Path(__file__).parents[1] / 'fixtures' / 'single_stage.ork', ork)
        self.write(dict(ork_file=ork.name, motor_dry_mass=.016))
        self.load(str(ork), self.path)
        self.assertEqual(self.state.motor_dry_mass, .016)
        self.assertEqual(self.state.rocket_dry_mass, .25)
        self.assertIn('design.ork + Airframe.json', self.window.airframe_label.text())
        self.assertIn('補完JSON', self.window.motor_dry_mass_input.toolTip())
        self.assertTrue(self.window.af_mass_input.isReadOnly())
        self.assertFalse(self.window.motor_dry_mass_input.isReadOnly())
        self.window.motor_dry_mass_input.setText('.018')
        self.assertEqual(self.state.motor_dry_mass, .018)
        self.window.lat_input.setValue(35)
        self.window.lon_input.setValue(139)
        with patch('utils.magnetic.declination_for_site', return_value=-7):
            self.assertEqual(self.controller._collect_params()['motor_dry_mass'], .018)
        self.load(str(ork), self.path)
        self.assertEqual(self.state.motor_dry_mass, .016)
        before = self.window.airframe_label.text()
        self.write(dict(motor_dry_mass=-1))
        self.load(str(ork), self.path)
        self.assertIn('failed', self.window._status_label.text())
        self.assertEqual(self.state.motor_dry_mass, .016)
        self.assertEqual(self.window.airframe_label.text(), before)

    def test_real_file_selection_dialog_loads_only_manually_chosen_ork(self):
        self.load()
        folder = self.path.parent / 'other'
        folder.mkdir()
        ork = folder / 'another_design.ork'
        shutil.copyfile(Path(__file__).parents[1] / 'fixtures' / 'single_stage.ork', ork)
        supplement = self.path.parent / 'measurements.json'
        supplement.write_text(json.dumps(dict(motor_dry_mass=.023)), encoding='utf-8')
        # A similarly named neighbor must not be loaded automatically.
        ork.with_suffix('.json').write_text(json.dumps(dict(motor_dry_mass=.099)))
        errors = []
        def choose():
            dialog = self.app.activeModalWidget()
            try:
                self.assertIsInstance(dialog, AirframeLoadDialog)
                self.assertEqual(dialog.ork_path.text(), '')
                self.assertFalse(hasattr(dialog, 'json_path'))
                self.assertFalse(hasattr(dialog, 'json_browse'))
                dialog.accept()
                self.assertTrue(dialog.isVisible())
                self.assertTrue(dialog.error_label.text())
                with patch('ui_qt.airframe_load_dialog.QFileDialog.getOpenFileName',
                           side_effect=[(str(ork), ''), ('', '')]):
                    dialog.ork_browse.click()
                    dialog.ork_browse.click()  # Cancelling keeps the chosen design.
                    self.assertEqual(dialog.ork_path.text(), str(ork))
                dialog.ork_path.setText(str(supplement))
                dialog.accept()
                self.assertTrue(dialog.isVisible())
                self.assertIn('.ork', dialog.error_label.text())
                dialog.ork_path.setText(str(ork))
                self.assertEqual(self.state.motor_dry_mass, AIRFRAME['motor_dry_mass'])
                dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).click()
            except BaseException as exc:
                errors.append(exc)
                dialog.reject()
        QTimer.singleShot(0, choose)
        self.window.sig_load_airframe_json_clicked.emit()
        if errors:
            raise errors[0]
        self.assertIsNone(self.state.motor_dry_mass)
        self.assertEqual(self.window.airframe_label.text(), 'another_design.ork')
        before = self.window.airframe_label.text()
        def cancel():
            dialog = self.app.activeModalWidget()
            dialog.ork_path.setText(str(ork))
            dialog.reject()
        QTimer.singleShot(0, cancel)
        self.window.sig_load_airframe_json_clicked.emit()
        self.assertIsNone(self.state.motor_dry_mass)
        self.assertEqual(self.window.airframe_label.text(), before)

    def test_two_file_staging_and_minimal_save(self):
        self.load()
        ork = self.path.parent / 'design.ork'
        shutil.copyfile(Path(__file__).parents[1] / 'fixtures' / 'single_stage.ork', ork)
        self.write(dict(ork_file=ork.name, motor_dry_mass=.016))
        def cancel():
            self.load(str(ork), self.path)
            self.assertEqual(self.state.motor_dry_mass, AIRFRAME['motor_dry_mass'])
            self.window._manual_dialog.reject()
        self.run_manual(cancel)
        self.assertEqual(self.state.rocket_dry_mass, AIRFRAME['mass'])
        def apply():
            self.load(str(ork), self.path)
            with patch('ui_qt.sim_controller.QFileDialog.getSaveFileName', return_value=(str(self.path), '')):
                self.window._manual_dialog.sig_save_json.emit()
            self.window._manual_dialog.sig_apply.emit()
        self.run_manual(apply)
        self.assertEqual(self.state.rocket_dry_mass, .25)
        self.assertEqual(self.state.motor_dry_mass, .016)
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8')),
                         {})

    def test_whole_openrocket_staged_cancel_apply_and_save_preserve_priority(self):
        self.load()
        fixture = Path(__file__).parents[1] / 'fixtures' / 'single_stage.ork'
        def cancel():
            self.load(str(fixture))
            self.assertEqual(float(self.window.af_mass_input.text()),.25)
            self.assertEqual(self.state.rocket_dry_mass,AIRFRAME['mass'])
            self.assertEqual(self.state.motor_dry_mass,AIRFRAME['motor_dry_mass'])
            self.window._manual_dialog.reject()
        self.run_manual(cancel)
        self.assertEqual(self.state.rocket_dry_mass,AIRFRAME['mass'])
        self.assertFalse(self.state.airframe_provenance)
        self.assertFalse(self.window.af_mass_input.isReadOnly())
        def apply():
            self.load(str(fixture))
            self.window.af_mass_input.setText('99')
            self.window.af_cp_input.setText('.01')
            self.window._manual_dialog.sig_apply.emit()
        self.run_manual(apply)
        self.assertEqual(self.state.rocket_dry_mass,.25)
        self.assertGreater(self.state.rocket_cp,.1)
        self.assertIsNone(self.state.motor_dry_mass)
        saved=self.path.parent/'imported.json'
        def save():
            with patch('ui_qt.sim_controller.QFileDialog.getSaveFileName',return_value=(str(saved),'')):
                self.window._manual_dialog.sig_save_json.emit()
            self.window._manual_dialog.reject()
        self.run_manual(save)
        self.assertTrue(saved.exists())
        data=load_airframe_config(fixture, saved)
        self.assertEqual(data['provenance']['sources']['mass'],'ork')
        self.assertIsNone(data['motor_dry_mass'])
        self.load(str(fixture), saved)
        self.assertTrue(self.window.af_mass_input.isReadOnly())

    def test_imported_parachute_wins_over_json_but_lag_is_manual(self):
        fixture = Path(__file__).parents[1] / 'fixtures' / 'single_stage.ork'
        self.load(str(fixture))
        para=self.path.parent/'parachute.json'
        para.write_text(json.dumps(dict(cd=.8,area=.1,lag=.5)),encoding='utf-8')
        with patch('ui_qt.sim_controller.QFileDialog.getOpenFileName',return_value=(str(para),'')):
            self.window.sig_load_para_json_clicked.emit()
        self.assertEqual(self.state.parachute_cd,1)
        self.assertAlmostEqual(self.state.parachute_area,np.pi*.25**2)
        self.assertEqual(self.state.parachute_lag,.5)

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
        self.assertNotIn('motor_dry_mass', values)

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
        self.assertCountEqual(names, ["Apply", "Cancel", "Load airframe", "Save airframe",
                                      'Edit Fin Outline…', 'Import Fin Shape (.ork)…'])
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
        before = len(self.state.wind_history_for_alt(1.5))
        with patch("ui_qt.sim_controller.OptimizationWorker"), \
             patch("ui_qt.sim_controller.SimulationWorker"), \
             patch.object(self.controller, "_validate_run_prerequisites", return_value=True), \
             patch.object(self.controller, "_collect_params", return_value={}):
            self.controller._on_phase1_clicked()
            self.assertTrue(self.controller._wind_timer.isActive())
            QTest.qWait(1250)
        self.assertGreater(len(self.state.wind_history_for_alt(1.5)), before)
        self.controller._wind_timer.stop()


if __name__ == "__main__":
    unittest.main()
