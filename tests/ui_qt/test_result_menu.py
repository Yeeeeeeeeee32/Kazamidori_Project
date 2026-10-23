"""Exercise menu actions against real Qt widgets and Matplotlib artists."""
import contextlib
import copy
import io
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kazamidori-mpl-test"))

import numpy as np
from PIL import Image
from matplotlib.contour import QuadContourSet
from PySide6.QtWidgets import QApplication
from main_qt import DEFAULT_CONFIG
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.sim_controller import SimController
from ui_qt.workers import SimulationWorker
from utils import map_downloader as maps
from utils.result_io import load_archive, save_archive


class TestResultMenu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.slot_errors = []
        stack.enter_context(patch("sys.excepthook", side_effect=lambda *args: self.slot_errors.append(args[1])))
        stack.enter_context(warnings.catch_warnings())
        warnings.filterwarnings("ignore", message="libpyside: Failed to disconnect")
        stack.enter_context(patch.object(maps, "OUT_DIR", self.root))
        self.network = stack.enter_context(patch.object(maps.requests.Session, "get", side_effect=AssertionError("Network used")))
        self.sample = load_archive(Path(__file__).parents[1] / "fixtures" / "result_archive.json")
        self.state = AppState(DEFAULT_CONFIG)
        self.window = AppWindow(self.state)
        self.controller = SimController(self.window, self.state)
        self.controller._wind_timer.stop()

    def tearDown(self):
        self.controller._tray_icon.hide()
        self.window.close()
        self.controller.deleteLater()
        self.window.deleteLater()
        self.app.processEvents()
        self.assertEqual(self.slot_errors, [], "An exception escaped a Qt signal handler")

    def load_result(self):
        path = self.root / "過去の結果.json"
        save_archive(path, self.sample)
        with patch("ui_qt.sim_controller.QFileDialog.getOpenFileName", return_value=(str(path), "")), \
             patch("ui_qt.sim_controller.SimulationWorker") as worker, \
             patch("ui_qt.sim_controller.MapDownloadWorker") as map_worker:
            self.window.action_import_results.trigger()
            worker.assert_not_called()
            map_worker.assert_not_called()
        self.assertIn("Results imported", self.window._status_label.text())

    def labels(self, ax):
        return ax.get_legend_handles_labels()[1]

    def test_import_restores_context_results_and_stops_live_tolerance(self):
        self.state.phase2_active = True
        self.state.launch_lat = 0
        self.state.launch_lon = 0
        self.state.cached_mc_scatter = np.ones((4, 2))
        self.state.show_scatter = False
        self.sample["result"]["landing_prob"] = 80
        self.load_result()
        self.assertEqual(self.state.simulation_result, self.sample["result"])
        self.assertEqual(self.state.nominal_result, self.sample["nominal"])
        self.assertEqual(self.window.lat_input.value(), 35.5)
        self.assertEqual(self.window.lon_input.value(), 139.5)
        self.assertEqual(self.window.mode_combo.currentText(), "自由")
        self.assertEqual(self.window.cep_prob_input.value(), 80)
        self.assertEqual(self.window.lbl_res_apogee.text(), "100.0 m")
        self.assertEqual(self.window.lbl_res_mc_avg_apo.text(), "101.0 m")
        self.assertEqual(self.state.cached_mc_scatter.shape, (4, 2))
        self.assertFalse(self.state.phase2_active)
        self.assertFalse(self.window.action_show_scatter.isChecked())
        self.assertNotIn("MC Scatter", self.labels(self.window.map_view.ax))
        self.assertIn("80% CEP", self.labels(self.window.map_view.ax))
        self.assertTrue(self.window.export_menu.isEnabled())
        self.network.assert_not_called()

    def test_view_toggles_change_both_plots_and_survive_percentile_edit(self):
        self.load_result()
        w = self.window
        self.assertIn("Burnout", self.labels(w.profile_ax))
        self.assertIn("CEP 90%", self.labels(w.profile_ax))
        self.assertTrue(any(isinstance(c, QuadContourSet) for c in w.profile_ax.collections))
        for name in ("kde", "cep", "scatter", "burnout", "apogee"):
            getattr(w, f"action_show_{name}").trigger()
        w.cep_prob_input.setValue(75)
        for ax in (w.profile_ax, w.map_view.ax):
            labels = self.labels(ax)
            for prefix in ("Burnout", "Apogee", "CEP", "MC", "KDE", "75% CEP"):
                self.assertFalse(any(label.startswith(prefix) for label in labels), labels)
            self.assertFalse(any(isinstance(c, QuadContourSet) for c in ax.collections))
        self.assertEqual(self.state.simulation_result["landing_prob"], 75)
        self.assertAlmostEqual(self.state.r90_radius, np.quantile(np.sqrt([13,5,10,18]),.75))
        self.assertEqual(len(self.state.simulation_result["trajectory_x"]), 4)
        self.state.show_cep = True
        self.assertTrue(w.action_show_cep.isChecked())
        self.assertIn("75% CEP", self.labels(w.map_view.ax))

    def test_event_toggles_also_work_without_phases(self):
        self.sample["result"].pop("phases")
        self.load_result()
        self.assertIn("Burnout", self.labels(self.window.profile_ax))
        self.window.action_show_apogee.trigger()
        self.window.action_show_burnout.trigger()
        self.assertFalse(any(s.startswith(("Burnout", "Apogee")) for s in self.labels(self.window.profile_ax)))

    def test_actual_map_and_3d_ellipse_align_with_kde_after_probability_change(self):
        from matplotlib.patches import Ellipse, Circle
        from core.monte_carlo import compute_error_ellipse, compute_kde_grid
        self.load_result()
        angle=np.deg2rad(12)
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        points=np.random.default_rng(173).normal(size=(600,2))*[9,1.5]
        points=points@rotation.T+[60,30]
        result=dict(self.state.simulation_result)
        result.update(ellipse=compute_error_ellipse(points),kde=compute_kde_grid(points),
                      mc_scatter_x=points[:,0].tolist(),mc_scatter_y=points[:,1].tolist())
        self.state.cached_mc_scatter=points
        self.state.simulation_result=result
        for probability in (90,75):
            self.window.cep_prob_input.setValue(probability)
            current=self.state.simulation_result['ellipse']
            patches=[p for p in self.window.map_view.ax.patches if type(p) is Ellipse]
            self.assertEqual(len(patches),1)
            major_tip=patches[0].get_patch_transform().transform([[1,0]])[0]
            expected=points.mean(axis=0)+current['a']*np.array([
                np.cos(current['angle_rad']),np.sin(current['angle_rad'])])
            np.testing.assert_allclose(major_tip,expected,atol=1e-10)
            lines=[line for line in self.window.profile_ax.lines if line.get_label().startswith('CEP ')]
            self.assertEqual(len(lines),1)
            x,y,_=lines[0].get_data_3d()
            np.testing.assert_allclose([x[0],y[0]],expected,atol=1e-10)
            circles=[p for p in self.window.map_view.ax.patches if type(p) is Circle and p.get_linestyle()=='--']
            self.assertTrue(any(np.allclose(p.center,points.mean(axis=0)) for p in circles))
            # Extract the largest KDE boundary and independently compare its axis.
            contours=[c for c in self.window.map_view.ax.collections if isinstance(c,QuadContourSet)]
            self.assertEqual(len(contours),1)
            polygon=max((p.vertices for p in contours[0].get_paths()),key=len)
            _,axes=np.linalg.eigh(np.cov(polygon,rowvar=False))
            direction=(major_tip-points.mean(axis=0))/current['a']
            self.assertGreater(abs(direction@axes[:,-1]),.99)

    def test_target_containment_rejects_biased_cloud_and_failed_trials(self):
        self.load_result()
        self.state.flight_mode='定点滞空'
        self.state.target_radius=10
        self.state.target_x=0
        self.state.target_y=0
        self.state.launch_lat=35.5
        self.state.launch_lon=139.5
        for center,attempted,expected in ((100,10,False),(0,20,False),(0,10,True)):
            points=[(center+i/100,0) for i in range(10)]
            stats=SimulationWorker._compute_stats(points,90)
            stats['n_attempted']=attempted
            nominal = copy.deepcopy(self.sample['result'])
            for axis in 'xyz':
                nominal[f'{axis}_vals'] = nominal.pop(f'trajectory_{axis}')
            nominal['t_vals'] = nominal.pop('t_hist')
            nominal['r_horiz'] = float(np.hypot(nominal['impact_x'], nominal['impact_y']))
            payload=SimulationWorker._package_result(nominal,points,stats,90,cancelled=False)
            payload.update(nominal_surf_spd=3,nominal_surf_dir=0)
            self.controller._on_mc_done(payload)
            self.assertEqual(self.state.phase2_active,expected)
        self.assertEqual(self.state._locked_mu,(0.,-3.))
        self.state.check_tolerance(3,0)

    def test_export_menu_writes_images_csv_and_restorable_json(self):
        self.load_result()
        kinds = ("profile", "map", "csv", "json")
        for action, kind in zip(self.window.export_menu.actions(), kinds):
            suffix = "png" if kind in ("profile", "map") else kind
            path = self.root / f"export_{kind}.{suffix}"
            with patch("ui_qt.app_window.QFileDialog.getSaveFileName", return_value=(str(path), "")):
                action.trigger()
            self.assertIn("Results exported", self.window._status_label.text())
            self.assertTrue(path.is_file())
            if suffix == "png":
                with Image.open(path) as image:
                    self.assertGreater(image.width, 100)
                    self.assertGreater(image.height, 100)
                    self.assertGreater(len(image.getcolors(image.width * image.height)), 10)
            elif suffix == "json":
                self.assertEqual(load_archive(path)["result"], self.sample["result"])
            else:
                self.assertIn("mc_impact,3,,23,13,0", path.read_text(encoding="utf-8-sig"))

    def test_invalid_import_and_cancel_preserve_current_result(self):
        self.load_result()
        before = copy.deepcopy(self.state.simulation_result)
        path = self.root / "bad.json"
        path.write_text('{"format": "airframe"}', encoding="utf-8")
        with patch("ui_qt.sim_controller.QFileDialog.getOpenFileName", return_value=(str(path), "")):
            self.window.action_import_results.trigger()
        self.assertIn("Import failed", self.window._status_label.text())
        self.assertEqual(self.state.simulation_result, before)
        for method in ("getOpenFileName", "getSaveFileName"):
            with patch(f"ui_qt.app_window.QFileDialog.{method}", return_value=("", "")):
                if method == "getOpenFileName":
                    self.window.action_import_results.trigger()
                else:
                    self.window.export_menu.actions()[0].trigger()
        self.assertEqual(self.state.simulation_result, before)

    def test_unavailable_and_busy_actions_and_export_error(self):
        self.assertFalse(self.window.export_menu.isEnabled())
        self.load_result()
        self.state.is_calculating = True
        self.assertFalse(self.window.export_menu.isEnabled())
        self.assertFalse(self.window.action_import_results.isEnabled())
        with patch("ui_qt.app_window.QFileDialog.getSaveFileName") as dialog:
            self.window._on_export_results("json")
            dialog.assert_not_called()
        self.state.is_calculating = False
        with patch("ui_qt.app_window.QFileDialog.getSaveFileName", return_value=(str(self.root / "fail.json"), "")), \
             patch("utils.result_io.atomic_write", side_effect=OSError("disk full")):
            self.window.export_menu.actions()[3].trigger()
        self.assertIn("Export failed", self.window._status_label.text())
        self.assertEqual(self.state.simulation_result, self.sample["result"])

    def test_advanced_settings_menu_opens_dialog(self):
        menu = next(a.menu() for a in self.window.menuBar().actions() if a.text() == "&Simulation")
        with patch.object(self.window._adv_dialog, "exec", return_value=0) as dialog:
            menu.actions()[0].trigger()
            dialog.assert_called_once()

    def test_worker_payload_can_be_exported_and_imported(self):
        self.load_result()
        nominal = copy.deepcopy(self.sample["result"])
        for axis in "xyz":
            nominal[f"{axis}_vals"] = nominal.pop(f"trajectory_{axis}")
        nominal["t_vals"] = nominal.pop("t_hist")
        nominal["r_horiz"] = 22.36
        scatter = [(17, 8), (19, 12), (21, 7), (23, 13)]
        stats = {**self.sample["result"], "mc_avg_score": float("-inf"), "mc_min_score": float("-inf")}
        payload = SimulationWorker._package_result(nominal, scatter, stats, 90, cancelled=False)
        self.state.nominal_result = nominal
        self.state.simulation_result = self.controller._adapt_for_window(payload)
        path = self.root / "worker.json"
        with patch("ui_qt.app_window.QFileDialog.getSaveFileName", return_value=(str(path), "")):
            self.window.export_menu.actions()[3].trigger()
        self.assertIn("Results exported", self.window._status_label.text())
        archive = load_archive(path)
        self.assertEqual(archive["result"]["t_vals"], [0, 1, 5, 15])
        with patch("ui_qt.sim_controller.QFileDialog.getOpenFileName", return_value=(str(path), "")):
            self.window.action_import_results.trigger()
        self.assertEqual(self.state.simulation_result["mc_scatter_x"], [17, 19, 21, 23])
        self.assertIsNone(self.state.simulation_result["mc_avg_score"])

    def test_run_and_stop_menu_use_controller(self):
        self.state.motor_cg_pos = None
        self.window.action_run.trigger()
        self.assertIn("Validation Error", self.window._status_label.text())
        worker = Mock()
        worker.isRunning.return_value = True
        self.controller._worker = worker
        self.state.is_calculating = True
        self.controller._set_run_buttons_enabled(False)
        self.window.action_stop.trigger()
        worker.stop.assert_called_once()
        self.assertFalse(self.window.action_run.isEnabled())
        self.controller._worker = None
        self.state.is_calculating = False


if __name__ == "__main__":
    unittest.main()
