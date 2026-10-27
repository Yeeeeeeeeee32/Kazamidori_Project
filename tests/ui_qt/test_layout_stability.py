import os
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QPushButton, QScrollArea
from main_qt import DEFAULT_CONFIG
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.sim_controller import SimController


@pytest.fixture
def ui():
    app = QApplication.instance() or QApplication([])
    state = AppState(DEFAULT_CONFIG)
    with patch('ui_qt.app_window.MapView.request_render'):
        win = AppWindow(state)
        controller = SimController(win,state)
        controller._wind_timer.stop()
        win._wind_refresh_timer.stop()
        yield app,state,win,controller
        controller._tray_icon.hide()
        win.close()
        controller.deleteLater()
        win.deleteLater()
        app.processEvents()


def test_single_run_dispatches_selected_mode_and_stop(ui):
    app,state,win,controller = ui
    assert win.findChild(QPushButton,'btn_phase1_run') is None
    with patch.object(controller,'_on_run_clicked') as normal, patch.object(
        controller,'_on_phase1_clicked') as search, patch.object(controller,'_worker') as worker:
        for mode in ('自由','高度','定点滞空','有翼','モンブラン'):
            win.mode_combo.setCurrentText(mode)
            win.findChild(QPushButton,'btn_run').click()
        assert normal.call_count == 1
        assert search.call_count == 4
        win.findChild(QPushButton,'btn_stop').setEnabled(True)
        win.findChild(QPushButton,'btn_stop').click()
        worker.stop.assert_called_once()


@pytest.mark.parametrize('size', [(960,640),(1280,800),(1600,900)])
def test_long_progress_and_results_keep_splitter_widths(ui,size,tmp_path):
    app,state,win,controller = ui
    win.resize(*size)
    win.show()
    win.mode_combo.setCurrentText('モンブラン')
    win._results_grp.show()
    app.processEvents()
    app.processEvents()
    before = (win.width(),win._main_splitter.sizes(),win._top_splitter.sizes())
    long = 'Phase 2: Monte Carlo verification (1000/1000) — elev=85.1234 azi=359.9999 ' * 12
    controller._on_worker_status(long)
    controller._on_progress(75)
    win._wind_status.setText(long)
    win.lbl_gpv_status.setText(long)
    win.lbl_koinobori_status.setText(long)
    win.update_status_indicator('NO-GO ' + long)
    win.mc_quality_label.setText(long)
    win.montblanc_result_label.setText(long)
    win.airframe_label.setText('long_design_name_' * 20 + '.ork')
    win.motor_label.setText('long_motor_name_' * 20 + '.csv')
    for _,value in win.res_labels.values():
        value.setText('12345678901234567890.123456')
    app.processEvents()
    app.processEvents()
    after = (win.width(),win._main_splitter.sizes(),win._top_splitter.sizes())
    assert after == before
    assert win._progress.format() == '%p%'
    assert win._phase_label.text() == win._phase_label.toolTip() == long
    assert win._status_label.toolTip() == long
    assert isinstance(win._params_panel,QScrollArea)
    assert win._params_panel.verticalScrollBar().maximum() > 0
    path = tmp_path / f'layout_{size[0]}.png'
    assert win.grab().save(str(path))
    hbar = win._params_panel.horizontalScrollBar()
    assert hbar.maximum() == 0 or hbar.isVisible()
    assert win._phase_label.isVisible()
    assert win._phase_label.mapTo(win,QPoint(win._phase_label.width(),0)).x() <= win.width()
