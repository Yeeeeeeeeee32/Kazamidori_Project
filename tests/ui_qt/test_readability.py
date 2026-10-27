import os
import re
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PySide6.QtCore import QDateTime, Qt
from PySide6.QtWidgets import QApplication, QPushButton, QLabel, QToolBox

from main_qt import DEFAULT_CONFIG
from ui_qt.app_state import AppState
from ui_qt.app_window import AppWindow
from ui_qt.sim_controller import SimController
from ui_qt.wind_data_dialog import WindDataDialog, WindDataWorker
from utils.result_io import load_archive
from utils.time_display import format_jst


@pytest.fixture
def ui():
    app = QApplication.instance() or QApplication([])
    state = AppState(DEFAULT_CONFIG)
    with patch('ui_qt.app_window.MapView.request_render'):
        win = AppWindow(state)
        controller = SimController(win, state)
        controller._wind_timer.stop()
        win._wind_refresh_timer.stop()
        yield state, win, controller
        controller._tray_icon.hide()
        win.close()
        controller.deleteLater()
        win.deleteLater()
        app.processEvents()


def test_upper_wind_is_mean_only_and_1_5m_keeps_history(ui):
    state, win, _ = ui
    now = time.monotonic()
    for stamp, speed, direction in ((now-20,2,359),(now-10,6,1)):
        with patch('time.monotonic', return_value=stamp):
            state.append_wind_nodes([dict(alt_m=1.5,speed_ms=speed+3,dir_deg=90),
                                     dict(alt_m=10,speed_ms=speed,dir_deg=direction)])
    win.update_wind_plot()
    lines = {line.get_label():line for line in win.wind_profile_ax.lines}
    np.testing.assert_allclose(lines['10 m'].get_ydata(), [4,4])
    assert len(win.wind_profile_ax.collections) == 1
    assert len(win.wind_profile_ax.collections[0].get_offsets()) == 2
    assert len(win.wind_ax.collections) == 1  # only 1.5 m shows past compass points
    assert '1.5 m' in lines
    assert win._wind_table.item(0,0).text() == '1.5 m'
    assert win._wind_table.item(1,1).text() == '4.0'
    assert float(win._wind_table.item(1,2).text()) < 2  # North, not South
    with patch('time.monotonic', return_value=now+100):
        win.update_wind_plot()
    np.testing.assert_allclose(win.wind_profile_ax.lines[1].get_ydata(), [4,4])


def test_jst_input_converts_across_midnight_and_ready_label_is_jst(ui, tmp_path):
    state, win, _ = ui
    state.phase4_options = dict(wind_time_utc='2026-10-23T15:00:00Z')
    dialog = WindDataDialog(state, win)
    assert dialog.target.dateTime().toString('yyyy-MM-dd HH:mm') == '2026-10-24 00:00'
    assert 'JST' in dialog.target.displayFormat()
    instant = QDateTime.fromString('2026-10-24T01:00:00+09:00', Qt.DateFormat.ISODate)
    dialog.target.setDateTime(instant)
    assert dialog._iso(dialog.target) == '2026-10-23T16:00:00Z'
    assert format_jst('2026-10-23T16:00:00Z') == '2026-10-24 01:00 JST'
    with patch.object(WindDataWorker, 'start'):
        dialog._run(str(tmp_path/'wind.csv'), False)
        assert dialog.worker.options['target'] == '2026-10-23T16:00:00Z'
        dialog.worker.ready.emit(str(tmp_path/'wind.csv'))
        dialog.worker.status.emit('2026-10-23T16:00:00+00:00')
        assert dialog.path_label.text() == 'Downloading: 2026-10-24 01:00 JST'
    assert state.phase4_options['wind_time_utc'] == '2026-10-23T16:00:00Z'
    dialog.close()


def test_buttons_and_go_nogo_have_plain_text_and_solid_phase1_color(ui):
    state, win, _ = ui
    emoji = re.compile('[\U0001f000-\U0001faff\u2600-\u27bf\u23e9-\u23fa\u25b6]')
    texts = [w.text() for cls in (QPushButton,QLabel) for w in win.findChildren(cls)]
    texts += [box.itemText(i) for box in win.findChildren(QToolBox) for i in range(box.count())]
    assert not any(emoji.search(text) for text in texts)
    assert 'qlineargradient' not in win.styleSheet()
    assert win.findChild(QPushButton,'btn_phase1_run') is None
    state.flight_mode = '定点滞空'
    state.set_wind_lock(0,-3,.2)
    state.phase2_active = True
    statuses = []
    state.tolerance_status_changed.connect(statuses.append)
    state.check_tolerance(3,0)
    state.check_tolerance(30,0)
    assert statuses == ['GO','NO-GO']


def test_3d_text_stays_in_figure_without_overlaps(ui, tmp_path):
    state, win, _ = ui
    state.simulation_result = load_archive(Path(__file__).parents[1]/'fixtures/result_archive.json')['result']
    for width, height in ((5,4.5),(3.6,3.4)):
        win.profile_fig.set_size_inches(width,height)
        for azimuth in (-150,-90,30,150):
            win._on_azim_changed(azimuth)
            win._layout_profile()
            win.profile_canvas.draw()
            renderer = win.profile_canvas.get_renderer()
            artists = [win._profile_header, win._profile_summary,
                       win.profile_canvas._time_label, win.profile_ax.get_legend()]
            bounds = [a.get_window_extent(renderer) for a in artists]
            figure = win.profile_fig.bbox
            axes_text = [axis.label for axis in (win.profile_ax.xaxis,win.profile_ax.yaxis,win.profile_ax.zaxis)]
            for axis, limits in zip((win.profile_ax.xaxis,win.profile_ax.yaxis,win.profile_ax.zaxis),
                                    (win.profile_ax.get_xlim(),win.profile_ax.get_ylim(),win.profile_ax.get_zlim())):
                axes_text += [tick.label1 for tick in axis.get_major_ticks()
                              if min(limits) <= tick.get_loc() <= max(limits) and tick.label1.get_visible()]
            for box in bounds + [text.get_window_extent(renderer) for text in axes_text]:
                assert figure.contains(box.x0,box.y0) and figure.contains(box.x1,box.y1)
            for i, first in enumerate(bounds):
                assert not any(first.overlaps(second) for second in bounds[i+1:])
    win.profile_fig.savefig(tmp_path/'profile_readability.png')
