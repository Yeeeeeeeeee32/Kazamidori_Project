"""Exercise the normal worker's successful run path, including KDE packaging."""
import os
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PySide6.QtWidgets import QApplication
from ui_qt.workers import SimulationWorker


@pytest.mark.parametrize('points', [[], [(1,1)]*4, [(0,0),(2,0),(0,3),(3,2),(1,1)]])
def test_free_mode_worker_finishes_with_or_without_kde(points):
    app = QApplication.instance() or QApplication([])
    worker = SimulationWorker(dict(flight_mode='自由', is_free_mode=True,
                                   elev=85, azim=0, mc_runs=5))
    errors, finished, mc_results = [], [], []
    worker.error.connect(errors.append)
    worker.sig_finished.connect(finished.append)
    worker.sig_mc_done.connect(mc_results.append)
    nominal = dict(ok=True, t_vals=[0,1,2], x_vals=[0,1,2], y_vals=[0,0,0],
                   z_vals=[0,10,0], impact_x=2,impact_y=0,r_horiz=2,
                   apogee_m=10,hang_time=2,apogee_idx=1,idx_burnout=0)
    with patch.object(worker, '_build_wind_profiles',return_value=([(0,0)],[(0,0)])), \
         patch.object(worker, '_build_sim_params',return_value={}), \
         patch('ui_qt.workers.simulate_once',return_value=nominal), \
         patch.object(worker, '_run_mc_loop',return_value=points):
        worker.run()
    assert not errors, errors
    assert len(finished) == len(mc_results) == 1
    assert finished[0]['has_sim_result']
    assert finished[0]['n_attempted'] == 5
    assert mc_results[0]['n_runs'] == len(points)
    assert finished[0]['kde'] == mc_results[0]['kde']
    assert len(mc_results[0]['mc_scatter_x_filtered']) == len(mc_results[0]['mc_scatter_y_filtered'])
    if len(set(points)) > 1:
        assert finished[0]['kde'] is not None
    else:
        assert finished[0]['kde'] is None
        assert mc_results[0]['mc_scatter_x_filtered'] == [p[0] for p in points]
