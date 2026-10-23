import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from utils.result_io import ResultFileError, load_archive, make_archive, save_archive, save_csv, validate_archive


class TestResultIO(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sample = load_archive(Path(__file__).parents[1] / "fixtures" / "result_archive.json")

    def test_numpy_round_trip_and_unavailable_score(self):
        result = self.sample["result"]
        result["trajectory_x"] = np.array(result["trajectory_x"])
        result["mc_avg_score"] = np.float64(-np.inf)
        result["mc_scatter_points"] = [{"x": np.float64(1), "score": -np.inf}]
        archive = make_archive(result, self.sample["nominal"], self.sample["context"])
        path = self.root / "結果.json"
        save_archive(path, archive)
        restored = load_archive(path)
        self.assertEqual(restored, archive)
        self.assertIsNone(restored["result"]["mc_avg_score"])
        self.assertNotIn("Infinity", path.read_text(encoding="utf-8"))
        result["trajectory_x"][0] = 999
        self.assertEqual(restored["result"]["trajectory_x"][0], 0)

    def test_old_ellipse_is_recomputed_once_from_scatter(self):
        source=Path(__file__).parents[1]/'fixtures'/'result_archive.json'
        raw=json.loads(source.read_text(encoding='utf-8'))
        self.assertNotIn('angle_convention',raw['result']['ellipse'])
        fixed=load_archive(source)
        from core.monte_carlo import compute_error_ellipse
        points=list(zip(raw['result']['mc_scatter_x'],raw['result']['mc_scatter_y']))
        self.assertEqual(fixed['result']['ellipse'],compute_error_ellipse(points,90))
        self.assertTrue(fixed['migration_notes'])
        path=self.root/'fixed.json'
        save_archive(path,fixed)
        self.assertEqual(load_archive(path),fixed)

    def test_csv_contains_all_samples_with_units(self):
        path = self.root / "results.csv"
        save_csv(path, self.sample)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[3], {"record_type": "trajectory", "index": "3", "time_s": "15",
                                  "east_m": "20", "north_m": "10", "altitude_m": "0"})
        self.assertEqual(rows[-1]["record_type"], "mc_impact")
        self.assertEqual(rows[-1]["north_m"], "13")

    def test_rejects_invalid_render_data(self):
        mutations = [
            lambda a: a.update(version=99),
            lambda a: a.update(format="airframe"),
            lambda a: a["context"].update(launch_lat=91),
            lambda a: a["context"].update(launch_lon=None),
            lambda a: a["result"].update(trajectory_y=[1]),
            lambda a: a["result"].update(t_hist=[0, 4, 2, 6]),
            lambda a: a["result"].update(apogee_m="bad"),
            lambda a: a["result"].update(landing_prob=100),
            lambda a: a["result"].update(land_x=None),
            lambda a: a["result"]["events"].update(apogee=[1, 2]),
            lambda a: a["result"]["ellipse"].update(a=-1),
            lambda a: a["result"]["kde"].update(Z=[[0, 0], [0, 0]]),
            lambda a: a["result"]["wind_nodes"][0].update(speed_ms="bad"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                archive = copy.deepcopy(self.sample)
                mutate(archive)
                with self.assertRaises(ResultFileError):
                    validate_archive(archive)

    def test_corrupt_json_and_nonfinite_coordinates_are_rejected(self):
        path = self.root / "broken.json"
        for text in ("{", "null", json.dumps(self.sample).replace('"land_x": 20', '"land_x": NaN')):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ResultFileError):
                load_archive(path)
        self.sample["result"]["trajectory_x"][0] = float("nan")
        with self.assertRaises(ResultFileError):
            make_archive(self.sample["result"], {}, self.sample["context"])

    def test_worker_time_alias_must_match_displayed_trajectory(self):
        result = self.sample["result"]
        result.pop("t_hist")
        result.update(t_vals=[0, 1], x_vals=[0, 1], y_vals=[0, 1], z_vals=[0, 1])
        with self.assertRaises(ResultFileError):
            validate_archive(self.sample)

    def test_failed_replace_keeps_previous_file_and_cleans_temporary(self):
        path = self.root / "results.json"
        path.write_bytes(b"previous result")
        with patch("utils.result_io.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                save_archive(path, self.sample)
        self.assertEqual(path.read_bytes(), b"previous result")
        self.assertEqual(list(self.root.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
