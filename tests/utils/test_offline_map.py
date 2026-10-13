"""Offline map publication must preserve a usable cache on every failure."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from utils import map_downloader as maps


class TestOfflineMapStorage(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(maps, "OUT_DIR", self.root))
        stack.enter_context(patch.object(maps.time, "sleep"))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.meta = {
            "center_lat": 35.422158, "center_lon": 139.422688,
            "extent_meters": [-1000, 1000, -1000, 1000],
            "image_size_px": [16, 16], "magnetic_declination": -7.5,
            "declination_model": "WMM",
        }
        (self.root / "map_meta.json").write_text(json.dumps(self.meta), encoding="utf-8")
        Image.new("RGB", (16, 16), "red").save(self.root / "background.png")
        self.original_meta = (self.root / "map_meta.json").read_bytes()

    def assert_old_map_preserved(self):
        self.assertEqual((self.root / "map_meta.json").read_bytes(), self.original_meta)
        self.assertEqual(maps.read_offline_map()[1].getpixel((0, 0)), (255, 0, 0, 255))

    def test_total_and_partial_download_failure_preserve_previous_map(self):
        for partial in (False, True):
            calls = 0

            def tile(*_):
                nonlocal calls
                calls += 1
                return Image.new("RGBA", (256, 256), "blue") if partial and calls == 1 else None

            with self.subTest(partial=partial), patch.object(maps, "_download_tile", side_effect=tile):
                with self.assertRaisesRegex(RuntimeError, "preserved"):
                    maps.download_offline_map(35.5, 139.5)
                self.assert_old_map_preserved()

    def test_success_switches_image_and_metadata_together(self):
        replace = maps.os.replace

        def check_commit(source, destination):
            # The old map is readable right until the metadata is committed.
            self.assert_old_map_preserved()
            return replace(source, destination)

        with patch.object(maps, "_download_tile", return_value=Image.new("RGBA", (256, 256), "blue")), \
                patch.object(maps.os, "replace", side_effect=check_commit):
            meta = maps.download_offline_map(35.5, 139.5, radius=100)
        stored, image = maps.read_offline_map()
        self.assertEqual(stored, meta)
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 255, 255))
        self.assertTrue((self.root / meta["image_file"]).exists())

    def test_metadata_write_failure_preserves_previous_map(self):
        replace = maps.os.replace

        def fail_commit(source, destination):
            if Path(destination).name == "map_meta.json":
                raise OSError("disk error")
            return replace(source, destination)

        with patch.object(maps, "_download_tile", return_value=Image.new("RGBA", (256, 256), "blue")), \
                patch.object(maps.os, "replace", side_effect=fail_commit):
            with self.assertRaises(OSError):
                maps.download_offline_map(35.5, 139.5, radius=100)
        self.assert_old_map_preserved()

    def test_invalid_coordinates_and_radius_do_not_touch_cache(self):
        for lat, lon, radius in ((-9999, 139, 100), (90, 139, 100),
                                 (35, 181, 100), (float("nan"), 0, 100),
                                 (35, 139, 0), (35, 139, float("inf"))):
            with self.subTest(lat=lat, lon=lon, radius=radius):
                with self.assertRaises(ValueError):
                    maps.download_offline_map(lat, lon, radius)
                self.assert_old_map_preserved()
        self.assertEqual(maps.validate_map_coordinates(0, 0), (0, 0))

    def test_corrupt_and_incomplete_images_are_rejected(self):
        (self.root / "background.png").write_bytes(b"not an image")
        with self.assertRaises(OSError):
            maps.read_offline_map()
        Image.new("RGBA", (16, 16), (0, 0, 0, 0)).save(self.root / "background.png")
        with self.assertRaisesRegex(ValueError, "missing tiles"):
            maps.read_offline_map()

    def test_wmm_receives_decimal_year_and_returns_japan_declination(self):
        with patch.object(maps, "GeoMag", wraps=maps.GeoMag) as model:
            declination = maps.compute_magnetic_declination(35.422158, 139.422688)
        self.assertLess(declination, -5)
        self.assertGreater(declination, -10)
        self.assertEqual(model.call_count, 1)


if __name__ == "__main__":
    unittest.main()
