"""Local WMM calculation and a per-site/date offline declination cache."""
import json
import math
import os
import tempfile
from datetime import date
from pathlib import Path

from pygeomag import GeoMag

CACHE = Path(__file__).resolve().parents[1] / 'assets/offline_map/declination_cache.json'


def declination_for_site(lat, lon, alt_m=0., *, cache_path=CACHE, today=None):
    today = today or date.today()
    lat, lon, alt_m = float(lat), float(lon), float(alt_m)
    if not all(math.isfinite(v) for v in (lat, lon, alt_m)) or not (-90 <= lat <= 90 and -180 <= lon <= 180 and -1000 <= alt_m <= 850000):
        raise ValueError('Invalid WMM coordinates or altitude')
    key = f'{lat:.6f},{lon:.6f},{alt_m:.1f},{today.isoformat()}'
    records = {}
    if cache_path is not None:
        try:
            records = json.loads(Path(cache_path).read_text(encoding='utf-8'))
            if not isinstance(records, dict):
                records = {}
            value = records.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value) and -180 <= value <= 180:
                return float(value)
        except (OSError, ValueError, TypeError):
            records = {}
    start, end = date(today.year, 1, 1), date(today.year + 1, 1, 1)
    year = today.year + (today - start).days / (end - start).days
    value = float(GeoMag().calculate(glat=lat, glon=lon, alt=alt_m / 1000., time=year).d)
    if not math.isfinite(value):
        raise ValueError('WMM returned invalid declination')
    if cache_path is not None:
        dest = Path(cache_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        records = dict(list(records.items())[-99:])
        records[key] = value
        fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, allow_nan=False)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return value


def true_bearing(bearing, reference, declination):
    return (float(bearing) + (float(declination) if reference == 'magnetic' else 0)) % 360


def input_bearing(bearing, reference, declination):
    return (float(bearing) - (float(declination) if reference == 'magnetic' else 0)) % 360
