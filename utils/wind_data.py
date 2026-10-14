"""Bounded-memory weather CSV cache and Open-Meteo hourly pressure-level fetch.

CSV is sorted by time_utc, with one row per level. alt_m is AGL; pressure_hpa
is converted at simulation time using the measured station pressure.
"""
import csv
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from core.atmosphere import pressure_altitude_agl

FIELDS = ('time_utc', 'alt_m', 'pressure_hpa', 'speed_ms', 'dir_deg', 'latitude', 'longitude', 'source')
LEVELS = (1000, 975, 950, 925, 900, 850, 800, 700, 600, 500)


def utc_time(value):
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Wind timestamps require a timezone (e.g. 2026-10-14T03:00:00Z)')
    return dt.astimezone(timezone.utc)


def _number(value, label):
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f'{label} must be finite')
    return v


def read_wind_snapshot(path, target_time, station_pressure_pa=101325., temperature_c=15.,
                       latitude=None, longitude=None, cancelled=lambda: False):
    """Scan CSV with O(one profile) memory; hold preceding sample for <1 hour.

    No fallback to unrelated dates/locations. Columns latitude/longitude are
    optional for manual files; when supplied they describe the requested site.
    """
    target = utc_time(target_time)
    selected, selected_time, previous = [], None, None
    levels = set()
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'time_utc', 'speed_ms', 'dir_deg'}
        if not required.issubset(reader.fieldnames or ()) or not {'alt_m', 'pressure_hpa'}.intersection(reader.fieldnames or ()):
            raise ValueError('Wind CSV requires time_utc,speed_ms,dir_deg and alt_m or pressure_hpa')
        for line, row in enumerate(reader, 2):
            if cancelled():
                raise RuntimeError('cancelled')
            try:
                stamp = utc_time(row['time_utc'])
                if previous is not None and stamp < previous:
                    raise ValueError('Rows must be sorted by time_utc')
                previous = stamp
                if stamp > target:
                    break
                if stamp != selected_time:
                    selected, levels, selected_time = [], set(), stamp
                if target - stamp >= timedelta(hours=1):
                    continue
                for key, site in (('latitude', latitude), ('longitude', longitude)):
                    if row.get(key) and site is not None and abs(_number(row[key], key) - float(site)) > .03:
                        raise ValueError('Wind cache belongs to a different launch site; fetch or load matching data')
                pressure = row.get('pressure_hpa')
                altitude = row.get('alt_m')
                if bool(pressure) == bool(altitude):
                    raise ValueError('Specify exactly one of alt_m or pressure_hpa')
                key = ('p' if pressure else 'z', _number(pressure or altitude, 'level'))
                if key in levels:
                    raise ValueError('Duplicate wind level')
                levels.add(key)
                if len(levels) > 500:
                    raise ValueError('Too many wind levels in one timestamp (limit 500)')
                speed, direction = _number(row['speed_ms'], 'speed'), _number(row['dir_deg'], 'direction')
                if speed < 0 or not 0 <= direction <= 360:
                    raise ValueError('Speed must be non-negative; direction must be 0–360 degrees FROM true north')
                z = pressure_altitude_agl(pressure, station_pressure_pa, temperature_c) if pressure else key[1]
                if not pressure and z < 0:
                    raise ValueError('alt_m must be metres above ground (>=0)')
                if z >= 0:
                    selected.append(dict(alt_m=z, speed_ms=speed, dir_deg=direction % 360))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'Wind CSV row {line}: {exc}') from exc
    if not selected:
        raise ValueError('No above-ground wind data for the target hour; check UTC time, pressure and CSV coverage')
    return sorted(selected, key=lambda n: n['alt_m'])


def fetch_wind_csv(path, latitude, longitude, start, end, *, cancelled=lambda: False,
                   progress=lambda *_: None, get=requests.get):
    """Fetch at most 24 hours per request; publish complete CSV atomically.

    Forecast API availability is enforced by the provider (past 92 / next 16
    days at most). Failed/cancelled downloads preserve the previous file.
    """
    begin, finish = utc_time(start), utc_time(end)
    if finish < begin or finish - begin > timedelta(days=108):
        raise ValueError('Invalid weather interval (maximum 108 days)')
    if any(t.minute or t.second or t.microsecond for t in (begin, finish)):
        raise ValueError('API start/end must be whole hours')
    lat, lon = _number(latitude, 'latitude'), _number(longitude, 'longitude')
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError('Invalid launch coordinates')
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dest.name + '.', suffix='.tmp', dir=dest.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            cursor = begin
            while cursor <= finish:
                if cancelled():
                    raise RuntimeError('cancelled')
                last = min(finish, cursor + timedelta(hours=23))
                variables = [f'wind_{kind}_{level}hPa' for level in LEVELS for kind in ('speed', 'direction')]
                with get('https://api.open-meteo.com/v1/forecast', params=dict(
                        latitude=lat, longitude=lon, hourly=','.join(variables),
                        start_hour=cursor.strftime('%Y-%m-%dT%H:%M'), end_hour=last.strftime('%Y-%m-%dT%H:%M'),
                        timezone='GMT', wind_speed_unit='ms'), timeout=(10, 30)) as response:
                    response.raise_for_status()
                    data = response.json()
                if data.get('error'):
                    raise ValueError(data.get('reason', 'Weather API error'))
                hourly, units = data['hourly'], data['hourly_units']
                times = hourly['time']
                expected = [cursor + timedelta(hours=i) for i in range(int((last-cursor).total_seconds()/3600)+1)]
                if times != [t.strftime('%Y-%m-%dT%H:%M') for t in expected]:
                    raise ValueError('Weather API returned incomplete or unexpected timestamps')
                for i, t in enumerate(expected):
                    for level in LEVELS:
                        skey, dkey = f'wind_speed_{level}hPa', f'wind_direction_{level}hPa'
                        if units.get(skey) != 'm/s' or units.get(dkey) != '°':
                            raise ValueError('Unexpected weather API wind units')
                        speed, direction = _number(hourly[skey][i], skey), _number(hourly[dkey][i], dkey)
                        if speed < 0 or not 0 <= direction <= 360:
                            raise ValueError('Invalid API wind values')
                        writer.writerow(dict(time_utc=t.isoformat(), pressure_hpa=level,
                                             speed_ms=speed, dir_deg=direction, latitude=lat,
                                             longitude=lon, source='Open-Meteo'))
                f.flush()
                progress(last.isoformat())
                cursor = last + timedelta(hours=1)
                del data, hourly, times
        if cancelled():
            raise RuntimeError('cancelled')
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return str(dest)
