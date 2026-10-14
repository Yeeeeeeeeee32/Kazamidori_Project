import csv
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from utils.wind_data import FIELDS, LEVELS, fetch_wind_csv, read_wind_snapshot

BASE = '2026-10-14T03:00:00Z'


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(time_utc=BASE, speed_ms=5, dir_deg=359, **row))


def test_only_target_hour_and_dynamic_pressure_levels_are_read(tmp_path):
    path = tmp_path/'wind.csv'
    write_csv(path, [dict(alt_m=3), dict(pressure_hpa=900)])
    profile = read_wind_snapshot(path, '2026-10-14T12:15:00+09:00')
    assert profile[0]['alt_m'] == 3
    assert profile[1]['alt_m'] == pytest.approx(988.49, abs=1)
    changed = read_wind_snapshot(path, BASE, 100000)
    assert changed[1]['alt_m'] < profile[1]['alt_m']
    with pytest.raises(ValueError, match='No above-ground'):
        read_wind_snapshot(path, '2026-10-14T04:00:00Z')
    with pytest.raises(ValueError, match='No above-ground'):
        read_wind_snapshot(path, '2026-10-14T02:00:00Z')


@pytest.mark.parametrize('rows,match', [
    ([dict(alt_m=3), dict(alt_m=3)], 'Duplicate'),
    ([dict(alt_m=3, pressure_hpa=900)], 'exactly one'),
    ([dict(alt_m='nan')], 'finite'),
    ([dict(alt_m=-10)], 'above ground'),
    ([dict(alt_m=3, latitude=30)], 'different launch site'),
])
def test_invalid_csv_does_not_silently_substitute_calm_wind(tmp_path, rows, match):
    path = tmp_path/'wind.csv'
    write_csv(path, rows)
    with pytest.raises(ValueError, match=match):
        read_wind_snapshot(path, BASE, latitude=35, longitude=139)


def fake_get(url, params, timeout):
    start = datetime.fromisoformat(params['start_hour'])
    end = datetime.fromisoformat(params['end_hour'])
    times = [(start+timedelta(hours=i)).strftime('%Y-%m-%dT%H:%M') for i in range(int((end-start).total_seconds()/3600)+1)]
    assert len(times) <= 24
    hourly, units = dict(time=times), {}
    for level in LEVELS:
        for kind, value, unit in [('speed', 5, 'm/s'), ('direction', 359, '°')]:
            key = f'wind_{kind}_{level}hPa'
            hourly[key], units[key] = [value]*len(times), unit
    response = MagicMock()
    response.__enter__.return_value = response
    response.json.return_value = dict(hourly=hourly, hourly_units=units)
    return response


def test_fetch_is_chunked_and_round_trips_csv(tmp_path):
    get = MagicMock(side_effect=fake_get)
    path = tmp_path/'wind.csv'
    fetch_wind_csv(path, 35, 139, BASE, '2026-10-16T03:00:00Z', get=get)
    assert get.call_count == 3
    result = read_wind_snapshot(path, '2026-10-16T03:00:00Z', latitude=35, longitude=139)
    assert len(result) == len(LEVELS)
    assert result[0]['speed_ms'] == 5


def test_api_failure_preserves_old_file_and_removes_partial_download(tmp_path):
    path = tmp_path/'wind.csv'
    path.write_text('previous data')
    get = MagicMock(side_effect=[fake_get('', dict(start_hour='2026-10-14T03:00', end_hour='2026-10-15T02:00'), 30), OSError('offline')])
    with pytest.raises(OSError, match='offline'):
        fetch_wind_csv(path, 35, 139, BASE, '2026-10-15T03:00:00Z', get=get)
    assert path.read_text() == 'previous data'
    assert list(tmp_path.iterdir()) == [path]


def test_fetch_cancellation_preserves_existing_file(tmp_path):
    path = tmp_path/'wind.csv'
    path.write_text('previous')
    with pytest.raises(RuntimeError, match='cancelled'):
        fetch_wind_csv(path, 35, 139, BASE, BASE, cancelled=lambda: True)
    assert path.read_text() == 'previous'


def test_wrong_api_units_and_missing_samples_rejected(tmp_path):
    for invalid in ('units', 'gap', 'null'):
        def get(url, params, timeout):
            response = fake_get(url, params, timeout)
            data = response.json.return_value
            if invalid == 'units':
                data['hourly_units']['wind_speed_1000hPa'] = 'km/h'
            elif invalid == 'gap':
                data['hourly']['time'] = []
            else:
                data['hourly']['wind_speed_1000hPa'][0] = None
            return response
        with pytest.raises((ValueError, TypeError)):
            fetch_wind_csv(tmp_path/'bad.csv', 35, 139, BASE, BASE, get=get)
        assert not (tmp_path/'bad.csv').exists()
