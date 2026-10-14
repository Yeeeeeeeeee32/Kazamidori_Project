import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from utils.magnetic import declination_for_site, input_bearing, true_bearing


def test_wmm_receives_altitude_in_km_and_offline_cache_matches_site_date(tmp_path):
    path = tmp_path/'wmm.json'
    model = MagicMock()
    model.calculate.return_value.d = -7.5
    with patch('utils.magnetic.GeoMag', return_value=model):
        result = declination_for_site(35, 139, 250, cache_path=path, today=date(2026, 7, 2))
    assert result == -7.5
    kwargs = model.calculate.call_args.kwargs
    assert kwargs['alt'] == .25
    assert kwargs['time'] == pytest.approx(2026.49863, abs=.00001)
    with patch('utils.magnetic.GeoMag', side_effect=RuntimeError('model unavailable')):
        assert declination_for_site(35, 139, 250, cache_path=path, today=date(2026, 7, 2)) == -7.5
        with pytest.raises(RuntimeError):
            declination_for_site(36, 139, 250, cache_path=path, today=date(2026, 7, 2))


def test_true_magnetic_conversion_sign_and_wrap():
    assert true_bearing(5, 'magnetic', -7.5) == 357.5
    assert input_bearing(357.5, 'magnetic', -7.5) == 5
    assert true_bearing(5, 'true', -7.5) == 5


def test_corrupt_cache_is_rebuilt_and_calculation_error_is_not_zero(tmp_path):
    path = tmp_path/'wmm.json'
    path.write_text('broken')
    value = declination_for_site(35, 139, cache_path=path, today=date(2026, 1, 1))
    assert -10 < value < -5
    assert len(json.loads(path.read_text())) == 1
    with pytest.raises(ValueError):
        declination_for_site(999, 139, cache_path=path)
