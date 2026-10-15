import gzip
import zipfile

import pytest

from core.fin_geometry import FinGeometryError, validate_fin_points
from utils.openrocket_fins import load_openrocket_fin_sets

POINTS = [[0,0], [.02,.03], [.05,.03], [.04,0]]


def freeform(name='Freeform', count=3, rotation=0):
    points = ''.join(f'<point x="{x}" y="{y}"/>' for x,y in POINTS)
    return (f'<freeformfinset><name>{name}</name><fincount>{count}</fincount>'
            f'<rotation>{rotation}</rotation><cant>2.5</cant><position type="bottom">0</position>'
            f'<finpoints>{points}</finpoints></freeformfinset>')


def design(parts):
    return f'<openrocket version="1.10"><rocket><subcomponents>{parts}</subcomponents></rocket></openrocket>'.encode()


def test_trapezoid_import_preserves_sweep_and_degrees(tmp_path):
    path = tmp_path/'fins.ork'
    content = ('<trapezoidfinset><name>Tail</name><fincount>4</fincount>'
               '<rootchord>.04</rootchord><tipchord>.02</tipchord><height>.03</height>'
               '<sweeplength>.01</sweeplength><cant>2.5</cant></trapezoidfinset>')
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('preview.png',b'not XML')
        z.writestr('rocket.ork',design(content))
    fin, = load_openrocket_fin_sets(path)
    assert fin['fin_shape'] == 'trapezoid'
    assert fin['fin_sweep'] == .01
    assert fin['fin_cant'] == 2.5
    assert fin['fin_count'] == 4


@pytest.mark.parametrize('compressed',[False,True])
def test_freeform_raw_xml_and_gzip(tmp_path, compressed):
    path = tmp_path/'fins.ork'
    data = design(freeform())
    path.write_bytes(gzip.compress(data) if compressed else data)
    fin, = load_openrocket_fin_sets(path)
    assert fin['fin_points'] == POINTS
    assert fin['fin_root'] == .04
    assert fin['fin_tip'] == pytest.approx(.03)
    assert fin['fin_span'] == .03


def test_identical_evenly_spaced_single_fins_merge_into_symmetric_set(tmp_path):
    path=tmp_path/'fins.ork'
    path.write_bytes(design(''.join(freeform(f'Fin {i}',1,a) for i,a in enumerate((0,120,-120)))))
    fin, = load_openrocket_fin_sets(path)
    assert fin['fin_count'] == 3
    assert fin['fin_points'] == POINTS
    path.write_bytes(design(''.join(freeform(f'Fin {i}',1,a) for i,a in enumerate((0,100,200)))))
    assert len(load_openrocket_fin_sets(path)) == 3


def test_multiple_sets_and_ellipse_are_read(tmp_path):
    path=tmp_path/'fins.ork'
    ellipse='<ellipticalfinset><name>Ellipse</name><fincount>3</fincount><rootchord>.06</rootchord><height>.03</height></ellipticalfinset>'
    path.write_bytes(design(freeform()+ellipse))
    a,b=load_openrocket_fin_sets(path)
    assert a['name'] == 'Freeform'
    assert len(b['fin_points']) == 33
    assert b['fin_root'] == pytest.approx(.06)
    assert b['fin_span'] == pytest.approx(.03)


@pytest.mark.parametrize('content', [b'broken', b'<RockSimDocument/>', design(''),
    b'<!DOCTYPE openrocket [<!ENTITY test "x">]><openrocket>&test;</openrocket>'])
def test_corrupt_or_unsupported_file_rejected(tmp_path, content):
    path=tmp_path/'bad.ork'
    path.write_bytes(content)
    with pytest.raises(FinGeometryError):
        load_openrocket_fin_sets(path)


@pytest.mark.parametrize('points', [[], [[0,0],[.03,.03],[.01,.03],[.04,0]],
    [[0,0],[.02,-.03],[.04,0]], [[0,0],[float('nan'),.03],[.04,0]],
    [[0,0],[.02,.03],[.04,0],[0,0]]])
def test_invalid_outlines_rejected(points):
    with pytest.raises(FinGeometryError):
        validate_fin_points(points)
