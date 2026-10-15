"""Read fin outlines from OpenRocket ZIP/XML files without changing airframe data.

Format: https://openrocket.readthedocs.io/en/latest/dev_guide/file_specification.html
Only outline, count and cant are imported; mass and axial mounting stay local.
"""
import gzip
import json
import math
import zipfile
from pathlib import Path

import defusedxml.ElementTree as ET
from defusedxml.common import DefusedXmlException
from core.fin_geometry import FinGeometryError, validate_fin_options

MAX_XML = 5 * 1024 * 1024


def read_openrocket_xml(path):
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
        if magic.startswith(b'PK'):
            with zipfile.ZipFile(path) as z:
                candidates = [info for info in z.infolist() if Path(info.filename).name == 'rocket.ork']
                if not candidates:
                    raise FinGeometryError('OpenRocket archive has no rocket.ork')
                info = candidates[0]
                if info.file_size > MAX_XML:
                    raise FinGeometryError('OpenRocket XML exceeds 5 MB')
                with z.open(info) as f:
                    payload = f.read(MAX_XML+1)
        else:
            opener = gzip.open if magic.startswith(b'\x1f\x8b') else open
            with opener(path, 'rb') as f:
                payload = f.read(MAX_XML+1)
        if len(payload) > MAX_XML:
            raise FinGeometryError('OpenRocket XML exceeds 5 MB')
        root = ET.fromstring(payload)
        if root.tag != 'openrocket':
            raise FinGeometryError('Not an OpenRocket design file')
        return root
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile, ET.ParseError, DefusedXmlException) as exc:
        raise FinGeometryError(f"Cannot read OpenRocket file: {exc}") from exc


def load_openrocket_fin_sets(path, *, root=None, include_elements=False):
    try:
        root = read_openrocket_xml(path) if root is None else root
        fins = []
        parents = {child: parent for parent in root.iter() for child in parent}
        for element in root.iter():
            if element.tag not in ('trapezoidfinset', 'freeformfinset', 'ellipticalfinset'):
                continue
            name = element.findtext('name', 'Fins')
            try:
                count = int(element.findtext('fincount', element.findtext('instancecount', '4')))
                if not 1 <= count <= 8:
                    raise FinGeometryError('Supported source fin count is 1–8')
                result = dict(name=name, fin_count=count,
                              fin_cant=float(element.findtext('cant', '0')))
                if element.tag == 'trapezoidfinset':
                    result.update(fin_shape='trapezoid', fin_root=float(element.findtext('rootchord')),
                                  fin_tip=float(element.findtext('tipchord')),
                                  fin_span=float(element.findtext('height')),
                                  fin_sweep=float(element.findtext('sweeplength', '0')))
                    if not all(math.isfinite(result[k]) for k in ('fin_root','fin_tip','fin_span')) or result['fin_root'] <= 0 or result['fin_tip'] < 0 or result['fin_span'] <= 0:
                        raise FinGeometryError('Invalid trapezoid dimensions')
                else:
                    if element.tag == 'freeformfinset':
                        points = [[float(p.attrib['x']), float(p.attrib['y'])] for p in element.findall('finpoints/point')]
                    else:
                        chord, span = float(element.findtext('rootchord')), float(element.findtext('height'))
                        if chord <= 0 or span <= 0:
                            raise FinGeometryError('Invalid elliptical dimensions')
                        points = [[chord/2*(1-math.cos(math.pi*i/32)), span*math.sin(math.pi*i/32)] for i in range(33)]
                        points[-1][1] = 0
                        result['name'] += ' (ellipse: 32 segments)'
                    result.update(fin_shape='freeform', fin_points=points)
                result.update(validate_fin_options(result))
                mount = element.find('axialoffset')
                if mount is None:
                    mount = element.find('position')
                result['_mount'] = (id(parents[element]),
                                    tuple(sorted(mount.attrib.items())) if mount is not None else (),
                                    mount.text if mount is not None else '')
                result['_elements'] = [element]
                result['_rotation'] = float(element.findtext('angleoffset', element.findtext('rotation', '0'))) % 360
                if not math.isfinite(result['_rotation']):
                    raise FinGeometryError('Invalid fin rotation')
                fins.append(result)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                raise FinGeometryError(f'{name}: {exc}') from exc
        if not fins:
            raise FinGeometryError('No supported fin sets found (trapezoidal, freeform or elliptical)')
        # OR can store an evenly spaced set as separate single-fin components.
        # Merge only identical outlines on the same mount with uniform spacing.
        grouped = {}
        for entry in fins:
            if entry['fin_count'] == 1:
                key = (entry['_mount'], json.dumps({k:v for k,v in entry.items()
                        if not k.startswith('_') and k not in ('name','fin_count')}, sort_keys=True))
                grouped.setdefault(key, []).append(entry)
        removed = set()
        for group in grouped.values():
            angles = sorted(entry['_rotation'] for entry in group)
            count = len(group)
            if 3 <= count <= 8 and all(abs((angles[(i+1)%count]-angles[i])%360 - 360/count) < 1e-5 for i in range(count)):
                group[0]['fin_count'] = count
                group[0]['_elements'] = [entry['_elements'][0] for entry in group]
                group[0]['name'] += f' ({count} evenly spaced fins)'
                removed.update(id(entry) for entry in group[1:])
        return [{k:v for k,v in entry.items() if not k.startswith('_') or (include_elements and k == '_elements')}
                for entry in fins if id(entry) not in removed]
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile, ET.ParseError, DefusedXmlException) as exc:
        raise FinGeometryError(f'Cannot read OpenRocket fins: {exc}') from exc
