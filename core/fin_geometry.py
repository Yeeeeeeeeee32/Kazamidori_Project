"""Fin outline validation shared by the editor, file loader and flight model."""
import math


class FinGeometryError(ValueError):
    pass


def validate_fin_points(raw):
    """Metres: root leading edge (0,0), x aft, y outward; close implicitly."""
    if not isinstance(raw, (list, tuple)) or not 3 <= len(raw) <= 200:
        raise FinGeometryError('Fin outline needs 3–200 vertices')
    try:
        points = [[float(x), float(y)] for x, y in raw]
    except (TypeError, ValueError, OverflowError):
        raise FinGeometryError('Each fin vertex must be [x, y] in metres') from None
    if any(not math.isfinite(v) or abs(v) > 10 for p in points for v in p):
        raise FinGeometryError('Fin coordinates must be finite and within ±10 m')
    eps = 1e-10
    if abs(points[0][0]) > eps or abs(points[0][1]) > eps:
        raise FinGeometryError('First vertex must be the root leading edge (0, 0)')
    if points[-1][0] <= eps or abs(points[-1][1]) > eps:
        raise FinGeometryError('Last vertex must be the root trailing edge (positive x, 0)')
    if any(p[1] <= eps for p in points[1:-1]):
        raise FinGeometryError('Intermediate vertices must be outside the body (y > 0)')
    if len({tuple(p) for p in points}) != len(points):
        raise FinGeometryError('Duplicate vertices; do not repeat (0, 0) to close the outline')
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def on(a, b, p):
        return abs(cross(a,b,p)) <= eps and all(min(a[k],b[k])-eps <= p[k] <= max(a[k],b[k])+eps for k in (0,1))
    edges = list(zip(points, points[1:] + points[:1]))
    for i, (a, b) in enumerate(edges):
        for j in range(i+1, len(edges)):
            if j == i+1 or (i == 0 and j == len(edges)-1):
                continue
            c, d = edges[j]
            if (cross(a,b,c)*cross(a,b,d) < 0 and cross(c,d,a)*cross(c,d,b) < 0) or any((on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b))):
                raise FinGeometryError('Fin outline must not cross or touch itself')
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in edges)) / 2
    if area <= 1e-10:
        raise FinGeometryError('Fin outline must have positive area')
    return points


def fin_dimensions(points):
    points = validate_fin_points(points)
    span = max(y for _,y in points)
    tip_x = [x for x,y in points if abs(y-span) < 1e-10]
    return dict(fin_root=points[-1][0], fin_tip=max(tip_x)-min(tip_x),
                fin_span=span, fin_sweep=min(tip_x))


def validate_fin_options(data):
    options = {}
    if any(k in data for k in ('fin_shape', 'fin_points')):
        shape = data.get('fin_shape', 'freeform' if data.get('fin_points') else 'trapezoid')
        if shape not in ('trapezoid', 'freeform'):
            raise FinGeometryError('Fin shape must be trapezoid or freeform')
        options['fin_shape'] = shape
        options['fin_points'] = validate_fin_points(data.get('fin_points')) if shape == 'freeform' else None
        if shape == 'freeform':
            options.update(fin_dimensions(options['fin_points']))
    for key, low, high in (('fin_sweep', -10, 10), ('fin_cant', -90, 90)):
        if key in data:
            if key == 'fin_sweep' and data[key] is None:
                if options.get('fin_shape') != 'freeform':
                    options[key] = None
                continue
            if isinstance(data[key], bool):
                raise FinGeometryError(f'{key} must be numeric')
            value = float(data[key])
            if not math.isfinite(value) or not low <= value <= high:
                raise FinGeometryError(f'{key} must be between {low} and {high}')
            if key != 'fin_sweep' or options.get('fin_shape') != 'freeform':
                options[key] = value
    return options
