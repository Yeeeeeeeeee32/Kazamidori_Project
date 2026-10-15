"""Import a single-stage OpenRocket design, with explicit data provenance.

No Java/OpenRocket installation or network access is needed. Geometry is read
from the design; mass properties are estimates from the component tree, not
claims to reproduce OpenRocket's mass calculator. Unsupported components never
silently contribute zero mass. Motors are excluded from airframe mass/CG.
See OpenRocket's file_specification and rocketcomponent/position/AxialMethod.
"""
import math
from pathlib import Path

import numpy as np

from core.airframe_parameters import resolve_airframe
from utils.openrocket_fins import read_openrocket_xml, load_openrocket_fin_sets


def _number(element, key, default=None):
    text = element.findtext(key)
    if text is None or text.strip() == 'auto':
        if default is None:
            raise ValueError(f'{element.tag}: {key} cannot be resolved')
        return float(default)
    # New OR versions persist the resolved radius after the `auto` keyword.
    value = float(text.removeprefix('auto ').strip())
    if not math.isfinite(value):
        raise ValueError(f'{element.tag}: {key} is not finite')
    return value


def _children(element):
    return list(element.find('subcomponents')) if element.find('subcomponents') is not None else []


def _length(element):
    if element.tag in ('trapezoidfinset', 'ellipticalfinset'):
        return _number(element, 'rootchord')
    if element.tag == 'freeformfinset':
        return float(element.findall('finpoints/point')[-1].attrib['x'])
    return _number(element, 'length', _number(element, 'packedlength', 0))


def _position(element, parent_pos, parent_length, previous_end):
    offset = element.find('axialoffset')
    if offset is None:
        offset = element.find('position')
    if offset is None:
        return previous_end if element.tag in ('nosecone', 'bodytube', 'transition') else parent_pos
    method = offset.get('method', offset.get('type', 'top'))
    value = float(offset.text)
    if not math.isfinite(value):
        raise ValueError('Non-finite component position')
    if method == 'absolute':
        return value
    if method == 'after':
        return previous_end + value
    fractions = {'top': 0, 'middle': .5, 'center': .5, 'bottom': 1}
    if method not in fractions:
        raise ValueError(f'Unsupported axial position method: {method}')
    return parent_pos + value + fractions[method] * (parent_length - _length(element))


def _polygon(points):
    pairs = list(zip(points, points[1:] + points[:1]))
    cross = [a[0]*b[1] - b[0]*a[1] for a, b in pairs]
    area2 = sum(cross)
    if abs(area2) < 1e-12:
        raise ValueError('Zero fin area')
    return abs(area2)/2, sum((a[0]+b[0])*c for (a,b),c in zip(pairs,cross))/(3*area2)


def _nose_volume_cg(element, radius):
    """Numerical axial-shell estimate including the aft shoulder and end cap."""
    length = _length(element)
    if length <= 0 or radius <= 0 or element.findtext('isflipped') == 'true':
        raise ValueError('Unsupported nose geometry')
    shape = element.findtext('shape')
    x = (np.arange(2048) + .5) * length / 2048
    if shape == 'conical':
        r = radius*x/length
    elif shape == 'ellipsoid':
        r = radius*np.sqrt(1-(1-x/length)**2)
    elif shape == 'ogive' and abs(_number(element, 'shapeparameter', 1)-1) < 1e-8:
        rho = (radius**2+length**2)/(2*radius)
        r = np.sqrt(rho**2-(length-x)**2)+radius-rho
    elif shape == 'haack' and abs(_number(element, 'shapeparameter', 0)) < 1e-8:
        theta = np.arccos(1-2*x/length)
        r = radius*np.sqrt((theta-np.sin(2*theta)/2)/math.pi)
    else:
        raise ValueError(f'Unsupported nose shape: {shape}')
    thickness = _number(element, 'thickness', 0)
    inner = np.zeros_like(r) if element.findtext('filled') == 'true' else np.maximum(0, r-thickness)
    dv = math.pi*(r*r-inner*inner)*length/2048
    volume, moment = float(dv.sum()), float((dv*x).sum())
    sr = _number(element, 'aftshoulderradius', 0)
    sl = _number(element, 'aftshoulderlength', 0)
    st = _number(element, 'aftshoulderthickness', 0)
    sv = math.pi*(sr*sr-max(0,sr-st)**2)*sl
    volume += sv
    moment += sv*(length+sl/2)
    if element.findtext('aftshouldercapped') == 'true':
        cap = math.pi*max(0,sr-st)**2*min(st,sl)
        volume += cap
        moment += cap*(length+sl-min(st,sl)/2)
    if volume <= 0:
        raise ValueError('Nose material volume is zero or unknown')
    return volume, moment/volume


def _component_mass_cg(element, context, fin_shapes):
    pos, parent = context[element]
    length = _length(element)
    kind = element.tag
    material = element.find('material')
    density = float(material.get('density')) if material is not None else None
    if density is not None and (not math.isfinite(density) or density < 0):
        raise ValueError('Invalid material density')
    mass, cg = None, length/2
    if kind in ('rocket', 'stage'):
        mass = 0.
    elif kind == 'masscomponent':
        mass = _number(element, 'mass')
    elif kind == 'nosecone':
        volume, cg = _nose_volume_cg(element, _number(element, 'aftradius'))
        mass = volume*density if density is not None else None
    elif kind in ('bodytube', 'innertube', 'tubecoupler', 'launchlug', 'centeringring', 'bulkhead', 'engineblock'):
        key = 'radius' if kind in ('bodytube','launchlug') else 'outerradius'
        parent_radius = None
        if parent is not None and parent.tag in ('bodytube','innertube','tubecoupler'):
            pk = 'radius' if parent.tag == 'bodytube' else 'outerradius'
            parent_radius = _number(parent, pk)-_number(parent,'thickness',0)
        outer = _number(element, key, parent_radius)
        if kind == 'bulkhead':
            inner = 0.
        elif kind in ('centeringring','engineblock'):
            # Auto inner radius surrounds an inner tube that overlaps this ring.
            tubes = [e for e,(p,par) in context.items() if par is parent and e.tag == 'innertube'
                     and p < pos+length and p+_length(e) > pos]
            auto_inner = max((_number(t,'outerradius') for t in tubes), default=0.)
            inner = _number(element,'innerradius',auto_inner)
        else:
            inner = 0 if element.findtext('filled') == 'true' else outer-_number(element,'thickness')
        if not 0 <= inner <= outer or outer <= 0:
            raise ValueError('Invalid tube/ring radii')
        mass = math.pi*(outer**2-inner**2)*length*density if density is not None else None
    elif element in fin_shapes:
        f = fin_shapes[element]
        points = f.get('fin_points') or [[0,0],[f['fin_sweep'],f['fin_span']],
                    [f['fin_sweep']+f['fin_tip'],f['fin_span']],[f['fin_root'],0]]
        area, cg = _polygon(points)
        tab_area = _number(element,'tabheight',0)*_number(element,'tablength',0)
        tabs = element.findall('tabposition')
        if tab_area:
            tab = tabs[-1] if tabs else None
            if tab is None:
                raise ValueError('Fin tab position is missing')
            tab_length = _number(element,'tablength')
            method = tab.get('relativeto','top')
            fraction = {'top':0,'front':0,'middle':.5,'center':.5,'bottom':1,'end':1}.get(method)
            if fraction is None:
                raise ValueError('Unknown fin tab position')
            tab_cg = float(tab.text)+fraction*(length-tab_length)+tab_length/2
            cg = (cg*area+tab_cg*tab_area)/(area+tab_area)
        if _number(element,'filletradius',0) > 0 or element.findtext('crosssection','square') != 'square':
            raise ValueError('Fin fillets / shaped cross-sections need measured mass and CG')
        count = int(element.findtext('fincount', element.findtext('instancecount','1')))
        mass = (area+tab_area)*_number(element,'thickness')*count*density if density is not None else None
    elif kind == 'parachute':
        line = element.find('linematerial')
        if density is not None and line is not None:
            mass = math.pi*(_number(element,'diameter')/2)**2*density
            mass += _number(element,'linecount')*_number(element,'linelength')*float(line.get('density'))
    elif kind == 'shockcord':
        mass = _number(element,'cordlength')*density if density is not None else None
    if mass is None:
        raise ValueError(f'{element.findtext("name",kind)}: 質量を計算できない部品です')
    if kind not in ('rocket','stage') and element not in fin_shapes:
        count = int(_number(element,'instancecount',1))
        cg += (count-1)*_number(element,'instanceseparation',0)/2
        mass *= count
    return mass, pos+cg


def load_openrocket_airframe(path):
    root = read_openrocket_xml(path)
    rocket = root.find('rocket')
    if rocket is None:
        raise ValueError('OpenRocket file has no rocket')
    configs = rocket.findall('motorconfiguration')
    selected = next((c for c in configs if c.get('default') == 'true'), None)
    if selected is None and len(configs) == 1:
        selected = configs[0]
    if selected is None and len(configs) > 1:
        raise ValueError('OpenRocketで使用するモーター構成を既定にして保存してください。')
    configid = selected.get('configid') if selected is not None else None
    inactive = {int(s.get('number')) for s in selected.findall('stage') if s.get('active') == 'false'} if selected is not None else set()
    stages = [s for i,s in enumerate(_children(rocket)) if i not in inactive and _children(s)]
    if len(stages) != 1 or stages[0].tag != 'stage':
        raise ValueError('機体全体の取込は有効な単段ロケットに対応しています。')
    stage = stages[0]
    external = _children(stage)
    if any(e.tag not in ('nosecone','bodytube') for e in external):
        raise ValueError('段差・ボートテール・ポッド付き機体は全体取込に未対応です。')
    noses = [e for e in external if e.tag == 'nosecone']
    bodies = [e for e in external if e.tag == 'bodytube']
    if len(noses) != 1 or not bodies or external[0] is not noses[0]:
        raise ValueError('先端ノーズと胴体を持つ機体が必要です。')
    nose = noses[0]
    radius = _number(bodies[0],'radius',_number(nose,'aftradius'))
    if any(abs(_number(b,'radius',radius)-radius)>1e-8 for b in bodies) or abs(_number(nose,'aftradius',radius)-radius)>1e-8:
        raise ValueError('直径が一定でない胴体は全体取込に未対応です。')
    shape = nose.findtext('shape')
    nose_shape = {'conical':'conical','ogive':'ogive','ellipsoid':'elliptical','haack':'vonKarman'}.get(shape)
    if nose_shape is None or nose.findtext('isflipped') == 'true':
        raise ValueError(f'未対応のノーズ形状です: {shape}')
    if (shape == 'ogive' and abs(_number(nose,'shapeparameter',1)-1)>1e-8 or
            shape == 'haack' and abs(_number(nose,'shapeparameter',0))>1e-8):
        raise ValueError('このノーズ形状係数はKazamidoriのモデルに変換できません。')
    # Drop inactive stages before finding/merging fins.
    for s in list(rocket.find('subcomponents')):
        if s is not stage:
            rocket.find('subcomponents').remove(s)
    fins = load_openrocket_fin_sets(path, root=root, include_elements=True)
    if len(fins) != 1 or not 3 <= fins[0]['fin_count'] <= 8:
        raise ValueError('全体取込には3〜8枚の対称フィン1組が必要です。')
    fin = fins[0]
    fin_shapes = {e:fin for e in fin['_elements']}
    context = {rocket:(0,None), stage:(0,rocket)}
    def locate(parent, start, length):
        end = start
        for e in _children(parent):
            if e.tag in ('podset','parallelstage','tubefinset'):
                raise ValueError(f'未対応の構成です: {e.tag}')
            pos = _position(e,start,length,end)
            context[e] = (pos,parent)
            locate(e,pos,_length(e))
            end = pos+_length(e)
    locate(stage,0,sum(_length(e) for e in external))
    previous = 0.
    for e in external:
        if abs(context[e][0]-previous)>1e-8:
            raise ValueError('胴体の重なり・間隔がある構成は全体取込に未対応です。')
        previous = context[e][0]+_length(e)
    length = previous
    direct = {k:v for k,v in fin.items() if k not in ('name','_elements')}
    direct.update(length=length,radius=radius,nose_length=_length(nose),nose_shape=nose_shape,
                  fin_pos=context[fin['_elements'][0]][0])
    if not 0 <= direct['fin_pos'] <= length-direct['fin_root']+1e-8:
        raise ValueError('フィンの翼根が機体全長の外にあります。Kazamidoriで位置を自動補正せず取込を中止しました。')
    calculated, warnings = {}, []

    def mass_tree(e):
        children = [mass_tree(c) for c in _children(e)]
        own_error = None
        try:
            mass, cg = _component_mass_cg(e,context,fin_shapes)
        except ValueError as exc:
            mass, cg, own_error = None, None, str(exc)
        overridden_mass = e.find('overridemass') is not None
        overridden_cg = e.find('overridecg') is not None
        include_mass = e.findtext('overridesubcomponentsmass', e.findtext('overridesubcomponents','false')) == 'true'
        include_cg = e.findtext('overridesubcomponentscg', e.findtext('overridesubcomponents','false')) == 'true'
        override_mass = _number(e,'overridemass') if overridden_mass else None
        override_cg = context[e][0]+_number(e,'overridecg') if overridden_cg else None
        if override_mass is not None and override_mass < 0:
            raise ValueError('Negative mass override')
        # Parent mass/CG overrides that include children can resolve unknown parts.
        if overridden_mass and include_mass and overridden_cg and include_cg:
            return override_mass,override_cg,[]
        if overridden_mass and not include_mass:
            mass = override_mass
        if overridden_cg and not include_cg:
            cg = override_cg
        errors = [err for _,_,errs in children for err in errs]
        if mass is None or (mass != 0 and cg is None):
            errors.append(own_error or '部品質量/CGが不明です')
        pieces = [(mass,cg)] + [(m,c) for m,c,_ in children]
        total = sum(m for m,_ in pieces) if all(m is not None for m,_ in pieces) else None
        if total is not None and all(m == 0 or c is not None for m,c in pieces):
            combined_cg = sum(m*(c or 0) for m,c in pieces)/total if total else context[e][0]
        else:
            combined_cg = None
        if overridden_mass and include_mass:
            total = override_mass
        if overridden_cg and include_cg:
            combined_cg = override_cg
        return total,combined_cg,errors if total is None or combined_cg is None else []

    mass,cg,errors = mass_tree(rocket)
    if mass is not None and mass > 0:
        calculated['mass'] = mass
    if cg is not None:
        calculated['cg'] = cg
    for key,tag,flag in (('mass','overridemass','overridesubcomponentsmass'),('cg','overridecg','overridesubcomponentscg')):
        for e in (rocket,stage):
            if e.find(tag) is not None and e.findtext(flag,e.findtext('overridesubcomponents','false')) == 'true':
                direct[key] = _number(e,tag)+(context[e][0] if key=='cg' else 0)
                break
    if calculated:
        warnings.append('質量・CGの自動計算は部品形状と材質による推定です（モーターを除く）。ノーズは数値積分、フィンは平板近似。')
    missing = [key for key in ('mass','cg') if key not in calculated and key not in direct]
    if missing:
        warnings.extend(dict.fromkeys(errors))
        warnings.append(f'全構成部品を計算できないため、{", ".join(missing)} は手入力が必要です。')
    motors = []
    for e,(pos,_) in context.items():
        mount = e.find('motormount')
        if mount is None:
            continue
        choices = [m for m in mount.findall('motor') if m.get('configid') == configid]
        if len(choices) == 1:
            motors.append((e,mount,choices[0]))
    motor_info = ''
    if len(motors) == 1:
        e,mount,motor = motors[0]
        direct['motor_pos'] = context[e][0]+_length(e)+_number(mount,'overhang',0)
        delay = motor.findtext('delay')
        if delay and delay != 'none':
            direct['backfire_delay'] = _number(motor,'delay')
        motor_info = ' '.join(filter(None,[motor.findtext('manufacturer'),motor.findtext('designation')]))
    elif len(motors)>1:
        raise ValueError('複数モーター構成は全体取込に未対応です。')
    warnings.append('モーター燃焼後質量と推力曲線は.ork内の型番から推測しません。Motor Specificationsで指定してください。')
    parachutes = [e for e in context if e.tag == 'parachute']
    parachute = {}
    if len(parachutes) == 1:
        p = parachutes[0]
        parachute['area'] = math.pi*(_number(p,'diameter')/2)**2
        parachute['cd'] = _number(p,'cd',.8)
        warnings.append('パラシュートの面積・Cdを取得。開傘ラグは別途入力してください（OpenRocketのdeploydelayとは別の値です）。')
    elif parachutes:
        warnings.append('パラシュートが複数あるため自動選択しません。Load Parachuteで使用する値を指定してください。')
    data = dict.fromkeys(('mass','cg','motor_pos','motor_dry_mass','backfire_delay'))
    data['provenance'] = dict(version=1, file=Path(path).name, name=rocket.findtext('name',''),
                              ork=direct, calculated=calculated, warnings=warnings,
                              motor=motor_info, parachute=parachute)
    return resolve_airframe(data)
