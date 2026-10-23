"""Paridad entre `web/dist/js/kde.js` (densidades del gráfico de crestas de Escaños) y `scipy.stats.gaussian_kde`
con el ancho de banda de Scott (se salta sin `node`)."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from scipy.stats import gaussian_kde

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KDE_JS = os.path.join(ROOT, 'web', 'dist', 'js', 'kde.js')
DIST = {'n_seats': 350, 'parties': ['PP', 'PSOE', 'VOX', 'UPN'],
        'seats': [[140, 106, 65, 1], [137, 110, 62, 1], [145, 100, 70, 1], [133, 112, 61, 1], [150, 95, 72, 1],
                  [139, 108, 60, 1], [141, 104, 66, 1], [136, 109, 63, 1]]}

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    script = 'import * as k from "file://{}"; const dist = {}; console.log(JSON.stringify({}));'.format(
        KDE_JS, json.dumps(DIST), expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_scott_bandwidth_and_density_match_scipy():
    values = np.array([row[0] for row in DIST['seats']], dtype=float)
    kde = gaussian_kde(values)  # Scott por defecto
    expected_bw = kde.factor * values.std(ddof=1)
    grid = [130, 135, 140, 145, 150, 155]
    got = run_js('[k.scottBandwidth([{}]), k.density([{}], {}, {})]'.format(
        ','.join(str(int(v)) for v in values), ','.join(str(int(v)) for v in values), json.dumps(grid), expected_bw))
    assert got[0] == pytest.approx(expected_bw)
    assert got[1] == pytest.approx(kde(np.array(grid, dtype=float)).tolist(), rel=1e-9)


def test_bandwidth_floor_counts_grid_and_range():
    assert run_js('k.scottBandwidth([1, 1, 1, 1])') == 0.5 and run_js('k.scottBandwidth([3])') == 0.5
    assert run_js('[...k.counts([5, 3, 5, 1]).entries()]') == [[1, 1], [3, 1], [5, 2]]
    assert run_js('k.seatGrid(0, 2, 5)') == [0, 0.5, 1, 1.5, 2]
    # Un escaño de margen a cada lado, nunca por debajo de 0, y un ancho mínimo para las distribuciones estrechas
    assert run_js('k.seatRange({min: 133, max: 150})') == [132, 151]
    assert run_js('k.seatRange({min: 0, max: 1})') == [0, 6] and run_js('k.seatRange({min: 1, max: 1}, 4)') == [0, 4]


def test_ridge_rows_scale_each_row_to_its_peak_over_its_own_range():
    data = run_js('k.ridgeData(dist, ["PP", "UPN", "NOEXISTE"], {majority: 176})')
    assert [row['name'] for row in data] == ['PP', 'UPN']  # el partido ausente se salta
    pp, upn = data
    assert (pp['x0'], pp['x1']) == (132, 151) and pp['grid'][0] == 132 and pp['grid'][-1] == 151 and len(pp['grid']) == 200
    assert max(pp['y']) == pytest.approx(1.0) and min(pp['y']) >= 0
    assert pp['median'] == 139.5 and pp['lo'] == 133 and pp['hi'] == 150 and pp['min'] == 133 and pp['n'] == 8
    assert sum(pp['pmf'].values()) == pytest.approx(1.0) and pp['pmf']['140'] == 0.125
    # UPN siempre con 1 escaño: rango mínimo, ancho de banda mínimo y curva finita con el pico en 1
    assert (upn['x0'], upn['x1']) == (0, 6) and upn['bw'] == 0.5 and max(upn['y']) == pytest.approx(1.0)
    assert all(np.isfinite(upn['y'])) and upn['pmf'] == {'1': 1.0}


def test_ridge_rows_carry_the_central_seats_when_given():
    """Cada fila lleva sus escaños del reparto central si se dan, y `null` si no."""
    rows = run_js('k.ridgeData(dist, ["PP", "PSOE"], {central: {PP: 141}})')
    assert rows[0]['central'] == 141 and rows[1]['central'] is None


def test_ridge_blocks_sum_the_parties_of_each_block():
    data = run_js('k.ridgeBlocks(dist, [{name: "Derecha", parties: ["PP", "VOX"]}, {name: "Vacio", parties: ["X"]}], {majority: 176})')
    assert [row['name'] for row in data] == ['Derecha']
    derecha = data[0]
    sums = sorted(row[0] + row[2] for row in DIST['seats'])
    assert derecha['min'] == sums[0] and derecha['max'] == sums[-1] and (derecha['x0'], derecha['x1']) == (sums[0] - 1, sums[-1] + 1)
    assert derecha['pmf'][str(sums[0])] == 0.125


def test_ridge_rows_can_share_one_axis():
    data = run_js('k.ridgeData(dist, ["PP", "UPN"], {majority: 176, range: [0, 210]})')
    assert [(row['x0'], row['x1']) for row in data] == [(0, 210), (0, 210)]
    assert data[0]['grid'][0] == 0 and data[0]['grid'][-1] == 210 and len(data[0]['grid']) == 200
    assert max(data[1]['y']) == pytest.approx(1.0)  # cada fila sigue a su propia escala
