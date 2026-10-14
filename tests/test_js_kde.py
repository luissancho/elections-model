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


def test_bandwidth_has_a_floor_and_counts_are_sorted():
    assert run_js('k.scottBandwidth([1, 1, 1, 1])') == 0.5 and run_js('k.scottBandwidth([3])') == 0.5
    assert run_js('[...k.counts([5, 3, 5, 1]).entries()]') == [[1, 1], [3, 1], [5, 2]]
    assert run_js('k.seatGrid(2, 0.5)') == [0, 0.5, 1, 1.5, 2]


def test_ridge_data_scales_each_party_to_its_peak_and_carries_the_quantiles():
    data = run_js('k.ridgeData(dist, ["PP", "UPN", "NOEXISTE"], {xMax: 200, step: 1, majority: 176})')
    assert data['grid'][0] == 0 and data['grid'][-1] == 200 and len(data['grid']) == 201
    assert [row['name'] for row in data['rows']] == ['PP', 'UPN']  # el partido ausente se salta
    pp, upn = data['rows']
    assert max(pp['y']) == pytest.approx(1.0) and len(pp['y']) == 201 and min(pp['y']) >= 0
    assert pp['median'] == 139.5 and pp['lo'] == 133 and pp['hi'] == 150 and pp['n'] == 8
    assert sum(pp['pmf'].values()) == pytest.approx(1.0) and pp['pmf']['140'] == 0.125
    # UPN siempre con 1 escaño: ancho de banda mínimo, curva finita con el pico en 1
    assert upn['bw'] == 0.5 and max(upn['y']) == pytest.approx(1.0) and upn['y'][1] == pytest.approx(1.0)
    assert all(np.isfinite(upn['y'])) and upn['pmf'] == {'1': 1.0}
