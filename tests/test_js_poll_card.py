"""La lógica pura de la tarjeta de un sondeo (`web/dist/js/poll-card.js`): filas del voto, escaños de la casa y anillo
del sondeo sobre sus series, comprobado con `node` (se salta sin él). El módulo importa `charts/series.js`, que lee
`window.echarts` al cargarse, así que el arnés carga el ECharts empaquetado antes."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIST = os.path.join(ROOT, 'web', 'dist')
ECHARTS = os.path.join(DIST, 'vendor', 'echarts-5.6.0.min.js')
MODULE = pathlib.Path(DIST, 'js', 'poll-card.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    """Evalúa `expr` con `pc` (el módulo) definido y devuelve su JSON."""
    script = (
        'import {{createRequire}} from "node:module"; '
        'const echarts = createRequire(import.meta.url)({}); globalThis.window = {{echarts}}; '
        'const pc = await import("{}"); console.log(JSON.stringify({})); process.exit(0);'
    ).format(json.dumps(ECHARTS), MODULE, expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT)

    return json.loads(out.stdout.strip().splitlines()[-1])


POLL = {'date': '2026-11-09', 'PP': 32.6, 'PSOE': 25.6, 'VOX': 19.3, 'ERC': None}


def test_vote_rows_carry_the_change_against_the_previous_delivery():
    rows = run_js('pc.voteRows({PP: 32.6, PSOE: 25.6, UP: 1.2}, ["PP", "PSOE", "UP"], {}, {PP: 31.4, PSOE: 25.6})')
    assert [(r['name'], r['change']) for r in rows] == [('PP', 1.2), ('PSOE', 0), ('UP', None), ('Otros', None)]
    assert all(r['change'] is None for r in run_js('pc.voteRows({PP: 32.6}, ["PP"], {})'))


def test_vote_rows_sort_the_parties_and_add_the_rest():
    rows = run_js('pc.voteRows(%s, ["PSOE", "PP", "VOX", "ERC"], {PP: 31.9, VOX: 18.0})' % json.dumps(POLL))
    assert rows == [{'name': 'PP', 'value': 32.6, 'avg': 31.9, 'change': None},
                    {'name': 'PSOE', 'value': 25.6, 'avg': None, 'change': None},
                    {'name': 'VOX', 'value': 19.3, 'avg': 18.0, 'change': None},
                    {'name': 'Otros', 'value': 22.5, 'avg': None, 'change': None}]
    full = run_js('pc.voteRows({PP: 60.0, PSOE: 39.7}, ["PP", "PSOE"])')
    assert [row['name'] for row in full] == ['PP', 'PSOE']  # a rest under 0.5 is not shown
    assert run_js('pc.voteRows({PP: 30, PSOE: 30}, ["PSOE", "PP"])')[0]['name'] == 'PSOE'  # ties keep the party order


def test_poll_seats_take_the_midpoints_label_the_ranges_and_count_the_rest():
    seats = run_js('pc.pollSeats({PP: [136, 138], PSOE: [100, 100], VOX: [67, 69]}, 350)')
    assert seats == {'seats': {'PP': 137, 'PSOE': 100, 'VOX': 68}, 'labels': {'PP': '136–138', 'PSOE': '100', 'VOX': '67–69'}, 'rest': 45}
    assert run_js('pc.pollSeats({PP: [200, 200], PSOE: [160, 160]}, 350)')['rest'] is None  # over the chamber: nothing to fill
    assert run_js('pc.pollSeats(null, 350)') == {'seats': {}, 'labels': {}, 'rest': None}


def test_highlight_series_ring_each_party_of_the_poll():
    series = run_js('pc.highlightSeries(%s, ["PP", "ERC"], {PP: "#1d84ce"})' % json.dumps(POLL))
    assert len(series) == 1 and series[0]['name'] == 'PP' and series[0]['id'] == 'highlight:PP' and series[0]['silent'] is True
    assert series[0]['itemStyle'] == {'color': 'transparent', 'borderColor': '#1d84ce', 'borderWidth': 3}
    assert series[0]['data'][0][1] == 32.6 and series[0]['tooltip'] == {'show': False}
