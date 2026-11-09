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


def test_vote_rows_sort_the_parties_and_add_the_rest():
    """The value of each party, from the largest, without the rest of the vote (Luis, 2026-11-09); without a previous
    delivery, no change."""
    rows = run_js('pc.voteRows(%s, ["PSOE", "PP", "VOX", "ERC"])' % json.dumps(POLL))
    assert rows == [{'name': 'PP', 'value': 32.6, 'change': None}, {'name': 'PSOE', 'value': 25.6, 'change': None},
                    {'name': 'VOX', 'value': 19.3, 'change': None}]
    assert run_js('pc.voteRows({PP: 30, PSOE: 30}, ["PSOE", "PP"])')[0]['name'] == 'PSOE'  # ties keep the party order


def test_vote_rows_carry_the_change_against_the_previous_delivery():
    rows = run_js('pc.voteRows({PP: 32.6, PSOE: 25.6, UP: 1.2}, ["PP", "PSOE", "UP"], {PP: 31.4, PSOE: 25.6})')
    assert [(r['name'], r['change']) for r in rows] == [('PP', 1.2), ('PSOE', 0), ('UP', None)]


def test_change_label_colours_the_change_and_writes_nothing_without_one():
    labels = run_js('[1.2, -0.6, 0, 0.04, null].map((change) => pc.changeLabel({change}))')
    assert labels[0] == {'text': '▲ 1,2', 'color': '#2ca02c', 'tooltip': '▲ 1,2 frente a la entrega anterior'}
    assert labels[1]['text'] == '▼ 0,6' and labels[1]['color'] == '#d62728'
    assert labels[2]['text'] == '–' and labels[3]['text'] == '–' and labels[2]['color'] == '#5a5f72'  # no change
    assert labels[4] is None  # no previous delivery, or no figure in it: nothing, not even a dash


def test_seat_changes_compare_the_midpoints_of_the_ranges():
    changes = run_js('pc.seatChanges({PP: [136, 138], PSOE: [99, 101], VOX: [60, 60]}, {PP: [130, 134], VOX: [62, 64], BNG: [2, 2]})')
    assert changes == {'PP': 5, 'VOX': -3}  # 137 − 132, 60 − 63; PSOE has no range before
    assert run_js('pc.seatChanges({PP: [136, 138]}, null)') == {} and run_js('pc.seatChanges(null, {PP: [1, 2]})') == {}


def test_poll_seats_take_the_midpoints_label_the_ranges_and_count_the_rest():
    seats = run_js('pc.pollSeats({PP: [136, 138], PSOE: [100, 100], VOX: [67, 69]}, 350)')
    assert seats == {'seats': {'PP': 137, 'PSOE': 100, 'VOX': 68}, 'labels': {'PP': '136–138', 'PSOE': '100', 'VOX': '67–69'}, 'rest': 45}
    assert run_js('pc.pollSeats({PP: [200, 200], PSOE: [160, 160]}, 350)')['rest'] is None  # over the chamber: nothing to fill
    assert run_js('pc.pollSeats(null, 350)') == {'seats': {}, 'labels': {}, 'rest': None}


def test_highlight_series_ring_each_party_of_the_poll():
    series = run_js('pc.highlightSeries(%s, ["PP", "ERC"], {PP: "#1d84ce"})' % json.dumps(POLL))
    assert len(series) == 1 and series[0]['name'] == 'PP' and series[0]['id'] == 'highlight:PP' and series[0]['silent'] is True
    # a ring a little bigger than a poll point (6 px), so that it does not hide the points around it
    assert series[0]['symbolSize'] == 11
    assert series[0]['itemStyle'] == {'color': 'transparent', 'borderColor': '#1d84ce', 'borderWidth': 2}
    assert series[0]['data'][0][1] == 32.6 and series[0]['tooltip'] == {'show': False}
