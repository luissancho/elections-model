"""Evolución de las publicaciones (`web/dist/js/charts/evolution.js` y `pages/evolution-wire.js`): un run por día, los
puntos de cada partido, el eje redondeado a unidades y la opción ECharts, comprobados con `node` (se salta sin él).
Los módulos de gráficos leen `window.echarts` al cargarse, así que el arnés carga el ECharts empaquetado antes."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIST = os.path.join(ROOT, 'web', 'dist')
ECHARTS = os.path.join(DIST, 'vendor', 'echarts-5.6.0.min.js')
EVOLUTION_JS = pathlib.Path(DIST, 'js', 'charts', 'evolution.js').as_uri()
BLOCKS_JS = pathlib.Path(DIST, 'js', 'blocks.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')

RUNS = [
    {'run_id': 'b05', 'run_at': '2026-10-05T12:00:00Z', 'as_of': '2026-10-05', 'backfill': True,
     'nowcast': {'parties': [{'name': 'PP', 'pct': 32.71, 'seats': 140}, {'name': 'PSOE', 'pct': 26.37, 'seats': 107}, {'name': 'VOX', 'pct': 17.93, 'seats': 63}]}},
    {'run_id': 'b06', 'run_at': '2026-10-06T12:00:00Z', 'as_of': '2026-10-06', 'backfill': True,
     'nowcast': {'parties': [{'name': 'PP', 'pct': 32.64, 'seats': 139}, {'name': 'PSOE', 'pct': 26.41, 'seats': 107}]}},
    {'run_id': 'l06', 'run_at': '2026-10-06T09:30:00Z', 'as_of': '2026-10-06',
     'nowcast': {'parties': [{'name': 'PP', 'pct': 32.6, 'seats': 139}, {'name': 'PSOE', 'pct': 26.45, 'seats': 108}, {'name': 'VOX', 'pct': 17.95, 'seats': 63}]}},
    {'run_id': 'l07a', 'run_at': '2026-10-07T08:00:00Z', 'as_of': '2026-10-07',
     'nowcast': {'parties': [{'name': 'PP', 'pct': 32.5, 'seats': 138}, {'name': 'PSOE', 'pct': 26.5, 'seats': 108}, {'name': 'VOX', 'pct': 18.0, 'seats': 64}]}},
    {'run_id': 'l07b', 'run_at': '2026-10-07T18:00:00Z', 'as_of': '2026-10-07',
     'nowcast': {'parties': [{'name': 'PP', 'pct': 32.4, 'seats': 138}, {'name': 'PSOE', 'pct': 26.55, 'seats': 108}, {'name': 'VOX', 'pct': 18.1, 'seats': 64}]}},
    {'run_id': 'x08', 'run_at': '2026-10-08T23:30:00Z',
     'nowcast': {'parties': [{'name': 'PP', 'pct': 32.3, 'seats': 137}, {'name': 'PSOE', 'pct': 26.6, 'seats': 109}, {'name': 'VOX', 'pct': None, 'seats': None}]}},
]
PARTIES = ['PP', 'PSOE', 'VOX']
DATA = {'runs': None, 'parties': PARTIES, 'selected': ['PP', 'PSOE'], 'colors': {'PP': '#1e5aa8', 'PSOE': '#e3001b', 'VOX': '#63be21'}}


def run_js(expr):
    """Evalúa `expr` con `e` (la evolución), `b` (los bloques) y `runs` (el fijo) definidos y devuelve su JSON (las
    funciones de la opción se pierden en la serialización, como en `JSON.stringify`)."""
    script = (
        'import {{createRequire}} from "node:module"; '
        'const echarts = createRequire(import.meta.url)({}); globalThis.window = {{echarts}}; '
        'const e = await import("{}"); const b = await import("{}"); const runs = {}; '
        'console.log(JSON.stringify({})); process.exit(0);'
    ).format(json.dumps(ECHARTS), EVOLUTION_JS, BLOCKS_JS, json.dumps(RUNS), expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout)


def option(**over):
    """`evolutionOption` sobre los runs diarios del fijo, con `over` encima de `DATA`."""
    return run_js('e.evolutionOption({...%s, runs: e.dailyRuns(runs), ...%s})' % (json.dumps(DATA), json.dumps(over)))


def test_daily_runs_keep_one_run_per_day_live_over_backfill_and_the_last_live():
    assert run_js('e.dailyRuns(runs).map((run) => run.run_id)') == ['b05', 'l06', 'l07b', 'x08']
    assert run_js('e.runDay(runs[5])') == '2026-10-08'  # without `as_of`: the day of `run_at`
    assert run_js('e.dailyRuns([])') == []


def test_evolution_points_skip_the_runs_without_the_party_and_carry_the_run_index():
    points = run_js('e.evolutionPoints(e.dailyRuns(runs), %s, "pct")' % json.dumps(PARTIES))
    assert [p['name'] for p in points] == PARTIES
    assert [p[1] for p in points[0]['points']] == [32.71, 32.6, 32.4, 32.3] and [p[2] for p in points[0]['points']] == [0, 1, 2, 3]
    assert [p[1] for p in points[2]['points']] == [17.93, 17.95, 18.1]  # VOX: null in the last run
    assert run_js('e.evolutionPoints(e.dailyRuns(runs), ["PP"], "seats")[0].points.map((p) => p[1])') == [140, 139, 138, 137]


def test_evolution_bounds_round_to_whole_units_around_the_visible_data():
    assert run_js('e.evolutionBounds({min: 31.53, max: 32.71})') == [31, 33]
    assert run_js('e.evolutionBounds({min: 8, max: 8})') == [8, 9]
    assert run_js('e.evolutionBounds({min: 32, max: 33})') == [32, 33]
    assert run_js('e.evolutionBounds({min: Infinity, max: -Infinity})') == [None, None]


def test_evolution_option_draws_a_line_per_party_with_hollow_retrospective_points():
    opt = option()
    assert [s['name'] for s in opt['series']] == PARTIES and all(s['type'] == 'line' for s in opt['series'])
    first = opt['series'][0]['data'][0]
    assert first['value'][1] == 32.71 and first['itemStyle'] == {'color': '#fff', 'borderColor': '#1e5aa8', 'borderWidth': 2}
    assert opt['series'][0]['data'][1][1:] == [32.6, 1]  # a live point is a plain `[time, value, index]`
    assert opt['legend'] == {'show': False, 'selectedMode': True, 'type': 'scroll', 'top': 0, 'data': PARTIES,
                             'selected': {'PP': True, 'PSOE': True, 'VOX': False}}
    assert opt['dataZoom'][0]['type'] == 'slider' and opt['xAxis']['type'] == 'time'
    assert run_js('typeof e.evolutionOption({...%s, runs: e.dailyRuns(runs)}).yAxis.min' % json.dumps(DATA)) == 'function'
    assert 'minInterval' not in opt['yAxis'] and option(field='seats')['yAxis']['minInterval'] == 1
    assert run_js('e.evolutionOption({runs: [], parties: []}).series') == []


def test_evolution_image_option_shows_a_plain_legend_of_the_names_shown_and_freezes_the_zoom():
    img = run_js('e.evolutionImageOption({...%s, runs: e.dailyRuns(runs)}, {startValue: 1, endValue: 2})' % json.dumps(DATA))
    assert img['legend'] == {'show': True, 'type': 'plain', 'selectedMode': False, 'top': 0, 'data': ['PP', 'PSOE']}
    assert img['dataZoom'] == [{'type': 'inside', 'filterMode': 'filter', 'disabled': True, 'startValue': 1, 'endValue': 2}]
    assert img['grid']['bottom'] == 8 and img['grid']['top'] == 40
