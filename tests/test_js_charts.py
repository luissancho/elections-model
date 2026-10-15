"""Opciones de ECharts de la diana (`web/dist/js/charts/bullseye.js`) y de las barras divergentes
(`charts/diverging.js`), comprobadas con `node` (se salta sin él). Los módulos de gráficos leen `window.echarts` al
cargarse, así que el arnés carga el ECharts empaquetado antes (sin `window`, para que zrender se sepa en node) y
después define `window` e importa el módulo."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIST = os.path.join(ROOT, 'web', 'dist')
ECHARTS = os.path.join(DIST, 'vendor', 'echarts-5.6.0.min.js')
BULLSEYE_JS = pathlib.Path(DIST, 'js', 'charts', 'bullseye.js').as_uri()
DIVERGING_JS = pathlib.Path(DIST, 'js', 'charts', 'diverging.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')

ROWS = [
    {'event': '2023-23J', 'scope': 'es', 'event_date': '2023-07-23', 'days': 7, 'Derecha': 0.7, 'Izquierda': -0.7},
    {'event': '2019-10N', 'scope': 'es', 'event_date': '2019-11-10', 'days': 5, 'Derecha': -2.1, 'Izquierda': 1.4},
    {'event': '2023-28M', 'scope': 'es-md', 'event_date': '2023-05-28', 'days': 4, 'Derecha': 0.8, 'Izquierda': None},
    {'event': '2021-4M', 'scope': 'es-md', 'event_date': '2021-05-04', 'days': 3, 'Derecha': 6.5, 'Izquierda': -4.2},
]


def run_js(expr, rows=ROWS):
    """Evalúa `expr` con `b` (diana), `d` (divergentes) y `rows` definidos y devuelve su JSON (las funciones de la
    opción se pierden en la serialización, como en `JSON.stringify`)."""
    script = (
        'import {{createRequire}} from "node:module"; '
        'const echarts = createRequire(import.meta.url)({}); globalThis.window = {{echarts}}; '
        'const b = await import("{}"); const d = await import("{}"); const rows = {}; '
        'console.log(JSON.stringify({})); process.exit(0);'
    ).format(json.dumps(ECHARTS), BULLSEYE_JS, DIVERGING_JS, json.dumps(rows), expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout)


def scatter(option):
    """Series de puntos y sus gemelas de etiquetas (silenciosas), en orden."""
    series = [s for s in option['series'] if s['type'] == 'scatter']
    return [s for s in series if not s.get('silent')], [s for s in series if s.get('silent')]


def test_bullseye_names_the_axes_at_the_edges_and_draws_small_points():
    option = run_js('b.bullseyeOption(rows, {blocks: ["Derecha", "Izquierda"], color: "#1d4f86"})')
    assert option['xAxis']['name'] == 'Derecha' and option['yAxis']['name'] == 'Izquierda'
    assert option['xAxis']['nameLocation'] == 'middle' and option['yAxis']['nameLocation'] == 'middle'
    # los ejes en los bordes (no sobre el cero) y la cruz del cero como series de línea, tras los tres anillos
    assert option['xAxis']['axisLine']['onZero'] is False and option['yAxis']['axisLine']['onZero'] is False
    assert option['xAxis']['axisLabel'] == {'showMinLabel': False, 'showMaxLabel': False}
    lines = [s for s in option['series'] if s['type'] == 'line']
    assert len(lines) == 5 and lines[3]['data'] == [[-11, 0], [11, 0]] and lines[4]['data'] == [[0, -11], [0, 11]]
    points, labels = scatter(option)
    assert len(points) == 1 and points[0]['symbolSize'] == 8 and points[0]['z'] == 10 and points[0]['color'] == '#1d4f86'
    assert points[0]['label'] == {'show': False} and 'tooltip' not in points[0]
    # la fila sin error en un bloque no se dibuja
    assert [p['row']['event'] for p in points[0]['data']] == ['2023-23J', '2019-10N', '2021-4M']
    # las etiquetas van en una gemela invisible y silenciosa, sin tooltip: solo los puntos lo abren
    assert len(labels) == 1 and labels[0]['itemStyle'] == {'opacity': 0} and labels[0]['tooltip'] == {'show': False}
    assert labels[0]['label']['show'] is True and labels[0]['label']['position'] == 'right' and labels[0]['labelLayout'] == {'hideOverlap': True}
    assert labels[0]['symbolSize'] == 8 and labels[0]['z'] == 10 and [p['row']['event'] for p in labels[0]['data']] == ['2023-23J', '2019-10N', '2021-4M']
    assert option['xAxis']['min'] == -11 and option['yAxis']['max'] == 11 and 'legend' not in option


def test_bullseye_draws_the_groups_on_top_over_the_others():
    option = run_js('b.bullseyeOption(rows, {blocks: ["Derecha", "Izquierda"], '
                    'groupOf: (row) => (row.scope === "es" ? "Generales" : "Autonómicas"), '
                    'colors: {Generales: "#1d4f86", "Autonómicas": "#c0672e"}, legend: true, '
                    'onTop: (group) => group === "Generales", alphaOf: (row) => (row.scope === "es" ? 1 : 0.5)})')
    points, labels = scatter(option)
    assert [s['name'] for s in points] == ['Generales', 'Autonómicas'] and [s['name'] for s in labels] == ['Generales', 'Autonómicas']
    assert [s['z'] for s in points] == [12, 10] and [s['z'] for s in labels] == [12, 10]
    assert points[0]['symbolSize'] == 8 and points[1]['color'] == '#c0672e' and labels[1]['color'] == '#c0672e'
    assert option['legend']['data'] == ['Generales', 'Autonómicas'] and option['grid']['bottom'] == 52
    # un punto atenuado: la opacidad en el punto y el color de la etiqueta en la gemela, nunca al revés (la gemela
    # debe seguir invisible)
    assert points[1]['data'][0]['itemStyle'] == {'opacity': 0.5} and 'label' not in points[1]['data'][0]
    assert labels[1]['data'][0]['label'] == {'color': 'rgba(26, 28, 34, 0.5)'} and 'itemStyle' not in labels[1]['data'][0]
    assert points[0]['data'][0]['itemStyle'] == {'opacity': 1} and labels[0]['data'][0]['label'] == {'color': 'rgba(26, 28, 34, 1)'}
    # sin `onTop`, todos los grupos al mismo nivel; sin `alphaOf`, los datos van sin estilos
    option = run_js('b.bullseyeOption(rows, {blocks: ["Derecha", "Izquierda"], groupOf: (row) => row.scope})')
    assert [s['z'] for s in option['series'] if s['type'] == 'scatter'] == [10, 10, 10, 10]
    points, labels = scatter(option)
    assert all(set(item) == {'value', 'row'} for series in points + labels for item in series['data'])


def test_diverging_grows_each_side_from_zero_as_magnitudes():
    rows = [{'name': 'GAD3', 'Derecha': 0.68, 'Izquierda': -0.68}, {'name': 'Sigma Dos', 'Derecha': None, 'Izquierda': 2.4},
            {'name': 'CIS', 'Derecha': -4.5, 'Izquierda': 3.9}]
    option = run_js('d.divergingOption(rows, {left: {key: "Izquierda", name: "Izquierda", color: "#c0392b"}, '
                    'right: {key: "Derecha", name: "Derecha", color: "#2c6fbb"}})', rows=rows)
    left, right = option['series']
    assert left['name'] == 'Izquierda' and right['name'] == 'Derecha' and left['stack'] == right['stack']
    assert left['data'] == [-0.68, -2.4, -3.9] and right['data'] == [0.68, None, 4.5]  # sin signo, cada lado al suyo
    assert left['color'] == '#c0392b' and right['color'] == '#2c6fbb'
    assert left['label']['position'] == 'left' and right['label']['position'] == 'right'
    assert left['markLine']['data'] == [{'xAxis': 0}]  # la línea del cero
    assert option['xAxis']['min'] == -option['xAxis']['max'] and option['xAxis']['max'] == pytest.approx(5.2)
    assert option['xAxis']['axisLabel']['showMinLabel'] is False
    assert option['yAxis']['data'] == ['GAD3', 'Sigma Dos', 'CIS'] and option['yAxis']['inverse'] is True
    assert option['yAxis']['axisLine']['onZero'] is False  # los nombres en el borde, no sobre el cero
    assert option['legend']['data'] == ['Izquierda', 'Derecha']
    # sin color, el azul neutro de las barras
    option = run_js('d.divergingOption(rows, {left: {key: "Izquierda", name: "I"}, right: {key: "Derecha", name: "D"}})', rows=rows)
    assert option['series'][0]['color'] == '#3a6ea5'
