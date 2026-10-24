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
HEMICYCLE_JS = pathlib.Path(DIST, 'js', 'charts', 'hemicycle.js').as_uri()
SERIES_JS = pathlib.Path(DIST, 'js', 'charts', 'series.js').as_uri()
BASE_JS = pathlib.Path(DIST, 'js', 'charts', 'base.js').as_uri()
RIDGE_JS = pathlib.Path(DIST, 'js', 'charts', 'ridge.js').as_uri()
PATH_JS = pathlib.Path(DIST, 'js', 'charts', 'path.js').as_uri()
WEIGHTS_JS = pathlib.Path(DIST, 'js', 'charts', 'weights.js').as_uri()
STACKED_JS = pathlib.Path(DIST, 'js', 'charts', 'stacked.js').as_uri()
CENSUS_JS = pathlib.Path(DIST, 'js', 'charts', 'census.js').as_uri()
SCATTER_JS = pathlib.Path(DIST, 'js', 'charts', 'scatter.js').as_uri()
SLOPE_JS = pathlib.Path(DIST, 'js', 'charts', 'slope.js').as_uri()
RADAR_JS = pathlib.Path(DIST, 'js', 'charts', 'radar.js').as_uri()
GROUPED_JS = pathlib.Path(DIST, 'js', 'charts', 'grouped.js').as_uri()
SERIES_WIRE_JS = pathlib.Path(DIST, 'js', 'pages', 'series-wire.js').as_uri()
MATRIX_JS = pathlib.Path(DIST, 'js', 'charts', 'matrix.js').as_uri()
RANKING_JS = pathlib.Path(DIST, 'js', 'charts', 'ranking.js').as_uri()
DISTANCE_JS = pathlib.Path(DIST, 'js', 'charts', 'distance.js').as_uri()
CATALOG_JS = pathlib.Path(DIST, 'js', 'catalog.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')

ROWS = [
    {'event': '2023-23J', 'scope': 'es', 'event_date': '2023-07-23', 'days': 7, 'Derecha': 0.7, 'Izquierda': -0.7},
    {'event': '2019-10N', 'scope': 'es', 'event_date': '2019-11-10', 'days': 5, 'Derecha': -2.1, 'Izquierda': 1.4},
    {'event': '2023-28M', 'scope': 'es-md', 'event_date': '2023-05-28', 'days': 4, 'Derecha': 0.8, 'Izquierda': None},
    {'event': '2021-4M', 'scope': 'es-md', 'event_date': '2021-05-04', 'days': 3, 'Derecha': 6.5, 'Izquierda': -4.2},
]


def run_js(expr, rows=ROWS):
    """Evalúa `expr` con `b` (diana), `d` (divergentes), `h` (hemiciclo), `s` (series), `base` (tema), `sw` (el
    cableado compartido del promedio), `c` (el catálogo), los gráficos del estudio CERA (`ce` censo, `sc`
    dispersión, `sl` pendientes, `ra` radar, `g` barras agrupadas), los del estudio del CIS (`mx` matriz, `rk` ranking, `ds` distancia) y `rows` definidos y devuelve su JSON (las
    funciones de la opción se pierden en la serialización, como en `JSON.stringify`)."""
    script = (
        'import {{createRequire}} from "node:module"; '
        'const echarts = createRequire(import.meta.url)({}); globalThis.window = {{echarts}}; '
        'const b = await import("{}"); const d = await import("{}"); const h = await import("{}"); '
        'const s = await import("{}"); const base = await import("{}"); const r = await import("{}"); '
        'const p = await import("{}"); const w = await import("{}"); const st = await import("{}"); '
        'const sw = await import("{}"); const c = await import("{}"); '
        'const ce = await import("{}"); const sc = await import("{}"); const sl = await import("{}"); '
        'const ra = await import("{}"); const g = await import("{}"); const mx = await import("{}"); '
        'const rk = await import("{}"); const ds = await import("{}"); const rows = {}; '
        'console.log(JSON.stringify({})); process.exit(0);'
    ).format(json.dumps(ECHARTS), BULLSEYE_JS, DIVERGING_JS, HEMICYCLE_JS, SERIES_JS, BASE_JS, RIDGE_JS, PATH_JS,
             WEIGHTS_JS, STACKED_JS, SERIES_WIRE_JS, CATALOG_JS, CENSUS_JS, SCATTER_JS, SLOPE_JS, RADAR_JS, GROUPED_JS,
             MATRIX_JS, RANKING_JS, DISTANCE_JS, json.dumps(rows), expr)
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
    assert option['xAxis']['axisLabel'] == {'color': '#5a5f72', 'fontSize': 10.5, 'showMinLabel': False, 'showMaxLabel': False}
    lines = [s for s in option['series'] if s['type'] == 'line']
    assert len(lines) == 5 and lines[3]['data'] == [[-11, 0], [11, 0]] and lines[4]['data'] == [[0, -11], [0, 11]]
    points, labels = scatter(option)
    assert len(points) == 1 and points[0]['symbolSize'] == 8 and points[0]['z'] == 10 and points[0]['color'] == '#1d4f86'
    assert points[0]['label'] == {'show': False} and 'tooltip' not in points[0]
    # la fila sin error en un bloque no se dibuja
    assert [p['row']['event'] for p in points[0]['data']] == ['2023-23J', '2019-10N', '2021-4M']
    # las etiquetas van en una gemela invisible y silenciosa, sin tooltip: sólo los puntos lo abren
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


def test_hemicycle_puts_the_labels_inside_and_marks_the_majority():
    option = run_js('h.hemicycleOption({PP: 140, PSOE: 107, VOX: 63, ERC: 9, BNG: 2}, {colors: {PP: "#1d84ce"}, nSeats: 350, majority: 176, width: 600, height: 330})')
    pies = [s for s in option['series'] if s['type'] == 'pie']
    assert len(pies) == 2 and pies[0]['startAngle'] == 180 and pies[0]['endAngle'] == 0
    # R = max(40, min(600 / 2 - 10, 330 - 36)) = 290; centro en [width / 2, height - 12]
    assert pies[0]['center'] == [300, 318] and pies[0]['radius'] == [150.8, 290]
    assert pies[1]['center'] == [300, 318] and pies[1]['radius'] == [145, 300]
    labels = {row['name']: row['label']['show'] for row in pies[0]['data']}
    assert labels == {'PP': True, 'PSOE': True, 'VOX': True, 'ERC': False, 'BNG': False}
    assert pies[0]['label']['position'] == 'inside' and pies[0]['labelLine']['show'] is False
    assert pies[0]['label']['rich']['seats']['fontSize'] == 24 and 'Outfit' in pies[0]['label']['rich']['seats']['fontFamily']
    assert pies[1]['silent'] is True and [round(d['value'], 2) for d in pies[1]['data']] == [175.65, 0.7, 173.65]
    texts = [g for g in option['graphic'] if g['type'] == 'text']
    # dos líneas bajo el total (Luis, 2026-10-20): «ESCAÑOS» y «MAYORÍA ABSOLUTA: 176», con 18 px entre ellas
    # («un poquito más de aire» que los 14 de una línea) y la pila subida otro tanto
    assert [g['style']['text'] for g in texts] == ['176', '350', 'ESCAÑOS', 'MAYORÍA ABSOLUTA: 176']
    assert [g['y'] for g in texts[1:]] == [318 - 48, 318 - 32, 318 - 14] and all(g['x'] == 300 for g in texts)
    assert [g for g in option['graphic'] if g['type'] == 'rect'][0]['shape'] == {'x': 284, 'y': 4, 'width': 32, 'height': 16, 'r': 8}
    assert pies[0]['data'][0]['itemStyle']['color'] == '#1d84ce' and pies[0]['data'][3]['itemStyle']['color'] == '#9e9e9e'
    assert option['tooltip']['borderRadius'] == 10


def test_hemicycle_writes_the_seat_delta_under_the_seats():
    """Con `deltas`, cada sector lleva su diferencia frente a la última elección bajo los escaños (flecha y magnitud, en
    el blanco de la etiqueta) y el tooltip la repite con el año; sin ella, la etiqueta de siempre."""
    data = run_js('h.hemicycleData({PP: 140, PSOE: 107}, {nSeats: 350, deltas: {PP: -25}})')
    assert [(d['name'], d['value'], d['delta']) for d in data] == [('PP', 140, -25), ('PSOE', 107, None)]
    assert run_js('h.hemicycleLabel({name: "PP", value: 140, data: {delta: -25}})') == '{name|PP}\n{seats|140}\n{delta|▼ 25}'
    assert run_js('h.hemicycleLabel({name: "PSOE", value: 107, data: {delta: null}})') == '{name|PSOE}\n{seats|107}'
    assert run_js('h.hemicycleLabel({name: "VOX", value: 33, data: {}})') == '{name|VOX}\n{seats|33}'
    option = run_js('h.hemicycleOption({PP: 140, PSOE: 107}, {nSeats: 350, majority: 176, width: 600, height: 330, deltas: {PP: -25}, year: 2023})')
    pie = [s for s in option['series'] if s['type'] == 'pie'][0]
    # un poco de aire sobre la flecha, bajo la cifra de escaños (Luis, 2026-10-17)
    assert pie['label']['rich']['delta'] == {'fontFamily': pie['label']['rich']['seats']['fontFamily'], 'fontWeight': 700, 'fontSize': 12, 'color': '#fff',
                                             'align': 'center', 'padding': [3, 0, 0, 0]}
    assert pie['data'][0]['delta'] == -25 and 'formatter' not in pie['label']  # la función no sobrevive al JSON


def test_stacked_bar_labels_each_segment_with_its_seats_and_delta():
    option = run_js('st.stackedOption([{name: "PP", seats: 140}, {name: "PSOE", seats: 107}, {name: "BNG", seats: 2}], {total: 350, majority: 176, deltas: {PP: -25, PSOE: 0}})')
    bars = [s for s in option['series'] if s['type'] == 'bar']
    # el nombre y los escaños en negrita y la flecha en peso normal (Luis, 2026-10-17); sin delta, sólo los escaños
    assert [s['label']['formatter'] for s in bars] == ['{seats|PP 140} {delta|▼ 25}', '{seats|PSOE 107} {delta|–}', '{seats|BNG 2}']
    assert all(s['label']['rich'] == {'seats': {'fontWeight': 700, 'color': '#fff'}, 'delta': {'fontWeight': 400, 'color': '#fff'}} for s in bars)
    assert [s['label']['show'] for s in bars] == [True, True, False] and option['xAxis']['max'] == 350
    assert [s['type'] for s in option['series']] == ['bar', 'bar', 'bar', 'line']
    plain = run_js('st.stackedOption([{name: "PP", seats: 140}], {total: 350})')
    assert plain['series'][0]['label']['formatter'] == '{seats|PP 140}' and len(plain['series']) == 1


def test_stacked_bar_draws_the_muted_segments_in_grey_with_muted_labels():
    """La barra de la calculadora de mayorías: lo que no se marca va en gris, con la etiqueta en el gris del texto."""
    option = run_js('st.stackedOption([{name: "PP", seats: 140}, {name: "PSOE", seats: 107}], '
                    '{total: 350, colors: {PP: "#1d84ce", PSOE: "#ef1c27"}, muted: ["PSOE"]})')
    pp, psoe = [s for s in option['series'] if s['type'] == 'bar']
    assert pp['color'] == '#1d84ce' and pp['label']['color'] == '#fff'
    assert psoe['color'] == '#d3d7df' and psoe['label']['color'] == '#5a5f72'
    assert psoe['label']['rich'] == {'seats': {'fontWeight': 700, 'color': '#5a5f72'}, 'delta': {'fontWeight': 400, 'color': '#5a5f72'}}


def test_hemicycle_shortens_the_centre_text_when_small():
    option = run_js('h.hemicycleOption({PP: 140, PSOE: 107}, {nSeats: 350, majority: 176, width: 358, height: 197})')
    assert [g['style']['text'] for g in option['graphic'] if g['type'] == 'text'] == ['176', '350', 'ESCAÑOS']


def test_hemicycle_without_majority_has_no_mark():
    option = run_js('h.hemicycleOption({A: 10, B: 5}, {nSeats: 15, width: 400, height: 220})')
    assert len([s for s in option['series'] if s['type'] == 'pie']) == 1
    assert [g['style']['text'] for g in option['graphic'] if g['type'] == 'text'] == ['15', 'ESCAÑOS']
    assert [g for g in option['graphic'] if g['type'] == 'rect'] == []


SERIES_ARGS = ("{series: {dates: ['2026-10-01', '2026-10-02'], parties: ['PP'], mean: {PP: [30, 31]}, lo: {PP: [29, 30]}, "
               "hi: {PP: [31, 32]}}, polls: {parties: ['PP'], polls: []}, projection: null, parties: ['PP'], selected: null, "
               "colors: {PP: '#1d84ce'}, asOf: '2026-10-02', when: '2026-10-02', window: 'all', legend: %s}")


def test_series_hidden_legend_keeps_the_component_and_the_theme():
    option = run_js('s.seriesOption(%s)' % (SERIES_ARGS % "'hidden'"))
    assert option['legend'] == {'show': False, 'selectedMode': True, 'type': 'scroll', 'top': 0, 'data': ['PP'], 'selected': {'PP': True}}
    assert option['grid']['top'] == 24 and option['yAxis']['axisLabel']['color'] == '#5a5f72'
    assert option['yAxis']['splitLine']['lineStyle']['color'] == 'rgba(0,0,0,.08)' and option['tooltip']['borderRadius'] == 10
    shown = run_js('s.seriesOption(%s)' % (SERIES_ARGS % 'true'))
    assert shown['legend']['show'] is True and shown['grid']['top'] == 40
    theme = run_js('base.THEME')
    assert theme['fontFamily'].startswith("'DM Sans'") and theme['headFamily'].startswith("'Outfit'") and theme['mutedColor'] == '#5a5f72'


RESULT_ARGS = SERIES_ARGS.replace("legend: %s", "legend: 'hidden', asOfLabel: 'A 6 días', result: {date: '2026-10-08', values: {PP: 31.5}}")


def test_series_result_draws_a_diamond_at_the_election_and_extends_the_axis():
    option = run_js('s.seriesOption(%s)' % RESULT_ARGS)
    election = run_js('s.dayTime("2026-10-08")')
    diamonds = [s for s in option['series'] if s['type'] == 'scatter' and s.get('symbol') == 'diamond']
    assert [(s['name'], s['data'], s['symbolSize'], s['z']) for s in diamonds] == [('PP', [[election, 31.5]], 12, 5)]
    assert diamonds[0]['itemStyle'] == {'borderColor': '#fff', 'borderWidth': 1.5}
    marks = option['series'][-1]['markLine']['data']
    assert [m['label']['formatter'] for m in marks] == ['A 6 días', 'Elección'] and marks[1]['xAxis'] == election
    assert option['dataZoom'][0]['endValue'] == election
    plain = run_js('s.seriesOption(%s)' % (SERIES_ARGS % "'hidden'"))
    assert not [s for s in plain['series'] if s.get('symbol') == 'diamond']
    assert [m['label']['formatter'] for m in plain['series'][-1]['markLine']['data']] == ['Estimación']
    assert run_js('s.axisDatesOf({dates: ["2026-10-01", "2026-10-02"]}, null, {date: "2026-10-08", values: {}})') == ['2026-10-01', '2026-10-02', '2026-10-08']
    assert run_js('s.axisDatesOf({dates: ["2026-10-01", "2026-10-02"]}, null, {date: "2026-10-02", values: {}})') == ['2026-10-01', '2026-10-02']


def test_series_window_of_three_months_keeps_the_result_in_view():
    option = run_js('s.seriesOption(%s)' % RESULT_ARGS.replace("window: 'all'", "window: '3m', anchor: '2026-10-08'"))
    assert option['dataZoom'][0]['endValue'] == run_js('s.dayTime("2026-10-08")')


def test_series_axis_follows_the_window_and_the_parties_shown():
    """El eje vertical no arranca en cero: se ajusta a los datos de la ventana y de los partidos visibles (Luis,
    2026-10-19), y sus marcas llevan un decimal cuando el tramo es tan estrecho que lo necesitan."""
    option = run_js('s.seriesOption(%s)' % (SERIES_ARGS % "'hidden'"))
    assert option['yAxis']['scale'] is True and 'min' not in option['yAxis'] and 'max' not in option['yAxis']
    assert option['dataZoom'][0]['filterMode'] == 'filter'  # la ventana recorta los datos y el eje se recalcula
    assert run_js('[s.axisPctLabel(32), s.axisPctLabel(32.2), s.axisPctLabel(0), s.axisPctLabel(32.000000001)]') == ['32%', '32,2%', '0%', '32%']

LAYER_ARGS = (SERIES_ARGS % "'hidden', projection: {dates: ['2026-10-02', '2026-10-03'], groups: {parties: {mean: {PP: [31, 31.5]}, "
              "lo: {PP: [30, 30]}, hi: {PP: [32, 33]}}}}, result: {date: '2026-10-08', values: {PP: 31.5}}, layers: %s")
LAYER_ARGS = LAYER_ARGS.replace("polls: {parties: ['PP'], polls: []}", "polls: {parties: ['PP'], polls: [{date: '2026-10-01', pollster: 'Casa A', PP: 30.5}]}")


def series_roles(option):
    """Papel de cada serie de la opción del promedio, deducido de su forma: `mean`, `band`, `poll`, `projection`,
    `result` y `marks`, sin repetir y ordenados."""
    roles = set()
    for entry in option['series']:
        if entry['name'] == 'marks':
            roles.add('marks')
        elif entry['type'] == 'scatter':
            roles.add('result' if entry.get('symbol') == 'diamond' else 'poll')
        elif entry.get('lineStyle', {}).get('opacity') == 0:
            roles.add('band')
        elif entry.get('lineStyle', {}).get('type') == 'dashed':
            roles.add('projection')
        else:
            roles.add('mean')
    return sorted(roles)


def test_series_layers_draw_the_polls_the_average_or_both():
    """«Sondeos / Promedio / Ambos» (Luis, 2026-10-23): la capa elegida decide qué se dibuja. Sólo sondeos: los
    puntos, el resultado y las marcas, sin líneas, bandas ni proyección; sólo promedio: sin puntos; ambas (por
    defecto): todo. El tooltip sigue a las series que quedan, y la imagen respeta la capa."""
    everything = ['band', 'marks', 'mean', 'poll', 'projection', 'result']
    assert series_roles(run_js('s.seriesOption(%s)' % (LAYER_ARGS % "'ambos'"))) == everything
    assert series_roles(run_js('s.seriesOption(%s)' % (SERIES_ARGS % "'hidden'"))) == ['band', 'marks', 'mean', 'poll']
    polls_only = run_js('s.seriesOption(%s)' % (LAYER_ARGS % "'sondeos'"))
    assert series_roles(polls_only) == ['marks', 'poll', 'result']
    assert [len(entry['data']) for entry in polls_only['series'] if entry['type'] == 'scatter'] == [1, 1]
    assert series_roles(run_js('s.seriesOption(%s)' % (LAYER_ARGS % "'promedio'"))) == ['band', 'marks', 'mean', 'projection', 'result']
    # the first series is now the poll points: its tooltip is the box of the poll, not a day of the average
    box = run_js("(() => { const o = s.seriesOption(%s); return o.tooltip.formatter({seriesIndex: 0, seriesName: 'PP', "
                 "dataIndex: 0, data: o.series[0].data[0]}); })()" % (LAYER_ARGS % "'sondeos'"))
    assert '<strong>Casa A</strong>' in box and 'PP 30,5' in box
    image = run_js('s.seriesImageOption(%s)' % (LAYER_ARGS % "'sondeos'"))
    assert series_roles(image) == ['marks', 'poll', 'result'] and image['legend']['data'] == ['PP']


def test_series_views_carry_the_result_of_each_view():
    initial = {'series': {'dates': ['2026-10-01'], 'parties': ['PP', 'VOX', 'PSOE'], 'mean': {'PP': [30], 'VOX': [10], 'PSOE': [28]},
                          'lo': {'PP': [29], 'VOX': [9], 'PSOE': [27]}, 'hi': {'PP': [31], 'VOX': [11], 'PSOE': [29]}},
               'polls': {'parties': ['PP', 'VOX', 'PSOE'], 'polls': []}, 'projection': None,
               'result': {'date': '2026-10-08', 'values': {'PP': 31, 'PSOE': 27.5}},
               'catalog': {'parties': [{'name': n, 'color': '#000000', 'fullname': n, 'block': None} for n in ('PP', 'VOX', 'PSOE')],
                           'bmaps': {'main': ['PP', 'PSOE'], 'blocks': {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE']}}}}
    views = run_js('sw.seriesViews(%s, new c.Catalog(%s.catalog))' % (json.dumps(initial), json.dumps(initial)))
    assert views['partidos']['result'] == initial['result'] and views['partidos']['defaults'] == ['PP', 'PSOE']
    assert views['bloques']['result'] == {'date': '2026-10-08', 'values': {'Derecha': 31, 'Izquierda': 27.5}}
    assert views['bloques']['parties'] == ['Derecha', 'Izquierda'] and views['bloques']['projection'] is None
    assert run_js('sw.seriesViews(%s, new c.Catalog({}))' % json.dumps({**initial, 'result': None}))['partidos']['result'] is None
    # the colour of a block is the one the page already uses for it (`block_colors`), else that of its first party
    painted = run_js('sw.seriesViews(%s, new c.Catalog(%s.catalog))' % (json.dumps({**initial, 'block_colors': {'Derecha': '#112233'}}), json.dumps(initial)))
    assert painted['bloques']['colors'] == {'Derecha': '#112233', 'Izquierda': '#000000'}


def test_ridge_rows_are_taller_on_desktop():
    assert run_js('r.ridgeRowHeight(1440)') == 110 and run_js('r.ridgeRowHeight(720)') == 110
    assert run_js('r.ridgeRowHeight(719)') == 64 and run_js('r.ridgeRowHeight(390)') == 64


PATH_POINTS = [
    {'event_date': '2021-05-04', 'scopes': ['es-md'], 'label': '4 may 2021 · Madrid', 'kind': 'regional', 'n_polls': 7, 'rating': 71.0},
    {'event_date': '2023-07-23', 'scopes': ['es'], 'label': 'Generales 2023', 'kind': 'general', 'n_polls': 14, 'rating': 80.0},
    {'event_date': '2024-05-12', 'scopes': ['es-ct'], 'label': '12 may 2024 · Cataluña', 'kind': 'regional', 'n_polls': 4, 'rating': None},
]
PATH_TODAY = {'event_date': '2026-10-15', 'label': 'Hoy', 'kind': 'today', 'rating': 82.4}


def test_rating_path_runs_from_zero_to_a_hundred_with_coloured_points_and_a_dashed_tail_to_today():
    option = run_js('p.ratingPathOption(rows, {today: %s, colors: {general: "#1d4f86", regional: "#c0672e"}})'
                    % json.dumps(PATH_TODAY), rows=PATH_POINTS)
    assert option['xAxis']['type'] == 'time' and option['yAxis']['min'] == 0 and option['yAxis']['max'] == 100
    names = [s['name'] for s in option['series']]
    assert names == ['Rating', 'Generales', 'Autonómicas', 'Hoy']
    line, general, regional, tail = option['series']
    # la línea une los puntos con rating (el de Cataluña, sin rating, no está) y no lleva símbolos
    assert line['type'] == 'line' and line['showSymbol'] is False and line['data'] == [['2021-05-04', 71.0], ['2023-07-23', 80.0]]
    assert general['type'] == 'scatter' and general['color'] == '#1d4f86' and [p['row']['label'] for p in general['data']] == ['Generales 2023']
    assert regional['type'] == 'scatter' and regional['color'] == '#c0672e' and [p['value'] for p in regional['data']] == [['2021-05-04', 71.0]]
    # el tramo final: discontinuo desde la última elección hasta hoy, con un punto hueco sólo en hoy
    assert tail['type'] == 'line' and tail['lineStyle']['type'] == 'dashed'
    assert [p['value'] for p in tail['data']] == [['2023-07-23', 80.0], ['2026-10-15', 82.4]]
    assert tail['data'][0]['symbol'] == 'none' and tail['data'][1]['symbol'] == 'emptyCircle'
    assert option['legend']['data'] == ['Generales', 'Autonómicas', 'Hoy'] and option['tooltip']['trigger'] == 'item'


def test_rating_path_without_today_or_regional_points_keeps_only_what_it_has():
    option = run_js('p.ratingPathOption(rows, {colors: {general: "#1d4f86", regional: "#c0672e"}})', rows=PATH_POINTS[1:2])
    assert [s['name'] for s in option['series']] == ['Rating', 'Generales'] and option['legend']['data'] == ['Generales']


def test_rating_path_ticks_show_the_year_on_new_year_and_the_day_otherwise():
    # ECharts places the ticks of a time axis at local midnight: 1 January must read as the year, not "31 dic"
    labels = run_js('[p.tickLabel(new Date(2024, 0, 1).getTime()), p.tickLabel(new Date(2024, 4, 12).getTime())]')
    assert labels == ['2024', '12 may']


# --- Fase 9: el gráfico de los pesos (`charts/weights.js`) ---
EXAMPLE = {
    'name': 'PP', 'color': '#1d84ce', 'date': '2026-10-08',
    'stats': {'date': '2026-10-08', 'mean': 33.1, 'lo': 32.4, 'hi': 33.8, 'err': 0.35, 'nobs': 2, 'neff': 1.8, 'bandwidth': 42.0},
    'rows': [
        {'date': '2026-09-24', 'pollster': 'GAD3', 'sponsor': 'ABC', 'sample_size': 1200, 'pct': 32.4, 'pct_adj': 32.9,
         'weight': 0.5, 'weight_kernel': 0.64, 'weight_over': 1.0, 'weight_sample': 0.8, 'weight_rating': 1.0},
        {'date': '2026-10-05', 'pollster': 'Sigma Dos', 'sponsor': None, 'sample_size': None, 'pct': 33.1, 'pct_adj': 33.0,
         'weight': 1.0, 'weight_kernel': 0.98, 'weight_over': 1.0, 'weight_sample': 1.0, 'weight_rating': 1.0},
    ],
    'line': {'dates': ['2026-09-24', '2026-10-01', '2026-10-08'], 'mean': [32.8, 32.95, 33.1],
             'lo': [32.1, 32.25, 32.4], 'hi': [33.5, 33.65, 33.8]},
    'kernel': {'dates': ['2026-09-24', '2026-10-01', '2026-10-08'], 'values': [0.946, 0.986, 1.0]},
}


def test_weights_option_sizes_the_polls_by_weight_and_draws_the_line_the_band_the_kernel_and_the_guides():
    option = run_js('w.weightsOption(%s)' % json.dumps(EXAMPLE))
    assert [s['name'] for s in option['series']] == ['Recta del día', 'Banda 95 %', 'Banda 95 %', 'Sondeos', 'Kernel', 'marks']
    line, band, points, kernel, marks = option['series'][0], option['series'][2], option['series'][3], option['series'][4], option['series'][5]
    assert line['color'] == '#1d84ce' and line['lineStyle']['width'] == 2 and line['data'][2][1] == 33.1
    assert band['areaStyle'] == {'color': '#1d84ce', 'opacity': 0.15} and band['stack'] == 'band:weights'
    assert [p['symbolSize'] for p in points['data']] == [13, 20]
    assert [p['itemStyle']['opacity'] for p in points['data']] == [0.625, 1] and points['data'][1]['value'][1:] == [33.1, 1]
    assert kernel['yAxisIndex'] == 1 and kernel['lineStyle']['type'] == 'dashed' and kernel['data'][2][1] == 1.0
    assert option['yAxis'][1]['max'] == 1 and option['yAxis'][1]['show'] is False and option['yAxis'][0]['scale'] is True
    assert option['yAxis'][0]['position'] == 'right' and option['grid']['right'] == 48 and option['grid']['left'] == 16   # los ticks del % a la derecha
    day = run_js('s.dayTime("2026-10-08")')
    assert marks['markLine']['data'] == [{'xAxis': day}, {'yAxis': 33.1}] and marks['markLine']['label'] == {'show': False}
    assert option['xAxis']['min'] == run_js('s.dayTime("2026-09-24")') and option['xAxis']['max'] == day
    assert option['legend']['data'] == ['Sondeos', 'Recta del día', 'Banda 95 %', 'Kernel'] and option['tooltip']['trigger'] == 'item'


def test_weights_poll_lines_name_the_sponsor_and_the_weight_factors():
    lines = run_js('w.pollLines(%s, "PP")' % json.dumps(EXAMPLE['rows'][0]))
    assert lines[0] == '<strong>GAD3 / ABC</strong>' and lines[1].endswith('· muestra 1200')   # es-ES: sin separador hasta 5 cifras
    assert lines[2:] == ['Publicada 32,4% · Corregida 32,9%', 'Peso 0,50', 'F(kernel) 0,64 · F(solape) 1,00 · F(muestra) 0,80 · F(rating) 1,00']
    lines = run_js('w.pollLines(%s, "PP")' % json.dumps(EXAMPLE['rows'][1]))
    assert lines[0] == '<strong>Sigma Dos</strong>' and lines[1].endswith('· muestra –')


def test_weights_option_gives_the_line_and_the_kernel_hover_symbols_for_their_tooltips():
    option = run_js('w.weightsOption(%s)' % json.dumps(EXAMPLE))
    for series in (option['series'][0], option['series'][4]):   # la recta y la campana
        assert series['showSymbol'] is True and series['symbol'] == 'circle' and series['symbolSize'] == 8
        assert series['itemStyle']['opacity'] == 0 and series['emphasis']['itemStyle']['opacity'] == 1


def test_weights_kernel_line_names_the_factor():
    assert run_js('w.kernelLine(0.946)') == 'F(kernel) 0,95'
    assert run_js('w.kernelLine(null)') == 'F(kernel) –'


def test_weights_option_can_draw_the_points_with_the_corrected_figure():
    # the Publicada/Corregida switch of the method page: the published figure by default, the one the line is fitted on
    option = run_js('w.weightsOption(%s)' % json.dumps(EXAMPLE))
    assert [p['value'][1] for p in option['series'][3]['data']] == [32.4, 33.1]
    option = run_js('w.weightsOption(%s, {field: "pct_adj"})' % json.dumps(EXAMPLE))
    assert [p['value'][1] for p in option['series'][3]['data']] == [32.9, 33.0]
    assert option['series'][3]['name'] == 'Sondeos' and [p['symbolSize'] for p in option['series'][3]['data']] == [13, 20]


def test_weights_value_axis_keeps_the_same_bounds_for_both_figures():
    # the union of both figures and the band (32.1 .. 33.8), half a point wider, rounded outwards: the axis stays still
    assert run_js('w.valueRange(%s, %s)' % (json.dumps(EXAMPLE['rows']), json.dumps(EXAMPLE['line']))) == {'min': 31, 'max': 35}
    for field in ('pct', 'pct_adj'):
        option = run_js('w.weightsOption(%s, {field: "%s"})' % (json.dumps(EXAMPLE), field))
        assert (option['yAxis'][0]['min'], option['yAxis'][0]['max']) == (31, 35), field
    # a corrected figure far from the published one widens the common range
    rows = [dict(EXAMPLE['rows'][0], pct_adj=38.2), EXAMPLE['rows'][1]]
    assert run_js('w.valueRange(%s, %s)' % (json.dumps(rows), json.dumps(EXAMPLE['line']))) == {'min': 31, 'max': 39}
    assert run_js('w.valueRange([], {mean: [], lo: [], hi: []})') == {}


# Gráficos del estudio CERA (`/analisis/voto-cera`)

def test_census_option_marks_every_date():
    marks = [{'date': '2022-10-21', 'label': 'Ley 20/2022'}, {'date': '2023-07-23', 'label': '23-J'},
             {'date': '2025-10-22', 'label': 'Fin del plazo'}, {'date': '2026-08-01', 'label': 'Censo 29-N'}]
    option = run_js('ce.censusOption(rows, %s)' % json.dumps(marks),
                    rows=[['2022-01-01', 2200000], ['2024-01-01', 2400000], ['2026-09-01', 2762829]])
    assert option['xAxis']['type'] == 'time' and 'axisLabel' in option['yAxis'] and option['yAxis']['scale'] is True
    [line] = option['series']
    assert line['type'] == 'line' and line['data'][2] == ['2026-09-01', 2762829] and line['smooth'] is True
    assert line['color'] == '#1d4f86' and 'areaStyle' in line and line['showSymbol'] is False
    # una línea vertical discontinua por marca, con su texto girado
    assert line['markLine']['data'] == [{'xAxis': m['date'], 'name': m['label']} for m in marks]
    assert line['markLine']['label']['rotate'] == 90 and line['markLine']['lineStyle']['type'] == 'dashed'
    assert run_js('ce.censusOption(rows, [], {color: "#123456"})', rows=[])['series'][0]['color'] == '#123456'
    assert run_js('ce.millions(2750000)') == '2,75 M'


def test_scatter_option_labels_only_the_named_provinces_and_draws_the_fit():
    rows = [{'name': 'Ourense', 'latam_share': 74.0, 'left_share': 29.7},
            {'name': 'Madrid', 'latam_share': 40.0, 'left_share': 50.0},
            {'name': 'Girona', 'latam_share': 23.0, 'left_share': 69.6}]
    option = run_js('sc.scatterOption(rows, {x: "latam_share", y: "left_share", name: "name", '
                    'labelled: ["Ourense", "Girona"], fit: [[20, 66], [80, 44]]})', rows=rows)
    points, fit = option['series']
    assert points['type'] == 'scatter' and points['color'] == '#1d4f86'
    assert [item['value'] for item in points['data']] == [[74.0, 29.7], [40.0, 50.0], [23.0, 69.6]]
    assert [item['name'] for item in points['data']] == ['Ourense', 'Madrid', 'Girona']
    assert [item['label']['show'] for item in points['data']] == [True, False, True]
    assert fit['type'] == 'line' and fit['data'] == [[20, 66], [80, 44]] and fit['lineStyle']['type'] == 'dashed'
    assert fit['showSymbol'] is False and fit['silent'] is True
    # sin recta, sólo los puntos; sin `labelled`, ninguna etiqueta
    option = run_js('sc.scatterOption(rows, {x: "latam_share", y: "left_share", name: "name"})', rows=rows)
    assert len(option['series']) == 1 and not any(item['label']['show'] for item in option['series'][0]['data'])


def test_slope_option_one_line_per_row_with_zero_reference():
    rows = [{'name': 'Aragón', 'from': 10.8, 'to': 9.2, 'color': '#e8912b'},
            {'name': 'Castilla y León', 'from': 8.9, 'to': -0.9, 'color': '#1d4f86'},
            {'name': 'Andalucía', 'from': 7.3, 'to': 4.9, 'color': '#2e9e5b'}]
    option = run_js('sl.slopeOption(rows, {labels: ["Comicio anterior", "2026"]})', rows=rows)
    lines = [s for s in option['series'] if s['type'] == 'line']
    assert len(lines) == 3 and [s['name'] for s in lines] == ['Aragón', 'Castilla y León', 'Andalucía']
    assert option['xAxis']['type'] == 'category' and option['xAxis']['data'] == ['Comicio anterior', '2026']
    assert lines[1]['color'] == '#1d4f86' and lines[1]['data'][0] == 8.9 and lines[1]['data'][1]['value'] == -0.9
    # la etiqueta sólo en el valor final
    assert all(s['data'][1]['label']['show'] is True and not isinstance(s['data'][0], dict) for s in lines)
    marked = [s for s in lines if 'markLine' in s]
    assert len(marked) == 1 and marked[0]['markLine']['data'] == [{'yAxis': 0}]
    low = option['yAxis']['min']
    assert low < -0.9 and marked[0]['markArea']['data'] == [[{'yAxis': low}, {'yAxis': 0}]]
    assert option['legend']['data'] == ['Aragón', 'Castilla y León', 'Andalucía']
    assert run_js('sl.slopeOption(rows)', rows=rows)['xAxis']['data'] == ['Comicio anterior', '2026']


def test_radar_option_two_series():
    indicators = [{'name': 'Participación CERA', 'max': 100}, {'name': 'Colocación dirigida', 'max': 100},
                  {'name': 'Voto a la izquierda', 'max': 100}]
    series = [{'name': 'Realidad hoy', 'value': [8.6, 0, 38.3], 'color': '#1d4f86', 'filled': True},
              {'name': 'Todo al máximo', 'value': [100, 100, 100], 'color': '#ce3b3b', 'filled': False}]
    option = run_js('ra.radarOption(%s, rows)' % json.dumps(indicators), rows=series)
    assert option['radar']['indicator'] == indicators
    [radar] = option['series']
    assert radar['type'] == 'radar' and len(radar['data']) == 2
    today, top = radar['data']
    assert today['name'] == 'Realidad hoy' and today['value'] == [8.6, 0, 38.3] and top['value'] == [100, 100, 100]
    assert today['lineStyle']['color'] == '#1d4f86' and today['label']['show'] is True and top['label']['show'] is False
    assert today['areaStyle']['opacity'] > top['areaStyle']['opacity']
    assert option['legend']['data'] == ['Realidad hoy', 'Todo al máximo'] and option['legend']['bottom'] == 0


def test_grouped_option_one_bar_series_per_entry_over_the_categories():
    series = [{'name': 'PSOE', 'data': [6.2, 7.1], 'color': '#ce3b3b'},
              {'name': 'Otras izquierdas', 'data': [-8.3, None], 'color': '#7b3fa0'}]
    option = run_js('g.groupedOption(["Castilla y León", "Andalucía"], rows)', rows=series)
    assert option['xAxis']['type'] == 'category' and option['xAxis']['data'] == ['Castilla y León', 'Andalucía']
    assert option['yAxis']['type'] == 'value' and option['legend']['data'] == ['PSOE', 'Otras izquierdas']
    psoe, other = option['series']
    assert psoe['type'] == other['type'] == 'bar' and psoe['color'] == '#ce3b3b' and 'stack' not in psoe
    assert [item['value'] for item in psoe['data']] == [6.2, 7.1] and other['data'][1]['value'] is None
    # la etiqueta del valor fuera de la barra: encima si es positivo, debajo si es negativo
    assert [item['label']['position'] for item in psoe['data'] + other['data'][:1]] == ['top', 'top', 'bottom']
    # en horizontal, las categorías en el eje Y de arriba abajo; los colores por barra y la leyenda abajo
    series = [{'name': 'Brecha', 'data': [284, 403], 'color': '#c7cdd8'},
              {'name': 'Neto', 'data': [{'value': -15, 'color': '#ce3b3b'}, {'value': 206, 'color': '#1d4f86'}]}]
    option = run_js('g.groupedOption(["Girona", "Cantabria"], rows, {horizontal: true, legend: "bottom"})', rows=series)
    assert option['yAxis']['data'] == ['Girona', 'Cantabria'] and option['yAxis']['inverse'] is True
    assert option['xAxis']['type'] == 'value' and option['legend']['bottom'] == 0
    net = option['series'][1]
    assert [item['itemStyle']['color'] for item in net['data']] == ['#ce3b3b', '#1d4f86']
    assert [item['label']['position'] for item in net['data']] == ['left', 'right']
    # apiladas: la misma pila, las etiquetas dentro (el texto propio de cada barra si lo hay) y los totales encima
    series = [{'name': 'Latinoamérica', 'data': [62000, 250000], 'labels': ['33 %', '59 %']},
              {'name': 'Resto', 'data': [127000, 172000]}]
    option = run_js('g.groupedOption(["Antes", "Después"], rows, {stacked: true, max: 100, '
                    'totals: ["+189.000", "+422.000 ×2,2"]})', rows=series)
    latam, rest, totals = option['series']
    assert latam['stack'] == rest['stack'] == totals['stack'] and option['yAxis']['max'] == 100
    assert [item['label']['formatter'] for item in latam['data']] == ['33 %', '59 %']
    assert latam['data'][0]['label']['position'] == 'inside' and rest['data'][0]['label']['position'] == 'inside'
    assert [item['value'] for item in totals['data']] == [0, 0] and totals['silent'] is True
    assert [item['label']['formatter'] for item in totals['data']] == ['+189.000', '+422.000 ×2,2']
    assert totals['data'][0]['label']['position'] == 'top' and option['legend']['data'] == ['Latinoamérica', 'Resto']


def test_bullseye_highlights_one_point_over_the_others():
    """`highlight`: un punto destacado (el modelo en el detalle de una elección), del tamaño de los demás y en su
    color, sin etiqueta, por encima de todos (también de los grupos destacados) y contado en la escala de los ejes;
    sin coordenadas, nada."""
    option = run_js('b.bullseyeOption(rows, {blocks: ["Derecha", "Izquierda"], color: "#1d4f86", '
                    'highlight: {row: {event: "Modelo (promedio)", Derecha: 14.0, Izquierda: -3.0}, color: "#c0672e"}})')
    [model] = [s for s in option['series'] if s.get('id') == 'highlight']
    assert model['type'] == 'scatter' and model['symbolSize'] == 8 and 'symbol' not in model
    assert model['z'] > 12 and model['color'] == '#c0672e'
    assert model['data'] == [{'value': [14.0, -3.0], 'row': {'event': 'Modelo (promedio)', 'Derecha': 14.0, 'Izquierda': -3.0}}]
    assert model['label'] == {'show': False}
    assert model['z'] > max(s.get('z', 0) for s in option['series'] if s.get('id') != 'highlight')
    assert option['xAxis']['max'] == 16  # 14 · 1,1 redondeado hacia arriba
    none = run_js('b.bullseyeOption(rows, {blocks: ["Derecha", "Izquierda"], highlight: {row: {Derecha: null}}})')
    assert not [s for s in none['series'] if s.get('id') == 'highlight']


def test_series_time_axis_names_ticks_by_what_they_are():
    """El eje temporal del promedio no rotula siempre día y mes: cada marca dice lo que es (año, mes o día,
    `format.fmtAxisDate`), con el año en negrita y sin solaparse."""
    label = run_js('s.seriesOption(%s).xAxis.axisLabel' % (SERIES_ARGS % "'hidden'"))
    assert label['color'] == '#5a5f72' and label['fontSize'] == 10.5 and label['hideOverlap'] is True
    assert label['rich']['year'] == {'color': '#1a1c22', 'fontSize': 10.5, 'fontWeight': 700}
    assert run_js('base.timeAxisLabel().formatter(new Date("2025-01-01T00:00:00").getTime(), 2)') == '{year|2025}'


IMAGE_ARGS = ("{series: {dates: ['2026-07-01', '2026-08-01', '2026-09-01', '2026-10-01'], parties: ['PP', 'VOX'], "
              "mean: {PP: [30, 31, 32, 33], VOX: [12, 13, 14, 15]}, lo: {PP: [29, 30, 31, 32], VOX: [11, 12, 13, 14]}, "
              "hi: {PP: [31, 32, 33, 34], VOX: [13, 14, 15, 16]}}, polls: {parties: ['PP', 'VOX'], polls: []}, "
              "projection: null, parties: ['PP', 'VOX'], selected: %s, colors: {PP: '#1d84ce', VOX: '#63be21'}, "
              "asOf: '2026-10-01', when: '2026-10-01', window: '3m', anchor: '2026-10-01', legend: 'hidden'}")


def test_series_image_draws_a_static_legend_of_the_names_shown_and_no_slider():
    """La imagen descargable del promedio (Luis, 2026-10-21): la leyenda se dibuja (no hay pastillas en el PNG),
    estática y sólo con lo que se ve; sin deslizador, con un zoom invisible en la ventana de la pantalla."""
    option = run_js('s.seriesImageOption(%s)' % (IMAGE_ARGS % "['VOX']"))
    assert option['legend']['show'] is True and option['legend']['selectedMode'] is False
    assert option['legend']['type'] == 'plain' and option['legend']['data'] == ['VOX']
    assert option['legend']['selected'] == {'PP': False, 'VOX': True}
    assert option['grid']['top'] == 40 and option['grid']['bottom'] == 8
    screen = run_js('s.seriesOption(%s).dataZoom[0]' % (IMAGE_ARGS % "['VOX']"))
    assert option['dataZoom'] == [{'type': 'inside', 'filterMode': 'filter', 'disabled': True,
                                   'startValue': screen['startValue'], 'endValue': screen['endValue']}]
    assert run_js('s.seriesImageOption(%s).legend.data' % (IMAGE_ARGS % 'null')) == ['PP', 'VOX']
    # the zoom of the screen, when given, replaces the window
    zoomed = run_js('s.seriesImageOption(%s, {startValue: 5, endValue: 9}).dataZoom[0]' % (IMAGE_ARGS % 'null'))
    assert zoomed == {'type': 'inside', 'filterMode': 'filter', 'disabled': True, 'startValue': 5, 'endValue': 9}


def test_chart_zoom_reads_the_window_given_and_the_drag():
    """`chartZoom` lee los extremos de la ventana del gráfico en pantalla, también tras arrastrar el deslizador; la
    imagen hecha con ese zoom enseña el mismo tramo del eje."""
    script = (
        "(() => {{ const make = (option) => {{ const chart = window.echarts.init(null, null, {{renderer: 'svg', ssr: true, width: 600, height: 400}}); "
        "chart.setOption(option); return chart; }}; "
        "const live = make(s.seriesOption({args})); const first = s.chartZoom(live); "
        "live.dispatchAction({{type: 'dataZoom', dataZoomIndex: 0, start: 40, end: 90}}); const dragged = s.chartZoom(live); "
        "const extent = (chart) => chart.getModel().getComponent('xAxis', 0).axis.scale.getExtent(); "
        "const image = make(s.seriesImageOption({args}, dragged)); "
        "return {{first, dragged, live: extent(live), same: JSON.stringify(extent(live)) === JSON.stringify(extent(image))}}; }})()"
    ).format(args=IMAGE_ARGS % 'null')
    out = run_js(script)
    assert out['first']['endValue'] == run_js('s.dayTime("2026-10-01")')
    assert out['dragged'] != out['first'] and [out['dragged']['startValue'], out['dragged']['endValue']] == out['live']
    assert out['same'] is True


def histogram_option(args):
    """Opción de `renderHistogram` con `args`, leída de un contenedor falso: `mountChart` reutiliza su `__chart`."""
    uri = pathlib.Path(DIST, 'js', 'charts', 'histogram.js').as_uri()
    return run_js('await (async () => { const hm = await import("%s"); let got = null; '
                  'hm.renderHistogram({__chart: {setOption: (o) => { got = o; }}}, [200, 201, 201, 202, 205], %s); '
                  'return got; })()' % (uri, args))


def test_histogram_caption_writes_its_lines_above_the_bars():
    """Las líneas junto al histograma de la calculadora de mayorías, como en los KDE: arriba a la izquierda, la cifra
    en negrita, en una franja propia sobre las barras (el margen superior crece 16 px por línea)."""
    option = histogram_option("{caption: [{label: 'Simulaciones', value: '1.000'}, {label: 'Mediana', value: '201'}]}")
    [text] = option['graphic']
    assert text['type'] == 'text' and (text['left'], text['top']) == (8, 4)
    assert text['style']['text'] == 'Simulaciones: {b|1.000}\nMediana: {b|201}' and text['style']['lineHeight'] == 16
    assert text['style']['rich']['b']['fontWeight'] == 600 and option['grid']['top'] == 24 + 2 * 16
    plain = histogram_option('{}')
    assert plain['graphic'] == [] and plain['grid']['top'] == 24


ROWS_JS = ('[{name: "PP", x0: 130, x1: 150, grid: [130, 140, 150], y: [0, 1, 0], median: 139.5, lo: 133, hi: 150, '
           'central: 141}, {name: "PSOE", x0: 95, x1: 115, grid: [95, 105, 115], y: [0, 1, 0], median: 105, lo: 100, '
           'hi: 112, central: null}, {name: "VOX", x0: 0, x1: 6, grid: [0, 3, 6], y: [0, 1, 0], median: 1, lo: 0, hi: 3, '
           'central: 0}]')


def ridge_option(rows_js, args):
    """Opción de `renderRidge` con `rows_js` y `args`, leída de un contenedor falso: `mountChart` reutiliza su
    `__chart`, y `renderRidge` fija `style.height` y lee `clientWidth`."""
    return run_js('(() => { let got = null; '
                  'r.renderRidge({__chart: {setOption: (o) => { got = o; }}, style: {}, clientWidth: 0}, %s, %s); '
                  'return got; })()' % (rows_js, args))


def test_ridge_marks_and_labels_the_central_seats_and_falls_back_to_the_median():
    """La marca gruesa y el rótulo de cada cresta dicen el reparto central; sin él, la mediana."""
    option = ridge_option(ROWS_JS, '{colors: {PP: "#1d4f86"}, majority: 176}')
    thick = [[m['xAxis'] for m in s['markLine']['data'] if m['lineStyle'].get('width') == 2] for s in option['series']]
    assert thick[2] == [0] and thick[:2] == [[141], [105]]
    texts = [g['style']['text'] for g in option['graphic'] if g['type'] == 'text']
    assert texts[0].startswith('{b|PP}\nReparto central: {b|141}') and texts[1].startswith('{b|PSOE}\nMediana: {b|105}')
    assert texts[2].startswith('{b|VOX}\nReparto central: {b|0}')


def test_matrix_option_colours_by_sign_and_bolds_the_general_elections():
    option = run_js('mx.matrixOption({events: [{short: "AN 18", label: "Andalucía 2018"}, {short: "GEN 23", label: "Generales 2023"}], houses: ["CIS", "GAD3"], cells: [[0, 0, 12.6], [1, 1, -4.0]]})')
    assert option['xAxis']['position'] == 'top' and option['yAxis']['inverse'] is True and option['series'][0]['type'] == 'heatmap'
    colours = [cell['itemStyle']['color'] for cell in option['series'][0]['data']]
    assert colours[0].startswith('rgba(214,39,40') and colours[1].startswith('rgba(29,79,134')


def test_ranking_option_marks_the_centre_the_pack_and_the_pollster():
    rows = [{'x': 93, 'y': 8.0, 'name': 'CIS', 'k': 38, 'n': 41, 'cis': True}, {'x': 77, 'y': 1.6, 'name': 'Sigma Dos', 'k': 23, 'n': 30, 'cis': False}]
    option = run_js('rk.rankingOption(%s)' % json.dumps(rows))
    kinds = [s['type'] for s in option['series']]
    assert kinds == ['line', 'line', 'scatter', 'scatter', 'scatter'] and option['yAxis']['interval'] == 2 and option['xAxis']['max'] == 100
    assert option['series'][3]['markArea']['label']['show'] is False and option['series'][4]['data'][0]['value'][:2] == [93, 8.0]


def test_distance_option_colours_the_pollster_and_the_consensus():
    option = run_js('ds.distanceOption([{name: "Opinometre", dist: 0.5}, {name: "Consenso (20 casas)", dist: 3.3, consensus: true}, {name: "CIS", dist: 4.2, cis: true}])')
    colours = [d['itemStyle']['color'] for d in option['series'][0]['data']]
    assert colours[2] == '#ce3b3b' and colours[1] == '#1a1c22' and option['xAxis']['axisLabel']['show'] is False


def test_bullseye_interval_and_highlights():
    option = run_js('b.bullseyeOption(rows, {blocks: ["Derecha", "Izquierda"], interval: 5, highlights: [{row: rows[0], color: "#ce3b3b", label: "CIS"}, {row: rows[1], color: "#1a1c22", symbol: "diamond", label: "Consenso"}]})')
    assert option['xAxis']['interval'] == 5 and option['xAxis']['max'] % 5 == 0 and option['xAxis']['max'] > 10
    marked = [s for s in option['series'] if s.get('id', '').startswith('highlight')]
    assert len(marked) == 2 and marked[1]['symbol'] == 'diamond' and marked[0]['label']['show'] is True


def test_scatter_diagonal_and_fixed_axes():
    option = run_js('sc.scatterOption([{x: 1, y: 2, name: "a"}], {x: "x", y: "y", name: "name", diagonal: true, axes: {xMin: -10, xMax: 15, yMin: -5, yMax: 15, interval: 5}})')
    assert option['xAxis']['min'] == -10 and option['yAxis']['interval'] == 5 and option['series'][-1]['data'] == [[-10, -10], [15, 15]]
