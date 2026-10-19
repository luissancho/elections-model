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
SERIES_WIRE_JS = pathlib.Path(DIST, 'js', 'pages', 'series-wire.js').as_uri()
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
    cableado compartido del promedio), `c` (el catálogo) y `rows` definidos y devuelve su JSON (las funciones de la
    opción se pierden en la serialización, como en `JSON.stringify`)."""
    script = (
        'import {{createRequire}} from "node:module"; '
        'const echarts = createRequire(import.meta.url)({}); globalThis.window = {{echarts}}; '
        'const b = await import("{}"); const d = await import("{}"); const h = await import("{}"); '
        'const s = await import("{}"); const base = await import("{}"); const r = await import("{}"); '
        'const p = await import("{}"); const w = await import("{}"); const st = await import("{}"); '
        'const sw = await import("{}"); const c = await import("{}"); const rows = {}; '
        'console.log(JSON.stringify({})); process.exit(0);'
    ).format(json.dumps(ECHARTS), BULLSEYE_JS, DIVERGING_JS, HEMICYCLE_JS, SERIES_JS, BASE_JS, RIDGE_JS, PATH_JS,
             WEIGHTS_JS, STACKED_JS, SERIES_WIRE_JS, CATALOG_JS, json.dumps(rows), expr)
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
    assert [g['style']['text'] for g in texts] == ['176', '350', 'ESCAÑOS · MAYORÍA ABSOLUTA 176']
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
    # el nombre y los escaños en negrita y la flecha en peso normal (Luis, 2026-10-17); sin delta, solo los escaños
    assert [s['label']['formatter'] for s in bars] == ['{seats|PP 140} {delta|▼ 25}', '{seats|PSOE 107} {delta|–}', '{seats|BNG 2}']
    assert all(s['label']['rich'] == {'seats': {'fontWeight': 700, 'color': '#fff'}, 'delta': {'fontWeight': 400, 'color': '#fff'}} for s in bars)
    assert [s['label']['show'] for s in bars] == [True, True, False] and option['xAxis']['max'] == 350
    assert [s['type'] for s in option['series']] == ['bar', 'bar', 'bar', 'line']
    plain = run_js('st.stackedOption([{name: "PP", seats: 140}], {total: 350})')
    assert plain['series'][0]['label']['formatter'] == '{seats|PP 140}' and len(plain['series']) == 1


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
    # el tramo final: discontinuo desde la última elección hasta hoy, con un punto hueco solo en hoy
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
    assert lines[2:] == ['Publicado 32,4% · corregido 32,9%', 'Peso 0,50', 'F(kernel) 0,64 · F(solape) 1,00 · F(muestra) 0,80 · F(rating) 1,00']
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
