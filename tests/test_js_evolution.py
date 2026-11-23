"""Evolución de las actualizaciones (`web/dist/js/charts/evolution.js` y `pages/evolution-wire.js`): un run por día, los
puntos de cada partido, el eje redondeado a unidades y la opción ECharts, comprobados con `node` (se salta sin él).
Los módulos de gráficos leen `window.echarts` al cargarse, así que el arnés carga el ECharts empaquetado antes."""
import json
import os
import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIST = os.path.join(ROOT, 'web', 'dist')
ECHARTS = os.path.join(DIST, 'vendor', 'echarts-5.6.0.min.js')
EVOLUTION_JS = pathlib.Path(DIST, 'js', 'charts', 'evolution.js').as_uri()
BLOCKS_JS = pathlib.Path(DIST, 'js', 'blocks.js').as_uri()
WIRE_JS = pathlib.Path(DIST, 'js', 'pages', 'evolution-wire.js').as_uri()
CATALOG_JS = pathlib.Path(DIST, 'js', 'catalog.js').as_uri()
CHANGES_JS = pathlib.Path(DIST, 'js', 'evolution-changes.js').as_uri()
FORMAT_JS = pathlib.Path(DIST, 'js', 'format.js').as_uri()

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
META = {'parties': [{'name': 'PP', 'color': '#1e5aa8', 'fullname': 'Partido Popular', 'block': 'Derecha'},
                    {'name': 'VOX', 'color': '#63be21', 'block': 'Derecha'}, {'name': 'PSOE', 'color': '#e3001b', 'block': 'Izquierda'}],
        'bmaps': {'main': ['PP', 'PSOE'], 'blocks': {'Derecha': ['PP', 'VOX'], 'Izquierda': ['PSOE'], 'Otros': []}}}
DATA = {'runs': None, 'parties': PARTIES, 'selected': ['PP', 'PSOE'], 'colors': {'PP': '#1e5aa8', 'PSOE': '#e3001b', 'VOX': '#63be21'}}


def run_js(expr):
    """Evalúa `expr` con `e` (la evolución), `b` (los bloques), `ew` (el cableado), `c` (el catálogo), `ch` (la tabla
    de cambios) y `runs` (el fijo) definidos y devuelve su JSON (las funciones de la opción se pierden en la serialización, como en
    `JSON.stringify`)."""
    script = (
        'import {{createRequire}} from "node:module"; '
        'const echarts = createRequire(import.meta.url)({}); globalThis.window = {{echarts}}; '
        'const e = await import("{}"); const b = await import("{}"); const ew = await import("{}"); '
        'const c = await import("{}"); const ch = await import("{}"); const runs = {}; '
        'console.log(JSON.stringify({})); process.exit(0);'
    ).format(json.dumps(ECHARTS), EVOLUTION_JS, BLOCKS_JS, WIRE_JS, CATALOG_JS, CHANGES_JS, json.dumps(RUNS), expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout)


def option(**over):
    """`evolutionOption` sobre los runs diarios del fijo, con `over` encima de `DATA`."""
    return run_js('e.evolutionOption({...%s, runs: e.dailyRuns(runs), ...%s})' % (json.dumps(DATA), json.dumps(over)))


def test_daily_runs_keep_the_last_run_of_each_day():
    # Luis, 2026-11-04: live and retrospective runs alike, the last `run_at` of the day (b06 at 12:00 over l06 at 09:30)
    assert run_js('e.dailyRuns(runs).map((run) => run.run_id)') == ['b05', 'b06', 'l07b', 'x08']
    assert run_js('e.runDay(runs[5])') == '2026-10-08'  # without `as_of`: the day of `run_at`
    assert run_js('e.dailyRuns([])') == []


def test_evolution_points_skip_the_runs_without_the_party_and_carry_the_run_index():
    points = run_js('e.evolutionPoints(e.dailyRuns(runs), %s, "pct")' % json.dumps(PARTIES))
    assert [p['name'] for p in points] == PARTIES
    assert [p[1] for p in points[0]['points']] == [32.71, 32.64, 32.4, 32.3] and [p[2] for p in points[0]['points']] == [0, 1, 2, 3]
    assert [p[1] for p in points[2]['points']] == [17.93, 18.1]  # VOX: missing in b06, null in the last run
    assert run_js('e.evolutionPoints(e.dailyRuns(runs), ["PP"], "seats")[0].points.map((p) => p[1])') == [140, 139, 138, 137]


def test_evolution_axis_follows_the_rule_of_the_average_multiples_of_five():
    # Luis, 2026-10-30: ticks every 5 on multiples of 5, as the average (2 → 0, 33 → 35); every 20 in seats, so the
    # whole chamber (0 to 145) keeps about as many lines as the vote axis
    bounds = lambda extent, field='pct': run_js('[e.evolutionOption({...%s, runs: e.dailyRuns(runs), field: "%s"}).yAxis.min(%s), '
                                                'e.evolutionOption({...%s, runs: e.dailyRuns(runs), field: "%s"}).yAxis.max(%s)]'
                                                % (json.dumps(DATA), field, extent, json.dumps(DATA), field, extent))
    assert bounds('{min: 1.9, max: 33.2}') == [0, 35]
    assert bounds('{min: 26.37, max: 26.77}') == [25, 30]
    assert bounds('{min: 107, max: 110}', 'seats') == [100, 120]
    assert bounds('{min: 8, max: 8}', 'seats') == [0, 20]
    assert bounds('{min: Infinity, max: -Infinity}') == [None, None]
    assert option()['yAxis']['interval'] == 5 and option(field='seats')['yAxis']['interval'] == 20


def test_evolution_option_draws_a_line_per_party_with_solid_points_and_donuts_on_change():
    opt = option()
    assert [s['name'] for s in opt['series']] == PARTIES and all(s['type'] == 'line' for s in opt['series'])
    # a filled circle: ECharts' default `emptyCircle` strokes with `color`, so a white hollow point would vanish
    assert all(s['symbol'] == 'circle' for s in opt['series'])
    pp = opt['series'][0]['data']
    assert pp[0][1:] == [32.71, 0]  # the first day: a plain `[time, value, index]`, retrospective or not
    # 32,7 → 32,6 → 32,4 → 32,3: every later day changes the figure shown, so every later point is a donut
    for index, value in ((1, 32.64), (2, 32.4), (3, 32.3)):
        assert pp[index]['value'][1:] == [value, index] and pp[index]['symbolSize'] == 8
        assert pp[index]['itemStyle'] == {'color': '#fff', 'borderColor': '#1e5aa8', 'borderWidth': 2}
    assert opt['legend'] == {'show': False, 'selectedMode': True, 'type': 'scroll', 'top': 0, 'data': PARTIES,
                             'selected': {'PP': True, 'PSOE': True, 'VOX': False}}
    assert opt['dataZoom'][0]['type'] == 'slider' and opt['xAxis']['type'] == 'time'
    assert run_js('typeof e.evolutionOption({...%s, runs: e.dailyRuns(runs)}).yAxis.min' % json.dumps(DATA)) == 'function'
    assert run_js('e.evolutionOption({runs: [], parties: []}).series') == []


def test_evolution_image_option_shows_a_plain_legend_of_the_names_shown_and_freezes_the_zoom():
    img = run_js('e.evolutionImageOption({...%s, runs: e.dailyRuns(runs)}, {startValue: 1, endValue: 2})' % json.dumps(DATA))
    assert img['legend'] == {'show': True, 'type': 'plain', 'selectedMode': False, 'top': 0, 'data': ['PP', 'PSOE']}
    assert img['dataZoom'] == [{'type': 'inside', 'filterMode': 'filter', 'disabled': True, 'startValue': 1, 'endValue': 2}]
    assert img['grid']['bottom'] == 8 and img['grid']['top'] == 40


MARKS = [{'date': '2026-10-05', 'label': 'Convocatoria', 'kind': 'call'},
         {'date': '2026-10-07', 'label': 'Campaña', 'kind': 'campaign'}]


def test_evolution_option_draws_the_key_dates_as_mark_lines():
    """Las fechas clave son líneas discontinuas de una serie aparte, que la leyenda no toca; sin ellas, ninguna."""
    drawn = option(marks=MARKS)
    marks = drawn['series'][-1]
    assert marks['name'] == 'marks' and 'marks' not in drawn['legend']['data'] and marks['data'] == []
    assert [m['label']['formatter'] for m in marks['markLine']['data']] == ['Convocatoria', 'Campaña']
    # the labels go above the plot, where no line of a party passes, each one towards the inside: in the first half
    # of the days it starts at its line (the call on the first day stays off the axis), in the second it ends there
    assert [m['label']['position'] for m in marks['markLine']['data']] == ['end', 'end']
    assert [m['label']['align'] for m in marks['markLine']['data']] == ['left', 'right']
    days = [m['xAxis'] for m in marks['markLine']['data']]
    assert days[1] - days[0] == 2 * 24 * 3600 * 1000  # the 5th and the 7th, as timestamps
    # a single day of runs, or none: the mark of that day starts at its line; nothing breaks without runs
    one = run_js('e.evolutionOption({...%s, runs: e.dailyRuns(runs).slice(0, 1), marks: %s})'
                 % (json.dumps(DATA), json.dumps(MARKS)))
    assert [m['label']['align'] for m in one['series'][-1]['markLine']['data']] == ['left', 'right']
    none = run_js('e.evolutionOption({...%s, runs: [], marks: %s})' % (json.dumps(DATA), json.dumps(MARKS)))
    assert len(none['series'][-1]['markLine']['data']) == 2
    assert all(entry['name'] != 'marks' for entry in option()['series'])
    assert [entry['name'] for entry in drawn['series'][:-1]] == [entry['name'] for entry in option()['series']]


def test_evolution_image_keeps_the_marks():
    """La imagen deja fuera los partidos apagados, pero no las marcas."""
    data = dict(DATA, marks=MARKS)
    image = run_js('e.evolutionImageOption({...%s, runs: e.dailyRuns(runs)})' % json.dumps(data))
    assert [entry['name'] for entry in image['series']] == ['PP', 'PSOE', 'marks']


def test_evolution_views_take_the_parties_given_in_their_order_with_the_main_ones_by_default():
    # the same names and order as the legend of the average (Luis, 2026-10-30): the list of the page, as given
    views = run_js('ew.evolutionViews(runs, new c.Catalog(%s), {parties: ["PP", "VOX", "PSOE"]})' % (json.dumps(META)))
    assert views['partidos']['parties'] == ['PP', 'VOX', 'PSOE'] and views['partidos']['defaults'] == ['PP', 'PSOE']
    assert views['partidos']['fullnames'] == {'PP': 'Partido Popular', 'VOX': 'VOX', 'PSOE': 'PSOE'} and views['partidos']['colors']['PP'] == '#1e5aa8'
    # without a list: the parties of the last run with a value, in the order of the run (VOX is null there)
    views = run_js('ew.evolutionViews(runs, new c.Catalog(%s))' % json.dumps(META))
    assert views['partidos']['parties'] == ['PP', 'PSOE'] and views['partidos']['defaults'] == ['PP', 'PSOE']
    assert [r['run_id'] for r in views['partidos']['runs']] == ['b05', 'b06', 'l07b', 'x08']
    assert views['bloques']['parties'] == ['Izquierda', 'Otros', 'Derecha'] and views['bloques']['defaults'] is None
    assert views['bloques']['colors'] == {'Izquierda': '#e3001b', 'Otros': '#9e9e9e', 'Derecha': '#1e5aa8'}
    assert views['bloques']['runs'][0]['nowcast']['parties'][0] == {'name': 'Derecha', 'pct': pytest.approx(50.64)}
    # the block view carries the map of the blocks, for the line naming the parties of each block
    assert views['bloques']['members'] == META['bmaps']['blocks'] and views['partidos'].get('members') is None


def test_evolution_views_without_blocks_or_main_parties():
    meta = {**META, 'bmaps': {'blocks': []}}
    views = run_js('ew.evolutionViews(runs, new c.Catalog(%s))' % json.dumps(meta))
    assert views['bloques'] is None and views['partidos']['defaults'] is None
    seats = run_js('ew.evolutionViews(runs, new c.Catalog(%s), {field: "seats", total: 350}).bloques.runs[0].nowcast.parties' % json.dumps(META))
    assert seats == [{'name': 'Derecha', 'seats': 203}, {'name': 'Izquierda', 'seats': 107}, {'name': 'Otros', 'seats': 40}]


def test_wire_evolution_switches_metric():
    """The Voto / Escaños switch picks `{field, formatter, total}` by metric, and the views are built with it."""
    metrics = '{pct: {field: "pct", formatter: f.fmtPct, total: 100}, seats: {field: "seats", formatter: f.fmtInt, total: 350}}'
    head = 'const f = await import("%s"); const m = %s; ' % (FORMAT_JS, metrics)
    out = run_js('await (async () => { %s const s = ew.evolutionMetric(m, "seats"), p = ew.evolutionMetric(m, "pct"), '
                 'x = ew.evolutionMetric(m, "nope"); return [s.field, s.total, s.formatter === f.fmtInt, p.field, p.total, '
                 'p.formatter === f.fmtPct, x.field]; })()' % head)
    assert out == ['seats', 350, True, 'pct', 100, True, 'pct']  # an unknown metric falls back to the first one
    views = run_js('await (async () => { %s return ew.evolutionViews(runs, new c.Catalog(%s), {parties: ["PP", "PSOE", "VOX"], '
                   '...ew.evolutionMetric(m, "seats")}); })()' % (head, json.dumps(META)))
    assert views['bloques']['runs'][0]['nowcast']['parties'] == [
        {'name': 'Derecha', 'seats': 203}, {'name': 'Izquierda', 'seats': 107}, {'name': 'Otros', 'seats': 40}]
    # the file of the image follows the metric: the voto name as the template writes it, «-evolucion-escanos-» for seats
    file = 'demoscopio-evolucion-es-2026-11-04.png'
    assert run_js('[ew.evolutionFile(%s, "pct"), ew.evolutionFile(%s, "seats"), ew.evolutionFile(ew.evolutionFile(%s, "seats"), "seats")]'
                  % ((json.dumps(file),) * 3)) == [file, 'demoscopio-evolucion-escanos-es-2026-11-04.png',
                                                   'demoscopio-evolucion-escanos-es-2026-11-04.png']
    labels = run_js('[ew.evolutionLabel("pct"), ew.evolutionLabel("seats")]')
    assert labels == ['Evolución de las actualizaciones: el voto de cada partido en cada actualización',
                      'Evolución de las actualizaciones: los escaños de cada partido en cada actualización']


def test_change_dates_pick_the_campaign_and_the_call_of_the_key_dates():
    """La tabla compara con el inicio de la campaña y la convocatoria: las fechas de esos tipos entre las fechas clave."""
    key_dates = [{'date': '2026-10-05', 'label': 'Convocatoria', 'kind': 'call'},
                 {'date': '2026-11-01', 'label': 'Debate', 'kind': 'key'},
                 {'date': '2026-11-13', 'label': 'Campaña', 'kind': 'campaign'}]
    assert run_js('ew.changeDates(%s)' % json.dumps(key_dates)) == {'campaign': '2026-11-13', 'call': '2026-10-05'}
    # before the call nothing is compared; other key dates are never a column
    assert run_js('[ew.changeDates([]), ew.changeDates(undefined), ew.changeDates(%s)]' % json.dumps(key_dates[1:2])) == [
        {'campaign': None, 'call': None}] * 3


def day_runs(field, values, name='VOX'):
    """Runs of consecutive days from 2026-10-05 with `values` of `name` in `field` (None: the party missing)."""
    return [{'run_id': 'r%d' % i, 'run_at': '2026-10-%02dT12:00:00Z' % (5 + i), 'as_of': '2026-10-%02d' % (5 + i),
             'nowcast': {'parties': [] if value is None else [{'name': name, field: value}]}} for i, value in enumerate(values)]


def donuts(field, values, name='VOX'):
    """Which points of `name` the option draws as donuts (objects) for the daily `values`."""
    data = run_js('e.evolutionOption({runs: %s, parties: ["%s"], field: "%s"}).series[0].data'
                  % (json.dumps(day_runs(field, values, name)), name, field))
    return [isinstance(point, dict) for point in data]


def test_donut_only_when_the_shown_figure_changes():
    # the daily drift of 0,01 without new polls keeps the figure at 18,9 until it reaches 19,0
    assert donuts('pct', [18.86, 18.87, 18.88, 18.96]) == [False, False, False, True]
    assert donuts('seats', [63, 63, 64]) == [False, False, True]


def test_donut_compares_with_the_last_day_of_the_same_name():
    # the day without VOX gives no point, so 18,1 compares with 18,0 (and 18,0 with nothing: the first point)
    assert donuts('pct', [17.95, None, 18.1]) == [False, True]
    assert donuts('pct', [17.95, None, 17.96]) == [False, False]


def test_shown_value_rounds_as_the_figure():
    assert run_js('[e.shownValue(18.25, "pct"), e.shownValue(63.4, "seats"), e.shownValue(null, "pct"), e.FIELD_DIGITS]') == \
        [18.3, 63, None, {'pct': 1, 'seats': 0}]


def test_tooltip_heads_with_the_day_and_marks_the_changes():
    script = ('const o = e.evolutionOption({runs: %s, parties: ["VOX"], field: "pct"}); '
              'o.series[0].data.map((point) => o.tooltip.formatter([{value: Array.isArray(point) ? point : point.value, '
              'seriesName: "VOX", marker: ""}]))')
    boxes = run_js('(() => { %s })()' % (script.replace('o.series[0].data.map', 'return o.series[0].data.map')
                                         % json.dumps(day_runs('pct', [18.86, 18.87, 18.96]))))
    assert [box.split('<br>')[0] for box in boxes] == ['<strong>5 oct 2026</strong>', '<strong>6 oct 2026</strong>',
                                                       '<strong>7 oct 2026</strong>']
    assert 'dlt' not in boxes[0] and 'dlt' not in boxes[1]
    assert boxes[2].endswith('VOX: 19,0% <span class="dlt up">▲ 0,1</span>')


# 05-10 to 15-10 without the 08: PP falls, PSOE drifts below the rounding today, SUMAR starts on the 06
CHANGE_DAYS = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
CHANGE_VALUES = {'PP': {5: 32.71, 7: 32.44, 14: 32.04, 15: 31.93}, 'PSOE': {5: 26.37, 7: 26.5, 14: 26.55, 15: 26.58},
                 'SUMAR': {7: 7.0, 14: 7.2, 15: 7.24}}


def change_runs():
    """Daily runs of `CHANGE_DAYS`: the values of `CHANGE_VALUES`, else PP 32.3, PSOE 26.5 and SUMAR 7.1 (none on the 05)."""
    default = {'PP': 32.3, 'PSOE': 26.5, 'SUMAR': 7.1}
    runs = []
    for day in CHANGE_DAYS:
        parties = [{'name': name, 'pct': CHANGE_VALUES[name].get(day, default[name]), 'seats': 100}
                   for name in ('PP', 'PSOE', 'SUMAR') if name != 'SUMAR' or day > 5]
        runs.append({'run_id': 'd%02d' % day, 'run_at': '2026-10-%02dT12:00:00Z' % day, 'as_of': '2026-10-%02d' % day,
                     'nowcast': {'parties': parties}})
    return runs


def changes(runs=None, names=('PP', 'PSOE', 'SUMAR'), **dates):
    """`evolutionChanges` del voto sobre `runs` (los de `change_runs` si no se dan), con las fechas clave `dates`
    (`campaign`, `call`)."""
    return run_js('ch.evolutionChanges(%s, %s, "pct", %s)' % (json.dumps(change_runs() if runs is None else runs),
                                                             json.dumps(list(names)), json.dumps(dates)))


def test_changes_compare_yesterday_the_campaign_and_the_call():
    """Tres columnas: el día anterior, el inicio de la campaña y la convocatoria."""
    out = changes(campaign='2026-10-12', call='2026-10-05')
    assert out['days'] == {'last': '2026-10-15', 'previous': '2026-10-14', 'campaign': '2026-10-12', 'call': '2026-10-05'}
    assert set(out) == {'days', 'rows'}  # fixed headers (Luis, 2026-11-04): nothing depends on today
    rows = {row['name']: row for row in out['rows']}
    assert [row['name'] for row in out['rows']] == ['PP', 'PSOE', 'SUMAR']
    # PP 31,9 against 32,0 / 32,3 / 32,7
    assert rows['PP'] == {'name': 'PP', 'value': 31.9, 'day': -0.1, 'campaign': -0.4, 'call': -0.8}
    # PSOE 26,6 against 26,6 (26.55 → 26.58: a drift below the rounding) / 26,5 / 26,4
    assert rows['PSOE'] == {'name': 'PSOE', 'value': 26.6, 'day': 0, 'campaign': 0.1, 'call': 0.2}


def test_changes_missing_name_gives_a_dash():
    """Un partido sin cifra el día de la convocatoria: nulo en los datos y una raya en su celda."""
    rows = {row['name']: row for row in changes(campaign='2026-10-12', call='2026-10-05')['rows']}
    # no SUMAR on the 05
    assert rows['SUMAR'] == {'name': 'SUMAR', 'value': 7.2, 'day': 0, 'campaign': 0.1, 'call': None}
    assert run_js('ch.evolutionChanges([], ["PP"], "pct")') is None
    html = table_html(campaign='2026-10-12', call='2026-10-05')
    sumar = html[html.index('data-name="SUMAR"'):]
    assert sumar[:sumar.index('</tr>')].endswith('<td><span class="dlt flat">–</span></td>')


def test_changes_with_a_single_day():
    assert changes(runs=change_runs()[:1]) is None
    assert changes(runs=[]) is None
    assert changes(runs=change_runs()[:1], campaign='2026-10-12', call='2026-10-05') is None


def test_changes_leave_out_a_date_the_runs_do_not_reach():
    """Sin fecha, antes de que empiece, con la serie empezada después o repetida en otra columna: sin columna."""
    bare = changes()
    assert bare['days'] == {'last': '2026-10-15', 'previous': '2026-10-14', 'campaign': None, 'call': None}
    assert all(row['campaign'] is None and row['call'] is None for row in bare['rows'])
    # the campaign starts on the last day, or later: two columns
    assert changes(campaign='2026-10-15', call='2026-10-05')['days']['campaign'] is None
    late = changes(campaign='2026-10-20', call='2026-10-05')
    assert late['days']['campaign'] is None and late['days']['call'] == '2026-10-05'
    assert all(row['campaign'] is None for row in late['rows']) and late['rows'][0]['call'] == -0.8
    # the series starts after the call
    assert changes(call='2026-10-01')['days']['call'] is None
    # the 08 has no run: the reference of a campaign that starts that day is the 07
    assert changes(campaign='2026-10-08', call='2026-10-05')['days']['campaign'] == '2026-10-07'
    # the day before is the day of the campaign: one column for both
    out = changes(campaign='2026-10-14', call='2026-10-05')
    assert out['days'] == {'last': '2026-10-15', 'previous': '2026-10-14', 'campaign': None, 'call': '2026-10-05'}
    # the day after the call: the day before is the call
    first = changes(runs=change_runs()[:2], campaign='2026-10-12', call='2026-10-05')
    assert first['days'] == {'last': '2026-10-06', 'previous': '2026-10-05', 'campaign': None, 'call': None}
    heads = re.findall(r'<th[^>]*>([^<]*)</th>', table_html(runs=change_runs()[:2], campaign='2026-10-12', call='2026-10-05'))
    assert heads == ['Partido', '6-oct', 'vs 5-oct']


def table_html(runs=None, field='pct', formatter='fmtPct', **dates):
    """`changesTableHtml` de los cambios de `runs` (los de `change_runs` si no se dan) con las fechas clave `dates`."""
    view = {'colors': {'PP': '#1e5aa8', 'PSOE': '#e3001b', 'SUMAR': '#e5007d'}, 'fullnames': {'PP': 'Partido Popular', 'PSOE': 'PSOE', 'SUMAR': 'Sumar'}}
    return run_js('await (async () => { const f = await import("%s"); return ch.changesTableHtml(ch.evolutionChanges(%s, ["PP", "PSOE", "SUMAR"], "%s", %s), %s, {field: "%s", formatter: f.%s}); })()'
                  % (pathlib.Path(DIST, 'js', 'format.js').as_uri(), json.dumps(change_runs() if runs is None else runs), field,
                     json.dumps(dates), json.dumps(view), field, formatter))


def test_changes_table_html():
    """Cabeceras con fecha («vs 14-oct»), con lo que es cada una en el título; la columna sin día no sale."""
    html = table_html(campaign='2026-10-12', call='2026-10-05')
    # Luis, 2026-11-04: the dates as headers, «d-mes», «vs» before those of the changes: the last run, the day
    # before and, since 2026-11-23, the start of the campaign and the call
    assert re.findall(r'<th[^>]*>([^<]*)</th>', html) == ['Partido', '15-oct', 'vs 14-oct', 'vs 12-oct', 'vs 5-oct']
    assert 'title="Frente al 14 oct 2026"' in html
    assert 'title="Frente al 12 oct 2026, inicio de la campaña"' in html
    assert 'title="Frente al 5 oct 2026, convocatoria"' in html
    # every name of the view, whatever the legend shows (Luis, 2026-11-04)
    assert re.findall(r'data-name="([^"]+)"', html) == ['PP', 'PSOE', 'SUMAR'] and 'hidden' not in html
    pp = html[html.index('data-name="PP"'):]
    pp = pp[:pp.index('</tr>')]
    assert 'title="Partido Popular"' in pp and '<span class="dot" style="background-color: #1e5aa8"></span>PP</span>' in pp
    assert '<strong>31,9%</strong>' in pp and '<span class="dlt down">▼ 0,4</span>' in pp and '<span class="dlt down">▼ 0,1</span>' in pp
    assert '<span class="dlt down">▼ 0,8</span>' in pp
    # before the campaign starts: the day before and the call
    two = table_html(call='2026-10-05')
    assert re.findall(r'<th[^>]*>([^<]*)</th>', two) == ['Partido', '15-oct', 'vs 14-oct', 'vs 5-oct']
    seats = table_html(field='seats', formatter='fmtInt')  # whole seats
    assert '<strong>100</strong>' in seats


def test_shown_value_rounds_like_the_formatter_on_block_sums():
    # a block sum carries float noise: 7.1499999999999995 is written 7,1 by `fmtPct`, so it is 7,1 and no change
    assert run_js('e.shownValue(7.1499999999999995, "pct")') == 7.1
    assert donuts('pct', [7.1, 7.1499999999999995], name='Izquierda') == [False, False]
    rows = run_js('ch.evolutionChanges(%s, ["Izquierda"], "pct").rows' % json.dumps(day_runs('pct', [7.1, 7.1499999999999995], 'Izquierda')))
    assert rows == [{'name': 'Izquierda', 'value': 7.1, 'day': 0, 'campaign': None, 'call': None}]


def test_tooltip_marker_keeps_the_party_colour_on_donuts():
    # ECharts builds the marker from the point's fill, white on a donut: the box draws it with the series colour
    script = ('(() => { const o = e.evolutionOption({runs: %s, parties: ["VOX"], colors: {VOX: "#63be21"}}); '
              'const white = window.echarts.format.getTooltipMarker("#fff"); '
              'return o.series[0].data.map((point) => o.tooltip.formatter([{value: Array.isArray(point) ? point : point.value, '
              'seriesName: "VOX", color: Array.isArray(point) ? "#63be21" : "#fff", marker: Array.isArray(point) ? "" : white}])); })()')
    boxes = run_js(script % json.dumps(day_runs('pct', [18.86, 18.87, 18.96])))
    assert all('background-color:#63be21' in box and 'background-color:#fff' not in box for box in boxes)
