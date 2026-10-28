"""Recuadro de un sondeo del gráfico del promedio (`web/dist/js/polls-tooltip.js`) y funciones puras de la tabla de
sondeos (`web/dist/js/polls-table.js`), comprobados con `node` (se salta sin él)."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

from mtpy.lib import pages

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TOOLTIP_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'polls-tooltip.js').as_uri()
TABLE_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'polls-table.js').as_uri()
DOM_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'pages', 'polls-dom.js').as_uri()
POLL = {'date': '2026-10-01', 'name': 'GAD3 / ABC', 'sample_size': None, 'PP': 31.2, 'PSOE': 28.9, 'VOX': 12.0}

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr, poll=POLL):
    script = ('import * as s from "{}"; import * as p from "{}"; const poll = {}; '
              'console.log(JSON.stringify({}));').format(TOOLTIP_JS, TABLE_JS, json.dumps(poll), expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_poll_tooltip_lines_sorts_parties_and_tolerates_missing_fields():
    lines = run_js('s.pollTooltipLines(poll, ["PSOE", "PP", "VOX", "ERC"], {PP: "#1", PSOE: "#2", VOX: "#3"})')
    assert lines[0].startswith('<strong>GAD3 / ABC') and 'muestra –' in lines[1]
    assert [l for l in lines[2:]] and 'PP 31,2' in lines[2] and 'PSOE 28,9' in lines[3] and 'VOX 12,0' in lines[4] and len(lines) == 5


def test_poll_tooltip_lines_show_the_date_the_sample_the_colour_and_escape_the_text():
    poll = {'date': '2026-10-01', 'pollster': 'A&B', 'sample_size': 1200, 'PP': 30.0, 'X<1>': 2.0}
    lines = run_js('s.pollTooltipLines(poll, ["PP", "X<1>"], {PP: "#123456"})', poll=poll)
    # sin `name`, la encuestadora; la fecha y la muestra en la segunda línea
    assert lines[0] == '<strong>A&amp;B</strong>' and lines[1].startswith('1 oct 2026 · muestra ')
    assert '1200' in lines[1].replace('.', '') and 'undefined' not in ''.join(lines) and 'null' not in ''.join(lines)
    assert 'background-color:#123456' in lines[2] and lines[2].endswith('PP 30,0')
    assert lines[3].endswith('X&lt;1&gt; 2,0')
    # sin nombre ni encuestadora, ni fecha: guiones, nunca "undefined"; sin el campo `rating`, nada de rating
    bare = run_js('s.pollTooltipLines(poll, ["PP"], {})', poll={'PP': None})
    assert bare == ['<strong>–</strong>', '– · muestra –']


def test_poll_tooltip_lines_show_the_rating_of_the_house():
    """Luis, 2026-10-28: el recuadro de cada punto lleva también el rating de la casa (0-100, un decimal, como la
    insignia de la tabla); un guión para las casas sin rating (agregadores y paneles, con el campo a null). El
    ejemplo vivo del método no pasa el campo y conserva su línea de factores."""
    poll = {'date': '2026-10-01', 'pollster': 'GAD3', 'sample_size': 1200, 'rating': 59.07, 'PP': 30.0}
    lines = run_js('s.pollTooltipLines(poll, ["PP"], {})', poll=poll)
    assert lines[1] == '1 oct 2026 · muestra 1200 · rating 59,1'
    assert run_js('s.pollTooltipLines(poll, ["PP"], {})', poll={**poll, 'rating': None})[1] == '1 oct 2026 · muestra 1200 · rating –'
    without = {key: value for key, value in poll.items() if key != 'rating'}
    assert run_js('s.pollTooltipLines(poll, ["PP"], {})', poll=without)[1] == '1 oct 2026 · muestra 1200'


def test_mean_tooltip_line_shows_the_average_its_interval_and_the_date():
    line = run_js('s.meanTooltipLine("PP", 33.1, 31.0, 35.3, "2026-10-13", "#1")')
    assert line.endswith('PP: 33,1% (31,0–35,3) · 13 oct 2026') and 'background-color:#1' in line
    # without bounds there is no parenthesis (the history of the series chart, whose band is no longer drawn)
    bare = run_js('s.meanTooltipLine("PP", 33.1, null, null, "2026-10-13", "#1")')
    assert bare.endswith('PP: 33,1% · 13 oct 2026') and '(' not in bare
    projected = run_js('s.meanTooltipLine("PP", 33.1, null, null, "2026-11-29", "#1", true)')
    assert projected.startswith('Proyección<br>') and projected.endswith('PP: 33,1% · 29 nov 2026')
    election = run_js('s.meanTooltipLine("PP", 31.4, 25.9, 37.0, "2026-11-29", "#1", "Elección")')
    assert election.startswith('Elección<br>') and election.endswith('PP: 31,4% (25,9–37,0) · 29 nov 2026')


TABLE_POLLS = [
    {'date': '2026-10-01', 'pollster': 'GAD3', 'sponsor': 'ABC', 'sample_size': 1010, 'PP': 31.1, 'PSOE': 26.1},
    {'date': '2026-09-04', 'pollster': 'CIS', 'sponsor': None, 'sample_size': None, 'PP': 25.5, 'PSOE': 31.0},
    {'date': '2026-09-02', 'pollster': '40dB', 'sponsor': 'Prisa', 'sample_size': 2000, 'PP': None, 'PSOE': None},
]


def table_js(expr):
    return run_js(expr.replace('POLLS', json.dumps(TABLE_POLLS)))


def test_result_tooltip_line_names_the_official_result_and_the_date():
    line = run_js('s.resultTooltipLine("PP", 33.06, "2023-07-23", "#1d84ce")')
    assert 'Resultado oficial' in line and '23 jul 2023' in line and '#1d84ce' in line and line.index('<br>') < line.index('PP')
    assert '33,1' in line


def test_months_labels_and_short_dates_of_the_polls_table():
    assert table_js('p.monthsOf(POLLS)') == ['2026-09', '2026-10']
    assert table_js('p.monthOf({date: "2026-10-01"})') == '2026-10' and table_js('p.monthOf({})') is None
    assert table_js('p.monthLabel("2023-08")') == 'Ago 2023' and table_js('p.monthLabel("2026-12")') == 'Dic 2026'
    assert table_js('p.shortDate("2026-10-01")') == '01/10/26' and table_js('p.shortDate(null)') == '–'
    assert table_js('p.PAGE_SIZE') == 15


def test_filter_polls_by_pollster_and_inclusive_months():
    assert table_js('p.filterPolls(POLLS, {pollsters: ["CIS"], from: null, to: null}).length') == 1
    assert table_js('p.filterPolls(POLLS, {pollsters: null, from: "2026-10", to: "2026-10"}).length') == 1
    assert table_js('p.filterPolls(POLLS, {pollsters: null, from: "2026-09", to: "2026-10"}).length') == 3
    assert table_js('p.filterPolls(POLLS, {pollsters: [], from: null, to: null}).length') == 0
    assert table_js('p.filterPolls([{pollster: "X"}], {pollsters: null, from: "2026-09", to: null}).length') == 0


def test_sort_polls_puts_missing_values_last_in_both_directions_and_breaks_ties_by_date():
    assert table_js('p.sortPolls(POLLS, "PP", false).map(x => x.pollster)') == ['GAD3', 'CIS', '40dB']
    assert table_js('p.sortPolls(POLLS, "PP", true).map(x => x.pollster)') == ['CIS', 'GAD3', '40dB']
    assert table_js('p.sortPolls(POLLS, "date", true).map(x => x.pollster)') == ['40dB', 'CIS', 'GAD3']
    assert table_js('p.sortPolls(POLLS, "pollster", true).map(x => x.pollster)') == ['40dB', 'CIS', 'GAD3']
    assert table_js('p.sortPolls(POLLS, "sample_size", false).map(x => x.pollster)') == ['40dB', 'GAD3', 'CIS']
    assert table_js('p.sortPolls(POLLS, "sample_size", true).map(x => x.pollster)') == ['GAD3', '40dB', 'CIS']
    tie = [{'pollster': 'A', 'date': '2026-01-01', 'PP': 1}, {'pollster': 'B', 'date': '2026-02-01', 'PP': 1}]
    assert run_js('p.sortPolls({}, "PP", true).map(x => x.pollster)'.format(json.dumps(tie))) == ['B', 'A']
    assert table_js('POLLS.map(x => x.pollster)') == ['GAD3', 'CIS', '40dB']  # the input is not reordered


def test_page_label_mean_and_leader_of_the_polls_table():
    assert run_js('p.pageOf([1, 2, 3, 4], 1, 3)') == [4] and run_js('p.pageOf([1, 2, 3, 4], 0, 3)') == [1, 2, 3]
    assert run_js('p.pageLabel(0, 15, 315)') == '1–15 de 315' and run_js('p.pageLabel(2, 5, 35)') == '31–35 de 35'
    assert run_js('p.pageLabel(0, 0, 0)') == '0 de 0'
    assert table_js('p.meanOf(POLLS, ["PP", "PSOE"])') == {'PP': 28.3, 'PSOE': 28.6}
    assert table_js('p.meanOf(POLLS.slice(2), ["PP"])') == {'PP': None}
    assert run_js('p.leaderOf({PP: 31.1, PSOE: 31.1, VOX: null})') == ['PP', 'PSOE']
    assert run_js('p.leaderOf({PP: null})') == []


def test_sort_polls_keeps_the_server_order_of_same_date_ties_and_reverses_it_ascending():
    polls = [{'pollster': 'first', 'date': '2026-10-01'}, {'pollster': 'second', 'date': '2026-10-01'},
             {'pollster': 'old', 'date': '2026-09-01'}]
    expr = 'p.sortPolls({}, "date", {}).map(x => x.pollster)'
    assert run_js(expr.format(json.dumps(polls), 'false')) == ['second', 'first', 'old']
    assert run_js(expr.format(json.dumps(polls), 'true')) == ['old', 'first', 'second']
    by_value = [{'pollster': 'a', 'date': '2026-10-01', 'PP': 1}, {'pollster': 'b', 'date': '2026-10-01', 'PP': 1}]
    assert run_js('p.sortPolls({}, "PP", false).map(x => x.pollster)'.format(json.dumps(by_value))) == ['b', 'a']


def test_polls_count_is_singular_for_one_and_the_month_filter_ignores_empty_limits():
    assert run_js('[p.pollsCount(1), p.pollsCount(0), p.pollsCount(315)]') == ['1 sondeo', '0 sondeos', '315 sondeos']
    undated = [{'pollster': 'CIS'}, {'pollster': 'GAD3'}]
    assert run_js('p.filterPolls({}, {{pollsters: null, from: "", to: ""}}).length'.format(json.dumps(undated))) == 2


def test_seats_view_uses_the_pollsters_projections():
    polls = [
        {'date': '2026-10-01', 'pollster': 'GAD3', 'PP': 31.1, 'PSOE': 26.1, 'seats': {'PP': [138, 142], 'PSOE': [100, 100]}},
        {'date': '2026-09-04', 'pollster': 'CIS', 'PP': 25.5, 'PSOE': 31.0},
        {'date': '2026-09-02', 'pollster': '40dB', 'PP': 33.0, 'PSOE': 27.0, 'seats': {'PP': [150, 150]}},
    ]
    js = json.dumps(polls)
    assert run_js('p.hasSeats({}[0]) && !p.hasSeats({}[1])'.format(js, js)) is True
    assert run_js('[p.pollValue({}[0], "PP", "esc"), p.pollValue({}[0], "PSOE", "esc"), p.pollValue({}[1], "PP", "esc"), p.pollValue({}[0], "PP")]'.format(js, js, js, js)) == [140, 100, None, 31.1]
    assert run_js('[p.seatsText({}[0], "PP"), p.seatsText({}[0], "PSOE"), p.seatsText({}[1], "PP")]'.format(js, js, js)) == ['138–142', '100', None]
    assert run_js('p.filterPolls({}, {{pollsters: null, from: null, to: null, mode: "esc"}}).map((x) => x.pollster)'.format(js)) == ['GAD3', '40dB']
    assert run_js('p.filterPolls({}, {{pollsters: null, from: null, to: null}}).length'.format(js)) == 3
    assert run_js('p.sortPolls({}, "PP", false, "esc").map((x) => x.pollster)'.format(js)) == ['40dB', 'GAD3', 'CIS']  # sin escaños, al final
    assert run_js('p.meanOf({}, ["PP", "PSOE"], "esc")'.format(js)) == {'PP': 145, 'PSOE': 100}  # puntos medios, enteros
    assert run_js('p.meanOf({}, ["PP"])'.format(js)) == {'PP': 29.9}


def test_fixed_rows_of_the_footer_follow_the_mode_and_the_validity_of_the_seats():
    last = {'horizon': 6, 'date': '2023-07-17', 'pct': {'PP': 32.4, 'PSOE': 31.0}, 'seats': {'PP': 140, 'PSOE': 118}, 'seats_valid': True}
    result = {'date': '2023-07-23', 'values': {'PP': 33.1, 'PSOE': 31.4}}
    seats = {'PP': 137, 'PSOE': 121}
    args = '{}, {}, {}, ["PP", "PSOE", "VOX"]'.format(json.dumps(last), json.dumps(result), json.dumps(seats))
    rows = run_js('p.fixedRows({})'.format(args))
    assert rows == [
        {'label': 'Promedio de sondeos', 'small': 'modelo · a 6 días', 'title': 'Congelado el 17 jul 2023: sólo los sondeos publicados hasta entonces',
         'values': {'PP': 32.4, 'PSOE': 31.0, 'VOX': None}, 'texts': None},
        {'label': 'Resultado', 'small': '23 jul 2023', 'title': None, 'values': {'PP': 33.1, 'PSOE': 31.4, 'VOX': None}, 'texts': None}]
    esc = run_js('p.fixedRows({}, "esc")'.format(args))
    assert esc[0]['values'] == {'PP': 140, 'PSOE': 118, 'VOX': None} and {k: v for k, v in esc[0]['texts'].items() if k != 'VOX'} == {'PP': '140', 'PSOE': '118'}
    assert esc[1]['values'] == {'PP': 137, 'PSOE': 121, 'VOX': None}
    invalid = run_js('p.fixedRows({}, "esc")'.format(args.replace('"seats_valid": true', '"seats_valid": false')))
    assert invalid[0]['values'] == {'PP': None, 'PSOE': None, 'VOX': None}
    assert run_js('p.fixedRows(null, {}, {}, ["PP"])'.format(json.dumps(result), json.dumps(seats))) == []


def test_sort_polls_by_rating_puts_the_unrated_last_in_both_directions():
    polls = [{'pollster': 'A', 'date': '2026-10-01', 'rating': 60.0}, {'pollster': 'B', 'date': '2026-09-01'},
             {'pollster': 'C', 'date': '2026-08-01', 'rating': 73.1}]
    expr = 'p.sortPolls({}, "rating", {}).map(x => x.pollster)'
    assert run_js(expr.format(json.dumps(polls), 'false')) == ['C', 'A', 'B']
    assert run_js(expr.format(json.dumps(polls), 'true')) == ['A', 'C', 'B']
    # the rating is a fixed column: the seats view sorts by it too (a party key would read the seat projections)
    expr = 'p.sortPolls({}, "rating", false, "esc").map(x => x.pollster)'
    assert run_js(expr.format(json.dumps(polls))) == ['C', 'A', 'B']


def test_polls_dom_fixed_columns_and_rating_colours_follow_the_server():
    """La tabla que redibuja el navegador lleva la columna del rating entre la encuestadora y la fecha, con los
    colores del degradado del servidor (`pages.RATING_COLORS`)."""
    script = 'import * as d from "{}"; console.log(JSON.stringify([d.FIXED_COLUMNS, d.RATING_COLORS]));'.format(DOM_JS)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    fixed, colors = json.loads(out.stdout)
    assert fixed == [['pollster', 'Encuestadora'], ['rating', 'Rating (0-100)'], ['date', 'Fecha'], ['sample_size', 'Muestra']]
    assert colors == list(pages.RATING_COLORS)
