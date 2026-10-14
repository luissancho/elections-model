"""Recuadro de un sondeo del gráfico del promedio (`web/dist/js/polls-tooltip.js`) y funciones puras de la tabla de
sondeos (`web/dist/js/polls-table.js`), comprobados con `node` (se salta sin él)."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TOOLTIP_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'polls-tooltip.js').as_uri()
TABLE_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'polls-table.js').as_uri()
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
    # sin nombre ni encuestadora, ni fecha: guiones, nunca "undefined"
    bare = run_js('s.pollTooltipLines(poll, ["PP"], {})', poll={'PP': None})
    assert bare == ['<strong>–</strong>', '– · muestra –']


def test_mean_tooltip_line_shows_the_average_its_interval_and_the_date():
    line = run_js('s.meanTooltipLine("PP", 33.1, 31.0, 35.3, "2026-10-13", "#1")')
    assert line.endswith('PP: 33,1 % (31,0–35,3) · 13 oct 2026') and 'background-color:#1' in line
    projected = run_js('s.meanTooltipLine("PP", 33.1, null, null, "2026-11-29", "#1", true)')
    assert projected.startswith('Proyección<br>') and projected.endswith('PP: 33,1 % (–) · 29 nov 2026')


TABLE_POLLS = [
    {'date': '2026-10-01', 'pollster': 'GAD3', 'sponsor': 'ABC', 'sample_size': 1010, 'PP': 31.1, 'PSOE': 26.1},
    {'date': '2026-09-04', 'pollster': 'CIS', 'sponsor': None, 'sample_size': None, 'PP': 25.5, 'PSOE': 31.0},
    {'date': '2026-09-02', 'pollster': '40dB', 'sponsor': 'Prisa', 'sample_size': 2000, 'PP': None, 'PSOE': None},
]


def table_js(expr):
    return run_js(expr.replace('POLLS', json.dumps(TABLE_POLLS)))


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
