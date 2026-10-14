"""Recuadro de un sondeo del gráfico del promedio (`web/dist/js/polls-tooltip.js`) y carga progresiva de la tabla de
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


def test_next_rows_pages_by_forty_and_stops():
    assert run_js('p.nextRows(Array.from({length: 95}, (_, i) => i), 40)') == list(range(40, 80))
    assert run_js('p.nextRows(Array.from({length: 95}, (_, i) => i), 80)') == list(range(80, 95))
    assert run_js('p.nextRows(Array.from({length: 30}, (_, i) => i), 30)') == []
    assert run_js('p.nextRows([1, 2, 3], 1, 1)') == [2]
    assert run_js('p.nextRows([], 0)') == []


def test_caption_counts_the_rows_shown_and_says_all_at_the_end():
    assert run_js('p.captionText(80, 315)') == 'Últimos 80 de 315 sondeos'
    assert run_js('p.captionText(315, 315)') == 'Todos los sondeos del ciclo (315)'
