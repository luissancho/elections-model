"""Lógica pura del histórico de errores (`web/dist/js/errors.js`), comprobada con `node` (se salta sin él)."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ERRORS_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'errors.js').as_uri()
ROWS = [
    {'event': '2023-23J', 'event_date': '2023-07-23', 'pollster': 'A', 'days': 7, 'error_avg': 1.0, 'error_blocks_abs': 2.0},
    {'event': '2023-23J', 'event_date': '2023-07-23', 'pollster': 'B', 'days': 13, 'error_avg': 3.0, 'error_blocks_abs': 4.0},
    {'event': '2019-10N', 'event_date': '2019-11-10', 'pollster': 'A', 'days': 20, 'error_avg': 2.0, 'error_blocks_abs': None},
]

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    script = 'import * as e from "{}"; const rows = {}; console.log(JSON.stringify({}));'.format(
        ERRORS_JS, json.dumps(ROWS), expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_filter_mean_and_weeks():
    assert len(run_js('e.filterRows(rows, {event: "2023-07-23"})')) == 2
    assert len(run_js('e.filterRows(rows, {pollster: "A"})')) == 2
    assert run_js('e.meanOf(rows, "error_blocks_abs")') == 3.0 and run_js('e.meanOf([], "error_avg")') is None
    assert run_js('e.weekOf(6)') == 1 and run_js('e.weekOf(13)') == 2 and run_js('e.weekOf(42)') == 6
    table = run_js('e.weeksTable(rows, "error_avg")')
    assert table['weeks'] == [1, 2, 3] and table['rows'][0]['event'] == '2023-23J' and table['rows'][-1]['event'] == 'Todas'
    assert table['rows'][-1]['cells']['1'] == {'n': 1, 'mean': 1.0}


def test_errors_hash_round_trip_and_unknown_values_fall_back():
    ctx = '{events: ["2023-07-23", "2019-11-10"], pollsters: ["A", "B"]}'
    assert run_js(f'e.parseErrorsHash("#eleccion=2019-11-10&casa=B&medida=error_avg", {ctx})') == {'event': '2019-11-10', 'pollster': 'B', 'measure': 'error_avg'}
    assert run_js(f'e.parseErrorsHash("#eleccion=2011-11-20&casa=Z&medida=x", {ctx})') == {'event': None, 'pollster': None, 'measure': 'error_blocks_abs'}
    assert run_js('e.errorsHash({event: "2019-11-10", pollster: null, measure: "error_blocks_abs"})') == '#eleccion=2019-11-10'
    assert run_js('e.errorsHash({event: null, pollster: null, measure: "error_blocks_abs"})') == ''


def test_weeks_table_cells_and_order():
    table = run_js('e.weeksTable(rows, "error_blocks_abs")')
    assert [row['event_date'] for row in table['rows']] == ['2023-07-23', '2019-11-10', None]
    # una fila sin la medida cuenta como sondeo pero no entra en la media
    assert table['rows'][1]['cells']['3'] == {'n': 1, 'mean': None}
    assert table['rows'][0]['cells'] == {'1': {'n': 1, 'mean': 2.0}, '2': {'n': 1, 'mean': 4.0}}
    assert run_js('e.weeksTable([], "error_avg")') == {'weeks': [], 'rows': [{'event': 'Todas', 'event_date': None, 'cells': {}}]}


def test_errors_hash_encodes_names_and_parses_them_back():
    ctx = '{events: ["2023-07-23"], pollsters: ["Sigma Dos", "40dB"]}'
    hash_ = run_js('e.errorsHash({event: null, pollster: "Sigma Dos", measure: "error_avg"})')
    assert hash_ == '#casa=Sigma%20Dos&medida=error_avg'
    assert run_js(f'e.parseErrorsHash({json.dumps(hash_)}, {ctx})') == {'event': None, 'pollster': 'Sigma Dos', 'measure': 'error_avg'}
    assert run_js(f'e.parseErrorsHash("#casa=%E0%A4%A", {ctx})') == {'event': None, 'pollster': None, 'measure': 'error_blocks_abs'}
    assert run_js(f'e.parseErrorsHash("", {ctx})') == {'event': None, 'pollster': None, 'measure': 'error_blocks_abs'}
    assert run_js(f'e.parseErrorsHash("#medida=toString", {ctx})')['measure'] == 'error_blocks_abs'
