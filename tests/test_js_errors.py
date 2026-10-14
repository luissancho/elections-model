"""Lógica pura del histórico de errores (`web/dist/js/errors.js`), comprobada con `node` (se salta sin él)."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ERRORS_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'errors.js').as_uri()


def row(scope, scope_name, year, date, event, pollster, days, avg, blocks_abs, **extra):
    return {'scope': scope, 'scope_name': scope_name, 'year': year, 'event_date': date, 'event': event,
            'pollster': pollster, 'days': days, 'error_avg': avg, 'error_blocks_abs': blocks_abs, **extra}


ROWS = [
    row('es', 'España', 2019, '2019-11-10', '10N', 'CIS', 20, 2.0, None, Derecha=1.0, Izquierda=-1.0),
    row('es', 'España', 2019, '2019-11-10', '10N', 'GAD3', 20, 4.0, 5.0, Derecha=1.0, Izquierda=-1.0),
    row('es-md', 'Madrid', 2021, '2021-05-04', '4M', 'CIS', 7, 5.0, 6.0, Derecha=1.0, Izquierda=-1.0),
    row('es-md', 'Madrid', 2021, '2021-05-04', '4M', 'GAD3', 7, 7.0, 8.0),
    row('es', 'España', 2023, '2023-07-23', '23J', 'CIS', 13, 1.0, 2.0, Derecha=1.0, Izquierda=-1.0),
    row('es', 'España', 2023, '2023-07-23', '23J', 'GAD3', 7, 3.0, 4.0, Derecha=1.0, Izquierda=-1.0),
]

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    script = 'import * as e from "{}"; const rows = {}; console.log(JSON.stringify({}));'.format(
        ERRORS_JS, json.dumps(ROWS), expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_filter_by_scope_event_pollster_and_years():
    assert len(run_js('e.filterRows(rows, {scope: "es"})')) == 4
    assert [r['year'] for r in run_js('e.filterRows(rows, {from: 2021, to: 2023})')] == [2021, 2021, 2023, 2023]
    assert run_js('e.filterRows(rows, {from: 2023, to: 2019})') == run_js('e.filterRows(rows, {from: 2019, to: 2023})')
    assert len(run_js('e.filterRows(rows, {pollster: "CIS", scope: "es-md"})')) == 1
    assert run_js('e.filterRows(rows, {event: "es:2023-07-23"})') == run_js('e.filterRows(rows, {from: 2023})')
    # same day, other scope: not the es rows
    other = run_js('e.filterRows([...rows, {...rows[0], scope: "es-md", scope_name: "Madrid", event_date: "2023-07-23", year: 2023}], {event: "es-md:2023-07-23"})')
    assert len(other) == 1 and other[0]['scope'] == 'es-md'
    assert run_js('e.eventKey(rows[0])') == 'es:2019-11-10'


def test_years_and_events():
    assert run_js('e.yearsOf(rows)') == [2019, 2021, 2023]
    events = run_js('e.eventsOf(rows, "error_avg")')
    assert [(x['scope'], x['event_date'], x['n'], x['mean']) for x in events] == [
        ('es', '2019-11-10', 2, 3.0), ('es-md', '2021-05-04', 2, 6.0), ('es', '2023-07-23', 2, 2.0)]
    assert events[1]['event'] == '4M' and events[1]['scope_name'] == 'Madrid'
    # the CIS 2019 row has no error_blocks_abs: it does not count
    assert run_js('e.eventsOf(rows, "error_blocks_abs")')[0] == {
        'event_date': '2019-11-10', 'event': '10N', 'name': None, 'scope': 'es', 'scope_name': 'España', 'n': 1, 'mean': 5.0}
    named = run_js('e.eventsOf(rows.map((r) => ({...r, name: r.event === "4M" ? "Elecciones Madrid Mayo 2021" : null})), "error_avg")')
    assert [x['name'] for x in named] == [None, 'Elecciones Madrid Mayo 2021', None]


def test_complements():
    out = run_js('e.complements(rows, {pollster: "CIS", from: 2021, to: 2023}, "error_avg")')
    assert out == [{'key': 'pollster', 'label': 'Otras encuestadoras', 'n': 2, 'mean': 5.0},
                   {'key': 'years', 'label': 'Otros años', 'n': 1, 'mean': 2.0}]
    assert run_js('e.complements(rows, {}, "error_avg")') == []
    # the extremes are no year filter
    assert run_js('e.complements(rows, {from: 2019, to: 2023}, "error_avg")') == []
    # "Otros ámbitos" is empty when the chosen election is in the chosen scope: omitted
    out = run_js('e.complements(rows, {event: "es:2023-07-23"}, "error_avg")')
    assert [(x['label'], x['n'], x['mean']) for x in out] == [('Otras elecciones', 4, 4.5)]
    out = run_js('e.complements(rows, {scope: "es-md"}, "error_avg")')
    assert [(x['label'], x['n'], x['mean']) for x in out] == [('Otros ámbitos', 4, 2.5)]
    assert run_js('e.complements(rows, {scope: "es", event: "es:2023-07-23"}, "error_avg")') == [
        {'key': 'event', 'label': 'Otras elecciones', 'n': 2, 'mean': 3.0}]


def test_trend_and_rest_points():
    assert run_js('e.trendPoints(rows, "error_avg", {window: 2})') == [
        ['2019-11-10', 3.0], ['2021-05-04', 4.5], ['2023-07-23', 4.0]]
    # default window 5 covers everything: (3*2 + 6*2 + 2*2) / 6
    assert run_js('e.trendPoints(rows, "error_avg")')[2] == [ '2023-07-23', 22 / 6]
    assert run_js('e.restPoints(rows, "CIS", "error_avg")') == [
        ['2019-11-10', 4.0], ['2021-05-04', 7.0], ['2023-07-23', 3.0]]
    assert run_js('e.restPoints(rows, "Nadie", "error_avg")') == []


def test_filter_mean_and_weeks():
    assert run_js('e.meanOf(rows, "error_blocks_abs")') == 5.0 and run_js('e.meanOf([], "error_avg")') is None
    assert run_js('e.weekOf(6)') == 1 and run_js('e.weekOf(13)') == 2 and run_js('e.weekOf(42)') == 6
    table = run_js('e.weeksTable(rows, "error_avg")')
    assert table['weeks'] == [1, 2, 3]
    assert [r['event'] for r in table['rows']] == ['23J · España', '4M · Madrid', '10N · España', 'Todas']
    assert table['rows'][-1]['cells']['1'] == {'n': 3, 'mean': 5.0}
    # with the full name of the election, the row is labelled by it alone (no scope)
    named = run_js('e.weeksTable(rows.map((r) => ({...r, name: r.event === "23J" ? "Elecciones Generales Julio 2023" : null})), "error_avg")')
    assert [r['event'] for r in named['rows']][:2] == ['Elecciones Generales Julio 2023', '4M · Madrid']


def test_weeks_table_groups_by_scope_and_date():
    table = run_js('e.weeksTable([...rows, {...rows[4], scope: "es-md", scope_name: "Madrid", event: "X"}], "error_avg")')
    assert len(table['rows']) == 5 and table['rows'][0]['event'] in ('23J · España', 'X · Madrid')


CTX = ('{scopes: ["es", "es-md"], events: ["es:2023-07-23", "es-md:2021-05-04", "es:2019-11-10"], '
       'pollsters: ["CIS", "GAD3"], years: [2019, 2021, 2023]}')


def test_errors_hash_parse_and_round_trip():
    assert run_js(f'e.parseErrorsHash("#ambito=es-md&desde=2021&hasta=2023&casa=CIS", {CTX})') == {
        'scope': 'es-md', 'event': None, 'pollster': 'CIS', 'from': 2021, 'to': None, 'measure': 'error_blocks_abs'}
    assert run_js(f'e.parseErrorsHash("#eleccion=es:2023-07-23", {CTX})')['event'] == 'es:2023-07-23'
    assert run_js(f'e.parseErrorsHash("#desde=2030&hasta=2019", {CTX})')['from'] is None
    assert run_js(f'e.parseErrorsHash("#desde=2030&hasta=2021", {CTX})')['to'] == 2021
    assert run_js(f'e.parseErrorsHash("#ambito=zz&eleccion=zz:1&casa=Z&medida=x", {CTX})') == {
        'scope': None, 'event': None, 'pollster': None, 'from': None, 'to': None, 'measure': 'error_blocks_abs'}
    state = '{scope: "es-md", event: "es-md:2021-05-04", pollster: "CIS", from: 2021, to: 2021, measure: "error_avg"}'
    hash_ = run_js(f'e.errorsHash({state})')
    assert hash_ == '#ambito=es-md&eleccion=es-md%3A2021-05-04&casa=CIS&desde=2021&hasta=2021&medida=error_avg'
    assert run_js(f'e.parseErrorsHash({json.dumps(hash_)}, {CTX})') == {
        'scope': 'es-md', 'event': 'es-md:2021-05-04', 'pollster': 'CIS', 'from': 2021, 'to': 2021, 'measure': 'error_avg'}
    assert run_js('e.errorsHash({scope: null, event: null, pollster: null, from: null, to: null, measure: "error_blocks_abs"})') == ''
    # An explicit empty `ambito=` (Todos with a page default) writes and parses as no scope.
    assert run_js('e.errorsHash({scope: "", event: null, pollster: null, from: null, to: null, measure: "error_blocks_abs"})') == '#ambito='
    assert run_js(f'e.parseErrorsHash("#ambito=&casa=CIS", {CTX})')['scope'] is None


CTX2 = ('{scopes: ["es", "es-cv"], events: ["es:2019-04-28", "es-cv:2019-04-28"], pollsters: [], years: [2019]}')
CTX3 = ('{defaultScope: "es-md", scopes: ["es", "es-md", "es-cv"], events: ["es-md:2019-05-26", "es-cv:2019-05-26"], '
        'pollsters: [], years: [2019]}')


def test_swapped_years_come_back_in_order():
    parsed = run_js(f'e.parseErrorsHash("#desde=2023&hasta=2021", {CTX})')
    assert (parsed['from'], parsed['to']) == (2021, 2023)


def test_weeks_table_without_valid_rows_is_empty():
    assert run_js('e.weeksTable([], "error_avg")') == {'weeks': [], 'rows': []}
    assert run_js('e.weeksTable([{...rows[0], days: null}], "error_avg")') == {'weeks': [], 'rows': []}


def test_legacy_event_hash_resolves_to_a_unique_key():
    legacy = '#eleccion=2023-07-23&casa=GAD3&medida=error_avg'
    assert run_js(f'e.parseErrorsHash("{legacy}", {CTX})') == {
        'scope': None, 'event': 'es:2023-07-23', 'pollster': 'GAD3', 'from': None, 'to': None, 'measure': 'error_avg'}
    assert run_js(f'e.parseErrorsHash("#eleccion=2019-04-28", {CTX2})')['event'] == 'es:2019-04-28'
    assert run_js(f'e.parseErrorsHash("#eleccion=2019-05-26", {CTX3})')['event'] == 'es-md:2019-05-26'
    both = CTX.replace('"es:2023-07-23",', '"es:2023-07-23", "es-md:2023-07-23",')
    assert run_js(f'e.parseErrorsHash("{legacy}", {both})')['event'] == 'es:2023-07-23'
    assert run_js(f'e.parseErrorsHash("#ambito=es-md&eleccion=2023-07-23", {both})')['event'] == 'es-md:2023-07-23'
    assert run_js(f'e.parseErrorsHash("#eleccion=2011-11-20", {CTX})')['event'] is None


def test_errors_hash_encodes_names_and_ignores_bad_escapes():
    hash_ = run_js('e.errorsHash({scope: null, event: null, pollster: "Sigma Dos", from: null, to: null, measure: "error_avg"})')
    assert hash_ == '#casa=Sigma%20Dos&medida=error_avg'
    ctx = '{scopes: [], events: [], pollsters: ["Sigma Dos"], years: []}'
    assert run_js(f'e.parseErrorsHash({json.dumps(hash_)}, {ctx})')['pollster'] == 'Sigma Dos'
    assert run_js(f'e.parseErrorsHash("#casa=%E0%A4%A", {ctx})')['pollster'] is None


def test_weeks_table_cells_and_order():
    table = run_js('e.weeksTable(rows, "error_blocks_abs")')
    assert [r['event_date'] for r in table['rows']] == ['2023-07-23', '2021-05-04', '2019-11-10', None]
    # a row without the measure counts as a poll but not in the mean
    assert table['rows'][2]['cells']['3'] == {'n': 2, 'mean': 5.0}
    assert table['rows'][0]['cells'] == {'1': {'n': 1, 'mean': 4.0}, '2': {'n': 1, 'mean': 2.0}}
    assert run_js('e.weeksTable([], "error_avg")') == {'weeks': [], 'rows': []}
