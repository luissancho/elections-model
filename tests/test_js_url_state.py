"""Estado compartido del fragmento de la URL (`web/dist/js/url-state.js`), comprobado con `node` (se salta sin él)."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
JS_DIR = pathlib.Path(ROOT, 'web', 'dist', 'js')
URL_STATE_JS = (JS_DIR / 'url-state.js').as_uri()
BLOCKS_JS = (JS_DIR / 'blocks.js').as_uri()
SEATS_JS = (JS_DIR / 'seats.js').as_uri()
IDS = ['territorio', 'escanos', 'tabla']

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    script = (f'import * as u from "{URL_STATE_JS}"; import * as b from "{BLOCKS_JS}"; import * as s from "{SEATS_JS}"; '
              f'const IDS = {json.dumps(IDS)}; console.log(JSON.stringify({expr}));')
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_key_lists():
    assert run_js('b.SERIES_KEYS') == ['ventana', 'vista', 'sel', 'mostrar']
    assert run_js('s.COALITION_KEYS') == ['coalition']


def test_merge_hash_keeps_the_other_controls():
    assert run_js('u.mergeHash("#escanos&coalition=PP,VOX", "#ventana=1a&vista=partidos", b.SERIES_KEYS)') == \
        '#escanos&coalition=PP,VOX&ventana=1a&vista=partidos'
    assert run_js('u.mergeHash("#escanos&coalition=PP&ventana=all", "#ventana=3m", b.SERIES_KEYS)') == \
        '#escanos&coalition=PP&ventana=3m'


def test_merge_hash_empty_cases():
    assert run_js('u.mergeHash("#coalition=PP", "", s.COALITION_KEYS)') == ''
    assert run_js('u.mergeHash("", "#coalition=PP", s.COALITION_KEYS)') == '#coalition=PP'
    assert run_js('u.mergeHash("#escanos&coalition=PP", "", s.COALITION_KEYS)') == '#escanos'


def test_with_section():
    assert run_js('u.withSection("#ventana=1a", "escanos")') == '#escanos&ventana=1a'
    assert run_js('u.withSection("#tabla&ventana=1a", "escanos")') == '#escanos&ventana=1a'
    assert run_js('u.withSection("#tabla", null)') == ''
    assert run_js('u.withSection("#tabla&ventana=1a", null)') == '#ventana=1a'


def test_section_target():
    assert run_js('u.sectionTarget("?mode=forecast&seccion=escanos", "#coalition=PP", IDS)') == 'escanos'
    assert run_js('u.sectionTarget("", "#territorio&ventana=1a", IDS)') == 'territorio'
    assert run_js('u.sectionTarget("?seccion=nope", "#coalition=PP", IDS)') is None


def test_drop_param():
    assert run_js('u.dropParam("?mode=forecast&seccion=escanos", "seccion")') == '?mode=forecast'
    assert run_js('u.dropParam("?seccion=escanos", "seccion")') == ''


def test_existing_readers_ignore_the_section_token():
    ctx = '{parties: ["PP"], blocks: [], windows: ["1a", "all"]}'
    assert run_js(f'b.parseSeriesHash("#escanos&ventana=1a", {ctx})')['window'] == '1a'
    assert run_js('s.parseCoalition("#escanos&coalition=PP,VOX", ["PP", "PSOE", "VOX"])') == ['PP', 'VOX']


def test_coalition_location_hash_puts_the_page_on_the_seats_section():
    """Un enlace compartido de la calculadora abre en «Escaños»: con partidos marcados el fragmento lleva la sección
    `escanos` (la única suelta, primero) y conserva el resto del estado; sin partidos la sección se queda como está."""
    assert run_js('s.coalitionLocationHash("", ["PP", "VOX"])') == '#escanos&coalition=PP,VOX'
    assert run_js('s.coalitionLocationHash("#ventana=1a", ["PP"])') == '#escanos&ventana=1a&coalition=PP'
    assert run_js('s.coalitionLocationHash("#tabla&ventana=1a", ["PP"])') == '#escanos&ventana=1a&coalition=PP'
    assert run_js('s.coalitionLocationHash("#escanos&coalition=PP&ventana=1a", ["PP", "VOX"])') == \
        '#escanos&ventana=1a&coalition=PP,VOX'
    assert run_js('s.coalitionLocationHash("#tabla&coalition=PP&ventana=1a", [])') == '#tabla&ventana=1a'
    assert run_js('s.coalitionLocationHash("#coalition=PP", [])') == ''
