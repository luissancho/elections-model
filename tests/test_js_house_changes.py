"""El selector % Voto / Escaños de «Cambios por encuestadora» (`web/dist/js/pages/house-changes-wire.js`), comprobado
con `node` y un documento de mentira (se salta sin él)."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIST = os.path.join(ROOT, 'web', 'dist')
ECHARTS = os.path.join(DIST, 'vendor', 'echarts-5.6.0.min.js')
MODULE = pathlib.Path(DIST, 'js', 'pages', 'house-changes-wire.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')

FAKES = (
    'const el = (hidden) => ({hidden, listeners: {}, addEventListener(type, fn) { this.listeners[type] = fn; }, '
    'querySelectorAll: () => [], querySelector: () => null}); '
    'const nodes = {"house-changes": el(false), "changes-metric": el(true), "changes-pct": el(false), '
    '"changes-seats": el(true)}; '
    'const button = {dataset: {file: "demoscopio-cambios-es-2026-11-11.png"}}; '
    'const pill = {textContent: "% Voto"}; '
    'const parts = {"#changes-download": button, ".card-head .pill": pill}; '
    'nodes["house-changes"].querySelector = (sel) => parts[sel] || null; '
    'const doc = {getElementById: (id) => nodes[id] || null}; '
    'const ids = ["changes-metric", "changes-pct", "changes-seats"]; '
    'const shown = () => ids.map((id) => (nodes[id] ? nodes[id].hidden : null)); '
    'const pick = (value) => nodes["changes-metric"].listeners.change({target: {value}}); '
)


def run_js(steps):
    """Ejecuta `steps` (sentencias que acaban en `return`) con `m` (el módulo) y el documento de `FAKES`; da su
    JSON."""
    script = (
        'import {{createRequire}} from "node:module"; '
        'const echarts = createRequire(import.meta.url)({}); globalThis.window = {{echarts}}; '
        'const m = await import("{}"); console.log(JSON.stringify((() => {{ {} {} }})())); process.exit(0);'
    ).format(json.dumps(ECHARTS), MODULE, FAKES, steps)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True,
                         cwd=ROOT)

    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_switch_shows_one_table_at_a_time():
    steps = ('m.wireHouseChanges(doc); const start = shown(); pick("seats"); const seats = shown(); pick("pct"); '
             'return [start, seats, shown()];')
    assert run_js(steps) == [[False, False, True], [False, True, False], [False, False, True]]


def test_changes_file_names_the_image_of_each_view():
    """La imagen de Escaños lleva su palabra en el nombre (como `evolutionFile`); la de voto, el del botón."""
    vote, seats = 'demoscopio-cambios-es-2026-11-11.png', 'demoscopio-cambios-escanos-es-2026-11-11.png'
    assert run_js('return m.changesFile("%s", "seats");' % vote) == seats
    assert run_js('return m.changesFile("%s", "seats");' % seats) == seats
    assert run_js('return m.changesFile("%s", "pct");' % seats) == vote
    assert run_js('return m.changesFile("%s", "pct");' % vote) == vote


def test_the_switch_renames_the_file_of_the_image_button():
    steps = ('m.wireHouseChanges(doc); const start = button.dataset.file; pick("seats"); '
             'const seats = button.dataset.file; pick("pct"); return [start, seats, button.dataset.file];')
    assert run_js(steps) == ['demoscopio-cambios-es-2026-11-11.png', 'demoscopio-cambios-escanos-es-2026-11-11.png',
                             'demoscopio-cambios-es-2026-11-11.png']


def test_the_switch_names_the_view_in_the_pill_of_the_title():
    """La pastilla del título dice qué vista lleva la imagen descargada, donde el selector no sale."""
    steps = ('m.wireHouseChanges(doc); const start = pill.textContent; pick("seats"); '
             'const seats = pill.textContent; pick("pct"); return [start, seats, pill.textContent];')
    assert run_js(steps) == ['% Voto', 'Escaños', '% Voto']


def test_wiring_without_the_card_or_without_seats_does_nothing():
    assert run_js('m.wireHouseChanges({getElementById: () => null}); return shown();') == [True, False, True]
    steps = 'delete nodes["changes-metric"]; delete nodes["changes-seats"]; m.wireHouseChanges(doc); return shown();'
    assert run_js(steps) == [None, False, None]
