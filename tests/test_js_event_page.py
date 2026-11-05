"""Página del evento (`web/dist/js/pages/promedios.js`) cargada en `node` con un documento mínimo (se salta sin `node`):
cada bloque se dibuja por su cuenta, así que un fallo en uno no impide los demás."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PAGE_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'pages', 'promedios.js').as_uri()
ECHARTS = os.path.join(ROOT, 'web', 'dist', 'vendor', 'echarts-5.6.0.min.js')

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def requests_of(initial):
    """URLs pedidas con `fetch` al montar la página sobre `initial`, con un documento que sólo tiene el bloque
    `initial-data` (el resto de la página no está, así que cada dibujo falla o no hace nada)."""
    script = (
        'import {{createRequire}} from "node:module"; '
        'const echarts = createRequire(import.meta.url)({}); '
        'const nodes = {{"initial-data": {{textContent: {}}}}}; '
        'globalThis.window = {{echarts, location: {{pathname: "/es", search: "", hash: ""}}, history: {{replaceState() {{}}}}, '
        'innerWidth: 1200, addEventListener() {{}}}}; '
        'globalThis.document = {{getElementById: (id) => nodes[id] || null, querySelector: () => null, '
        'querySelectorAll: () => [], addEventListener() {{}}}}; '
        'const requests = []; globalThis.fetch = async (href) => {{ requests.push(href); return {{ok: false, status: 503}}; }}; '
        'console.error = () => {{}}; '
        'await import("{}"); await new Promise((resolve) => setTimeout(resolve, 50)); '
        'console.log(JSON.stringify(requests)); process.exit(0);'
    ).format(json.dumps(ECHARTS), json.dumps(json.dumps(initial)), PAGE_JS)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout)


def test_the_seats_are_wired_even_when_the_results_fail():
    """Un fallo al dibujar el hemiciclo (aquí, un `headline` sin el modo de la página) no se salta los escaños: la
    calculadora sigue pidiendo sus simulaciones a `dist_href`."""
    meta = {'majority': 176, 'n_seats': 350, 'parties': [{'name': 'PP', 'color': '#1e5aa8'}], 'bmaps': {}}
    initial = {'state': {'scope': 'es', 'mode': 'forecast', 'run': 'r', 'pinned': False}, 'meta': meta, 'headline': {},
               'summary': {'parties': [{'name': 'PP', 'seats': 140}]}, 'dist_href': '/api/v1/forecast/es/forecast/dist?run=r'}
    assert requests_of(initial) == ['/api/v1/forecast/es/forecast/dist?run=r']
