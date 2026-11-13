"""Los nombres de las imágenes de «Escaños que bailan» y la línea del filtro del mapa (`web/dist/js/seat-odds.js`),
comprobados con `node` (se salta sin él). El módulo no importa nada, así que el arnés no carga ECharts."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODULE = pathlib.Path(ROOT, 'web', 'dist', 'js', 'seat-odds.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    """Evalúa `expr` con `s` (el módulo) definido y devuelve su JSON."""
    script = 'const s = await import("{}"); console.log(JSON.stringify({}));'.format(MODULE, expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT)

    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_image_is_named_after_what_is_on_view():
    assert run_js("s.imageFile('demoscopio-escanos-es-mapa', ['psoe', 'en-juego'])") == 'demoscopio-escanos-es-mapa-psoe-en-juego.png'
    assert run_js("s.imageFile('demoscopio-escanos-es-mapa', ['', ''])") == 'demoscopio-escanos-es-mapa.png'
    assert run_js("s.imageFile('demoscopio-escanos-es', ['columnas'])") == 'demoscopio-escanos-es-columnas.png'
    assert run_js("[s.filterLine('PSOE', 'En juego'), s.filterLine('PSOE', ''), s.filterLine('', 'Seguros'), "
                  "s.filterLine('', '')]") == ['PSOE · En juego', 'PSOE', 'Seguros', '']


def test_the_canonical_of_a_tile_keeps_the_origin_and_drops_the_query():
    # the address of a tile keeps the mode and the run of the page; its canonical, as the server writes it, does not
    assert run_js("s.canonicalFor('https://demoscopio.es/es/circunscripciones', "
                  "'/es/circunscripciones/cadiz?mode=nowcast&run=20261113-094116')") == (
        'https://demoscopio.es/es/circunscripciones/cadiz')
    assert run_js("s.canonicalFor('http://127.0.0.1:8771/es/circunscripciones/la-coruna', "
                  "'/es/circunscripciones/cadiz')") == 'http://127.0.0.1:8771/es/circunscripciones/cadiz'
