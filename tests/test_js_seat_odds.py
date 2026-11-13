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


def test_a_card_is_out_of_view_only_when_none_of_it_shows():
    assert run_js("[s.outOfView({top: 900, bottom: 1300}, 800), s.outOfView({top: 700, bottom: 1300}, 800), "
                  "s.outOfView({top: -500, bottom: -10}, 800), s.outOfView({top: -500, bottom: 10}, 800)]") == [
        True, False, True, False]
    assert run_js("s.imageFile('demoscopio-escanos-es', ['madrid', 'psoe', 'en-juego'])") == (
        'demoscopio-escanos-es-madrid-psoe-en-juego.png')
