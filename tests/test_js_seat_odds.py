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
    assert run_js("s.imageFile('demoscopio-escanos-es', ['madrid', 'psoe', 'en-juego'])") == (
        'demoscopio-escanos-es-madrid-psoe-en-juego.png')
    assert run_js("s.imageFile('demoscopio-escanos-es', ['cadiz', '', 'seguros'])") == 'demoscopio-escanos-es-cadiz-seguros.png'
    assert run_js("s.imageFile('demoscopio-escanos-es', ['madrid', '', ''])") == 'demoscopio-escanos-es-madrid.png'
    assert run_js("s.imageFile('demoscopio-escanos-es', ['columnas'])") == 'demoscopio-escanos-es-columnas.png'
    assert run_js("[s.filterLine('PSOE', 'En juego'), s.filterLine('PSOE', ''), s.filterLine('', 'Seguros'), "
                  "s.filterLine('', '')]") == ['PSOE · En juego', 'PSOE', 'Seguros', '']


def test_a_card_is_out_of_view_only_when_none_of_it_shows():
    assert run_js("[s.outOfView({top: 900, bottom: 1300}, 800), s.outOfView({top: 700, bottom: 1300}, 800), "
                  "s.outOfView({top: -500, bottom: -10}, 800), s.outOfView({top: -500, bottom: 10}, 800)]") == [
        True, False, True, False]
    # with bars stuck to the window (52 px at the top, 57 at the foot of a window of 844), only the free band between
    # them counts: a box under the foot bar or under the top bar shows nothing; one pixel inside the band shows
    band = '844, 52, 57'
    assert run_js("[s.outOfView({{top: 800, bottom: 1300}}, {0}), s.outOfView({{top: 787, bottom: 1300}}, {0}), "
                  "s.outOfView({{top: 786, bottom: 1300}}, {0}), s.outOfView({{top: -400, bottom: 40}}, {0}), "
                  "s.outOfView({{top: -400, bottom: 52}}, {0}), s.outOfView({{top: -400, bottom: 53}}, {0}), "
                  "s.outOfView({{top: 720, bottom: 1286}}, {0})]".format(band)) == [
        True, True, False, True, True, False, False]
    # no bars at all: the whole window, as without the two arguments
    assert run_js("[s.outOfView({top: 800, bottom: 1300}, 844, 0, 0), s.outOfView({top: 844, bottom: 1300}, 844, 0, 0), "
                  "s.outOfView({top: -400, bottom: 1}, 844, 0, 0), s.outOfView({top: -400, bottom: 0}, 844, 0, 0)]") == [
        False, True, False, True]


def test_the_free_band_is_what_the_bars_stuck_to_the_window_leave():
    # the top bars cover down to the lowest of their bottoms (the freeze banner and the header under it), the foot bars
    # up from the highest of their tops; no bars, no cover; a bar beyond the window covers no more than the window
    assert run_js("s.coveredBand([{top: 0, bottom: 48}, {top: 46, bottom: 103}], [{top: 787, bottom: 844}], 844)") == {
        'top': 103, 'foot': 57}
    assert run_js("s.coveredBand([], [], 844)") == {'top': 0, 'foot': 0}
    assert run_js("s.coveredBand([{top: 0, bottom: 52}], [], 640)") == {'top': 52, 'foot': 0}
    assert run_js("s.coveredBand([{top: -10, bottom: 900}], [{top: -5, bottom: 700}], 640)") == {'top': 640, 'foot': 640}
