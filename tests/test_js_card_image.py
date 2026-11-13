"""La imagen descargable de una tarjeta entera (`web/dist/js/card-image.js`): lo que se quita de la copia, el tamaño
del lienzo y las opciones de los gráficos de una tarjeta, comprobado con `node` (se salta sin él). El módulo importa
`share.js`, que lee `window.echarts` al cargarse, así que el arnés carga el ECharts empaquetado antes."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIST = os.path.join(ROOT, 'web', 'dist')
ECHARTS = os.path.join(DIST, 'vendor', 'echarts-5.6.0.min.js')
MODULE = pathlib.Path(DIST, 'js', 'card-image.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    """Evalúa `expr` con `c` (el módulo) definido y devuelve su JSON."""
    script = (
        'import {{createRequire}} from "node:module"; '
        'const echarts = createRequire(import.meta.url)({}); globalThis.window = {{echarts}}; '
        'const c = await import("{}"); console.log(JSON.stringify({})); process.exit(0);'
    ).format(json.dumps(ECHARTS), MODULE, expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT)

    return json.loads(out.stdout.strip().splitlines()[-1])


def test_card_image_layout_and_stripped_parts():
    """La tarjeta se dibuja a 1200 px de ancho y al doble de densidad desde la librería vendorizada; de la copia salen
    los botones de la tarjeta, los «Ver más» y el «Cómo leerlo» plegado."""
    assert run_js('c.CARD_IMAGE') == {'width': 1200, 'scale': 2, 'vendor': '/dist/vendor/html2canvas-1.4.1.min.js'}
    assert run_js('c.STRIPPED') == '.card-img-btn, .tool-btn, button.more, details.how'


def test_chart_options_take_the_mounted_charts_with_an_id():
    """`chartOptions` devuelve el id y la opción de cada contenedor `.chart` con un gráfico montado y con id; omite los
    contenedores sin gráfico o sin id (no se podrían volver a dibujar ni encontrar en la copia)."""
    card = ('{querySelectorAll: () => [{id: "a", __chart: {getOption: () => ({series: [1]})}}, '
            '{id: "", __chart: {getOption: () => ({})}}, {id: "b"}, {id: "c", __chart: {getOption: () => ({x: 2})}}]}')
    assert run_js('c.chartOptions(%s)' % card) == [{'id': 'a', 'option': {'series': [1]}, 'imageOption': None},
                                                       {'id': 'c', 'option': {'x': 2}, 'imageOption': None}]


def test_chart_options_carry_the_image_option_of_a_chart_and_image_option_of_picks_it():
    """El hemiciclo de la tarjeta de un sondeo se redibuja al tamaño de la imagen con `el.__imageOption`; los gráficos sin
    ella siguen con la opción de la pantalla."""
    card = ('{querySelectorAll: () => [{id: "a", __chart: {getOption: () => ({series: [1]})}, __imageOption: (w, h) => ({w, h})}, '
            '{id: "b", __chart: {getOption: () => ({x: 2})}}]}')
    assert run_js('c.chartOptions(%s).map((chart) => [chart.id, typeof chart.imageOption])' % card) == [['a', 'function'], ['b', 'object']]
    assert run_js('c.chartOptions(%s)[1].imageOption' % card) is None
    assert run_js('c.imageOptionOf({option: {a: 1}, imageOption: (w, h) => ({w, h})}, 500, 300)') == {'w': 500, 'h': 300}
    assert run_js('c.imageOptionOf({option: {a: 1}, imageOption: null}, 500, 300)') == {'a': 1}


def test_wire_card_images_looks_for_the_square_button_or_for_the_selector_given():
    """Una tarjeta puede descargar su imagen con otro botón (el «Descargar» de la cabecera): `wireCardImages` busca
    el cuadrado por defecto y el selector que se le pase."""
    wire = ('(() => { const seen = []; const root = {querySelectorAll: (sel) => { seen.push(sel); return []; }}; '
            'c.wireCardImages(root); c.wireCardImages(root, null, "#changes-download"); return seen; })()')
    assert run_js(wire) == ['button.card-img-btn', '#changes-download']


def test_fit_overflow_widens_the_card_by_what_its_marked_scrollers_hide():
    """Una tabla más ancha que la tarjeta (`data-shot-fit`) ensancha la copia lo que desborda; las ocultas (ancho 0) y
    las que caben no cuentan."""
    card = ('{style: {}, getBoundingClientRect: () => ({width: 1136}), querySelectorAll: (sel) => '
            'sel === "[data-shot-fit]" ? %s : []}')
    fit = '(() => { const card = %s; return [c.fitOverflow(card), card.style.width || null]; })()'
    wide = card % '[{scrollWidth: 1500, clientWidth: 1100}, {scrollWidth: 0, clientWidth: 0}]'
    assert run_js(fit % wide) == [400, '1536px']
    assert run_js(fit % (card % '[{scrollWidth: 900, clientWidth: 1100}]')) == [0, None]


def test_free_fluid_svgs_drops_the_screen_size_html2canvas_writes_on_the_marked_svgs():
    """html2canvas escribe en la copia el estilo calculado de cada SVG de la página: un SVG que se dimensiona por CSS
    (`data-shot-fluid`, el hemiciclo de «Escaños que bailan») se quedaría con el tamaño de la pantalla (294 px en un
    móvil). `freeFluidSvgs` le quita el ancho y el alto escritos y deja el resto de su estilo."""
    free = ('(() => { const removed = []; const svg = {style: {removeProperty: (p) => removed.push(p)}}; '
            'const card = {querySelectorAll: (sel) => (sel === "[data-shot-fluid]" ? [svg] : [])}; '
            'return [c.freeFluidSvgs(card), removed]; })()')
    assert run_js(free) == [1, ['width', 'height', 'inline-size', 'block-size']]
    assert run_js('c.freeFluidSvgs({querySelectorAll: () => []})') == 0
