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
    assert run_js('c.chartOptions(%s)' % card) == [{'id': 'a', 'option': {'series': [1]}}, {'id': 'c', 'option': {'x': 2}}]
