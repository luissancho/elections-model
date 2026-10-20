"""La imagen descargable de un gráfico (`web/dist/js/share.js`): la composición del título, el subtítulo, el gráfico
y la firma en un lienzo de tamaño fijo, comprobada con `node` (se salta sin él). El módulo importa el tema de
`charts/base.js`, que lee `window.echarts` al cargarse, así que el arnés carga el ECharts empaquetado antes."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIST = os.path.join(ROOT, 'web', 'dist')
ECHARTS = os.path.join(DIST, 'vendor', 'echarts-5.6.0.min.js')
SHARE_JS = pathlib.Path(DIST, 'js', 'share.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')

TEXTS = {'title': 'Elecciones generales · 29 nov 2026', 'subtitle': 'Hemiciclo · Estimación a 19 oct 2026',
         'sign': 'demoscopio.es · @luis_sancho'}


def run_js(expr):
    """Evalúa `expr` con `s` (el módulo) definido y devuelve su JSON."""
    script = (
        'import {{createRequire}} from "node:module"; '
        'const echarts = createRequire(import.meta.url)({}); globalThis.window = {{echarts}}; '
        'const s = await import("{}"); console.log(JSON.stringify({})); process.exit(0);'
    ).format(json.dumps(ECHARTS), SHARE_JS, expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout)


def test_image_layout_stacks_the_title_the_subtitle_the_chart_and_the_signature():
    """Un lienzo de 800 px a escala 2: el título arriba a la izquierda en la fuente de los titulares, el subtítulo
    apagado debajo, el gráfico entre los mismos márgenes que los textos y la firma pequeña y apagada abajo a la
    derecha."""
    layout = run_js('s.imageLayout({})'.format(json.dumps(TEXTS)))
    assert layout['width'] == 800 and layout['scale'] == 2
    texts = {text['role']: text for text in layout['texts']}
    assert [text['role'] for text in layout['texts']] == ['title', 'subtitle', 'sign']
    assert texts['title'] == {'role': 'title', 'text': TEXTS['title'], 'x': 28, 'y': 28, 'align': 'left',
                              'font': "700 22px 'Outfit', 'DM Sans', sans-serif", 'color': '#1a1c22'}
    assert texts['subtitle'] == {'role': 'subtitle', 'text': TEXTS['subtitle'], 'x': 28, 'y': 56, 'align': 'left',
                                 'font': "500 14px 'DM Sans', system-ui, sans-serif", 'color': '#5a5f72'}
    assert layout['chart'] == {'x': 28, 'y': 88, 'width': 744, 'height': 440}
    assert texts['sign'] == {'role': 'sign', 'text': TEXTS['sign'], 'x': 772, 'y': 536, 'align': 'right',
                             'font': "600 12px 'DM Sans', system-ui, sans-serif", 'color': '#5a5f72'}
    assert layout['height'] == 572


def test_image_layout_skips_the_missing_texts():
    """Sin subtítulo el gráfico sube; sin firma la imagen acaba bajo el gráfico."""
    layout = run_js('s.imageLayout({title: "Elecciones generales", sign: "demoscopio.es"})')
    assert [text['role'] for text in layout['texts']] == ['title', 'sign']
    assert layout['chart']['y'] == 68 and layout['height'] == 552
    bare = run_js('s.imageLayout({})')
    assert bare['texts'] == [] and bare['chart'] == {'x': 28, 'y': 0, 'width': 744, 'height': 440} and bare['height'] == 440


def test_image_layout_takes_its_size_from_the_options():
    layout = run_js('s.imageLayout({title: "T", width: 600, chartHeight: 330, scale: 3})')
    assert layout['width'] == 600 and layout['scale'] == 3 and layout['chart'] == {'x': 28, 'y': 68, 'width': 544, 'height': 330}
    assert layout['height'] == 68 + 330
