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


SIDE = {'width': 300, 'height': 520, 'items': []}


def test_image_layout_leaves_room_for_a_side_panel_and_centres_the_shorter():
    """Con un panel a la derecha (la tabla del hemiciclo, Luis 2026-10-21) el gráfico se estrecha, el panel se
    pega al margen derecho y el más bajo de los dos se centra en vertical frente al más alto."""
    layout = run_js('s.imageLayout({title: "T", subtitle: "S", width: 1200, chartHeight: 360, side: %s})' % json.dumps(SIDE))
    assert layout['chart'] == {'x': 28, 'y': 88 + 80, 'width': 1200 - 56 - 300 - 32, 'height': 360}
    assert layout['side'] == {'x': 1200 - 28 - 300, 'y': 88, 'width': 300, 'height': 520, 'items': []}
    assert layout['height'] == 88 + 520
    short = run_js('s.imageLayout({width: 1200, chartHeight: 360, side: %s})' % json.dumps({**SIDE, 'height': 160}))
    assert short['chart']['y'] == 0 and short['side']['y'] == 100 and short['height'] == 360
    assert run_js('s.imageLayout({}).side') is None


TABLE = {
    'columns': [{'label': 'Partido', 'align': 'left'}, {'label': '% Voto', 'align': 'right'},
                {'label': 'vs 2023', 'align': 'right'}, {'label': 'Rango 95%', 'align': 'left'}],
    'rows': [
        [{'text': 'PP', 'bold': True, 'dot': '#1d84ce'}, {'text': '32,7%', 'bold': False},
         {'text': '▼ 0,4', 'bold': True, 'color': 'rgb(214, 39, 40)'},
         {'text': '118–160', 'range': {'left': 30, 'width': 10, 'mark': 40, 'line': 50, 'color': '#1d84ce'}}],
        [{'text': 'UPN', 'bold': True, 'dot': '#0a5ca8'}, {'text': '0,2%', 'bold': False}, {'text': '–', 'bold': True},
         {'text': '0–2', 'range': {'left': 0, 'width': 1, 'mark': 0, 'line': None, 'color': '#0a5ca8'}}],
    ],
}


def table_drawing(table=TABLE):
    """`tableDrawing` con una medida falsa: 7 px por carácter, sea cual sea la fuente."""
    return run_js('s.tableDrawing(%s, (text) => 7 * [...text].length)' % json.dumps(table))


def test_table_drawing_sizes_the_columns_to_their_widest_cell_and_draws_the_site_rows():
    drawing = table_drawing()
    # Partido: dot 9 + gap 7 + 3 chars; % Voto: 6 chars of the header; vs 2023: 7; Rango: bar 110 + 8 + 7 chars
    widths = [16 + max(7 * 7, 16 + 21), 16 + 42, 16 + 49, 16 + 110 + 8 + 49]
    assert drawing['width'] == sum(widths) and drawing['height'] == 30 + 2 * 30
    texts = [item for item in drawing['items'] if item['type'] == 'text']
    heads = [t for t in texts if t['y'] == 15]
    assert [t['text'] for t in heads] == ['PARTIDO', '% VOTO', 'VS 2023', 'RANGO 95%']
    assert heads[0]['align'] == 'left' and heads[1]['align'] == 'right' and heads[1]['x'] == widths[0] + widths[1] - 8
    assert heads[3]['align'] == 'left' and heads[3]['x'] == sum(widths[:3]) + 8  # the range column is aligned left
    assert all(t['baseline'] == 'middle' for t in texts)
    pp = {t['text']: t for t in texts if t['y'] == 45}
    assert pp['PP']['x'] == 8 + 9 + 7 and pp['PP']['font'].startswith('700 13px')
    assert pp['32,7%']['font'].startswith('400 13px') and pp['32,7%']['color'] == '#1a1c22'
    assert pp['▼ 0,4']['color'] == 'rgb(214, 39, 40)'
    assert pp['118–160']['align'] == 'left' and pp['118–160']['x'] == sum(widths[:3]) + 8 + 110 + 8  # after its bar
    circles = [item for item in drawing['items'] if item['type'] == 'circle']
    assert circles == [{'type': 'circle', 'x': 12.5, 'y': 45, 'r': 4.5, 'color': '#1d84ce'},
                       {'type': 'circle', 'x': 12.5, 'y': 75, 'r': 4.5, 'color': '#0a5ca8'}]
    lines = [item for item in drawing['items'] if item['type'] == 'line' and item['dash'] is None]
    assert [(line['y1'], line['width'], line['color']) for line in lines] == [
        (30, 1.5, 'rgba(0,0,0,.15)'), (60, 1, 'rgba(0,0,0,.08)'), (90, 1, 'rgba(0,0,0,.08)')]


def test_table_drawing_lines_up_the_range_bars_and_draws_the_band_the_mark_and_the_majority():
    drawing = table_drawing()
    start = drawing['width'] - (16 + 110 + 8 + 49) + 8
    rects = [item for item in drawing['items'] if item['type'] == 'rect']
    tracks = [r for r in rects if r['color'] == '#e3e6ec']
    assert [(r['x'], r['y'], r['width'], r['height']) for r in tracks] == [(start, 39, 110, 12), (start, 69, 110, 12)]
    band, mark = [r for r in rects if r['color'] == '#1d84ce']
    assert (band['x'], band['width'], band['opacity']) == (start + 33, 11, 0.45)
    assert (mark['x'], mark['y'], mark['width'], mark['height'], mark['opacity']) == (start + 44 - 1, 37, 3, 16, 1)
    majority = [item for item in drawing['items'] if item['type'] == 'line' and item['dash']]
    assert len(majority) == 1 and majority[0]['x1'] == start + 55 and majority[0]['color'] == '#9298ac'


def test_file_slug_drops_accents_and_joins_words_with_dashes():
    assert run_js('["Sigma Dos / El Mundo", "40dB", "Céleste-Tel", "  CIS  "].map(s.fileSlug)') == [
        'sigma-dos-el-mundo', '40db', 'celeste-tel', 'cis']


def test_image_layout_stacks_several_charts_with_a_gap():
    """La imagen de la calculadora de mayorías (Luis, 2026-10-26) lleva dos gráficos, la barra y el histograma,
    uno bajo otro con el mismo hueco que separa los textos; `chart` sigue siendo el primero."""
    layout = run_js('s.imageLayout({title: "T", subtitle: "S", sign: "F", charts: [{height: 120}, {height: 300}]})')
    assert layout['charts'] == [{'x': 28, 'y': 88, 'width': 744, 'height': 120},
                                {'x': 28, 'y': 88 + 120 + 12, 'width': 744, 'height': 300}]
    assert layout['chart'] == layout['charts'][0]
    assert layout['height'] == 88 + 120 + 12 + 300 + 8 + 16 + 20
    single = run_js('s.imageLayout({chartHeight: 330})')
    assert single['charts'] == [single['chart']] == [{'x': 28, 'y': 0, 'width': 744, 'height': 330}]


def test_image_layout_draws_a_note_chip_under_the_subtitle():
    """El texto de la calculadora («2 partidos · 200 escaños · P(mayoría absoluta): 98,8 %») va como en la página:
    un chip de fondo suave bajo el subtítulo, verde cuando la mayoría es probable; su anchura la mide quien llama."""
    layout = run_js('s.imageLayout({title: "T", subtitle: "S", chartHeight: 120, '
                    'note: {text: "2 partidos · 200 escaños", width: 180, ok: false}})')
    assert layout['note'] == {'x': 28, 'y': 88, 'width': 180 + 22, 'height': 30, 'r': 10, 'text': '2 partidos · 200 escaños',
                              'font': "600 13px 'DM Sans', system-ui, sans-serif", 'color': '#1a1c22', 'background': '#eceef2'}
    assert layout['chart']['y'] == 88 + 30 + 12 and layout['height'] == 130 + 120
    ok = run_js('s.imageLayout({chartHeight: 120, note: {text: "ok", width: 20, ok: true}})')
    assert ok['note']['color'] == '#15803d' and ok['note']['background'] == '#ecfdf3' and ok['note']['y'] == 28
    assert run_js('s.imageLayout({}).note') is None


def test_image_layout_puts_a_bottom_panel_under_the_charts_centred():
    """La tabla de «Reparto de escaños» (Luis, 2026-10-26) va bajo la barra, centrada; si es más ancha que la imagen,
    la imagen se ensancha hasta darle sitio."""
    layout = run_js('s.imageLayout({title: "T", sign: "F", chartHeight: 120, bottom: {width: 600, height: 390, items: []}})')
    assert layout['chart'] == {'x': 28, 'y': 68, 'width': 744, 'height': 120}
    assert layout['bottom'] == {'x': 100, 'y': 68 + 120 + 12, 'width': 600, 'height': 390, 'items': []}
    assert layout['height'] == 200 + 390 + 8 + 16 + 20
    wide = run_js('s.imageLayout({chartHeight: 120, bottom: {width: 900, height: 60, items: []}})')
    assert wide['width'] == 956 and wide['chart']['width'] == 900 and wide['bottom']['x'] == 28
    assert run_js('s.imageLayout({}).bottom') is None
