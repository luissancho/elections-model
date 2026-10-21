"""Los degradados de `web/dist/js/format.js` (`shade`, `shadeAround`) siguen a los filtros de
`mtpy/lib/pages/filters.py` (`shade`, `shade_around`), que pintan las mismas celdas en el servidor; comprobados con
`node` (se salta sin él)."""
import os
import pathlib
import re
import shutil
import subprocess

import pytest

from mtpy.lib import pages

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FORMAT_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'format.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    script = 'import * as f from "{}"; console.log(JSON.stringify({}));'.format(FORMAT_JS, expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return out.stdout.strip().strip('"')


def rgba(text):
    """Los cuatro números de un `rgba()`, para comparar sin depender de cómo se escribe el cero ('0.0' en Python, '0'
    en JavaScript); cualquier otro texto, tal cual."""
    return [float(x) for x in re.findall(r'[\d.]+', text)] if text.startswith('rgba') else text


def test_shade_follows_the_python_filter():
    for x, start, end, color in ((2, 0, 4, '#d62728'), (9, 0, 4, '#d62728'), (0.5, 1, 0, '#d62728'), (0, 0, 4, '#d62728'),
                                 (-3, 0, -8, '#d62728'), (1.14, 0, 8, '#2ca02c'), (0.37, 1, 0, '#7f7f7f')):
        assert rgba(run_js(f'f.shade({x}, {start}, {end}, "{color}")')) == rgba(pages.shade(x, start, end, color)), (x, start, end)
    assert run_js('f.shade(1.5, 1, 2, "#7f7f7f", 0.4)') == pages.shade(1.5, 1, 2, '#7f7f7f', alpha=0.4) == 'rgba(127, 127, 127, 0.2)'
    for expr in ('f.shade(null, 0, 4, "#d62728")', 'f.shade(undefined, 0, 4, "#d62728")', 'f.shade(NaN, 0, 4, "#d62728")',
                 'f.shade(1, 2, 2, "#d62728")', 'f.shade(1, 0, 4, "red")', 'f.shade(1, 0, 4, "#abc")'):
        assert run_js(expr) == 'transparent', expr


def test_shade_around_follows_the_python_filter():
    for x in (-4, -20, 0, 0.5, 4, 1.14, -1.14, 20):
        js = run_js(f'f.shadeAround({x}, 0, 8, "#d62728", "#2ca02c")')
        assert rgba(js) == rgba(pages.shade_around(x, 0, 8, '#d62728', '#2ca02c')), x
    assert run_js('f.shadeAround(0.5, 1, 1, "#d62728", "#7f7f7f")') == pages.shade_around(0.5, 1, 1, '#d62728', '#7f7f7f')
    assert run_js('f.shadeAround(1.5, 1, 1, "#d62728", "#7f7f7f")') == pages.shade_around(1.5, 1, 1, '#d62728', '#7f7f7f')
    for expr in ('f.shadeAround(null, 0, 8, "#d62728", "#2ca02c")', 'f.shadeAround(3, 0, 0, "#d62728", "#2ca02c")',
                 'f.shadeAround(3, 0, -1, "#d62728", "#2ca02c")', 'f.shadeAround(NaN, 0, 8, "#d62728", "#2ca02c")'):
        assert run_js(expr) == 'transparent', expr


def test_fmt_delta_follows_the_badge_filter():
    """`fmtDelta` escribe la flecha y la magnitud de una diferencia, el texto de `delta_badge` en Python."""
    for x, digits in ((6, 0), (-1, 0), (0, 0), (0.04, 1), (-12.38, 1), (1.0, 1), (-0.05, 1), (-25, 0)):
        assert run_js(f'f.fmtDelta({x}, {digits})') == pages.delta_badge(x, digits)['text'], (x, digits)
    assert run_js('f.fmtDelta(null)') == run_js('f.fmtDelta(undefined)') == run_js('f.fmtDelta(NaN)') == '–'


def test_gradient_follows_the_python_filter():
    """`gradient` mezcla los colores de una escala de varios tramos como el filtro `gradient` de Python (el badge del
    rating de «Todas las encuestas»)."""
    stops = '["#d62728", "#7f7f7f", "#2ca02c"]'
    for x in (0, 5, 25, 50, 73.08, 75, 100, -10, 140, 0.9, 49.999):
        assert run_js(f'f.gradient({x}, 0, 100, {stops})') == pages.gradient(x, 0, 100, pages.RATING_COLORS), x
    assert run_js(f'f.gradient(75, 100, 0, {stops})') == pages.gradient(75, 100, 0, pages.RATING_COLORS) == '#ab5354'
    assert run_js('f.gradient(0.5, 0, 1, ["#000000", "#ffffff"])') == '#808080'
    for expr in (f'f.gradient(null, 0, 100, {stops})', f'f.gradient(undefined, 0, 100, {stops})',
                 f'f.gradient(NaN, 0, 100, {stops})', f'f.gradient(3, 1, 1, {stops})',
                 'f.gradient(3, 0, 100, ["#d62728"])', 'f.gradient(3, 0, 100, ["red", "#7f7f7f"])',
                 'f.gradient(3, 0, 100, [])', 'f.gradient(3, 0, 100, "#d62728")'):
        assert run_js(expr) == 'transparent', expr


def test_axis_date_names_each_tick_by_what_it_is():
    """Eje temporal de los promedios: el 1 de enero es el año (en negrita, `{year|…}`), el día 1 de otro mes es el
    mes y cualquier otro día, día y mes; la primera marca redonda (índice 1: ECharts da el 0 al extremo izquierdo,
    que no dibuja) lleva siempre el año, para que una ventana corta no se quede sin él."""
    t = lambda iso: f'new Date("{iso}T00:00:00").getTime()'  # noqa: E731
    assert run_js(f'f.fmtAxisDate({t("2025-01-01")}, 3)') == '{year|2025}'
    assert run_js(f'f.fmtAxisDate({t("2025-02-01")}, 3)') == 'feb'
    assert run_js(f'f.fmtAxisDate({t("2025-10-04")}, 3)') == '4 oct'
    assert run_js(f'f.fmtAxisDate({t("2026-05-01")}, 1)') == 'may {year|2026}'
    assert run_js(f'f.fmtAxisDate({t("2026-07-17")}, 1)') == '17 jul {year|2026}'
    assert run_js(f'f.fmtAxisDate({t("2026-07-05")}, 0)') == '5 jul {year|2026}'
    assert run_js(f'f.fmtAxisDate({t("2026-01-01")}, 1)') == '{year|2026}'
    assert run_js(f'f.fmtAxisDate({t("2026-08-01")}, 2)') == 'ago'


def test_fmt_count_marks_the_thousands_as_the_python_filter():
    """`fmtCount` agrupa también los millares de cuatro cifras, como `fmt_num`; `fmtInt` no (para que un año no
    salga con punto)."""
    for x in (1000, 12000, 999, 0, 1234567, -2500):
        assert run_js(f'f.fmtCount({x})') == pages.fmt_num(x), x
    assert run_js('f.fmtInt(2026)') == '2026' and run_js('f.fmtCount(null)') == '–'
