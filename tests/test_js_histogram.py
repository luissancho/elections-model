"""Opciones de `renderHistogram` (`web/dist/js/charts/histogram.js`) con un ECharts simulado en `node` (se salta
sin él): por defecto, el % de simulaciones en escaños; con `share: false`, el recuento de sondeos en puntos."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HISTOGRAM_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'charts', 'histogram.js').as_uri()
STUB = ('let last = null; globalThis.window = {matchMedia: () => ({matches: false}), echarts: {'
        'format: {encodeHTML: (s) => s}, init: () => ({setOption: (o) => { last = o; }, resize() {}})}};'
        'globalThis.echarts = window.echarts; globalThis.ResizeObserver = class { observe() {} };')

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(call):
    """Draw with `call` (an expression on `h`) and return the y data, y tick labels, first tooltip and marks."""
    script = STUB + ('const h = await import("{}"); {};'
              'const y = last.yAxis; const tip = last.tooltip.formatter([{{dataIndex: 0}}]);'
              'console.log(JSON.stringify({{data: last.series[0].data, ticks: [0, 1, 2].map(y.axisLabel.formatter),'
              'minInterval: y.minInterval ?? null, tip, color: last.series[0].color,'
              'marks: last.series[0].markLine.data.map((d) => d.label.formatter),'
              'lines: last.series[0].markLine.data.map((d) => [d.lineStyle.width ?? null, d.lineStyle.type, d.label.show ?? null])}}));').format(HISTOGRAM_JS, call)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_default_draws_the_share_of_simulations_in_seats():
    out = run_js('h.renderHistogram({}, [170, 171, 171, 175], {median: 171, lo: 170, hi: 175, width: 1})')
    assert out['data'][:2] == [25.0, 50.0] and out['ticks'] == ['0%', '1%', '2%'] and out['minInterval'] is None
    assert out['tip'] == '<strong>170 escaños: 25,0%</strong><br>Mediana: 171 · rango 95%: 170–175'
    assert out['marks'] == ['171'] and out['color'] == '#3a6ea5'  # THEME.barColor, the neutral blue


def test_counts_in_points_with_the_mean_marked():
    out = run_js('h.renderHistogram({}, [0.4, 2.3, 2.9, 5.1], {width: 1, unit: "puntos", share: false,'
                 'label: (s) => `${s} a ${s + 1} puntos`, median: 2.675, markLabel: "Media: 2,7",'
                 'countLabel: (n) => `${n} ${n === 1 ? "sondeo" : "sondeos"}`})')
    assert out['data'] == [1, 0, 2, 0, 0, 1] and out['ticks'] == ['0', '1', '2'] and out['minInterval'] == 1
    assert out['tip'] == '<strong>0 a 1 puntos: 1 sondeo</strong><br>Media: 2,7'
    assert out['marks'] == ['Media: 2,7']


def test_bounds_draw_thin_lines_at_the_interval_and_a_thick_one_at_the_median_without_labels():
    out = run_js('h.renderHistogram({}, [120, 130, 135, 135, 140, 150], {width: 1, bounds: true, lo: 120, hi: 150,'
                 'median: 135, markLabels: false})')
    assert out['lines'] == [[1, 'solid', False], [3, 'solid', False], [1, 'solid', False]]
    assert out['marks'] == [None, None, None]


def test_default_marks_keep_their_labels_and_have_no_bounds():
    out = run_js('h.renderHistogram({}, [170, 171, 171, 175], {median: 171, lo: 170, hi: 175, width: 1})')
    assert out['marks'] == ['171'] and out['lines'] == [[None, 'solid', None]]


SERIES_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'charts', 'series.js').as_uri()


@pytest.mark.parametrize('legend, show, selected_mode', [(False, False, None), (True, True, None), ('readonly', True, False)])
def test_series_legend_is_hidden_selectable_or_read_only(legend, show, selected_mode):
    script = STUB + ('const s = await import("{}"); s.renderSeries({{}}, {{series: {{dates: ["2026-10-01"], mean: {{A: [30]}}, lo: {{A: [28]}}, hi: {{A: [32]}}}},'
                     'polls: {{polls: []}}, projection: null, parties: ["A"], asOf: "2026-10-01", when: "2026-10-01",'
                     'legend: {}}}); console.log(JSON.stringify(last.legend));').format(SERIES_JS, json.dumps(legend))
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    option = json.loads(out.stdout)
    assert option['show'] is show and option.get('selectedMode') == selected_mode and option['selected'] == {'A': True}
