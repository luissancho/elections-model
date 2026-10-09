"""Ventana inicial del gráfico del promedio (`web/dist/js/range.js`), comprobada con `node` (se salta sin él)."""
import json
import os
import pathlib
import shutil
import subprocess
from datetime import date, timedelta

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RANGE_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'range.js').as_uri()
DATES = ['2026-01-01', '2026-03-01', '2026-06-01', '2026-10-13']

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    script = 'import * as r from "{}"; const dates = {}; console.log(JSON.stringify({}));'.format(
        RANGE_JS, json.dumps(DATES), expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def days_before(iso, days):
    return (date.fromisoformat(iso) - timedelta(days=days)).isoformat()


def test_last_window_counts_back_from_the_anchor_and_clamps_to_the_first_date():
    assert run_js('r.lastWindow(dates, 180, "2026-10-03")') == {'startValue': days_before('2026-10-03', 180), 'endValue': '2026-10-13'}
    assert run_js('r.lastWindow(dates, 180)') == {'startValue': days_before('2026-10-13', 180), 'endValue': '2026-10-13'}
    assert run_js('r.lastWindow(dates, 400, "2026-10-03")') == {'startValue': '2026-01-01', 'endValue': '2026-10-13'}
    assert run_js('r.lastWindow([], 180)') is None
    # un ancla posterior al eje no invierte la ventana: el inicio se limita a la última fecha
    assert run_js('r.lastWindow(["2026-01-01", "2026-02-01"], 10, "2026-06-01")') == {'startValue': '2026-02-01', 'endValue': '2026-02-01'}


def test_series_window_maps_the_control_choices():
    assert run_js('r.seriesWindow("6m", dates, "2026-10-03")') == {'startValue': days_before('2026-10-03', 180), 'endValue': '2026-10-13'}
    assert run_js('r.seriesWindow("all", dates, "2026-10-03")') == {'startValue': '2026-01-01', 'endValue': '2026-10-13'}
    assert run_js('r.seriesWindow("nope", dates, "2026-10-03")') == {'startValue': '2026-01-01', 'endValue': '2026-10-13'}
    assert run_js('r.seriesWindow("6m", [], null)') is None
    # una clave heredada de Object no es una opción
    assert run_js('r.seriesWindow("toString", dates, "2026-10-03")') == {'startValue': '2026-01-01', 'endValue': '2026-10-13'}
    assert run_js('r.DEFAULT_WINDOW') == '6m'
