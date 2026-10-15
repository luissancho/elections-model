"""Puntos del sparkline de las tarjetas de partido (`web/dist/js/sparkline.js`), comprobados con `node` (se salta
sin él)."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SPARKLINE_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'sparkline.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    """Evalúa `expr` con el módulo importado como `m` y devuelve su JSON."""
    script = 'const m = await import("{}"); console.log(JSON.stringify({})); process.exit(0);'.format(SPARKLINE_JS, expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(out.stdout)


def test_sparkline_points_scale_the_values_and_skip_the_gaps():
    assert run_js('m.sparklinePoints([1, null, 3, 2], 180, 28)') == [[0, 25], [120, 3], [180, 14]]
    assert run_js('m.sparklinePoints([5, 5], 180, 28)') == [[0, 14], [180, 14]]
    assert run_js('m.sparklinePoints([], 180, 28)') == []
