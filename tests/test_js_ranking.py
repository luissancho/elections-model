"""Visibilidad de las filas del ranking de /casas (`web/dist/js/ranking.js`), comprobada con `node` (se salta sin él)."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RANKING_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'ranking.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def visibility(generals, only_generals, shown):
    script = 'import * as r from "{}"; console.log(JSON.stringify(r.rankingVisibility({}, {}, {})));'.format(
        RANKING_JS, json.dumps(generals), json.dumps(only_generals), shown)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_ranking_visibility_without_filter_shows_the_first_rows():
    result = visibility([0, 2, 0, 1], False, 2)
    assert result == {'visible': [True, True, False, False], 'filtered': [False] * 4, 'more': True}


def test_ranking_visibility_with_filter_skips_rows_without_generals():
    result = visibility([0, 2, 0, 1, 3], True, 2)
    assert result['filtered'] == [True, False, True, False, False]
    assert result['visible'] == [False, True, False, True, False] and result['more'] is True


def test_ranking_visibility_with_filter_and_nothing_left():
    assert visibility([0, 1], True, 50)['more'] is False
