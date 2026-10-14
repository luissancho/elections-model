"""Paridad entre `web/dist/js/seats.js` y la regla de cuantiles de `Stat` (se salta sin `node`)."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from mtpy.core.utils.stat import Stat

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SEATS_JS = os.path.join(ROOT, 'web', 'dist', 'js', 'seats.js')
DIST = {'n_seats': 350, 'parties': ['PP', 'PSOE', 'VOX'],
        'seats': [[140, 106, 65], [137, 110, 62], [145, 100, 70], [133, 112, 61], [150, 95, 72], [139, 108, 60], [141, 104, 66], [136, 109, 63]]}

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def run_js(expr):
    script = 'import * as s from "file://{}"; const dist = {}; console.log(JSON.stringify({}));'.format(SEATS_JS, json.dumps(DIST), expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_summarize_matches_stat_quantiles():
    values = np.array([row[0] + row[2] for row in DIST['seats']], dtype=float)
    stat = Stat(values, dropna=True)
    got = run_js('s.summarize(s.coalitionSeats(dist, ["PP", "VOX"]), 176)')
    assert got == {'n': 8, 'median': stat.median(), 'lo': stat.quantile(0.025), 'hi': stat.quantile(0.975),
                   'min': 194.0, 'max': 222.0, 'pMajority': 1.0}


def test_column_and_coalition_ignore_unknown_parties():
    assert run_js('s.column(dist, "PSOE")') == [row[1] for row in DIST['seats']]
    assert run_js('s.column(dist, "ERC")') == []
    assert run_js('s.coalitionSeats(dist, ["ERC"])') == []
    assert run_js('s.coalitionSeats(dist, ["PP", "ERC"])') == [row[0] for row in DIST['seats']]
    assert run_js('s.summarize([], 176)') is None


def test_histogram_bins_cover_the_range_without_gaps():
    assert run_js('s.binWidth([0, 0, 1], 40)') == 1 and run_js('s.binWidth([100, 180], 40)') == 3
    assert run_js('s.histogram([3, 4, 4, 9], 2)') == {'starts': [2, 4, 6, 8], 'counts': [1, 2, 0, 1]}
    assert run_js('s.histogram([0, 0, 0], 1)') == {'starts': [0], 'counts': [3]}


def test_coalition_hash_round_trip():
    assert run_js('s.parseCoalition("#coalition=VOX,PP,ERC", dist.parties)') == ['PP', 'VOX']
    assert run_js('s.parseCoalition("#other=1", dist.parties)') is None
    assert run_js('s.parseCoalition("#coalition=", dist.parties)') == []
    assert run_js('s.coalitionHash(["PP", "VOX"])') == '#coalition=PP,VOX' and run_js('s.coalitionHash([])') == ''
