"""Funciones puras de "Resultados por territorio" (`web/dist/js/territory.js`), comprobadas con `node` (se salta sin él)."""
import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TERRITORY_JS = pathlib.Path(ROOT, 'web', 'dist', 'js', 'territory.js').as_uri()

pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node no disponible')


def _row(region_id, region, name, pct, lo, hi):
    return {'region_id': region_id, 'region': region, 'name': name, 'pct': pct, 'pct_lo': None, 'pct_hi': None,
            'seats': None, 'seats_mean': None, 'seats_lo': lo, 'seats_hi': hi, 'p_seats': None}


REGIONS = [
    {'id': 28, 'name': 'Madrid', 'seats': 6, 'votes': 600, 'community': {'code': 13, 'name': 'Madrid'}},
    {'id': 8, 'name': 'Barcelona', 'seats': 4, 'votes': 400, 'community': {'code': 9, 'name': 'Cataluña'}},
    {'id': 17, 'name': 'Girona', 'seats': 2, 'votes': 200, 'community': {'code': 9, 'name': 'Cataluña'}},
]
ROWS = [
    _row(28, 'Madrid', 'PP', 40, 2, 4), _row(8, 'Barcelona', 'PP', 30, 1, 3), _row(17, 'Girona', 'PP', 20, 1, 3),
    _row(28, 'Madrid', 'PSOE', 35, 2, 4), _row(8, 'Barcelona', 'PSOE', 30, 1, 3), _row(17, 'Girona', 'PSOE', 20, 0, 1),
    _row(28, 'Madrid', 'VOX', 10, 0, 1), _row(8, 'Barcelona', 'VOX', 5, 0, 0), _row(17, 'Girona', 'VOX', 0, 0, 0),
]
DISTRICTS = {'parties': ['PP', 'PSOE', 'VOX'], 'regions': REGIONS, 'rows': ROWS}
SCENARIO = {'parties': ['PP', 'PSOE', 'VOX'], 'rows': [
    {'region_id': 0, 'region': 'es', 'seats': [7, 5, 0]}, {'region_id': 28, 'region': 'Madrid', 'seats': [3, 3, 0]},
    {'region_id': 8, 'region': 'Barcelona', 'seats': [2, 2, 0]}, {'region_id': 17, 'region': 'Girona', 'seats': [2, 0, 0]},
]}
SUMMARY = {'parties': [{'name': 'PP', 'pct': 33}, {'name': 'PSOE', 'pct': 30}, {'name': 'VOX', 'pct': 7}]}


def run_js(expr, districts=DISTRICTS):
    script = ('import * as t from "{}"; const d = {}, s = {}, u = {}; console.log(JSON.stringify({}));'
              ).format(TERRITORY_JS, json.dumps(districts), json.dumps(SCENARIO), json.dumps(SUMMARY), expr)
    out = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_territory_rows_by_province_in_seats_drop_parties_without_seats_and_carry_ranges():
    out = run_js('t.territoryRows(d, s, u, {level: "prov", mode: "esc"})')
    assert out['parties'] == ['PP', 'PSOE']
    madrid = next(row for row in out['rows'] if row['id'] == 28)
    assert madrid['values'] == {'PP': 3, 'PSOE': 3} and madrid['total'] == 6 and madrid['ranges']['PP'] == '2–4'
    assert out['total'] == {'values': {'PP': 7, 'PSOE': 5}, 'total': 12}


def test_territory_rows_by_community_weight_the_vote_by_the_votes_of_each_province():
    out = run_js('t.territoryRows(d, s, u, {level: "ccaa", mode: "pct"})')
    assert out['parties'] == ['PP', 'PSOE', 'VOX'] and len(out['rows']) == 2
    catalonia = next(row for row in out['rows'] if row['name'] == 'Cataluña')
    assert catalonia['id'] is None and catalonia['total'] == '100 %'
    assert round(catalonia['values']['PP'], 1) == 26.7 and catalonia['ranges'] == {}
    assert out['total'] == {'values': {'PP': 33, 'PSOE': 30, 'VOX': 7}, 'total': '100 %'}


def test_territory_rows_by_community_add_up_the_seats():
    out = run_js('t.territoryRows(d, s, u, {level: "ccaa", mode: "esc"})')
    catalonia = next(row for row in out['rows'] if row['name'] == 'Cataluña')
    assert catalonia['values'] == {'PP': 4, 'PSOE': 2} and catalonia['total'] == 6


def test_territory_rows_fall_back_to_provinces_without_communities_or_votes():
    bare = dict(DISTRICTS, regions=[{k: v for k, v in r.items() if k not in ('community', 'votes')} for r in REGIONS])
    out = run_js('t.territoryRows(d, s, u, {level: "ccaa", mode: "esc"})', districts=bare)
    assert len(out['rows']) == 3 and out['rows'][0]['id'] == 28


def test_territory_rows_leave_the_seats_empty_when_the_scenario_lacks_a_region():
    lonely = dict(DISTRICTS, regions=REGIONS + [{'id': 1, 'name': 'Álava', 'seats': 1}])
    out = run_js('t.territoryRows(d, s, u, {level: "prov", mode: "esc"})', districts=lonely)
    alava = next(row for row in out['rows'] if row['id'] == 1)
    assert alava['values'] == {'PP': None, 'PSOE': None} and alava['total'] == 1


def test_winner_tint_and_sort():
    assert run_js('t.winnerOf({PP: 3, PSOE: 3, VOX: 0})') == ['PP', 'PSOE']
    assert run_js('t.winnerOf({PP: null})') == [] and run_js('t.winnerOf({PP: 0})') == []
    assert run_js('[t.tintAlpha(3, 2, 4), t.tintAlpha(5, 5, 5), t.tintAlpha(null, 0, 1), t.tintAlpha(0, 0, 1)]') == [0.09, 0.18, 0, 0]
    assert run_js('t.columnRange([{values: {PP: 3}}, {values: {PP: 0}}, {values: {PP: 7}}, {values: {PP: null}}], "PP")') == {'min': 3, 'max': 7}
    first = 't.territoryRows(d, s, u, {level: "prov", mode: "esc"}).rows'
    assert [r['name'] for r in run_js('t.sortTerritory(%s, "PP", false)' % first)] == ['Madrid', 'Barcelona', 'Girona']
    assert [r['name'] for r in run_js('t.sortTerritory(%s, "name", true)' % first)] == ['Barcelona', 'Girona', 'Madrid']
    assert [r['name'] for r in run_js('t.sortTerritory(%s, "PSOE", true)' % first)] == ['Girona', 'Barcelona', 'Madrid']
