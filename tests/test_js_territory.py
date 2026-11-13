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


def test_territory_rows_by_province_in_seats_keep_every_party_and_carry_ranges():
    """VOX no saca escaño en ningún sitio y aun así es columna (Luis, 2026-10-30): ceros, no ausencias."""
    out = run_js('t.territoryRows(d, s, u, {level: "prov", mode: "esc"})')
    assert out['parties'] == ['PP', 'PSOE', 'VOX']
    madrid = next(row for row in out['rows'] if row['id'] == 28)
    assert madrid['values'] == {'PP': 3, 'PSOE': 3, 'VOX': 0} and madrid['total'] == 6 and madrid['ranges']['PP'] == '2–4'
    assert out['total'] == {'values': {'PP': 7, 'PSOE': 5, 'VOX': 0}, 'total': 12}


def test_territory_rows_by_community_weight_the_vote_by_the_votes_of_each_province():
    out = run_js('t.territoryRows(d, s, u, {level: "ccaa", mode: "pct"})')
    assert out['parties'] == ['PP', 'PSOE', 'VOX'] and len(out['rows']) == 2
    catalonia = next(row for row in out['rows'] if row['name'] == 'Cataluña')
    assert catalonia['id'] is None and catalonia['total'] == '100%'
    assert round(catalonia['values']['PP'], 1) == 26.7 and catalonia['ranges'] == {}
    assert out['total'] == {'values': {'PP': 33, 'PSOE': 30, 'VOX': 7}, 'total': '100%'}


def test_territory_columns_keep_the_seats_order_in_the_vote_view_and_break_ties_by_vote():
    """Las mismas columnas en el mismo orden en Escaños y en % Voto (Luis, 2026-10-30): por escaños del reparto
    central y, entre los que no sacan ninguno, por voto."""
    # VOX vota más que PSOE y aun así va detrás en % Voto, porque tiene menos escaños
    summary = '{parties: [{name: "PP", pct: 33}, {name: "PSOE", pct: 5}, {name: "VOX", pct: 7}]}'
    for mode in ('pct', 'esc'):
        out = run_js('t.territoryRows(d, s, %s, {level: "prov", mode: "%s"}).parties' % (summary, mode))
        assert out == ['PP', 'PSOE', 'VOX'], mode
    # sin escaños para PSOE ni VOX, el voto desempata: VOX antes
    scenario = '{...s, rows: s.rows.map((r) => ({...r, seats: [r.seats[0], 0, 0]}))}'
    for mode in ('pct', 'esc'):
        out = run_js('t.territoryRows(d, %s, %s, {level: "prov", mode: "%s"}).parties' % (scenario, summary, mode))
        assert out == ['PP', 'VOX', 'PSOE'], mode


def test_territory_rows_by_community_add_up_the_seats():
    out = run_js('t.territoryRows(d, s, u, {level: "ccaa", mode: "esc"})')
    catalonia = next(row for row in out['rows'] if row['name'] == 'Cataluña')
    assert catalonia['values'] == {'PP': 4, 'PSOE': 2, 'VOX': 0} and catalonia['total'] == 6


def test_territory_rows_fall_back_to_provinces_without_communities_or_votes():
    bare = dict(DISTRICTS, regions=[{k: v for k, v in r.items() if k not in ('community', 'votes')} for r in REGIONS])
    out = run_js('t.territoryRows(d, s, u, {level: "ccaa", mode: "esc"})', districts=bare)
    assert len(out['rows']) == 3 and out['rows'][0]['id'] == 28


def test_territory_rows_leave_the_seats_empty_when_the_scenario_lacks_a_region():
    lonely = dict(DISTRICTS, regions=REGIONS + [{'id': 1, 'name': 'Álava', 'seats': 1}])
    out = run_js('t.territoryRows(d, s, u, {level: "prov", mode: "esc"})', districts=lonely)
    alava = next(row for row in out['rows'] if row['id'] == 1)
    assert alava['values'] == {'PP': None, 'PSOE': None, 'VOX': None} and alava['total'] == 1


def test_winner_tint_and_sort():
    assert run_js('t.winnerOf({PP: 3, PSOE: 3, VOX: 0})') == ['PP', 'PSOE']
    assert run_js('t.winnerOf({PP: null})') == [] and run_js('t.winnerOf({PP: 0})') == []
    assert run_js('[t.tintAlpha(3, 2, 4), t.tintAlpha(5, 5, 5), t.tintAlpha(null, 0, 1), t.tintAlpha(0, 0, 1)]') == [0.09, 0.18, 0, 0]
    assert run_js('t.columnRange([{values: {PP: 3}}, {values: {PP: 0}}, {values: {PP: 7}}, {values: {PP: null}}], "PP")') == {'min': 3, 'max': 7}
    first = 't.territoryRows(d, s, u, {level: "prov", mode: "esc"}).rows'
    assert [r['name'] for r in run_js('t.sortTerritory(%s, "PP", false)' % first)] == ['Madrid', 'Barcelona', 'Girona']
    assert [r['name'] for r in run_js('t.sortTerritory(%s, "name", true)' % first)] == ['Barcelona', 'Girona', 'Madrid']
    assert [r['name'] for r in run_js('t.sortTerritory(%s, "PSOE", true)' % first)] == ['Girona', 'Barcelona', 'Madrid']


def test_can_group_needs_votes_and_two_communities_and_ranking_and_ranges_are_defined():
    assert run_js('t.canGroup(d.regions)') is True
    assert run_js('t.canGroup(d.regions.map(r => ({...r, votes: undefined})))') is False
    assert run_js('t.canGroup(d.regions.slice(0, 1))') is False
    # sin resumen (voto desconocido): por escaños y luego por nombre, sin NaN
    out = run_js('t.territoryRows(d, s, {parties: []}, {level: "prov", mode: "pct"}).parties')
    assert out == ['PP', 'PSOE', 'VOX']
    # sin límites del rango, la celda no lleva rango
    gap = dict(DISTRICTS, rows=[dict(r, seats_lo=None) if r['name'] == 'PP' else r for r in ROWS])
    out = run_js('t.territoryRows(d, s, u, {level: "prov", mode: "esc"}).rows[0].ranges', districts=gap)
    assert out['PP'] is None and out['PSOE'] == '2–4'


def test_community_vote_counts_a_province_where_the_party_does_not_stand_as_zero():
    regions = [
        {'id': 35, 'name': 'Las Palmas', 'seats': 8, 'votes': 100, 'community': {'code': 5, 'name': 'Canarias'}},
        {'id': 38, 'name': 'S. C. Tenerife', 'seats': 7, 'votes': 100, 'community': {'code': 5, 'name': 'Canarias'}},
        REGIONS[0],
    ]
    rows = [_row(28, 'Madrid', 'PP', 40, 2, 4), _row(28, 'Madrid', 'VOX', 10, 0, 1), _row(35, 'Las Palmas', 'VOX', 12, 0, 1), _row(38, 'S. C. Tenerife', 'VOX', None, None, None),
            _row(35, 'Las Palmas', 'PP', 30, 1, 3), _row(38, 'S. C. Tenerife', 'PP', 20, 1, 3)]
    districts = {'parties': ['PP', 'VOX'], 'regions': regions, 'rows': rows}
    out = run_js('t.territoryRows(d, s, u, {level: "ccaa", mode: "pct"})', districts)
    canarias = next(row for row in out['rows'] if row['name'] == 'Canarias')
    assert canarias['values']['VOX'] == 6.0 and canarias['values']['PP'] == 25.0


def test_region_rows_mirror_python():
    """`regionRows` es el espejo de `region_rows` de Python: sin las filas sin voto y por escaños, luego por voto."""
    from mtpy.lib.pages.escanos import region_rows

    def row(name, pct, seats):
        return dict(_row(8, 'Barcelona', name, pct, 0, 1), seats=seats)

    rows = [row('VOX', 12.5, 2), row('PSOE', None, None), row('PP', 25.0, 5), _row(17, 'Girona', 'PP', 20, 1, 3)]
    districts = dict(DISTRICTS, rows=rows)
    out = run_js('t.regionRows(d, 8)', districts)
    assert [r['name'] for r in out] == ['PP', 'VOX']
    assert out == region_rows(districts, 8)
    # a tie in seats is broken by the vote, the order of the input kept on a full tie (stable)
    tied = [row('A', 10.0, 2), row('B', 15.0, 2), row('C', 10.0, 2), row('D', 5.0, 3)]
    districts = dict(DISTRICTS, rows=tied)
    out = run_js('t.regionRows(d, 8)', districts)
    assert [r['name'] for r in out] == ['D', 'B', 'A', 'C'] and out == region_rows(districts, 8)
    assert run_js('t.regionRows(d, 99)', districts) == []


def test_region_href_lands_on_the_seats_section_with_the_district():
    """El enlace de una circunscripción de la tabla, como lo escribe la plantilla: la página del evento en la sección
    «Escaños» (`seccion=escanos#escanos`, 2026-11-13; antes `territorio`), con el modo y, fijado, el run."""
    state = '{scope: "es", mode: "forecast", run: "20261008-120000", pinned: false}'
    assert run_js('t.regionHref(%s, 28)' % state) == '/es?mode=forecast&region=28&seccion=escanos#escanos'
    pinned = '{scope: "es", mode: "nowcast", run: "20261008-120000", pinned: true}'
    assert run_js('t.regionHref(%s, 8)' % pinned) == \
        '/es?mode=nowcast&run=20261008-120000&region=8&seccion=escanos#escanos'


def test_the_detail_of_a_district_no_longer_links_the_page_that_was():
    """«Escaños que bailan» es una tarjeta de la misma sección (2026-11-13): el detalle del territorio pierde su enlace
    «Escaño a escaño en…» y el script, el código que lo apuntaba a cada circunscripción."""
    wire = pathlib.Path(ROOT, 'web', 'dist', 'js', 'pages', 'seats-wire.js').read_text(encoding='utf-8')
    assert not [name for name in ('region-odds', 'odds_hrefs', 'showOddsLink', 'Escaño a escaño') if name in wire]
