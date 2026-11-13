"""La colocación de los dibujos de «Escaños que bailan»: casillas del mapa y columnas."""
from mtpy.lib import pages
from tests.test_pages_seat_odds import ODDS

POSITIONS = {11: {'abbr': 'CA', 'col': 3, 'row': 8, 'w': 1, 'h': 1}, 19: {'abbr': 'GU', 'col': 6, 'row': 3, 'w': 1, 'h': 1},
             15: {'abbr': 'C', 'col': 0, 'row': 0, 'w': 1, 'h': 1}}


def test_map_tiles_place_each_district_and_sort_its_options():
    views = pages.district_views(ODDS)
    grid = pages.map_tiles(views, POSITIONS, ODDS['parties'])
    assert grid['placed'] and [tile['label'] for tile in grid['tiles']] == ['C', 'GU', 'CA']  # by row, then column
    cadiz = grid['tiles'][2]
    assert (cadiz['name'], cadiz['slug'], cadiz['col'], cadiz['row'], cadiz['w'], cadiz['h']) == ('Cádiz', 'cadiz', 3, 8, 1, 1)
    assert [(o['party'], o['state'], o['k']) for o in cadiz['options']] == [
        ('PP', 'safe', 1), ('PP', 'safe', 2), ('PP', 'likely', 3), ('PSOE', 'safe', 1), ('PSOE', 'safe', 2),
        ('PSOE', 'play', 3), ('VOX', 'safe', 1), ('VOX', 'likely', 2), ('AA', 'likely', 1)]
    assert len(pages.scope_options(views)) == 22 and pages.scope_options(views)[0]['district'] == 'Cádiz'


def test_a_scope_without_every_position_draws_its_tiles_in_a_row_by_name():
    views = pages.district_views(ODDS)
    for positions in ({}, {11: POSITIONS[11]}):  # no catalogue columns; one district placed and two not
        grid = pages.map_tiles(views, positions, ODDS['parties'])
        assert not grid['placed'] and [tile['label'] for tile in grid['tiles']] == ['Cádiz', 'Guadalajara', 'La Coruña']
        assert all(tile['col'] is None and tile['w'] is None for tile in grid['tiles'])
    empty = dict(ODDS, rows=[row for row in ODDS['rows'] if row['region_id'] != 19])  # a district without options
    tiles = pages.map_tiles(pages.district_views(empty), POSITIONS, ODDS['parties'])['tiles']
    assert [(tile['label'], len(tile['options'])) for tile in tiles] == [('C', 10), ('GU', 0), ('CA', 9)]


def test_the_columns_follow_the_order_given_with_the_rest_last_and_fall_back_to_one_per_party():
    options = pages.scope_options(pages.district_views(ODDS))
    blocks = {'Izquierda': ['PSOE', 'AA', 'FA'], 'Derecha': ['PP', 'VOX'], 'Otros': []}  # BNG is in no block; FA has no option
    columns = pages.party_columns(options, ODDS['parties'], blocks)
    # the parties of Izquierda and Derecha in the order given (that of the table of results), not block by block
    assert [(c['name'], c['safe'], c['likely'], c['play']) for c in columns] == [
        ('PP', 6, 1, 1), ('PSOE', 4, 1, 1), ('VOX', 2, 1, 1), ('AA', 0, 1, 0), ('Otros', 2, 0, 1)]
    assert columns[-1]['members'] == ['BNG'] and sum(len(c['options']) for c in columns) == 22
    assert [o['state'] for o in columns[0]['options']] == ['safe'] * 6 + ['likely', 'play']
    ranked = pages.party_columns(options, ['BNG', 'VOX', 'PSOE', 'PP', 'AA'], blocks)
    assert [c['name'] for c in ranked] == ['VOX', 'PSOE', 'PP', 'AA', 'Otros']  # «Otros» last wherever its parties rank
    bare = pages.party_columns(options, ODDS['parties'], None)
    assert [c['name'] for c in bare] == ['PP', 'PSOE', 'VOX', 'AA', 'BNG']
    assert pages.party_columns(options, ODDS['parties'], {'Derecha': ['PP']}) == bare  # the blocks of the test bundle


def test_the_axis_of_the_columns_ends_with_the_mark_above_the_tallest_one():
    # a mark every 50 options, up to the first one at or above the tallest column: PP's 141 options sit under a 150
    ticks = pages.column_ticks
    assert ticks([{'options': [0] * 141}, {'options': [0] * 121}, {'options': [0] * 2}]) == [50, 100, 150]
    assert ticks([{'options': [0] * 100}]) == [50, 100] and ticks([{'options': [0] * 101}]) == [50, 100, 150]
    assert ticks([{'options': [0] * 8}]) == [50] and ticks([{'options': [0] * 50}]) == [50]
    assert ticks([]) == [] and ticks([{'options': []}]) == []  # nothing to measure: no axis
    columns = pages.party_columns(pages.scope_options(pages.district_views(ODDS)), ODDS['parties'], None)
    assert ticks(columns) == [50]
