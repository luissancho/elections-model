"""La colocación de los dibujos de «Escaños que bailan»: casillas del mapa, plazas del hemiciclo y columnas."""
from mtpy.lib import pages
from tests.test_pages_circunscripciones import ODDS

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
