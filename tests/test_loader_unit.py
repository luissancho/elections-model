"""Tests de los cargadores de Wikipedia que no necesitan base de datos ni red (M11).

Los fixtures de `tests/fixtures/wikipedia/` son artículos recortados (encabezados y tablas) de la
instantánea del 01-10-2026; ver `build_fixtures.py`.
"""
import os

import pandas as pd
import pytest
from lxml import html

from mtpy.lib.loader import WikipediaLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data')
FIXTURES = os.path.join(ROOT, 'tests', 'fixtures', 'wikipedia')


class AnyKey(dict):
    """Mapa de ids que acepta cualquier nombre (todas las casas y patrocinadores mapeados)."""

    def __contains__(self, key):
        return key is not None

    def __getitem__(self, key):
        return abs(hash(key)) % 10 ** 6 + 1


def load(page):
    with open(os.path.join(FIXTURES, page + '.html'), 'rb') as fh:
        return html.fromstring(fh.read())


def bare_loader(scope, maps=None, party_ids=None):
    """Cargador sin base de datos: sólo los atributos que usa la lectura de tablas."""
    loader = WikipediaLoader.__new__(WikipediaLoader)
    loader.scope = scope
    loader.charsep = '–'
    loader.maps = maps or {'parties': {}, 'pollsters': {}, 'sponsors': {}, 'scopes': {}}
    loader.parties_colmap = loader.get_colmap('parties')
    loader.parties_idmap = party_ids or {}
    loader.pollsters_colmap = loader.get_colmap('pollsters')
    loader.pollsters_idmap = AnyKey()
    loader.sponsors_colmap = loader.get_colmap('sponsors')
    loader.sponsors_idmap = AnyKey()
    loader.parties_missing, loader.pollsters_missing, loader.sponsors_missing = [], [], []

    return loader


# (filas tr, columnas de partido, columna Lead, sondeos leídos): instantánea del 01-10-2026.
# Sondeos = filas que no están en negrita y cuya última celda es numérica.
CASES = {
    '2023_Madrilenian_regional_election': (59, 6, 10, 50),
    'Next_Madrilenian_regional_election': (26, 8, 12, 14),
    '2022_Castilian-Leonese_regional_election': (77, 11, 15, 65),
    '2019_Asturian_regional_election': (36, 9, 13, 25),      # tabla bajo el h2, sin h3
}


@pytest.mark.parametrize('page', CASES)
def test_poll_table_header_and_rows(page):
    n_tr, n_parties, lead_col, n_polls = CASES[page]
    table = WikipediaLoader.find_poll_table(load(page))
    assert len(table.xpath('.//tr')) == n_tr
    parties, lead = WikipediaLoader.read_header(table)
    assert (len(parties), lead) == (n_parties, lead_col)
    rows = bare_loader('es-md').read_table(table, page[:4] if page[0].isdigit() else '2026')
    assert len(rows) == n_polls
    assert not any('regional election' in row['pollster'] for row in rows)   # sin filas de resultado


def test_madrid_2023_header_keys_and_first_poll():
    table = WikipediaLoader.find_poll_table(load('2023_Madrilenian_regional_election'))
    parties, _ = WikipediaLoader.read_header(table)
    assert parties == {
        4: "People's_Party_of_the_Community_of_Madrid", 5: 'Más_Madrid',
        6: "Spanish_Socialist_Workers'_Party_of_the_Community_of_Madrid", 7: 'Vox_(political_party)',
        8: 'Unidas_Podemos', 9: 'Citizens_(Spanish_political_party)'
    }
    first = bare_loader('es-md').read_table(table, '2023')[0]
    assert (first['pollster'], first['sponsor']) == ('GAD3', 'RTVE-FORTA')
    assert (first['start_date'], first['end_date']) == (pd.Timestamp('2023-05-12'), pd.Timestamp('2023-05-27'))
    pp = first['results'][0]
    assert (pp['pct'], pp['seats_min'], pp['seats_max']) == (49.5, 70, 72)


def test_next_madrid_reads_the_last_party_column():
    parties, lead = WikipediaLoader.read_header(WikipediaLoader.find_poll_table(load('Next_Madrilenian_regional_election')))
    assert parties[11] == 'Se_Acabó_La_Fiesta' and lead == 12


def test_non_numeric_cells_are_skipped():
    # Foco 3: '[f]', '?', '–' y 'Tie' no rompen la lectura ni dejan resultados sin porcentaje
    table = WikipediaLoader.find_poll_table(load('2022_Castilian-Leonese_regional_election'))
    rows = bare_loader('es-cl').read_table(table, '2022')
    assert all(pd.notnull(r['pct']) and r['pct'] > 0 for row in rows for r in row['results'])


def test_scope_alias_wins_and_unknown_party_is_reported():
    # Foco 2: el alias del ámbito prevalece; lo no mapeado se lista, no se asigna
    maps = {
        'parties': {'PP': ['People%27s_Party_(Spain)']}, 'pollsters': {}, 'sponsors': {},
        'scopes': {'es-md': {'parties': {
            'PP': ['People%27s_Party_of_the_Community_of_Madrid'], 'MM': ['M%C3%A1s_Madrid']
        }}}
    }
    loader = bare_loader('es-md', maps=maps, party_ids={'PP': 2, 'MM': 600})
    colmap = loader.get_colmap('parties')
    assert colmap["People's_Party_of_the_Community_of_Madrid"] == 'PP' and colmap['Más_Madrid'] == 'MM'
    assert colmap["People's_Party_(Spain)"] == 'PP'
    rows = loader.read_table(WikipediaLoader.find_poll_table(load('2023_Madrilenian_regional_election')), '2023')
    assert {r['party'] for r in rows[0]['results'] if r['party_id']} == {'PP', 'MM'}
    assert 'Vox_(political_party)' in loader.parties_missing
    assert all(r['party_id'] is None for r in rows[0]['results'] if r['party'] not in ('PP', 'MM'))
