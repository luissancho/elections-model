"""Tests de los cargadores de Wikipedia que no necesitan base de datos ni red (M11).

Los fixtures de `tests/fixtures/wikipedia/` son artículos recortados (encabezados y tablas) de la
instantánea del 01-10-2026; ver `build_fixtures.py`.
"""
import copy
import os

import pandas as pd
import pytest
from lxml import html

from mtpy.core.utils.helpers import is_number
from mtpy.lib.loader import WikipediaLoader, apply_poll_fixes, normalized

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


def bare_loader(scope, maps=None, party_ids=None, party_names=None):
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
    loader.party_names = {normalized(n) for n in party_names or ()}
    loader.pollsters_skipped, loader.sponsors_skipped, loader.parties_skipped = [], [], []

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


def first_poll_cols(table):
    """Cells (`td`) of the first poll row of a poll table (no style: the election result row is shaded)."""
    for tr in table.xpath('.//tr'):
        cols = tr.xpath('./td')
        if len(cols) > 3 and 'style' not in tr.attrib and is_number((cols[-1].text_content() or '').strip()):
            return cols
    raise AssertionError('no poll row')


def test_read_cols_ignores_the_party_columns_of_skip_parties():
    maps = {'parties': {}, 'pollsters': {}, 'sponsors': {}, 'scopes': {}, 'skip': {'parties': ['Más_Madrid']}}
    loader = bare_loader('es-md', maps=maps, party_ids={"People's_Party_of_the_Community_of_Madrid": 1})
    table = WikipediaLoader.find_poll_table(load('2023_Madrilenian_regional_election'))
    parties, lead = WikipediaLoader.read_header(table)
    row = loader.read_cols(first_poll_cols(table), parties, lead, '2023')
    assert 'Más_Madrid' not in [r['party'] for r in row['results']]
    assert loader.parties_skipped == ['Más_Madrid'] and 'Más_Madrid' not in loader.parties_missing
    assert [r for r in row['results'] if r['party_id'] is None]  # the other unmapped columns stay without id


def test_read_cols_maps_a_skipped_key_that_the_scope_aliases():
    maps = {'parties': {}, 'pollsters': {}, 'sponsors': {}, 'scopes': {'es-md': {'parties': {'Podemos_Madrid': ['Más_Madrid']}}},
            'skip': {'parties': ['Más_Madrid']}}
    loader = bare_loader('es-md', maps=maps, party_ids={'Podemos_Madrid': 7})
    table = WikipediaLoader.find_poll_table(load('2023_Madrilenian_regional_election'))
    parties, lead = WikipediaLoader.read_header(table)
    row = loader.read_cols(first_poll_cols(table), parties, lead, '2023')
    assert 7 in [r['party_id'] for r in row['results']]
    assert 'Más_Madrid' not in loader.parties_skipped


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


def test_fetch_page_reads_the_cache_before_the_network(tmp_path):
    # Con la página en caché no se toca la red (el dominio no existe)
    from mtpy.lib.loader import fetch_page
    (tmp_path / 'Some_page.html').write_bytes(b'<html>cached</html>')
    assert fetch_page('https://invalid.invalid/wiki/Some_page', cache_dir=str(tmp_path)) == b'<html>cached</html>'


def test_parse_dates_variants_and_unparseable_cells():
    # Foco 3: fechas con espacios, con el inicio desconocido o sin día; lo ilegible devuelve None
    loader = bare_loader('es-md')
    ts = pd.Timestamp
    assert loader.parse_dates('5–7 Apr 2005', '2005') == (ts('2005-04-05'), ts('2005-04-07'))
    assert loader.parse_dates('30 Mar–6 Apr 2005', '2005') == (ts('2005-03-30'), ts('2005-04-06'))
    assert loader.parse_dates('17 Mar – 17 Apr 2011', '2011') == (ts('2011-03-17'), ts('2011-04-17'))
    assert loader.parse_dates('28 Dec–3 Jan 2019', '2019') == (ts('2018-12-28'), ts('2019-01-03'))
    assert loader.parse_dates('?–16 May 2015', '2015') == (ts('2015-05-16'), ts('2015-05-16'))
    for bad in ['?', '–', '42.9', 'Dec 2019', 'Mar–Apr 2024']:
        assert loader.parse_dates(bad, '2019') is None, bad


# --- Resultados (Results › Overall y Distribution by constituency) ---

from mtpy.lib.loader import WikipediaResultsLoader


def results(page):
    overall, const = WikipediaResultsLoader.find_results_tables(load(page))
    rows, totals = WikipediaResultsLoader.parse_overall(overall)
    return rows.set_index('abbr'), totals, None if const is None else WikipediaResultsLoader.parse_constituencies(const)


def test_overall_castile_and_leon_2022():
    rows, totals, const = results('2022_Castilian-Leonese_regional_election')
    assert rows.loc['PP', ['key', 'votes', 'pct', 'seats']].tolist() == ["People's_Party_of_Castile_and_León", 382157, 31.40, 31]
    assert {k: totals[k] for k in ['votes', 'invalid', 'counted', 'abstentions', 'registered', 'seats']} == {
        'votes': 1217164, 'invalid': 13435, 'counted': 1230599, 'abstentions': 864024, 'registered': 2094623, 'seats': 81
    }
    assert totals['blank'] > 0 and rows['seats'].sum() == 81
    assert rows['votes'].sum() + totals['blank'] == totals['votes']
    cell = const.set_index(['district', 'key'])
    assert const['district'].nunique() == 9 and const['seats'].sum() == 81
    assert cell.loc[('Ávila', "People's_Party_of_Castile_and_León")].tolist()[-2:] == [34.0, 3]
    assert cell.loc[('León', "Leonese_People's_Union"), 'pct'] == 21.3
    assert cell.loc[('Soria', 'Empty_Spain'), 'pct'] == 42.7
    assert const.groupby('district')['seats'].sum().to_dict() == {
        'Ávila': 7, 'Burgos': 11, 'León': 13, 'Palencia': 7, 'Salamanca': 10, 'Segovia': 6, 'Soria': 5,
        'Valladolid': 15, 'Zamora': 7
    }


def test_canaries_2023_has_a_regional_list_and_two_vote_blocks():
    rows, totals, const = results('2023_Canarian_regional_election')
    assert (totals['votes'], totals['blank'], totals['seats']) == (912219, 15947, 70)
    assert rows.loc['PSOE', ['votes', 'pct', 'seats']].tolist() == [247811, 27.17, 23]
    cell = const.set_index(['district', 'key'])
    assert const['district'].nunique() == 8 and const['seats'].sum() == 70
    assert cell.loc[('Regional', 'Socialist_Party_of_the_Canaries')].tolist()[-2:] == [32.4, 4]
    assert cell.loc[('El Hierro', 'Independent_Herrenian_Group')].tolist()[-2:] == [26.3, 1]


def test_asturias_2023_constituencies_are_not_provinces():
    _, totals, const = results('2023_Asturian_regional_election')
    assert const.groupby('district')['seats'].sum().to_dict() == {'Central': 34, 'Eastern': 5, 'Western': 6}
    assert (totals['votes'], totals['registered'], totals['seats']) == (537023, 958658, 45)


def test_madrid_2023_is_a_single_district():
    rows, totals, const = results('2023_Madrilenian_regional_election')
    assert const is None
    assert rows.loc['PP', ['votes', 'pct', 'seats']].tolist() == [1599186, 47.32, 70]
    assert (totals['votes'], totals['registered'], totals['seats']) == (3379477, 5211710, 135)


def test_split_votes_is_exact():
    out = WikipediaResultsLoader.split_votes(100, pd.Series({'a': 1, 'b': 1, 'c': 1}))
    assert out.tolist() == [34, 33, 33]
    assert WikipediaResultsLoader.split_votes(1217164, pd.Series({'x': 3., 'y': 7.})).sum() == 1217164


def frames(page, scope, event_date, **kwargs):
    rows, totals, const = results(page)
    districts = pd.read_csv(os.path.join(DATA, 'es-districts.csv'))
    names = {key: key for key in rows['key'].dropna()}       # identidad: todo mapeado
    return WikipediaResultsLoader.build_frames(
        scope, event_date, rows.reset_index(), totals, const, districts.loc[districts['scope'] == scope], names, **kwargs
    )


def test_build_frames_castile_and_leon_2022():
    data, res = frames('2022_Castilian-Leonese_regional_election', 'es-cl', '2022-02-13')
    total = data.set_index('region_id').loc[0]
    assert total['votes'] == 1217164 and total['seats'] == 81 and not total['estimated']
    parts = data.loc[data['region_id'] > 0]
    assert sorted(parts['region_id']) == [5, 9, 24, 34, 37, 40, 42, 47, 49]
    assert parts['estimated'].all() and parts['votes'].sum() == 1217164 and parts['seats'].sum() == 81
    votes0 = res.loc[res['region_id'] == 0, 'votes'].sum()
    assert abs(votes0 - (total['votes'] - total['blank'])) <= 0.001 * total['votes']
    assert res.loc[res['region_id'] > 0, 'seats'].sum() == 81


def test_build_frames_official_votes_replace_the_estimate():
    official = pd.DataFrame({'region_id': [5], 'votes': [90000]})
    data, _ = frames('2022_Castilian-Leonese_regional_election', 'es-cl', '2022-02-13', official=official)
    avila = data.set_index('region_id').loc[5]
    assert avila['votes'] == 90000 and not avila['estimated']


def test_build_frames_single_district_and_regional_list():
    data, res = frames('2023_Madrilenian_regional_election', 'es-md', '2023-05-28')
    assert sorted(data['region_id']) == [0, 28]
    assert data.set_index('region_id').loc[28, 'seats'] == 135
    data, _ = frames('2023_Canarian_regional_election', 'es-cn', '2023-05-28')
    by_id = data.set_index('region_id')
    assert sorted(by_id.index) == [0, 100, 101, 102, 103, 104, 105, 106, 107]
    assert by_id.loc[100, 'seats'] == 9 and by_id.loc[range(101, 108), 'votes'].sum() == 912219


def test_unknown_constituency_raises_with_its_name():
    # Foco 4: sin el alias, el error nombra la circunscripción
    rows, totals, const = results('2023_Asturian_regional_election')
    districts = pd.read_csv(os.path.join(DATA, 'es-districts.csv'))
    districts = districts.loc[districts['scope'] == 'es-as'].assign(aliases='')
    with pytest.raises(ValueError, match='Eastern'):
        WikipediaResultsLoader.build_frames('es-as', '2023-05-28', rows.reset_index(), totals, const, districts, {})


def test_skipped_pollsters_and_sponsors_are_not_reported():
    # Sondeos internos de partido: la casa de la lista `skip` se descarta sin figurar como pendiente;
    # un patrocinador de la lista deja el sondeo sin patrocinador (id 0), tampoco pendiente
    maps = {
        'parties': {}, 'pollsters': {}, 'sponsors': {}, 'scopes': {},
        'skip': {'pollsters': ['GAD3'], 'sponsors': ['La Razón']}
    }
    loader = bare_loader('es-md', maps=maps)
    loader.pollsters_idmap = {'NC Report': 7}
    loader.sponsors_idmap = {}
    rows = loader.read_table(WikipediaLoader.find_poll_table(load('2023_Madrilenian_regional_election')), '2023')
    assert {row['pollster'] for row in rows} == {'NC Report'}
    assert all(row['sponsor_id'] == 0 for row in rows)
    assert 'GAD3' not in loader.pollsters_missing and 'Sigma Dos' in loader.pollsters_missing
    assert 'La Razón' not in loader.sponsors_missing


def test_columns_mapped_to_the_same_party_are_merged():
    # Dos columnas del mismo sondeo que acaban en el mismo partido (miembros de una coalición posterior):
    # un solo resultado con la suma, no dos filas con la misma clave
    maps = {
        'parties': {}, 'pollsters': {}, 'sponsors': {},
        'scopes': {'es-md': {'parties': {'X': ["People's_Party_of_the_Community_of_Madrid", 'Más_Madrid']}}}
    }
    loader = bare_loader('es-md', maps=maps, party_ids={'X': 9})
    first = loader.read_table(WikipediaLoader.find_poll_table(load('2023_Madrilenian_regional_election')), '2023')[0]
    merged = [r for r in first['results'] if r['party'] == 'X']
    assert len(merged) == 1
    assert merged[0]['pct'] == pytest.approx(49.5 + 14.3)
    assert (merged[0]['seats_min'], merged[0]['seats_max']) == (90, 92)


def test_build_frames_seat_fixes_correct_the_article():
    # Erratas del artículo (Cataluña 2010: CiU 8 en Lleida, fueron 9): correcciones por circunscripción y candidatura
    fixes = pd.DataFrame({'region_id': [5], 'key': ["People's_Party_of_Castile_and_León"], 'seats': [4]})
    data, res = frames('2022_Castilian-Leonese_regional_election', 'es-cl', '2022-02-13', seat_fixes=fixes)
    assert data.set_index('region_id').loc[5, 'seats'] == 8
    avila = res.loc[(res['region_id'] == 5) & (res['party'] == "People's_Party_of_Castile_and_León")]
    assert avila['seats'].tolist() == [4]
    with pytest.raises(ValueError, match='not in the table'):
        frames('2022_Castilian-Leonese_regional_election', 'es-cl', '2022-02-13',
               seat_fixes=pd.DataFrame({'region_id': [5], 'key': ['Nobody'], 'seats': [1]}))


def test_party_name_prefers_the_abbreviation_alias():
    # Andalucía 2022: la fila de Por Andalucía enlaza a IULV-CA pero su abreviatura es PorA
    names = {'United_Left/The_Greens–Assembly_for_Andalusia': 'IU', 'PorA': 'PorA'}
    name = WikipediaResultsLoader.party_name
    assert name('United_Left/The_Greens–Assembly_for_Andalusia', 'PorA', names) == 'PorA'
    assert name('United_Left/The_Greens–Assembly_for_Andalusia', 'IULV–CA', names) == 'IU'
    assert name(None, 'XYZ', names) == '-'


def test_parties_outside_the_constituency_table_get_flat_district_shares():
    # Un partido con resultado en el conjunto pero sin columna en la tabla por circunscripción (sin escaños)
    # recibe en cada circunscripción su cuota del conjunto: es la base del swing si después crece (VOX 2019)
    data, res = frames('2022_Castilian-Leonese_regional_election', 'es-cl', '2022-02-13')
    pacma = res.loc[res['party'] == 'Animalist_Party_Against_Mistreatment_of_Animals']
    assert sorted(pacma['region_id']) == [0, 5, 9, 24, 34, 37, 40, 42, 47, 49]
    by_region = pacma.set_index('region_id')['pct']
    # 0,54 % donde cabe; menos en Soria, donde Soria ¡Ya! deja poco voto para el resto
    assert by_region.loc[9] == 0.54 and (by_region.drop(0) <= 0.54).all() and by_region.loc[42] < 0.54
    assert pacma['seats'].sum() == 0
    # Las cuotas de cada circunscripción siguen sumando 100 con los votos en blanco
    blank = data.set_index('region_id')['blank'] / data.set_index('region_id')['votes'] * 100
    total = res.loc[res['region_id'] > 0].groupby('region_id')['pct'].sum() + blank.reindex(range(1, 53)).dropna()
    assert total.between(99.5, 100.5).all()


# --- Revisión final de la rama: caché, columnas sin enlace, celdas vacías, guardas del cargador por lotes ---

def fixture_bytes(page):
    with open(os.path.join(FIXTURES, page + '.html'), 'rb') as fh:
        return fh.read()


def test_read_data_passes_refresh_to_the_download(monkeypatch):
    # Sin esto, `--refresh` no llegaba a los cargadores y se releía siempre la copia en caché
    from mtpy.lib import loader as loader_module
    calls = []

    def fake_fetch(url, cache_dir=None, refresh=False):
        calls.append((url, cache_dir, refresh))
        return fixture_bytes('2023_Madrilenian_regional_election')

    monkeypatch.setattr(loader_module, 'fetch_page', fake_fetch)

    polls = bare_loader('es-md')
    polls.params, polls.years, polls.verbose, polls.event_date = {'polls': 'https://x/wiki/A'}, None, 0, '2023-05-28'
    polls.cache_dir, polls.refresh = 'cache', True
    polls.read_data()
    assert calls == [('https://x/wiki/A', 'cache', True)] and len(polls.data) == 50

    results = WikipediaResultsLoader.__new__(WikipediaResultsLoader)
    results.scope, results.event_date, results.params = 'es-md', '2023-05-28', {'results': 'https://x/wiki/B'}
    results.parties_colmap, results.cache_dir, results.refresh = {}, 'cache', True
    results.read_data()
    assert calls[-1] == ('https://x/wiki/B', 'cache', True) and results.overall_totals['seats'] == 135


POLLS_TABLE = '''<table class="wikitable">
<tr><th rowspan="2">Polling firm/Commissioner</th><th rowspan="2">Fieldwork date</th><th rowspan="2">Sample size</th>
<th rowspan="2">Turnout</th><th><a href="/wiki/PP_X">logo</a></th><th>JUEx</th><th rowspan="2">Lead</th></tr>
<tr><th></th><th></th></tr>
<tr><td>GAD3/RTVE</td><td>1–2 May 2023</td><td>1,000</td><td>?</td><td>40.0</td><td>2.5</td><td>10.0</td></tr>
<tr><td>Sigma Dos</td><td>3 May 2023</td><td></td><td>?</td><td>41.0</td><td>3.0</td><td>9.0</td></tr>
<tr><td></td><td>4 May 2023</td><td>800</td><td>?</td><td>42.0</td><td>3.5</td><td>8.0</td></tr>
</table>'''


def test_party_column_without_link_uses_its_text():
    # Foco 2: una columna de partido sin enlace (JUEx) no puede desaparecer: su texto es la clave
    table = html.fromstring(POLLS_TABLE)
    parties, lead = WikipediaLoader.read_header(table)
    assert parties == {4: 'PP_X', 5: 'JUEx'} and lead == 6
    loader = bare_loader('es-ex')
    rows = loader.read_table(table, '2023')
    assert 'JUEx' in loader.parties_missing and [r['party'] for r in rows[0]['results']] == ['PP_X', 'JUEx']


def test_empty_cells_do_not_abort_the_table():
    # Foco 3: una celda vacía de muestra deja el tamaño sin valor; una de casa descarta la fila
    rows = bare_loader('es-ex').read_table(html.fromstring(POLLS_TABLE), '2023')
    assert [row['pollster'] for row in rows] == ['GAD3', 'Sigma Dos']
    assert rows[0]['sample_size'] == 1000 and pd.isnull(rows[1]['sample_size'])


def run_load_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('run_load', os.path.join(ROOT, 'load', 'run_load.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_load_rejects_the_national_scope():
    # El cargador por lotes es de los ámbitos autonómicos: con `es` borraría su inventario y sus `featured`
    module = run_load_module()
    catalogue = pd.read_csv(os.path.join(DATA, 'es-scopes.csv')).set_index('scode')
    assert module.check_scopes(['es-md', 'es-cl'], catalogue) == ['es-md', 'es-cl']
    for bad in (['es'], ['es-md', 'es-xx']):
        with pytest.raises(ValueError, match='regional scopes'):
            module.check_scopes(bad, catalogue)


def test_merge_urls_keeps_a_hand_corrected_upcoming_event():
    module = run_load_module()
    existing = {'2027-06-20': {'results': 'next'}, '2023-05-28': {'results': 'a', 'polls': 'a'}}
    rebuilt = {'2027-06-27': {'results': 'next', 'polls': 'next'}, '2023-05-28': {'results': 'a', 'polls': 'a'},
               '2019-05-26': {'results': 'b', 'polls': 'b'}}
    merged, kept = module.merge_urls(existing, rebuilt, today='2026-10-01')
    # La fecha del evento próximo ya inventariado se respeta (pudo corregirse a mano), con las URL nuevas
    assert list(merged) == ['2027-06-20', '2023-05-28', '2019-05-26'] and kept == ('2027-06-20', '2027-06-27')
    assert merged['2027-06-20'] == {'results': 'next', 'polls': 'next'}
    merged, kept = module.merge_urls({}, rebuilt, today='2026-10-01')
    assert merged == rebuilt and kept is None


def test_effective_refresh_downloads_again_when_saving():
    module = run_load_module()
    assert module.effective_refresh(refresh=False, save=True, cached=False) is True
    assert module.effective_refresh(refresh=False, save=True, cached=True) is False
    assert module.effective_refresh(refresh=False, save=False, cached=False) is False
    assert module.effective_refresh(refresh=True, save=False, cached=True) is True


def test_party_column_with_only_a_logo_uses_its_caption():
    # Una columna con sólo el logo (enlace a `File:`) se identifica por el título de la imagen
    table = html.fromstring(POLLS_TABLE.replace('<th>JUEx</th>', '<th><a href="/wiki/File:Logo_JUEx.svg" title="JUEx"></a></th>'))
    parties, lead = WikipediaLoader.read_header(table)
    assert parties == {4: 'PP_X', 5: 'JUEx'} and lead == 6


def test_run_load_compute_phases_delegate_to_the_library(monkeypatch):
    """Los pasos `compute` y `ratings` del cargador son los de `mtpy.lib.compute` (una sola implementación), con
    el `verbose` del Computer un nivel por debajo del suyo."""
    from mtpy.lib import compute
    calls = []
    monkeypatch.setattr(compute, 'compute_scope', lambda scope, **kw: calls.append(('compute', scope, kw)) or {})
    monkeypatch.setattr(compute, 'rate_scope', lambda scope, **kw: calls.append(('ratings', scope, kw)) or {})
    module = run_load_module()
    module.compute_scope('es-md', save=False, verbose=2)
    module.rate_scope('es-md', save=True, verbose=0)
    assert calls == [('compute', 'es-md', {'save': False, 'verbose': 1}), ('ratings', 'es-md', {'save': True, 'verbose': 0})]


# --- Correcciones curadas de sondeos (`apply_poll_fixes`, data/polls/{scope}/{event_date}.csv) ----------

def poll_fix_frames():
    key = {'event_date': pd.Timestamp('2026-11-29'), 'event_scope': 'es'}
    celeste = {**key, 'date': pd.Timestamp('2026-09-21'), 'pollster_id': 7, 'sponsor_id': 3}
    socio = {**key, 'date': pd.Timestamp('2026-10-11'), 'pollster_id': 9, 'sponsor_id': 0}
    polls = pd.DataFrame([
        {**celeste, 'pollster': 'Celeste Tel', 'sponsor': 'Onda Cero', 'parties': 2},
        {**socio, 'pollster': 'SocioMétrica', 'sponsor': None, 'parties': 1},
    ])
    empty = {'seats': None, 'seats_min': None, 'seats_max': None}
    results = pd.DataFrame([
        {**celeste, 'party_id': 5, 'party': 'SUMAR', 'pct': 6.4, 'seats': 7, 'seats_min': 6, 'seats_max': 8},
        {**celeste, 'party_id': 1, 'party': 'PP', 'pct': 33.0, **empty},
        {**socio, 'party_id': 5, 'party': 'SUMAR', 'pct': 5.8, **empty},
    ])
    ids = ({'SUMAR': 5, 'COMPROMIS': 401, 'PP': 1}, {'Celeste Tel': 7, 'SocioMétrica': 9}, {'Onda Cero': 3, 'El Español': 4})
    return polls, results, ids


def poll_fixes(rows):
    return pd.DataFrame(rows, columns=['pollster', 'date', 'sponsor', 'party', 'pct', 'seats', 'note'])


def test_poll_fixes_set_and_create_rows():
    polls, results, ids = poll_fix_frames()
    fixes = poll_fixes([
        ['Celeste Tel', '2026-09-21', 'Onda Cero', 'SUMAR', 5.1, None, 'Compromís aparte'],
        ['Celeste Tel', '2026-09-21', 'Onda Cero', 'COMPROMIS', 1.3, 3, None],
        ['SocioMétrica', '2026-10-11', None, 'SUMAR', 4.9, None, None],
        ['SocioMétrica', '2026-10-11', None, 'COMPROMIS', 0.9, 3, None],
    ])
    out_polls, out_results, missing = apply_poll_fixes(polls, results, fixes, *ids)
    assert missing == []
    rows = out_results.set_index(['pollster_id', 'party'])
    assert rows.loc[(7, 'SUMAR'), ['pct', 'seats', 'seats_min', 'seats_max']].tolist() == [5.1, 7, 6, 8]
    assert rows.loc[(7, 'COMPROMIS'), ['pct', 'seats', 'seats_min', 'seats_max', 'party_id']].tolist() == [1.3, 3, 3, 3, 401]
    assert rows.loc[(7, 'COMPROMIS'), ['event_scope', 'sponsor_id']].tolist() == ['es', 3]
    assert rows.loc[(9, 'SUMAR'), 'pct'] == 4.9 and rows.loc[(9, 'COMPROMIS'), 'pct'] == 0.9
    assert out_polls.set_index('pollster_id')['parties'].to_dict() == {7: 3, 9: 2}
    assert results.shape[0] == 3 and polls['parties'].tolist() == [2, 1]  # entradas intactas


def test_poll_fixes_remove_a_party_and_report_unread_polls():
    polls, results, ids = poll_fix_frames()
    fixes = poll_fixes([
        ['Celeste Tel', '2026-09-21', 'Onda Cero', 'PP', None, None, 'fuera'],
        ['SocioMétrica', '2026-10-11', None, 'COMPROMIS', None, None, 'no estaba: nada que quitar'],
        ['Celeste Tel', '2026-10-30', 'Onda Cero', 'SUMAR', 5.0, None, 'sondeo no leído'],
    ])
    out_polls, out_results, missing = apply_poll_fixes(polls, results, fixes, *ids)
    assert missing == ['Celeste Tel 2026-10-30']
    assert out_results['party'].tolist() == ['SUMAR', 'SUMAR']
    assert out_polls.set_index('pollster_id')['parties'].to_dict() == {7: 1, 9: 1}


@pytest.mark.parametrize('row, match', [
    (['Celeste Tel', '2026-09-21', 'Onda Cero', 'COMPROMÍS', 1.3, 3, None], 'party'),
    (['Celeste-Tel', '2026-09-21', 'Onda Cero', 'SUMAR', 5.1, None, None], 'pollster'),
    (['Celeste Tel', '2026-09-21', 'Onda Zero', 'SUMAR', 5.1, None, None], 'sponsor'),
])
def test_poll_fixes_reject_unknown_names(row, match):
    polls, results, ids = poll_fix_frames()
    with pytest.raises(ValueError, match=match):
        apply_poll_fixes(polls, results, poll_fixes([row]), *ids)


def test_poll_fixes_reject_duplicates():
    polls, results, ids = poll_fix_frames()
    fixes = poll_fixes([['Celeste Tel', '2026-09-21', 'Onda Cero', 'SUMAR', 5.1, None, None]] * 2)
    with pytest.raises(ValueError, match='duplicate'):
        apply_poll_fixes(polls, results, fixes, *ids)


def test_build_series_applies_the_curated_poll_fixes(monkeypatch):
    import types
    from mtpy.lib import loader as loader_module

    loader = bare_loader('es')
    loader.event_date, loader.params, loader.exclude, loader.verbose = '2026-11-29', {}, None, 0
    loader.pollsters = pd.DataFrame({'id': [7], 'mtype': ['cati']})
    loader.m_polls = types.SimpleNamespace(key=['event_date', 'event_scope', 'date', 'pollster_id', 'sponsor_id'],
                                           format_data=lambda df, **kwargs: df)
    loader.m_results = types.SimpleNamespace(key=loader.m_polls.key + ['party_id'], format_data=lambda df, **kwargs: df)
    loader.data = [{
        'pollster_id': 7, 'pollster': 'Celeste Tel', 'sponsor_id': 3, 'sponsor': 'Onda Cero', 'name': 'Celeste Tel / Onda Cero',
        'date': pd.Timestamp('2026-09-21'), 'start_date': pd.Timestamp('2026-09-16'), 'end_date': pd.Timestamp('2026-09-21'),
        'sample_size': 1100, 'ctype': None,
        'results': [{'party_id': 5, 'party': 'SUMAR', 'pct': 6.4, 'seats': None, 'seats_min': None, 'seats_max': None}],
    }]
    fixes = pd.DataFrame({'pollster': ['Celeste Tel'], 'date': ['2026-09-21'], 'sponsor': ['Onda Cero'],
                          'party': ['SUMAR'], 'pct': [5.1], 'seats': [None], 'note': [None]})
    names = []
    loader.app = types.SimpleNamespace(data=types.SimpleNamespace(
        exists=lambda name: names.append(name) or name == 'polls/es/2026-11-29.csv', read_csv=lambda name: fixes))
    calls = []

    def fake_apply(polls, results, fixes, parties, pollsters, sponsors):
        calls.append((fixes, parties, pollsters, sponsors))
        return polls, results, ['Celeste Tel 2026-10-30']

    monkeypatch.setattr(loader_module, 'apply_poll_fixes', fake_apply)
    loader.build_series()

    assert names == ['polls/es/2026-11-29.csv']
    assert calls[0][0] is fixes and calls[0][1] is loader.parties_idmap
    assert loader.fixes_missing == ['Celeste Tel 2026-10-30']


def test_poll_fixes_new_row_without_seats_keeps_the_frame_clean():
    # Una casa que da cuota a Compromís sin escaños: fila nueva con escaños vacíos, sin avisos de pandas
    # (pytest.ini convierte FutureWarning en error) y con la columna de escaños utilizable
    polls, results, ids = poll_fix_frames()
    fixes = poll_fixes([['Celeste Tel', '2026-09-21', 'Onda Cero', 'COMPROMIS', 1.3, None, None]])
    out_polls, out_results, missing = apply_poll_fixes(polls, results, fixes, *ids)
    row = out_results.set_index(['pollster_id', 'party']).loc[(7, 'COMPROMIS')]
    assert row['pct'] == 1.3 and pd.isnull(row['seats']) and pd.isnull(row['seats_min']) and pd.isnull(row['seats_max'])
    assert out_results['seats'].dtype == results['seats'].dtype
    assert out_polls.set_index('pollster_id')['parties'].to_dict() == {7: 3, 9: 1}


def fake_cols(table, pollster, sponsor):
    """Cells of the first poll row of `table` with the first one rewritten as `pollster/sponsor`."""
    for row in table.xpath('.//tr'):
        cells = row.xpath('./td')
        last = ''.join(cells[-1].xpath('.//text()')).strip() if cells else ''
        if len(cells) > 3 and not row.xpath('./td[1]/b') and is_number(last):
            cols = copy.deepcopy(cells)
            for node in cols[0].xpath('.//text()'):
                parent = node.getparent()
                if node.is_text:
                    parent.text = ''
                else:
                    parent.tail = ''
            cols[0].text = f'{pollster}/{sponsor}'
            return cols
    raise AssertionError('no poll row in the fixture')


def test_discard_rules_for_pollsters_and_sponsors():
    from mtpy.lib.loader import discarded_pollster, discarded_sponsor, normalized
    parties = {normalized('Sumar'), normalized('Partido Popular'), normalized('IMC')}
    assert discarded_pollster('CIS (Sondaxe)', [], parties) and discarded_pollster('CIS (Demócrata)', [], parties)
    assert not discarded_pollster('CIS', [], parties)
    assert discarded_pollster('SUMAR', [], parties) and discarded_pollster('partido popular', [], parties)
    assert discarded_pollster('PSPV', ['PSPV'], parties) and not discarded_pollster('GAD3', ['PSPV'], parties)
    assert discarded_sponsor('Sumar', [], parties) and discarded_sponsor('PSN', ['PSN'], parties)
    assert not discarded_sponsor('El Mundo', ['PSN'], parties)


def test_read_cols_skips_discarded_names_only_when_they_have_no_id():
    """A pollster without id named like a party or «CIS (...)» is dropped silently; with an id it keeps its poll
    (IMC); a sponsor without id named like a party enters with sponsor_id 0 and no warning."""
    maps = {'parties': {}, 'pollsters': {}, 'sponsors': {}, 'scopes': {},
            'skip': {'pollsters': ['PSPV'], 'sponsors': ['PSN']}}
    loader = bare_loader('es', maps=maps, party_names=['Sumar', 'IMC'])
    loader.pollsters_idmap = {'IMC': 110, 'GAD3': 1}
    loader.sponsors_idmap = {'El Mundo': 5}
    table = WikipediaLoader.find_poll_table(load('2023_Madrilenian_regional_election'))
    parties, lead = WikipediaLoader.read_header(table)
    rows = {}
    for name in ('IMC', 'Sumar', 'CIS (Sondaxe)', 'PSPV', 'GAD3'):
        cols = fake_cols(table, pollster=name, sponsor='Sumar' if name == 'GAD3' else 'El Mundo')
        rows[name] = loader.read_cols(cols, parties, lead, '2023')
    assert rows['IMC']['pollster_id'] == 110 and rows['GAD3']['pollster_id'] == 1
    assert rows['Sumar'] == {} and rows['CIS (Sondaxe)'] == {} and rows['PSPV'] == {}
    assert rows['GAD3']['sponsor_id'] == 0
    assert loader.pollsters_missing == [] and loader.sponsors_missing == []
    assert sorted(loader.pollsters_skipped) == ['CIS (Sondaxe)', 'PSPV', 'Sumar']
    assert loader.sponsors_skipped == ['Sumar']
