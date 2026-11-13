"""«Escaños que bailan», la tarjeta de la página del evento (`/{scope}`, sección `#escanos`): de la parte `seat-odds`
a las opciones de escaño de cada circunscripción, el mapa, las fichas y las columnas; y las direcciones de la página
aparte que hubo (`/{scope}/circunscripciones`), que llevan a la tarjeta."""
import copy
import os
import re

import pytest

from mtpy.lib import bundle, pages, webapi
from tests.test_api_asgi import call
from tests.test_pages import api, assert_public_wording, counting, html, visible_text  # noqa: F401
from tests.test_webapi_unit import RUN, fixture_data, write_bundle


def odds_row(region_id, name, seats, *reach):
    return {'region_id': region_id, 'name': name, 'seats': seats, 'reach': list(reach)}


ODDS = {
    'n_sim': 10000, 'parties': ['PP', 'PSOE', 'VOX', 'AA', 'BNG', 'UP'],
    'regions': [{'id': 11, 'name': 'Cádiz', 'seats': 8}, {'id': 19, 'name': 'Guadalajara', 'seats': 3},
                {'id': 15, 'name': 'La Coruña', 'seats': 8}],
    'rows': [
        odds_row(11, 'PP', 3, 10000, 10000, 8010, 730, 10), odds_row(11, 'PSOE', 2, 10000, 9950, 4110, 80),
        odds_row(11, 'VOX', 2, 10000, 8530, 570), odds_row(11, 'AA', 1, 7580),
        odds_row(11, 'UP', 0, 2400),                      # never reaches 25%: no option anywhere
        odds_row(19, 'PP', 1, 10000, 60), odds_row(19, 'PSOE', 1, 9990), odds_row(19, 'VOX', 1, 9950),
        odds_row(15, 'PP', 3, 10000, 10000, 9900, 3000),  # the third, exactly at 99%: safe; the fourth, in play
        odds_row(15, 'PSOE', 2, 10000, 9899, 2499),       # 98,99%: likely, labelled 98%; 24,99%: not drawn
        odds_row(15, 'VOX', 1, 3657),
        odds_row(15, 'BNG', 2, 10000, 9950, 2500, 99),    # the third, exactly at 25%: in play
    ],
}


def by_name(district):
    return {row['name']: row for row in district['rows']}


def test_states_follow_the_three_thresholds_to_the_exact_count():
    edges = pages.seat_tokens('X', [9900, 9899, 7500, 7499, 2500, 2499], 10000)
    assert [(t['k'], t['state'], t['label'], t['alpha']) for t in edges] == [
        (1, 'safe', None, None), (2, 'likely', '98%', 0.68), (3, 'likely', '75%', 0.42),
        (4, 'play', '75%', 0.26), (5, 'play', '25%', 0.06)]
    assert [pages.seat_state(share) for share in (1.0, 0.99, 0.9899, 0.75, 0.7499, 0.25, 0.2499, 0.0)] == [
        'safe', 'safe', 'likely', 'likely', 'play', 'play', None, None]
    assert (pages.token_alpha('likely', 0.801), pages.token_alpha('play', 0.5), pages.token_alpha('safe', 1.0)) == (
        0.475, 0.16, None)
    assert edges[0]['title'] == '1.º de X: seguro'
    assert edges[3]['title'] == '4.º de X: lo consigue en el 75% de las simulaciones'
    assert pages.option_title(4, 'PSOE', '51%', 'Alicante') == (
        '4.º de PSOE en Alicante: lo consigue en el 51% de las simulaciones')
    assert pages.option_title(1, 'PSOE', None, 'Cádiz') == '1.º de PSOE en Cádiz: seguro'
    assert pages.seat_tokens('X', [], 10000) == [] and 'central' not in edges[0] and not hasattr(pages, 'OPEN_FLOOR')


@pytest.mark.parametrize('share, label', [(0.9899, '98%'), (0.985, '98%'), (0.805, '81%'), (0.5, '50%'),
                                           (0.25, '25%'), (0.2549, '25%'),
                                           # exact halves that the binary floats put a hair below: rounded up too
                                           (0.285, '29%'), (0.565, '57%'), (0.575, '58%'),
                                           (2850 / 10000, '29%'), (5650 / 10000, '57%'), (5750 / 10000, '58%')])
def test_token_label_is_a_whole_percentage_between_25_and_98(share, label):
    assert pages.token_label(share) == label


def test_district_views_count_the_options_of_each_state():
    cadiz, guadalajara, coruna = pages.district_views(ODDS)
    assert (cadiz['name'], cadiz['slug'], cadiz['seats']) == ('Cádiz', 'cadiz', 8) and coruna['slug'] == 'la-coruna'
    assert [(d['safe'], d['likely'], d['play']) for d in (cadiz, guadalajara, coruna)] == [(5, 3, 1), (3, 0, 0), (6, 1, 3)]
    # only the parties with some option (UP has none), by the expected seats among the drawn ones
    assert [row['name'] for row in cadiz['rows']] == ['PP', 'PSOE', 'VOX', 'AA']
    assert [row['name'] for row in coruna['rows']] == ['PP', 'BNG', 'PSOE', 'VOX']
    assert [(r['safe'], r['likely'], r['play']) for r in cadiz['rows']] == [(2, 1, 0), (2, 0, 1), (1, 1, 0), (0, 1, 0)]
    assert by_name(cadiz)['PSOE']['expected'] == pytest.approx(2.406) and 'seats' not in by_name(cadiz)['PSOE']


def test_the_default_district_the_totals_and_the_foot_sentence():
    districts = pages.district_views(ODDS)
    assert pages.default_district(districts)['name'] == 'La Coruña'  # three options in play
    tie = [{'name': 'Zamora', 'play': 1, 'seats': 3}, {'name': 'Ávila', 'play': 1, 'seats': 3},
           {'name': 'Soria', 'play': 1, 'seats': 2}]
    assert pages.default_district(tie)['name'] == 'Ávila'
    assert pages.default_district(tie + [{'name': 'Madrid', 'play': 1, 'seats': 38}])['name'] == 'Madrid'
    assert pages.scope_totals(districts) == {'seats': 19, 'safe': 14, 'likely': 4, 'play': 4, 'options': 22}
    assert pages.options_sentence(22, 19) == ('Hay 22 opciones para 19 escaños: a un mismo escaño pueden optar '
                                              'varios partidos.')
    assert pages.options_sentence(19, 19) == '' and pages.options_sentence(17, 19) == ''


def test_the_texts_of_the_states():
    districts = pages.district_views(ODDS)
    assert [pages.state_text(s, n) for s, n in [('safe', 1), ('safe', 5), ('likely', 1), ('likely', 3), ('play', 1),
                                                ('play', 2)]] == ['1 seguro', '5 seguros', '1 probable',
                                                                  '3 probables', '1 en juego', '2 en juego']
    cadiz = by_name(districts[0])
    assert pages.tokens_note(cadiz['PP']) == '2 seguros · 1 probable' and pages.tokens_note(cadiz['AA']) == '1 probable'
    assert pages.tokens_reading('PP', cadiz['PP']) == 'PP: 2 seguros; 3.º, 80%'
    # the list of the closest seats («Los más reñidos», then «Los más disputados») left the page (Luis, 2026-11-13)
    assert not [name for name in ('closest_seats', 'closest_sentence', 'CLOSEST_SHOWN') if hasattr(pages, name)]


def sentence_of(seats, safe_by_party, possible):
    """The sentence of a district with these safe seats by party and these parties with some option not safe."""
    safe = sum(safe_by_party.values())
    names = list(safe_by_party) + [name for name in possible if name not in safe_by_party]
    rows = [{'name': name, 'safe': safe_by_party.get(name, 0), 'likely': 0, 'play': int(name in possible)}
            for name in names]
    return pages.district_sentence({'seats': seats, 'safe': safe, 'rows': rows})


@pytest.mark.parametrize('seats, safe, possible, text', [
    (1, {'PP': 1}, [], 'El escaño es seguro (PP 1).'),
    (3, {'PP': 2, 'PSOE': 1}, [], 'Los 3 escaños son seguros (PP 2, PSOE 1).'),
    (4, {'PP': 1, 'VOX': 0}, ['PP', 'VOX', 'PSOE'],
     'De los 4 escaños, 1 es seguro (PP 1). Los otros 3 se los disputan PP, VOX y PSOE.'),
    (3, {'PP': 1, 'PSOE': 1}, ['PP', 'PSOE'], 'De los 3 escaños, 2 son seguros (PP 1, PSOE 1). El otro se lo disputan PP y PSOE.'),
    (3, {'PP': 1, 'PSOE': 1}, ['PSOE'], 'De los 3 escaños, 2 son seguros (PP 1, PSOE 1). Para el otro, quien más opciones tiene es PSOE.'),
    (4, {'PP': 1, 'PSOE': 1}, ['PP'], 'De los 4 escaños, 2 son seguros (PP 1, PSOE 1). Para los otros 2, quien más opciones tiene es PP.'),
    (3, {'PP': 1, 'PSOE': 1}, [], 'De los 3 escaños, 2 son seguros (PP 1, PSOE 1).'),
    (4, {'PP': 0}, ['PP', 'PSOE', 'IU'], 'Ninguno de los 4 escaños es seguro: se los disputan PP, PSOE e IU.'),
    (4, {'PP': 0}, ['PP'], 'Ninguno de los 4 escaños es seguro: quien más opciones tiene es PP.'),
    (4, {'PP': 0}, [], 'Ninguno de los 4 escaños es seguro.'),
    (1, {'PP': 0}, ['PP', 'PSOE'], 'El escaño no es seguro: se lo disputan PP y PSOE.'),
    (1, {'PP': 0}, ['PP'], 'El escaño no es seguro: quien más opciones tiene es PP.'),
    (1, {'PP': 0}, [], 'El escaño no es seguro.'),
    (1, {}, [], 'El escaño no es seguro.'),  # a district without any option
    (2, {'A': 3}, [], 'Los 2 escaños son seguros (A 3).'),  # more safe options than seats: never below zero
])
def test_district_sentence_names_the_parties_that_dispute_the_seats_not_safe(seats, safe, possible, text):
    # the seats nobody has safe go to the parties with some likely option or some option in play
    assert sentence_of(seats, safe, possible) == text


def write_odds(fs, data=ODDS, modes=bundle.MODES):
    """Sobrescribe la parte `seat-odds` del run del bundle de prueba."""
    writer = bundle.BundleWriter(fs)
    for mode in modes:
        writer.write_json(bundle.path_part('es', RUN, 'seat-odds', mode), 'seat-odds', data, 'es', run_id=RUN, mode=mode)
    webapi.site().files.clear()
    webapi.site().manifest_cache_clear()  # the presence of the part is remembered under the TTL of the pointers


@pytest.fixture
def odds_site(api, fresh_app):
    write_odds(fresh_app.fs)
    return api


RETIRED = ('posible', 'en el aire', 'aspirante', 'reparto central', 'Central')


def test_only_ine_province_codes_make_the_districts_provinces():
    # the catalogue keeps the INE code of a province as its id and numbers every other district from 100
    provinces = [{'id': 11, 'name': 'Cádiz'}, {'id': 52, 'name': 'Melilla'}]
    assert pages.districts_are_provinces(provinces)
    assert not pages.districts_are_provinces(provinces + [{'id': 100, 'name': 'Lista autonómica'}])
    assert not pages.districts_are_provinces([{'id': 101, 'name': 'Central'}, {'id': 102, 'name': 'Occidental'}])


def test_the_positions_of_the_map_come_from_the_catalogue(api):
    positions = webapi.district_positions('es')
    assert len(positions) == 52 and positions[31] == {'abbr': 'NA', 'col': 6, 'row': 0, 'w': 1, 'h': 1}
    assert positions[28] == {'abbr': 'M', 'col': 4, 'row': 3, 'w': 2, 'h': 2} and webapi.district_positions('es-an') == {}
    assert all(type(positions[28][key]) is int for key in ('col', 'row', 'w', 'h'))


def test_the_texts_of_a_row_and_its_drawing():
    assert pages.tokens_note({'safe': 1, 'likely': 3, 'play': 2}) == '1 seguro · 3 probables · 2 en juego'
    assert pages.tokens_note({'safe': 0, 'likely': 0, 'play': 1}) == '1 en juego'
    assert pages.tokens_note({'safe': 1, 'likely': 0, 'play': 0}) == '1 seguro'
    assert pages.tokens_note({'safe': 0, 'likely': 0, 'play': 0}) == '—'
    row = by_name(pages.district_views(ODDS)[1])['PP']  # Guadalajara: one safe seat, the second below 25%
    assert pages.tokens_reading('PP', row) == 'PP: 1 seguro' and len(row['tokens']) == 1
    assert pages.tokens_reading('VOX', by_name(pages.district_views(ODDS)[2])['VOX']) == 'VOX: 1.º, 37%'
    assert [pages.threshold_label(share) for share in (pages.SAFE_SHARE, pages.LIKELY_SHARE, pages.PLAY_SHARE)] == [
        '99%', '75%', '25%'] and pages.STATES == ('safe', 'likely', 'play')
    assert pages.tint('#1d84ce', 0.2) == 'rgba(29, 132, 206, 0.2)' and pages.tint('#fff', 0.5) == 'rgba(255, 255, 255, 0.5)'
    assert pages.tint(None, 0.5) == 'rgba(158, 158, 158, 0.5)'  # the grey of the others
    drawn = pages.drawn_row(by_name(pages.district_views(ODDS)[0])['PP'], 'PP', '#1d84ce')
    assert [token['fill'] for token in drawn['tokens']] == [None, None, 'rgba(29, 132, 206, 0.475)']
    assert (drawn['color'], drawn['reading'], drawn['note']) == (
        '#1d84ce', 'PP: 2 seguros; 3.º, 80%', '2 seguros · 1 probable')
    token = pages.seat_tokens('PP', [7580], 10000)[0]
    assert pages.drawn_token(token, '#1d84ce') == dict(token, color='#1d84ce', fill=pages.tint('#1d84ce', token['alpha']))
    safe = pages.seat_tokens('PP', [10000], 10000)[0]
    assert pages.drawn_token(safe, '#1d84ce') == dict(safe, color='#1d84ce', fill=None) and 'color' not in safe


def test_the_event_bar_of_a_subpage_links_the_event_page_and_drops_the_seats_section_without_districts(api, tmp_path):
    """The helper of the page of a poll (`sondeos`), whose tests do not reach a run without districts."""
    context = pages.common_context({'scope': 'es', 'mode': 'nowcast', 'run': RUN}, None, event=True)
    bar = pages.subpage_event_bar(context)
    query = '?mode=nowcast&run=' + RUN
    assert [entry['href'] for entry in bar['sections']] == [
        '/es' + query + '#' + key for key in ('promedio', 'resultados', 'escanos', 'evolucion', 'tabla')]
    assert context['event_bar']['sections'][0]['href'] == '#promedio'  # a copy: the bar of the context is untouched
    os.remove(os.path.join(str(tmp_path), bundle.PREFIX, bundle.path_part('es', RUN, 'districts', 'nowcast')))
    webapi.site().files.clear()
    assert [entry['id'] for entry in pages.subpage_event_bar(context)['sections']] == [
        'promedio', 'resultados', 'evolucion', 'tabla']


@pytest.mark.parametrize('path, query, location', [
    ('/es/circunscripciones', '', '/es#escanos'),
    ('/es/circunscripciones/cadiz', 'mode=nowcast', '/es?mode=nowcast#escanos'),
    ('/es/circunscripciones', 'partido=psoe&vista=partidos&circunscripcion=cadiz&mode=nowcast&run=' + RUN,
     '/es?mode=nowcast&run=' + RUN + '#escanos'),
    ('/es/circunscripciones', 'vista=renidos', '/es#escanos'),
])
def test_the_addresses_of_the_page_that_was_go_to_the_card(odds_site, path, query, location):
    status, headers, _ = call(odds_site, path, query=query)
    assert status == 301 and headers['location'] == location


def test_the_addresses_of_the_page_that_was_need_a_published_scope_and_not_the_part(api, fresh_app, tmp_path):
    # a scope outside the catalogue, or in it without a published forecast: the 404 of the event page
    for path in ('/es-zz/circunscripciones', '/es-md/circunscripciones', '/es-md/circunscripciones/madrid'):
        status, headers, body = call(api, path)
        assert status == 404 and headers['content-type'].startswith('text/html') and 'location' not in headers, path
    # a district that does not exist, or a run without the part: the event page, which says what it has
    remove_odds(tmp_path)
    for path in ('/es/circunscripciones', '/es/circunscripciones/no-existe'):
        status, headers, _ = call(api, path, query='mode=nowcast')
        assert status == 301 and headers['location'] == '/es?mode=nowcast#escanos', path


def test_nothing_of_the_page_that_was_is_left():
    gone = ('seat_odds_context', 'seat_odds_links', 'district_href', 'party_view', 'party_sentence', 'hemicycle_layout')
    assert not [name for name in gone if hasattr(pages, name)]
    css = site_css()
    assert not [rule for rule in ('.so-hemi', '.so-tabs', '.so-pills', '.so-by-party', '.so-settled', '#vistas',
                                  '#partidos', '#grafico', '#mapa', '#provincias') if rule in css]
    root = os.path.join(os.path.dirname(__file__), '..')
    assert not os.path.exists(os.path.join(root, 'web', 'templates', 'circunscripciones.html'))
    assert not os.path.exists(os.path.join(root, 'web', 'dist', 'js', 'pages', 'circunscripciones.js'))


DISTRICTS = {  # the districts of `ODDS` in the `districts` part, so that the detail of the event page shows one of them
    'parties': ['PP'],
    'regions': [dict(region, community=None, votes=1000) for region in ODDS['regions']],
    'rows': [{'region_id': region['id'], 'region': region['name'], 'name': 'PP', 'pct': 30.0, 'pct_lo': 27.0,
              'pct_hi': 33.0, 'seats': 3, 'seats_median': 3.0, 'seats_mean': 3.1, 'seats_lo': 2, 'seats_hi': 4,
              'p_seats': 1.0} for region in ODDS['regions']],
}


def write_event_districts(fs, districts=DISTRICTS):
    """Sobrescribe la parte `districts` del run en los dos modos, para el detalle de la página del evento."""
    writer = bundle.BundleWriter(fs)
    for mode in bundle.MODES:
        writer.write_json(bundle.path_part('es', RUN, 'districts', mode), 'districts', districts, 'es', run_id=RUN, mode=mode)
    webapi.site().files.clear()


def remove_odds(tmp_path):
    """Quita la parte `seat-odds` del run en los dos modos (un run publicado antes de que existiera)."""
    for mode in bundle.MODES:
        os.remove(os.path.join(str(tmp_path), bundle.PREFIX, bundle.path_part('es', RUN, 'seat-odds', mode)))
    webapi.site().files.clear()
    webapi.site().manifest_cache_clear()  # the presence of the part is remembered under the TTL of the pointers


def event_page(site, query='mode=forecast'):
    status, _, body = call(site, '/es', query=query)
    return status, re.sub(r'\s+', ' ', html(body))


def seat_odds_card(page):
    """La tarjeta «Escaños que bailan» de una página del evento ya colapsada: de su `<section>` a su cierre."""
    start = page.index('<section class="card shot so-card" id="seat-odds"')
    return page[start:page.index('</section>', start)]


def test_the_event_page_carries_the_card_of_the_seats_that_dance(odds_site, fresh_app):
    write_event_districts(fresh_app.fs)
    status, page = event_page(odds_site)
    section = page[page.index('<section class="event-section" id="escanos"'):page.index('id="evolucion"')]
    assert status == 200 and section.index('id="seat-odds"') < section.index('id="districts"')
    card = section[section.index('id="seat-odds"'):section.index('id="districts"')]
    assert 'Escaños que bailan' in card and 'según las 10.000 simulaciones del modelo.' in card
    assert re.findall(r'name="sv" value="(map|cols)"[^>]*?( checked)?>([^<]+)<', card) == [
        ('map', ' checked', 'Provincias'), ('cols', '', 'Partidos')]
    assert re.findall(r'name="d" value="([a-z-]+)"', card) == ['la-coruna', 'guadalajara', 'cadiz']
    assert re.findall(r'name="d" value="([a-z-]+)"[^>]* checked', card) == ['la-coruna']  # most options in play
    assert re.findall(r'class="so-ficha" id="ficha-([a-z-]+)"', card) == ['cadiz', 'guadalajara', 'la-coruna']
    assert re.findall(r'name="mp" value="([a-z-]+)"', card) == ['all', 'pp', 'psoe', 'vox', 'aa', 'bng']
    assert re.findall(r'class="so-col" data-col="([^"]+)"', card) == ['PP', 'PSOE', 'VOX', 'AA', 'BNG']
    assert 'Hay 22 opciones para 19 escaños' in card and card.count('<details class="how"') == 1
    assert 'De los 8 escaños, 5 son seguros (PP 2, PSOE 2, VOX 1).' in card
    # the rules of the run, in the head: one per party for the filter, one per district for its card
    assert '#seat-odds:has(#mp-psoe:checked) .so-dots i:not(.p-psoe){opacity:.08}' in page
    assert '#seat-odds:has(#d-cadiz:checked) #ficha-cadiz{display:block}' in page
    # its CSV among the downloads of the page
    downloads = page[page.index('id="tabla"'):]
    assert ('<a href="/api/v1/forecast/es/forecast/seat-odds?run={}&amp;format=csv">Escaños uno a uno por '
            'circunscripción</a>').format(RUN) in downloads
    # what did not come along, and what it replaces
    gone = ('so-hemi', 'so-pills', 'id="partidos"', '/circunscripciones', 'region-odds', 'odds_hrefs', 'reñid')
    assert not [piece for piece in gone if piece in page]
    text = visible_text(card)
    assert not [word for word in RETIRED + ('disputados',) if word in text]
    assert_public_wording(page, 'evento con la tarjeta')
    # nothing of the card collides with the rest of the page
    ids = re.findall(r'\sid="([^"]+)"', page)
    assert len(ids) == len(set(ids)), [name for name in set(ids) if ids.count(name) > 1]
    for name in ('sv', 'mp', 'ms', 'd'):
        assert page.count('name="{}"'.format(name)) == card.count('name="{}"'.format(name)) > 0


def test_the_card_speaks_of_three_states_with_their_exact_marks(odds_site):
    card = seat_odds_card(event_page(odds_site)[1])
    cadiz = card[card.index('id="ficha-cadiz"'):card.index('id="ficha-guadalajara"')]
    for chip in ('<span class="chip so-chip safe"><b>5</b> seguros</span>',
                 '<span class="chip so-chip likely"><b>3</b> probables</span>',
                 '<span class="chip so-chip play"><b>1</b> en juego</span>'):
        assert chip in cadiz
    assert ('<i class="so-o likely" style="--c: #1d84ce; --f: rgba(29, 132, 206, 0.475)" '
            'title="3.º de PP: lo consigue en el 80% de las simulaciones"></i>') in cadiz
    assert '<span class="so-pct">80%</span>' in cadiz and '>7%<' not in card  # under 25%: neither drawn nor counted
    assert 'aria-label="PP: 2 seguros; 3.º, 80%"' in cadiz and '<div class="so-note">2 seguros · 1 probable</div>' in cadiz
    assert 'De los 8 escaños, 5 son seguros (PP 2, PSOE 2, VOX 1). Los otros 3 se los disputan PP, PSOE, VOX y AA.' in cadiz
    # one meaning per word: the legend says the three states with their bands, each with its own sample circle
    start = card.index('class="so-legend"')
    legend = card[start:card.index('</div>', start)]
    assert re.findall(r'<i class="so-o (safe|likely|play)" aria-hidden="true"></i>([^<]+)</span>', legend) == [
        ('safe', 'Seguro: 99% o más'), ('likely', 'Probable: del 75% al 99%'), ('play', 'En juego: del 25% al 75%')]
    assert 'probable si lo consigue entre el 75% y el 99% y está en juego entre el 25% y el 75%' in card


def test_the_map_of_the_card_places_every_district_and_counts_the_options_of_each_state(odds_site):
    card = seat_odds_card(event_page(odds_site)[1])
    # the radio group of the tiles is named as the view of the map
    assert '<div class="so-map" role="radiogroup" aria-label="Provincias">' in card
    grid = card[card.index('class="so-map"'):card.index('class="so-legend"')]
    assert 'title="Cádiz · 8 escaños" style="grid-column: 4 / span 1; grid-row: 9 / span 1"' in grid
    assert '<span class="so-ab">CA</span>' in grid and 'aria-label="Cádiz, 8 escaños"' in grid
    assert grid.count('class="so-o ') == 22 and grid.count('class="so-o play p-') == 4
    assert '<i class="so-o safe p-pp" style="--c: #1d84ce" title="1.º de PP: seguro"></i>' in grid
    states = card[card.index('id="map-state"'):card.index('</fieldset>', card.index('id="map-state"'))]
    assert re.findall(r'>([^<>]+) <small>(\d+)</small>', states) == [
        ('Todos', '22'), ('Seguros', '14'), ('Probables', '4'), ('En juego', '4')]
    assert '<span class="so-hint">Toca una provincia para ver su detalle.</span>' in card


def test_the_map_without_positions_and_the_foot_without_more_options_than_seats(odds_site, fresh_app, monkeypatch):
    monkeypatch.setattr('mtpy.lib.pages.seat_odds.district_positions', lambda scope: {})
    card = seat_odds_card(event_page(odds_site)[1])
    loose = card[card.index('class="so-map so-map-loose"'):card.index('class="so-legend"')]
    assert re.findall(r'<span class="so-ab">([^<]+)</span>', loose) == ['Cádiz', 'Guadalajara', 'La Coruña']
    assert 'grid-column' not in loose
    monkeypatch.undo()
    safe_only = dict(ODDS, rows=[odds_row(row['region_id'], row['name'], row['seats'], *[n for n in row['reach'] if n >= 9900])
                                 for row in ODDS['rows']])
    write_odds(fresh_app.fs, safe_only)  # 14 options for 19 seats, and districts where nobody disputes what is left
    card = seat_odds_card(event_page(odds_site)[1])
    assert 'opciones para' not in card and 'De los 8 escaños, 5 son seguros (PP 2, PSOE 2, VOX 1).' in card
    assert 'se los disputan' not in card


def test_the_block_of_a_district_draws_a_circle_per_option_and_labels_only_those_not_safe(odds_site):
    card = seat_odds_card(event_page(odds_site)[1])
    block = card[card.index('id="ficha-la-coruna"'):card.index('class="so-view-cols"')]
    assert '<h3>La Coruña</h3>' in block
    labels = re.findall(r'<span class="so-pct">([^<]+)</span>', block)
    assert sorted(labels) == ['25%', '30%', '37%', '98%'] and not {'99%', '100%', '24%'} & set(labels)
    states = re.findall(r'<span class="so-token (safe|likely|play)"><i class="so-o \1" ', block)
    assert (states.count('safe'), states.count('likely'), states.count('play')) == (6, 1, 3)
    assert re.findall(r'<i class="so-o [a-z]+" [^>]*title="([^"]+)"', block)[:3] == [
        '1.º de PP: seguro', '2.º de PP: seguro', '3.º de PP: seguro']
    assert '<i class="so-o safe" style="--c: #1d84ce" title="1.º de PP: seguro"></i>' in block  # solid: no fill
    closed = card[card.index('id="ficha-guadalajara"'):card.index('id="ficha-la-coruna"')]
    assert 'Los 3 escaños son seguros (PP 1, PSOE 1, VOX 1).' in closed
    assert '<span class="chip so-chip safe"><b>3</b> seguros</span>' in closed and 'so-chip likely' not in closed


def test_the_card_follows_the_mode_of_the_event_page(odds_site, fresh_app):
    nowcast = copy.deepcopy(ODDS)
    nowcast['rows'][3] = odds_row(11, 'AA', 1, 2500)  # AA's seat in Cádiz: 76% in the election mode, 25% today
    write_odds(fresh_app.fs, nowcast, modes=('nowcast',))
    card = seat_odds_card(event_page(odds_site, '')[1])  # less than 60 days away: the forecast
    assert 'aria-label="AA: 1.º, 76%"' in card and 'aria-label="AA: 1.º, 25%"' not in card
    today = seat_odds_card(event_page(odds_site, 'mode=nowcast')[1])
    assert 'aria-label="AA: 1.º, 25%"' in today and 'aria-label="AA: 1.º, 76%"' not in today


OLD_RUN = '20261001-120000'


def test_a_pinned_run_shows_the_card_of_that_run(odds_site, fresh_app):
    write_bundle(fresh_app.fs, run_id=OLD_RUN, manifest=False)  # an older run; the latest is still `RUN`
    older = copy.deepcopy(ODDS)
    older['rows'][3] = odds_row(11, 'AA', 1, 2500)  # AA's seat in Cádiz: 76% in the latest run, 25% in the older one
    writer = bundle.BundleWriter(fresh_app.fs)
    for mode in bundle.MODES:
        writer.write_json(bundle.path_part('es', OLD_RUN, 'seat-odds', mode), 'seat-odds', older, 'es', run_id=OLD_RUN,
                          mode=mode)
    webapi.site().files.clear()
    webapi.site().manifest_cache_clear()
    latest = seat_odds_card(event_page(odds_site)[1])
    assert 'aria-label="AA: 1.º, 76%"' in latest and 'aria-label="AA: 1.º, 25%"' not in latest
    status, page = event_page(odds_site, 'mode=forecast&run=' + OLD_RUN)
    pinned = seat_odds_card(page)
    assert status == 200 and 'aria-label="AA: 1.º, 25%"' in pinned and 'aria-label="AA: 1.º, 76%"' not in pinned
    assert '/seat-odds?run={}&amp;format=csv'.format(OLD_RUN) in page  # and its CSV is that run's


def test_the_card_and_its_section_come_only_with_something_to_show(odds_site, api, fresh_app, tmp_path):
    # the part without a territory table (the test bundle has one: its districts go): the section with the card
    # alone, and its entry in the index
    write_event_districts(fresh_app.fs, {'parties': ['PP'], 'regions': [], 'rows': []})
    status, page = event_page(odds_site)
    section = page[page.index('<section class="event-section" id="escanos"'):page.index('id="evolucion"')]
    assert 'id="seat-odds"' in section and 'id="districts"' not in section and '<a href="#escanos"' in page
    # a territory table without the part (a pinned run older than the part): the table alone, no rule of the card
    write_event_districts(fresh_app.fs)
    remove_odds(tmp_path)
    status, page = event_page(api)
    assert status == 200 and 'id="districts"' in page and 'id="seat-odds"' not in page and '#seat-odds:has' not in page


def test_the_card_switches_its_views_and_its_districts_without_javascript():
    css = site_css()
    # the radio of the columns hides the map and its card; any other state hides the columns
    assert re.search(r'#seat-odds:has\(#sv-cols:checked\) \.so-view-map, '
                     r'#seat-odds:not\(:has\(#sv-cols:checked\)\) \.so-view-cols\s*\{\s*display:\s*none', css)
    # side by side from 1,200 px of window: the map up to 760 px, the card of the district beside it, without the
    # summary of each row
    wide = css[css.index('@media (min-width: 1200px)'):]
    wide = wide[:wide.index('\n}')]
    assert re.search(r'\.so-split \{[^}]*display:\s*grid;[^}]*grid-template-columns:\s*minmax\(0, 760px\) '
                     r'minmax\(0, 1fr\)', wide)
    assert re.search(r'\.so-split \.so-note \{\s*display:\s*none', wide)
    # the filter of states, one rule per state; every card hidden but the one of the tile checked
    for state in ('safe', 'likely', 'play'):
        assert re.search(r'#seat-odds:has\(#ms-{0}:checked\) \.so-dots i:not\(\.{0}\)\s*\{{\s*opacity:\s*\.08'.format(state),
                         css)
    assert re.search(r'(^|\n)\.so-ficha\s*\{\s*display:\s*none', css)
    # without `:has()` every card and both views show, and the selector of views, which would do nothing, goes
    fallback = css[css.index('@supports not selector(:has(*))', css.index('.so-ficha')):]
    fallback = fallback[:fallback.index('\n}')]
    assert re.search(r'\.so-ficha\s*\{\s*display:\s*block', fallback) and re.search(r'#so-view\s*\{\s*display:\s*none',
                                                                                   fallback)
    assert 'so-view-map' not in fallback and 'so-view-cols' not in fallback  # hidden only through `:has()`
    # the dropdown of districts only under 720 px, and only once the script reveals it
    assert re.search(r'#seat-odds #district-select\s*\{\s*display:\s*none', css)
    narrow = css[css.index('@media (max-width: 719px)', css.index('#seat-odds #district-select')):]
    assert re.search(r'#seat-odds #district-select:not\(\[hidden\]\)\s*\{\s*display:\s*block', narrow)


def test_the_block_of_a_district_lands_below_the_header_and_below_the_banner_only_where_it_sticks():
    # the page script scrolls a block into view: below the sticky header (56 px) and the freeze banner when it shows;
    # on phones the banner is not sticky and the header is 52 px, so the block keeps its 68 px with the banner too
    css = site_css()
    assert re.search(r'(^|\n)#seat-odds \.so-ficha \{\s*scroll-margin-top:\s*68px', css)
    assert re.search(r'(^|\n)body:has\(#freeze-banner:not\(\[hidden\]\)\) #seat-odds \.so-ficha \{\s*'
                     r'scroll-margin-top:\s*114px', css)
    narrow = css[css.index('@media (max-width: 719px)', css.index('#seat-odds #district-select')):]
    narrow = narrow[:narrow.index('\n}')]
    assert re.search(r'body:has\(#freeze-banner:not\(\[hidden\]\)\) #seat-odds \.so-ficha \{\s*scroll-margin-top:\s*68px',
                     narrow)


def test_on_phones_the_block_of_a_district_lands_above_the_tab_bar():
    # under 720 px the tab bar sticks to the foot of the window (57 px and the safe area): a block the page script
    # scrolls up from below the map ends above it, not under it
    css = site_css()
    narrow = css[css.index('@media (max-width: 719px)', css.index('#seat-odds #district-select')):]
    narrow = narrow[:narrow.index('\n}')]
    assert re.search(r'#seat-odds \.so-ficha \{\s*scroll-margin-bottom:\s*calc\(68px \+ env\(safe-area-inset-bottom\)\)',
                     narrow)
    phones = css[css.index('.tabbar {', css.index('.tabbar { display: none; }') + 1):]
    assert re.match(r'\.tabbar \{\s*position:\s*sticky;\s*bottom:\s*0', phones)


def test_a_part_without_simulations_or_with_a_single_district_takes_down_no_card(odds_site, api, fresh_app):
    assert 'id="seat-odds"' in event_page(odds_site)[1]
    single = fixture_data('seat-odds')
    assert len(single['regions']) == 1 and single['n_sim']
    for data in (dict(ODDS, n_sim=0), single):
        write_odds(fresh_app.fs, data)
        assert pages.published_seat_odds('es', RUN, 'forecast') is None
        status, page = event_page(api)
        assert status == 200 and 'seat-odds' not in page  # neither the card, nor its rules, nor its CSV


def test_two_renders_of_the_event_page_without_the_part_read_it_once(odds_site, api, fresh_app, tmp_path):
    remove_odds(tmp_path)
    call(api, '/es')  # the other run files stay in the LRU
    fs = counting(fresh_app)
    assert call(api, '/es')[0] == 200 and call(api, '/es')[0] == 200
    assert [name for method, name in fs.names if 'seat-odds' in name] == [
        ('/'.join((bundle.PREFIX, bundle.path_part('es', RUN, 'seat-odds', 'forecast'))))]
    # with the part, its bytes come from the LRU: the second render, card and all, reads nothing of it
    write_odds(fresh_app.fs)
    fs = counting(fresh_app)
    renders = [call(api, '/es') for _ in range(2)]
    assert all(status == 200 and b'id="seat-odds"' in body for status, _, body in renders)
    assert len([name for method, name in fs.names if 'seat-odds' in name]) == 1


def test_the_image_of_the_card_says_the_election_the_estimate_and_its_date(odds_site):
    page = event_page(odds_site)[1]
    card = seat_odds_card(page)
    # once, above both views; only in the image (`.card:not(.shot-image) .shot-only` is hidden): the same district
    # exists in the general and in a regional scope, and the figures change every day
    stamps = re.findall(r'<p class="so-stamp shot-only">([^<]+)</p>', card)
    assert len(stamps) == 1 and stamps[0].startswith('Elecciones generales 2026 · Pronóstico para el ')
    assert re.search(r' · Actualizado: \d', stamps[0])  # the day the run was computed, written out


def write_blocks(fs, blocks):
    """Sobrescribe la parte `meta` del run del bundle de prueba con estos bloques de la web."""
    meta = fixture_data('meta')
    meta = meta | {'bmaps': meta['bmaps'] | {'blocks': blocks}}
    bundle.BundleWriter(fs).write_json(bundle.path_part('es', RUN, 'meta'), 'meta', meta, 'es', run_id=RUN)
    webapi.site().files.clear()
    webapi.site().manifest_cache_clear()


def write_summary(fs, rows, blocks=()):
    """Sobrescribe la parte `summary` del run en los dos modos con estos partidos (siglas, escaños, voto) y, si se dan,
    con una fila por bloque (la de `Derecha` del fixture con otro nombre: la tabla de bloques de la página del evento
    busca la de cada bloque del mapa)."""
    summary = fixture_data('summary')
    base = summary['parties'][0]
    summary = summary | {'parties': [base | {'name': name, 'seats': seats, 'pct': pct} for name, seats, pct in rows]}
    if blocks:
        summary['blocks'] = [summary['blocks'][0] | {'name': name} for name in blocks]
    writer = bundle.BundleWriter(fs)
    for mode in bundle.MODES:
        writer.write_json(bundle.path_part('es', RUN, 'summary', mode), 'summary', summary, 'es', run_id=RUN, mode=mode)
    webapi.site().files.clear()
    webapi.site().manifest_cache_clear()


def test_the_columns_follow_the_table_of_results_and_name_the_district_of_each_circle(odds_site, fresh_app):
    card = seat_odds_card(event_page(odds_site)[1])
    columns = card[card.index('class="so-view-cols"'):]
    assert ('<p class="so-lede">Cada círculo es una opción de escaño en alguna provincia. Abajo, los seguros; encima, '
            'los probables y los que están en juego.</p>') in columns
    # the blocks of the test bundle are incomplete: one column per party and no «Otros»
    assert re.findall(r'class="so-col" data-col="([^"]+)"', columns) == ['PP', 'PSOE', 'VOX', 'AA', 'BNG']
    # the circles of the columns are decoration: the labels under them carry the figures; their titles name the district
    assert len(re.findall(r'class="so-col" data-col="[^"]+" aria-hidden="true">', columns)) == 5
    cols = columns[columns.index('class="so-cols"'):columns.index('class="so-wlabels"')]
    assert 'title="1.º de PP en La Coruña: seguro"' in cols and 'title="1.º de PP: ' not in cols
    assert 'title="1.º de VOX en La Coruña: lo consigue en el 37% de las simulaciones"' in cols
    assert '350' not in visible_text(columns) and 'mayoría' not in columns  # never the chamber
    write_blocks(fresh_app.fs, {'Izquierda': ['PSOE', 'AA'], 'Derecha': ['PP', 'VOX'], 'Otros': ['BNG']})
    # the columns follow the table of results of the event page: by seats and then by vote, «Otros» last
    write_summary(fresh_app.fs, [('PSOE', 7, 28.0), ('VOX', 7, 30.0), ('PP', 4, 20.0), ('AA', 1, 2.0), ('BNG', 1, 3.0)],
                  blocks=('Izquierda', 'Derecha', 'Otros'))
    card = seat_odds_card(event_page(odds_site)[1])
    assert re.findall(r'class="so-col" data-col="([^"]+)"', card) == ['VOX', 'PSOE', 'PP', 'AA', 'Otros']
    labels = card[card.index('class="so-wlabels"'):]
    assert re.findall(r'([A-Za-z]+)</b><small>', labels[:labels.index('class="so-legend"')]) == [
        'VOX', 'PSOE', 'PP', 'AA', 'Otros']
    assert 'Bajo cada partido: seguros · probables · en juego. Otros:' in card


def site_css():
    return open(os.path.join(os.path.dirname(__file__), '..', 'web', 'dist', 'css', 'site.css'), encoding='utf-8').read()


def test_the_note_of_a_row_sits_beside_its_circles_and_not_at_the_far_edge_of_the_card():
    # the column of the circles takes what its longest row needs and no more, so the note («2 seguros · 1 probable»)
    # follows it; with `1fr` it took the rest of a card that now spans the page and pushed the note to the right edge
    css = site_css()
    assert re.search(r'\.so-grid \{[^}]*grid-template-columns:\s*max-content minmax\(0, max-content\) max-content;'
                     r'[^}]*justify-content:\s*start', css)
    # beside, not stuck to them: 42 px from the last circle of the longest row (Luis, 2026-11-13), 38 px of its own
    # on top of the 4 px between columns
    assert re.search(r'\.so-note \{\s*padding-left:\s*38px', css)
    assert re.search(r'\.so-grid \{[^}]*gap:\s*6px 4px', css)


def test_the_columns_make_room_for_the_top_mark_of_their_axis(odds_site):
    card = seat_odds_card(event_page(odds_site)[1])
    waffle = card[card.index('class="so-waffle"'):card.index('class="so-wlabels"')]
    # the tallest column of the test bundle has 8 options: one mark, at 50, above it; the chart is as tall as that mark
    assert waffle.startswith('class="so-waffle" style="--top: 50">')
    assert re.findall(r'class="so-tick" style="bottom: calc\((\d+) / var\(--wn\)', waffle) == ['50']
    assert re.search(r'\.so-waffle \{[^}]*min-height:\s*calc\(var\(--top, 0\) / var\(--wn\) \* '
                     r'\(var\(--wsq\) \+ var\(--wgap\)\)\)', site_css())


def test_the_long_titles_of_the_cards_wrap_beside_their_accent_bar():
    # the site's `.card-head h2` wraps its text under the bar when it does not fit a line (phones); not in this card
    assert re.search(r'\.so-card \.card-head h2\s*\{\s*flex-wrap:\s*nowrap', site_css())


def test_a_safe_circle_has_no_seam_between_its_fill_and_its_border():
    # the fill of the others stops at their border (`padding-box`): it is lighter than the border; a safe circle is
    # one colour, so its fill runs under its border and the anti-aliased edge between them never shows as a ring
    css = site_css()
    assert re.search(r'\.so-o \{[^}]*background-clip:\s*padding-box', css)
    assert re.search(r'\.so-o\.safe \{\s*background-clip:\s*border-box', css)


def test_the_card_carries_what_its_script_needs_to_name_its_image_and_pick_a_district(odds_site, fresh_app):
    write_event_districts(fresh_app.fs)
    card = seat_odds_card(event_page(odds_site)[1])
    # the image is named after what is on view: the server writes that of the district open and the base
    assert '<section class="card shot so-card" id="seat-odds" data-file-base="demoscopio-escanos-es">' in card
    assert re.findall(r'class="card-img-btn" hidden data-file="([^"]+)"', card) == [
        'demoscopio-escanos-es-la-coruna.png']
    # the line of the filters, for the image only, is born hidden and empty: the script writes it
    assert '<p class="so-filter-line shot-only" id="map-filter-line" hidden></p>' in card
    assert 'id="mp-all" data-slug="" data-label=""' in card
    assert 'id="mp-psoe" data-slug="psoe" data-label="PSOE"' in card
    assert 'id="ms-play" data-slug="en-juego" data-label="En juego"' in card
    assert 'id="sv-map" data-slug=""' in card and 'id="sv-cols" data-slug="columnas"' in card
    # the dropdown of the phones, hidden until the script reveals it: every district by name, the one open chosen,
    # with a label in the word of the card
    select = card[card.index('<select id="district-select" hidden>'):card.index('</select>')]
    assert re.findall(r'<option value="([a-z-]+)"( selected)?>([^<]+)</option>', select) == [
        ('cadiz', '', 'Cádiz · 8 escaños'), ('guadalajara', '', 'Guadalajara · 3 escaños'),
        ('la-coruna', ' selected', 'La Coruña · 8 escaños')]
    assert '<label for="district-select" class="sr-only">Provincia</label>' in card
    # the copy the image is drawn from hides the selector of views and both filters (never removes them: the rules of
    # `:has()` read their radios), and the tile chosen loses its dark border
    css = site_css()
    assert all('so-filter' in re.search(r'<fieldset class="([^"]+)" id="{}"'.format(name), card).group(1)
               for name in ('so-view', 'map-party', 'map-state'))
    assert re.search(r'\.shot-image \.so-filter\s*\{\s*display:\s*none', css)
    assert re.search(r'\.shot-image \.so-tile:has\(input:checked\)\s*\{[^}]*'
                     r'border:\s*1px solid var\(--color-line-strong\)', css)
    # nor does it say to tap a tile: the hint of the lede stays on the page
    assert re.search(r'\.shot-image \.so-hint\s*\{\s*display:\s*none', css)
    # the lede of the card is styled in the card only
    assert re.search(r'(^|\n)#seat-odds \.so-lede\s*\{', css) and not re.search(r'(^|\n)\.so-lede\s*\{', css)


def test_a_card_whose_districts_are_not_provinces_says_circunscripcion(odds_site, fresh_app):
    # the islands of Canarias and its «Lista autonómica», the three districts of Asturias: never «provincia»
    islands = [{'id': 101, 'name': 'Lista autonómica', 'seats': 2}]
    write_odds(fresh_app.fs, dict(ODDS, regions=ODDS['regions'] + islands,
                                  rows=ODDS['rows'] + [odds_row(101, 'PP', 1, 10000, 5000)]))
    page = event_page(odds_site)[1]
    card = seat_odds_card(page)
    assert '<label for="district-select" class="sr-only">Circunscripción</label>' in card
    assert re.findall(r'id="sv-map"[^>]*>([^<]+)<', card) == ['Circunscripciones']
    # a district without a tile in the catalogue: the tiles go in a row, and their group is named as the view
    assert '<div class="so-map so-map-loose" role="radiogroup" aria-label="Circunscripciones">' in card
    assert ('con el borde discontinuo si está en juego. <span class="so-hint">Toca una circunscripción para ver su '
            'detalle.</span>') in card
    assert 'Cada círculo es una opción de escaño en alguna circunscripción. Abajo, los seguros' in card
    assert 'provincia' not in visible_text(card).lower()
    assert_public_wording(page, 'circunscripciones')


def test_every_element_the_card_script_looks_for_is_in_the_card(odds_site, fresh_app):
    # read from the script itself, so that a renamed id, radio or class on either side fails here and not on a phone
    script = open(os.path.join(os.path.dirname(__file__), '..', 'web', 'dist', 'js', 'pages', 'seat-odds-wire.js'),
                  encoding='utf-8').read()
    ids = set(re.findall(r"getElementById\('([\w-]+)'\)", script))
    prefixes = set(re.findall(r"getElementById\(`([\w-]+)\$\{", script))
    radios = re.search(r'const RADIOS = \{([^}]+)\}', script).group(1)
    names = set(re.findall(r":\s*'(\w+)'", radios))
    # the elements it looks for by tag and class, down from the card (`querySelector`) or up from a target (`closest`)
    selectors = re.findall(r"(?:querySelector|closest)\('(\w+)\.([\w-]+)'\)", script)
    assert ids and prefixes and names == {'sv', 'mp', 'ms', 'd'} and ('label', 'so-tile') in selectors
    write_event_districts(fresh_app.fs)
    page = event_page(odds_site)[1]
    card = seat_odds_card(page)
    for element in ids:
        assert 'id="{}"'.format(element) in card, element
    for prefix in prefixes:
        assert 'id="{}'.format(prefix) in card, prefix
    for name in names:
        # each group only in the card; those the image is named after, each radio with its `data-slug` (and the
        # filters with their `data-label`, the words of the line)
        assert page.count('name="{}"'.format(name)) == card.count('name="{}"'.format(name)) > 0, name
    for name in ('sv', 'mp', 'ms'):
        radios = re.findall(r'<input type="radio" name="{}"[^>]*>'.format(name), card)
        assert radios and all('data-slug="' in radio for radio in radios), name
        assert name == 'sv' or all('data-label="' in radio for radio in radios), name
    for tag, cls in selectors:
        assert re.search(r'<{} [^>]*class="[^"]*\b{}\b'.format(tag, cls), card), (tag, cls)


def test_the_card_is_wired_before_the_page_scrolls_to_its_section():
    # under 720 px the script reveals the dropdown of districts (some 50 px): wired after the scroll to the section of
    # the address, it would push `#evolucion` and `#tabla` down once the page had landed on them
    script = open(os.path.join(os.path.dirname(__file__), '..', 'web', 'dist', 'js', 'pages', 'promedios.js'),
                  encoding='utf-8').read()
    paint = script[script.index('function paint()'):]
    assert -1 < paint.index('tryDraw(wireSeatOdds') < paint.index('tryDraw(scrollToSection')
