"""La página «Escaños que bailan» (`/{scope}/circunscripciones`): de la parte `seat-odds` a las fichas y sus vistas."""
import copy
import os
import re

import pytest

from mtpy.lib import bundle, pages, webapi
from tests.test_api_asgi import call
from tests.test_pages import api, assert_public_wording, counting, html, initial, visible_text  # noqa: F401
from tests.test_webapi_unit import RUN, fixture_data


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


def test_the_party_view_the_closest_seats_and_the_texts_of_the_states():
    districts = pages.district_views(ODDS)
    view = pages.party_view(districts, 'PSOE')
    assert (view['safe'], view['likely'], view['play']) == (4, 1, 1) and 'seats' not in view
    assert [row['district']['name'] for row in view['rows']] == ['Cádiz', 'La Coruña']
    assert view['settled'] == [{'name': 'Guadalajara', 'slug': 'guadalajara', 'safe': 1}]
    closest = pages.closest_seats(districts)
    assert len(closest) == 8 and closest[0] == {'district': 'Cádiz', 'slug': 'cadiz', 'party': 'PSOE', 'k': 3,
                                                'share': 0.411, 'label': '41%', 'state': 'play'}
    assert [(c['district'], c['party'], c['k']) for c in closest[1:4]] == [
        ('La Coruña', 'VOX', 1), ('La Coruña', 'PP', 4), ('La Coruña', 'BNG', 3)]
    assert (closest[-1]['party'], closest[-1]['label'], closest[-1]['state']) == ('PSOE', '98%', 'likely')
    assert [pages.state_text(s, n) for s, n in [('safe', 1), ('safe', 5), ('likely', 1), ('likely', 3), ('play', 1),
                                                ('play', 2)]] == ['1 seguro', '5 seguros', '1 probable',
                                                                  '3 probables', '1 en juego', '2 en juego']
    cadiz = by_name(districts[0])
    assert pages.tokens_note(cadiz['PP']) == '2 seguros · 1 probable' and pages.tokens_note(cadiz['AA']) == '1 probable'
    assert pages.tokens_reading('PP', cadiz['PP']) == 'PP: 2 seguros; 3.º, 80%'
    assert [pages.closest_sentence(n) for n in (0, 1, 8)] == [
        'No hay ningún escaño probable ni en juego.', 'El único escaño que no es seguro. La raya marca el 50%.',
        'Los 8 escaños más cercanos al cara o cruz. La raya marca el 50%.']


def test_a_party_without_options_has_an_empty_view_and_the_closest_seats_keep_their_limit():
    districts = pages.district_views(ODDS)
    assert pages.party_view(districts, 'PSOE')['rows'][0]['district'] == {
        'id': 11, 'name': 'Cádiz', 'slug': 'cadiz', 'seats': 8}
    assert pages.party_view(districts, 'UP') == {
        'name': 'UP', 'safe': 0, 'likely': 0, 'play': 0, 'rows': [], 'settled': []}
    assert len(pages.closest_seats(districts, limit=4)) == 4


@pytest.mark.parametrize('safe, likely, play, text', [
    (66, 30, 25, 'Tiene 66 escaños seguros, 30 probables y 25 en juego.'), (2, 0, 0, 'Tiene 2 escaños seguros.'),
    (1, 0, 0, 'Tiene 1 escaño seguro.'), (3, 0, 2, 'Tiene 3 escaños seguros y 2 en juego.'),
    (0, 2, 1, 'Tiene 2 escaños probables y 1 en juego.'), (0, 0, 1, 'Tiene 1 escaño en juego.'),
])
def test_party_sentence_names_only_the_states_with_some_option(safe, likely, play, text):
    assert pages.party_sentence({'safe': safe, 'likely': likely, 'play': play}) == text


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
    (3, {'PP': 1, 'PSOE': 1}, ['PSOE'], 'De los 3 escaños, 2 son seguros (PP 1, PSOE 1). El otro sólo está al alcance de PSOE.'),
    (4, {'PP': 1, 'PSOE': 1}, ['PP'], 'De los 4 escaños, 2 son seguros (PP 1, PSOE 1). Los otros 2 sólo están al alcance de PP.'),
    (3, {'PP': 1, 'PSOE': 1}, [], 'De los 3 escaños, 2 son seguros (PP 1, PSOE 1).'),
    (4, {'PP': 0}, ['PP', 'PSOE', 'IU'], 'Ninguno de los 4 escaños es seguro: se los disputan PP, PSOE e IU.'),
    (4, {'PP': 0}, ['PP'], 'Ninguno de los 4 escaños es seguro: sólo están al alcance de PP.'),
    (4, {'PP': 0}, [], 'Ninguno de los 4 escaños es seguro.'),
    (1, {'PP': 0}, ['PP', 'PSOE'], 'El escaño no es seguro: se lo disputan PP y PSOE.'),
    (1, {'PP': 0}, ['PP'], 'El escaño no es seguro: sólo está al alcance de PP.'),
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


def page_of(site, path, query=''):
    status, _, body = call(site, path, query=query)
    return status, re.sub(r'\s+', ' ', html(body))


RETIRED = ('posible', 'en el aire', 'aspirante', 'reparto central', 'Central')


def test_the_three_views_speak_of_three_states_and_never_of_the_central_allocation(odds_site):
    landing = page_of(odds_site, '/es/circunscripciones')[1]
    assert '<h2>La Coruña</h2>' in landing and ('Qué escaños tiene seguros cada partido, cuáles son probables y '
                                                'cuáles están en juego, según las 10.000 simulaciones del modelo.') in landing
    assert re.findall(r'<a href="[^"]*"[^>]*>(Provincias|Partidos|Los más reñidos)</a>', landing) == [
        'Provincias', 'Partidos', 'Los más reñidos']
    assert 'id="seat-kpis"' not in landing and 'so-list' not in landing and 'name="orden"' not in landing
    assert '<title>Demoscopio · Escaños que bailan · Elecciones generales 2026</title>' in landing
    cadiz = page_of(odds_site, '/es/circunscripciones/cadiz')[1]
    for chip in ('<span class="chip so-chip safe"><b>5</b> seguros</span>',
                 '<span class="chip so-chip likely"><b>3</b> probables</span>',
                 '<span class="chip so-chip play"><b>1</b> en juego</span>'):
        assert chip in cadiz
    assert ('<i class="so-o likely" style="--c: #1d84ce; --f: rgba(29, 132, 206, 0.475)" '
            'title="3.º de PP: lo consigue en el 80% de las simulaciones"></i>') in cadiz
    assert '<span class="so-pct">80%</span>' in cadiz and '>7%<' not in cadiz  # under 25%: neither drawn nor counted
    assert 'aria-label="PP: 2 seguros; 3.º, 80%"' in cadiz and '<div class="so-note">2 seguros · 1 probable</div>' in cadiz
    assert 'De los 8 escaños, 5 son seguros (PP 2, PSOE 2, VOX 1). Los otros 3 se los disputan PP, PSOE, VOX y AA.' in cadiz
    legend = cadiz[cadiz.index('class="so-legend"'):]
    assert 'Seguro: 99% o más' in legend and 'Probable: del 75% al 99%' in legend and 'En juego: del 25% al 75%' in legend
    assert 'probable si lo consigue entre el 75% y el 99% y está en juego entre el 25% y el 75%' in cadiz
    form = cadiz[cadiz.index('id="district-pick"'):cadiz.index('</form>')]
    assert re.findall(r'<option value="[a-z-]+"( selected)?>([^<]+)</option>', form) == [
        (' selected', 'Cádiz · 8 escaños'), ('', 'Guadalajara · 3 escaños'), ('', 'La Coruña · 8 escaños')]
    party = page_of(odds_site, '/es/circunscripciones', 'partido=psoe')[1]
    assert 'Tiene 4 escaños seguros, 1 probable y 1 en juego.' in party
    assert '<b>Sólo seguros (1):</b> Guadalajara 1.' in party
    closest = page_of(odds_site, '/es/circunscripciones', 'vista=renidos')[1]
    assert 'Los 8 escaños más cercanos al cara o cruz. La raya marca el 50%.' in closest
    assert closest.count('<tr role="row">') == 1 + 8
    for label, page in (('landing', landing), ('cadiz', cadiz), ('party', party), ('closest', closest)):
        text = visible_text(page[page.index('id="content"'):])
        assert not [word for word in RETIRED if word in text], label
        assert_public_wording(page, label)
    assert call(odds_site, '/es/circunscripciones', query='partido=up')[0] == 404  # a party without options



def test_the_map_draws_every_district_as_the_label_of_a_radio_and_every_card_is_on_the_page(odds_site):
    page = page_of(odds_site, '/es/circunscripciones/cadiz')[1]
    assert '<h2>Dónde bailan los escaños, provincia a provincia</h2>' in page
    grid = page[page.index('class="so-map"'):page.index('class="so-legend"')]
    assert re.findall(r'name="d" value="([a-z-]+)"', grid) == ['la-coruna', 'guadalajara', 'cadiz']  # by row and column
    assert re.findall(r'name="d" value="([a-z-]+)"[^>]* checked', grid) == ['cadiz']
    assert 'title="Cádiz · 8 escaños" style="grid-column: 4 / span 1; grid-row: 9 / span 1"' in grid
    assert '<span class="so-ab">CA</span>' in grid and 'aria-label="Cádiz, 8 escaños"' in grid
    assert grid.count('class="so-o ') == 22 and grid.count('class="so-o play p-') == 4
    assert '<i class="so-o safe p-pp" style="--c: #1d84ce" title="1.º de PP: seguro"></i>' in grid
    assert re.findall(r'name="mp" value="([a-z-]+)"', page) == ['all', 'pp', 'psoe', 'vox', 'aa', 'bng']  # never UP
    states = page[page.index('id="map-state"'):page.index('class="so-map"')]
    assert re.findall(r'>([^<>]+) <small>(\d+)</small>', states) == [
        ('Todos', '22'), ('Seguros', '14'), ('Probables', '4'), ('En juego', '4')]
    # the rules of the run: one per party for the filter, one per district for its card
    assert '#mapa:has(#mp-psoe:checked) .so-dots i:not(.p-psoe){opacity:.08}' in page
    assert '#provincias:has(#d-cadiz:checked) #ficha-cadiz{display:block}' in page
    # without `:has()` no card stays hidden: every one shows
    css = open(os.path.join(os.path.dirname(__file__), '..', 'web', 'dist', 'css', 'site.css'), encoding='utf-8').read()
    assert re.search(r'@supports not selector\(:has\(\*\)\)\s*\{\s*\.so-ficha\s*\{\s*display:\s*block', css)
    assert re.findall(r'class="card shot so-card so-ficha" id="ficha-([a-z-]+)"', page) == ['cadiz', 'guadalajara', 'la-coruna']
    assert 'Hay 22 opciones para 19 escaños: a un mismo escaño pueden optar varios partidos.' in page
    assert '<title>Demoscopio · Escaños que bailan en Cádiz · Elecciones generales 2026</title>' in page
    landing = page_of(odds_site, '/es/circunscripciones')[1]
    assert re.findall(r'name="d" value="([a-z-]+)"[^>]* checked', landing) == ['la-coruna']
    text = visible_text(page[page.index('id="content"'):])
    assert not [word for word in RETIRED if word in text]
    assert_public_wording(page, 'provincias')


def test_the_map_without_positions_and_the_foot_without_more_options_than_seats(odds_site, fresh_app, monkeypatch):
    monkeypatch.setattr('mtpy.lib.pages.circunscripciones.district_positions', lambda scope: {})
    page = page_of(odds_site, '/es/circunscripciones')[1]
    loose = page[page.index('class="so-map so-map-loose"'):page.index('class="so-legend"')]
    assert re.findall(r'<span class="so-ab">([^<]+)</span>', loose) == ['Cádiz', 'Guadalajara', 'La Coruña']
    assert 'grid-column' not in loose
    monkeypatch.undo()
    safe_only = dict(ODDS, rows=[odds_row(row['region_id'], row['name'], row['seats'], *[n for n in row['reach'] if n >= 9900])
                                 for row in ODDS['rows']])
    write_odds(fresh_app.fs, safe_only)  # 14 options for 19 seats, and districts where nobody disputes what is left
    page = page_of(odds_site, '/es/circunscripciones/cadiz')[1]
    assert 'opciones para' not in page and 'De los 8 escaños, 5 son seguros (PP 2, PSOE 2, VOX 1).' in page
    assert 'se los disputan' not in page

def test_a_district_card_draws_a_circle_per_option_and_labels_only_those_not_safe(odds_site):
    status, page = page_of(odds_site, '/es/circunscripciones/la-coruna')
    start = page.index('id="ficha-la-coruna"')
    card = page[start:page.index('class="so-legend"', start)]
    # every title of the site starts with its name (`base.html`)
    assert status == 200 and 'La Coruña' in card and '<title>Demoscopio · Escaños que bailan en La Coruña' in page
    labels = re.findall(r'<span class="so-pct">([^<]+)</span>', card)
    assert sorted(labels) == ['25%', '30%', '37%', '98%'] and not {'99%', '100%', '24%'} & set(labels)
    states = re.findall(r'<span class="so-token (safe|likely|play)"><i class="so-o \1" ', card)
    assert (states.count('safe'), states.count('likely'), states.count('play')) == (6, 1, 3)
    assert re.findall(r'<i class="so-o [a-z]+" [^>]*title="([^"]+)"', card)[:3] == [
        '1.º de PP: seguro', '2.º de PP: seguro', '3.º de PP: seguro']
    assert '<i class="so-o safe" style="--c: #1d84ce" title="1.º de PP: seguro"></i>' in card  # solid: no fill
    assert '<link rel="canonical" href="' in page and page.count('<details class="how"') == 3  # one per card
    page = page_of(odds_site, '/es/circunscripciones/guadalajara')[1]
    start = page.index('id="ficha-guadalajara"')
    closed = page[start:page.index('</section>', start)]
    assert 'Los 3 escaños son seguros (PP 1, PSOE 1, VOX 1).' in closed
    assert '<span class="chip so-chip safe"><b>3</b> seguros</span>' in closed and 'so-chip likely' not in closed


def test_the_landing_is_shared_as_the_page_and_each_view_has_its_description(odds_site):
    page = page_of(odds_site, '/es/circunscripciones')[1]
    assert '<h1 class="headline">Escaños que bailan</h1>' in page
    # the address without a district, a party or a view is shared as the page, not as its default district (which
    # changes between runs); its card is still that district's, with the same image
    assert '<meta property="og:title" content="Demoscopio · Escaños que bailan · Elecciones generales 2026">' in page
    assert ('<meta name="description" content="Qué escaños tiene seguros cada partido en cada circunscripción, cuáles '
            'son probables y cuáles están en juego, según las simulaciones del modelo para Elecciones generales '
            '2026.">') in page
    assert 'data-file="demoscopio-escanos-es-la-coruna.png"' in page and '/dist/js/pages/circunscripciones.js' in page
    assert 'data-file="demoscopio-escanos-es-mapa.png"' in page
    assert 'aria-label="VOX: 1.º, 37%"' in page and '<div class="so-note">1 en juego</div>' in page
    cadiz = page_of(odds_site, '/es/circunscripciones/cadiz')[1]
    assert '<title>Demoscopio · Escaños que bailan en Cádiz · Elecciones generales 2026</title>' in cadiz
    assert ('<meta name="description" content="Los escaños de Cádiz, uno a uno: cuáles tiene seguros cada partido, '
            'cuáles son probables y cuáles están en juego, según las simulaciones del modelo para Elecciones '
            'generales 2026.">') in cadiz
    # the dropdown sits between the map and the cards (it only shows under 720 px)
    assert cadiz.index('id="mapa"') < cadiz.index('id="district-pick"') < cadiz.index('id="ficha"')


def test_the_page_follows_the_mode_and_keeps_it_in_its_links(odds_site, fresh_app):
    nowcast = copy.deepcopy(ODDS)
    nowcast['rows'][3] = odds_row(11, 'AA', 1, 2500)  # AA's seat in Cádiz: 76% in the election mode, 25% today
    write_odds(fresh_app.fs, nowcast, modes=('nowcast',))
    assert '>76%<' in page_of(odds_site, '/es/circunscripciones/cadiz')[1]  # less than 60 days away: forecast
    today = page_of(odds_site, '/es/circunscripciones/cadiz', 'mode=nowcast')[1]
    assert '>25%<' in today and '>76%<' not in today
    assert 'href="/es?mode=nowcast#territorio"' in today  # the index of the event bar goes back to the event page
    # the districts linked by the other views keep the mode
    closest = page_of(odds_site, '/es/circunscripciones', 'vista=renidos&mode=nowcast')[1]
    assert 'href="/es/circunscripciones/la-coruna?mode=nowcast#ficha"' in closest


def test_the_page_does_not_exist_without_the_part_or_with_a_single_district(api, odds_site, fresh_app, tmp_path):
    assert call(odds_site, '/es/circunscripciones/no-existe')[0] == 404
    status, headers, _ = call(odds_site, '/es/circunscripciones', query='circunscripcion=cadiz&mode=nowcast')
    # the form without JavaScript: on to the address of the district, keeping the mode
    # with the fragment of the card: on a phone each change of district lands on it, not at the top of the page
    assert status == 301 and headers['location'] == '/es/circunscripciones/cadiz?mode=nowcast#ficha'
    write_odds(fresh_app.fs, fixture_data('seat-odds'))
    status, page = page_of(api, '/es/circunscripciones')
    assert status == 404 and 'Este run no tiene el desglose escaño a escaño.' in page  # one district
    remove_odds(tmp_path)
    assert call(api, '/es/circunscripciones')[0] == 404  # a run published before the part existed


def test_a_pinned_run_stays_in_every_link_of_the_page_and_the_canonical_drops_it(odds_site):
    status, page = page_of(odds_site, '/es/circunscripciones/cadiz', 'mode=nowcast&run=' + RUN)
    query = '?mode=nowcast&amp;run=' + RUN
    assert status == 200
    closest = page_of(odds_site, '/es/circunscripciones', 'vista=renidos&mode=nowcast&run=' + RUN)[1]
    assert 'href="/es/circunscripciones/la-coruna{}#ficha"'.format(query) in closest  # a district linked by a view
    # the tabs of the views land on themselves (`#vistas`), not at the top of the page
    assert '<nav class="seg lg so-tabs" id="vistas" aria-label="Vistas">' in page
    assert 'href="/es/circunscripciones{}#vistas"'.format(query) in page  # the tab of the view
    assert 'href="/es/circunscripciones{}&amp;vista=partidos#vistas"'.format(query) in page  # the tab of the parties
    assert 'href="/es/circunscripciones{}&amp;vista=renidos#vistas"'.format(query) in page
    assert 'href="/es{}#tabla"'.format(query) in page  # the index of the event bar
    form = page[page.index('id="district-pick"'):page.index('</form>')]
    assert '<input type="hidden" name="run" value="{}">'.format(RUN) in form and 'value="nowcast"' in form
    assert '<option value="cadiz" selected>Cádiz · 8 escaños</option>' in form
    assert '<option value="guadalajara">Guadalajara · 3 escaños</option>' in form
    assert '<link rel="canonical" href="https://demoscopio.es/es/circunscripciones/cadiz">' in page
    status, headers, _ = call(odds_site, '/es/circunscripciones', query='mode=nowcast&run={}&circunscripcion=cadiz'.format(RUN))
    assert status == 301 and headers['location'] == '/es/circunscripciones/cadiz?mode=nowcast&run={}#ficha'.format(RUN)


def test_the_positions_of_the_map_come_from_the_catalogue(api):
    positions = webapi.district_positions('es')
    assert len(positions) == 52 and positions[31] == {'abbr': 'NA', 'col': 6, 'row': 0, 'w': 1, 'h': 1}
    assert positions[28] == {'abbr': 'M', 'col': 4, 'row': 3, 'w': 2, 'h': 2} and webapi.district_positions('es-an') == {}
    assert all(type(positions[28][key]) is int for key in ('col', 'row', 'w', 'h'))


def test_the_texts_of_a_row_and_the_link_of_a_district():
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
    assert pages.district_href('es', 'la-coruna', '?mode=nowcast') == '/es/circunscripciones/la-coruna?mode=nowcast#ficha'


def test_the_event_bar_of_a_subpage_links_the_event_page_and_drops_the_territory_without_districts(api, tmp_path):
    """The helper shared by the page of a poll and this page: neither page's tests reach a run without districts."""
    context = pages.common_context({'scope': 'es', 'mode': 'nowcast', 'run': RUN}, None, event=True)
    bar = pages.subpage_event_bar(context)
    query = '?mode=nowcast&run=' + RUN
    assert [entry['href'] for entry in bar['sections']] == [
        '/es' + query + '#' + key for key in ('promedio', 'escanos', 'evolucion', 'territorio', 'tabla')]
    assert context['event_bar']['sections'][0]['href'] == '#promedio'  # a copy: the bar of the context is untouched
    os.remove(os.path.join(str(tmp_path), bundle.PREFIX, bundle.path_part('es', RUN, 'districts', 'nowcast')))
    webapi.site().files.clear()
    assert [entry['id'] for entry in pages.subpage_event_bar(context)['sections']] == [
        'promedio', 'escanos', 'evolucion', 'tabla']


def test_the_party_view_lists_its_doubtful_districts_and_sums_the_settled_ones(odds_site):
    status, page = page_of(odds_site, '/es/circunscripciones', 'partido=psoe')
    assert status == 200 and '<title>Demoscopio · Escaños que bailan: PSOE' in page
    assert re.findall(r'class="so-party-btn[^"]*" href="[^"]*partido=([a-z]+)#partidos"', page) == [
        'pp', 'psoe', 'vox', 'aa', 'bng']
    card = page[page.index('id="ficha"'):page.index('class="so-legend"', page.index('id="ficha"'))]
    assert re.findall(r'href="/es/circunscripciones/([a-z-]+)\?mode=forecast#ficha"', card) == ['cadiz', 'la-coruna']
    assert 'Tiene 4 escaños seguros, 1 probable y 1 en juego.' in card and 'reparto central' not in card
    assert re.findall(r'<span class="chip so-chip ([a-z]+)"><b>(\d+)</b> ([^<]+)</span>', card) == [
        ('safe', '4', 'seguros'), ('likely', '1', 'probable'), ('play', '1', 'en juego')]
    assert re.findall(r'<div class="so-h[^"]*">([^<]*)</div>', card) == [
        'Circunscripción', 'Escaños', 'Escaño a escaño', '']
    assert 'aria-label="Cádiz: 2 seguros; 3.º, 41%"' in card and 'aria-label="La Coruña: 1 seguro; 2.º, 98%"' in card
    assert 'Sólo seguros (1):' in card and 'Guadalajara 1' in card
    assert '<link rel="canonical" href="https://demoscopio.es/es/circunscripciones?partido=psoe">' in page
    assert 'data-file="demoscopio-escanos-es-psoe.png"' in page and page.count('<details class="how"') == 1
    assert ('<meta name="description" content="Los escaños de PSOE, circunscripción a circunscripción: cuáles tiene '
            'seguros, cuáles son probables y cuáles están en juego, según las simulaciones del modelo para Elecciones '
            'generales 2026.">') in page
    assert_public_wording(page, 'circunscripciones por partido')
    assert call(odds_site, '/es/circunscripciones', query='partido=nadie')[0] == 404
    kept = page_of(odds_site, '/es/circunscripciones', 'partido=pp&mode=nowcast')[1]
    assert 'href="/es/circunscripciones/cadiz?mode=nowcast#ficha"' in kept
    assert 'class="so-party-btn on" href="/es/circunscripciones?mode=nowcast&amp;partido=pp#partidos"' in kept
    # the district of the path wins over the party
    assert '<title>Demoscopio · Escaños que bailan en Cádiz' in page_of(odds_site, '/es/circunscripciones/cadiz', 'partido=pp')[1]


def test_the_closest_view_ranks_the_options_not_safe_by_their_distance_to_a_coin_toss(odds_site):
    status, page = page_of(odds_site, '/es/circunscripciones', 'vista=renidos')
    table = page[page.index('id="closest"'):]
    # under 720 px each row stacks in two lines (`display: grid`), so the table carries its ARIA roles
    assert table.startswith('id="closest" role="table"') and table.count('scope="col" role="columnheader"') == 3
    assert re.findall(r'role="columnheader">([^<]+)</th>', table) == ['Circunscripción', 'Escaño', 'Lo consigue en']
    rows = re.findall(r'<tr role="row">(.*?)</tr>', table[table.index('<tbody role="rowgroup">'):])
    assert status == 200 and len(rows) == 8 and all(row.count('role="cell"') == 3 for row in rows)
    assert 'Cádiz' in rows[0] and '3.º de PSOE' in rows[0] and '<b>41%</b>' in rows[0]
    assert 'La Coruña' in rows[1] and '1.º de VOX' in rows[1] and '<b>37%</b>' in rows[1]
    assert '1.º de PP' not in table and '<b>98%</b>' in rows[-1]  # never a safe one; the likely ones too
    assert '<link rel="canonical" href="https://demoscopio.es/es/circunscripciones?vista=renidos">' in page
    assert 'Los 8 escaños más cercanos al cara o cruz. La raya marca el 50%.' in page
    assert '<title>Demoscopio · Los escaños más reñidos · Elecciones generales 2026' in page
    assert 'class="seg lg so-tabs"' in page and re.findall(r'<a href="[^"]*"( class="on")', page) == [' class="on"']
    assert_public_wording(page, 'circunscripciones, los más reñidos')
    assert call(odds_site, '/es/circunscripciones', query='vista=otra')[0] == 400


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


def test_the_event_page_links_the_page_and_the_card_of_the_district_shown(odds_site, api, fresh_app, tmp_path):
    write_event_districts(fresh_app.fs)
    status, _, body = call(odds_site, '/es', query='mode=nowcast')
    page = re.sub(r'\s+', ' ', html(body))
    territory = page[page.index('id="territorio"'):page.index('id="tabla"')]
    assert status == 200 and 'href="/es/circunscripciones?mode=nowcast"' in territory and 'Escaños que bailan' in territory
    assert re.search(r'<a id="region-odds"[^>]*href="/es/circunscripciones/[a-z-]+\?mode=nowcast#ficha"', territory)
    assert initial(body)['odds_hrefs']['11'] == '/es/circunscripciones/cadiz?mode=nowcast#ficha'
    assert 'seat-odds' in page[page.index('id="escanos"'):]  # its CSV among the downloads
    remove_odds(tmp_path)
    status, bare = page_of(api, '/es')
    assert status == 200 and '/es/circunscripciones' not in bare and 'region-odds' not in bare and 'seat-odds' not in bare


def test_the_results_card_links_the_page_at_the_foot_of_its_tables(odds_site, api, fresh_app, tmp_path):
    # the same link as the territory card, under the tables of «Resultados» and above its distributions (Luis, 2026-11-12)
    write_event_districts(fresh_app.fs)
    link = ('<p class="more"><a class="more-link" href="/es/circunscripciones?mode=nowcast">'
            'Escaños que bailan: los seguros y los que están en el aire ›</a></p>')
    page = page_of(odds_site, '/es', 'mode=nowcast')[1]
    results = page[page.index('id="seats"'):page.index('id="coalition"')]
    assert results.count(link) == 1 and page.count(link) == 2  # here and at the foot of the territory card
    assert results.rindex('</table>') < results.index(link) < results.index('id="ridge-toggle"')
    remove_odds(tmp_path)
    bare = page_of(api, '/es')[1]
    assert 'circunscripciones' not in bare[bare.index('id="seats"'):bare.index('id="coalition"')]


def test_the_links_of_the_event_page_keep_a_pinned_run(odds_site, fresh_app):
    write_event_districts(fresh_app.fs)
    status, _, body = call(odds_site, '/es', query='mode=nowcast&run=' + RUN)
    page = re.sub(r'\s+', ' ', html(body))
    assert status == 200 and 'href="/es/circunscripciones?mode=nowcast&amp;run={}"'.format(RUN) in page
    hrefs = initial(body)['odds_hrefs']
    assert hrefs['11'] == '/es/circunscripciones/cadiz?mode=nowcast&run={}#ficha'.format(RUN)
    assert all('run={}#ficha'.format(RUN) in href for href in hrefs.values()) and len(hrefs) == 3


def test_the_link_of_a_district_without_card_is_hidden_and_a_single_district_gives_no_links(odds_site, api, fresh_app):
    # the detail shows a district that is not in the part: its link is there, hidden and without href, for the script
    soria = {'id': 42, 'name': 'Soria', 'seats': 2}
    districts = dict(DISTRICTS, regions=DISTRICTS['regions'] + [dict(soria, community=None, votes=1000)],
                     rows=DISTRICTS['rows'] + [dict(DISTRICTS['rows'][0], region_id=42, region='Soria')])
    write_event_districts(fresh_app.fs, districts)
    page = page_of(odds_site, '/es', 'region=42')[1]
    assert '<p class="more" hidden><a id="region-odds" class="more-link">Escaño a escaño en Soria ›</a></p>' in page
    assert 'href="/es/circunscripciones?mode=forecast"' in page  # the link to the page stays
    write_odds(fresh_app.fs, fixture_data('seat-odds'))  # one district: no page, so no links
    status, bare = page_of(api, '/es')
    assert status == 200 and '/es/circunscripciones' not in bare and 'region-odds' not in bare and 'seat-odds' not in bare


def test_the_links_of_the_event_page_do_not_compute_the_tokens(odds_site, fresh_app, monkeypatch):
    def fail(data):
        raise AssertionError('district_views called')

    monkeypatch.setattr('mtpy.lib.pages.circunscripciones.district_views', fail)
    links = pages.seat_odds_links('es', RUN, 'forecast', '?mode=forecast')
    assert links == {'page': '/es/circunscripciones?mode=forecast',
                     'districts': {11: '/es/circunscripciones/cadiz?mode=forecast#ficha',
                                   19: '/es/circunscripciones/guadalajara?mode=forecast#ficha',
                                   15: '/es/circunscripciones/la-coruna?mode=forecast#ficha'}}


def test_a_part_without_simulations_takes_down_no_page(odds_site, api, fresh_app):
    write_odds(fresh_app.fs, dict(ODDS, n_sim=0))
    assert pages.published_seat_odds('es', RUN, 'forecast') is None
    status, page = page_of(api, '/es')
    assert status == 200 and '/es/circunscripciones' not in page and 'seat-odds' not in page
    status, page = page_of(api, '/es/circunscripciones')
    assert status == 404 and 'Este run no tiene el desglose escaño a escaño.' in page


def test_two_renders_of_the_event_page_without_the_part_read_it_once(odds_site, api, fresh_app, tmp_path):
    remove_odds(tmp_path)
    call(api, '/es')  # the other run files stay in the LRU
    fs = counting(fresh_app)
    assert call(api, '/es')[0] == 200 and call(api, '/es')[0] == 200
    assert [name for method, name in fs.names if 'seat-odds' in name] == [
        ('/'.join((bundle.PREFIX, bundle.path_part('es', RUN, 'seat-odds', 'forecast'))))]
    # with the part, its bytes come from the LRU: the second render reads nothing of it
    write_odds(fresh_app.fs)
    fs = counting(fresh_app)
    assert call(api, '/es')[0] == 200 and call(api, '/es')[0] == 200
    assert len([name for method, name in fs.names if 'seat-odds' in name]) == 1


def test_the_image_of_both_cards_says_the_election_the_estimate_and_its_date(odds_site):
    for query, cards in (('', ('mapa', 'ficha-la-coruna')), ('partido=psoe', ('ficha',))):
        page = page_of(odds_site, '/es/circunscripciones', query)[1]
        suffix = re.search(r'<span class="pill brand">([^<]+)</span>', page).group(1)  # the pill of the event bar
        for card_id in cards:
            start = page.index('id="{}"'.format(card_id))
            card = page[start:page.index('class="so-legend"', start)]
            stamp = re.search(r'<p class="so-stamp shot-only">([^<]+)</p>', card)
            # only in the image (`.card:not(.shot-image) .shot-only` is hidden): the same district exists in the
            # general and in a regional scope, and the figures change every day
            assert stamp and stamp.group(1).startswith('Elecciones generales 2026 · {} · Actualizado: '.format(suffix))
            assert re.search(r' · Actualizado: \d', stamp.group(1))  # the day the run was computed, written out


@pytest.mark.parametrize('path, query, canonical', [
    # the route of a district ignores the party, the view and the dropdown
    ('/es/circunscripciones/cadiz', 'partido=pp&vista=x&circunscripcion=', '/es/circunscripciones/cadiz'),
    # the party wins over the view
    ('/es/circunscripciones', 'partido=pp&vista=renidos', '/es/circunscripciones?partido=pp'),
    # the dropdown never stays (an empty one is ignored by the page)
    ('/es/circunscripciones', 'vista=renidos&circunscripcion=&mode=nowcast', '/es/circunscripciones?vista=renidos'),
    ('/es/circunscripciones', 'vista=partidos&mode=nowcast', '/es/circunscripciones?vista=partidos'),
])
def test_the_canonical_keeps_only_what_the_page_shows(odds_site, path, query, canonical):
    status, page = page_of(odds_site, path, query)
    assert status == 200 and '<link rel="canonical" href="https://demoscopio.es{}">'.format(canonical) in page


def test_the_party_wins_over_the_view_and_the_errors_and_units_read_well(odds_site):
    page = page_of(odds_site, '/es/circunscripciones', 'partido=psoe&vista=renidos')[1]
    assert '<title>Demoscopio · Escaños que bailan: PSOE' in page and 'id="closest"' not in page
    status, page = page_of(odds_site, '/es/circunscripciones/no-existe')
    assert status == 404 and 'Circunscripción no encontrada.' in page
    # one meaning per word: the legend says the three states with their bands, each with its own sample circle
    for query in ('', 'partido=pp'):
        page = page_of(odds_site, '/es/circunscripciones', query)[1]
        start = page.index('class="so-legend"')
        legend = page[start:page.index('</div>', start)]
        assert re.findall(r'<i class="so-o (safe|likely|play)" aria-hidden="true"></i>([^<]+)</span>', legend) == [
            ('safe', 'Seguro: 99% o más'), ('likely', 'Probable: del 75% al 99%'), ('play', 'En juego: del 25% al 75%')]


def test_a_party_with_only_safe_seats_has_no_grid_and_says_so(odds_site, fresh_app):
    settled = copy.deepcopy(ODDS)
    settled['rows'][3] = odds_row(11, 'AA', 1, 10000)  # AA's seat in Cádiz, safe: nothing of AA likely nor in play
    write_odds(fresh_app.fs, settled)
    status, page = page_of(odds_site, '/es/circunscripciones', 'partido=aa')
    card = page[page.index('id="ficha"'):page.index('class="so-legend"', page.index('id="ficha"'))]
    assert status == 200 and 'Tiene 1 escaño seguro.' in card
    assert 'so-grid' not in card and '<b>Sólo seguros (1):</b> Cádiz 1.' in card
    assert 'chip so-chip safe' in card and 'chip so-chip likely' not in card and 'chip so-chip play' not in card


def write_blocks(fs, blocks):
    """Sobrescribe la parte `meta` del run del bundle de prueba con estos bloques de la web."""
    meta = fixture_data('meta')
    meta = meta | {'bmaps': meta['bmaps'] | {'blocks': blocks}}
    bundle.BundleWriter(fs).write_json(bundle.path_part('es', RUN, 'meta'), 'meta', meta, 'es', run_id=RUN)
    webapi.site().files.clear()
    webapi.site().manifest_cache_clear()


def test_the_parties_tab_draws_the_hemicycle_and_the_columns_and_opens_on_the_hemicycle(odds_site, fresh_app):
    status, page = page_of(odds_site, '/es/circunscripciones', 'vista=partidos')
    assert status == 200 and '<title>Demoscopio · Escaños que bailan, partido a partido · Elecciones generales 2026</title>' in page
    assert re.findall(r'<a href="[^"]*"( class="on" aria-current="page")?>(Provincias|Partidos|Los más reñidos)</a>', page) == [
        ('', 'Provincias'), (' class="on" aria-current="page"', 'Partidos'), ('', 'Los más reñidos')]
    assert 'href="/es/circunscripciones?mode=forecast&amp;vista=partidos#vistas"' in page
    assert re.findall(r'name="pf" value="(hemi|cols)"[^>]*?( checked)?>', page) == [('hemi', ' checked'), ('cols', '')]
    assert '<h2 class="so-form-hemi">El hemiciclo que baila</h2>' in page
    assert '<h2 class="so-form-cols">Las opciones de escaño de cada partido</h2>' in page
    start = page.index('<svg class="so-hemi-svg"')
    svg = page[start:page.index('</svg>', start)]  # the icons of the header and of the image button come first
    assert 'viewBox="0 20 1000 520" role="img" aria-label="Hemiciclo con las opciones de escaño de cada partido según su probabilidad"' in svg
    assert svg.count('<circle') == 22 and svg.count('stroke-dasharray') == 4 and svg.count('<title>') == 22
    for count in ('<b>14</b> seguros', '<b>4</b> probables', '<b>4</b> en juego'):
        assert count in page[page.index('class="so-hemi-core"'):]
    # the order of the hemicycle (`hemicycle_order`): the test catalogue only gives a block to PP (Derecha), so the
    # parties without one go after it, in the order of the catalogue
    assert re.findall(r'<b>([A-Z]+)</b><em>([^<]+)</em>', page) == [
        ('PP', '6 · 1 · 1'), ('PSOE', '4 · 1 · 1'), ('VOX', '2 · 1 · 1'), ('AA', '0 · 1 · 0'), ('BNG', '2 · 0 · 1')]
    assert ('Tras cada partido: seguros · probables · en juego. Hay 22 opciones para 19 escaños: a un mismo escaño '
            'pueden optar varios partidos.') in page
    # the blocks of the test bundle are incomplete: one column per party and no «Otros»
    assert re.findall(r'class="so-col" data-col="([^"]+)"', page) == ['PP', 'PSOE', 'VOX', 'AA', 'BNG']
    assert 'href="/es/circunscripciones?mode=forecast&amp;partido=psoe#partidos"' in page and 'id="ficha"' not in page
    assert '350' not in visible_text(page[page.index('id="grafico"'):page.index('id="partidos"')])  # never the chamber
    assert 'mayoría' not in page[page.index('id="grafico"'):page.index('id="partidos"')]
    write_blocks(fresh_app.fs, {'Izquierda': ['PSOE', 'AA'], 'Derecha': ['PP', 'VOX'], 'Otros': ['BNG']})
    page = page_of(odds_site, '/es/circunscripciones', 'partido=psoe')[1]
    assert re.findall(r'class="so-col" data-col="([^"]+)"', page) == ['PSOE', 'AA', 'PP', 'VOX', 'Otros']
    assert 'Bajo cada partido: seguros · probables · en juego. Otros:' in page
    assert 'id="ficha"' in page and 'Tiene 4 escaños seguros, 1 probable y 1 en juego.' in page
    text = visible_text(page[page.index('id="content"'):])
    assert not [word for word in RETIRED if word in text]
    assert_public_wording(page, 'partidos')
    assert call(odds_site, '/es/circunscripciones', query='vista=otra')[0] == 400
