"""La página «Escaños que bailan» (`/{scope}/circunscripciones`): de la parte `seat-odds` a las fichas y sus vistas."""
import copy
import os
import re

import pytest

from mtpy.lib import bundle, pages, webapi
from tests.test_api_asgi import call
from tests.test_pages import api, assert_public_wording, html, initial  # noqa: F401
from tests.test_webapi_unit import RUN, fixture_data


def odds_row(region_id, name, seats, *reach):
    return {'region_id': region_id, 'name': name, 'seats': seats, 'reach': list(reach)}


ODDS = {
    'n_sim': 10000, 'parties': ['PP', 'PSOE', 'VOX', 'AA', 'BNG'],
    'regions': [{'id': 11, 'name': 'Cádiz', 'seats': 8}, {'id': 19, 'name': 'Guadalajara', 'seats': 3},
                {'id': 15, 'name': 'La Coruña', 'seats': 8}],
    'rows': [
        odds_row(11, 'PP', 3, 10000, 10000, 8010, 730, 10), odds_row(11, 'PSOE', 2, 10000, 9950, 4110, 80),
        odds_row(11, 'VOX', 2, 10000, 8530, 570), odds_row(11, 'AA', 1, 7580),
        odds_row(19, 'PP', 1, 10000, 60), odds_row(19, 'PSOE', 1, 9990), odds_row(19, 'VOX', 1, 9950),
        odds_row(15, 'PP', 3, 10000, 10000, 9900, 3000),  # the third, exactly at 99%: safe
        odds_row(15, 'PSOE', 2, 10000, 9899, 2400),       # the second, at 98,99%: open, labelled 98%
        odds_row(15, 'VOX', 1, 3657),                     # a central seat that comes out in 37% of the simulations
        odds_row(15, 'BNG', 2, 10000, 9950, 100, 99),     # the third, exactly at 1%: drawn; the fourth is not
    ],
}


def by_name(district):
    return {row['name']: row for row in district['rows']}


def test_tokens_follow_the_two_thresholds_to_the_exact_count():
    pp = pages.seat_tokens('PP', [10000, 10000, 8010, 730, 10], 3, 10000)
    assert [(t['k'], t['safe'], t['central'], t['label']) for t in pp] == [
        (1, True, True, None), (2, True, True, None), (3, False, True, '80%'), (4, False, False, '7%')]
    assert (pp[2]['alpha'], pp[0]['alpha']) == (round(0.08 + 0.42 * 0.801, 3), None)
    assert pp[1]['title'] == '2.º de PP: seguro'
    assert pp[2]['title'] == '3.º de PP: lo consigue en el 80% de las simulaciones'
    edges = pages.seat_tokens('X', [9900, 9899, 100, 99], 0, 10000)
    assert [(t['k'], t['safe'], t['label']) for t in edges] == [(1, True, None), (2, False, '98%'), (3, False, '1%')]
    assert pages.seat_tokens('X', [], 1, 10000) == []


@pytest.mark.parametrize('share, label', [(0.9899, '98%'), (0.985, '98%'), (0.805, '81%'), (0.5, '50%'),
                                           (0.0149, '1%'), (0.01, '1%')])
def test_token_label_is_a_whole_percentage_between_1_and_98(share, label):
    assert pages.token_label(share) == label


def test_district_views_count_the_safe_and_the_open_seats():
    cadiz, guadalajara, coruna = pages.district_views(ODDS)
    assert (cadiz['name'], cadiz['slug'], cadiz['seats']) == ('Cádiz', 'cadiz', 8) and coruna['slug'] == 'la-coruna'
    assert (cadiz['safe'], cadiz['open'], cadiz['contenders']) == (5, 3, 6)
    assert [row['name'] for row in cadiz['rows']] == ['PP', 'PSOE', 'VOX', 'AA']  # central seats, then expected seats
    assert [(row['safe'], row['open']) for row in cadiz['rows']] == [(2, 2), (2, 1), (1, 2), (0, 1)]
    assert by_name(cadiz)['PSOE']['expected'] == pytest.approx(2.414)
    assert (guadalajara['safe'], guadalajara['open'], guadalajara['contenders']) == (3, 0, 0)
    assert (coruna['safe'], coruna['open'], coruna['contenders']) == (6, 2, 5)
    vox = by_name(coruna)['VOX']['tokens']
    assert [(t['safe'], t['central'], t['label']) for t in vox] == [(False, True, '37%')]


def test_district_views_keep_a_party_with_a_central_seat_and_no_tokens_and_never_count_below_zero():
    data = {'n_sim': 100, 'parties': ['A', 'B', 'C'], 'regions': [{'id': 1, 'name': 'Uno', 'seats': 2}],
            'rows': [odds_row(1, 'A', 1, 100, 100, 100), odds_row(1, 'B', 1), odds_row(1, 'C', 0, 40)]}
    uno = pages.district_views(data)[0]
    assert [(row['name'], len(row['tokens'])) for row in uno['rows']] == [('A', 3), ('B', 0), ('C', 1)]
    assert (uno['safe'], uno['open'], uno['contenders']) == (3, 0, 1)  # three safe seats for two: never minus one


def test_district_order_totals_and_sentences():
    districts = pages.district_views(ODDS)
    assert [d['name'] for d in pages.district_order(districts)] == ['Cádiz', 'La Coruña', 'Guadalajara']
    tie = [{'name': 'Ávila', 'open': 1, 'seats': 3}, {'name': 'Zamora', 'open': 1, 'seats': 3},
           {'name': 'Soria', 'open': 1, 'seats': 2}, {'name': 'Madrid', 'open': 1, 'seats': 38}]
    assert [d['name'] for d in pages.district_order(tie)] == ['Madrid', 'Ávila', 'Zamora', 'Soria']
    assert pages.scope_totals(districts) == {'seats': 19, 'safe': 14, 'open': 5, 'contenders': 11, 'live': 2, 'districts': 3}
    cadiz, guadalajara, _ = districts
    assert pages.district_sentence(cadiz) == ('De los 8 escaños, 5 son seguros (PP 2, PSOE 2, VOX 1). '
                                              'Los otros 3 se los disputan 6 aspirantes.')
    assert pages.district_sentence(guadalajara) == 'Los 3 escaños son seguros (PP 1, PSOE 1, VOX 1).'


def test_party_view_lists_the_most_doubtful_districts_first():
    view = pages.party_view(pages.district_views(ODDS), 'PSOE')
    assert (view['name'], view['seats'], view['safe'], view['open']) == ('PSOE', 5, 4, 3)
    assert [row['district']['name'] for row in view['rows']] == ['Cádiz', 'La Coruña']
    assert view['rows'][0]['district'] == {'id': 11, 'name': 'Cádiz', 'slug': 'cadiz', 'seats': 8}
    assert view['settled'] == [{'name': 'Guadalajara', 'slug': 'guadalajara', 'safe': 1}]
    assert pages.party_view(pages.district_views(ODDS), 'UP') == {
        'name': 'UP', 'seats': 0, 'safe': 0, 'open': 0, 'rows': [], 'settled': []}


def test_closest_seats_are_the_open_ones_nearest_to_a_coin_toss():
    closest = pages.closest_seats(pages.district_views(ODDS))
    assert len(closest) == 11
    assert closest[0] == {'district': 'Cádiz', 'slug': 'cadiz', 'party': 'PSOE', 'k': 3, 'share': 0.411,
                          'label': '41%', 'central': False}
    assert [(c['district'], c['party'], c['k']) for c in closest[1:4]] == [
        ('La Coruña', 'VOX', 1), ('La Coruña', 'PP', 4), ('Cádiz', 'AA', 1)]
    assert (closest[-1]['party'], closest[-1]['label']) == ('BNG', '1%')
    assert len(pages.closest_seats(pages.district_views(ODDS), limit=4)) == 4


def sentence_of(seats, safe_by_party, contenders):
    safe = sum(safe_by_party.values())
    return pages.district_sentence({'seats': seats, 'safe': safe, 'open': max(seats - safe, 0), 'contenders': contenders,
                                    'rows': [{'name': name, 'safe': n} for name, n in safe_by_party.items()]})


@pytest.mark.parametrize('seats, safe, contenders, text', [
    (1, {'PP': 1}, 0, 'El escaño es seguro (PP 1).'),
    (3, {'PP': 2, 'PSOE': 1}, 0, 'Los 3 escaños son seguros (PP 2, PSOE 1).'),
    (4, {'PP': 1, 'VOX': 0}, 5, 'De los 4 escaños, 1 es seguro (PP 1). Los otros 3 se los disputan 5 aspirantes.'),
    (3, {'PP': 1, 'PSOE': 1}, 2, 'De los 3 escaños, 2 son seguros (PP 1, PSOE 1). El otro se lo disputan 2 aspirantes.'),
    (3, {'PP': 1, 'PSOE': 1}, 1, 'De los 3 escaños, 2 son seguros (PP 1, PSOE 1). El otro se lo disputa 1 aspirante.'),
    (4, {'PP': 1, 'PSOE': 1}, 1, 'De los 4 escaños, 2 son seguros (PP 1, PSOE 1). Los otros 2 se los disputa 1 aspirante.'),
    (3, {'PP': 1, 'PSOE': 1}, 0, 'De los 3 escaños, 2 son seguros (PP 1, PSOE 1).'),
    (4, {'PP': 0}, 6, 'Ninguno de los 4 escaños es seguro: se los disputan 6 aspirantes.'),
    (4, {'PP': 0}, 1, 'Ninguno de los 4 escaños es seguro: se los disputa 1 aspirante.'),
    (4, {'PP': 0}, 0, 'Ninguno de los 4 escaños es seguro.'),
    (1, {'PP': 0}, 2, 'El escaño no es seguro: se lo disputan 2 aspirantes.'),
    (1, {'PP': 0}, 1, 'El escaño no es seguro: se lo disputa 1 aspirante.'),
    (1, {'PP': 0}, 0, 'El escaño no es seguro.'),
])
def test_district_sentence_agrees_with_the_number_of_seats_and_contenders(seats, safe, contenders, text):
    assert sentence_of(seats, safe, contenders) == text


def write_odds(fs, data=ODDS, modes=bundle.MODES):
    """Sobrescribe la parte `seat-odds` del run del bundle de prueba."""
    writer = bundle.BundleWriter(fs)
    for mode in modes:
        writer.write_json(bundle.path_part('es', RUN, 'seat-odds', mode), 'seat-odds', data, 'es', run_id=RUN, mode=mode)
    webapi.site().files.clear()


@pytest.fixture
def odds_site(api, fresh_app):
    write_odds(fresh_app.fs)
    return api


def page_of(site, path, query=''):
    status, _, body = call(site, path, query=query)
    return status, re.sub(r'\s+', ' ', html(body))


def test_the_page_opens_on_the_district_with_most_open_seats(odds_site):
    status, page = page_of(odds_site, '/es/circunscripciones')
    assert status == 200 and '<h1 class="headline">Escaños que bailan</h1>' in page
    kpis = page[page.index('class="kpis"'):page.index('class="seg lg so-tabs"')]
    assert [int(n) for n in re.findall(r'class="v"[^>]*>(\d+)', kpis)] == [14, 5, 11, 2]
    assert 'de 19' in kpis and 'de 3' in kpis
    items = re.findall(r'class="so-item[^"]*" href="([^"#?]+)', page)
    assert items == ['/es/circunscripciones/cadiz', '/es/circunscripciones/la-coruna', '/es/circunscripciones/guadalajara']
    card = page[page.index('id="ficha"'):]
    assert 'De los 8 escaños, 5 son seguros (PP 2, PSOE 2, VOX 1). Los otros 3 se los disputan 6 aspirantes.' in card
    assert re.findall(r'class="so-token[^"]*"[^>]*title="([^"]+)"', card)[:3] == [
        '1.º de PP: seguro', '2.º de PP: seguro', '3.º de PP: lo consigue en el 80% de las simulaciones']
    assert '<link rel="canonical" href="' in page and page.count('<details class="how"') >= 1
    assert_public_wording(page, 'circunscripciones')


def test_a_district_has_its_own_address_and_never_labels_an_open_seat_as_sure(odds_site):
    status, page = page_of(odds_site, '/es/circunscripciones/la-coruna')
    card = page[page.index('id="ficha"'):page.index('class="so-legend"')]
    # every title of the site starts with its name (`base.html`)
    assert status == 200 and 'La Coruña' in card and '<title>Demoscopio · Escaños que bailan en La Coruña' in page
    labels = re.findall(r'<span class="so-pct[^"]*">([^<]+)</span>', card)
    assert set(labels) == {'30%', '98%', '24%', '37%', '1%'} and not {'99%', '100%', '0%'} & set(labels)
    assert len(re.findall(r'class="so-token safe"', card)) == 6
    assert len(re.findall(r'class="so-token open out"', card)) == 3 and len(re.findall(r'class="so-token open"', card)) == 2
    closed = page_of(odds_site, '/es/circunscripciones/guadalajara')[1]
    assert 'Los 3 escaños son seguros (PP 1, PSOE 1, VOX 1).' in closed and 'decidida' in closed


def test_the_page_follows_the_mode_and_keeps_it_in_its_links(odds_site, fresh_app):
    nowcast = copy.deepcopy(ODDS)
    nowcast['rows'][3] = odds_row(11, 'AA', 1, 2500)  # AA's seat in Cádiz: 76% in the election mode, 25% today
    write_odds(fresh_app.fs, nowcast, modes=('nowcast',))
    assert '>76%<' in page_of(odds_site, '/es/circunscripciones/cadiz')[1]  # less than 60 days away: forecast
    today = page_of(odds_site, '/es/circunscripciones/cadiz', 'mode=nowcast')[1]
    assert '>25%<' in today and '>76%<' not in today
    assert 'href="/es/circunscripciones/la-coruna?mode=nowcast#ficha"' in today
    assert 'href="/es?mode=nowcast#territorio"' in today  # the index of the event bar goes back to the event page


def test_the_page_does_not_exist_without_the_part_or_with_a_single_district(api, odds_site, fresh_app, tmp_path):
    assert call(odds_site, '/es/circunscripciones/no-existe')[0] == 404
    status, headers, _ = call(odds_site, '/es/circunscripciones', query='circunscripcion=cadiz&mode=nowcast')
    # the form without JavaScript: on to the address of the district, keeping the mode
    assert status == 301 and headers['location'] == '/es/circunscripciones/cadiz?mode=nowcast'
    write_odds(fresh_app.fs, fixture_data('seat-odds'))
    status, page = page_of(api, '/es/circunscripciones')
    assert status == 404 and 'Este run no tiene el desglose escaño a escaño.' in page  # one district
    for mode in bundle.MODES:
        os.remove(os.path.join(str(tmp_path), bundle.PREFIX, bundle.path_part('es', RUN, 'seat-odds', mode)))
    webapi.site().files.clear()
    assert call(api, '/es/circunscripciones')[0] == 404  # a run published before the part existed


def test_a_pinned_run_stays_in_every_link_of_the_page_and_the_canonical_drops_it(odds_site):
    status, page = page_of(odds_site, '/es/circunscripciones/cadiz', 'mode=nowcast&run=' + RUN)
    query = '?mode=nowcast&amp;run=' + RUN
    assert status == 200
    assert 'href="/es/circunscripciones/guadalajara{}#ficha"'.format(query) in page
    assert 'href="/es/circunscripciones{}"'.format(query) in page  # the tab of the view
    assert 'href="/es/circunscripciones{}&amp;partido=pp"'.format(query) in page  # the first party with a token
    assert 'href="/es/circunscripciones{}&amp;vista=renidos"'.format(query) in page
    assert 'href="/es{}#tabla"'.format(query) in page  # the index of the event bar
    form = page[page.index('id="district-pick"'):page.index('</form>')]
    assert '<input type="hidden" name="run" value="{}">'.format(RUN) in form and 'value="nowcast"' in form
    assert '<option value="cadiz" selected>Cádiz · 8 escaños · 3 en el aire</option>' in form
    assert '<option value="guadalajara">Guadalajara · 3 escaños · decidida</option>' in form
    assert '<link rel="canonical" href="https://demoscopio.es/es/circunscripciones/cadiz">' in page
    status, headers, _ = call(odds_site, '/es/circunscripciones', query='mode=nowcast&run={}&circunscripcion=cadiz'.format(RUN))
    assert status == 301 and headers['location'] == '/es/circunscripciones/cadiz?mode=nowcast&run=' + RUN


def test_the_list_sorts_itself_without_javascript_and_the_rows_read_aloud(odds_site):
    page = page_of(odds_site, '/es/circunscripciones')[1]
    # the order by open seats is the order of the markup; the alphabetical one, a position for the CSS
    assert re.findall(r'style="--by-open: (\d+); --by-name: (\d+)"', page) == [('1', '1'), ('2', '3'), ('3', '2')]
    assert 'name="orden" value="az">A–Z' in page and 'class="so-item on" href="/es/circunscripciones/cadiz?' in page
    assert 'aria-label="PP: 2 seguros; 3.º, 80%; 4.º, 7%"' in page and 'aria-label="AA: 1.º, 76%"' in page
    assert '<div class="so-note">2 seguros · 2 en el aire</div>' in page and '<div class="so-note">1 en el aire</div>' in page
    legend = page[page.index('class="so-legend"'):]
    assert 'lo consigue en el 99% o más de las simulaciones' in legend and 'por debajo del 1% no se dibuja' in legend
    assert 'data-file="demoscopio-escanos-es-cadiz.png"' in page and '/dist/js/pages/circunscripciones.js' in page
    assert ('<meta name="description" content="Los escaños de Cádiz, uno a uno: cuáles tiene seguros cada partido y cuáles '
            'están en el aire, según las simulaciones del modelo para Elecciones generales 2026.">') in page
    assert 'según las 10.000 simulaciones del modelo.' in page


def test_the_texts_of_a_row_and_the_link_of_a_district():
    assert pages.tokens_note({'safe': 2, 'open': 2}) == '2 seguros · 2 en el aire'
    assert pages.tokens_note({'safe': 1, 'open': 0}) == '1 seguro' and pages.tokens_note({'safe': 0, 'open': 0}) == '—'
    row = by_name(pages.district_views(ODDS)[1])['PP']  # Guadalajara: one safe seat, the second below 1%
    assert pages.tokens_reading('PP', row) == 'PP: 1 seguro'
    assert pages.tokens_reading('B', {'safe': 0, 'tokens': []}) == 'B: ningún escaño llega al 1%'
    assert (pages.threshold_label(pages.SAFE_SHARE), pages.threshold_label(pages.OPEN_FLOOR)) == ('99%', '1%')
    assert pages.tint('#1d84ce', 0.2) == 'rgba(29, 132, 206, 0.2)' and pages.tint('#fff', 0.5) == 'rgba(255, 255, 255, 0.5)'
    assert pages.tint(None, 0.5) == 'rgba(158, 158, 158, 0.5)'  # the grey of the others
    drawn = pages.drawn_row(by_name(pages.district_views(ODDS)[0])['PP'], 'PP', '#1d84ce')
    assert [token['fill'] for token in drawn['tokens']][:3] == [None, None, pages.tint('#1d84ce', drawn['tokens'][2]['alpha'])]
    assert pages.district_href('es', 'la-coruna', '?mode=nowcast') == '/es/circunscripciones/la-coruna?mode=nowcast#ficha'
