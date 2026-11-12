"""La página «Escaños que bailan» (`/{scope}/circunscripciones`): de la parte `seat-odds` a las fichas y sus vistas."""
import copy
import os
import re

import pytest

from mtpy.lib import bundle, pages, webapi
from tests.test_api_asgi import call
from tests.test_pages import api, assert_public_wording, counting, html, initial  # noqa: F401
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
                                           (0.0149, '1%'), (0.01, '1%'),
                                           # exact halves that the binary floats put a hair below: rounded up too
                                           (0.145, '15%'), (0.285, '29%'), (0.565, '57%'), (0.575, '58%'),
                                           (1450 / 10000, '15%'), (5750 / 10000, '58%')])
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
    webapi.site().manifest_cache_clear()  # the presence of the part is remembered under the TTL of the pointers


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
    assert 'href="/es/circunscripciones/guadalajara{}#ficha"'.format(query) in page
    # the tabs of the views land on themselves (`#vistas`), not at the top of the page
    assert '<nav class="seg lg so-tabs" id="vistas" aria-label="Vistas">' in page
    assert 'href="/es/circunscripciones{}#vistas"'.format(query) in page  # the tab of the view
    assert 'href="/es/circunscripciones{}&amp;partido=pp#vistas"'.format(query) in page  # the first party with a token
    assert 'href="/es/circunscripciones{}&amp;vista=renidos#vistas"'.format(query) in page
    assert 'href="/es{}#tabla"'.format(query) in page  # the index of the event bar
    form = page[page.index('id="district-pick"'):page.index('</form>')]
    assert '<input type="hidden" name="run" value="{}">'.format(RUN) in form and 'value="nowcast"' in form
    assert '<option value="cadiz" selected>Cádiz · 8 escaños · 3 en el aire</option>' in form
    assert '<option value="guadalajara">Guadalajara · 3 escaños · decidida</option>' in form
    assert '<link rel="canonical" href="https://demoscopio.es/es/circunscripciones/cadiz">' in page
    status, headers, _ = call(odds_site, '/es/circunscripciones', query='mode=nowcast&run={}&circunscripcion=cadiz'.format(RUN))
    assert status == 301 and headers['location'] == '/es/circunscripciones/cadiz?mode=nowcast&run={}#ficha'.format(RUN)


def test_the_list_sorts_itself_without_javascript_and_the_rows_read_aloud(odds_site):
    page = page_of(odds_site, '/es/circunscripciones')[1]
    # the order by open seats is the order of the markup; the alphabetical one, a position for the CSS
    assert re.findall(r'style="--by-open: (\d+); --by-name: (\d+)"', page) == [('1', '1'), ('2', '3'), ('3', '2')]
    assert 'name="orden" value="az">A–Z' in page and 'class="so-item on" href="/es/circunscripciones/cadiz?' in page
    # the seats of each item, read aloud with their unit
    assert '<small title="8 escaños">8<span class="sr-only"> escaños</span></small>' in page
    assert 'aria-label="PP: 2 seguros; 3.º, 80%; 4.º, 7%"' in page and 'aria-label="AA: 1.º, 76%"' in page
    assert '<div class="so-note">2 seguros · 2 en el aire</div>' in page and '<div class="so-note">1 en el aire</div>' in page
    legend = page[page.index('class="so-legend"'):]
    assert 'lo consigue en el 99% o más de las simulaciones' in legend and 'por debajo del 1% no se dibuja' in legend
    assert 'data-file="demoscopio-escanos-es-cadiz.png"' in page and '/dist/js/pages/circunscripciones.js' in page
    assert 'según las 10.000 simulaciones del modelo.' in page
    # the address without a district, a party or a view is shared as the page, not as its first district (which
    # changes between runs); its card is still that district's, with the same image
    assert '<title>Demoscopio · Escaños que bailan · Elecciones generales 2026</title>' in page
    assert ('<meta name="description" content="Qué escaños tiene seguros cada partido en cada circunscripción y cuáles '
            'están en el aire, según las simulaciones del modelo para Elecciones generales 2026.">') in page
    assert '<meta property="og:title" content="Demoscopio · Escaños que bailan · Elecciones generales 2026">' in page
    assert '<h2>Cádiz</h2>' in page
    cadiz = page_of(odds_site, '/es/circunscripciones/cadiz')[1]
    assert '<title>Demoscopio · Escaños que bailan en Cádiz · Elecciones generales 2026</title>' in cadiz
    assert ('<meta name="description" content="Los escaños de Cádiz, uno a uno: cuáles tiene seguros cada partido y cuáles '
            'están en el aire, según las simulaciones del modelo para Elecciones generales 2026.">') in cadiz


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
    assert re.findall(r'class="so-party-btn[^"]*" href="[^"]*partido=([a-z]+)#vistas"', page) == [
        'pp', 'psoe', 'vox', 'aa', 'bng']
    card = page[page.index('id="ficha"'):page.index('class="so-legend"')]
    assert re.findall(r'href="/es/circunscripciones/([a-z-]+)\?mode=forecast#ficha"', card) == ['cadiz', 'la-coruna']
    assert 'Tiene 4 escaños seguros y aspira a otros 3 en 2 circunscripciones.' in card
    assert 'Sin nada en el aire (1):' in card and 'Guadalajara 1' in card
    assert '<link rel="canonical" href="https://demoscopio.es/es/circunscripciones?partido=psoe">' in page
    assert 'data-file="demoscopio-escanos-es-psoe.png"' in page and page.count('<details class="how"') == 1
    assert ('<meta name="description" content="Los escaños de PSOE, circunscripción a circunscripción: cuáles tiene '
            'seguros y cuáles están en el aire, según las simulaciones del modelo para Elecciones generales 2026.">') in page
    assert_public_wording(page, 'circunscripciones por partido')
    assert call(odds_site, '/es/circunscripciones', query='partido=nadie')[0] == 404
    kept = page_of(odds_site, '/es/circunscripciones', 'partido=pp&mode=nowcast')[1]
    assert 'href="/es/circunscripciones/cadiz?mode=nowcast#ficha"' in kept
    assert 'class="so-party-btn on" href="/es/circunscripciones?mode=nowcast&amp;partido=pp#vistas"' in kept
    # the district of the path wins over the party
    assert '<title>Demoscopio · Escaños que bailan en Cádiz' in page_of(odds_site, '/es/circunscripciones/cadiz', 'partido=pp')[1]


def test_the_closest_view_ranks_the_open_seats_by_their_distance_to_a_coin_toss(odds_site):
    status, page = page_of(odds_site, '/es/circunscripciones', 'vista=renidos')
    table = page[page.index('id="closest"'):]
    # under 720 px each row stacks in two lines (`display: grid`), so the table carries its ARIA roles
    assert table.startswith('id="closest" role="table"') and table.count('scope="col" role="columnheader"') == 4
    rows = re.findall(r'<tr role="row">(.*?)</tr>', table[table.index('<tbody role="rowgroup">'):])
    assert status == 200 and len(rows) == 11
    assert 'Cádiz' in rows[0] and '3.º de PSOE' in rows[0] and '41%' in rows[0] and 'no se lo da' in rows[0]
    assert 'La Coruña' in rows[1] and '1.º de VOX' in rows[1] and 'se lo da' in rows[1] and 'no se lo da' not in rows[1]
    # the central state is in every row, on every width: no class that hides it on phones, and «Central: » before it
    # there (`so-narrow` hides only from 720 px up, where the column has its header)
    central = r'<td class="so-central" role="cell"><span class="so-narrow">Central: </span>(?:no )?se lo da</td>'
    assert all(re.search(central, row) for row in rows) and not any('so-wide' in row for row in rows)
    assert '<link rel="canonical" href="https://demoscopio.es/es/circunscripciones?vista=renidos">' in page
    assert 'Los 11 escaños más cercanos al cara o cruz, de los 11 que están en el aire. La raya marca el 50%.' in page
    assert '<title>Demoscopio · Los escaños más reñidos · Elecciones generales 2026' in page
    assert 'class="seg lg so-tabs"' in page and re.findall(r'<a href="[^"]*"( class="on")', page) == [' class="on"']
    assert_public_wording(page, 'circunscripciones, los más reñidos')
    assert call(odds_site, '/es/circunscripciones', query='vista=otra')[0] == 400


def test_the_sentences_of_the_party_and_of_the_closest_seats_agree_with_their_numbers():
    assert pages.party_sentence({'safe': 4, 'open': 3, 'rows': [1, 2]}) == (
        'Tiene 4 escaños seguros y aspira a otros 3 en 2 circunscripciones. Primero, las más dudosas.')
    assert pages.party_sentence({'safe': 1, 'open': 1, 'rows': [1]}) == 'Tiene 1 escaño seguro y aspira a otro en 1 circunscripción.'
    assert pages.party_sentence({'safe': 2, 'open': 0, 'rows': []}) == 'Tiene 2 escaños seguros y ninguno en el aire.'
    assert pages.closest_sentence(0, 0) == 'Ningún escaño está en el aire.'
    assert pages.closest_sentence(1, 1) == 'El único escaño que está en el aire. La raya marca el 50%.'
    assert pages.closest_sentence(40, 55).startswith('Los 40 escaños más cercanos al cara o cruz, de los 55 que')


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
    for query in ('', 'partido=psoe'):
        page = page_of(odds_site, '/es/circunscripciones', query)[1]
        suffix = re.search(r'<span class="pill brand">([^<]+)</span>', page).group(1)  # the pill of the event bar
        card = page[page.index('id="ficha"'):page.index('class="so-legend"')]
        stamp = re.search(r'<p class="so-stamp shot-only">([^<]+)</p>', card)
        # only in the image (`.card:not(.shot-image) .shot-only` is hidden): the same district exists in the general
        # and in a regional scope, and the figures change every day
        assert stamp and stamp.group(1).startswith('Elecciones generales 2026 · {} · Actualizado: '.format(suffix))
        assert re.search(r' · Actualizado: \d', stamp.group(1))  # the day the run was computed, written out


@pytest.mark.parametrize('path, query, canonical', [
    # the route of a district ignores the party, the view and the dropdown
    ('/es/circunscripciones/cadiz', 'partido=pp&vista=x&circunscripcion=', '/es/circunscripciones/cadiz'),
    # the party wins over the view
    ('/es/circunscripciones', 'partido=pp&vista=renidos', '/es/circunscripciones?partido=pp'),
    # the dropdown never stays (an empty one is ignored by the page)
    ('/es/circunscripciones', 'vista=renidos&circunscripcion=&mode=nowcast', '/es/circunscripciones?vista=renidos'),
])
def test_the_canonical_keeps_only_what_the_page_shows(odds_site, path, query, canonical):
    status, page = page_of(odds_site, path, query)
    assert status == 200 and '<link rel="canonical" href="https://demoscopio.es{}">'.format(canonical) in page


def test_the_party_wins_over_the_view_and_the_errors_and_units_read_well(odds_site):
    page = page_of(odds_site, '/es/circunscripciones', 'partido=psoe&vista=renidos')[1]
    assert '<title>Demoscopio · Escaños que bailan: PSOE' in page and 'id="closest"' not in page
    status, page = page_of(odds_site, '/es/circunscripciones/no-existe')
    assert status == 404 and 'Circunscripción no encontrada.' in page
    kpis = page_of(odds_site, '/es/circunscripciones')[1]
    assert '<div class="l">Aspirantes</div><div class="v">11<small>a esos 5</small></div>' in kpis


def test_a_party_with_only_safe_seats_has_no_grid_and_says_so(odds_site, fresh_app):
    settled = copy.deepcopy(ODDS)
    settled['rows'][3] = odds_row(11, 'AA', 1, 10000)  # AA's seat in Cádiz, safe: nothing of AA in the air
    write_odds(fresh_app.fs, settled)
    status, page = page_of(odds_site, '/es/circunscripciones', 'partido=aa')
    card = page[page.index('id="ficha"'):page.index('class="so-legend"')]
    assert status == 200 and 'Tiene 1 escaño seguro y ninguno en el aire.' in card
    assert 'so-grid' not in card and '<b>Sin nada en el aire (1):</b> Cádiz 1.' in card
