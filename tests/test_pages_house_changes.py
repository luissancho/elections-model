"""La tarjeta «Los cambios de cada casa» de la página del evento: `pages.house_changes` y su matriz en `/{scope}`."""
import re

from mtpy.lib import bundle, pages, webapi
from mtpy.lib.pages.filters import DASH, gradient
from mtpy.lib.pages.slugs import poll_detail_href
from tests.test_api_asgi import call
from tests.test_pages import api, assert_public_wording, html  # noqa: F401  (fixtures y ayudas de las páginas)
from tests.test_webapi_unit import RUN, fixture_data

AS_OF = '2026-11-11'
NAMES = ['PP', 'PSOE', 'UP']
COLORS = {'PP': '#1d84ce', 'PSOE': '#ef1c27', 'UP': '#7b4977'}


def poll(date, pollster_id, pollster, sponsor, rating, seats=None, **values):
    """Registro de la parte `polls`: la casa, el medio, el rating, los escaños que publica y una cifra por partido."""
    record = {'date': date, 'pollster_id': pollster_id, 'pollster': pollster, 'sponsor': sponsor, 'rating': rating,
              **values}
    return record | ({'seats': seats} if seats else {})


SIGMA_SEATS = ({'PP': [133, 139], 'PSOE': [97, 103]}, {'PP': [131, 137], 'PSOE': [101, 106]})
SW_SEATS = ({'PP': [116, 116]}, {'PP': [113, 119], 'PSOE': [127, 132]})
POLLS = [
    poll('2026-10-13', 1, 'Sigma Dos', 'El Mundo', 65.9, PP=30.0, PSOE=26.0),  # before the window
    poll('2026-10-30', 1, 'Sigma Dos', 'El Mundo', 65.9, SIGMA_SEATS[0], PP=31.1, PSOE=24.6, UP=1.9),
    poll('2026-11-06', 1, 'Sigma Dos', 'El mundo', 65.9, SIGMA_SEATS[1], PP=31.0, PSOE=25.2, UP=1.6),
    poll('2026-10-14', 2, 'GAD3', 'Mediaset', 71.0, PP=31.4, PSOE=25.0),  # one delivery per outlet: no line with two
    poll('2026-11-05', 2, 'GAD3', 'ABC', 71.0, PP=31.9, PSOE=24.5),
    poll('2026-11-08', 3, 'SW Demoscopia', 'La Séptima', 25.5, SW_SEATS[0], PP=28.5, PSOE=31.1, UP=1.2),
    poll('2026-11-09', 3, 'SW Demoscopia', 'La Séptima', 25.5, SW_SEATS[1], PP=28.5, PSOE=31.2),
    poll('2026-10-28', None, 'Sondaxe', None, None, PP=29.2, PSOE=26.9),
    poll('2026-11-04', None, 'Sondaxe', None, None, PP=29.7, PSOE=27.3, UP=2.2),
    poll('2026-10-14', 5, 'IMOP', 'El Confidencial', 62.8, PP=28.0),  # the first day of the window
    poll('2026-11-06', 5, 'IMOP', 'El Confidencial', 62.8, PP=29.8),
    poll('2026-10-13', 6, '40dB', 'Prisa', 60.0, PP=30.0),  # the day before the window
    poll('2026-11-04', 6, '40dB', 'Prisa', 60.0, PP=30.5),
]


def test_house_changes_compare_the_last_two_deliveries_of_each_line_in_the_window():
    changes = pages.house_changes(POLLS, NAMES, COLORS, AS_OF, 'es')
    view = changes['pct']
    assert changes['days'] == 28 == pages.HOUSE_CHANGES_DAYS
    assert view['parties'] == [{'name': name, 'color': COLORS[name]} for name in NAMES]
    assert [row['pollster'] for row in view['rows']] == ['Sigma Dos', 'IMOP', 'SW Demoscopia', 'Sondaxe']
    sigma, imop, sw, sondaxe = view['rows']
    assert sigma['cells'] == [{'prev': '31,1', 'value': '31,0', 'change': -0.1},
                              {'prev': '24,6', 'value': '25,2', 'change': 0.6},
                              {'prev': '1,9', 'value': '1,6', 'change': -0.3}]
    assert (sigma['rating'], sigma['rating_color']) == ('65,9', gradient(65.9, 0, 100, pages.RATING_COLORS))
    assert (sigma['sponsor'], sigma['prev_date'], sigma['last_date']) == ('El mundo', '30 Oct', '6 Nov')
    assert sigma['gap_days'] == 7
    assert sigma['href'] == pages.poll_href(POLLS[2])
    assert sigma['prev_href'] == poll_detail_href('es', POLLS[1])
    assert sigma['last_href'] == poll_detail_href('es', POLLS[2])
    assert imop['cells'] == [{'prev': '28,0', 'value': '29,8', 'change': 1.8}, None, None]
    assert sw['cells'] == [{'prev': '28,5', 'value': '28,5', 'change': 0.0},
                           {'prev': '31,1', 'value': '31,2', 'change': 0.1}, None]  # UP: gone from the last delivery
    assert sw['gap_days'] == 1
    assert sondaxe['cells'][2] == {'prev': None, 'value': '2,2', 'change': None}  # UP: new in the last delivery
    assert (sondaxe['href'], sondaxe['rating'], sondaxe['rating_color'], sondaxe['sponsor']) == (None, DASH, None, None)
    assert view['counts'] == {'PP': {'up': 2, 'down': 1, 'flat': 1}, 'PSOE': {'up': 3, 'down': 0, 'flat': 0},
                              'UP': {'up': 0, 'down': 1, 'flat': 0}}


def test_house_changes_seats_are_the_houses_own_ranges_by_their_midpoints():
    view = pages.house_changes(POLLS, NAMES, COLORS, AS_OF, 'es')['seats']
    assert [party['name'] for party in view['parties']] == ['PP', 'PSOE']
    assert [row['pollster'] for row in view['rows']] == ['Sigma Dos', 'SW Demoscopia']
    assert view['rows'][0]['cells'] == [{'prev': f'133{DASH}139', 'value': f'131{DASH}137', 'change': -2},
                                        {'prev': f'97{DASH}103', 'value': f'101{DASH}106', 'change': 4}]
    assert view['rows'][1]['cells'] == [{'prev': '116', 'value': f'113{DASH}119', 'change': 0},
                                        {'prev': None, 'value': f'127{DASH}132', 'change': None}]
    assert view['counts'] == {'PP': {'up': 0, 'down': 1, 'flat': 1}, 'PSOE': {'up': 1, 'down': 0, 'flat': 0}}
    without = [{key: value for key, value in record.items() if key != 'seats'} for record in POLLS]
    assert pages.house_changes(without, NAMES, COLORS, AS_OF, 'es')['seats'] is None


def test_house_changes_without_two_deliveries_in_the_window_or_without_the_day_is_none():
    assert pages.house_changes([], NAMES, COLORS, AS_OF, 'es') is None
    assert pages.house_changes(POLLS[:2] + POLLS[3:5] + POLLS[11:], NAMES, COLORS, AS_OF, 'es') is None
    for day in (None, '', 'ayer'):
        assert pages.house_changes(POLLS, NAMES, COLORS, day, 'es') is None
    week = pages.house_changes(POLLS, NAMES, COLORS, AS_OF, 'es', days=7)
    assert [row['pollster'] for row in week['pct']['rows']] == ['SW Demoscopia'] and week['days'] == 7


def test_house_changes_same_day_deliveries_rating_ties_and_no_scope():
    polls = [poll('2026-11-05', 9, 'Zeta', 'M', 50.0, PP=30.0), poll('2026-11-05', 9, 'Zeta', 'M', 50.0, PP=30.4),
             poll('2026-11-01', 8, 'Alfa', None, 50.0, PP=29.0),
             poll('2026-11-03', 8, 'Alfa', None, 50.0, PP=29.0, PSOE=27.0)]
    view = pages.house_changes(polls, NAMES, COLORS, AS_OF)['pct']
    assert [row['pollster'] for row in view['rows']] == ['Alfa', 'Zeta']  # the same rating: by name
    zeta = view['rows'][1]
    assert zeta['gap_days'] == 0 and zeta['cells'][0] == {'prev': '30,0', 'value': '30,4', 'change': 0.4}
    assert (zeta['prev_href'], zeta['last_href']) == (None, None)
    assert [party['name'] for party in view['parties']] == ['PP', 'PSOE']  # UP has no figure in any row
    assert view['rows'][0]['cells'][1] == {'prev': None, 'value': '27,0', 'change': None}
    assert view['counts']['PSOE'] == {'up': 0, 'down': 0, 'flat': 0}


PAGE_POLLS = [
    poll('2026-09-29', 7, 'Sigma Dos', 'El Mundo', 65.9, {'PP': [133, 139]}, PP=31.1),
    poll('2026-10-06', 7, 'Sigma Dos', 'El Mundo', 65.9, {'PP': [131, 137]}, PP=31.0, PSOE=25.2),
    poll('2026-10-06', 3, 'GAD3', 'Mediaset', 71.0, PP=31.4, PSOE=25.0),
    poll('2026-10-07', 3, 'GAD3', 'Mediaset', 71.0, PP=31.4),
    poll('2026-10-02', None, 'Sondaxe', None, None, PP=29.2, PSOE=26.9),
    poll('2026-10-02', None, 'Sondaxe', None, None, PP=29.7, PSOE=27.3),
]


def write_polls(fresh_app, polls, parties=('PP', 'PSOE')):
    """Sustituye los sondeos de la parte `polls` del run de pruebas."""
    part = fixture_data('polls') | {'parties': list(parties), 'polls': polls}
    writer = bundle.BundleWriter(fresh_app.fs)
    writer.write_json(bundle.path_part('es', RUN, 'polls'), 'polls', part, 'es', run_id=RUN)
    webapi.site().files.clear()


def changes_card(api, query=''):
    """La tarjeta de `/es` en una línea, y la página entera."""
    page = re.sub(r'\s+', ' ', html(call(api, '/es', query=query)[2]))
    start = page.index('<section class="card shot" id="house-changes"')
    return page[start:page.index('</section>', start)], page


def test_event_page_has_no_changes_card_without_a_line_with_two_deliveries(api):
    assert 'id="house-changes"' not in html(call(api, '/es')[2])


def test_event_page_changes_card_follows_the_average_with_both_views(api, fresh_app):
    write_polls(fresh_app, PAGE_POLLS)
    card, page = changes_card(api)
    assert page.index('id="series"') < page.index('id="house-changes"') < page.index('id="escanos"')
    assert '<h2>Los cambios de cada casa</h2>' in card
    assert 'class="card-img-btn" hidden data-file="demoscopio-cambios-es-2026-10-08.png"' in card
    assert '<fieldset id="changes-metric" class="seg" hidden>' in card
    radios = re.findall(r'<label><input type="radio" name="hcmetric" value="(\w+)"( checked)?> ([^<]+)</label>', card)
    assert radios == [('pct', ' checked', '% Voto'), ('seats', '', 'Escaños')]
    pct, seats = card.split('id="changes-seats" data-shot-fit hidden>')
    assert '<div class="hc-wrap" id="changes-pct" data-shot-fit>' in pct
    # by rating, the house without one last
    assert pct.index('<b>GAD3</b>') < pct.index('<b>Sigma Dos</b>') < pct.index('<b>Sondaxe</b>')
    assert re.search(r'<span class="rt" style="background: #[0-9a-f]{6}">65,9</span>', pct)
    assert ('El Mundo · <a href="/es/sondeos/2026-09-29/sigma-dos/el-mundo">29 Sep</a> → '
            '<a href="/es/sondeos/2026-10-06/sigma-dos/el-mundo">6 Oct</a> (7 días)') in pct
    assert '(1 día)' in pct and '(mismo día)' in pct
    sondaxe = pct[pct.index('<b>Sondaxe</b>'):]
    sondaxe = sondaxe[:sondaxe.index('</th>')]
    assert '<a href="/casas/' not in sondaxe and 'class="rt"' not in sondaxe  # no id, no rating
    assert re.search(r'<span class="dlt down">▼ 0,1</span>\s*<small>31,1 → 31,0</small>', pct)
    assert re.search(r'<span class="dlt flat">=</span>\s*<small>31,4 → 31,4</small>', pct)
    assert re.search(r'<td>\s*<small>25,2</small>\s*</td>', pct)  # Sigma Dos, PSOE: new in the last delivery
    assert '<td class="na">—</td>' in pct  # GAD3, PSOE: no figure in the last delivery
    foot = pct[pct.index('<tfoot>'):]
    assert 'Suben · bajan · igual' in foot
    count = '<span class="dlt up">▲ {}</span> <span class="dlt down">▼ {}</span> <span class="dlt flat">= {}</span>'
    assert foot.count(count.format(1, 1, 1)) == 1 and count.format(1, 0, 0) in foot  # PP, then PSOE
    assert '<b>GAD3</b>' not in seats and '<b>Sondaxe</b>' not in seats
    assert re.search(r'<span class="dlt down">▼ 2</span>\s*<small>133–139 → 131–137</small>', seats)
    for phrase in ('La comparación es siempre de una casa consigo misma', 'Mira el conjunto, no una celda',
                   'no un reparto nuestro', 'aquí sólo se ve hacia dónde lo mueve'):
        assert phrase in card, phrase
    assert_public_wording(page, '/es con los cambios de cada casa')
    assert 'id="house-changes"' in changes_card(api, 'mode=nowcast')[1]


def test_event_page_changes_card_without_seats_has_no_switch(api, fresh_app):
    write_polls(fresh_app, [{key: value for key, value in record.items() if key != 'seats'}
                            for record in PAGE_POLLS])
    card, _ = changes_card(api)
    assert 'id="changes-metric"' not in card and 'id="changes-seats"' not in card and 'id="changes-pct"' in card
