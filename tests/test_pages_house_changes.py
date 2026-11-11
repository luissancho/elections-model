"""La tarjeta «Los cambios de cada casa» de la página del evento: `pages.house_changes` y su matriz en `/{scope}`."""
from mtpy.lib import pages
from mtpy.lib.pages.filters import DASH, gradient
from mtpy.lib.pages.slugs import poll_detail_href

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
