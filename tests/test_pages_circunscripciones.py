"""La página «Escaños que bailan» (`/{scope}/circunscripciones`): de la parte `seat-odds` a las fichas y sus vistas."""
import pytest

from mtpy.lib import pages


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
