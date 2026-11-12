"""
The seats that dance (``/{scope}/circunscripciones``): from the ``seat-odds`` part to the tokens of every district and
the views built on them (per district, per party, the closest seats).

A seat is *safe* when the party reaches it in at least ``SAFE_SHARE`` of the simulations and *open* when it does so in
at least ``OPEN_FLOOR`` but fewer than ``SAFE_SHARE``; below ``OPEN_FLOOR`` it is neither drawn nor counted. All
functions are pure: they take the data of the part and return plain dicts and lists.
"""
from .escanos import _sort_name
from .slugs import slugify

SAFE_SHARE = 0.99
OPEN_FLOOR = 0.01
CLOSEST_SHOWN = 40


def token_label(share: float) -> str:
    """
    Label of an open token: a whole percentage, half rounded up, kept between 1% and 98%.

    Parameters
    ----------
    share : float
        Share of the simulations in which the party reaches the seat (0 to 1).

    Returns
    -------
    str
        The percentage with the ``%`` attached; never 0%, 99% nor 100%.
    """
    return '{}%'.format(min(98, max(1, int(share * 100 + 0.5))))


def seat_tokens(name: str, reach: list, seats: int, n_sim: int) -> list:
    """
    Tokens of a party in a district: one per seat it reaches in at least ``OPEN_FLOOR`` of the simulations.

    Parameters
    ----------
    name : str
        Party name, used in the title of each token.
    reach : list
        ``reach[k - 1]`` is the number of simulations in which the party wins at least ``k`` seats.
    seats : int
        Seats of the party in the central allocation.
    n_sim : int
        Number of simulations.

    Returns
    -------
    list
        One dict per token, in seat order: ``k``, ``share``, ``safe``, ``central`` (``k <= seats``), ``label`` and
        ``alpha`` (``None`` when safe) and ``title``.
    """
    tokens = []
    for k, count in enumerate(reach, start=1):
        share = count / n_sim
        if share < OPEN_FLOOR:
            continue
        safe = share >= SAFE_SHARE
        label = None if safe else token_label(share)
        what = 'seguro' if safe else 'lo consigue en el {} de las simulaciones'.format(label)
        tokens.append({
            'k': k,
            'share': share,
            'safe': safe,
            'central': k <= seats,
            'label': label,
            'alpha': None if safe else round(0.08 + 0.42 * share, 3),
            'title': '{}.º de {}: {}'.format(k, name, what),
        })
    return tokens


def district_views(data: dict) -> list:
    """
    One view per district of the ``seat-odds`` part, in the order of the part.

    Parameters
    ----------
    data : dict
        Data of the ``seat-odds`` part (``n_sim``, ``regions``, ``rows``).

    Returns
    -------
    list
        Dicts with ``id``, ``name``, ``slug``, ``seats``, ``safe``, ``open`` (``max(seats - safe, 0)``),
        ``contenders`` (open tokens) and ``rows``: one per party with some token or some central seat, with
        ``name``, ``seats``, ``tokens``, ``safe``, ``open`` and ``expected`` (mean seats), by central seats and then
        expected seats, both descending.
    """
    n_sim = data['n_sim']
    views = []
    for region in data['regions']:
        rows = []
        for row in data['rows']:
            if row['region_id'] != region['id']:
                continue
            tokens = seat_tokens(row['name'], row['reach'], row['seats'], n_sim)
            if not tokens and not row['seats']:
                continue
            rows.append({
                'name': row['name'],
                'seats': row['seats'],
                'tokens': tokens,
                'safe': sum(token['safe'] for token in tokens),
                'open': sum(not token['safe'] for token in tokens),
                'expected': sum(row['reach']) / n_sim,
            })
        rows.sort(key=lambda row: (-row['seats'], -row['expected']))
        safe = sum(row['safe'] for row in rows)
        views.append({
            'id': region['id'],
            'name': region['name'],
            'slug': slugify(region['name']),
            'seats': region['seats'],
            'safe': safe,
            'open': max(region['seats'] - safe, 0),
            'contenders': sum(row['open'] for row in rows),
            'rows': rows,
        })
    return views


def district_order(districts: list) -> list:
    """
    Districts sorted by how much is in play.

    Parameters
    ----------
    districts : list
        Views from ``district_views``.

    Returns
    -------
    list
        A new list: more open seats first; on a tie, more seats, then the name with accents folded.
    """
    return sorted(districts, key=lambda d: (-d['open'], -d['seats'], _sort_name(d['name'])))


def scope_totals(districts: list) -> dict:
    """
    Totals of a scope.

    Parameters
    ----------
    districts : list
        Views from ``district_views``.

    Returns
    -------
    dict
        ``seats``, ``safe``, ``open``, ``contenders``, ``live`` (districts with open seats) and ``districts``.
    """
    return {
        'seats': sum(d['seats'] for d in districts),
        'safe': sum(d['safe'] for d in districts),
        'open': sum(d['open'] for d in districts),
        'contenders': sum(d['contenders'] for d in districts),
        'live': sum(d['open'] > 0 for d in districts),
        'districts': len(districts),
    }


def district_sentence(district: dict) -> str:
    """
    Plain-language summary of a district: how many seats are safe, for whom, and how many contenders share the rest.

    Parameters
    ----------
    district : dict
        View from ``district_views``.

    Returns
    -------
    str
        The sentence, in Spanish.
    """
    seats, safe, left, contenders = district['seats'], district['safe'], district['open'], district['contenders']
    listing = ', '.join('{} {}'.format(row['name'], row['safe']) for row in district['rows'] if row['safe'])
    if left == 0:
        if seats == 1:
            return 'El escaño es seguro ({}).'.format(listing)
        return 'Los {} escaños son seguros ({}).'.format(seats, listing)
    # The verb agrees with the number of contenders; with none the clause is dropped.
    if contenders == 0:
        dispute = ''
    elif contenders == 1:
        dispute = 'disputa 1 aspirante'
    else:
        dispute = 'disputan {} aspirantes'.format(contenders)
    if safe == 0:
        if seats == 1:
            return 'El escaño no es seguro: se lo {}.'.format(dispute) if dispute else 'El escaño no es seguro.'
        if dispute:
            return 'Ninguno de los {} escaños es seguro: se los {}.'.format(seats, dispute)
        return 'Ninguno de los {} escaños es seguro.'.format(seats)
    how_many = '1 es seguro' if safe == 1 else '{} son seguros'.format(safe)
    sentence = 'De los {} escaños, {} ({}).'.format(seats, how_many, listing)
    if not dispute:
        return sentence
    if left == 1:
        return '{} El otro se lo {}.'.format(sentence, dispute)
    return '{} Los otros {} se los {}.'.format(sentence, left, dispute)


def _coin_toss_gap(tokens: list) -> float:
    """
    Distance to 50% of the open token nearest to it.

    Parameters
    ----------
    tokens : list
        Tokens of a party in a district, with at least one open.

    Returns
    -------
    float
        The smallest ``abs(share - 0.5)`` among the open tokens.
    """
    return min(abs(token['share'] - 0.5) for token in tokens if not token['safe'])


def party_view(districts: list, name: str) -> dict:
    """
    View of a party across the districts of a scope.

    Parameters
    ----------
    districts : list
        Views from ``district_views``.
    name : str
        Party name.

    Returns
    -------
    dict
        ``name``, ``seats`` (central), ``safe``, ``open``, ``rows`` (districts where it has an open token, with
        ``district``, ``seats``, ``tokens``, ``safe`` and ``open``, the one nearest to a coin toss first) and
        ``settled`` (``name``, ``slug``, ``safe`` of the districts where it has only safe tokens, most seats first).
    """
    rows, settled = [], []
    totals = {'seats': 0, 'safe': 0, 'open': 0}
    for district in districts:
        row = next((r for r in district['rows'] if r['name'] == name), None)
        if row is None:
            continue
        for key in totals:
            totals[key] += row[key]
        if row['open']:
            rows.append({
                'district': {key: district[key] for key in ('id', 'name', 'slug', 'seats')},
                'seats': row['seats'],
                'tokens': row['tokens'],
                'safe': row['safe'],
                'open': row['open'],
            })
        elif row['safe']:
            settled.append({'name': district['name'], 'slug': district['slug'], 'safe': row['safe']})
    rows.sort(key=lambda row: _coin_toss_gap(row['tokens']))
    settled.sort(key=lambda item: -item['safe'])
    return {'name': name, **totals, 'rows': rows, 'settled': settled}


def closest_seats(districts: list, limit: int = CLOSEST_SHOWN) -> list:
    """
    The open tokens nearest to a coin toss.

    Parameters
    ----------
    districts : list
        Views from ``district_views``.
    limit : int, default CLOSEST_SHOWN
        Maximum number of seats returned.

    Returns
    -------
    list
        Dicts with ``district``, ``slug``, ``party``, ``k``, ``share``, ``label`` and ``central``, nearest to 50% first.
    """
    seats = [
        {'district': district['name'], 'slug': district['slug'], 'party': row['name'], 'k': token['k'],
         'share': token['share'], 'label': token['label'], 'central': token['central']}
        for district in districts for row in district['rows'] for token in row['tokens'] if not token['safe']
    ]
    return sorted(seats, key=lambda seat: abs(seat['share'] - 0.5))[:limit]
