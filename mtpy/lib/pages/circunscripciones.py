"""
The seats that dance (``/{scope}/circunscripciones``): from the ``seat-odds`` part to the tokens of every district and
the views built on them (per district, per party, the closest seats).

A seat is *safe* when the party reaches it in at least ``SAFE_SHARE`` of the simulations and *open* when it does so in
at least ``OPEN_FLOOR`` but fewer than ``SAFE_SHARE``; below ``OPEN_FLOOR`` it is neither drawn nor counted. All
functions above ``seat_odds_context`` are pure: they take the data of the part and return plain dicts and lists; the
context adds on top of them what the page needs (links, colours, the texts of each row).
"""
import re
from typing import Optional
from urllib.parse import urlencode

from ...core.api import HttpError
from ..webapi import site
from .common import OTHERS_COLOR, bundle_part, common_context, subpage_event_bar
from .escanos import _sort_name
from .slugs import slugify

SAFE_SHARE = 0.99
OPEN_FLOOR = 0.01
CLOSEST_SHOWN = 40
CARD_ANCHOR = '#ficha'   # the card of a district or a party
VIEWS_ANCHOR = '#vistas'  # the tabs of the views


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


def threshold_label(share: float) -> str:
    """
    A threshold as the texts of the page write it.

    Parameters
    ----------
    share : float
        Share of the simulations (``SAFE_SHARE``, ``OPEN_FLOOR``).

    Returns
    -------
    str
        The whole percentage with the ``%`` attached (``'99%'``, ``'1%'``).
    """
    return '{}%'.format(int(share * 100 + 0.5))


def tokens_note(row: dict) -> str:
    """
    Summary of the tokens of a row, beside them: how many safe and how many open.

    Parameters
    ----------
    row : dict
        A row of ``district_views`` (``safe``, ``open``).

    Returns
    -------
    str
        ``'2 seguros · 2 en el aire'``, ``'1 seguro'``, ``'1 en el aire'``; ``'—'`` without tokens.
    """
    parts = []
    if row['safe']:
        parts.append('1 seguro' if row['safe'] == 1 else '{} seguros'.format(row['safe']))
    if row['open']:
        parts.append('{} en el aire'.format(row['open']))
    return ' · '.join(parts) or '—'


def tokens_reading(name: str, row: dict) -> str:
    """
    The tokens of a row read aloud: the label of the image that draws them.

    Parameters
    ----------
    name : str
        What the row is (the party in the view of a district).
    row : dict
        A row of ``district_views`` (``safe``, ``tokens``).

    Returns
    -------
    str
        ``'PP: 2 seguros; 3.º, 80%; 4.º, 7%'``; without tokens, that none reaches ``OPEN_FLOOR``.
    """
    parts = []
    if row['safe']:
        parts.append('1 seguro' if row['safe'] == 1 else '{} seguros'.format(row['safe']))
    parts += ['{}.º, {}'.format(token['k'], token['label']) for token in row['tokens'] if not token['safe']]
    if not parts:
        return '{}: ningún escaño llega al {}'.format(name, threshold_label(OPEN_FLOOR))
    return '{}: {}'.format(name, '; '.join(parts))


def tint(color: str, alpha: float) -> str:
    """
    A colour with an opacity, for the fill of an open token.

    Parameters
    ----------
    color : str
        Hex colour ``#rgb``, ``#rrggbb`` or ``#rrggbbaa`` (its alpha is ignored); anything else takes
        ``OTHERS_COLOR``.
    alpha : float
        Opacity, 0 to 1.

    Returns
    -------
    str
        ``'rgba(r, g, b, alpha)'``.
    """
    if not isinstance(color, str) or not re.fullmatch(r'#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})', color):
        color = OTHERS_COLOR
    digits = color[1:]
    if len(digits) == 3:
        digits = ''.join(char * 2 for char in digits)
    r, g, b = (int(digits[i:i + 2], 16) for i in (0, 2, 4))
    return 'rgba({}, {}, {}, {})'.format(r, g, b, round(alpha, 3))


def drawn_row(row: dict, name: str, color: str) -> dict:
    """
    A row of tokens ready to draw: its colour, its texts and the fill of each open token.

    Parameters
    ----------
    row : dict
        A row of ``district_views``.
    name : str
        What the row is, for ``tokens_reading``.
    color : str
        Colour of the party.

    Returns
    -------
    dict
        A copy of ``row`` with ``color``, ``reading`` (``tokens_reading``), ``note`` (``tokens_note``) and its
        ``tokens`` with ``fill``: ``tint`` of the colour with the ``alpha`` of the token, ``None`` when safe.
    """
    tokens = [dict(token, fill=None if token['safe'] else tint(color, token['alpha'])) for token in row['tokens']]
    return dict(row, tokens=tokens, color=color, reading=tokens_reading(name, row), note=tokens_note(row))


def district_href(scope: str, slug: str, query: str = '') -> str:
    """
    Link to the card of a district on the page.

    Parameters
    ----------
    scope : str
        Scope code.
    slug : str
        Slug of the district (``district_views``).
    query : str, optional
        ``event_query`` of the page that links it, so that the link keeps ``mode`` and ``run``.

    Returns
    -------
    str
        ``'/{scope}/circunscripciones/{slug}{query}#ficha'``.
    """
    return '/{}/circunscripciones/{}{}{}'.format(scope, slug, query, CARD_ANCHOR)


def _with_param(query: str, key: str, value: str) -> str:
    """
    A query string with one more parameter.

    Parameters
    ----------
    query : str
        ``'?...'`` or ``''``.
    key, value : str
        The parameter.

    Returns
    -------
    str
        ``query`` followed by ``key=value``, joined with ``&`` (or opened with ``?`` when ``query`` is empty).
    """
    return query + ('&' if query else '?') + urlencode([(key, value)])


def party_sentence(party: dict) -> str:
    """
    Plain-language summary of a party across the districts: its safe seats and where it can still win more.

    Parameters
    ----------
    party : dict
        View from ``party_view`` (``safe``, ``open``, ``rows``).

    Returns
    -------
    str
        The sentence, in Spanish; ``'Primero, las más dudosas.'`` closes it only when there is more than one district
        to order.
    """
    safe, left, count = party['safe'], party['open'], len(party['rows'])
    own = '1 escaño seguro' if safe == 1 else '{} escaños seguros'.format(safe)
    if not left:
        return 'Tiene {} y ninguno en el aire.'.format(own)
    more = 'otro' if left == 1 else 'otros {}'.format(left)
    where = '1 circunscripción' if count == 1 else '{} circunscripciones'.format(count)
    sentence = 'Tiene {} y aspira a {} en {}.'.format(own, more, where)
    return sentence + (' Primero, las más dudosas.' if count > 1 else '')


def closest_sentence(shown: int, contenders: int) -> str:
    """
    Plain-language introduction of the list of the closest seats.

    Parameters
    ----------
    shown : int
        Seats in the list.
    contenders : int
        Open tokens of the whole scope.

    Returns
    -------
    str
        The sentence, in Spanish.
    """
    if not shown:
        return 'Ningún escaño está en el aire.'
    if shown == 1:
        return 'El único escaño que está en el aire. La raya marca el 50%.'
    return 'Los {} escaños más cercanos al cara o cruz, de los {} que están en el aire. La raya marca el 50%.'.format(
        shown, contenders)


def _party_names(data: dict, views: list) -> list:
    """
    The parties of the part with some token, in the order of the part.

    Parameters
    ----------
    data : dict
        Data of the ``seat-odds`` part.
    views : list
        Views from ``district_views``.

    Returns
    -------
    list
        Party names.
    """
    drawn = {row['name'] for view in views for row in view['rows'] if row['tokens']}
    return [name for name in data.get('parties') or [] if name in drawn]


def _district_card(chosen: dict, parties: dict) -> dict:
    """
    The part of the context specific to the view of a district.

    Parameters
    ----------
    chosen : dict
        The district (``district_views`` with ``href``).
    parties : dict
        Catalogue of the scope (name to ``color``).

    Returns
    -------
    dict
        ``district`` (its rows drawn with the colour of each party), ``sentence``, ``page_title`` and ``image_slug``.
    """
    rows = [drawn_row(row, row['name'], parties[row['name']]['color']) for row in chosen['rows']]
    return {
        'district': dict(chosen, rows=rows),
        'sentence': district_sentence(chosen),
        'page_title': 'Escaños que bailan en {}'.format(chosen['name']),
        'image_slug': chosen['slug'],
    }


def _party_card(views: list, name: str, parties: dict, scope: str, query: str) -> dict:
    """
    The part of the context specific to the view of a party.

    Parameters
    ----------
    views : list
        Views from ``district_views``.
    name : str
        Party name.
    parties : dict
        Catalogue of the scope (name to ``color`` and ``fullname``).
    scope, query : str
        Scope code and ``event_query`` of the page, for the link of each district.

    Returns
    -------
    dict
        ``party`` (``party_view`` with ``fullname``, ``color`` and its rows drawn, each with the ``href`` of its
        district), ``sentence``, ``page_title`` and ``image_slug``.
    """
    view = party_view(views, name)
    color = parties[name]['color']
    rows = [dict(drawn_row(row, row['district']['name'], color),
                 href=district_href(scope, row['district']['slug'], query)) for row in view['rows']]
    return {
        'party': dict(view, fullname=parties[name]['fullname'], color=color, rows=rows),
        'sentence': party_sentence(view),
        'page_title': 'Escaños que bailan: {}'.format(name),
        'image_slug': slugify(name),
    }


def _closest_card(views: list, parties: dict, scope: str, query: str) -> dict:
    """
    The part of the context specific to the list of the closest seats.

    Parameters
    ----------
    views : list
        Views from ``district_views``.
    parties : dict
        Catalogue of the scope (name to ``color``).
    scope, query : str
        Scope code and ``event_query`` of the page, for the link of each district.

    Returns
    -------
    dict
        ``closest`` (``closest_seats`` with ``color`` and ``href``), ``sentence`` and ``page_title``.
    """
    closest = [dict(seat, color=parties[seat['party']]['color'], href=district_href(scope, seat['slug'], query))
               for seat in closest_seats(views)]
    return {
        'closest': closest,
        'sentence': closest_sentence(len(closest), scope_totals(views)['contenders']),
        'page_title': 'Los escaños más reñidos',
    }


def published_seat_odds(scope: str, run: str, mode: str) -> Optional[dict]:
    """
    The ``seat-odds`` part of a run when the page «Escaños que bailan» exists for it.

    Parameters
    ----------
    scope : str
        Scope code.
    run : str
        Run id.
    mode : str
        Simulation mode.

    Returns
    -------
    dict or None
        The data of the part; ``None`` when the run has no such part or it has fewer than two districts.
    """
    data = bundle_part(site().run_file, scope, run, 'seat-odds', mode)

    if data is None or len(data.get('regions') or []) < 2:
        return None

    return data


def seat_odds_links(scope: str, run: str, mode: str, query: str) -> Optional[dict]:
    """
    Links from the event page to the page «Escaños que bailan» and to the card of each district.

    Parameters
    ----------
    scope : str
        Scope code.
    run : str
        Run id the event page shows.
    mode : str
        Simulation mode.
    query : str
        ``event_query`` of the event page, so that the links keep ``mode`` and ``run``.

    Returns
    -------
    dict or None
        ``{'page': '/{scope}/circunscripciones{query}', 'districts': {region_id: href}}`` with the href of
        ``district_href``; ``None`` when the page does not exist for the run (``published_seat_odds``).
    """
    data = published_seat_odds(scope, run, mode)

    if data is None:
        return None

    return {
        'page': '/{}/circunscripciones{}'.format(scope, query),
        'districts': {view['id']: district_href(scope, view['slug'], query) for view in district_views(data)},
    }


def seat_odds_context(state: dict, district: Optional[str] = None, party: Optional[str] = None,
                      view: Optional[str] = None) -> dict:
    """
    Template context of the page «Escaños que bailan» (``/{scope}/circunscripciones[/{district}]``).

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``), the scope taken from the route.
    district : str, optional
        Slug of the district in the URL; it wins over ``party`` and ``view``.
    party : str, optional
        Slug (``slugify``) of the acronym of a party with some token: the view of that party.
    view : str, optional
        ``'renidos'``: the list of the closest seats. ``party`` wins over it. Without any of the three, the view of
        the first of ``district_order``.

    Returns
    -------
    dict
        ``common_context`` of an event page, its ``event_bar['sections']`` linking the sections of the event page
        (``'/{scope}{query}#{id}'``, without ``territorio`` when the run has no districts), plus ``view``
        (``'district'``, ``'party'`` or ``'closest'``), ``totals`` (``scope_totals``), ``kpis`` (rows of ``m.kpis``), ``districts``
        (``district_order``, each with ``href``, ``district_href`` keeping the query, and ``rank_name``, its
        position in alphabetical order), ``sentence`` and the data of the view: ``district`` (the chosen one, its rows
        ``drawn_row`` with the colour of each party), or ``parties`` (the buttons: ``name``, ``fullname``, ``color``,
        ``href``, ``on``) and ``party`` (``party_view`` with ``fullname``, ``color`` and the rows ``drawn_row`` with
        the ``href`` of their district), or ``closest`` (``closest_seats`` with ``color`` and ``href``); ``tabs`` (``key``, ``label``, ``href`` keeping the query and
        ``on`` of the three views), ``n_sim``, ``thresholds`` (``safe`` and ``floor``, ``threshold_label`` of
        ``SAFE_SHARE`` and ``OPEN_FLOOR``, for the legend), ``page_title``, ``image_file`` (of the card of a district
        or a party) and ``image_sign``.

    Raises
    ------
    HttpError
        404 ``seat odds not published`` when the run shown has no ``seat-odds`` part in the mode or it has fewer
        than two districts; 404 ``region not found`` for an unknown ``district``; 404 ``party not found`` when ``party`` is not
        a party of the part with some token; 400 ``invalid view`` for a ``view`` other than ``renidos``; those of ``common_context``.
    """
    context = common_context(state, None, event=True)
    full, query, parties = context['state'], context['query'], context['catalog']
    scope, mode, run = full['scope'], full['mode'], full['run']
    data = published_seat_odds(scope, run, mode)

    if data is None:
        raise HttpError(404, 'seat odds not published')

    views = district_order(district_views(data))
    by_name = sorted(views, key=lambda view: _sort_name(view['name']))
    rank = {view['id']: position for position, view in enumerate(by_name, start=1)}
    districts = [dict(view, href=district_href(scope, view['slug'], query), rank_name=rank[view['id']])
                 for view in views]

    names = _party_names(data, views)
    by_slug = {slugify(name): name for name in names}

    if district is not None:
        chosen = next((item for item in districts if item['slug'] == district), None)

        if chosen is None:
            raise HttpError(404, 'region not found')

        shown, card = 'district', _district_card(chosen, parties)
    elif party is not None:
        if party not in by_slug:
            raise HttpError(404, 'party not found')

        shown, card = 'party', _party_card(views, by_slug[party], parties, scope, query)
    elif view is not None:
        if view != 'renidos':
            raise HttpError(400, 'invalid view')

        shown, card = 'closest', _closest_card(views, parties, scope, query)
    else:
        shown, card = 'district', _district_card(districts[0], parties)

    totals = scope_totals(views)
    # the index links the sections of the event page, keeping the mode and the pinned run
    context['event_bar'] = subpage_event_bar(context)
    base = '/{}/circunscripciones'.format(scope)
    # the view of a party opens on the first party of the part with some token; the tabs and the party buttons land
    # on the tabs (`#vistas`): on a phone the card of the view is far below the top of the page
    tabs = [
        {'key': 'district', 'label': 'Por circunscripción', 'href': base + query + VIEWS_ANCHOR},
        {'key': 'party', 'label': 'Por partido',
         'href': base + (_with_param(query, 'partido', slugify(names[0])) if names else query) + VIEWS_ANCHOR},
        {'key': 'closest', 'label': 'Los más reñidos',
         'href': base + _with_param(query, 'vista', 'renidos') + VIEWS_ANCHOR},
    ]

    context.update({
        'view': shown,
        'totals': totals,
        'kpis': [
            {'label': 'Escaños seguros', 'value': totals['safe'], 'unit': 'de {}'.format(totals['seats']),
             'color': None},
            {'label': 'Escaños en el aire', 'value': totals['open'], 'unit': '', 'color': None},
            {'label': 'Aspirantes', 'value': totals['contenders'], 'unit': 'a esos {}'.format(totals['open']),
             'color': None},
            {'label': 'Circunscripciones con algo en juego', 'value': totals['live'],
             'unit': 'de {}'.format(totals['districts']), 'color': None},
        ],
        'districts': districts,
        'parties': [{'name': name, 'fullname': parties[name]['fullname'], 'color': parties[name]['color'],
                     'href': base + _with_param(query, 'partido', slugify(name)) + VIEWS_ANCHOR,
                     'on': shown == 'party' and card['party']['name'] == name} for name in names],
        'tabs': [dict(tab, on=tab['key'] == shown) for tab in tabs],
        'n_sim': data['n_sim'],
        'thresholds': {'safe': threshold_label(SAFE_SHARE), 'floor': threshold_label(OPEN_FLOOR)},
        'image_sign': '{} · {}'.format(context['site_host'], context['author_handle']),
    })
    image_slug = card.pop('image_slug', None)
    context.update(card)

    if image_slug:
        context['image_file'] = 'demoscopio-escanos-{}-{}.png'.format(scope, image_slug)

    return context
