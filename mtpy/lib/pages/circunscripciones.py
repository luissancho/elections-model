"""
The seats that dance (``/{scope}/circunscripciones``): from the ``seat-odds`` part to the seat options of every district
and the views built on them (per district, per party, the closest seats).

A *seat option* is the k-th seat of a party in a district, reached by a simulation when the party wins ``k`` seats or
more there. Its share of the simulations sets its state: *safe* from ``SAFE_SHARE``, *likely* from ``LIKELY_SHARE``
and *in play* from ``PLAY_SHARE``; below ``PLAY_SHARE`` it is neither drawn nor counted. The shares are compared
unrounded. All functions above ``seat_odds_context`` are pure: they take the data of the part and return plain dicts
and lists; the context adds on top of them what the page needs (links, colours, the texts of each row).
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
LIKELY_SHARE = 0.75
PLAY_SHARE = 0.25
STATES = ('safe', 'likely', 'play')  # the states of a seat option, from the surest
# the word of each state, singular and plural: the texts of the module and the chips of the page (`state_words`)
STATE_WORDS = {'safe': ('seguro', 'seguros'), 'likely': ('probable', 'probables'), 'play': ('en juego', 'en juego')}
CLOSEST_SHOWN = 40
CARD_ANCHOR = '#ficha'   # the card of a district or a party
VIEWS_ANCHOR = '#vistas'  # the tabs of the views


def seat_state(share: float) -> Optional[str]:
    """
    State of a seat option from its share of the simulations, compared unrounded.

    Parameters
    ----------
    share : float
        Share of the simulations in which the party reaches the seat (0 to 1).

    Returns
    -------
    str or None
        ``'safe'`` from ``SAFE_SHARE``, ``'likely'`` from ``LIKELY_SHARE``, ``'play'`` from ``PLAY_SHARE``; ``None``
        below it (not an option).
    """
    if share >= SAFE_SHARE:
        return 'safe'
    if share >= LIKELY_SHARE:
        return 'likely'
    if share >= PLAY_SHARE:
        return 'play'
    return None


def token_label(share: float) -> str:
    """
    Label of a seat option that is not safe: a whole percentage, half rounded up, kept between 25% and 98%.

    Parameters
    ----------
    share : float
        Share of the simulations in which the party reaches the seat (0 to 1).

    Returns
    -------
    str
        The percentage with the ``%`` attached; never below 25%, nor 99% nor 100%.
    """
    # the epsilon rounds up the exact halves that the binary floats put a hair below (0.145 * 100 is 14.4999...)
    return '{}%'.format(min(98, max(25, int(share * 100 + 0.5 + 1e-9))))


def token_alpha(state: str, share: float) -> Optional[float]:
    """
    Opacity of the fill of a seat option: the share moves it inside the band of its state.

    Parameters
    ----------
    state : str
        ``seat_state`` of the option.
    share : float
        Share of the simulations in which the party reaches the seat (0 to 1).

    Returns
    -------
    float or None
        ``None`` when safe (solid); 0.42 to 0.68 when likely, 0.06 to 0.26 when in play; rounded to 3 decimals.
    """
    if state == 'safe':
        return None
    if state == 'likely':
        return round(0.42 + 0.26 * (share - LIKELY_SHARE) / (SAFE_SHARE - LIKELY_SHARE), 3)
    return round(0.06 + 0.20 * (share - PLAY_SHARE) / (LIKELY_SHARE - PLAY_SHARE), 3)


def seat_tokens(name: str, reach: list, n_sim: int) -> list:
    """
    Seat options of a party in a district: one per seat it reaches in at least ``PLAY_SHARE`` of the simulations.

    Parameters
    ----------
    name : str
        Party name, used in the title of each option.
    reach : list
        ``reach[k - 1]`` is the number of simulations in which the party wins at least ``k`` seats.
    n_sim : int
        Number of simulations.

    Returns
    -------
    list
        One dict per option, in seat order: ``k``, ``share``, ``state`` (``seat_state``), ``label`` and ``alpha``
        (``None`` when safe) and ``title``.
    """
    tokens = []
    for k, count in enumerate(reach, start=1):
        share = count / n_sim
        state = seat_state(share)
        if state is None:
            continue
        safe = state == 'safe'
        label = None if safe else token_label(share)
        what = 'seguro' if safe else 'lo consigue en el {} de las simulaciones'.format(label)
        tokens.append({
            'k': k,
            'share': share,
            'state': state,
            'label': label,
            'alpha': token_alpha(state, share),
            'title': '{}.º de {}: {}'.format(k, name, what),
        })
    return tokens


def _count_states(tokens: list) -> dict:
    """
    How many seat options of each state.

    Parameters
    ----------
    tokens : list
        Options from ``seat_tokens``.

    Returns
    -------
    dict
        One count per state of ``STATES``.
    """
    return {state: sum(token['state'] == state for token in tokens) for state in STATES}


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
        Dicts with ``id``, ``name``, ``slug``, ``seats``, the options of each state (``safe``, ``likely``, ``play``)
        and ``rows``: one per party with some option, with ``name``, ``tokens``, its options of each state and
        ``expected`` (the sum of the shares of its options), by ``expected`` descending (on a tie, in the order of
        the part).
    """
    n_sim = data['n_sim']
    views = []
    for region in data['regions']:
        rows = []
        for row in data['rows']:
            if row['region_id'] != region['id']:
                continue
            tokens = seat_tokens(row['name'], row['reach'], n_sim)
            if not tokens:
                continue
            rows.append({
                'name': row['name'],
                'tokens': tokens,
                **_count_states(tokens),
                'expected': sum(token['share'] for token in tokens),
            })
        rows.sort(key=lambda row: -row['expected'])
        views.append({
            'id': region['id'],
            'name': region['name'],
            'slug': slugify(region['name']),
            'seats': region['seats'],
            **{state: sum(row[state] for row in rows) for state in STATES},
            'rows': rows,
        })
    return views


def default_district(districts: list) -> dict:
    """
    The district the page opens on when the address names none.

    Parameters
    ----------
    districts : list
        Views from ``district_views`` (``name``, ``seats``, ``play``).

    Returns
    -------
    dict
        The one with most options in play; on a tie, the one with more seats, then the first by name with accents
        folded.
    """
    return min(districts, key=lambda d: (-d['play'], -d['seats'], _sort_name(d['name'])))


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
        ``seats``, the options of each state (``safe``, ``likely``, ``play``) and ``options``, their sum.
    """
    totals = {key: sum(d[key] for d in districts) for key in ('seats',) + STATES}
    totals['options'] = sum(totals[state] for state in STATES)
    return totals


def options_sentence(options: int, seats: int) -> str:
    """
    The foot of a chart when the seat options outnumber the seats.

    Parameters
    ----------
    options : int
        Seat options drawn (``scope_totals``).
    seats : int
        Seats of the scope.

    Returns
    -------
    str
        The sentence, in Spanish, when ``options > seats``; ``''`` otherwise.
    """
    if options <= seats:
        return ''
    return 'Hay {} opciones para {} escaños: a un mismo escaño pueden optar varios partidos.'.format(options, seats)


def _state_word(state: str, count: int) -> str:
    """
    The word of a state, agreeing with a count.

    Parameters
    ----------
    state : str
        One of ``STATES``.
    count : int
        How many options.

    Returns
    -------
    str
        ``'seguro'``, ``'probables'``, ``'en juego'``...
    """
    return STATE_WORDS[state][0 if count == 1 else 1]


def state_text(state: str, count: int) -> str:
    """
    A count of options of a state, in words.

    Parameters
    ----------
    state : str
        One of ``STATES``.
    count : int
        How many options.

    Returns
    -------
    str
        ``'1 seguro'``, ``'5 seguros'``, ``'3 probables'``, ``'2 en juego'``.
    """
    return '{} {}'.format(count, _state_word(state, count))


def _join_names(names: list) -> str:
    """
    Spanish enumeration of some names.

    Parameters
    ----------
    names : list of str
        Names, in the order to say them.

    Returns
    -------
    str
        ``'A'``, ``'A y B'``, ``'A, B y C'``; «e» instead of «y» before a last name that starts with the sound /i/
        (``'PP e IU'``).
    """
    if len(names) < 2:
        return ''.join(names)
    last = names[-1].lower()
    link = 'e' if last.startswith(('i', 'hi')) else 'y'
    return '{} {} {}'.format(', '.join(names[:-1]), link, names[-1])


def district_sentence(district: dict) -> str:
    """
    Plain-language summary of a district: how many seats are safe, for whom, and which parties dispute the others
    (those with some likely option or some option in play).

    Parameters
    ----------
    district : dict
        View from ``district_views`` (``seats``, ``safe`` and ``rows`` with ``name``, ``safe``, ``likely``, ``play``).

    Returns
    -------
    str
        The sentence, in Spanish. The other ``max(seats - safe, 0)`` seats «se los disputan» the parties named; with a
        single party they «sólo están al alcance de» it, and without any the clause is dropped.
    """
    seats, safe = district['seats'], district['safe']
    left = max(seats - safe, 0)
    listing = ', '.join('{} {}'.format(row['name'], row['safe']) for row in district['rows'] if row['safe'])
    if left == 0:
        if seats == 1:
            return 'El escaño es seguro ({}).'.format(listing)
        return 'Los {} escaños son seguros ({}).'.format(seats, listing)
    rivals = [row['name'] for row in district['rows'] if row['likely'] or row['play']]
    single = left == 1
    if not rivals:
        dispute = ''
    elif len(rivals) == 1:
        dispute = 'sólo {} al alcance de {}'.format('está' if single else 'están', rivals[0])
    else:
        dispute = 'se {} disputan {}'.format('lo' if single else 'los', _join_names(rivals))
    if safe == 0:
        head = 'El escaño no es seguro' if seats == 1 else 'Ninguno de los {} escaños es seguro'.format(seats)
        return '{}: {}.'.format(head, dispute) if dispute else head + '.'
    how_many = '1 es seguro' if safe == 1 else '{} son seguros'.format(safe)
    sentence = 'De los {} escaños, {} ({}).'.format(seats, how_many, listing)
    if not dispute:
        return sentence
    return '{} {} {}.'.format(sentence, 'El otro' if single else 'Los otros {}'.format(left), dispute)


def _coin_toss_gap(tokens: list) -> float:
    """
    Distance to 50% of the option not safe nearest to it.

    Parameters
    ----------
    tokens : list
        Options of a party in a district, with at least one not safe.

    Returns
    -------
    float
        The smallest ``abs(share - 0.5)`` among the options that are not safe.
    """
    return min(abs(token['share'] - 0.5) for token in tokens if token['state'] != 'safe')


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
        ``name``, its options of each state (``safe``, ``likely``, ``play``), ``rows`` (districts where it has some
        option not safe, with ``district``, ``tokens`` and its options of each state, the one whose option not safe
        is nearest to a coin toss first) and ``settled`` (``name``, ``slug``, ``safe`` of the districts where all its
        options are safe, most seats first).
    """
    rows, settled = [], []
    totals = dict.fromkeys(STATES, 0)
    for district in districts:
        row = next((r for r in district['rows'] if r['name'] == name), None)
        if row is None:
            continue
        for state in STATES:
            totals[state] += row[state]
        if row['likely'] or row['play']:
            rows.append({
                'district': {key: district[key] for key in ('id', 'name', 'slug', 'seats')},
                'tokens': row['tokens'],
                **{state: row[state] for state in STATES},
            })
        elif row['safe']:
            settled.append({'name': district['name'], 'slug': district['slug'], 'safe': row['safe']})
    rows.sort(key=lambda row: _coin_toss_gap(row['tokens']))
    settled.sort(key=lambda item: -item['safe'])
    return {'name': name, **totals, 'rows': rows, 'settled': settled}


def closest_seats(districts: list, limit: int = CLOSEST_SHOWN) -> list:
    """
    The seat options not safe nearest to a coin toss.

    Parameters
    ----------
    districts : list
        Views from ``district_views``.
    limit : int, default CLOSEST_SHOWN
        Maximum number of options returned.

    Returns
    -------
    list
        Dicts with ``district``, ``slug``, ``party``, ``k``, ``share``, ``label`` and ``state``, nearest to 50% first.
    """
    seats = [
        {'district': district['name'], 'slug': district['slug'], 'party': row['name'], 'k': token['k'],
         'share': token['share'], 'label': token['label'], 'state': token['state']}
        for district in districts for row in district['rows'] for token in row['tokens'] if token['state'] != 'safe'
    ]
    return sorted(seats, key=lambda seat: abs(seat['share'] - 0.5))[:limit]


def threshold_label(share: float) -> str:
    """
    A threshold as the texts of the page write it.

    Parameters
    ----------
    share : float
        Share of the simulations (``SAFE_SHARE``, ``LIKELY_SHARE``, ``PLAY_SHARE``).

    Returns
    -------
    str
        The whole percentage with the ``%`` attached (``'99%'``, ``'25%'``).
    """
    return '{}%'.format(int(share * 100 + 0.5))


def tokens_note(row: dict) -> str:
    """
    Summary of the options of a row, beside them: how many of each state.

    Parameters
    ----------
    row : dict
        A row of ``district_views`` or of ``party_view`` (``safe``, ``likely``, ``play``).

    Returns
    -------
    str
        ``'2 seguros · 1 probable'``, ``'1 en juego'``: only the states with some option; ``'—'`` without any.
    """
    return ' · '.join(state_text(state, row[state]) for state in STATES if row[state]) or '—'


def tokens_reading(name: str, row: dict) -> str:
    """
    The options of a row read aloud: the label of the image that draws them.

    Parameters
    ----------
    name : str
        What the row is (the party in the view of a district).
    row : dict
        A row of ``district_views`` (``safe``, ``tokens``).

    Returns
    -------
    str
        ``'PP: 2 seguros; 3.º, 80%; 4.º, 30%'``: the safe ones counted, the others one by one with their label.
    """
    parts = [state_text('safe', row['safe'])] if row['safe'] else []
    parts += ['{}.º, {}'.format(token['k'], token['label']) for token in row['tokens'] if token['state'] != 'safe']
    return '{}: {}'.format(name, '; '.join(parts))


def tint(color: str, alpha: float) -> str:
    """
    A colour with an opacity, for the fill of a seat option that is not safe.

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


def drawn_token(token: dict, color: str) -> dict:
    """
    A seat option ready to draw as a circle: its colour and the fill of its state.

    Parameters
    ----------
    token : dict
        An option from ``seat_tokens``.
    color : str
        Colour of the party.

    Returns
    -------
    dict
        A copy of ``token`` with ``color`` and ``fill``: ``tint`` of the colour with the ``alpha`` of the option,
        ``None`` when safe (the circle is solid).
    """
    return dict(token, color=color, fill=None if token['alpha'] is None else tint(color, token['alpha']))


def drawn_row(row: dict, name: str, color: str) -> dict:
    """
    A row of seat options ready to draw: its colour, its texts and each option drawn (``drawn_token``).

    Parameters
    ----------
    row : dict
        A row of ``district_views`` or of ``party_view``.
    name : str
        What the row is, for ``tokens_reading``.
    color : str
        Colour of the party.

    Returns
    -------
    dict
        A copy of ``row`` with ``color``, ``reading`` (``tokens_reading``), ``note`` (``tokens_note``) and its
        ``tokens`` through ``drawn_token``.
    """
    tokens = [drawn_token(token, color) for token in row['tokens']]
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
    Plain-language summary of a party across the districts: its seat options of each state.

    Parameters
    ----------
    party : dict
        View from ``party_view`` (``safe``, ``likely``, ``play``).

    Returns
    -------
    str
        The sentence, in Spanish, naming only the states with some option, with «escaño(s)» after the first figure:
        ``'Tiene 66 escaños seguros, 30 probables y 25 en juego.'``, ``'Tiene 1 escaño en juego.'``.
    """
    parts = []
    for state in STATES:
        count = party[state]
        if not count:
            continue
        noun = '' if parts else (' escaño' if count == 1 else ' escaños')
        parts.append('{}{} {}'.format(count, noun, _state_word(state, count)))
    return 'Tiene {}.'.format(_join_names(parts)) if parts else ''


def closest_sentence(shown: int) -> str:
    """
    Plain-language introduction of the list of the closest seats.

    Parameters
    ----------
    shown : int
        Seat options in the list.

    Returns
    -------
    str
        The sentence, in Spanish.
    """
    if not shown:
        return 'No hay ningún escaño probable ni en juego.'
    if shown == 1:
        return 'El único escaño que no es seguro. La raya marca el 50%.'
    return 'Los {} escaños más cercanos al cara o cruz. La raya marca el 50%.'.format(shown)


def _party_names(data: dict, views: list) -> list:
    """
    The parties of the part with some seat option, in the order of the part.

    Parameters
    ----------
    data : dict
        Data of the ``seat-odds`` part.
    views : list
        Views from ``district_views`` (their rows are the parties with some option).

    Returns
    -------
    list
        Party names.
    """
    drawn = {row['name'] for view in views for row in view['rows']}
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
        'sentence': closest_sentence(len(closest)),
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
        The data of the part; ``None`` when the run has no such part, it has no simulations (``n_sim`` missing or
        0) or it has fewer than two districts.
    """
    reader = site()
    # every run published before the part lacks it, and the LRU of the run files keeps only the files that exist:
    # its presence is remembered under the TTL of the pointers, so the event page does not read the bundle again on
    # every render; when present, the bytes come from the LRU
    present = reader.cached(('seat-odds', scope, run, mode),
                            lambda: bundle_part(reader.run_file, scope, run, 'seat-odds', mode) is not None)

    if not present:
        return None

    data = bundle_part(reader.run_file, scope, run, 'seat-odds', mode)

    if data is None or not data.get('n_sim') or len(data.get('regions') or []) < 2:
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
        ``district_href``, built from the ``regions`` of the part alone (no token is computed); ``None`` when the
        page does not exist for the run (``published_seat_odds``).
    """
    data = published_seat_odds(scope, run, mode)

    if data is None:
        return None

    return {
        'page': '/{}/circunscripciones{}'.format(scope, query),
        # the keys are the ids of the regions, ints; in the JSON embedded in the event page (`odds_hrefs`) they become
        # strings
        'districts': {region['id']: district_href(scope, slugify(region['name']), query) for region in data['regions']},
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
        Slug (``slugify``) of the acronym of a party with some seat option: the view of that party.
    view : str, optional
        ``'renidos'``: the list of the closest seats. ``party`` wins over it. Without any of the three, the view of
        ``default_district``, under the title of the page itself (``landing``).

    Returns
    -------
    dict
        ``common_context`` of an event page, its ``event_bar['sections']`` linking the sections of the event page
        (``'/{scope}{query}#{id}'``, without ``territorio`` when the run has no districts), plus ``view``
        (``'district'``, ``'party'`` or ``'closest'``), ``landing`` (none of ``district``, ``party`` and ``view``
        given), ``totals`` (``scope_totals``), ``districts`` (``district_views`` sorted by name with accents folded,
        each with ``href``, ``district_href`` keeping the query), ``sentence`` and the data of the view: ``district``
        (the chosen one, its rows ``drawn_row`` with the colour of each party), or ``parties`` (the buttons: ``name``,
        ``fullname``, ``color``, ``href`` keeping the query and landing on the tabs, ``on``) and ``party``
        (``party_view`` with ``fullname``, ``color`` and the rows ``drawn_row`` with the ``href`` of their district),
        or ``closest`` (``closest_seats`` with ``color`` and ``href``); ``tabs`` (``key``, ``label``, ``href`` keeping
        the query and landing on the tabs, and ``on`` of the three views), ``n_sim``, ``thresholds`` (``safe``,
        ``likely`` and ``play``, ``threshold_label`` of ``SAFE_SHARE``, ``LIKELY_SHARE`` and ``PLAY_SHARE``, for the
        legend and the «Cómo leerlo»), ``state_words`` (``STATE_WORDS``, for the chips), ``page_title``
        (``'Escaños que bailan'`` on the landing), ``image_file`` (of the card of a district or a party) and
        ``image_sign``.

    Raises
    ------
    HttpError
        404 ``seat odds not published`` when the run shown has no ``seat-odds`` part in the mode (or one without
        simulations, or with fewer than two districts: ``published_seat_odds``); 404 ``region not found`` for an
        unknown ``district``; 404 ``party not found`` when ``party`` is not a party of the part with some option; 400
        ``invalid view`` for a ``view`` other than ``renidos``; those of ``common_context``.
    """
    context = common_context(state, None, event=True)
    full, query, parties = context['state'], context['query'], context['catalog']
    scope, mode, run = full['scope'], full['mode'], full['run']
    data = published_seat_odds(scope, run, mode)

    if data is None:
        raise HttpError(404, 'seat odds not published')

    views = district_views(data)
    districts = [dict(view, href=district_href(scope, view['slug'], query))
                 for view in sorted(views, key=lambda view: _sort_name(view['name']))]

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
        # the address of the page itself: it is shared as the page, not as its default district (which changes
        # between runs); the card and its image stay that district's
        shown, card = 'district', dict(_district_card(default_district(districts), parties),
                                       page_title='Escaños que bailan')

    # the index links the sections of the event page, keeping the mode and the pinned run
    context['event_bar'] = subpage_event_bar(context)
    base = '/{}/circunscripciones'.format(scope)
    # the view of a party opens on the first party of the part with some option; the tabs and the party buttons land
    # on the tabs (`#vistas`): on a phone the card of the view is far below the top of the page
    tabs = [
        {'key': 'district', 'label': 'Provincias', 'href': base + query + VIEWS_ANCHOR},
        {'key': 'party', 'label': 'Partidos',
         'href': base + (_with_param(query, 'partido', slugify(names[0])) if names else query) + VIEWS_ANCHOR},
        {'key': 'closest', 'label': 'Los más reñidos',
         'href': base + _with_param(query, 'vista', 'renidos') + VIEWS_ANCHOR},
    ]

    context.update({
        'view': shown,
        'landing': district is None and party is None and view is None,
        'totals': scope_totals(views),
        'districts': districts,
        'parties': [{'name': name, 'fullname': parties[name]['fullname'], 'color': parties[name]['color'],
                     'href': base + _with_param(query, 'partido', slugify(name)) + VIEWS_ANCHOR,
                     'on': shown == 'party' and card['party']['name'] == name} for name in names],
        'tabs': [dict(tab, on=tab['key'] == shown) for tab in tabs],
        'n_sim': data['n_sim'],
        'thresholds': {'safe': threshold_label(SAFE_SHARE), 'likely': threshold_label(LIKELY_SHARE),
                       'play': threshold_label(PLAY_SHARE)},
        'state_words': STATE_WORDS,
        'image_sign': '{} · {}'.format(context['site_host'], context['author_handle']),
    })
    image_slug = card.pop('image_slug', None)
    context.update(card)

    if image_slug:
        context['image_file'] = 'demoscopio-escanos-{}-{}.png'.format(scope, image_slug)

    return context
