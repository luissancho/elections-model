"""
«Escaños que bailan», the card of the event page (``/{scope}``, section ``escanos``): from the ``seat-odds`` part to
the seat options of every district, the map, the block of each district and the columns of the parties.

A *seat option* is the k-th seat of a party in a district, reached by a simulation when the party wins ``k`` seats or
more there. Its share of the simulations sets its state: *safe* from ``SAFE_SHARE``, *likely* from ``LIKELY_SHARE``
and *in play* from ``PLAY_SHARE``; below ``PLAY_SHARE`` it is neither drawn nor counted. The shares are compared
unrounded. The public functions down to ``drawn_row`` are pure: they take the data of the part and return plain dicts
and lists; ``published_seat_odds`` reads the part, and ``seat_odds_card`` builds the card on top of them (colours,
the texts of each row) and of the layout of ``seat_charts`` (the tiles of the map, the columns and their axis).
"""
import re
from typing import Optional

from ..webapi import district_positions, site
from .common import OTHERS_COLOR, Catalog, bundle_part
from .escanos import _sort_name
from .promedios import result_rows
from .seat_charts import REST_COLUMN, STATES, column_ticks, map_tiles, party_columns, state_counts
from .slugs import slugify

SAFE_SHARE = 0.99
LIKELY_SHARE = 0.75
PLAY_SHARE = 0.25
# the word of each state, singular and plural: the texts of the module and the chips of the card (`state_words`)
STATE_WORDS = {'safe': ('seguro', 'seguros'), 'likely': ('probable', 'probables'), 'play': ('en juego', 'en juego')}
# the catalogue (`data/es-districts.csv`) keeps the INE code (1 to 52) of a province as the id of its district and
# numbers every other district from 100 (`tests/test_data_files.py` asserts it)
FIRST_NON_PROVINCE_ID = 100


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
    # the epsilon rounds up the exact halves that the binary floats put a hair below (0.285 * 100 is 28.4999...)
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


def option_title(k: int, party: str, label: Optional[str], district: Optional[str] = None) -> str:
    """
    The title of a seat option: which seat, of whom and how likely.

    Parameters
    ----------
    k : int
        Seat of the party (1 for its first).
    party : str
        Party name.
    label : str or None
        ``token_label`` of the option; ``None`` when safe.
    district : str, optional
        Name of its district, for the columns of the parties (in a tile or in the block of a district it is already
        given).

    Returns
    -------
    str
        ``'3.º de PP: lo consigue en el 78% de las simulaciones'``, ``'2.º de PP: seguro'``; with ``district``,
        ``'4.º de PSOE en Alicante: lo consigue en el 51% de las simulaciones'``.
    """
    what = 'seguro' if label is None else 'lo consigue en el {} de las simulaciones'.format(label)
    where = ' en {}'.format(district) if district else ''
    return '{}.º de {}{}: {}'.format(k, party, where, what)


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
        (``None`` when safe) and ``title`` (``option_title``).
    """
    tokens = []
    for k, count in enumerate(reach, start=1):
        share = count / n_sim
        state = seat_state(share)
        if state is None:
            continue
        label = None if state == 'safe' else token_label(share)
        tokens.append({
            'k': k,
            'share': share,
            'state': state,
            'label': label,
            'alpha': token_alpha(state, share),
            'title': option_title(k, name, label),
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
                **state_counts(tokens),
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
    The district the card opens on: its tile checked and its block shown.

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


def scope_options(districts: list) -> list:
    """
    Every seat option of a scope.

    Parameters
    ----------
    districts : list
        Views from ``district_views``.

    Returns
    -------
    list
        The options of ``seat_tokens``, each with ``party`` and ``district`` (its name), in the order of the districts
        and of their rows.
    """
    return [dict(token, party=row['name'], district=district['name'])
            for district in districts for row in district['rows'] for token in row['tokens']]


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


def districts_are_provinces(regions: list) -> bool:
    """
    Whether every district of a part is a province, so that the card can call them «provincias».

    Parameters
    ----------
    regions : list
        ``regions`` of the ``seat-odds`` part (``id``).

    Returns
    -------
    bool
        ``True`` when every id is an INE province code, below ``FIRST_NON_PROVINCE_ID`` (the islands of Canarias and
        its «Lista autonómica», those of Baleares and the three districts of Asturias are not).
    """
    return all(region['id'] < FIRST_NON_PROVINCE_ID for region in regions)


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
        single party, it is the one «quien más opciones tiene» for them (never that nobody else can win them: other
        parties may reach them below ``PLAY_SHARE``), and without any the clause is dropped.
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
    if safe == 0:
        head = 'El escaño no es seguro' if seats == 1 else 'Ninguno de los {} escaños es seguro'.format(seats)
        if not rivals:
            return head + '.'
        if len(rivals) == 1:
            return '{}: quien más opciones tiene es {}.'.format(head, rivals[0])
        return '{}: se {} disputan {}.'.format(head, 'lo' if single else 'los', _join_names(rivals))
    how_many = '1 es seguro' if safe == 1 else '{} son seguros'.format(safe)
    sentence = 'De los {} escaños, {} ({}).'.format(seats, how_many, listing)
    if not rivals:
        return sentence
    if len(rivals) == 1:
        others = 'el otro' if single else 'los otros {}'.format(left)
        return '{} Para {}, quien más opciones tiene es {}.'.format(sentence, others, rivals[0])
    others = 'El otro' if single else 'Los otros {}'.format(left)
    return '{} {} se {} disputan {}.'.format(sentence, others, 'lo' if single else 'los', _join_names(rivals))


def threshold_label(share: float) -> str:
    """
    A threshold as the texts of the card write it.

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
        A row of ``district_views`` (``safe``, ``likely``, ``play``).

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
        Party name.
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
        A row of ``district_views``.
    name : str
        Party name, for ``tokens_reading``.
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


def published_seat_odds(scope: str, run: str, mode: str) -> Optional[dict]:
    """
    The ``seat-odds`` part of a run when the card «Escaños que bailan» exists for it.

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


def _district_block(district: dict, parties: Catalog) -> dict:
    """
    The block of a district, beside or under the map: its head, its sentence and its rows.

    Parameters
    ----------
    district : dict
        A view from ``district_views``.
    parties : Catalog
        Catalogue of the scope (name to ``fullname``, ``color`` and ``block``; a party outside it gets its name and the
        grey of the others).

    Returns
    -------
    dict
        ``slug``, ``name``, ``seats``, its options of each state (``safe``, ``likely``, ``play``, for its chips),
        ``sentence`` (``district_sentence``) and ``rows``: each through ``drawn_row`` with the colour of its party, and
        its ``fullname``.
    """
    rows = [dict(drawn_row(row, row['name'], parties[row['name']]['color']), fullname=parties[row['name']]['fullname'])
            for row in district['rows']]

    return {
        **{key: district[key] for key in ('slug', 'name', 'seats') + STATES},
        'sentence': district_sentence(district),
        'rows': rows,
    }


def _map_view(districts: list, data: dict, parties: Catalog, scope: str) -> dict:
    """
    The view of the map: its tiles and the block of every district.

    Parameters
    ----------
    districts : list
        Views from ``district_views``, sorted by name.
    data : dict
        Data of the ``seat-odds`` part (its ``parties``, the order of the options of a tile).
    parties : Catalog
        Catalogue of the scope (name to ``fullname``, ``color`` and ``block``).
    scope : str
        Scope code, for the positions of its districts (``district_positions``).

    Returns
    -------
    dict
        ``grid`` (``map_tiles`` with the ``district_positions`` of the scope, each option drawn with the colour of its
        party, ``drawn_token``, and with ``party_slug``, the class the filter of the parties dims) and ``cards``
        (``_district_block`` of every district, by name).
    """
    grid = map_tiles(districts, district_positions(scope), data.get('parties') or [])
    tiles = [dict(tile, options=[dict(drawn_token(option, parties[option['party']]['color']),
                                      party_slug=slugify(option['party'])) for option in tile['options']])
             for tile in grid['tiles']]

    return {'grid': dict(grid, tiles=tiles), 'cards': [_district_block(district, parties) for district in districts]}


def _columns_view(views: list, order: list, blocks: Optional[dict], parties: Catalog) -> dict:
    """
    The view of the columns: every seat option of the scope in the column of its party.

    Parameters
    ----------
    views : list
        Views from ``district_views``.
    order : list
        The parties in the order of the table of results of the event page (``result_rows``).
    blocks : dict, optional
        ``bmaps.blocks`` of the run (``party_columns``).
    parties : Catalog
        Catalogue of the scope (name to ``fullname``, ``color`` and ``block``).

    Returns
    -------
    dict
        ``columns`` (``party_columns``, each with ``color``, ``None`` for ``REST_COLUMN``, and its options drawn, each
        with the ``option_title`` that names its district), ``ticks`` (``column_ticks``) and ``rest`` (``name`` and
        ``color`` of the parties of ``REST_COLUMN``, empty without it).
    """
    # a chart of the whole scope: the title of each circle names its district
    options = [dict(option, title=option_title(option['k'], option['party'], option['label'], option['district']))
               for option in scope_options(views)]
    columns = [dict(column, color=None if column['name'] == REST_COLUMN else parties[column['name']]['color'],
                    options=[drawn_token(option, parties[option['party']]['color']) for option in column['options']])
               for column in party_columns(options, order, blocks)]
    rest = next((column['members'] for column in columns if column['name'] == REST_COLUMN), [])

    return {
        'columns': columns,
        'ticks': column_ticks(columns),
        'rest': [{'name': name, 'color': parties[name]['color']} for name in rest],
    }


def seat_odds_card(scope: str, run: str, mode: str, meta: dict, parties: Catalog, summary: dict,
                   sign: str) -> Optional[dict]:
    """
    The card «Escaños que bailan» of the event page (``/{scope}``, section ``escanos``): the map with the block of
    every district beside it, and the columns of the parties.

    Parameters
    ----------
    scope : str
        Scope code.
    run : str
        Run id the event page shows.
    mode : str
        Simulation mode.
    meta : dict
        Data of the ``meta`` part of the run (``bmaps.blocks``, for the columns).
    parties : Catalog
        Catalogue of the scope (name to ``fullname``, ``color`` and ``block``; a party outside it gets its name and the
        grey of the others).
    summary : dict
        Data of the ``summary`` part of the mode: the columns follow the order of its table of results
        (``result_rows``).
    sign : str
        Signature of the image of the card (``'{site_host} · {author_handle}'``).

    Returns
    -------
    dict or None
        ``None`` when the run has no card (``published_seat_odds``). Otherwise ``n_sim``, ``totals``
        (``scope_totals``), ``foot`` (``options_sentence`` of the totals), ``thresholds`` (``threshold_label`` of
        ``SAFE_SHARE``, ``LIKELY_SHARE`` and ``PLAY_SHARE``), ``state_words`` (``STATE_WORDS``), ``district_word``
        (``'provincia'`` when ``districts_are_provinces``, ``'circunscripción'`` otherwise), ``view_label`` (the label
        of the view of the map and of its tiles: ``'Provincias'`` or ``'Circunscripciones'``), ``parties`` (those with
        some option, in the order of the part: ``name``, ``slug``, ``fullname``, ``color``; the filter of the map),
        ``grid`` and ``cards`` (``_map_view``: the tiles of the map and the block of every district, by name),
        ``chosen`` (the ``slug`` of ``default_district``), ``columns``, ``ticks`` and ``rest`` (``_columns_view``, in
        the order of ``result_rows(summary)``), ``image_base`` (``'demoscopio-escanos-{scope}'``), ``image_file`` (the
        image of the map with the block of ``chosen``) and ``image_sign``.
    """
    data = published_seat_odds(scope, run, mode)

    if data is None:
        return None

    views = district_views(data)
    districts = sorted(views, key=lambda view: _sort_name(view['name']))
    chosen = default_district(districts)
    provinces = districts_are_provinces(data['regions'])
    order = [row['name'] for row in result_rows(summary)]
    blocks = ((meta or {}).get('bmaps') or {}).get('blocks')
    shown = _map_view(districts, data, parties, scope)
    charts = _columns_view(views, order, blocks, parties)
    totals = scope_totals(views)
    image_base = 'demoscopio-escanos-{}'.format(scope)

    return {
        'n_sim': data['n_sim'],
        'totals': totals,
        'foot': options_sentence(totals['options'], totals['seats']),
        'thresholds': {'safe': threshold_label(SAFE_SHARE), 'likely': threshold_label(LIKELY_SHARE),
                       'play': threshold_label(PLAY_SHARE)},
        'state_words': STATE_WORDS,
        'district_word': 'provincia' if provinces else 'circunscripción',
        'view_label': 'Provincias' if provinces else 'Circunscripciones',
        'parties': [{'name': name, 'slug': slugify(name), 'fullname': parties[name]['fullname'],
                     'color': parties[name]['color']} for name in _party_names(data, views)],
        'grid': shown['grid'],
        'cards': shown['cards'],
        'chosen': chosen['slug'],
        'columns': charts['columns'],
        'ticks': charts['ticks'],
        'rest': charts['rest'],
        'image_base': image_base,
        'image_file': '{}-{}.png'.format(image_base, chosen['slug']),
        'image_sign': sign,
    }
