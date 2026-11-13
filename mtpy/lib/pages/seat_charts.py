"""
The layout of the drawings of «Escaños que bailan», the card of the event page: where each seat option goes, in the
tiles of the map and in the columns of the parties.

The functions are pure: they take the views of ``seat_odds.district_views`` (or the options of ``scope_options``) and
return plain dicts and lists, with the options still undrawn (the card passes them through ``drawn_token``). This
module imports nothing from ``seat_odds``, which imports it: the states of a seat option (``STATES``) and their count
(``state_counts``) live here.
"""
from typing import Optional

from .escanos import _sort_name

STATES = ('safe', 'likely', 'play')  # the states of a seat option, from the surest
REST_COLUMN = 'Otros'  # the last column of the parties: every party outside the blocks Izquierda and Derecha


def _option_key(option: dict, order: list) -> tuple:
    """
    Sort key of a seat option inside a tile: by party, then by state, then by seat.

    Parameters
    ----------
    option : dict
        A seat option with ``party``, ``state`` and ``k``.
    order : list
        Party names in the order of the part; a party not in it goes last.

    Returns
    -------
    tuple
        The position of the party in ``order``, of the state in ``STATES`` and ``k``.
    """
    party = order.index(option['party']) if option['party'] in order else len(order)

    return party, STATES.index(option['state']), option['k']


def map_tiles(districts: list, positions: dict, order: list) -> dict:
    """
    The tiles of the map of a scope: one per district, with its seat options.

    Parameters
    ----------
    districts : list
        Views from ``district_views`` (``id``, ``name``, ``slug``, ``seats`` and ``rows`` with ``name`` and
        ``tokens``).
    positions : dict
        ``district_positions`` of the scope: ``{region_id: {'abbr', 'col', 'row', 'w', 'h'}}``.
    order : list
        Party names in the order of the part, for the order of the options.

    Returns
    -------
    dict
        ``placed``, whether every district has a position, and ``tiles``: ``id``, ``name``, ``slug``, ``seats``,
        ``label``, ``col``, ``row``, ``w``, ``h`` and ``options`` (its seat options, each with ``party``, by party,
        state and ``k``). Placed, ``label`` is the abbreviation and the tiles go by row and column; otherwise
        ``label`` is the name, ``col``, ``row``, ``w`` and ``h`` are ``None`` and the tiles go by name with accents
        folded.
    """
    placed = bool(districts) and all(district['id'] in positions for district in districts)
    tiles = []

    for district in districts:
        options = [dict(token, party=row['name']) for row in district['rows'] for token in row['tokens']]
        tile = {key: district[key] for key in ('id', 'name', 'slug', 'seats')}
        tile['options'] = sorted(options, key=lambda option: _option_key(option, order))

        if placed:
            where = positions[district['id']]
            tile.update(label=where['abbr'], col=where['col'], row=where['row'], w=where['w'], h=where['h'])
        else:
            tile.update(label=district['name'], col=None, row=None, w=None, h=None)

        tiles.append(tile)

    if placed:
        tiles.sort(key=lambda tile: (tile['row'], tile['col']))
    else:
        tiles.sort(key=lambda tile: _sort_name(tile['name']))

    return {'placed': placed, 'tiles': tiles}


def state_counts(options: list) -> dict:
    """
    How many seat options of each state.

    Parameters
    ----------
    options : list
        Seat options with ``state`` (those of ``seat_tokens`` or ``scope_options``).

    Returns
    -------
    dict
        One count per state of ``STATES``.
    """
    return {state: sum(option['state'] == state for option in options) for state in STATES}


def _drawn_names(options: list, order: list) -> list:
    """
    The parties with some seat option, in a given order.

    Parameters
    ----------
    options : list
        Seat options with ``party``.
    order : list
        Party names; the parties with options outside it go after them, in the order they first appear.

    Returns
    -------
    list
        Party names, each once.
    """
    present = list(dict.fromkeys(option['party'] for option in options))

    return [name for name in order if name in present] + [name for name in present if name not in order]


def party_columns(options: list, order: list, blocks: Optional[dict]) -> list:
    """
    The columns of the seat options of a scope: one per party of the blocks Izquierda and Derecha, and the rest.

    Parameters
    ----------
    options : list
        Seat options with ``party``, ``state`` and ``share`` (``scope_options``).
    order : list
        Party names in the order of the columns (that of the table of results of the event page); a party with
        options outside it goes last.
    blocks : dict, optional
        ``bmaps.blocks`` of the run (block name to party names). Valid when it has both ``Izquierda`` and
        ``Derecha``.

    Returns
    -------
    list
        One dict per column: ``name``, ``members`` (its parties), its options of each state (``safe``, ``likely``,
        ``play``) and ``options``, by state, by party (in ``order``) and by share descending. With valid ``blocks``,
        one column per party of Izquierda or Derecha with some option, in ``order`` (not block by block), and a last
        one, ``REST_COLUMN``, with every other party with options, if any; otherwise one column per party with
        options, in ``order``.
    """
    names = _drawn_names(options, order)

    if isinstance(blocks, dict) and 'Izquierda' in blocks and 'Derecha' in blocks:
        own = set(blocks['Izquierda'] or []) | set(blocks['Derecha'] or [])
        main = [name for name in names if name in own]
        rest = [name for name in names if name not in own]
        groups = [(name, [name]) for name in main] + ([(REST_COLUMN, rest)] if rest else [])
    else:
        groups = [(name, [name]) for name in names]

    columns = []

    for name, members in groups:
        mine = sorted((option for option in options if option['party'] in members),
                      key=lambda option: (STATES.index(option['state']), names.index(option['party']),
                                          -option['share']))
        columns.append({'name': name, 'members': members, **state_counts(mine), 'options': mine})

    return columns


def column_ticks(columns: list) -> list:
    """
    The marks of the axis of the columns: one every 50 seat options, the last one above the tallest column.

    Parameters
    ----------
    columns : list
        Columns with ``options`` (``party_columns``).

    Returns
    -------
    list
        ``[50, 100, ...]`` up to the first mark at or above the size of the tallest column (``[50, 100, 150]`` for
        141 options), so that every column ends under a mark; empty without options.
    """
    tallest = max((len(column['options']) for column in columns), default=0)
    top = -(-tallest // 50) * 50

    return list(range(50, top + 1, 50))
