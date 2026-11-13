"""
The layout of the drawings of «Escaños que bailan» (``/{scope}/circunscripciones``): where each seat option goes.

The functions are pure: they take the views of ``circunscripciones.district_views`` and return plain dicts and lists,
with the options still undrawn (the page passes them through ``drawn_token``). This module imports nothing from
``circunscripciones``, which imports it: the states of a seat option (``STATES``) live here.
"""
from .escanos import _sort_name

STATES = ('safe', 'likely', 'play')  # the states of a seat option, from the surest


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
