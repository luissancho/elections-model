"""
Houses page context.
"""
from typing import Optional

from .. import webapi
from ..webapi import query_pollster, query_ratings, site
from .common import api_url, bundle_part, common_context
from .filters import fmt_num

MTYPE_LABELS = {'public': 'pública', 'private': 'privada'}
RANKING_KEYS = ('pollster_id', 'pollster', 'rating', 'quality', 'num_polls', 'num_events', 'error_avg',
                'error_blocks', 'bias_blocks')
PATH_KEYS = ('event_date', 'rating', 'rating_adj', 'error_blocks', 'bias_blocks', 'num_polls')
HERDING_KEYS = ('event_date', 'n', 'ratio', 'p_value')
ANALYSIS_CSV = (
    ('house-effects', None, 'Efectos de casa por partido y bloque'),
    ('herding', None, 'Herding por casa'),
    ('herding', 'events', 'Herding por casa y ciclo'),
    ('poll-errors', None, 'Error de los sondeos en cada elección'),
)



def analysis_part(scope: str, name: str) -> Optional[dict]:
    """
    Data of a part of the published houses analysis of a scope.

    Parameters
    ----------
    scope : str
        Scope code.
    name : str
        Part name, one of ``bundle.ANALYSIS_PARTS``.

    Returns
    -------
    dict or None
        The ``data`` member of the part envelope, or ``None`` when the part is not published.

    Raises
    ------
    HttpError
        Any error of the bundle other than the 404 of an unpublished analysis.
    """
    return bundle_part(site().analysis, scope, name)


def ranking_rows(ratings, herding: Optional[dict]) -> list:
    """
    Rows of the pollsters ranking.

    Parameters
    ----------
    ratings : pandas.DataFrame
        Ratings of one event (``webapi.query_ratings``).
    herding : dict or None
        Data of the ``herding`` analysis part, or ``None`` when it is not published.

    Returns
    -------
    list of dict
        The pollsters with a rating, by ``rating`` descending (ties keep the input order), with
        the keys of ``RANKING_KEYS`` (``None`` when a column is missing) plus ``herding``, the
        pooled herding of the analysis for that pollster name or ``None``. ``pollster_id`` is an
        integer, for the profile links.
    """
    if ratings.empty or 'rating' not in ratings.columns:
        return []

    pooled = {row['pollster']: row.get('herding') for row in (herding or {}).get('rows') or []}
    rows = []

    for record in webapi.frame_to_records(ratings):
        if record.get('rating') is None:
            continue

        row = {key: record.get(key) for key in RANKING_KEYS}
        row['pollster_id'] = int(row['pollster_id'])
        row['herding'] = pooled.get(row['pollster'])
        rows.append(row)

    return sorted(rows, key=lambda row: row['rating'], reverse=True)


def ranking_headline(ranking: list) -> Optional[str]:
    """
    Headline of the houses page.

    Parameters
    ----------
    ranking : list of dict
        Rows of ``ranking_rows``, best rated first.

    Returns
    -------
    str or None
        The number of rated houses and the best rated one with its rating; ``None`` without ranking.
    """
    if not ranking:
        return None

    best = ranking[0]

    return '{} casas con rating para la próxima elección; la mejor valorada es {} ({}).'.format(
        len(ranking), best['pollster'], fmt_num(best['rating'], 1))


def profile_headline(name: str, current: dict) -> str:
    """
    Headline of a pollster profile.

    Parameters
    ----------
    name : str
        Pollster name.
    current : dict
        ``rating`` and ``quality`` of its next-election row (``None`` when missing).

    Returns
    -------
    str
        The rating and prior quality of the house in plain words.
    """
    return '{}: rating {} de 100, calidad previa {}.'.format(
        name, fmt_num(current.get('rating'), 1), fmt_num(current.get('quality'), 1))


def house_effects_grid(part: dict) -> dict:
    """
    Grid of the house effects of the current cycle: one row per pollster, one cell per column.

    Parameters
    ----------
    part : dict
        Data of the ``house-effects`` analysis part (``parties``, ``blocks``, ``rows``).

    Returns
    -------
    dict
        ``{'columns': parties + blocks, 'rows': [{'pollster', 'cells'}]}``; each cell is
        ``{'cycle', 'hist', 'trend'}`` (all ``None`` when the analysis has no row for it). Only
        pollsters with some ``cycle`` are kept, in the order of the analysis, then sorted by the
        ``cycle`` of the first block descending (of the first column without blocks), ``None``
        last.
    """
    blocks = list(part.get('blocks') or [])
    columns = list(part.get('parties') or []) + blocks
    values = {}

    for row in part.get('rows') or []:
        values.setdefault(row['pollster'], {})[row['name']] = row

    rows = []

    for pollster, by_name in values.items():
        cells = [{key: (by_name.get(name) or {}).get(key) for key in ('cycle', 'hist', 'trend')}
                 for name in columns]

        if any(cell['cycle'] is not None for cell in cells):
            rows.append({'pollster': pollster, 'cells': cells})

    if columns:
        index = columns.index(blocks[0]) if blocks else 0
        rows.sort(key=lambda row: (row['cells'][index]['cycle'] is None, -(row['cells'][index]['cycle'] or 0)))

    return {'columns': columns, 'rows': rows}


def herding_rows(part: dict) -> list:
    """
    Rows of the herding table.

    Parameters
    ----------
    part : dict
        Data of the ``herding`` analysis part.

    Returns
    -------
    list of dict
        ``part['rows']`` by ``herding`` ascending (the most herded first), ``None`` last.
    """
    rows = part.get('rows') or []

    return sorted(rows, key=lambda row: (row.get('herding') is None, row.get('herding') or 0))


def profile_rows(records: list, keys: tuple) -> list:
    """
    Rows of a profile table, by event date.

    Parameters
    ----------
    records : list of dict
        JSON records of a table of the pollster (``webapi.query_pollster``).
    keys : tuple of str
        Keys to keep, in order (``None`` when a record lacks one).

    Returns
    -------
    list of dict
        The records reduced to ``keys``, sorted by ``event_date`` (stable).
    """
    rows = [{key: record.get(key) for key in keys} for record in records]

    return sorted(rows, key=lambda row: row.get('event_date') or '')


def effects_table(records: list) -> dict:
    """
    House effects of a pollster by election and party.

    Parameters
    ----------
    records : list of dict
        Its rows of the house effects table (``event_date``, ``party``, ``dev_result_c``...).

    Returns
    -------
    dict
        ``{'columns': parties, 'rows': [{'event_date', 'cells'}]}``: the parties in order of first
        appearance, one row per election by date and the ``dev_result_c`` of each party
        (``None`` when missing). An election without any value (the next one, which has no
        result yet) has no row.
    """
    columns = []
    values = {}

    for record in records:
        party = record.get('party')

        if party not in columns:
            columns.append(party)

        values.setdefault(record.get('event_date'), {})[party] = record.get('dev_result_c')

    rows = [
        {'event_date': event_date, 'cells': [values[event_date].get(party) for party in columns]}
        for event_date in sorted(values, key=lambda value: value or '')
    ]

    return {'columns': columns, 'rows': [row for row in rows if any(cell is not None for cell in row['cells'])]}


def casas_context(state: dict, active: str = '/casas') -> dict:
    """
    Template context of the houses page.

    The ranking comes from the database; the house effects and the herding from the houses
    analysis of the scope, whose ``meta`` part marks it as complete.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    active : str, optional
        ``href`` of the page in the navigation.

    Returns
    -------
    dict
        The common context plus ``ranking`` (``ranking_rows``), ``grid`` (``house_effects_grid``
        or ``None``), ``herding`` (``herding_rows`` or ``None``), ``analysis_missing``,
        ``pollster_ids`` (``{name: id}`` of the pollsters rated in the event, for the links),
        ``csv_links`` and ``initial`` (``state``, ``meta``, ``ranking`` and ``grid``).

    Raises
    ------
    HttpError
        503 without a bundle or without a database; 404 for an unpublished scope or a missing
        run.
    """
    from .elecciones import with_abs  # lazy: elecciones imports this module

    context = common_context(state, active)
    full = context['state']
    scope = full['scope']
    ratings = query_ratings(scope)
    missing = analysis_part(scope, 'meta') is None
    effects = None if missing else analysis_part(scope, 'house-effects')
    herding = None if missing else analysis_part(scope, 'herding')
    ranking = with_abs(ranking_rows(ratings, herding))
    grid = house_effects_grid(effects) if effects is not None else None
    links = [{'label': 'Ratings de la próxima elección', 'href': api_url(scope, 'ratings', fmt='csv', kind='data')}]

    if not missing:
        links += [
            {'label': label, 'href': api_url(scope, name, fmt='csv', kind='analysis', table=table)}
            for name, table, label in ANALYSIS_CSV
        ]

    context['ranking'] = ranking
    context['headline_text'] = ranking_headline(ranking)
    context['grid'] = grid
    context['herding'] = herding_rows(herding) if herding is not None else None
    context['analysis_missing'] = missing
    context['pollster_ids'] = {
        record['pollster']: int(record['pollster_id'])
        for record in webapi.frame_to_records(ratings)
        if record.get('pollster') is not None and record.get('pollster_id') is not None
    }
    context['csv_links'] = links
    context['initial'] = {'state': full, 'meta': context['meta'], 'ranking': ranking, 'grid': grid}

    return context


def casa_context(state: dict, pollster_id: int, active: str = '/casas') -> dict:
    """
    Template context of the profile of a pollster.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    pollster_id : int
        Pollster id validated by ``webapi.check_pollster``.
    active : str, optional
        ``href`` of the page in the navigation.

    Returns
    -------
    dict
        The common context plus ``profile`` (``webapi.query_pollster``), ``mtype`` (the type of
        the pollster through ``MTYPE_LABELS``: the raw value when unknown, ``None`` when missing),
        ``current`` (the ``rating`` and ``quality`` of its last ratings row in the scope, the next
        election; both
        ``None`` without ratings: the catalogue keeps only the national rating), ``path`` (its
        ratings by election, ``PATH_KEYS``), ``effects_table`` (``effects_table``), ``herding_table``
        (its herding by cycle, ``HERDING_KEYS``), ``bullseye`` (its rows of the ``poll-errors``
        analysis part: ``event``, ``event_date``, ``days`` and the error of each block, by date;
        empty without analysis), ``blocks`` (the blocks of that part), ``analysis_missing`` and
        ``initial`` (``state``, ``meta``, ``path``, ``bullseye`` and ``blocks``).

    Raises
    ------
    HttpError
        503 without a bundle or without a database; 404 for an unpublished scope, a missing run
        or an unknown pollster.
    """
    from .elecciones import with_abs  # lazy: elecciones imports this module

    context = common_context(state, active)
    full = context['state']
    scope = full['scope']
    profile = query_pollster(pollster_id, scope)
    missing = analysis_part(scope, 'meta') is None
    errors = (None if missing else analysis_part(scope, 'poll-errors')) or {}
    blocks = list(errors.get('blocks') or [])
    name = profile['pollster'].get('name')
    bullseye = profile_rows(
        [row for row in errors.get('rows') or [] if row.get('pollster') == name],
        ('event', 'event_date', 'days') + tuple(blocks),
    )
    path = with_abs(profile_rows(profile['ratings'], PATH_KEYS))
    latest = profile_rows(profile['ratings'], ('rating', 'quality', 'event_date'))

    mtype = profile['pollster'].get('mtype')

    context['profile'] = profile
    context['mtype'] = MTYPE_LABELS.get(mtype, mtype) if mtype not in (None, '') else None
    context['current'] = latest[-1] if latest else {'rating': None, 'quality': None, 'event_date': None}
    context['path'] = path
    context['headline_text'] = profile_headline(name, context['current'])
    context['effects_table'] = effects_table(profile['house_effects'])
    context['herding_table'] = profile_rows(profile['herding'], HERDING_KEYS)
    context['bullseye'] = bullseye
    context['blocks'] = blocks
    context['analysis_missing'] = missing
    context['initial'] = {'state': full, 'meta': context['meta'], 'path': path, 'bullseye': bullseye,
                          'blocks': blocks}

    return context
