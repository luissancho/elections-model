"""
Houses page context.
"""
from typing import Callable, Optional, Sequence

from .. import webapi
from ..webapi import query_pollster, query_ratings, site
from .common import analysis_part, api_url, common_context
from .elecciones import error_blocks, history_events, history_rows, history_scopes, poll_error_parts
from .filters import fmt_num, fmt_date

RATING_SCOPE = 'es'
# Rows of the ranking the page shows before "Cargar más".
RANKING_SHOWN = 50
MTYPE_LABELS = {'public': 'pública', 'private': 'privada'}
RANKING_KEYS = ('pollster_id', 'pollster', 'rating', 'quality', 'num_polls', 'num_events', 'error_avg',
                'error_blocks', 'bias_blocks')
PATH_KEYS = ('event_date', 'rating', 'rating_adj', 'error_blocks', 'bias_blocks', 'num_polls')
HERDING_KEYS = ('event_scope', 'event_date', 'n', 'ratio', 'p_value')
ANALYSIS_CSV = (
    ('house-effects', None, 'Efectos de encuestadora por partido y bloque'),
    ('herding', None, 'Rebaño por encuestadora'),
    ('herding', 'events', 'Rebaño por encuestadora y ciclo'),
    ('poll-errors', None, 'Error de los sondeos en cada elección'),
)



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
        The pollsters with a rating and at least one poll (``num_polls``; a pollster without polls has its
        prior quality as rating, nothing to rank), by ``rating`` descending (ties keep the input order), with
        the keys of ``RANKING_KEYS`` (``None`` when a column is missing) plus ``herding``, the pooled herding of
        the analysis for that pollster name or ``None``. ``pollster_id`` is an integer, for the profile links.
    """
    if ratings.empty or 'rating' not in ratings.columns:
        return []

    pooled = {row['pollster']: row.get('herding') for row in (herding or {}).get('rows') or []}
    rows = []

    for record in webapi.frame_to_records(ratings):
        if record.get('rating') is None or not record.get('num_polls'):
            continue

        row = {key: record.get(key) for key in RANKING_KEYS}
        row['pollster_id'] = int(row['pollster_id'])
        row['herding'] = pooled.get(row['pollster'])
        rows.append(row)

    return sorted(rows, key=lambda row: row['rating'], reverse=True)


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


def election_label(scope: str, date: str, names: dict, scope_names: dict) -> str:
    """
    Name of an election in the tables of a profile.

    Parameters
    ----------
    scope : str
        Scope code of the election.
    date : str
        Its ISO date.
    names : dict
        ``{(scope, date): name}`` of the published elections.
    scope_names : dict
        ``{code: name}`` of the catalogue scopes.

    Returns
    -------
    str
        The published name or, for an election without a page (the current cycle, an old election), its date
        and the name of its scope, which the date alone would not tell apart.
    """
    return names.get((scope, date)) or '{} · {}'.format(fmt_date(date), scope_names.get(scope, scope))


def effects_table(records: list, label: Optional[Callable] = None, order: Sequence[str] = ()) -> dict:
    """
    House effects of a pollster by election and party.

    Parameters
    ----------
    records : list of dict
        Its rows of the house effects table (``event_scope``, ``event_date``, ``party``, ``dev_result_c``...);
        a row without ``event_scope`` belongs to ``RATING_SCOPE``.
    label : callable, optional
        ``(scope, date) -> str``, the name of each election; its date by default.
    order : sequence of str, optional
        Party names that go first, in this order (the catalogue of the run); the rest follow by first
        appearance.

    Returns
    -------
    dict
        ``{'columns': parties, 'rows': [{'scope', 'event_date', 'label', 'cells'}]}``: one row per election by
        date (ties in catalogue order of the records) with the ``dev_result_c`` of each party (``None`` when
        missing). An election without any value (the next one, which has no result yet) has no row.
    """
    seen = []
    values = {}

    for record in records:
        party = record.get('party')

        if party not in seen:
            seen.append(party)

        key = (record.get('event_scope') or RATING_SCOPE, record.get('event_date'))
        values.setdefault(key, {})[party] = record.get('dev_result_c')

    columns = [name for name in order if name in seen] + [name for name in seen if name not in order]
    rows = [
        {'scope': scope, 'event_date': date, 'label': label(scope, date) if label else fmt_date(date),
         'cells': [values[(scope, date)].get(party) for party in columns]}
        for scope, date in sorted(values, key=lambda key: (key[1] or '', key[0]))
    ]

    return {'columns': columns, 'rows': [row for row in rows if any(cell is not None for cell in row['cells'])]}


def casas_context(state: dict, active: str = '/casas') -> dict:
    """
    Template context of the houses page.

    The ranking is always the national one (``RATING_SCOPE``), whatever the scope: the rating of a
    pollster does not depend on it, and its herding column comes from the national analysis for the same
    reason. The page renders every row and hides those beyond ``RANKING_SHOWN`` behind "Cargar más". The
    ratings come from the database; the house effects and the herding cards from the houses analysis of the
    scope (chosen in the house-effects card), whose ``meta`` part marks it as complete.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    active : str, optional
        ``href`` of the page in the navigation.

    Returns
    -------
    dict
        The common context plus ``ranking`` (``ranking_rows``), ``ranking_shown`` (``RANKING_SHOWN``) and
        ``ranking_hidden`` (rows beyond them, behind "Cargar más"), ``error_max`` (the largest ``error_avg`` of
        the ranking, the fullest shade of that column; 0 without one), ``grid`` (``house_effects_grid`` or
        ``None``), ``herding`` (``herding_rows`` or ``None``), ``analysis_missing``, ``pollster_ids``
        (``{name: id}`` of the pollsters rated in the event, for the links), ``ranking_event`` (ISO date of the
        election the ranking is for, ``None`` without ratings), ``csv_links`` and ``initial`` (``state``,
        ``meta`` and ``grid``).

    Raises
    ------
    HttpError
        503 without a bundle or without a database; 404 for an unpublished scope or a missing
        run.
    """
    context = common_context(state, active)
    full = context['state']
    scope = full['scope']
    ratings = query_ratings(RATING_SCOPE)
    missing = analysis_part(scope, 'meta') is None
    effects = None if missing else analysis_part(scope, 'house-effects')
    herding = None if missing else analysis_part(scope, 'herding')
    if scope == RATING_SCOPE:
        national_herding = herding
    else:
        national_herding = None if analysis_part(RATING_SCOPE, 'meta') is None else analysis_part(RATING_SCOPE, 'herding')
    ranking = ranking_rows(ratings, national_herding)
    grid = house_effects_grid(effects) if effects is not None else None
    links = [{'label': 'Ratings de la próxima elección general', 'href': api_url(RATING_SCOPE, 'ratings', fmt='csv', kind='data')}]

    if not missing:
        links += [
            {'label': label, 'href': api_url(scope, name, fmt='csv', kind='analysis', table=table)}
            for name, table, label in ANALYSIS_CSV
        ]

    context['ranking'] = ranking
    context['ranking_shown'] = RANKING_SHOWN
    context['ranking_hidden'] = max(0, len(ranking) - RANKING_SHOWN)
    context['error_max'] = max((row['error_avg'] for row in ranking if row.get('error_avg') is not None), default=0)
    context['grid'] = grid
    context['herding'] = herding_rows(herding) if herding is not None else None
    context['analysis_missing'] = missing
    context['pollster_ids'] = {
        record['pollster']: int(record['pollster_id'])
        for record in webapi.frame_to_records(ratings)
        if record.get('pollster') is not None and record.get('pollster_id') is not None
    }
    context['ranking_event'] = (
        str(ratings['event_date'].max())[:10] if 'event_date' in ratings.columns and not ratings.empty else None)
    context['csv_links'] = links
    context['initial'] = {'state': full, 'meta': context['meta'], 'grid': grid}

    return context


def profile_kpis(current: dict, mtype: Optional[str]) -> list:
    """
    The KPI row of a pollster profile: its rating, its prior quality and its type.

    Parameters
    ----------
    current : dict
        The latest rating row (``rating``, ``quality``).
    mtype : str or None
        Label of the type of the pollster.

    Returns
    -------
    list of dict
        Three ``{'label', 'value', 'unit', 'support', 'ok', 'color'}`` rows for ``_macros.kpis``.
    """
    return [
        {'label': 'Rating', 'value': fmt_num(current.get('rating'), 1), 'unit': '', 'support': 'entra así en el promedio',
         'ok': False, 'color': None},
        {'label': 'Calidad previa', 'value': fmt_num(current.get('quality'), 1), 'unit': '', 'support': '', 'ok': False,
         'color': None},
        {'label': 'Tipo', 'value': mtype or '–', 'unit': '', 'support': '', 'ok': False, 'color': None},
    ]


def casa_context(state: dict, pollster_id: int, active: str = '/casas') -> dict:
    """
    Template context of the profile of a pollster.

    The profile is global, whatever the scope of the request: the ratings are the national ones
    (``RATING_SCOPE``; the rating is the same number in every scope), the bullseye has its last poll in every
    published election of every scope with analysis (the same elections as the error history) and the house
    effects and the herding tables cover the elections and cycles of every scope.

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
        The common context plus ``kpis`` (``profile_kpis``), ``profile`` (``webapi.query_pollster`` of every scope), ``mtype`` (the type of
        the pollster through ``MTYPE_LABELS``: the raw value when unknown, ``None`` when missing), ``current``
        (the ``rating`` and ``quality`` of its last national ratings row, the next general election; both
        ``None`` without ratings), ``path`` (its national ratings by election, ``PATH_KEYS``),
        ``effects_table`` (``effects_table`` with the elections named by ``election_label``), ``herding_table``
        (its herding by cycle, ``HERDING_KEYS`` plus ``label``, by date), ``bullseye`` (its rows of the
        ``poll-errors`` parts of the published elections: ``event``, ``event_date``, ``year``, ``days``,
        ``scope``, ``name`` and the error of each block, by date; empty without analysis), ``bullseye_scopes``
        (``{code, name}`` of the scopes with a bullseye row, in catalogue order) and ``bullseye_years`` (the
        years of the rows, ascending), the options of the bullseye filters, ``blocks`` (the pair of
        ``error_blocks``), ``analysis_missing`` (no scope has a published analysis) and ``initial`` (``state``,
        ``meta``, ``path``, ``bullseye``, ``blocks`` and ``scopes`` = ``bullseye_scopes``).

    Raises
    ------
    HttpError
        503 without a bundle or without a database; 404 for an unpublished scope, a missing run
        or an unknown pollster.
    """
    context = common_context(state, active)
    full = context['state']
    profile = query_pollster(pollster_id, RATING_SCOPE, every_scope=True)
    name = profile['pollster'].get('name')
    scopes = history_scopes()
    parts = poll_error_parts(scopes)
    events = history_events(scopes)
    names = {(event['scope'], event['date']): event['name'] for event in events if event.get('name')}
    scope_names = {row['code']: row['name'] for row in site().scopes()['scopes']}
    blocks = error_blocks(scopes, parts)

    def label(scope, date):
        """Name of an election of the tables."""
        return election_label(scope, date, names, scope_names)

    bullseye = sorted(
        (dict({key: row.get(key) for key in ('event', 'event_date', 'year', 'days', 'scope') + tuple(blocks)},
              name=label(row['scope'], row['event_date']))
         for row in history_rows(scopes, events, parts) if row.get('pollster') == name),
        key=lambda row: row['event_date'] or '')
    polled = {row['scope'] for row in bullseye}
    bullseye_scopes = [{'code': scope['code'], 'name': scope['name']} for scope in scopes if scope['code'] in polled]
    bullseye_years = sorted({row['year'] for row in bullseye if row.get('year') is not None})
    path = profile_rows(profile['ratings'], PATH_KEYS)
    latest = profile_rows(profile['ratings'], ('rating', 'quality', 'event_date'))
    mtype = profile['pollster'].get('mtype')

    context['profile'] = profile
    context['mtype'] = MTYPE_LABELS.get(mtype, mtype) if mtype not in (None, '') else None
    context['current'] = latest[-1] if latest else {'rating': None, 'quality': None, 'event_date': None}
    context['kpis'] = profile_kpis(context['current'], context['mtype'])
    context['path'] = path
    context['effects_table'] = effects_table(profile['house_effects'], label, order=list(context['catalog']))
    context['herding_table'] = [dict(row, label=label(row.get('event_scope') or RATING_SCOPE, row['event_date']))
                                for row in profile_rows(profile['herding'], HERDING_KEYS)]
    context['bullseye'] = bullseye
    context['bullseye_scopes'] = bullseye_scopes
    context['bullseye_years'] = bullseye_years
    context['blocks'] = blocks
    context['analysis_missing'] = not any(scope['analysis'] for scope in scopes)
    context['initial'] = {'state': full, 'meta': context['meta'], 'path': path, 'bullseye': bullseye,
                          'blocks': blocks, 'scopes': bullseye_scopes}

    return context
