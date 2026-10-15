"""
Past elections page context: the list of published elections and the detail of one.
"""
from typing import Callable, Optional

from ...core.api import HttpError
from ..webapi import check_date, event_params, site
from .common import Catalog, analysis_part, api_url, block_color, bundle_part, common_context, published_scopes
from .filters import fmt_num
from .promedios import BLOCK_ORDER, chart_polls, lead_strip, poll_months, polls_table, pollster_options

SUMMARY_KEYS = ('n_polls', 'n_pollsters', 'error_avg', 'error_blocks', 'bias_blocks', 'bias')
# Block drawn to the left in the chart that splits the error of the last polls by block (its mate goes right).
LEFT_BLOCK = 'Izquierda'
HISTORY_COLUMNS = ('scope', 'scope_name', 'year', 'event_date', 'event', 'pollster', 'days', 'error_avg',
                   'error_blocks', 'error_blocks_abs', 'bias')
EVENT_CSV = (
    ('pollsters', None, 'Encuestadoras en esta elección', False),
    ('polls', None, 'Sondeos de esta elección, con su error', False),
    ('cycle', None, 'Sondeos del ciclo, tal como se publicaron', False),
    ('results', None, 'Resultados oficiales', False),
    ('model', 'shares', 'Modelo: voto a cada horizonte', True),
    ('model', 'seats', 'Modelo: escaños a cada horizonte', True),
    ('model', 'blocks', 'Modelo: bloques a cada horizonte', True),
)
# Rows of the results table the detail shows before "Ver más" when the election has no list of parties in
# ``data/params.json`` (``parties.event``), or when none of its results is in that list.
RESULTS_SHOWN = 15
# Keys of a last-poll row the detail embeds for its charts (plus the signed error of each block).
LAST_KEYS = ('pollster', 'pollster_id', 'days', 'sample_size', 'error_avg', 'error_blocks', 'error_blocks_abs')


def event_part(scope: str, date: str, name: str) -> Optional[dict]:
    """
    Data of a part of a published past election.

    Parameters
    ----------
    scope : str
        Scope code.
    date : str
        Election date.
    name : str
        Part name, one of ``bundle.EVENT_PARTS``.

    Returns
    -------
    dict or None
        The ``data`` member of the part envelope, or ``None`` when the part is not published.

    Raises
    ------
    HttpError
        Any error of the bundle other than the 404 of an unpublished part.
    """
    return bundle_part(site().event, scope, date, name)


def with_abs(rows: list) -> list:
    """
    Copy rows adding the absolute value of their signed block error.

    Parameters
    ----------
    rows : list of dict
        Rows with the signed ``error_blocks`` of the published bundle.

    Returns
    -------
    list of dict
        New dicts with ``error_blocks_abs`` and ``last_error_blocks_abs`` (absolute values of ``error_blocks`` and
        ``last_error_blocks`` (this one only on rows that have it), ``None`` when missing).
    """
    result = []

    for row in rows:
        error = row.get('error_blocks')
        new = dict(row, error_blocks_abs=None if error is None else abs(error))

        if 'last_error_blocks' in row:
            last = row['last_error_blocks']
            new['last_error_blocks_abs'] = None if last is None else abs(last)

        result.append(new)

    return result


def last_poll_rows(rows: list) -> list:
    """
    The last poll of each pollster of a past election.

    Parameters
    ----------
    rows : list of dict
        Rows of the ``polls`` part (``with_abs``): one per poll with ``pollster`` and ``days``.

    Returns
    -------
    list of dict
        One copy per pollster of its poll with the fewest ``days`` (the first one of the part on a tie, as the
        rows come ascending by ``days``; a row without a pollster is skipped), from the smallest to the largest
        ``error_blocks_abs`` (``None`` last).
    """
    last = {}

    for row in rows:
        name = row.get('pollster')

        if name is None:
            continue

        current = last.get(name)
        days = row.get('days')

        if current is None or (days is not None and (current.get('days') is None or days < current['days'])):
            last[name] = row

    return sorted((dict(row) for row in last.values()),
                  key=lambda row: (row['error_blocks_abs'] is None, row['error_blocks_abs'] or 0.0))


def published_events(scope: str) -> list:
    """
    Summary of every published past election of a scope.

    Parameters
    ----------
    scope : str
        Scope code.

    Returns
    -------
    list of dict
        One dict per published date, newest first, with ``date``, ``name``, ``n_polls``,
        ``n_pollsters``, ``error_avg``, ``error_blocks`` (signed), ``error_blocks_abs``, ``bias_blocks``,
        ``bias`` (by block) and ``has_model``, all from its ``meta`` part. A date without ``meta`` is skipped.
    """
    events = []

    for date in sorted(site().events(scope), reverse=True):
        meta = event_part(scope, date, 'meta')

        if meta is None:
            continue

        summary = meta.get('summary') or {}
        event = {'date': date, 'name': meta.get('name')}
        event.update({key: summary.get(key) for key in SUMMARY_KEYS})
        event['has_model'] = bool(meta.get('has_model'))
        events.append(event)

    return with_abs(events)


def last_results(scope: str) -> Optional[dict]:
    """
    Official results of the most recent published election of a scope, by party.

    Parameters
    ----------
    scope : str
        Scope code.

    Returns
    -------
    dict or None
        ``{'date': ISO date, 'year': int, 'parties': {name: {'seats', 'pct'}}}`` of the newest published election
        with a ``results`` part; ``None`` without one.
    """
    for event in published_events(scope):
        results = event_part(scope, event['date'], 'results')

        if results and results.get('rows'):
            return {
                'date': event['date'],
                'year': int(event['date'][:4]),
                'parties': {row['party']: {'seats': row.get('seats'), 'pct': row.get('pct')} for row in results['rows']},
            }

    return None


def error_rows(scope: str, dates: list, errors: Optional[dict] = None) -> Optional[list]:
    """
    Rows of the published ``poll-errors`` analysis of the given elections.

    Only the published elections count (the analysis has every election of the scope with results and polls), so
    the filters, the histogram, the bullseye and the weeks table speak of the same elections as the list.

    Parameters
    ----------
    scope : str
        Scope code.
    dates : list of str
        Dates of the published elections.
    errors : dict, optional
        The ``poll-errors`` part of the scope when the caller has already read it; read here when not given.

    Returns
    -------
    list of dict or None
        The rows whose ``event_date`` is in ``dates``, in their order, with ``error_blocks_abs`` (``with_abs``);
        ``None`` when the analysis of the scope is not published (no ``meta`` part).
    """
    if analysis_part(scope, 'meta') is None:
        return None

    errors = errors if errors is not None else analysis_part(scope, 'poll-errors') or {}
    wanted = set(dates)

    return with_abs([row for row in errors.get('rows') or [] if row.get('event_date') in wanted])


def history_scopes() -> list:
    """
    Scopes whose past elections are published, in catalogue order.

    Returns
    -------
    list of dict
        ``{'code', 'name', 'analysis', 'linkable'}`` for each catalogue scope with at least one published
        election; ``analysis`` is whether its houses analysis is published (``meta`` part) and ``linkable``
        whether the scope has a published run (the detail of its elections exists). A scope with analysis but
        without elections does not appear.
    """
    linkable = {row['code'] for row in published_scopes()}

    return [{'code': row['code'], 'name': row['name'], 'analysis': analysis_part(row['code'], 'meta') is not None,
             'linkable': row['code'] in linkable}
            for row in site().scopes()['scopes'] if site().events(row['code'])]


def history_events(scopes: list) -> list:
    """
    Published elections of several scopes in one list.

    Parameters
    ----------
    scopes : list of dict
        ``history_scopes`` rows.

    Returns
    -------
    list of dict
        ``published_events`` of each scope with ``scope``, ``scope_name`` and ``href`` added (``None`` when the scope is
        not ``linkable``: its detail would be a 404), newest first and, on the same date, in catalogue order.
    """
    events = []

    for scope in scopes:
        for event in published_events(scope['code']):
            href = f"/elecciones?scope={scope['code']}&date={event['date']}" if scope.get('linkable') else None
            event.update(scope=scope['code'], scope_name=scope['name'], href=href)
            events.append(event)

    order = {scope['code']: index for index, scope in enumerate(scopes)}
    events.sort(key=lambda event: order[event['scope']])
    events.sort(key=lambda event: event['date'], reverse=True)

    return events


def history_rows(scopes: list, events: list, parts: Optional[dict] = None) -> list:
    """
    Error rows of the published elections of every scope with analysis.

    Parameters
    ----------
    scopes : list of dict
        ``history_scopes`` rows; only those with ``analysis`` are read.
    events : list of dict
        ``history_events`` rows.
    parts : dict, optional
        ``{code: poll-errors part}`` as ``poll_error_parts`` returns; read here when not given.

    Returns
    -------
    list of dict
        ``error_rows`` of each scope with ``scope``, ``scope_name`` and ``year`` (int) added, reduced to
        ``HISTORY_COLUMNS`` plus the block columns of the scope.
    """
    rows = []
    parts = poll_error_parts(scopes) if parts is None else parts

    for scope in scopes:
        if not scope['analysis']:
            continue

        code = scope['code']
        part = parts.get(code) or {}
        blocks = list(part.get('blocks') or [])

        for row in error_rows(code, [event['date'] for event in events if event['scope'] == code], part) or []:
            row = dict(row, scope=code, scope_name=scope['name'], year=int(row['event_date'][:4]))
            rows.append({key: row.get(key) for key in HISTORY_COLUMNS + tuple(blocks)})

    return rows


def poll_error_parts(scopes: list) -> dict:
    """
    The ``poll-errors`` part of every scope with analysis, each read once.

    Parameters
    ----------
    scopes : list of dict
        ``history_scopes`` rows.

    Returns
    -------
    dict
        ``{code: part}`` for the scopes with analysis; an unpublished part is ``{}``.
    """
    return {scope['code']: analysis_part(scope['code'], 'poll-errors') or {} for scope in scopes if scope['analysis']}


def error_blocks(scopes: list, parts: dict) -> list:
    """
    Blocks of the bullseye axes: the pair of the first analysed scope that has at least two.

    Parameters
    ----------
    scopes : list of dict
        ``history_scopes`` rows.
    parts : dict
        ``poll_error_parts`` result.

    Returns
    -------
    list of str
        The blocks of the first analysed scope with two or more (``es`` first if analysed, then catalogue
        order); the bullseye leaves out the rows of the other scopes that lack those columns. Empty when no
        scope has two blocks.
    """
    ordered = sorted((scope for scope in scopes if scope['analysis']), key=lambda scope: scope['code'] != 'es')

    for scope in ordered:
        blocks = list((parts.get(scope['code']) or {}).get('blocks') or [])

        if len(blocks) >= 2:
            return blocks

    return []


def elecciones_context(state: dict, active: str = '/elecciones') -> dict:
    """
    Template context of the list of past elections, of every scope.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``); its raw ``scope`` (before ``common_context`` resolves
        it) only preselects the scope filter.
    active : str, optional
        ``href`` of the page in the navigation.

    Returns
    -------
    dict
        The common context with ``scopes`` replaced by ``history_scopes``, plus ``events``
        (``history_events``), ``errors`` (``history_rows``), ``error_blocks`` (``error_blocks``),
        ``filter_events`` (``key`` ``scope:date``, ``date``, ``name``, ``scope`` and ``scope_name``),
        ``filter_pollsters`` (house names, alphabetical), ``years`` (ascending), ``analysis_missing`` (no scope
        has analysis), ``selected_scope`` (the raw ``scope`` when it has analysis, else ``None``),
        only with events, ``initial`` (``state``, ``scopes``,
        ``events``: ``filter_events``, ``errors``, ``error_blocks``, ``years`` and ``selected_scope``).

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope or a missing run.
    """
    raw_scope = state['scope']
    context = common_context(state, active)
    scopes = history_scopes()
    events = history_events(scopes)
    parts = poll_error_parts(scopes)
    errors = history_rows(scopes, events, parts)
    with_analysis = {scope['code'] for scope in scopes if scope['analysis']}
    filter_events = [{'key': f"{event['scope']}:{event['date']}", 'date': event['date'], 'name': event['name'],
                      'scope': event['scope'], 'scope_name': event['scope_name']} for event in events]

    context['scopes'] = scopes
    context['events'] = events
    context['analysis_missing'] = not with_analysis
    context['errors'] = errors
    context['error_blocks'] = error_blocks(scopes, parts)
    context['filter_events'] = filter_events
    context['filter_pollsters'] = sorted({row['pollster'] for row in errors if row.get('pollster')}, key=str.casefold)
    context['years'] = sorted({row['year'] for row in errors})
    context['selected_scope'] = raw_scope if raw_scope in with_analysis else None

    if events:
        context['initial'] = {'state': context['state'], 'scopes': scopes, 'events': filter_events,
                              'errors': errors, 'error_blocks': context['error_blocks'], 'years': context['years'],
                              'selected_scope': context['selected_scope']}

    return context


def resolve_event(scope: str, date: str) -> str:
    """
    Validate an election date and check that it is published.

    Parameters
    ----------
    scope : str
        Scope code.
    date : str
        Raw ``date`` parameter.

    Returns
    -------
    str
        The validated date.

    Raises
    ------
    HttpError
        400 ``invalid date``; 404 ``event not published``.
    """
    date = check_date(date)

    if date not in site().events(scope):
        raise HttpError(404, 'event not published')

    return date


def shown_parties(params: dict) -> Optional[list]:
    """
    Parties the results table of an election shows before "Ver más", from its parameters.

    Parameters
    ----------
    params : dict
        Parameters of the election (``webapi.event_params``).

    Returns
    -------
    list of str or None
        ``parties.event`` of the parameters, or ``None`` when the election has no such list.
    """
    parties = (params.get('parties') or {}).get('event') if isinstance(params, dict) else None

    return list(parties) if isinstance(parties, (list, tuple)) else None


def mark_shown(rows: list, parties: Optional[list], limit: int = RESULTS_SHOWN) -> list:
    """
    Copy results rows marking the ones the table shows before "Ver más".

    Parameters
    ----------
    rows : list of dict
        Results rows with ``party``.
    parties : list of str or None
        Parties to show (``shown_parties``), or ``None`` to show the first rows.
    limit : int, optional
        Rows shown without a list, or when no row is in it.

    Returns
    -------
    list of dict
        Each row with ``shown``: whether its party is in ``parties``; without a list, or when none of the rows
        is in it, whether the row is among the first ``limit``.
    """
    wanted = set(parties) if parties is not None else set()

    if any(row.get('party') in wanted for row in rows):
        return [dict(row, shown=row.get('party') in wanted) for row in rows]

    return [dict(row, shown=index < limit) for index, row in enumerate(rows)]


def party_blocks(names: list, params: dict, event_blocks: Optional[dict], catalog: Catalog) -> dict:
    """
    Block of each party of a past election.

    Parameters
    ----------
    names : list of str
        Parties of the results.
    params : dict
        Parameters of the election (``webapi.event_params``); ``bmaps.blocks`` lists the parties of each block.
    event_blocks : dict or None
        ``blocks`` of the event meta: the parties of each of the two main blocks.
    catalog : Catalog
        Party catalogue of the run, whose parties have a block.

    Returns
    -------
    dict
        ``{name: block}`` by the first source that lists the party: the parameters, the event meta, the run
        catalogue; ``None`` when none does.
    """
    maps = []

    for source in ((params.get('bmaps') or {}).get('blocks') if isinstance(params, dict) else None, event_blocks):
        if isinstance(source, dict):
            maps.append({party: block for block, parties in source.items() for party in (parties or [])})

    blocks = {}

    for name in names:
        block = next((mapping[name] for mapping in maps if name in mapping), None)
        blocks[name] = block if block is not None else catalog[name].get('block')

    return blocks


def event_block_rows(rows: list, blocks: dict, color: Callable) -> list:
    """
    Rows of the results table by block of a past election.

    Parameters
    ----------
    rows : list of dict
        Results rows with ``party``, ``pct`` and ``seats``.
    blocks : dict
        Block of each party (``party_blocks``); a party without one counts for no block.
    color : callable
        ``block -> colour``.

    Returns
    -------
    list of dict
        ``{'name', 'pct', 'seats', 'color'}`` per block with some party: the sums of their ``pct`` (2 decimals)
        and ``seats`` (``None`` when no party has the value); the blocks of ``BLOCK_ORDER`` first, in that
        order, then the others by first appearance.
    """
    totals = {}

    for row in rows:
        block = blocks.get(row.get('party'))

        if block is None:
            continue

        total = totals.setdefault(block, {'pct': None, 'seats': None})

        for key in ('pct', 'seats'):
            value = row.get(key)

            if value is not None:
                total[key] = value if total[key] is None else total[key] + value

    order = [block for block in BLOCK_ORDER if block in totals] + [block for block in totals if block not in BLOCK_ORDER]

    return [{'name': block, 'pct': None if totals[block]['pct'] is None else round(totals[block]['pct'], 2),
             'seats': totals[block]['seats'], 'color': color(block)} for block in order]


def majority_of(n_seats) -> Optional[int]:
    """
    Absolute majority of a chamber.

    Parameters
    ----------
    n_seats : int or None
        Seats of the chamber.

    Returns
    -------
    int or None
        ``n_seats // 2 + 1``, or ``None`` when ``n_seats`` is not a positive integer.
    """
    if isinstance(n_seats, bool) or not isinstance(n_seats, int) or n_seats <= 0:
        return None

    return n_seats // 2 + 1


def split_sides(blocks: list) -> Optional[dict]:
    """
    Sides of the chart that splits the block error of the last poll of each house.

    Parameters
    ----------
    blocks : list of str
        Blocks of the election, as published.

    Returns
    -------
    dict or None
        ``{'left': block, 'right': block}`` with the first two blocks: ``LEFT_BLOCK`` goes left when it is the
        first one, otherwise the second block goes left and the first right (the notebook's deviation chart);
        ``None`` with fewer than two blocks.
    """
    if len(blocks) < 2:
        return None

    first, second = blocks[0], blocks[1]

    if first == LEFT_BLOCK:
        return {'left': first, 'right': second}

    return {'left': second, 'right': first}


def chart_parties(catalog: Catalog, *groups: list) -> list:
    """
    Parties of the browser's catalogue of the detail of an election.

    Parameters
    ----------
    catalog : Catalog
        Party catalogue of the run.
    *groups : list of str
        Lists of party names (the parties of the polls table, those of the results).

    Returns
    -------
    list of str
        Every name of the groups once: those of the run catalogue first, in its order, then the rest by
        appearance.
    """
    wanted = []

    for group in groups:
        for name in group:
            if name not in wanted:
                wanted.append(name)

    known = [name for name in catalog if name in wanted]

    return known + [name for name in wanted if name not in known]


def horizon_tables(model: dict) -> list:
    """
    Tables of the backtest of an election, one per horizon.

    Parameters
    ----------
    model : dict
        Data of the ``model`` part (``horizons``, ``shares``, ``seats``, ``blocks``).

    Returns
    -------
    list of dict
        By horizon ascending: ``{'horizon', 'seats_valid', 'shares', 'seats', 'blocks'}``. ``seats_valid``
        is ``False`` when any seats or blocks row of the horizon has it ``False`` (a case whose seats
        could not be projected). ``shares`` and ``seats``
        have the ``main`` parties (``name``, ``official``, ``model``; ``shares`` also
        ``error = model - official``, rounded to 2 decimals); ``blocks`` the ``block``,
        ``official_seats``, ``model_seats``, ``p_majority`` and ``majority`` of each block.
    """
    tables = []

    for horizon in sorted(model.get('horizons') or []):
        shares = [
            {'name': row['party'], 'official': row['official'], 'model': row['model'],
             'error': None if row['official'] is None or row['model'] is None
             else round(row['model'] - row['official'], 2)}
            for row in model.get('shares') or [] if row['horizon'] == horizon and row.get('main')
        ]
        seats = [
            {'name': row['party'], 'official': row['official'], 'model': row['model']}
            for row in model.get('seats') or [] if row['horizon'] == horizon and row.get('main')
        ]
        blocks = [
            {key: row.get(key) for key in ('block', 'official_seats', 'model_seats', 'p_majority', 'majority')}
            for row in model.get('blocks') or [] if row['horizon'] == horizon
        ]
        seats_valid = all(row.get('seats_valid') is not False
                          for part in ('seats', 'blocks') for row in model.get(part) or []
                          if row['horizon'] == horizon)
        tables.append({'horizon': horizon, 'seats_valid': seats_valid, 'shares': shares, 'seats': seats,
                       'blocks': blocks})

    return tables


def eleccion_context(state: dict, date: str, active: str = '/elecciones') -> dict:
    """
    Template context of the detail of a past election.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    date : str
        Raw ``date`` parameter, validated by ``resolve_event``.
    active : str, optional
        ``href`` of the page in the navigation.

    Returns
    -------
    dict
        The common context plus ``event`` (its ``meta``), ``results_rows`` (as published, with the
        ``fullname`` and ``color`` of the run catalogue, or the fallback of an unknown party, and ``shown``:
        ``mark_shown`` with the ``parties.event`` of the election in ``data/params.json``, else the first
        ``RESULTS_SHOWN``), ``results_hidden`` (the rows not shown, behind "Ver más"), ``block_result_rows``
        (``event_block_rows``: the results by block, ``party_blocks`` from the parameters, the event meta and the
        run catalogue, coloured by ``block_color``; empty when no party has a block), ``n_seats``,
        ``last_rows`` (``last_poll_rows``: the last poll of each pollster, from the smallest block error),
        ``poll_rows``, ``blocks``, ``split_blocks`` (``split_sides``: the block of each side of the chart that
        splits the error of the last polls, ``None`` without two blocks), ``horizon_tables``, ``has_model``,
        ``csv_links`` (the model ones only with a backtest), the keys of the "Todas las encuestas" table of
        Promedios drawn from the ``cycle`` part (``has_cycle``, ``table``, ``table_parties``, ``table_total``,
        ``has_seats``, ``months``, ``pollsters`` and ``csv_href``; ``has_cycle`` is false for an election
        published before the part existed, and the table then gives way to a notice) and ``initial`` (``state``,
        ``meta``, ``results`` (``n_seats``, ``majority`` (``majority_of``), ``rows`` (``name`` and ``seats``
        of each party, as published) and ``blocks`` (``name`` and ``seats`` of each block) for the hemicycle),
        ``last`` (``LAST_KEYS`` and the signed error of each block of every last poll, for the bars, the split
        by block and the bullseye), ``blocks``, ``split`` (``split_blocks``), ``block_colors`` (``{block:
        colour}`` of the error blocks and of the result blocks, from the run's ``bmaps``, ``block_color``),
        ``poll_errors`` (``error_avg`` and ``error_blocks_abs`` of each poll, for the error histogram),
        ``catalog`` (``parties`` of the table and of the results, ``chart_parties``, for the browser's catalogue)
        and, with the cycle, ``polls`` (``chart_polls``)).

    Raises
    ------
    HttpError
        400 for an invalid date; 404 for an unpublished election, scope or missing run; 503 without a bundle.
    """
    context = common_context(state, active)
    full = context['state']
    scope = full['scope']
    date = resolve_event(scope, date)
    meta = event_part(scope, date, 'meta')

    if meta is None:
        raise HttpError(404, 'event not published')

    polls = event_part(scope, date, 'polls') or {}
    cycle = event_part(scope, date, 'cycle')
    results = event_part(scope, date, 'results') or {}
    model = event_part(scope, date, 'model') or {}
    has_model = bool(meta.get('has_model'))
    blocks = list(polls.get('blocks') or meta.get('blocks') or [])
    poll_rows = with_abs(polls.get('rows') or [])
    last_rows = last_poll_rows(poll_rows)
    catalog = context['catalog']
    params = event_params(scope, date)
    results_rows = mark_shown([
        {'party': row['party'], 'fullname': catalog[row['party']]['fullname'],
         'color': catalog[row['party']]['color'], 'pct': row.get('pct'), 'seats': row.get('seats')}
        for row in results.get('rows') or []
    ], shown_parties(params))
    blocks_of = party_blocks([row['party'] for row in results_rows], params, meta.get('blocks'), catalog)
    block_rows = event_block_rows(results_rows, blocks_of, lambda block: block_color(block, context['meta'], catalog))
    links = [
        {'label': label, 'href': api_url(scope, name, fmt='csv', kind='event', date=date, table=table)}
        for name, table, label, needs_model in EVENT_CSV if has_model or not needs_model
    ]

    summary = dict(meta.get('summary') or {})
    error = summary.get('error_blocks')
    summary['error_blocks_abs'] = None if error is None else abs(error)
    cycle_polls = (cycle or {}).get('polls') or []
    table_parties = list((cycle or {}).get('parties') or [])
    colors = {name: catalog[name]['color'] for name in table_parties}
    n_seats = results.get('n_seats')
    split = split_sides(blocks)
    parties = chart_parties(catalog, table_parties, [row['party'] for row in results_rows])
    context['event'] = dict(meta, summary=summary)
    context['results_rows'] = results_rows
    context['results_hidden'] = sum(1 for row in results_rows if not row['shown'])
    context['pct_max'] = max((row['pct'] for row in results_rows if row.get('pct') is not None), default=0)
    context['majority'] = majority_of(n_seats)
    context['lead'] = lead_strip([{'name': row['party'], 'seats': row.get('seats')} for row in results_rows],
                                 majority_of(n_seats), with_first=False)
    context['block_result_rows'] = block_rows
    context['n_seats'] = n_seats
    context['last_rows'] = last_rows
    context['poll_rows'] = poll_rows
    context['blocks'] = blocks
    context['split_blocks'] = split
    context['horizon_tables'] = horizon_tables(model) if has_model else []
    context['has_model'] = has_model
    context['csv_links'] = links
    context['has_cycle'] = cycle is not None
    context['table_parties'] = table_parties
    context['table'] = polls_table(cycle_polls, table_parties, colors)
    context['table_total'] = len(cycle_polls)
    context['has_seats'] = any(poll.get('seats') for poll in cycle_polls)
    context['months'] = poll_months(cycle_polls)
    context['pollsters'] = pollster_options(cycle_polls)
    context['csv_href'] = api_url(scope, 'cycle', fmt='csv', kind='event', date=date)
    context['initial'] = {
        'state': full,
        'meta': meta,
        'results': {'n_seats': n_seats, 'majority': majority_of(n_seats),
                    'rows': [{'name': row['party'], 'seats': row['seats']} for row in results_rows],
                    'blocks': [{'name': row['name'], 'seats': row['seats']} for row in block_rows]},
        'last': [dict({key: row.get(key) for key in LAST_KEYS}, **{block: row.get(block) for block in blocks})
                 for row in last_rows],
        'blocks': blocks,
        'split': split,
        'block_colors': {block: block_color(block, context['meta'], catalog) for block in blocks}
        | {row['name']: row['color'] for row in block_rows},
        'poll_errors': [{key: row.get(key) for key in ('error_avg', 'error_blocks_abs')} for row in poll_rows],
        'catalog': {'parties': [dict(catalog[name], name=name) for name in parties]},
    }

    if cycle is not None:
        context['initial']['polls'] = chart_polls(cycle)

    return context
