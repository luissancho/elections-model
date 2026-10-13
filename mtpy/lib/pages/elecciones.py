"""
Past elections page context: the list of published elections and the detail of one.
"""
from typing import Optional

from ...core.api import HttpError
from ..webapi import check_date, site
from .casas import analysis_part
from .common import api_url, bundle_part, common_context
from .filters import fmt_num

SUMMARY_KEYS = ('n_polls', 'n_pollsters', 'error_avg', 'error_blocks', 'bias_blocks', 'bias')
EVENT_CSV = (
    ('pollsters', None, 'Casas en esta elección', False),
    ('polls', None, 'Sondeos de esta elección', False),
    ('results', None, 'Resultados oficiales', False),
    ('model', 'shares', 'Modelo: voto a cada horizonte', True),
    ('model', 'seats', 'Modelo: escaños a cada horizonte', True),
    ('model', 'blocks', 'Modelo: bloques a cada horizonte', True),
)


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


def error_rows(scope: str, dates: list) -> Optional[list]:
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

    Returns
    -------
    list of dict or None
        The rows whose ``event_date`` is in ``dates``, in their order, with ``error_blocks_abs`` (``with_abs``);
        ``None`` when the analysis of the scope is not published (no ``meta`` part).
    """
    if analysis_part(scope, 'meta') is None:
        return None

    errors = analysis_part(scope, 'poll-errors') or {}
    wanted = set(dates)

    return with_abs([row for row in errors.get('rows') or [] if row.get('event_date') in wanted])


def history_headline(n_events: int, errors: Optional[list]) -> str:
    """
    Headline of the error history: the published elections and the mean block error of the last poll of each house.

    Parameters
    ----------
    n_events : int
        Number of published elections.
    errors : list of dict or None
        ``error_rows`` of those elections (``None`` without analysis).

    Returns
    -------
    str
        The sentence with the mean of ``error_blocks_abs``, or only the count when there is no analysis or no
        finite value to average.
    """
    values = [row['error_blocks_abs'] for row in errors or [] if row.get('error_blocks_abs') is not None]

    if not values:
        return f'{n_events} elecciones analizadas.'

    mean = fmt_num(sum(values) / len(values), 1)

    return (f'En las {n_events} elecciones analizadas, el último sondeo de cada casa se desvió de media {mean} '
            'puntos en el margen entre bloques.')


def elecciones_context(state: dict, active: str = '/elecciones') -> dict:
    """
    Template context of the list of past elections.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    active : str, optional
        ``href`` of the page in the navigation.

    Returns
    -------
    dict
        The common context plus ``events`` (``published_events``, each with its ``href``),
        ``blocks`` (those of the newest election; empty without events), ``analysis_missing``, ``errors``
        (``error_rows`` of the published elections, empty without analysis), ``error_blocks`` (the ``blocks`` of
        the ``poll-errors`` part), ``filter_events`` (``date`` and ``name`` of each election, newest first),
        ``filter_pollsters`` (the house names of ``errors``, alphabetical), ``headline_text``
        (``history_headline``) and, only with events, ``initial``
        (``state``, ``sector``: the average error by date ascending, ``blocks``, ``errors``, ``error_blocks`` and
        ``events``: ``filter_events``).

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope or a missing run.
    """
    context = common_context(state, active)
    scope = context['state']['scope']
    events = published_events(scope)
    blocks = []

    if events:
        meta = event_part(scope, events[0]['date'], 'meta') or {}
        blocks = list(meta.get('blocks') or [])

    for event in events:
        event['href'] = '/elecciones' + context['scope_query'] + '&date=' + event['date']

    errors = error_rows(scope, [event['date'] for event in events])
    analysis = analysis_part(scope, 'poll-errors') if errors is not None else None
    error_blocks = list((analysis or {}).get('blocks') or [])
    filter_events = [{'date': event['date'], 'name': event['name']} for event in events]

    context['events'] = events
    context['blocks'] = blocks
    context['analysis_missing'] = errors is None
    context['errors'] = errors or []
    context['error_blocks'] = error_blocks
    context['filter_events'] = filter_events
    context['filter_pollsters'] = sorted({row['pollster'] for row in errors or [] if row.get('pollster')},
                                         key=str.casefold)
    context['headline_text'] = history_headline(len(events), errors)

    if events:
        sector = [{key: event[key] for key in ('date', 'error_avg', 'error_blocks', 'error_blocks_abs', 'bias_blocks')}
                  for event in reversed(events)]
        context['initial'] = {'state': context['state'], 'sector': sector, 'blocks': blocks,
                              'errors': context['errors'], 'error_blocks': error_blocks, 'events': filter_events}

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
        ``fullname`` and ``color`` of the run catalogue, or the fallback of an unknown party),
        ``n_seats``, ``pollster_rows``, ``poll_rows``, ``blocks``, ``horizon_tables``, ``has_model``,
        ``csv_links`` (the model ones only with a backtest) and ``initial`` (``state``, ``meta``,
        ``pollsters``, ``bullseye``, ``blocks`` and ``poll_errors``: ``error_avg`` and ``error_blocks_abs`` of
        each poll, for the error histogram).

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

    pollsters = event_part(scope, date, 'pollsters') or {}
    polls = event_part(scope, date, 'polls') or {}
    results = event_part(scope, date, 'results') or {}
    model = event_part(scope, date, 'model') or {}
    has_model = bool(meta.get('has_model'))
    blocks = list(pollsters.get('blocks') or meta.get('blocks') or [])
    pollster_rows = sorted(with_abs(pollsters.get('rows') or []),
                           key=lambda row: (row['error_blocks_abs'] is None, row['error_blocks_abs'] or 0.0))
    catalog = context['catalog']
    results_rows = [
        {'party': row['party'], 'fullname': catalog[row['party']]['fullname'],
         'color': catalog[row['party']]['color'], 'pct': row.get('pct'), 'seats': row.get('seats')}
        for row in results.get('rows') or []
    ]
    links = [
        {'label': label, 'href': api_url(scope, name, fmt='csv', kind='event', date=date, table=table)}
        for name, table, label, needs_model in EVENT_CSV if has_model or not needs_model
    ]

    summary = dict(meta.get('summary') or {})
    error = summary.get('error_blocks')
    summary['error_blocks_abs'] = None if error is None else abs(error)
    context['event'] = dict(meta, summary=summary)
    context['results_rows'] = results_rows
    context['n_seats'] = results.get('n_seats')
    context['pollster_rows'] = pollster_rows
    context['poll_rows'] = with_abs(polls.get('rows') or [])
    context['blocks'] = blocks
    context['horizon_tables'] = horizon_tables(model) if has_model else []
    context['has_model'] = has_model
    context['csv_links'] = links
    context['initial'] = {
        'state': full,
        'meta': meta,
        'pollsters': [{key: row.get(key)
                       for key in ('pollster', 'error_blocks', 'error_blocks_abs', 'last_error_blocks')}
                      for row in pollster_rows],
        'bullseye': [dict({'pollster': row['pollster'], 'days': row.get('last_days')},
                          **{block: row.get('last_' + block) for block in blocks})
                     for row in pollster_rows],
        'blocks': blocks,
        'poll_errors': [{key: row.get(key) for key in ('error_avg', 'error_blocks_abs')}
                        for row in context['poll_rows']],
    }

    return context
