"""
Past elections page context: the list of published elections and the detail of one.
"""
from typing import Optional

from ...core.api import HttpError
from .. import bundle
from ..webapi import check_date, site
from .common import api_url, common_context

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
    try:
        return bundle.loads(site().event(scope, date, name))['data']
    except HttpError as error:
        if error.status == 404:
            return None

        raise


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
        ``n_pollsters``, ``error_avg``, ``error_blocks``, ``bias_blocks``, ``bias`` (by block)
        and ``has_model``, all from its ``meta`` part. A date without ``meta`` is skipped.
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

    return events


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
        ``blocks`` (those of the newest election; empty without events) and, only with events,
        ``initial`` (``state``, ``sector``: the average error by date ascending, and ``blocks``).

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
        event['href'] = '/elecciones' + context['query'] + '&date=' + event['date']

    context['events'] = events
    context['blocks'] = blocks

    if events:
        sector = [{key: event[key] for key in ('date', 'error_avg', 'error_blocks', 'bias_blocks')}
                  for event in reversed(events)]
        context['initial'] = {'state': context['state'], 'sector': sector, 'blocks': blocks}

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
        By horizon ascending: ``{'horizon', 'shares', 'seats', 'blocks'}``. ``shares`` and ``seats``
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
        tables.append({'horizon': horizon, 'shares': shares, 'seats': seats, 'blocks': blocks})

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
        ``pollsters``, ``bullseye`` and ``blocks``).

    Raises
    ------
    HttpError
        400 for an invalid date; 404 for an unpublished election, scope or missing run; 503 without a bundle.
    """
    context = common_context(state, active)
    full = context['state']
    scope = full['scope']
    date = resolve_event(scope, date)
    meta = event_part(scope, date, 'meta') or {}
    pollsters = event_part(scope, date, 'pollsters') or {}
    polls = event_part(scope, date, 'polls') or {}
    results = event_part(scope, date, 'results') or {}
    model = event_part(scope, date, 'model') or {}
    has_model = bool(meta.get('has_model'))
    blocks = list(pollsters.get('blocks') or meta.get('blocks') or [])
    pollster_rows = pollsters.get('rows') or []
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

    context['event'] = meta
    context['results_rows'] = results_rows
    context['n_seats'] = results.get('n_seats')
    context['pollster_rows'] = pollster_rows
    context['poll_rows'] = polls.get('rows') or []
    context['blocks'] = blocks
    context['horizon_tables'] = horizon_tables(model) if has_model else []
    context['has_model'] = has_model
    context['csv_links'] = links
    context['initial'] = {
        'state': full,
        'meta': meta,
        'pollsters': [{key: row.get(key) for key in ('pollster', 'error_blocks', 'last_error_blocks')}
                      for row in pollster_rows],
        'bullseye': [dict({'pollster': row['pollster'], 'days': row.get('last_days')},
                          **{block: row.get('last_' + block) for block in blocks})
                     for row in pollster_rows],
        'blocks': blocks,
    }

    return context
