"""
Context of the backtest subpage of the method (``/metodo/backtest``): at its head, the model scored like a pollster
in every published election with a backtest, of any scope (its estimate at the shortest horizon, its «last poll»,
measured with the definitions of the pollsters and placed among their last polls); then the published backtest of
the scope by horizon, by election and by block.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from statistics import median
from typing import Optional

from ..webapi import site
from .casas import PART_READERS, block_bar_colors, effects_table, election_label
from .common import api_url, bundle_part, common_context
from .elecciones import (error_blocks, event_part, history_events, history_scopes, horizon_tables, last_poll_rows,
                         poll_error_parts, table_blocks, with_abs, with_model_row, year_range)
from .filters import _madrid_tz, fmt_num

# The KPIs of the head card, in order
MODEL_KPIS = ('Elecciones', 'Error medio', 'Error de bloques', 'Frente a las encuestadoras')


def better_than(rows: list) -> Optional[float]:
    """
    Share of the pollsters of an election whose last poll missed the margin between the blocks by more than the
    model (a tie counts half).

    Parameters
    ----------
    rows : list of dict
        ``with_model_row`` of the election: the last poll of each pollster plus the model row (``model`` true),
        with ``error_blocks_abs``.

    Returns
    -------
    float or None
        From 0 (every pollster did better) to 1 (the model did better than all of them); ``None`` without the
        error of the model or without pollsters with one.
    """
    model = next((row for row in rows if row.get('model')), None)
    errors = [row['error_blocks_abs'] for row in rows if not row.get('model') and row.get('error_blocks_abs') is not None]

    if model is None or model.get('error_blocks_abs') is None or not errors:
        return None

    ours = model['error_blocks_abs']

    return round(sum(1.0 if error > ours else 0.5 if error == ours else 0.0 for error in errors) / len(errors), 3)


def model_kpis(rows: list) -> list:
    """
    The KPIs of the model as a pollster (``MODEL_KPIS``), for ``_macros.kpis``.

    Parameters
    ----------
    rows : list of dict
        ``model_record`` rows.

    Returns
    -------
    list of dict
        ``{'label', 'value', 'unit', 'support', 'ok', 'color'}``: the elections, the mean of the error of the main
        parties, the mean of the absolute error of the margin and the median, over the elections, of the share of
        pollsters the model did better than.
    """
    def mean(values):
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None

    shares = [row['better_than'] for row in rows if row.get('better_than') is not None]
    better = median(shares) * 100 if shares else None

    return [
        {'label': MODEL_KPIS[0], 'value': str(len(rows)), 'unit': '', 'support': 'con backtest, de todos los ámbitos',
         'ok': False, 'color': None},
        {'label': MODEL_KPIS[1], 'value': fmt_num(mean([row['error_avg'] for row in rows]), 1), 'unit': '',
         'support': 'puntos, partidos principales', 'ok': False, 'color': None},
        {'label': MODEL_KPIS[2], 'value': fmt_num(mean([row['error_blocks_abs'] for row in rows]), 1), 'unit': '',
         'support': 'puntos de margen, sin signo', 'ok': False, 'color': None},
        {'label': MODEL_KPIS[3], 'value': fmt_num(better, 0), 'unit': '%',
         'support': 'de encuestadoras por detrás (mediana)', 'ok': False, 'color': None},
    ]


def model_record(today=None) -> dict:
    """
    The model scored like a pollster in every published election with a backtest.

    Parameters
    ----------
    today : datetime.date, optional
        The end of the years of the bullseye filter; today in Madrid by default.

    Returns
    -------
    dict
        ``rows`` (one per election, newest first: ``scope``, ``event_date``, ``year``, ``label`` and ``name`` (the
        name of the election, ``election_label``), ``event`` (its month, ``YYYY-MM``: the short label of its point
        in the bullseye, with the scope), ``days`` (the horizon),
        ``error_avg``, ``error_blocks`` (signed, first block of the margin of the election minus the second),
        ``error_blocks_abs``, the error of each block, ``better_than`` and ``n_pollsters`` (the pollsters with a
        last poll) and ``parties`` (``{'party', 'result'}``: model minus result of each main party)), ``kpis``
        (``model_kpis``), ``blocks`` (the two blocks of the axes of the bullseye, ``error_blocks``), ``scopes``
        (``{code, name}`` of the scopes with a row, in catalogue order) and ``years`` (``year_range`` of the rows).
    """
    scopes = history_scopes()
    events = [event for event in history_events(scopes) if event.get('has_model')]
    names = {(event['scope'], event['date']): event['name'] for event in events if event.get('name')}
    scope_names = {row['code']: row['name'] for row in site().scopes()['scopes']}
    wanted = [(event['scope'], event['date'], part) for event in events for part in ('model', 'polls')]

    # each part is one read of the bundle (cached under the TTL afterwards)
    with ThreadPoolExecutor(max_workers=PART_READERS) as pool:
        parts = dict(zip(wanted, pool.map(lambda key: event_part(*key), wanted)))

    rows = []

    for event in events:
        scope, date = event['scope'], event['date']
        model = parts[(scope, date, 'model')] or {}
        polls = parts[(scope, date, 'polls')] or {}
        blocks = list(polls.get('blocks') or [])
        tables = horizon_tables(model, blocks)

        if not tables:
            continue

        table = tables[0]
        last = with_model_row(last_poll_rows(with_abs(polls.get('rows') or [])), table, blocks)
        label = election_label(scope, date, names, scope_names)
        errors = {row['block']: row.get('error') for row in table.get('blocks') or []}
        error = table.get('error_blocks')
        rows.append(dict(
            {'scope': scope, 'event_date': date, 'year': int(date[:4]), 'label': label, 'name': label, 'event': date[:7],
             'days': table['horizon'], 'error_avg': table.get('error_avg'), 'error_blocks': error,
             'error_blocks_abs': None if error is None else abs(error), 'better_than': better_than(last),
             'n_pollsters': sum(1 for row in last if not row.get('model')),
             'parties': [{'event_scope': scope, 'event_date': date, 'party': row['name'], 'result': row['error']}
                         for row in table['rows']]},
            **errors))

    rows.sort(key=lambda row: row['event_date'], reverse=True)
    present = {row['scope'] for row in rows}
    day = today or datetime.now(_madrid_tz()).date()

    return {
        'rows': rows,
        'kpis': model_kpis(rows),
        'blocks': error_blocks(scopes, poll_error_parts(scopes)),
        'scopes': [{'code': scope['code'], 'name': scope['name']} for scope in scopes if scope['code'] in present],
        'years': year_range({row['year'] for row in rows}, day),
    }


def model_context(context: dict) -> dict:
    """
    Add the model scored like a pollster to the context of the backtest subpage.

    Parameters
    ----------
    context : dict
        The context of ``backtest_context``, with ``meta`` and ``catalog`` (for the colours of the margin bars).

    Returns
    -------
    dict
        ``context`` plus ``model_record`` (``model_record``; without rows, the cards are not drawn), ``model_last``
        (``effects_table`` of the ``parties`` of every row, by the order of the catalogue), ``model_table_blocks``
        (``table_blocks`` of the axes: the columns of the bullseye table, the left block first) and
        ``model_block_colors`` (``block_bar_colors``: the colour of each side of the bar of «Sesgo bloques»).
    """
    record = model_record()
    labels = {(row['scope'], row['event_date']): row['label'] for row in record['rows']}
    context['model_record'] = record
    context['model_last'] = effects_table([party for row in record['rows'] for party in row['parties']],
                                          lambda scope, date: labels.get((scope, date), date),
                                          order=list(context['catalog']), key='result')
    context['model_table_blocks'] = table_blocks(record['blocks'])
    context['model_block_colors'] = block_bar_colors(context['meta'], context['catalog'])[1]

    return context


HORIZON_COLUMNS = (
    'horizon', 'n_cases', 'n_seats_cases', 'mae_shares', 'mae_last_poll', 'mae_mean_4w', 'mae_prev_result',
    'cov_shares50', 'cov_shares80', 'cov_shares95', 'mae_seats', 'cov_seats95', 'crps_seats', 'brier_vs',
    'mae_seats_h', 'cov_shares95_h', 'brier_vs_h',
)
CASE_COLUMNS = ('event_date', 'horizon', 'mae_shares', 'mae_last_poll', 'mae_seats', 'cov_shares95', 'brier_vs',
                'seats_valid')
BLOCK_COLUMNS = ('event_date', 'horizon', 'block', 'official_seats', 'model_seats', 'p_majority', 'majority',
                 'seats_valid')
CSV_LABELS = (
    ('metrics', 'Métricas por elección y horizonte'),
    ('by-horizon', 'Métricas por horizonte'),
    ('blocks', 'Bloques por elección y horizonte'),
    ('shares', 'Voto por partido, elección y horizonte'),
    ('seats', 'Escaños por partido, elección y horizonte'),
)


def backtest_part(scope: str, name: str) -> Optional[dict]:
    """
    Data of a part of the published backtest of a scope.

    Parameters
    ----------
    scope : str
        Scope code.
    name : str
        Part name, one of ``bundle.BACKTEST_PARTS``.

    Returns
    -------
    dict or None
        The ``data`` member of the part envelope, or ``None`` when the backtest is not published.

    Raises
    ------
    HttpError
        Any error of the bundle other than the 404 of an unpublished part.
    """
    return bundle_part(site().backtest, scope, name)


def project(rows: list, columns: tuple) -> list:
    """
    Keep only some columns of a list of rows.

    Parameters
    ----------
    rows : list of dict
        Rows as published.
    columns : tuple of str
        Columns to keep.

    Returns
    -------
    list of dict
        One dict per row with exactly ``columns``; a missing value is ``None``.
    """
    return [{column: row.get(column) for column in columns} for row in rows or []]


def case_rows(rows: list) -> list:
    """
    Sort rows of cases (``event_date``, ``horizon``) for display.

    Parameters
    ----------
    rows : list of dict
        Rows with ``event_date`` and ``horizon``.

    Returns
    -------
    list of dict
        Newest election first and, within each, the shortest horizon first.
    """
    by_horizon = sorted(rows, key=lambda row: row['horizon'])

    return sorted(by_horizon, key=lambda row: row['event_date'], reverse=True)


def backtest_context(state: dict, active: str = '/metodo') -> dict:
    """
    Template context of the backtest subpage of the method.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    active : str, optional
        ``href`` of the page in the navigation.

    Returns
    -------
    dict
        The common context, with ``filter_action`` set to ``/metodo/backtest``, plus ``published`` and the model
        scored like a pollster in every election with a backtest, of every scope (``modelo.model_context``; with
        rows, ``initial`` also carries ``bullseye``, ``bullseye_blocks`` and ``bullseye_scopes``, for its bullseye).
        With a published backtest: ``backtest`` (its ``meta``),
        ``horizon_rows``, ``case_rows``, ``block_rows``, ``csv_links`` and ``initial`` (``state``,
        ``horizons`` and, aligned with them, the ``mae``, ``coverage`` and ``seats`` series; ``seats`` holds
        ``mae``, the mean absolute seat error, the same in both modes since the headline is the central
        allocation, so the horizon run's ``mae_seats_h`` stays in the table rows and the CSV only).

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope or a missing run.
    """
    context = common_context(state, active)
    context['filter_action'] = '/metodo/backtest'
    scope = context['state']['scope']
    meta = backtest_part(scope, 'meta')
    context['published'] = meta is not None
    model_context(context)
    record = context['model_record']
    bullseye = {'state': context['state'], 'bullseye': record['rows'], 'bullseye_blocks': record['blocks'],
                'bullseye_scopes': record['scopes']} if record['rows'] else {}

    if meta is None:
        if bullseye:
            context['initial'] = bullseye
        return context

    by_horizon = backtest_part(scope, 'by-horizon') or {}
    metrics = backtest_part(scope, 'metrics') or {}
    blocks = backtest_part(scope, 'blocks') or {}
    horizon_rows = sorted(project(by_horizon.get('rows'), HORIZON_COLUMNS), key=lambda row: row['horizon'])

    def series(column: str) -> list:
        return [row[column] for row in horizon_rows]

    context['backtest'] = meta
    context['horizon_rows'] = horizon_rows
    context['case_rows'] = case_rows(project(metrics.get('rows'), CASE_COLUMNS))
    context['block_rows'] = case_rows(project(blocks.get('rows'), BLOCK_COLUMNS))
    context['csv_links'] = [{'label': label, 'href': api_url(scope, name, fmt='csv', kind='backtest')}
                            for name, label in CSV_LABELS]
    context['initial'] = {
        'state': context['state'],
        'horizons': series('horizon'),
        'mae': {'model': series('mae_shares'), 'last_poll': series('mae_last_poll'),
                'mean_4w': series('mae_mean_4w'), 'prev_result': series('mae_prev_result')},
        'coverage': {'50': series('cov_shares50'), '80': series('cov_shares80'), '95': series('cov_shares95')},
        'seats': {'mae': series('mae_seats')},
        **bullseye,
    }

    return context
