"""
Model accuracy page context: the published backtest by horizon, by election and by block.
"""
from typing import Optional

from ...core.api import HttpError
from .. import bundle
from ..webapi import site
from .common import api_url, common_context

HORIZON_COLUMNS = (
    'horizon', 'n_cases', 'n_seats_cases', 'mae_shares', 'mae_last_poll', 'mae_mean_4w', 'mae_prev_result',
    'cov_shares50', 'cov_shares80', 'cov_shares95', 'mae_seats', 'cov_seats95', 'crps_seats', 'brier_vs',
    'mae_seats_h', 'cov_shares95_h', 'brier_vs_h',
)
CASE_COLUMNS = ('event_date', 'horizon', 'mae_shares', 'mae_last_poll', 'mae_seats', 'cov_shares95', 'brier_vs',
                'seats_valid')
BLOCK_COLUMNS = ('event_date', 'horizon', 'block', 'official_seats', 'model_seats', 'p_majority', 'majority')
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
    try:
        return bundle.loads(site().backtest(scope, name))['data']
    except HttpError as error:
        if error.status == 404:
            return None

        raise


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


def modelo_context(state: dict, active: str = '/modelo') -> dict:
    """
    Template context of the model accuracy page.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    active : str, optional
        ``href`` of the page in the navigation.

    Returns
    -------
    dict
        The common context plus ``published``. With a published backtest: ``backtest`` (its ``meta``),
        ``horizon_rows``, ``case_rows``, ``block_rows``, ``csv_links`` and ``initial`` (``state``,
        ``horizons`` and, aligned with them, ``mae``, ``coverage`` and ``seats`` series).

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope or a missing run.
    """
    context = common_context(state, active)
    scope = context['state']['scope']
    meta = backtest_part(scope, 'meta')
    context['published'] = meta is not None

    if meta is None:
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
        'seats': {'mae': series('mae_seats'), 'mae_h': series('mae_seats_h'), 'cov95': series('cov_seats95')},
    }

    return context
