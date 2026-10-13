"""
Average page context.
"""
from typing import Optional

from .common import api_url, common_context, part
from .filters import DASH, fmt_date, fmt_int, fmt_num

TABLE_ROWS = 40


def chart_polls(polls: dict) -> dict:
    """
    Polls part reduced to the columns the chart draws.

    Parameters
    ----------
    polls : dict
        Data of the polls part (``parties``, ``columns``, ``polls``, ``results``).

    Returns
    -------
    dict
        ``{'parties': list, 'polls': list of dict}`` where each poll keeps ``date``,
        ``pollster`` and one value per party.
    """
    parties = list(polls.get('parties') or [])
    keys = ['date', 'pollster'] + parties

    return {
        'parties': parties,
        'polls': [{key: poll.get(key) for key in keys if key in poll} for poll in polls.get('polls') or []],
    }


def table_rows(polls: list, parties: list, n: int = TABLE_ROWS) -> list:
    """
    Formatted rows of the polls table, the most recent first.

    Parameters
    ----------
    polls : list of dict
        Polls of the polls part (``date``, ``pollster``, ``sponsor``, ``sample_size`` and one
        percentage per party), in publication order.
    parties : list of str
        Parties to show, in column order.
    n : int, optional
        Maximum number of rows.

    Returns
    -------
    list of list of str
        The last ``n`` polls by date (ties keep the later published first) as ``date``,
        pollster, sponsor, sample size and one percentage per party, all formatted.
    """
    latest = sorted(reversed(polls), key=lambda poll: poll.get('date') or '', reverse=True)[:n]

    return [
        [
            fmt_date(poll.get('date')),
            poll.get('pollster') or DASH,
            poll.get('sponsor') or DASH,
            fmt_int(poll.get('sample_size')),
        ] + [fmt_num(poll.get(name), 1) for name in parties]
        for poll in latest
    ]


def promedio_context(state: dict, active: Optional[str] = None, event: bool = False, active_suffix: str = '') -> dict:
    """
    Template context of the poll average page.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    active : str or None, optional
        ``href`` of the page in the navigation (``None``: the average is an event page).
    event : bool, optional
        Whether it is rendered as an event page (see ``common_context``).
    active_suffix : str, optional
        Suffix of the current tab of the event bar.

    Returns
    -------
    dict
        The common context plus ``table_parties`` (the main-bloc parties present in the
        series, or all of them), ``table_rows`` (``table_rows``), ``n_polls``, ``csv_href``,
        ``series_subtitle``, ``series_title`` and ``initial`` (``state``, ``meta``, ``series``, ``polls``,
        ``projection`` and ``vote``, the data the chart is drawn from).

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope, a missing run or a missing part.
    """
    context = common_context(state, active, event=event, active_suffix=active_suffix)
    full, meta = context['state'], context['meta']
    scope, mode, run = full['scope'], full['mode'], full['run']
    series = part(scope, run, 'series')
    polls = part(scope, run, 'polls')
    vote = part(scope, run, 'vote', mode=mode)
    names = series.get('parties') or []
    main = (meta.get('bmaps') or {}).get('main') or []

    if mode == 'nowcast':
        subtitle = 'a ' + fmt_date(meta.get('as_of'))
    else:
        subtitle = '{} → {}'.format(fmt_date(meta.get('as_of')), fmt_date(vote.get('when')))

    table_parties = [name for name in main if name in names] or list(names)
    context['table_parties'] = table_parties
    context['table_rows'] = table_rows(polls.get('polls') or [], table_parties)
    context['n_polls'] = meta.get('n_polls')
    context['csv_href'] = api_url(scope, 'polls', run=run, fmt='csv')
    context['series_subtitle'] = subtitle
    context['series_title'] = 'Promedio de sondeos' if mode == 'nowcast' else 'Promedio de sondeos y proyección'
    context['initial'] = {
        'state': full,
        'meta': meta,
        'series': series,
        'polls': chart_polls(polls),
        'projection': part(scope, run, 'projection', mode=mode),
        'vote': vote,
    }

    return context
