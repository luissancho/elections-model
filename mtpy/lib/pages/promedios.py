"""
Context of the Promedios tab of an event (``/{scope}``): the poll average with its window, party and block views,
the hemicycle and the results by party and by block, the evolution of the publications, one house against the average
and the polls of the cycle.
"""
from collections import Counter
from typing import Optional

from .. import bundle
from ..webapi import site
from .common import OTHERS_COLOR, Catalog, api_url, block_color, common_context, part
from .filters import DASH, _is_missing, _parse_datetime, fmt_date, fmt_int, fmt_num

PAGE_SIZE = 15
MONTH_LABELS = ('Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic')
BLOCK_ORDER = ('Derecha', 'Izquierda', 'Regionalista', 'Separatista')


def result_rows(summary: dict) -> list:
    """
    Rows of the table of results by party.

    Parameters
    ----------
    summary : dict
        Data of the summary part (``parties``).

    Returns
    -------
    list of dict
        ``{'name', 'pct', 'seats', 'p_first'}`` per party of the summary, in its order and unformatted.
    """
    return [
        {key: row.get(key) for key in ('name', 'pct', 'seats', 'p_first')}
        for row in (summary or {}).get('parties') or []
    ]


def block_result_rows(summary: dict, meta: dict, parties: Catalog) -> list:
    """
    Rows of the table of results by block.

    Parameters
    ----------
    summary : dict
        Data of the summary part (``blocks``).
    meta : dict
        Data of the meta part.
    parties : Catalog
        Party catalogue of the run.

    Returns
    -------
    list of dict
        ``{'name', 'pct', 'seats', 'p_majority', 'color'}`` per block with parties (``seats`` not ``None``),
        unformatted: the blocks of ``BLOCK_ORDER`` first, in that order, and then the others in the order of
        the summary.
    """
    blocks = [row for row in (summary or {}).get('blocks') or [] if row.get('seats') is not None]
    blocks.sort(key=lambda row: BLOCK_ORDER.index(row['name']) if row['name'] in BLOCK_ORDER else len(BLOCK_ORDER))

    return [
        {key: row.get(key) for key in ('name', 'pct', 'seats', 'p_majority')}
        | {'color': block_color(row['name'], meta, parties)}
        for row in blocks
    ]


def poll_name(poll: dict) -> str:
    """
    Name of a poll as the loader builds it: the pollster, plus ``' / '`` and the sponsor when there is one.

    Parameters
    ----------
    poll : dict
        Poll record (``pollster``, ``sponsor``).

    Returns
    -------
    str
        ``'{pollster} / {sponsor}'``, or the pollster alone when the sponsor is missing or empty; an empty
        string without a pollster.
    """
    pollster = poll.get('pollster') or ''
    sponsor = poll.get('sponsor')

    return pollster + ' / ' + sponsor if sponsor else pollster


def chart_polls(polls: dict) -> dict:
    """
    Polls part reduced to the columns the charts draw.

    Parameters
    ----------
    polls : dict
        Data of the polls part (``parties``, ``columns``, ``polls``, ``results``).

    Returns
    -------
    dict
        ``{'parties': list, 'polls': list of dict}`` where each poll keeps ``date``, ``pollster_id``, ``seats``
        (the pollster's projections, when it has them),
        ``pollster``, ``sponsor``, ``start_date``, ``end_date`` and one value per party (the keys it has),
        plus ``name`` (``poll_name``) and ``sample_size`` (``None`` when missing): the box of each poll of the
        charts and the table of all the polls are drawn from them.
    """
    parties = list(polls.get('parties') or [])
    keys = ['date', 'pollster_id', 'pollster', 'sponsor', 'start_date', 'end_date', 'seats'] + parties

    return {
        'parties': parties,
        'polls': [
            {key: poll.get(key) for key in keys if key in poll}
            | {'name': poll_name(poll), 'sample_size': poll.get('sample_size')}
            for poll in polls.get('polls') or []
        ],
    }


def short_date(value) -> str:
    """
    Format a date as ``'01/10/26'``.

    Parameters
    ----------
    value : str or datetime.date or None
        ISO date or date-time.

    Returns
    -------
    str
        ``dd/mm/aa``, or ``DASH`` when it is missing or invalid.
    """
    parsed = _parse_datetime(value)

    return DASH if parsed is None else '{:02d}/{:02d}/{:02d}'.format(parsed.day, parsed.month, parsed.year % 100)


def poll_months(polls: list) -> list:
    """
    Months with polls, for the "Desde" and "Hasta" selectors of the polls table.

    Parameters
    ----------
    polls : list of dict
        Polls of the polls part (``date``); a poll without a valid date is not counted.

    Returns
    -------
    list of dict
        ``{'value': 'YYYY-MM', 'label': 'Ago 2023'}`` per month, ascending.
    """
    months = set()

    for poll in polls:
        parsed = _parse_datetime(poll.get('date'))

        if parsed is not None:
            months.add((parsed.year, parsed.month))

    return [
        {'value': '{:04d}-{:02d}'.format(year, month), 'label': '{} {}'.format(MONTH_LABELS[month - 1], year)}
        for year, month in sorted(months)
    ]


def _leaders(values: list) -> list:
    """
    Flags of the highest value.

    Parameters
    ----------
    values : list of float or None
        One value per party.

    Returns
    -------
    list of bool
        ``True`` where the value equals the maximum of the present ones (every tied value); all ``False``
        without values.
    """
    present = [value for value in values if not _is_missing(value)]

    return [not _is_missing(value) and value == max(present) for value in values]


def polls_table(polls: list, parties: list, colors: dict, n: Optional[int] = PAGE_SIZE) -> dict:
    """
    Formatted rows of the polls table, the most recent first, with the simple mean of all the polls.

    Parameters
    ----------
    polls : list of dict
        Polls of the polls part (``date``, ``pollster``, ``sponsor``, ``sample_size``, ``start_date``,
        ``end_date`` and one percentage per party), in publication order.
    parties : list of str
        Parties to show, in column order.
    colors : dict
        Colour by party name; a party without one gets ``OTHERS_COLOR``.
    n : int, optional
        Maximum number of rows; ``None`` for all of them. The mean always covers every poll.

    Returns
    -------
    dict
        ``{'rows', 'mean', 'total'}``: ``total`` is ``len(polls)``; each row of the last ``n`` polls by date
        (ties keep the later published first) is ``{'pollster', 'sponsor', 'date' (dd/mm/aa), 'fieldwork'
        (``None`` without fieldwork dates), 'sample', 'cells'}`` with a ``{'value', 'leader', 'color'}`` cell
        per party (``leader``: the highest of the row, every tied one); ``mean`` is ``{'n', 'cells'}``, the
        unweighted mean of all the polls (``n`` is ``total``) at one decimal in the same cells, the figure the
        browser shows before any filter.
    """
    latest = sorted(reversed(polls), key=lambda poll: poll.get('date') or '', reverse=True)[:n]
    rows = []

    for poll in latest:
        start, end = poll.get('start_date'), poll.get('end_date')
        fieldwork = None

        if start or end:
            fieldwork = 'Trabajo de campo: {} – {}'.format(fmt_date(start or end), fmt_date(end or start))

        values = [poll.get(name) for name in parties]
        rows.append({
            'pollster': poll.get('pollster') or DASH,
            'sponsor': poll.get('sponsor') or None,
            'date': short_date(poll.get('date')),
            'fieldwork': fieldwork,
            'sample': fmt_int(poll.get('sample_size')),
            'cells': [
                {'value': fmt_num(value, 1), 'leader': leader, 'color': colors.get(name, OTHERS_COLOR)}
                for name, value, leader in zip(parties, values, _leaders(values))
            ],
        })

    means = []

    for name in parties:
        present = [poll[name] for poll in polls if not _is_missing(poll.get(name))]
        means.append(round(sum(present) / len(present), 1) if present else None)

    return {
        'rows': rows,
        'mean': {
            'n': len(polls),
            'cells': [
                {'value': fmt_num(value, 1), 'leader': leader, 'color': colors.get(name, OTHERS_COLOR)}
                for name, value, leader in zip(parties, means, _leaders(means))
            ],
        },
        'total': len(polls),
    }


def pollster_options(polls: list) -> list:
    """
    Options of the house selector: the houses with polls in the cycle.

    Parameters
    ----------
    polls : list of dict
        Polls of the polls part (``pollster``); a poll without a house is not counted.

    Returns
    -------
    list of dict
        ``{'name', 'n'}`` per house, by number of polls descending and then by name.
    """
    counts = Counter(poll.get('pollster') for poll in polls if poll.get('pollster'))

    return [{'name': name, 'n': n} for name, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def round_series(series: dict, digits: int = 1) -> dict:
    """
    Series part with the ``mean``, ``lo`` and ``hi`` values rounded, to keep the page light.

    Parameters
    ----------
    series : dict
        Data of the series part (``dates``, ``parties``, ``mean``, ``lo``, ``hi``).
    digits : int, optional
        Decimals kept.

    Returns
    -------
    dict
        A copy of ``series`` with every number of ``mean``, ``lo`` and ``hi`` rounded to ``digits``
        decimals; missing values stay ``None`` and the other members are kept as they are.
    """
    def rounded(values):
        return [None if _is_missing(value) else round(value, digits) for value in values]

    out = dict(series)

    for key in ('mean', 'lo', 'hi'):
        out[key] = {name: rounded(values) for name, values in (series.get(key) or {}).items()}

    return out


def promedios_context(state: dict) -> dict:
    """
    Template context of the Promedios tab of an event.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``), the scope taken from the route.

    Returns
    -------
    dict
        ``common_context`` of an event page plus ``headline`` (the headline of the mode), ``result_rows``
        and ``block_result_rows`` (the tables of results of the summary), ``table_parties`` (every party of the
        polls part), ``table`` (``polls_table``, the ``PAGE_SIZE`` most recent polls the server renders),
        ``table_total`` (the polls of the polls part), ``months`` (``poll_months``), ``csv_href``,
        ``pollsters`` (``pollster_options``), ``blocks`` (the names of ``meta.bmaps.blocks``, empty without
        them) and ``initial`` (``state``, ``meta``, ``headline``,
        ``series`` (``round_series``), ``polls`` (``chart_polls``, which also feeds the whole table, filtered,
        sorted and paginated in the browser), ``projection``, ``vote``, ``summary`` and ``runs``).

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope, a missing run or a missing part.
    """
    context = common_context(state, None, event=True)
    full, meta, parties = context['state'], context['meta'], context['catalog']
    scope, mode, run = full['scope'], full['mode'], full['run']
    headline = part(scope, run, 'headline')
    current = headline[mode]
    series = part(scope, run, 'series')
    polls = part(scope, run, 'polls')
    poll_rows = polls.get('polls') or []
    bmaps = meta.get('bmaps') or {}
    blocks = bmaps.get('blocks')

    context['headline'] = current
    summary = part(scope, run, 'summary', mode=mode)
    context['result_rows'] = result_rows(summary)
    context['block_result_rows'] = block_result_rows(summary, meta, parties)
    context['table_parties'] = list(polls.get('parties') or [])
    colors = {name: parties[name]['color'] for name in context['table_parties']}
    context['table'] = polls_table(poll_rows, context['table_parties'], colors)
    context['table_total'] = len(poll_rows)
    context['has_seats'] = any(poll.get('seats') for poll in poll_rows)  # the % de voto / Escaños switch
    context['months'] = poll_months(poll_rows)
    context['csv_href'] = api_url(scope, 'polls', run=run, fmt='csv')
    context['pollsters'] = pollster_options(poll_rows)
    context['blocks'] = list(blocks) if isinstance(blocks, dict) else []
    context['initial'] = {
        'state': full,
        'meta': meta,
        'headline': headline,
        'series': round_series(series),
        'polls': chart_polls(polls),
        'projection': part(scope, run, 'projection', mode=mode),
        'vote': part(scope, run, 'vote', mode=mode),
        'summary': summary,
        'runs': bundle.loads(site().history(scope))['data'],
    }

    return context
