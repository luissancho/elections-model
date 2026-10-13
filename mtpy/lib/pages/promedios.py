"""
Context of the Promedios tab of an event (``/{scope}``): the poll average with its window, party and block views,
the headline table, vote and seats, majorities, the evolution of the publications, one house against the average
and the polls of the cycle.
"""
from collections import Counter
from typing import Optional

from .. import bundle
from ..webapi import site
from .common import Catalog, api_url, block_color, common_context, part
from .filters import DASH, _is_missing, fmt_date, fmt_int, fmt_num, fmt_pct, fmt_prob

TABLE_ROWS = 40
DEFINITIVE_PREFIX = 'Pronóstico definitivo: '


def majority_rows(p_majority: dict, meta: dict, parties: Catalog) -> list:
    """
    Rows of the absolute majority block, one per block in the given order.

    Parameters
    ----------
    p_majority : dict
        Probability of an absolute majority by block.
    meta : dict
        Data of the meta part.
    parties : Catalog
        Party catalogue of the run.

    Returns
    -------
    list of dict
        ``{'block', 'p', 'width', 'color'}``; ``width`` is the bar width in percent (0-100,
        one decimal).
    """
    rows = []

    for block, p in (p_majority or {}).items():
        share = 0.0 if _is_missing(p) else max(0.0, min(1.0, float(p)))
        rows.append({
            'block': block,
            'p': p,
            'width': round(share * 100, 1),
            'color': block_color(block, meta, parties),
        })

    return rows


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
        ``{'parties': list, 'polls': list of dict}`` where each poll keeps ``date``, ``pollster_id``,
        ``pollster`` and one value per party (the keys it has), plus ``name`` (``poll_name``) and
        ``sample_size`` (``None`` when missing), shown in the box of each poll.
    """
    parties = list(polls.get('parties') or [])
    keys = ['date', 'pollster_id', 'pollster'] + parties

    return {
        'parties': parties,
        'polls': [
            {key: poll.get(key) for key in keys if key in poll}
            | {'name': poll_name(poll), 'sample_size': poll.get('sample_size')}
            for poll in polls.get('polls') or []
        ],
    }


def table_rows(polls: list, parties: list, n: Optional[int] = TABLE_ROWS) -> list:
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
        Maximum number of rows; ``None`` for all of them.

    Returns
    -------
    list of list of str
        The last ``n`` polls by date (ties keep the later published first) as ``date``, poll name
        (``poll_name``), sample size and one percentage per party, all formatted.
    """
    latest = sorted(reversed(polls), key=lambda poll: poll.get('date') or '', reverse=True)[:n]

    return [
        [
            fmt_date(poll.get('date')),
            poll_name(poll) or DASH,
            fmt_int(poll.get('sample_size')),
        ] + [fmt_num(poll.get(name), 1) for name in parties]
        for poll in latest
    ]


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


def headline_text(current: dict, mode: str, meta: dict, definitive: bool) -> Optional[str]:
    """
    One-line headline of the page in plain words.

    Parameters
    ----------
    current : dict
        Headline of the mode shown (``parties`` with ``name``, ``pct``, ``seats``; ``p_majority``).
    mode : str
        ``'nowcast'`` or ``'forecast'``.
    meta : dict
        Data of the meta part (``event_date``).
    definitive : bool
        Whether the forecast shown is the final one (frozen publication, latest run).

    Returns
    -------
    str or None
        ``'Si hoy hubiera elecciones, {name} ganaría con el {pct} % del voto y {seats} escaños; probabilidad
        de mayoría absoluta de {block}: {p}.'`` in nowcast; in forecast it opens with ``'El {event_date}'``,
        after ``'Pronóstico definitivo: '`` (and in lower case) when ``definitive``. The leader is the party with
        the highest ``pct`` and the block that with the highest ``p_majority``; without probabilities the sentence
        ends after the seats. The party is named by its short name. ``None`` when the headline has no parties.
    """
    rows = current.get('parties') or []

    if not rows:
        return None

    leader = max(rows, key=lambda row: row.get('pct') or 0)
    sentence = '{} ganaría con el {} del voto y {} escaños'.format(
        leader['name'], fmt_pct(leader.get('pct')), fmt_int(leader.get('seats')))
    probs = {block: p for block, p in (current.get('p_majority') or {}).items() if not _is_missing(p)}

    if probs:
        block = max(probs, key=probs.get)
        sentence += '; probabilidad de mayoría absoluta de {}: {}'.format(block, fmt_prob(probs[block]))

    if mode == 'nowcast':
        return 'Si hoy hubiera elecciones, {}.'.format(sentence)

    text = 'El {}, {}.'.format(fmt_date(meta.get('event_date')), sentence)

    return DEFINITIVE_PREFIX + text[0].lower() + text[1:] if definitive else text


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
        ``common_context`` of an event page plus ``headline`` (the headline of the mode), ``majority``
        (``majority_rows``), ``headline_text`` (``headline_text``), ``table_parties`` (the ``main`` parties
        present in the series, or all of them), ``table_rows`` (``table_rows``, the first ``TABLE_ROWS``),
        ``table_total`` (the polls of the polls part), ``n_polls``, ``csv_href``,
        ``pollsters`` (``pollster_options``), ``blocks`` (the names of ``meta.bmaps.blocks``, empty without
        them), ``block_colors`` (``block_color`` of each block, for its checkbox), ``main`` (the parties of
        ``meta.bmaps.main`` present in the series) and ``initial`` (``state``, ``meta``, ``headline``,
        ``series`` (``round_series``), ``polls`` (``chart_polls``), ``projection``, ``vote``, ``summary`` and
        ``runs``, the data the charts are drawn from; plus ``table``, ``{'parties', 'rows'}`` with every row of
        the polls table, only when ``table_total > TABLE_ROWS``, for the "Cargar más" button).

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
    names = series.get('parties') or []
    main = [name for name in bmaps.get('main') or [] if name in names]
    blocks = bmaps.get('blocks')
    definitive = mode == 'forecast' and context['frozen'] and run == site().latest_run(scope)

    context['headline'] = current
    context['majority'] = majority_rows(current.get('p_majority'), meta, parties)
    context['headline_text'] = headline_text(current, mode, meta, definitive)
    context['table_parties'] = main or list(names)
    context['table_rows'] = table_rows(poll_rows, context['table_parties'])
    context['table_total'] = len(poll_rows)
    context['n_polls'] = meta.get('n_polls')
    context['csv_href'] = api_url(scope, 'polls', run=run, fmt='csv')
    context['pollsters'] = pollster_options(poll_rows)
    context['blocks'] = list(blocks) if isinstance(blocks, dict) else []
    context['block_colors'] = {block: block_color(block, meta, parties) for block in context['blocks']}
    context['main'] = main
    context['initial'] = {
        'state': full,
        'meta': meta,
        'headline': headline,
        'series': round_series(series),
        'polls': chart_polls(polls),
        'projection': part(scope, run, 'projection', mode=mode),
        'vote': part(scope, run, 'vote', mode=mode),
        'summary': part(scope, run, 'summary', mode=mode),
        'runs': bundle.loads(site().history(scope))['data'],
    }

    if context['table_total'] > TABLE_ROWS:
        context['initial']['table'] = {
            'parties': context['table_parties'],
            'rows': table_rows(poll_rows, context['table_parties'], n=None),
        }

    return context
