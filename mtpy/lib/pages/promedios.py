"""
Context of the Promedios tab of an event (``/{scope}``): the poll average with its window, party and block views,
the hemicycle and the results by party and by block, the evolution of the publications, one house against the average
and the polls of the cycle.
"""
from collections import Counter
from typing import Optional

from .. import bundle
from ..webapi import site
from .common import OTHERS_COLOR, Catalog, api_url, block_color, bundle_part, common_context, ink_color, part
from .filters import (DASH, _is_missing, _parse_datetime, delta_text, fmt_date, fmt_date_short, fmt_int, fmt_num, fmt_pct,
                      fmt_prob, fmt_range)

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
        ``{'name', 'pct', 'seats', 'seats_lo', 'seats_hi', 'p_first'}`` per party of the summary, in its order
        and unformatted.
    """
    return [
        {key: row.get(key) for key in ('name', 'pct', 'seats', 'seats_lo', 'seats_hi', 'p_first')}
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
        {key: row.get(key) for key in ('name', 'pct', 'seats', 'seats_lo', 'seats_hi', 'p_majority')}
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


def poll_sample(poll: dict):
    """
    Sample size of a poll record (``sample_size`` in the published part, ``sample`` in older records).

    Parameters
    ----------
    poll : dict
        Poll record.

    Returns
    -------
    int or float or None
        The sample, or ``None``.
    """
    return poll.get('sample_size', poll.get('sample'))


def leader_of(rows: list) -> Optional[dict]:
    """
    Row with the most seats (ties broken by the vote share).

    Parameters
    ----------
    rows : list of dict
        Party rows with ``seats`` and ``pct``.

    Returns
    -------
    dict or None
        The leading row, or ``None`` without rows.
    """
    if not rows:
        return None

    return max(rows, key=lambda row: (row.get('seats') or 0, row.get('pct') or 0))


def kpi_rows(summary: dict, meta: dict, parties: Catalog, polls: list) -> list:
    """
    The KPI row of Promedios: leading party, its seats, the absolute majority, the polls of the cycle and the
    last poll.

    Parameters
    ----------
    summary : dict
        Data of the summary part of the mode shown.
    meta : dict
        Data of the meta part (``n_seats``, ``n_polls``, ``n_pollsters``, ``bmaps``).
    parties : Catalog
        Party catalogue of the run.
    polls : list of dict
        Polls of the polls part (empty without the part).

    Returns
    -------
    list of dict
        Up to five ``{'label', 'value', 'unit', 'support', 'ok', 'color'}``: the majority KPI needs
        ``summary.p_majority`` and the last-poll KPI needs polls.
    """
    rows = summary.get('parties') or []
    leader = leader_of(rows)
    n_seats = summary.get('n_seats') or meta.get('n_seats')
    kpis = []

    if leader:
        kpis.append({'label': 'Primer partido', 'value': leader['name'], 'unit': fmt_pct(leader.get('pct')) + ' del voto',
                     'support': 'P(primero) ' + fmt_prob(leader.get('p_first'), 0), 'ok': False,
                     'color': parties[leader['name']]['color']})
        kpis.append({'label': 'Escaños del líder', 'value': fmt_int(leader.get('seats')), 'unit': 'de ' + fmt_int(n_seats),
                     'support': 'Rango 95 %: ' + fmt_range(leader.get('seats_lo'), leader.get('seats_hi'), 0),
                     'ok': False, 'color': None})

    probs = summary.get('p_majority') or {}

    if probs:
        block = max(probs, key=probs.get)
        seats = next((row.get('seats') for row in (summary.get('vs') or []) + (summary.get('blocks') or [])
                      if row.get('name') == block), None)
        members = ((meta.get('bmaps') or {}).get('vs') or {}).get(block) or []
        support = ' + '.join(members) + ': ' + fmt_int(seats) + ' escaños' if members and seats is not None else ''
        kpis.append({'label': 'Mayoría absoluta', 'value': fmt_prob(probs[block], 0), 'unit': block, 'support': support,
                     'ok': probs[block] >= 0.5, 'color': None})

    dated = [poll for poll in polls if poll.get('date')]
    first = min(dated, key=lambda poll: poll['date'])['date'] if dated else None
    kpis.append({'label': 'Sondeos del ciclo', 'value': fmt_int(meta.get('n_polls')),
                 'unit': 'de ' + fmt_int(meta.get('n_pollsters')) + ' encuestadoras',
                 'support': 'Desde el ' + fmt_date(first) if first else '', 'ok': False, 'color': None})

    if dated:
        last = max(dated, key=lambda poll: poll['date'])
        sample = poll_sample(last)
        bits = [bit for bit in ('para ' + last['sponsor'] if last.get('sponsor') else '',
                                fmt_int(sample) + ' entrevistas' if sample else '') if bit]
        kpis.append({'label': 'Último sondeo', 'value': fmt_date_short(last['date']), 'unit': last.get('pollster') or '',
                     'support': ' · '.join(bits), 'ok': False, 'color': None})

    return kpis


def party_cards(summary: dict, meta: dict, parties: Catalog, results: Optional[dict]) -> list:
    """
    The party cards of Promedios: the main parties of the run with their seats, vote and deltas.

    Parameters
    ----------
    summary : dict
        Data of the summary part of the mode shown.
    meta : dict
        Data of the meta part (``bmaps.main``).
    parties : Catalog
        Party catalogue of the run.
    results : dict or None
        ``last_results`` of the scope (deltas against it), or ``None``.

    Returns
    -------
    list of dict
        One card per party of ``bmaps.main`` present in the summary (the six with most seats without the map):
        ``name``, ``fullname``, ``color``, ``rank`` (by seats, ties by vote), ``seats``, ``pct``, ``delta_seats``
        (``'−25'``), ``delta_pct`` (``'+0,0'``), both ``None`` without results, and ``new`` (not in the results).
    """
    rows = summary.get('parties') or []
    by_name = {row['name']: row for row in rows}
    ranked = sorted(rows, key=lambda row: (-(row.get('seats') or 0), -(row.get('pct') or 0)))
    ranks = {row['name']: index + 1 for index, row in enumerate(ranked)}
    main = [name for name in ((meta.get('bmaps') or {}).get('main') or []) if name in by_name]
    names = main or [row['name'] for row in ranked[:6]]
    previous = (results or {}).get('parties') or {}
    cards = []

    for name in names:
        row = by_name[name]
        before = previous.get(name)
        cards.append({
            'name': name,
            'fullname': parties[name]['fullname'],
            'color': parties[name]['color'],
            'rank': ranks[name],
            'seats': row.get('seats'),
            'pct': row.get('pct'),
            'delta_seats': delta_text((row.get('seats') or 0) - (before.get('seats') or 0)) if before and results else None,
            'delta_pct': delta_text((row.get('pct') or 0) - (before.get('pct') or 0), 1) if before and results else None,
            'new': bool(results) and before is None,
        })

    return cards


def lead_strip(rows: list, majority, with_first: bool = True) -> Optional[dict]:
    """
    The strip under a hemicycle: the leading party and its distance to the absolute majority.

    Parameters
    ----------
    rows : list of dict
        Party rows (``name``, ``seats``, ``pct``, ``p_first`` optional).
    majority : int or None
        Absolute majority of the chamber.
    with_first : bool, optional
        Whether the strip carries ``p_first`` (not in the detail of a past election).

    Returns
    -------
    dict or None
        ``{'name', 'seats', 'short', 'over', 'p_first'}``: ``short`` is the seats missing for the majority (0 when
        reached), ``over`` whether it is reached, ``p_first`` the probability (``None`` without it or without
        ``with_first``); ``None`` without rows or majority.
    """
    leader = leader_of(rows)

    if leader is None or majority is None:
        return None

    seats = leader.get('seats') or 0

    return {
        'name': leader['name'],
        'seats': seats,
        'short': max(0, int(majority) - int(seats)),
        'over': seats >= majority,
        'p_first': leader.get('p_first') if with_first else None,
    }


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
        and ``block_result_rows`` (the tables of results of the summary), ``pct_max`` (the widest vote bar),
        ``n_seats``, ``majority``, ``kpis`` (``kpi_rows``), ``cards`` (``party_cards``), ``results_year`` (of the
        last election with results, for the deltas), ``lead`` (``lead_strip``), ``table_parties`` (every party of the
        polls part), ``table`` (``polls_table``, the ``PAGE_SIZE`` most recent polls the server renders),
        ``table_total`` (the polls of the polls part), ``months`` (``poll_months``), ``csv_href``,
        ``pollsters`` (``pollster_options``), ``blocks`` (the names of ``meta.bmaps.blocks``, empty without
        them) and ``initial`` (``state``, ``meta``, ``headline``,
        ``series`` (``round_series``), ``polls`` (``chart_polls``, which also feeds the whole table, filtered,
        sorted and paginated in the browser), ``projection``, ``vote``, ``summary``, ``runs``, ``cards`` (name
        and colour of the party cards, for the sparklines) and ``inks`` (the ink over the colour of every party
        and block, for the pills of the legend)).

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
    # a run without the polls part still shows the average: no polls table, no last-poll KPI
    polls = bundle_part(site().run_file, scope, run, 'polls') or {'parties': [], 'polls': []}
    poll_rows = polls.get('polls') or []
    bmaps = meta.get('bmaps') or {}
    blocks = bmaps.get('blocks')

    context['headline'] = current
    summary = part(scope, run, 'summary', mode=mode)
    # imported here: ``elecciones`` imports this module
    from .elecciones import last_results
    results = last_results(scope)
    context['result_rows'] = result_rows(summary)
    context['block_result_rows'] = block_result_rows(summary, meta, parties)
    context['pct_max'] = max((row['pct'] for row in context['result_rows'] if row.get('pct') is not None), default=0)
    context['n_seats'] = summary.get('n_seats') or meta.get('n_seats')
    context['majority'] = summary.get('majority') or meta.get('majority')
    context['kpis'] = kpi_rows(summary, meta, parties, poll_rows)
    context['cards'] = party_cards(summary, meta, parties, results)
    context['results_year'] = results['year'] if results else None
    context['lead'] = lead_strip(summary.get('parties') or [], context['majority'])
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
        'cards': [{'name': card['name'], 'color': card['color']} for card in context['cards']],
        # ink of every party and block of the series charts, for the pills of the legend when they are on
        'inks': {**{name: ink_color(parties[name]['color']) for name in series.get('parties') or []},
                 **{block: ink_color(block_color(block, meta, parties)) for block in context['blocks']}},
    }

    return context
