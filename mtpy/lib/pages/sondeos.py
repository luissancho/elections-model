"""
The page of a poll (``/{scope}/sondeos/{date}/{pollster}[/{sponsor}]``): the lookup of the poll named by the URL and
of the previous delivery of its house and sponsor, the rows and figures of its cards and the template context.
"""
from typing import Optional
from urllib.parse import urlsplit

from ...core.api import HttpError
from ..webapi import site
from .casas import mtype_label
from .common import block_color, bundle_part, common_context, ink_color, part
from .elecciones import chart_parties
from .filters import DASH, _is_missing, _parse_datetime, fmt_date, fmt_int, fmt_num, gradient
from .promedios import RATING_COLORS, chart_polls, round_series
from .slugs import poll_detail_href, pollster_href, slugify


def find_poll(polls: list, date: str, pollster: str, sponsor: Optional[str] = None) -> dict:
    """
    Resolve the poll that a detail URL names.

    Parameters
    ----------
    polls : list
        Records of the ``polls`` part of a scope.
    date : str
        Day of the poll, ``YYYY-MM-DD``.
    pollster : str
        House segment of the URL: the slug of its name or its id.
    sponsor : str, optional
        Sponsor segment of the URL (the slug of the sponsor). Without it, the polls of the house that day without a
        sponsor slug are preferred, and if there are none, all the polls of the house that day.

    Returns
    -------
    dict
        The poll record, when exactly one poll matches.

    Raises
    ------
    HttpError
        404 ``poll not found`` when no poll or more than one matches.
    """
    day_polls = [poll for poll in polls
                 if str(poll.get('date'))[:10] == date
                 and pollster in {slugify(poll.get('pollster')), str(poll.get('pollster_id'))}]

    if sponsor is not None:
        candidates = [poll for poll in day_polls if slugify(poll.get('sponsor') or '') == sponsor]
    else:
        candidates = [poll for poll in day_polls if not slugify(poll.get('sponsor') or '')] or day_polls

    if len(candidates) != 1:
        raise HttpError(404, 'poll not found')

    return candidates[0]


def seats_text(seats_range) -> Optional[str]:
    """
    Text of the seats range of a party in a poll.

    Parameters
    ----------
    seats_range : list or None
        ``[low, high]`` seats.

    Returns
    -------
    str or None
        ``'136–138'``, or ``'100'`` when both ends coincide; ``None`` when there is no range.
    """
    if not seats_range or len(seats_range) < 2 or _is_missing(seats_range[0]) or _is_missing(seats_range[1]):
        return None

    low, high = int(seats_range[0]), int(seats_range[1])

    return str(low) if low == high else f'{low}{DASH}{high}'


def _has_value(value) -> bool:
    """
    Tell whether a poll value is a number.

    Parameters
    ----------
    value : object
        Value of a party in a poll or an average.

    Returns
    -------
    bool
        True for a number that is not NaN or infinite.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not _is_missing(value)


def previous_poll(polls: list, poll: dict) -> Optional[dict]:
    """
    Previous delivery of the same house and sponsor before a poll.

    Parameters
    ----------
    polls : list
        Records of the ``polls`` part of a scope, in its order.
    poll : dict
        One of the records of ``polls`` (found by identity; by date, house and sponsor otherwise).

    Returns
    -------
    dict or None
        Among the polls of the same house (``pollster_id``, or ``pollster`` when ``poll`` has no id, as the chart of
        the house of ``sondeo_context``) and sponsor (compared by its slug, as ``find_poll`` and the URLs), ordered by
        date and then by position in the part, the last one strictly before ``poll``; ``None`` when there is none.
    """
    house_key = 'pollster_id' if poll.get('pollster_id') is not None else 'pollster'
    sponsor_slug = slugify(poll.get('sponsor') or '')

    def same_line(other: dict) -> bool:
        return other.get(house_key) == poll.get(house_key) and slugify(other.get('sponsor') or '') == sponsor_slug

    keys = [(str(other.get('date'))[:10], index) for index, other in enumerate(polls) if same_line(other)]
    own = [index for index, other in enumerate(polls) if other is poll]

    if not own:
        own = [index for index, other in enumerate(polls)
               if same_line(other) and str(other.get('date'))[:10] == str(poll.get('date'))[:10]][:1]

    key = (str(poll.get('date'))[:10], own[0] if own else len(polls))
    earlier = [item for item in keys if item < key]

    return polls[max(earlier)[1]] if earlier else None


def fieldwork_overlaps(previous: dict, poll: dict) -> bool:
    """
    Tell whether the fieldwork of an earlier delivery reaches into the fieldwork of a poll.

    Parameters
    ----------
    previous : dict
        Poll record of the earlier delivery (``end_date``, else ``date``).
    poll : dict
        Poll record (``start_date``).

    Returns
    -------
    bool
        True when the last fieldwork day of ``previous`` is on or after the first fieldwork day of ``poll``; False
        when either day is unknown.
    """
    end = _parse_datetime(previous.get('end_date')) or _parse_datetime(previous.get('date'))
    start = _parse_datetime(poll.get('start_date'))

    return end is not None and start is not None and end.date() >= start.date()


def _seats_mid(seats_range) -> Optional[int]:
    """
    Midpoint of a seats range.

    Parameters
    ----------
    seats_range : list or None
        ``[low, high]`` seats.

    Returns
    -------
    int or None
        ``round((low + high) / 2)``; ``None`` when there is no range.
    """
    if seats_text(seats_range) is None:
        return None

    return round((int(seats_range[0]) + int(seats_range[1])) / 2)


def vote_rows(poll: dict, parties: list, colors: dict, previous: Optional[dict] = None) -> list:
    """
    Rows of the vote table of a poll: its vote and seats by party, against the previous delivery of the same house and
    sponsor.

    Parameters
    ----------
    poll : dict
        Poll record (a value per party, ``seats``).
    parties : list
        Parties of the scope, in their order (it breaks the ties).
    colors : dict
        ``{party: colour}``.
    previous : dict, optional
        Previous delivery of the same house and sponsor.

    Returns
    -------
    list
        One dict per party with a number in the poll, by descending value: ``name``, ``color``, ``value``,
        ``seats`` (``None`` when missing), ``change`` (value minus its figure in the previous delivery,
        one decimal) and ``seats_change`` (difference of the midpoints of the two ranges); ``None`` when the previous
        delivery lacks the figure.
    """
    seats = poll.get('seats') or {}
    prev_seats = (previous or {}).get('seats') or {}
    rows = []

    for name in parties:
        value = poll.get(name)

        if not _has_value(value):
            continue

        prev = (previous or {}).get(name)
        mid, prev_mid = _seats_mid(seats.get(name)), _seats_mid(prev_seats.get(name))
        rows.append({'name': name, 'color': colors.get(name), 'value': float(value),
                     'seats': seats_text(seats.get(name)),
                     'change': round(value - prev, 1) if _has_value(prev) else None,
                     'seats_change': None if mid is None or prev_mid is None else mid - prev_mid})

    return sorted(rows, key=lambda row: -row['value'])


def weight_card(poll: dict, weights: Optional[dict]) -> dict:
    """
    Content of the weight card of a poll: the factors of its weight and the row of the ``weights`` part.

    Parameters
    ----------
    poll : dict
        Poll record (``weight``, ``weight_over``, ``weight_sample``, ``weight_rating``).
    weights : dict or None
        The published ``weights`` part, or ``None`` when the scope has none.

    Returns
    -------
    dict
        ``factors`` (label and two-decimal value of F(solape), F(muestra) and F(rating); the total weight is in
        ``poll_kpis``), ``has_part``, ``today`` (``date``, ``kernel``, ``weight``,
        ``party`` of the row of the poll for its biggest party, or ``None``), ``status`` (``None`` without part,
        ``'today'``, ``'zero'``, ``'outside'`` or ``'absent'``) and ``overlap`` (a weight of 0 whose overlap factor
        is 0 too).
    """
    factors = [{'label': label, 'value': fmt_num(poll.get(key), 2)}
               for label, key in (('F(solape)', 'weight_over'), ('F(muestra)', 'weight_sample'),
                                  ('F(rating)', 'weight_rating'))]
    card = {'factors': factors, 'has_part': weights is not None, 'today': None, 'status': None, 'overlap': False}

    if weights is None:
        return card

    weight = poll.get('weight')

    if _is_missing(weight) or weight == 0:
        card['status'] = 'zero'
        card['overlap'] = not _is_missing(poll.get('weight_over')) and poll.get('weight_over') == 0

        return card

    date = str(poll.get('date'))[:10]
    sponsor = poll.get('sponsor') or None
    rows = [row for row in weights.get('rows') or []
            if str(row.get('date'))[:10] == date and row.get('pollster_id') == poll.get('pollster_id')
            and (row.get('sponsor') or None) == sponsor and _has_value(poll.get(row.get('name')))]

    if rows:
        row = max(rows, key=lambda r: poll[r['name']])
        card['today'] = {'date': date, 'kernel': fmt_num(row.get('weight_kernel'), 2),
                         'weight': fmt_num(row.get('weight'), 2), 'party': row['name']}
        card['status'] = 'today'
    elif any(line.get('dates') and date < str(line['dates'][0])[:10] for line in (weights.get('line') or {}).values()):
        card['status'] = 'outside'
    else:
        card['status'] = 'absent'

    return card


def poll_kpis(poll: dict) -> list:
    """
    Figures of a poll in the average, at the head of its weight card.

    Parameters
    ----------
    poll : dict
        Poll record.

    Returns
    -------
    list
        The ``kpis`` of the page layout (``label``, ``value``, ``unit`` and ``color``, ``None`` but on the rating):
        the rating of the house (0-100, coloured as the badge of the polls table) and the weight of the poll in the
        average. A missing value is a dash.
    """
    rating = poll.get('rating')

    return [{'label': 'Rating', 'value': fmt_num(rating, 1), 'unit': '',
             'color': gradient(rating, 0, 100, RATING_COLORS) if _has_value(rating) else None},
            {'label': 'Peso en el promedio', 'value': fmt_num(poll.get('weight'), 2), 'unit': '', 'color': None}]


def sample_text(poll: dict) -> Optional[str]:
    """
    Sample of a poll for the lead line of its card.

    Parameters
    ----------
    poll : dict
        Poll record (``sample_size``).

    Returns
    -------
    str or None
        ``'1.200 entrevistas'``, ``None`` without a sample (the lead line then leaves it out).
    """
    sample = poll.get('sample_size')

    return f'{fmt_int(sample)} entrevistas' if _has_value(sample) and sample > 0 else None


def poll_title(poll: dict) -> str:
    """
    Title of a poll: the house, plus ``' para '`` and the outlet that commissioned it when there is one.

    Parameters
    ----------
    poll : dict
        Poll record (``pollster``, ``sponsor``).

    Returns
    -------
    str
        For example ``'NC Report para La Razón'``.
    """
    sponsor = poll.get('sponsor')

    return f"{poll.get('pollster')} para {sponsor}" if sponsor else str(poll.get('pollster'))


def fieldwork_text(poll: dict) -> Optional[str]:
    """
    Fieldwork dates of a poll with the year.

    Parameters
    ----------
    poll : dict
        Poll record (``start_date``, ``end_date``).

    Returns
    -------
    str or None
        ``'29 oct 2026 – 30 oct 2026'``, a single date when they coincide or only one is known, ``None`` without
        any valid date.
    """
    days = []

    for key in ('start_date', 'end_date'):
        text = fmt_date(poll.get(key))

        if text != DASH and text not in days:
            days.append(text)

    return f'{days[0]} {DASH} {days[1]}' if len(days) == 2 else (days[0] if days else None)


def sheet_rows(poll: dict, house_href: str, event_label: str, run_text: str) -> list:
    """
    Rows of the data sheet of a poll.

    Parameters
    ----------
    poll : dict
        Poll record.
    house_href : str
        Link to the profile of the house.
    event_label : str
        Text of the election the poll belongs to.
    run_text : str
        Text of the model update shown.

    Returns
    -------
    list
        ``{'label', 'text', 'href'}`` rows; the source links to the URL of the poll when it is an ``http`` or
        ``https`` address with a host or, without one, names the Wikipedia list (without a link).
    """
    url = poll.get('url').strip() if isinstance(poll.get('url'), str) else None

    try:
        parts = urlsplit(url) if url else None
        host = parts.hostname if parts is not None and parts.scheme.lower() in ('http', 'https') else None
    except ValueError:  # a malformed address (a broken IPv6 host, a bad port): no link
        host = None

    if host:
        source = {'text': host, 'href': url}
    else:
        source = {'text': 'Wikipedia, lista de sondeos', 'href': None}

    return [{'label': 'Encuestadora', 'text': poll.get('pollster'), 'href': house_href},
            {'label': 'Medio', 'text': poll.get('sponsor') or '—', 'href': None},
            {'label': 'Tipo', 'text': mtype_label(poll.get('mtype')), 'href': None},
            {'label': 'Trabajo de campo', 'text': fieldwork_text(poll) or fmt_date(poll.get('date')), 'href': None},
            {'label': 'Muestra', 'text': fmt_int(poll.get('sample_size')), 'href': None},
            {'label': 'Rating', 'text': fmt_num(poll.get('rating'), 1), 'href': None},
            {'label': 'Peso', 'text': fmt_num(poll.get('weight'), 2), 'href': None},
            {'label': 'Fuente', **source},
            {'label': 'Elección', 'text': event_label, 'href': None},
            {'label': 'Actualización del modelo', 'text': run_text, 'href': None}]


def sondeo_context(state: dict, date: str, pollster: str, sponsor: Optional[str] = None) -> dict:
    """
    Template context of the page of a poll.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``), the scope taken from the route.
    date : str
        Day of the poll in the URL (already checked), the last day of its fieldwork.
    pollster : str
        House segment of the URL (``find_poll``).
    sponsor : str, optional
        Sponsor segment of the URL.

    Returns
    -------
    dict
        ``common_context`` of an event page, its ``event_bar['sections']`` linking the sections of the event page
        (``'/{scope}{query}#{id}'``, without ``territorio`` when the run has no districts), plus ``poll``,
        ``poll_title``, ``house`` (``name``, ``href``, ``type``), ``fieldwork`` (``fieldwork_text``), ``sample_text``
        (``sample_text``), ``poll_date``,
        ``event_label`` (the event and its date), ``kpis`` (``poll_kpis``, at the head of the weight card),
        ``vote_rows``, ``previous`` (``date``, ``href`` keeping the query, ``title``, ``fieldwork``
        (``fieldwork_text``, ``None`` without dates) and ``overlap`` (``fieldwork_overlaps``) of the previous delivery
        of the house and sponsor, ``None`` without one), ``has_seats``, ``weight``
        (``weight_card``, its ``factors`` with an empty ``unit`` and no ``color`` for ``m.kpis``), ``weights_date``
        (the day of the ``weights`` part, ``None`` without it), ``sheet`` (``sheet_rows``), ``back_href`` (the polls
        table of the event page), ``detail_href``, ``image_file`` and ``image_sign`` (the image of the card),
        ``chart_file`` (the image of the chart of the house), ``blocks`` (the names of ``meta.bmaps.blocks``) and
        ``initial``: ``state``, ``meta``, ``series`` (``round_series``), ``polls`` (``chart_polls`` with the polls of
        the house only), ``projection`` and ``vote`` (the parts of the mode), ``catalog`` (``parties`` and ``bmaps``,
        as on the event page), ``inks``, ``poll`` (``date``, ``pollster_id``, ``sponsor``, ``pollster`` and the
        percentages), ``previous`` (``values``, the shares, and ``seats``, the ranges, of the previous delivery, or
        ``None``), ``seats`` (``None`` without them), ``n_seats`` and ``majority``. The card of the poll draws the poll
        and its change against the previous delivery, never the average (Luis, 2026-11-09).

    Raises
    ------
    HttpError
        404 ``poll not found`` without a ``polls`` part or without the poll; those of ``common_context``.
    """
    context = common_context(state, None, event=True)
    full, meta, parties, query = context['state'], context['meta'], context['catalog'], context['query']
    scope, mode, run = full['scope'], full['mode'], full['run']
    polls = bundle_part(site().run_file, scope, run, 'polls')

    if polls is None:
        raise HttpError(404, 'poll not found')

    poll = find_poll(polls.get('polls') or [], date, pollster, sponsor)
    poll_parties = list(polls.get('parties') or [])
    series = part(scope, run, 'series')
    weights = bundle_part(site().run_file, scope, run, 'weights')
    colors = {name: parties[name]['color'] for name in poll_parties}
    bmaps = meta.get('bmaps') or {}
    blocks = bmaps.get('blocks')
    blocks = list(blocks) if isinstance(blocks, dict) else []
    event = context['event_bar']
    event_label = '{} · {}'.format(event['title'], fmt_date(event['date']))
    house = {'name': poll.get('pollster'), 'href': pollster_href(poll.get('pollster'), poll.get('pollster_id')),
             'type': mtype_label(poll.get('mtype'))}
    run_text = 'Actualizado el ' + fmt_date(meta.get('run_at')) + (' (run anclado)' if full['pinned'] else '')
    weight = weight_card(poll, weights)
    house_slug = slugify(poll.get('pollster')) or str(poll.get('pollster_id'))
    sponsor_slug = slugify(poll.get('sponsor') or '')
    # the index links the sections of the event page, keeping the mode and the pinned run; the territory only when
    # the run has districts, as the event page does
    districts = bundle_part(site().run_file, scope, run, 'districts', mode) or {}
    missing = set() if districts.get('regions') else {'territorio'}
    context['event_bar'] = dict(event, sections=[dict(entry, href='/' + scope + query + '#' + entry['id'])
                                                 for entry in event['sections'] if entry['id'] not in missing])
    # the polls of the chart of the house: the same id, or the same name when the poll has no id
    key = 'pollster_id' if poll.get('pollster_id') is not None else 'pollster'
    house_polls = dict(polls, polls=[other for other in polls.get('polls') or [] if other.get(key) == poll.get(key)])
    run_parties = [entry['name'] for entry in meta.get('parties') or []]
    seats = poll.get('seats') or None
    prev = previous_poll(polls.get('polls') or [], poll)
    previous = None if prev is None else {'date': fmt_date(prev['date']), 'title': poll_title(prev),
                                          'href': poll_detail_href(scope, prev) + query,
                                          'fieldwork': fieldwork_text(prev), 'overlap': fieldwork_overlaps(prev, poll)}

    context.update({
        'poll': poll,
        'poll_title': poll_title(poll),
        'house': house,
        'fieldwork': fieldwork_text(poll),
        'sample_text': sample_text(poll),
        'poll_date': fmt_date(date),
        'event_label': event_label,
        'kpis': poll_kpis(poll),
        'vote_rows': vote_rows(poll, poll_parties, colors, prev),
        'previous': previous,
        'has_seats': bool(seats),
        'weight': dict(weight, factors=[dict(factor, unit='', color=None) for factor in weight['factors']]),
        'weights_date': weights.get('date') if weights is not None else None,
        'sheet': sheet_rows(poll, house['href'], event_label, run_text),
        'back_href': '/' + scope + query + '#tabla',
        'detail_href': poll_detail_href(scope, poll),
        'image_file': 'demoscopio-sondeo-{}-{}-{}{}.png'.format(scope, date, house_slug,
                                                                '-' + sponsor_slug if sponsor_slug else ''),
        'image_sign': '{} · {}'.format(context['site_host'], context['author_handle']),
        'chart_file': 'demoscopio-encuestadora-{}-{}-{}.png'.format(scope, house_slug, date),
        'blocks': blocks,
    })
    context['initial'] = {
        'state': full,
        'meta': meta,
        'series': round_series(series),
        'polls': chart_polls(house_polls, scope),
        'projection': part(scope, run, 'projection', mode=mode),
        'vote': part(scope, run, 'vote', mode=mode),
        'catalog': {'parties': [dict(parties[name], name=name)
                                for name in chart_parties(parties, run_parties, poll_parties)], 'bmaps': bmaps},
        'inks': {**{name: ink_color(parties[name]['color'])
                    for name in run_parties + poll_parties + list(series.get('parties') or [])},
                 **{block: ink_color(block_color(block, meta, parties)) for block in blocks}},
        'poll': {'date': poll.get('date'), 'pollster_id': poll.get('pollster_id'), 'sponsor': poll.get('sponsor'),
                 'pollster': poll.get('pollster'), **{name: poll[name] for name in poll_parties if name in poll}},
        'previous': None if prev is None else {
            'values': {name: prev[name] for name in poll_parties if _has_value(prev.get(name))},
            'seats': prev.get('seats') or None},
        'seats': seats,
        'n_seats': meta.get('n_seats'),
        'majority': meta.get('majority'),
    }

    return context
