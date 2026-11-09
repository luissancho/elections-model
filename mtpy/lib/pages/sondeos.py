"""
The page of a poll: the lookup of the poll named by the URL and the average of the day it was published.
"""
from typing import Optional
from urllib.parse import urlsplit

from ...core.api import HttpError
from .casas import mtype_label
from .filters import DASH, _is_missing, fmt_date, fmt_int, fmt_num, gradient
from .promedios import RATING_COLORS
from .slugs import slugify


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


def series_at(series: dict, date: str) -> dict:
    """
    Average of the last day of a series up to a date.

    Parameters
    ----------
    series : dict
        A ``series`` part: ``dates`` (ISO, ascending), ``parties`` and ``mean`` (``{party: [values]}``).
    date : str
        Day to look up, ``YYYY-MM-DD``.

    Returns
    -------
    dict
        ``{party: mean}`` of the last day of the series that is not after ``date`` (``None`` for a party without a
        value that day); ``{}`` when the series has no such day.
    """
    index = None

    for i, day in enumerate(series.get('dates') or []):
        if day <= date:
            index = i

    if index is None:
        return {}

    mean = series.get('mean') or {}

    return {name: mean[name][index] if name in mean and index < len(mean[name]) else None
            for name in series.get('parties') or []}


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


def vote_rows(poll: dict, parties: list, average: dict, colors: dict, effects: Optional[dict] = None) -> list:
    """
    Rows of the vote table of a poll: its parties against the average of the day.

    Parameters
    ----------
    poll : dict
        Poll record (a value per party, ``seats``).
    parties : list
        Parties of the scope, in their order (it breaks the ties).
    average : dict
        ``{party: mean}`` of the day of the poll.
    colors : dict
        ``{party: colour}``.
    effects : dict, optional
        ``{party: house effect}``.

    Returns
    -------
    list
        One dict per party with a number in the poll, by descending value: ``name``, ``color``, ``value``, ``avg``,
        ``diff`` (value minus average, one decimal), ``effect`` and ``seats`` (``None`` when missing).
    """
    effects = effects or {}
    seats = poll.get('seats') or {}
    rows = []

    for name in parties:
        value = poll.get(name)

        if not _has_value(value):
            continue

        avg = average.get(name)
        avg = float(avg) if _has_value(avg) else None
        rows.append({'name': name, 'color': colors.get(name), 'value': float(value), 'avg': avg,
                     'diff': None if avg is None else round(value - avg, 1), 'effect': effects.get(name),
                     'seats': seats_text(seats.get(name))})

    return sorted(rows, key=lambda row: -row['value'])


def distance_to_average(poll: dict, parties: list, average: dict) -> Optional[float]:
    """
    Mean absolute gap between a poll and the average, over the parties that have both figures.

    Parameters
    ----------
    poll : dict
        Poll record.
    parties : list
        Parties to consider.
    average : dict
        ``{party: mean}`` of the day of the poll.

    Returns
    -------
    float or None
        The mean of ``|poll - average|`` to one decimal, or ``None`` without any shared party.
    """
    gaps = [abs(poll[name] - average[name]) for name in parties
            if _has_value(poll.get(name)) and _has_value(average.get(name))]

    return round(sum(gaps) / len(gaps), 1) if gaps else None


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
        ``factors`` (label and two-decimal value), ``has_part``, ``today`` (``date``, ``kernel``, ``weight``,
        ``party`` of the row of the poll for its biggest party, or ``None``), ``status`` (``None`` without part,
        ``'today'``, ``'zero'``, ``'outside'`` or ``'absent'``) and ``overlap`` (a weight of 0 whose overlap factor
        is 0 too).
    """
    factors = [{'label': label, 'value': fmt_num(poll.get(key), 2)}
               for label, key in (('Solape', 'weight_over'), ('Muestra', 'weight_sample'),
                                  ('Rating', 'weight_rating'), ('Peso', 'weight'))]
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


def poll_kpis(poll: dict, distance: Optional[float]) -> list:
    """
    Figures of the header of a poll page.

    Parameters
    ----------
    poll : dict
        Poll record.
    distance : float or None
        Distance to the average (``distance_to_average``).

    Returns
    -------
    list
        The ``kpis`` of the page layout: rating (coloured), weight, sample and distance.
    """
    rating = poll.get('rating')

    return [{'label': 'Rating', 'value': fmt_num(rating, 1), 'unit': '',
             'color': gradient(rating, 0, 100, RATING_COLORS) if _has_value(rating) else None},
            {'label': 'Peso en el promedio', 'value': fmt_num(poll.get('weight'), 2), 'unit': ''},
            {'label': 'Muestra', 'value': fmt_int(poll.get('sample_size')), 'unit': 'entrevistas'},
            {'label': 'Distancia al promedio', 'value': fmt_num(distance, 1), 'unit': 'pts'}]


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
        ``{'label', 'text', 'href'}`` rows; the source links to the URL of the poll or, without it, names the
        Wikipedia list.
    """
    url = poll.get('url')

    if url:
        source = {'text': urlsplit(url).hostname or url, 'href': url}
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
