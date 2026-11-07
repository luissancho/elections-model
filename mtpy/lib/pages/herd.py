"""
View models of the page of the herd study (``/analisis/rebano``): the verdicts of its two questions, the three KPIs,
the bars of the elections, the squares of the two comparisons, the sentence of the chart by weeks and the one of the
live card.

``analisis.herd_context`` assembles them from the published parts (``meta``, ``summary``, ``elections``, ``current``,
``weeks``); each function takes the data it reads and returns what the template renders, with ``None`` or an empty
structure for what the parts do not have, never an error.
"""
from datetime import date, datetime, timedelta
from typing import Optional

from .cera import MONTHS_LONG, _get
from .common import days_left as days_to
from .filters import DASH, _madrid_tz, fmt_int, fmt_num

HERD = 'rebano'
# Significance level of the sign test
ALPHA = 0.05
# Below this probability the chance is written «menos de 1 entre 1.000»; above ``COIN`` it is «como una moneda»
RARE, COIN = 0.001, 0.2
# Days between the start of the LOREG ban on polls and the election
VEDA_DAYS = 6
# The bars are clipped at this change (±100 %)
BAR_CLIP = 1.0
MINUS = '−'


def _is_num(x) -> bool:
    """Whether ``x`` is a real number (not a bool)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _date(iso: Optional[str]) -> Optional[date]:
    """The ``date`` of an ISO string, or ``None`` when it is not one."""
    try:
        return date.fromisoformat((iso or '')[:10])
    except ValueError:
        return None


def day_month(iso: Optional[str]) -> Optional[str]:
    """«29 de noviembre», or ``None`` without a valid date."""
    day = _date(iso)

    return '{} de {}'.format(day.day, MONTHS_LONG[day.month - 1]) if day else None


def short_day(iso: Optional[str]) -> Optional[str]:
    """The day of an election as the press writes it: «29-N»."""
    day = _date(iso)

    return '{}-{}'.format(day.day, MONTHS_LONG[day.month - 1][0].upper()) if day else None


def names(items: list) -> str:
    """A list in prose: «A, B y C» (``''`` when empty)."""
    return ', '.join(items[:-1]) + ' y ' + items[-1] if len(items) > 1 else ''.join(items)


def today() -> date:
    """Today in Europe/Madrid, the day the texts of the page count from (the tests replace it)."""
    return datetime.now(_madrid_tz()).date()


def percent(change, signed: bool = True) -> str:
    """
    A relative change as the page writes it: «−45 %», «+623 %» (the typographic minus, a space before the sign).

    Parameters
    ----------
    change : float or None
        The change as a fraction (``-0.45``).
    signed : bool, optional
        Whether a positive figure carries «+».

    Returns
    -------
    str
        The percentage rounded to the unit, a dash when ``change`` is missing.
    """
    if not _is_num(change):
        return DASH

    value = round(change * 100)

    if value < 0:
        return '{}{} %'.format(MINUS, fmt_int(-value))

    return '{}{} %'.format('+' if signed and value else '', fmt_int(value))


def percent_fine(change) -> str:
    """
    ``percent`` with one decimal when the change rounds to 0 % without being 0: «−0,3 %», «+0,4 %».

    Parameters
    ----------
    change : float or None
        The change as a fraction.

    Returns
    -------
    str
        As ``percent``, except for a non-zero change below half a point, written with its sign and one decimal.
    """
    if not _is_num(change) or round(change * 100) or not round(change * 100, 1):
        return percent(change)

    value = round(change * 100, 1)

    return '{}{} %'.format(MINUS if value < 0 else '+', fmt_num(abs(value), 1))


def chance_text(p: Optional[float]) -> Optional[str]:
    """
    The probability of a sign test in words.

    Parameters
    ----------
    p : float or None
        The ``sign_p`` of a comparison.

    Returns
    -------
    str or None
        ``None`` without ``p``; «menos de 1 entre 1.000» below 0,001; «como una moneda» above 0,2; otherwise «1 entre
        769».
    """
    if not _is_num(p):
        return None

    if p < RARE:
        return 'menos de 1 entre 1.000'

    if p > COIN:
        return 'como una moneda'

    return '1 entre {}'.format(fmt_int(round(1 / p)))


def _significant(comparison: Optional[dict]) -> bool:
    """Whether a comparison falls in more than half of its elections with a sign-test p value below ``ALPHA``."""
    n, k, p = (_get(comparison, key) for key in ('n', 'k', 'sign_p'))

    return bool(n) and _is_num(k) and k > n / 2 and _is_num(p) and p < ALPHA


def _comparison(summary: Optional[dict], key: str) -> dict:
    """A comparison of the ``summary`` part (``{}`` when missing)."""
    return _get(summary, 'comparisons', key) or {}


def verdicts(summary: Optional[dict]) -> dict:
    """
    The verdicts of the two questions of the study.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.

    Returns
    -------
    dict
        ``{'q1': {'label', 'tone'}, 'q2': {...}}``: question 1 reads ``far_final`` (do the houses get closer on the
        whole?), question 2 ``mid_final`` and ``far_mid`` (is it in the last weeks?).
    """
    far_final = _comparison(summary, 'far_final')
    mid_final, far_mid = _comparison(summary, 'mid_final'), _comparison(summary, 'far_mid')
    n = far_final.get('n') or 0

    if not n:
        first = ('SIN DATOS', 'q2')
    elif _significant(far_final):
        first = ('SÍ, EN CONJUNTO', 'q1')
    elif (far_final.get('k') or 0) <= n / 2:
        first = ('NO', 'q3')
    else:
        first = ('NO SE PUEDE DECIR', 'q2')

    if not mid_final.get('n'):
        second = ('SIN DATOS', 'q2')
    elif not _significant(mid_final):
        second = ('NO', 'q3')
    elif _significant(far_mid):
        second = ('SÍ, DESDE ANTES', 'q1')
    else:
        second = ('SÍ, EN LAS TRES ÚLTIMAS SEMANAS', 'q1')

    return {'q1': {'label': first[0], 'tone': first[1]}, 'q2': {'label': second[0], 'tone': second[1]}}


def flags(meta: Optional[dict], current: Optional[dict]) -> dict:
    """
    The conditions and the dates of the texts of the page.

    The days to the election count from ``today()``, not from the ``as_of`` of the run: inside the veda the forecast
    is refused and the daily republication keeps the last run, so its ``days_left`` would stay frozen.

    Parameters
    ----------
    meta : dict or None
        The ``meta`` part.
    current : dict or None
        The ``current`` block (the election in progress), ``None`` without it.

    Returns
    -------
    dict
        ``has_run``, ``held`` (today is the election day or later), ``days_left`` (from ``today()``), ``year``,
        ``election_day`` («29 de noviembre»), ``veda_date`` (the election minus ``VEDA_DAYS``), ``after_veda``,
        ``final_start`` («9 de noviembre», from ``windows.final``) and ``final_started`` (today is that day or later).
    """
    election = _get(meta, 'election') or {}
    day = _date(election.get('date'))
    days_left = days_to(election.get('date'), today())
    final = _date(_get(current, 'windows', 'final'))

    return {
        'has_run': bool(current),
        'held': _is_num(days_left) and days_left <= 0,
        'days_left': days_left,
        'year': str(day.year) if day else None,
        'election_day': day_month(election.get('date')),
        'veda_date': day_month((day - timedelta(days=VEDA_DAYS)).isoformat()) if day else None,
        'after_veda': _is_num(days_left) and days_left <= VEDA_DAYS,
        'final_start': day_month(final.isoformat()) if final else None,
        'final_started': bool(final) and today() >= final,
    }


def _odds(p: Optional[float]) -> Optional[str]:
    """«por azar, 1 entre 769» (or the coin) for the support line of a KPI, ``None`` without ``p``."""
    text = chance_text(p)

    if text is None or text == 'como una moneda':
        return text

    return 'por azar, ' + text


def kpis(summary: Optional[dict], current: Optional[dict], flag: dict) -> list:
    """
    The three rows of the ``m.kpis`` macro.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    current : dict or None
        The ``current`` block.
    flag : dict
        ``flags(meta, current)``.

    Returns
    -------
    list of dict
        ``{'label', 'value', 'unit', 'support', 'ok', 'color'}`` per KPI. The third is the change of the election in
        progress (the last weeks against 3 to 8 weeks before, or the first comparison when the last weeks have no
        pairs yet) and, without a run, the count of generales and autonómicas.
    """
    row = {'unit': '', 'ok': False, 'color': None}
    far_final, mid_final = _comparison(summary, 'far_final'), _comparison(summary, 'mid_final')
    far_mid = _comparison(summary, 'far_mid')
    count = '{} de {}'

    rows = [
        dict(row, label='Elecciones en que las encuestadoras se acercan al final',
             value=count.format(fmt_int(far_final.get('k')), fmt_int(far_final.get('n'))),
             support=_odds(far_final.get('sign_p')) or ''),
        dict(row, label='En las tres últimas semanas',
             value=count.format(fmt_int(mid_final.get('k')), fmt_int(mid_final.get('n'))),
             support='hasta tres semanas antes, {}: {}'.format(
                 count.format(fmt_int(far_mid.get('k')), fmt_int(far_mid.get('n'))),
                 chance_text(far_mid.get('sign_p')) or DASH)),
    ]

    if not current:
        kinds = _get(summary, 'kinds') or {}
        rows.append(dict(row, label='Generales y autonómicas',
                         value='{} y {}'.format(fmt_int(_get(kinds, 'generales', 'n') or 0),
                                                fmt_int(_get(kinds, 'autonómicas', 'n') or 0)),
                         support='elecciones de cada clase con sondeos lejos y al final'))
        return rows

    own = _get(current, 'mid_final')
    versus = 'entre 3 y 8 semanas antes'
    comparison = own or _get(current, 'far_mid')
    support = ['de separación entre las mismas parejas de casas, frente a ' + versus if own else
               'de separación entre las mismas parejas de casas, frente a más de 8 semanas antes']

    if _is_num(flag['days_left']) and flag['days_left'] > 0:
        support.append('a {} {}'.format(fmt_int(flag['days_left']), 'día' if flag['days_left'] == 1 else 'días'))

    if not flag['final_started'] and flag['final_start']:
        support.append('el tramo final empieza el ' + flag['final_start'])

    rows.append(dict(row, label='{}, en directo'.format(current.get('label') or 'Elección en curso'),
                     value=percent(_get(comparison, 'change')) if comparison else DASH,
                     support=' · '.join(support)))

    return rows


def bars(rows: list) -> list:
    """
    The bars of question 1: one per election of the ``far_final`` comparison, from the largest fall to the largest rise.

    Parameters
    ----------
    rows : list of dict
        The rows of the ``elections`` part.

    Returns
    -------
    list of dict
        ``{'label', 'text', 'side', 'width'}``: ``text`` the signed percentage (``percent_fine``), ``side``
        ``'down'`` when the election ``falls`` (the same count as the squares and the KPIs, even when the published
        change rounds to zero), ``'up'`` otherwise, and ``width`` the length of the bar in % of half the axis, clipped
        at 100.
    """
    chosen = sorted((row for row in rows or [] if row.get('comparison') == 'far_final'
                     and _is_num(row.get('change'))), key=lambda row: row['change'])

    return [{'label': row.get('label'), 'text': percent_fine(row['change']),
             'side': 'down' if row.get('falls', row['change'] < 0) else 'up',
             'width': round(min(BAR_CLIP, abs(row['change'])) * 100, 1)} for row in chosen]


def squares(comparison: Optional[dict]) -> dict:
    """
    The squares of a comparison: one per election, filled when the houses get closer.

    Returns
    -------
    dict
        ``{'k', 'n', 'cells', 'chance'}``: ``cells`` is ``[True] * k + [False] * (n - k)``.
    """
    n, k = (_get(comparison, 'n') or 0), (_get(comparison, 'k') or 0)

    return {'k': k, 'n': n, 'cells': [True] * k + [False] * max(0, n - k),
            'chance': chance_text(_get(comparison, 'sign_p'))}


def live_rows(current: Optional[dict]) -> list:
    """
    The figures of the live card: one row per comparison of the election in progress that already has pairs of houses.

    Parameters
    ----------
    current : dict or None
        The ``current`` block.

    Returns
    -------
    list of dict
        ``{'key', 'pairs', 'r_a', 'r_b', 'change'}`` for ``far_mid`` and ``mid_final`` (in that order, those that are
        not ``None``); ``change`` is written by ``percent``, ``r_a`` and ``r_b`` are the dispersion of the earlier and
        of the later period.
    """
    rows = []

    for key in ('far_mid', 'mid_final'):
        block = _get(current, key)

        if block:
            rows.append({'key': key, 'pairs': block.get('pairs'), 'r_a': block.get('r_a'), 'r_b': block.get('r_b'),
                         'change': percent(block.get('change'))})

    return rows


def live_text(current: Optional[dict], flag: dict) -> Optional[str]:
    """
    The sentence of the live card.

    Returns
    -------
    str or None
        Once the election is held, that it enters the count with the scrutiny; before the final stretch, the day it
        starts; during it, how much the same pairs of houses differ against 3 to 8 weeks before; ``None`` without a
        run.
    """
    if not current:
        return None

    if flag['held']:
        return 'Elección celebrada: entra en la cuenta con el escrutinio.'

    if not flag['final_started']:
        if not flag['final_start']:
            return None

        return 'El tramo de las tres últimas semanas empieza el {}.'.format(flag['final_start'])

    change = _get(current, 'mid_final', 'change')

    if not _is_num(change):
        return 'Por ahora, el tramo de las tres últimas semanas no tiene parejas de casas que comparar.'

    value = round(abs(change) * 100)

    if not value:
        return 'Por ahora, las mismas parejas de casas difieren igual que entre 3 y 8 semanas antes.'

    return 'Por ahora, las mismas parejas de casas difieren un {} % {} que entre 3 y 8 semanas antes.'.format(
        fmt_int(value), 'menos' if change < 0 else 'más')


def _bucket_words(lo, hi) -> str:
    """
    A bucket of weeks before the election in words.

    Parameters
    ----------
    lo, hi : int
        First and last week of the bucket.

    Returns
    -------
    str
        «en las tres últimas semanas» for ``0-2``, «a más de 27 semanas» for ``27-52`` (open on the far side) and
        «entre 3 y 4 semanas antes» for any other.
    """
    if lo == 0:
        return 'en las tres últimas semanas'

    if hi is not None and hi >= 52:
        return 'a más de {} semanas'.format(lo)

    return 'entre {} y {} semanas antes'.format(lo, hi)


def weeks_text(weeks: Optional[dict]) -> Optional[str]:
    """
    The sentence of the chart of the separation by weeks (its ``aria-label``).

    Parameters
    ----------
    weeks : dict or None
        The ``weeks`` part (``{'buckets', 'rows', 'mean'}``, farthest bucket first).

    Returns
    -------
    str or None
        «La media de las 7 generales pasadas pasa de 1,65 a más de 27 semanas a 1,07 en las tres últimas; Generales
        2026 va en 1,06 entre 3 y 4 semanas antes.»: the number of past elections with any ratio, the mean of the first and of the
        last bucket that have one and, when the election in progress has a ratio, the one of its bucket nearest to the
        election; ``None`` without any mean.
    """
    mean = [row for row in _get(weeks, 'mean') or [] if _is_num(row.get('ratio'))]

    if not mean:
        return None

    rows = [row for row in _get(weeks, 'rows') or [] if _is_num(row.get('ratio'))]
    past = {row.get('event_date') for row in rows if not row.get('current')}
    text = 'La media de las {} generales pasadas pasa de {} a más de {} semanas a {} en las tres últimas'.format(
        fmt_int(len(past)), fmt_num(mean[0]['ratio'], 2), mean[0].get('lo'), fmt_num(mean[-1]['ratio'], 2))
    current = [row for row in rows if row.get('current')]

    if current:
        nearest = min(current, key=lambda row: row.get('lo') or 0)
        text += '; {} va en {} {}'.format(nearest.get('label'), fmt_num(nearest['ratio'], 2),
                                          _bucket_words(nearest.get('lo'), nearest.get('hi')))

    return text + '.'


def weeks_since(weeks: Optional[dict]) -> Optional[str]:
    """
    The year of the first past election of the chart of the separation by weeks.

    Parameters
    ----------
    weeks : dict or None
        The ``weeks`` part.

    Returns
    -------
    str or None
        The year of the earliest past election with a ratio in any bucket («2008»), ``None`` without one (the page
        then leaves the chart out).
    """
    dates = [row.get('event_date') or '' for row in _get(weeks, 'rows') or []
             if not row.get('current') and _is_num(row.get('ratio'))]

    return min(dates)[:4] if dates else None


def _whole(x) -> int:
    """A fraction as the whole percentage the page writes (``round(abs(x) * 100)``)."""
    return round(abs(x) * 100)


def ci_text(comparison: Optional[dict]) -> Optional[str]:
    """
    The interval of the median change of a comparison in words.

    Parameters
    ----------
    comparison : dict or None
        A comparison of the ``summary`` part, with its ``ci`` (``{'lo', 'hi'}`` as fractions).

    Returns
    -------
    str or None
        «entre el 3 y el 35 %» for an interval of falls, «entre una subida del 19 % y una bajada del 28 %» when it
        crosses zero, «entre el 5 y el 12 % de subida» for rises, «en torno al 15 %» when both bounds round to the same
        figure; ``None`` without a valid ``ci``.
    """
    lo, hi = _get(comparison, 'ci', 'lo'), _get(comparison, 'ci', 'hi')

    if not (_is_num(lo) and _is_num(hi)):
        return None

    rise = lo >= 0 and hi > 0
    first, second = sorted((_whole(lo), _whole(hi)))
    suffix = ' de subida' if rise else ''

    if lo < 0 < hi:
        return 'entre una subida del {} % y una bajada del {} %'.format(fmt_int(_whole(hi)), fmt_int(_whole(lo)))

    if first == second:
        return 'en torno al {} %{}'.format(fmt_int(first), suffix)

    return 'entre el {} y el {} %{}'.format(fmt_int(first), fmt_int(second), suffix)


def pooled_text(comparison: Optional[dict], later: str, earlier: str) -> Optional[str]:
    """
    The sentence of the joint test of a comparison.

    Parameters
    ----------
    comparison : dict or None
        A comparison of the ``summary`` part, with its ``pooled`` block.
    later, earlier : str
        The two periods in words («al final», «lejos de la votación»).

    Returns
    -------
    str or None
        «Con todas las parejas de todas las elecciones juntas, la separación al final es un 27 % menor que lejos de la
        votación: por azar, menos de 1 entre 1.000.»; ``None`` without ``pooled``.
    """
    change, p = _get(comparison, 'pooled', 'change'), _get(comparison, 'pooled', 'p')

    if not _is_num(change):
        return None

    value = _whole(change)
    odds = _odds(p)
    tail = ': {}'.format(odds) if odds else ''
    head = 'Con todas las parejas de todas las elecciones juntas, la separación {} '.format(later)

    if not value:
        return '{}es igual que {}{}.'.format(head, earlier, tail)

    return '{}es un {} % {} que {}{}.'.format(head, fmt_int(value), 'menor' if change < 0 else 'mayor', earlier, tail)


def _pooled_clause(comparison: Optional[dict], span: str) -> Optional[str]:
    """One clause of ``pooled_both_text`` («un 23 % menos en las tres últimas semanas que … (odds)»), or ``None``."""
    change = _get(comparison, 'pooled', 'change')

    if not _is_num(change):
        return None

    value = _whole(change)
    odds = _odds(_get(comparison, 'pooled', 'p'))
    text = 'lo mismo ' + span

    if value:
        text = 'un {} % {} {}'.format(fmt_int(value), 'menos' if change < 0 else 'más', span)

    return text + (' ({})'.format(odds) if odds else '')


def pooled_both_text(mid_final: Optional[dict], far_mid: Optional[dict]) -> Optional[str]:
    """
    The joint test of question 2 in one sentence.

    Parameters
    ----------
    mid_final, far_mid : dict or None
        The comparisons of the last weeks against 3 to 8 weeks before and of 3 to 8 weeks against farther.

    Returns
    -------
    str or None
        «Con todas las parejas juntas: un 23 % menos en las tres últimas semanas que entre 3 y 8 semanas antes (...), y
        un 6 % menos hasta tres semanas antes que lejos (...).» with the clauses that have ``pooled``; ``None`` without
        any.
    """
    clauses = [_pooled_clause(mid_final, 'en las tres últimas semanas que entre 3 y 8 semanas antes'),
               _pooled_clause(far_mid, 'hasta tres semanas antes que lejos')]
    clauses = [clause for clause in clauses if clause]

    if not clauses:
        return None

    return 'Con todas las parejas juntas: {}.'.format(', y '.join(clauses))


def _robustness(summary: Optional[dict]) -> dict:
    """The ``robustness`` block of the ``summary`` part (``{}`` when missing)."""
    return _get(summary, 'robustness') or {}


def _default_min_rows(summary: Optional[dict]):
    """The ``min_rows`` of the default cell of the robustness grid (the first of the list without a default row)."""
    block = _robustness(summary)

    for row in block.get('rows') or []:
        if row.get('default'):
            return row.get('min_rows')

    return (block.get('min_rows') or [None])[0]


def _joined_ranges(rows: list, key: str) -> Optional[str]:
    """«entre el 23 y el 27 %» over the ``change`` of the rows of a comparison, ``None`` without any."""
    values = [_whole(row['change']) for row in rows if row.get('comparison') == key and _is_num(row.get('change'))]

    if not values:
        return None

    low, high = min(values), max(values)

    return 'en torno al {} %'.format(fmt_int(low)) if low == high else 'entre el {} y el {} %'.format(
        fmt_int(low), fmt_int(high))


def robustness_text(summary: Optional[dict]) -> Optional[str]:
    """
    The sentence that opens the robustness note.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part, with its ``robustness`` block.

    Returns
    -------
    str or None
        The range of the joint fall at the end and in the last three weeks over the grid of days between polls (at the
        default minimum of comparisons per pair of houses), the lowest and highest count of elections that get closer
        (by the fraction k/n, ties to the fewer days) and the warning about the larger minimum; ``None`` without any
        joint change.
    """
    block = _robustness(summary)
    minimum = _default_min_rows(summary)
    rows = [row for row in block.get('rows') or [] if row.get('min_rows') == minimum]
    far = _joined_ranges(rows, 'far_final')
    mid = _joined_ranges(rows, 'mid_final')

    if far is None or mid is None:
        return None

    days = [fmt_int(day) for day in block.get('pair_days') or []]
    counts = [row for row in rows if row.get('comparison') == 'far_final' and row.get('n')]
    listed = ', '.join(days[:-1]) + ' o ' + days[-1] if len(days) > 1 else ''.join(days)
    text = ('Con {} días entre los dos sondeos, la bajada conjunta al final queda {}, y en las tres últimas '
            'semanas {}').format(listed, far, mid)

    if counts:
        low = min(counts, key=lambda row: (row['k'] / row['n'], row['pair_days']))
        high = min(counts, key=lambda row: (-row['k'] / row['n'], row['pair_days']))
        cell = '{} de {} con {}'

        if low is high:
            text += '; el recuento de elecciones que se acercan al final es de ' + cell.format(
                fmt_int(low['k']), fmt_int(low['n']), '{} días'.format(fmt_int(low['pair_days'])))
        else:
            text += '; el recuento de elecciones que se acercan al final va de {} días a {}'.format(
                cell.format(fmt_int(low['k']), fmt_int(low['n']), fmt_int(low['pair_days'])),
                cell.format(fmt_int(high['k']), fmt_int(high['n']), fmt_int(high['pair_days'])))

    text += '.'
    larger = [value for value in block.get('min_rows') or [] if _is_num(value) and value != minimum]

    if larger:
        text += (' Con un mínimo de {} comparaciones por pareja de casas (dos parejas de sondeos) quedan pocas '
                 'elecciones: la tabla las enseña.').format(fmt_int(max(larger)))

    return text


def _robustness_cell(row: Optional[dict]) -> dict:
    """The three figures of a robustness cell of one comparison: dashes without elections."""
    if not row or not row.get('n'):
        return {'count': DASH, 'change': DASH, 'chance': DASH}

    return {'count': '{} de {}'.format(fmt_int(row.get('k')), fmt_int(row.get('n'))),
            'change': percent(row.get('change')), 'chance': chance_text(row.get('p')) or DASH}


def robustness_rows(summary: Optional[dict]) -> list:
    """
    The rows of the robustness table: one per cell of the grid of days between polls and minimum of comparisons.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part, with its ``robustness`` block.

    Returns
    -------
    list of dict
        ``{'pair_days', 'min_rows', 'default', 'far_final': {'count', 'change', 'chance'}, 'mid_final': {...}}``, the
        default cell first and not repeated in the grid, then by days and minimum; ``[]`` without rows.
    """
    block = _robustness(summary)
    rows = block.get('rows') or []

    if not rows:
        return []

    cells = {}

    for row in rows:
        cells.setdefault((row.get('pair_days'), row.get('min_rows')), {})[row.get('comparison')] = row

    keys = [(days, minimum) for days in block.get('pair_days') or [] for minimum in block.get('min_rows') or []]
    keys += [key for key in cells if key not in keys]
    default = [key for key in cells if any(row.get('default') for row in cells[key].values())][:1]
    ordered = default + [key for key in keys if key in cells and key not in default]

    return [{'pair_days': days, 'min_rows': minimum, 'default': (days, minimum) in default,
             'far_final': _robustness_cell(cells[(days, minimum)].get('far_final')),
             'mid_final': _robustness_cell(cells[(days, minimum)].get('mid_final'))} for days, minimum in ordered]
