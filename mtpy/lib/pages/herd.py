"""
View models of the page of the herd study (``/analisis/rebano``): the verdicts, the three KPI rows, the three tests of
chance, the CIS table, the houses of the window in progress, the house × cycle matrix, the conditioned sentences and
the ``initial`` island with what the two charts draw.

``analisis.herd_context`` assembles them from the published parts; each function takes the data it reads and returns
what the template renders, with ``None`` or an empty structure for what the parts do not have, never an error.
"""
from datetime import date, datetime, timedelta
from typing import Optional

from .cera import MONTHS_LONG, _get
from .common import days_left as days_to
from .filters import DASH, _madrid_tz, fmt_int, fmt_num
from .lean import house_href

HERD = 'rebano'
QUESTIONS = ('¿Se acercan los sondeos cuando llega la elección?',
             '¿Más de lo que cabría esperar sólo por el azar de sus muestras?',
             '¿Quién va en rebaño en {}?')
# Dispersion at or below which the polls differ «as if they all drew their samples from the same urn» (the threshold
# of the subtitle in ``publish.herd.URN_RATIO``; the pages do not import ``publish``)
URN_RATIO = 1.1
# Days between the start of the LOREG ban on polls and the election
VEDA_DAYS = 6
# Upper bounds (exclusive) of the red shade classes of the matrix; from 0,9 to ``NEUTRAL_TOP`` inclusive the neutral
# ``h4`` (the grey band of ``matrix.js``, ``neutral = [0.9, 1.2]``), up to ``SHADE_TOP`` inclusive ``h5``, above ``h6``
SHADES = ((0.5, 'h1'), (0.7, 'h2'), (0.9, 'h3'))
NEUTRAL_TOP = 1.2
SHADE_TOP = 1.6
# Dispersion of the rest of the cycle above which «each house has its own recipe», and the top of the final ratios
# that «still differ, but little»
COOKING_RATIO, CLOSE_RATIO = 1.2, 1.3
# Defaults of the windows, of the parties measured and of the days of the local consensus when ``meta.params`` lacks
# them (spec D4)
WINDOW_DAYS, FINAL_DAYS, MIN_POLLS = 56, 21, 3
N_SERIES, LOCAL_WINDOW = 4, 10
LUCK = '1 = lo que cabría esperar sólo por el azar de las muestras'


def _is_num(x) -> bool:
    """Whether ``x`` is a real number (not a bool)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _num(x, digits: int = 2) -> str:
    """A ratio with two decimals (``fmt_num``; a dash when missing)."""
    return fmt_num(x, digits)


def names(items: list) -> str:
    """A list in prose: «A, B y C» (``''`` when empty)."""
    return ', '.join(items[:-1]) + ' y ' + items[-1] if len(items) > 1 else ''.join(items)


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


def today() -> date:
    """Today in Europe/Madrid, the day the texts of the page count from (the tests replace it)."""
    return datetime.now(_madrid_tz()).date()


def flags(summary: Optional[dict], meta: Optional[dict]) -> dict:
    """
    The conditions and the dates of the texts of the page (spec D6).

    The days to the election count from ``today()``, not from the ``as_of`` of the run: inside the veda the forecast
    is refused and the daily republication keeps the last run, so its ``days_left`` would stay frozen.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    meta : dict or None
        The ``meta`` part.

    Returns
    -------
    dict
        ``has_run`` (the summary has the cycle in progress), ``like_final`` (its whole cycle already as low as the
        final weeks of the past ones), ``all_narrowed`` (every past cycle narrowed), ``final_like_chance``
        (``mean_final <= URN_RATIO``), ``year`` (of the election in progress, ``None`` without it), ``election_day``
        («29 de noviembre»), ``election_short`` («29-N»), ``days_left`` (from ``today()`` to the election, ``None``
        without its date), ``veda_date`` (the election minus ``VEDA_DAYS``, «23 de noviembre»), ``after_veda``
        (``days_left <= VEDA_DAYS``), ``held`` (``days_left <= 0``: the election is held, its scrutiny pending),
        ``since_year``, ``window_weeks``, ``final_weeks``, ``min_polls``, ``n_series``, ``local_window``,
        ``window_start`` (the first day of the final window of the cycle in progress) and the thresholds of the texts
        (``shade_top``, ``cooking_ratio``, ``close_ratio``).
    """
    current = _get(summary, 'current')
    election = _get(meta, 'election') or {}
    params = _get(meta, 'params') or {}
    day = _date(election.get('date'))
    days_left = days_to(election.get('date'), today())
    n, k = _get(summary, 'n_cycles'), _get(summary, 'k_narrowed')
    mean_final = _get(summary, 'mean_final')
    window = params.get('window_days') or WINDOW_DAYS

    return {
        'has_run': bool(current),
        'like_final': bool(_get(current, 'like_final')),
        'all_narrowed': bool(n) and k == n,
        'final_like_chance': _is_num(mean_final) and mean_final <= URN_RATIO,
        'year': str(day.year) if day else None,
        'election_day': day_month(election.get('date')),
        'election_short': short_day(election.get('date')),
        'days_left': days_left,
        'veda_date': day_month((day - timedelta(days=VEDA_DAYS)).isoformat()) if day else None,
        'after_veda': _is_num(days_left) and days_left <= VEDA_DAYS,
        'held': _is_num(days_left) and days_left <= 0,
        'since_year': (params.get('since') or '')[:4] or None,
        'window_weeks': window // 7,
        'final_weeks': (params.get('final_days') or FINAL_DAYS) // 7,
        'min_polls': params.get('min_polls') or MIN_POLLS,
        'n_series': params.get('n_series') or N_SERIES,
        'local_window': params.get('local_window') or LOCAL_WINDOW,
        'window_start': day_month((day - timedelta(days=window)).isoformat()) if day else None,
        'shade_top': SHADE_TOP, 'cooking_ratio': COOKING_RATIO, 'close_ratio': CLOSE_RATIO,
    }


def verdict_rows(summary: Optional[dict], meta: Optional[dict] = None) -> list:
    """
    The three questions of the study with their verdict (spec D6).

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    meta : dict or None, optional
        The ``meta`` part, for the year of the third question («este año» without an election).

    Returns
    -------
    list of dict
        ``{'label', 'tone', 'question'}`` per question: P1 «SÍ» (``q1``) when every past cycle narrowed, else
        «EN k DE n» (``q2``); P2 «SÍ» (``q1``) when every cycle fell below 1, «NO» (``q3``) when none did, else «EN k
        DE n» (``q2``); P3 (``qb``) the houses below 1 of those measurable in the window in progress, «k DE n
        MEDIBLE(S)», so a lone house is not read as the only one in the herd; «NINGUNA» without them or without run and
        «AÚN SIN DATOS» when no house reaches ``min_polls`` polls yet.
    """
    n = _get(summary, 'n_cycles') or 0
    narrowed, below = _get(summary, 'k_narrowed') or 0, _get(summary, 'k_below_one') or 0
    current = bool(_get(summary, 'current'))
    herd = (_get(summary, 'houses_now', 'k') or 0) if current else 0
    measurable = (_get(summary, 'houses_now', 'n') or 0) if current else 0
    first = ('SÍ', 'q1') if n and narrowed == n else ('EN {} DE {}'.format(narrowed, n), 'q2')

    if n and below == n:
        second = ('SÍ', 'q1')
    elif below == 0:
        second = ('NO', 'q3')
    else:
        second = ('EN {} DE {}'.format(below, n), 'q2')

    if current and measurable == 0:
        third = ('AÚN SIN DATOS', 'qb')
    elif herd == 0:
        third = ('NINGUNA', 'qb')
    else:
        third = ('{} DE {} MEDIBLE{}'.format(herd, measurable, '' if measurable == 1 else 'S'), 'qb')
    year = flags(summary, meta)['year'] or 'este año'
    questions = QUESTIONS[:2] + (QUESTIONS[2].format(year),)

    return [{'label': label, 'tone': tone, 'question': question}
            for (label, tone), question in zip((first, second, third), questions)]


def past_rows(cycles: list) -> list:
    """The rows of the past cycles of the ``cycles`` part, in its order."""
    return [row for row in cycles if not row.get('current')]


def _last_house(houses: list, cycles: list) -> Optional[dict]:
    """The house with the lowest ``h`` in the final window of the last past cycle (the CIS apart), or ``None``."""
    past = past_rows(cycles)
    rows = [row for row in houses if past and row.get('event_date') == past[-1].get('event_date')
            and not row.get('cis') and _is_num(row.get('h'))]

    return min(rows, key=lambda row: row['h']) if rows else None


def kpis(summary: Optional[dict], meta: Optional[dict], cycles: list = (), houses: list = ()) -> list:
    """
    The three rows of the ``m.kpis`` macro.

    Parameters
    ----------
    summary, meta : dict or None
        The ``summary`` and ``meta`` parts.
    cycles, houses : list of dict, optional
        The rows of the ``cycles`` and ``houses`` parts: without a run, the second and third KPI speak of the final
        window of the last past cycle.

    Returns
    -------
    list of dict
        ``{'label', 'value', 'unit', 'support', 'ok', 'color'}`` per KPI.
    """
    f = flags(summary, meta)
    row = {'unit': '', 'ok': False, 'color': None}
    n = _get(summary, 'n_cycles')
    count = '{} de {}'.format(fmt_int(_get(summary, 'k_narrowed')), fmt_int(n))
    first = dict(row, label='Ciclos en que los sondeos se acercaron en la recta final', value=count,
                 support='la dispersión media de los sondeos pasa de {} en el resto del ciclo a {} en las últimas {} '
                         'semanas y a {} en las últimas {} · {}'.format(
                             _num(_get(summary, 'mean_before')), _num(_get(summary, 'mean_last')), f['window_weeks'],
                             _num(_get(summary, 'mean_final')), f['final_weeks'], LUCK))
    current = _get(summary, 'current')

    if not current:
        past = past_rows(list(cycles))
        last = past[-1] if past else {}
        house = _last_house(list(houses), list(cycles))
        label = last.get('label') or '–'
        # «en rebaño» only when the lowest house of the window went below 1
        third = ('Casa más en rebaño en las últimas {} semanas de {}' if house and house['h'] < 1 else
                 'Casa que menos oscila en las últimas {} semanas de {}').format(f['window_weeks'], label)

        return [first,
                dict(row, label='Dispersión de {}, últimas {} semanas'.format(label, f['window_weeks']),
                     value=_num(_get(last, 'last', 'ratio')),
                     support='en el resto del ciclo, {} · {}'.format(_num(_get(last, 'before', 'ratio')), LUCK)),
                dict(row, label=third, value=_num(house['h']) if house else DASH,
                     support='{}, con {} sondeos con muestra · {}'.format(house['pollster'], fmt_int(house.get('k')),
                                                                          LUCK)
                     if house else 'ninguna casa con {} sondeos con muestra en esa ventana'.format(f['min_polls']))]

    # once the election is held the days no longer count: the figure is the one of the close of the polls
    when = 'al cierre de los sondeos' if f['held'] else 'a {} días del {}'.format(fmt_int(f['days_left']),
                                                                                  f['election_day'] or '–')
    second = '{} · en el resto del ciclo, el {}.º más bajo de los {}'.format(
        when, fmt_int(current.get('rank_before')), fmt_int((n or 0) + 1))
    now = _get(summary, 'houses_now') or {}
    herd = ['{} {}'.format(house.get('pollster'), _num(house.get('h'))) for house in now.get('herd') or []]
    cis = _get(summary, 'cis', 'current', 'ratio')

    if not herd and now.get('n'):
        herd = ['ninguna de las casas con {} sondeos con muestra oscila menos de lo que cabría esperar sólo por el '
                'azar'.format(f['min_polls'])]
    elif not herd:
        herd = ['ninguna casa tiene aún {} sondeos con muestra en la ventana'.format(f['min_polls'])]

    pending = len(now.get('pending') or [])

    if now.get('n') and pending:
        herd.append('otra casa aún sin sondeos suficientes' if pending == 1 else
                    'otras {} casas aún sin sondeos suficientes'.format(fmt_int(pending)))

    if _is_num(cis):
        herd.append('el CIS, aparte: ' + _num(cis, 1))

    return [first,
            dict(row, label=('Dispersión de los sondeos de {}' if f['held'] else
                             'Dispersión de los sondeos de {}, hoy').format(f['year'] or current.get('label') or '–'),
                 value=_num(_get(current, 'cycle', 'ratio')),
                 support=second + (' · ya como una recta final' if f['like_final'] else '')),
            dict(row, label='Casas en rebaño en las últimas {} semanas'.format(f['window_weeks']),
                 value='{} de {}'.format(fmt_int(now.get('k') or 0), fmt_int(now.get('n') or 0)),
                 support=' · '.join(herd))]


def final_split(summary: Optional[dict], cycles: list) -> dict:
    """
    The past cycles by their dispersion in the last weeks: below 1 (``summary.below_one``) and the others.

    Returns
    -------
    dict
        ``below`` and ``others`` (``[{'label', 'ratio'}]`` in the order of ``cycles``) and ``lo``/``hi`` (the range
        of the others' ratios, ``None`` without them).
    """
    below = set(_get(summary, 'below_one') or [])
    rows = [{'label': row.get('label'), 'ratio': _get(row, 'final', 'ratio')} for row in past_rows(cycles)
            if _is_num(_get(row, 'final', 'ratio'))]
    others = [row for row in rows if row['label'] not in below]
    ratios = [row['ratio'] for row in others]

    return {'below': [row for row in rows if row['label'] in below], 'others': others,
            'lo': min(ratios) if ratios else None, 'hi': max(ratios) if ratios else None}


def tests(summary: Optional[dict], cycles: list = (), meta: Optional[dict] = None) -> list:
    """
    The three boxes of question 2.

    Parameters
    ----------
    summary : dict or None
        The ``summary`` part.
    cycles : list of dict, optional
        The rows of the ``cycles`` part, for the ratios of the cycles below 1.
    meta : dict or None, optional
        The ``meta`` part, for the length of the last weeks.

    Returns
    -------
    list of dict
        ``{'key', 'label', 'value', 'note', 'text', 'hi'}`` for ``below``, ``mean`` and ``sign``; only the first is
        highlighted; ``value`` is a dash and ``text`` ``None`` when the part lacks the figure.
    """
    n, k = _get(summary, 'n_cycles'), _get(summary, 'k_below_one')
    before, final = _get(summary, 'mean_before'), _get(summary, 'mean_final')
    below = final_split(summary, list(cycles))['below']

    if k:
        first = '{}: menos variación de la que cabría esperar sólo por el azar de las muestras.'.format(
            names(['{} ({})'.format(row['label'], _num(row['ratio'])) for row in below] or
                  list(_get(summary, 'below_one') or [])))
    else:
        first = ('En ninguno bajó de 1: los sondeos variaron al menos lo que cabría esperar por el azar de sus '
                 'muestras.')

    if not _is_num(final):
        second = None
    elif final <= URN_RATIO:
        second = ('Las diferencias entre casas, que en el resto del ciclo dejan la dispersión en {}, desaparecen en '
                  'la recta final: queda sólo el ruido de las muestras.').format(_num(before))
    elif _is_num(before) and final < before:
        second = ('Las diferencias entre casas se reducen en la recta final, desde {}, pero no desaparecen: siguen '
                  'por encima de lo que cabría esperar sólo por el azar de las muestras.').format(_num(before))
    else:
        second = 'Las diferencias entre casas no se reducen en la recta final.'

    sign = _get(summary, 'sign') or {}
    p = sign.get('p')
    odds = None if not _is_num(p) or p <= 0 else 'menos de 1 entre 1.000' if p < 1e-3 else \
        '1 entre ' + fmt_int(round(1 / p))
    final_weeks = flags(summary, meta)['final_weeks']

    return [
        {'key': 'below', 'label': '1 · Ciclos por debajo de 1 en las últimas {} semanas'.format(final_weeks),
         'value': '{} de {}'.format(fmt_int(k), fmt_int(n)), 'note': None, 'text': first if n else None, 'hi': True},
        {'key': 'mean', 'label': '2 · Media de los {} ciclos, últimas {} semanas'.format(fmt_int(n), final_weeks),
         'value': _num(final), 'note': 'desde ' + _num(before) if _is_num(before) else None, 'text': second,
         'hi': False},
        {'key': 'sign', 'label': '3 · Probabilidad de una caída así por azar', 'value': odds or DASH, 'note': None,
         'text': '{} de {} ciclos con la dispersión a la baja al entrar en la recta final: una moneda al aire lo '
                 'haría con una probabilidad de {}.'.format(fmt_int(sign.get('k')), fmt_int(sign.get('n')), odds)
         if odds else None, 'hi': False},
    ]


def cis_rows(cycles: list) -> list:
    """The table «El CIS, aparte»: ``{'label', 'current', 'k', 'ratio', 'own', 'cycle'}`` per cycle of the part."""
    return [{'label': row.get('label'), 'current': bool(row.get('current')), 'k': _get(row, 'cis', 'last', 'k'),
             'ratio': _get(row, 'cis', 'last', 'ratio'), 'own': _get(row, 'cis', 'own', 'h'),
             'cycle': _get(row, 'cis', 'cycle', 'ratio')} for row in cycles]


def _p_text(p) -> str:
    """A p value with two decimals, «< 0,01» below that."""
    return '< 0,01' if _is_num(p) and p < 0.005 else _num(p)


def houses_now_rows(summary: Optional[dict], houses: list) -> list:
    """
    The houses measured in the final window of the cycle in progress (the CIS apart), by ``h``.

    Returns
    -------
    list of dict
        The ``houses`` rows with ``href`` (profile), ``herd`` (``h < 1``), ``p_text`` and ``bar`` (the width of the
        bar in %, ``min(1, max(0, 1 − h)) · 100``); ``[]`` without a run.
    """
    if not _get(summary, 'current'):
        return []

    rows = sorted((row for row in houses if row.get('current') and not row.get('cis')),
                  key=lambda row: (not _is_num(row.get('h')), row.get('h') or 0))

    return [dict(row, href=house_href(row.get('pollster')), herd=_is_num(row.get('h')) and row['h'] < 1,
                 p_text=_p_text(row.get('p')),
                 bar=round(min(1, max(0, 1 - row['h'])) * 100) if _is_num(row.get('h')) else 0) for row in rows]


def pending_text(summary: Optional[dict]) -> Optional[str]:
    """The houses with too few polls in the window in progress, «GAD3 (2), 40dB (1)», or ``None``."""
    pending = _get(summary, 'houses_now', 'pending') or []

    return ', '.join('{} ({})'.format(row.get('pollster'), fmt_int(row.get('k'))) for row in pending) or None


def chance_text(summary: Optional[dict], houses: list) -> Optional[str]:
    """
    The caveat under the table of the cycle in progress when a house below 1 is not yet beyond chance (``p`` of 0,05
    or more): with three or four polls a low ``h`` still happens without herding.

    Returns
    -------
    str or None
        «Con 3 sondeos, el 0,43 de NC Report aún puede ser casualidad: sin rebaño, una oscilación tan baja saldría 17
        veces de cada 100.» for one house; «Con tan pocos sondeos, el rebaño de A y B aún puede ser casualidad (p-valor
        de 0,05 o más).» for several; ``None`` without such a house.
    """
    rows = [row for row in houses_now_rows(summary, houses) if row['herd'] and _is_num(row.get('p'))
            and row['p'] >= 0.05]

    if not rows:
        return None

    if len(rows) == 1:
        row = rows[0]
        return ('Con {} sondeos, el {} de {} aún puede ser casualidad: sin rebaño, una oscilación tan baja saldría {} '
                'veces de cada 100.').format(fmt_int(row.get('k')), _num(row['h']), row.get('pollster'),
                                             fmt_int(round(row['p'] * 100)))

    return ('Con tan pocos sondeos, el rebaño de {} aún puede ser casualidad (p-valor de 0,05 o más).'
            .format(names([row.get('pollster') for row in rows])))


def cycle_ratio_text(houses: list, summary: Optional[dict]) -> Optional[str]:
    """
    The note under the table of the cycle in progress: the same measure over the whole cycle (``h_cycle``, the one of
    /casas) of its first three houses that have it.

    Returns
    -------
    str or None
        «En el ciclo entero (desde 2023), la misma medida da SocioMétrica 0,78, NC Report 0,58 y Sigma Dos 0,88: la
        recta final acentúa lo que ya se veía.» (the closing clause only when every one is lower in the final window);
        ``None`` without a run or without a house with ``h_cycle``.
    """
    rows = [row for row in houses_now_rows(summary, houses) if _is_num(row.get('h_cycle'))][:3]

    if not rows:
        return None

    past = sorted(row.get('event_date') or '' for row in houses if not row.get('current'))
    since = ' (desde {})'.format(past[-1][:4]) if past and past[-1] else ''
    text = 'En el ciclo entero{}, la misma medida da {}'.format(
        since, names(['{} {}'.format(row['pollster'], _num(row['h_cycle'])) for row in rows]))

    if all(_is_num(row.get('h')) and row['h'] < row['h_cycle'] for row in rows):
        text += ': la recta final acentúa lo que ya se veía'

    return text + '.'


def shade_class(h) -> str:
    """
    The class of a cell of the matrix: ``h1``..``h3`` below 0,9 (``SHADES``), the neutral ``h4`` from 0,9 to 1,2 both
    included (``NEUTRAL_TOP``, as the grey band of ``matrix.js``), ``h5`` up to ``SHADE_TOP`` included, ``h6`` above;
    ``z`` without ``h``.
    """
    if not _is_num(h):
        return 'z'

    if h <= NEUTRAL_TOP:
        return next((name for top, name in SHADES if h < top), 'h4')

    return 'h5' if h <= SHADE_TOP else 'h6'


def _below_counts(houses: list) -> dict:
    """``{pollster: (cycles with h < 1, cycles measured)}`` of the houses but the CIS."""
    counts = {}

    for row in houses:
        if row.get('cis') or not row.get('pollster'):
            continue

        k, n = counts.get(row['pollster'], (0, 0))
        counts[row['pollster']] = (k + (_is_num(row.get('h')) and row['h'] < 1), n + 1)

    return counts


def matrix(houses: list, cycles: list) -> dict:
    """
    The house × cycle matrix of the final windows, for ``matrix.js`` and the table without JavaScript.

    Returns
    -------
    dict
        ``events`` (``{'label', 'short', 'current'}`` per cycle, in the order of ``cycles``), ``houses`` (the houses of
        the cycle in progress first by ``h``, then by cycles below 1 descending and name, the CIS last) and ``cells``
        (``[x, y, h, k, p]`` per row with ``h``: the column, the row, the measure, the polls and the p value, for the
        tooltip).
    """
    columns = {row.get('event_date'): x for x, row in enumerate(cycles)}
    counts = _below_counts(houses)
    now = sorted((row for row in houses if row.get('current') and not row.get('cis') and _is_num(row.get('h'))),
                 key=lambda row: row['h'])
    order = [row['pollster'] for row in now]
    order += sorted((name for name in counts if name not in order),
                    key=lambda name: (-counts[name][0], name.casefold()))
    order += sorted({row['pollster'] for row in houses if row.get('cis') and row.get('pollster')} - set(order))
    y_of = {name: y for y, name in enumerate(order)}

    return {
        'events': [{'label': row.get('label'), 'short': row.get('label'), 'current': bool(row.get('current'))}
                   for row in cycles],
        'houses': order,
        'cells': [[columns[row['event_date']], y_of[row['pollster']], row['h'], row.get('k'), row.get('p')]
                  for row in houses
                  if row.get('event_date') in columns and row.get('pollster') in y_of and _is_num(row.get('h'))],
    }


def matrix_rows(data: dict) -> list:
    """The rows of the matrix table: ``{'name', 'href', 'cells': [{'h', 'shade'}]}`` with a cell per column."""
    grid = {(cell[0], cell[1]): cell[2] for cell in data.get('cells') or []}
    width = len(data.get('events') or [])

    return [{'name': name, 'href': house_href(name),
             'cells': [{'h': grid.get((x, y)), 'shade': shade_class(grid.get((x, y)))} for x in range(width)]}
            for y, name in enumerate(data.get('houses') or [])]


def repeaters(houses: list) -> list:
    """
    The houses that went with the herd in two or more final windows, by count descending and name, in prose: the first
    «NC Report ha ido en rebaño en la recta final de 4 de sus 5 ciclos» (or «de los 5 ciclos en que la medimos»), the
    others «Celeste Tel en 4 de 5»; the CIS apart.
    """
    counts = sorted(((name, k, n) for name, (k, n) in _below_counts(houses).items() if k >= 2),
                    key=lambda item: (-item[1], item[0].casefold()))
    texts = []

    for i, (name, k, n) in enumerate(counts):
        if i:
            texts.append('{} en {} de {}'.format(name, k, n))
        elif k == n:
            texts.append('{} ha ido en rebaño en la recta final de los {} ciclos en que la medimos'.format(name, n))
        else:
            texts.append('{} ha ido en rebaño en la recta final de {} de sus {} ciclos'.format(name, k, n))

    return texts


def matrix_notes(houses: list) -> dict:
    """
    The two conditions of the closing sentences of question 3.

    Returns
    -------
    dict
        ``none_escaped``: there are houses with three or more cycles and every one went below 1 in some of them;
        ``cis_alone``: the CIS has a cell above ``SHADE_TOP`` and no other house has one.
    """
    veterans = [k for k, n in _below_counts(houses).values() if n >= 3]
    high = [row for row in houses if _is_num(row.get('h')) and row['h'] > SHADE_TOP]

    return {'none_escaped': bool(veterans) and all(veterans),
            'cis_alone': bool(high) and all(row.get('cis') for row in high)}


def accuracy_text(summary: Optional[dict]) -> Optional[str]:
    """
    How the houses that went with the herd erred against the others, over the past cycles with a result
    (``summary.accuracy``); ``None`` without it.
    """
    acc = _get(summary, 'accuracy') or {}
    herd, rest = acc.get('herd') or {}, acc.get('rest') or {}

    if not herd or not rest:
        return None

    text = ('Las casas que iban en rebaño en la recta final erraron {} puntos de media en el voto de su último sondeo '
            'y {} en el margen entre bloques; las demás, {} y {}. Cuenta cada casa en cada ciclo con resultado: {} en '
            'total, {} en rebaño y {} no.').format(
        _num(herd.get('error_avg'), 1), _num(herd.get('error_blocks'), 1), _num(rest.get('error_avg'), 1),
        _num(rest.get('error_blocks'), 1), fmt_int(acc.get('n')), fmt_int(herd.get('k')), fmt_int(rest.get('k')))
    pairs = [(herd.get(key), rest.get(key)) for key in ('error_avg', 'error_blocks')]

    if not all(_is_num(a) and _is_num(b) for a, b in pairs):
        return text

    if all(a < b for a, b in pairs):
        return text + ' Por ahora, ir en rebaño no les ha hecho fallar más.'

    if all(a > b for a, b in pairs):
        return text + ' Por ahora, las que iban en rebaño fallaron más.'

    return text + ' Ninguno de los dos grupos acierta más en las dos medidas.'


def result_text(houses: list, cycles: list, meta: Optional[dict]) -> Optional[str]:
    """
    The line of the election of the study once it has a result: the houses of its final window with the error of
    their last poll, split by ``h < 1``. The election is ``meta.election`` or, without it, the newest cycle.

    Returns
    -------
    str or None
        The sentence, or ``None`` while the election has no past-cycle rows with errors or a group is empty.
    """
    target = _get(meta, 'election', 'date') or max((row.get('event_date') or '' for row in cycles), default=None)
    rows = [row for row in houses if row.get('event_date') == target and not row.get('current')
            and not row.get('cis') and _is_num(row.get('h')) and _is_num(row.get('last_error_avg'))]
    groups = [[row for row in rows if row['h'] < 1], [row for row in rows if row['h'] >= 1]]

    if not all(groups):
        return None

    def mean(group: list, key: str) -> Optional[float]:
        values = [abs(row[key]) if key == 'last_error_blocks' else row[key] for row in group if _is_num(row.get(key))]
        return sum(values) / len(values) if values else None

    return ('En {}, las {} casas que iban en rebaño erraron {} puntos de media en el voto de su último sondeo y {} en '
            'el margen entre bloques; las otras {}, {} y {}.').format(
        rows[0].get('label'), len(groups[0]), _num(mean(groups[0], 'last_error_avg'), 1),
        _num(mean(groups[0], 'last_error_blocks'), 1), len(groups[1]), _num(mean(groups[1], 'last_error_avg'), 1),
        _num(mean(groups[1], 'last_error_blocks'), 1))


def biggest_drops(cycles: list, count: int = 2) -> list:
    """The past cycles with the largest falls of the dispersion into the final window (``change < 0``), largest
    first."""
    rows = [row for row in past_rows(cycles) if _is_num(row.get('change')) and row['change'] < 0]

    return sorted(rows, key=lambda row: row['change'])[:count]


def island(parts: dict) -> dict:
    """
    What the two charts of the page draw (the page script reads it).

    Parameters
    ----------
    parts : dict
        The published parts by name (``{}`` for a missing one).

    Returns
    -------
    dict
        ``weeks`` (the part as is) and ``matrix`` (``matrix`` of the ``houses`` and ``cycles`` rows).
    """
    return {'weeks': parts.get('weeks') or {},
            'matrix': matrix(_get(parts.get('houses'), 'rows') or [], _get(parts.get('cycles'), 'rows') or [])}
