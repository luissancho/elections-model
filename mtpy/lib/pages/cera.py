"""
View models of the cards of the CERA study page (``/analisis/voto-cera``), one per question, and their helpers.

``analisis.cera_context`` assembles them; each takes the data of the parts it reads and returns what its card
renders, with ``None`` for what the parts do not have (the templates guard it), never an error.
"""
from typing import Optional

from .filters import fmt_num, fmt_pct

# Question 1: the countries of its table (the most additions after the law), the provinces shown before «Ver las
# 52 provincias», the countries its summary quotes (by their INE name) and the p-value below which it says «< »
COUNTRIES_SHOWN = 6
PROVINCES_SHOWN = 8
LATAM_QUOTED = ('ARGENTINA', 'MÉXICO')
EXILE_QUOTED = 'CUBA'
P_FLOOR = 0.001
# Level under which the change of slope of the trend-break regression is called confirmed
SIGNIFICANCE = 0.05
# A pace (additions after / before) from which the cell is green (it doubles) and below which it is red (it slows)
PACE_FAST = 2
PACE_SLOW = 1
# Words that stay in lower case in the country names of the INE (``ESTADOS UNIDOS DE AMÉRICA``)
NAME_PARTICLES = ('de', 'del', 'la', 'las', 'los', 'y', 'e')
# Questions 2 and 3: the two blocks the texts compare, Madrid in the rows of ``prediction.last_seat`` (its INE code),
# the distance from 2 of the ratio of the participations (rest / Latin American) still called «la mitad», the share
# of the new vote that goes to the left with a uniform split (the observed column of the sensitivity grids) and
# the two census scenarios of the prediction with the labels of the selector
LEFT, RIGHT = 'Izquierda', 'Derecha'
MADRID = 28
HALF_TOLERANCE = 0.25
UNIFORM_SPLIT = 0.5
SCENARIO_LABELS = {'convocatoria': 'Censo de la convocatoria', 'sin_litigio': 'Sin las inscripciones en litigio'}
SCENARIOS = tuple(SCENARIO_LABELS)
# Rows of the table of each scenario of question 3 (label, key of the 2023 value, key of the projection)
PREDICTION_ROWS = (('Votantes CERA', 'voters_2023', 'voters_projected'),
                   ('Votos a candidaturas', 'votes_2023', 'votes_projected'))
BLOCK_LABELS = {LEFT: 'Escaños izquierda (PSOE + SUMAR)', RIGHT: 'Escaños derecha (PP + VOX)'}


def _get(data: Optional[dict], *keys):
    """
    Value of a chain of keys of nested dicts.

    Parameters
    ----------
    data : dict or None
        Outer dict.
    *keys : str
        Keys, from the outer dict inwards.

    Returns
    -------
    object
        The value, or ``None`` when a level is missing or is not a dict.
    """
    for key in keys:
        if not isinstance(data, dict):
            return None

        data = data.get(key)

    return data


def _months(start: Optional[str], end: Optional[str]) -> Optional[int]:
    """
    Months between two first-of-month dates.

    Parameters
    ----------
    start, end : str or None
        ISO dates ``YYYY-MM-DD``.

    Returns
    -------
    int or None
        ``end - start`` in months, or ``None`` without two valid dates.
    """
    try:
        y1, m1 = (int(piece) for piece in start.split('-')[:2])
        y2, m2 = (int(piece) for piece in end.split('-')[:2])
    except (AttributeError, ValueError):
        return None

    return 12 * (y2 - y1) + m2 - m1


def _country_name(name: Optional[str]) -> Optional[str]:
    """
    Country name of the INE in the case of the prose (``'ESTADOS UNIDOS DE AMÉRICA'`` → ``'Estados Unidos de
    América'``); a name already in mixed case stays as it is.

    Parameters
    ----------
    name : str or None
        Name of the country.

    Returns
    -------
    str or None
        The name, words capitalised except ``NAME_PARTICLES`` after the first.
    """
    if not name or not name.isupper():
        return name

    words = name.lower().split()

    return ' '.join(word if index and word in NAME_PARTICLES else word.capitalize() for index, word in enumerate(words))


def _p_value(p: Optional[float]) -> Optional[str]:
    """
    A p-value as the texts quote it: ``'< 0,001'`` under ``P_FLOOR``, else ``'= 0,042'``.

    Parameters
    ----------
    p : float or None
        The p-value.

    Returns
    -------
    str or None
        The comparison and the value, or ``None`` without one.
    """
    if p is None:
        return None

    return '< ' + fmt_num(P_FLOOR, 3) if p < P_FLOOR else '= ' + fmt_num(p, 3)


def _pace_tone(ratio: Optional[float]) -> Optional[str]:
    """
    Colour class of a pace of additions (after / before) in the table of countries.

    Parameters
    ----------
    ratio : float or None
        The pace.

    Returns
    -------
    str or None
        ``'pos'`` from ``PACE_FAST`` (it doubles), ``'neg'`` under ``PACE_SLOW`` (it slows), else ``None``.
    """
    if ratio is None:
        return None

    return 'pos' if ratio >= PACE_FAST else 'neg' if ratio < PACE_SLOW else None


def _question_1(census: dict, figures: dict) -> dict:
    """
    View model of the card of question 1 (``#p1``): its two tables, the texts of its charts and its summary.

    Parameters
    ----------
    census : dict
        Data of the ``census`` part.
    figures : dict
        ``_figures`` of the page (``p_slope``).

    Returns
    -------
    dict
        - ``countries``: the ``COUNTRIES_SHOWN`` countries with the most additions in the second period of the
          article, ``{'name', 'g1', 'g2', 'ratio', 'tone'}`` (``tone`` ``'pos'`` from ``PACE_FAST``, ``'neg'``
          under ``PACE_SLOW``, else ``None``).
        - ``provinces``: every row of ``election_provinces`` with the census at the start of the second period
          (``c2_start`` of its row of ``provinces``) as ``start``, ``election``, ``growth = election - start``,
          ``pct`` (in % of ``start``) and ``shown`` (the first ``PROVINCES_SHOWN``), sorted by ``growth``, the
          largest first; ``provinces_hidden`` counts the rest.
        - ``start_label``: the month of that start as ``'MM-YYYY'`` (the header of the table).
        - ``periods``: the two periods of the article (``census.periods.article``), ``months`` the length of each
          and ``equal`` whether they last the same.
        - ``series``: ``{'from', 'to', 'first', 'last'}``, the dates and values at both ends of the monthly series.
        - ``latam``: the rows of ``LATAM_QUOTED`` and ``exile`` the one of ``EXILE_QUOTED`` (or ``None``), among
          all the countries, for the summary; ``exile_slow`` whether the pace of ``exile`` is under ``PACE_FAST``
          (the summary says it «apenas se mueve» only then).
        - ``p_slope``: ``_p_value`` of the change of slope, and ``slope_significant`` whether it is under
          ``SIGNIFICANCE``.
    """
    countries = [dict(row, name=_country_name(row.get('name'))) for row in census.get('countries') or []]
    by_name = {(row['name'] or '').upper(): row for row in countries}
    starts = {row.get('region_id'): row.get('c2_start') for row in census.get('provinces') or []}
    provinces = []

    for row in census.get('election_provinces') or []:
        start, election = starts.get(row.get('region_id')), row.get('election')
        growth = election - start if start is not None and election is not None else None
        provinces.append({'name': row.get('name'), 'start': start, 'election': election, 'growth': growth,
                          'pct': 100 * growth / start if growth is not None and start else None})

    provinces.sort(key=lambda row: (row['growth'] is None, -(row['growth'] or 0)))

    for index, row in enumerate(provinces):
        row['shown'] = index < PROVINCES_SHOWN

    periods = _get(census, 'periods', 'article') or []
    months = [_months(*period) for period in periods if len(period) == 2]
    series = census.get('series') or {}
    dates, values = series.get('dates') or [None], series.get('values') or [None]
    start = periods[1][0] if len(periods) > 1 and periods[1] else None

    return {
        'countries': [{'name': row['name'], 'g1': row.get('g1'), 'g2': row.get('g2'), 'ratio': row.get('ratio'),
                       'tone': _pace_tone(row.get('ratio'))}
                      for row in sorted(countries, key=lambda row: -(row.get('g2') or 0))[:COUNTRIES_SHOWN]],
        'provinces': provinces,
        'provinces_hidden': max(len(provinces) - PROVINCES_SHOWN, 0),
        'start_label': '{}-{}'.format(start[5:7], start[:4]) if start else None,
        'periods': periods,
        'months': months,
        'equal': len(set(months)) == 1 and None not in months,
        'series': {'from': dates[0], 'to': dates[-1], 'first': values[0], 'last': values[-1]},
        'latam': [by_name[name] for name in LATAM_QUOTED if name in by_name],
        'exile': by_name.get(EXILE_QUOTED),
        'exile_slow': (by_name.get(EXILE_QUOTED) or {}).get('ratio') is not None
                      and by_name[EXILE_QUOTED]['ratio'] < PACE_FAST,
        'p_slope': _p_value(figures.get('p_slope')),
        'slope_significant': figures.get('p_slope') is not None and figures['p_slope'] < SIGNIFICANCE,
    }


def _year(iso: Optional[str]) -> Optional[str]:
    """
    Year of an ISO date.

    Parameters
    ----------
    iso : str or None
        ISO date ``YYYY-MM-DD`` (or any string starting with the year).

    Returns
    -------
    str or None
        The four digits of the year, or ``None`` without a date.
    """
    return iso[:4] if isinstance(iso, str) and len(iso) >= 4 else None


def _every(rows: list, test, *keys) -> bool:
    """
    Whether every row has the values of ``keys`` and they pass ``test``: the gate of a sentence that asserts a
    direction for all of them.

    Parameters
    ----------
    rows : list of dict
        Rows to check.
    test : callable
        Takes the values of ``keys`` of a row, in order.
    *keys : str
        Keys of the values.

    Returns
    -------
    bool
        ``False`` without rows or when a row misses a value.
    """
    values = [[row.get(key) for key in keys] for row in rows]

    return bool(values) and all(None not in value and test(*value) for value in values)


def _pct_label(x: Optional[float]) -> Optional[str]:
    """
    A rate of 0 to 1 as the headers of the grids write it: ``0.09`` → ``'9%'``, ``0.095`` → ``'9,5%'``.

    Parameters
    ----------
    x : float or None
        The rate.

    Returns
    -------
    str or None
        The percentage, without decimals when it is whole, or ``None`` without a rate.
    """
    if x is None:
        return None

    pct = 100 * x

    return fmt_pct(pct, 0 if abs(pct - round(pct)) < 1e-9 else 1)


def _times(x: Optional[float]) -> Optional[str]:
    """
    A factor as the scenarios table writes it: ``2.0`` → ``'×2'``, ``1.5`` → ``'×1,5'``, ``1.187`` →
    ``'×1,19'``.

    Parameters
    ----------
    x : float or None
        The factor.

    Returns
    -------
    str or None
        ``×`` and the factor with up to two decimals, the trailing zeros dropped, or ``None`` without one.
    """
    if x is None:
        return None

    return '×' + fmt_num(x, 2).rstrip('0').rstrip(',')


def _regional_pairs(vote: dict) -> list:
    """
    The regional elections of ``vote.processes`` with both steps (the previous election and the last one).

    Parameters
    ----------
    vote : dict
        Data of the ``vote`` part.

    Returns
    -------
    list of dict
        ``{'community', 'from', 'to', 'margin_from', 'margin_to', 'left_from', 'left_to', 'd_psoe',
        'd_other_left', 'd_block'}`` per community, in the order of ``processes``: the years of both elections,
        the CERA margin left − right and the left share of each, and the changes of the ``last`` row.
    """
    steps = {}

    for row in vote.get('processes') or []:
        if row.get('kind') == 'autonomicas' and row.get('step') in ('previous', 'last'):
            steps.setdefault(row.get('community'), {})[row['step']] = row

    return [{'community': community, 'from': _year(rows['previous'].get('date')), 'to': _year(rows['last'].get('date')),
             'margin_from': rows['previous'].get('margin'), 'margin_to': rows['last'].get('margin'),
             'left_from': rows['previous'].get('left_share'), 'left_to': rows['last'].get('left_share'),
             'd_psoe': rows['last'].get('d_psoe'), 'd_other_left': rows['last'].get('d_other_left'),
             'd_block': rows['last'].get('d_block')}
            for community, rows in steps.items() if 'previous' in rows and 'last' in rows]


def _madrid(placement: dict, census: dict, prediction: dict, params: dict) -> dict:
    """
    The table of the «imán» of Madrid and the facts of its note.

    Parameters
    ----------
    placement : dict
        Data of the ``placement`` part (``madrid``).
    census : dict
        Data of the ``census`` part (``periods.article``, for the headers).
    prediction : dict
        Data of the ``prediction`` part (``last_seat``).
    params : dict
        ``meta.params`` (``madrid_countries``).

    Returns
    -------
    dict
        ``rows`` (``{'label', 'stock', 'p1', 'p2'}`` per row of ``placement.madrid``: «Madrid, n países» for the
        countries together, «Madrid, {country}» for each one); ``labels`` (the headers ``stock``, ``p1`` and ``p2``,
        with the years of the article periods); ``countries`` (``params.madrid_countries`` in the case of the
        prose); ``magnet`` (whether Madrid took more of the additions before the law than its share of the census);
        ``seat`` (``None`` without Madrid in ``last_seat``, else ``{'magnitude', 'largest', 'favoured'}``: its seats,
        whether no province has more and the candidature the net CERA effect favours, the runner-up with a positive
        net and the winner with a negative one).
    """
    countries = [_country_name(name) for name in params.get('madrid_countries') or []]
    rows = []

    for row in placement.get('madrid') or []:
        if row.get('label') == 'Conjunto':
            label = 'Madrid, {} países'.format(len(countries)) if countries else 'Madrid, conjunto'
        else:
            label = 'Madrid, ' + (_country_name(row.get('label')) or '')

        rows.append({'label': label, 'stock': row.get('stock_share'), 'p1': row.get('share_p1'),
                     'p2': row.get('share_p2')})

    periods = [period for period in _get(census, 'periods', 'article') or [] if len(period) == 2]
    labels = {'stock': 'Stock', 'p1': 'Altas antes de la ley', 'p2': 'Altas después'}

    if len(periods) == 2:
        (s1, e1), (s2, e2) = periods
        labels = {'stock': 'Stock ' + s2[:4], 'p1': 'Altas {}–{}'.format(s1[:4], e1[2:4]),
                  'p2': 'Altas {}–{}'.format(s2[:4], e2[2:4])}

    together = rows[0] if rows and (placement.get('madrid') or [{}])[0].get('label') == 'Conjunto' else {}
    last_seat = prediction.get('last_seat') or []
    row = next((row for row in last_seat if row.get('region_id') == MADRID), None)
    seat = None

    if row is not None:
        net = row.get('net')
        magnitudes = [other.get('magnitude') for other in last_seat if other.get('magnitude') is not None]
        seat = {'magnitude': row.get('magnitude'),
                'largest': row.get('magnitude') is not None and row['magnitude'] == max(magnitudes),
                'favoured': row.get('runner_up') if net is not None and net > 0
                else row.get('winner') if net is not None and net < 0 else None}

    return {'rows': rows, 'labels': labels, 'countries': countries, 'seat': seat,
            'magnet': together.get('p1') is not None and together.get('stock') is not None
            and together['p1'] > together['stock']}


def _question_2(parts: dict, figures: dict) -> dict:
    """
    View model of the card of question 2 (``#p2``): the gates and figures of its chart titles, its band, the
    Madrid table and its summary.

    Parameters
    ----------
    parts : dict
        The data of every part by name (``{}`` for a missing one).
    figures : dict
        ``_figures`` of the page (``latam_rate``, ``rest_rate``).

    Returns
    -------
    dict
        - ``processes``: how many processes ``participation.origin`` has (the participation by origin).
        - ``half``: whether the rest participates about twice as much as the Latin American new electors (within
          ``HALF_TOLERANCE``), ``lower`` whether they participate less at all.
        - ``ecological``: ``{'n', 'r', 'min_voters', 'year'}`` (``year`` the one of ``params.split_source``, the
          scrutiny of the regression).
        - ``communities``: ``_regional_pairs`` of ``vote``; ``years`` the first and the last year among them (or
          ``None``).
        - ``margins_fall`` and ``left_falls``: whether the margin left − right and the left share fall in every
          one of them; ``absorbs`` whether in every one the PSOE grows while the rest of the left falls;
          ``block_steady`` whether in every one the left block moves less than the PSOE.
        - ``sensitive``: ``{'magnitude', 'period', 'stock', 'growth', 'stock_countries', 'growth_countries',
          'down'}`` from ``placement.sensitive.article`` (``down``: the sensitive provinces take a smaller share
          of the additions than of the census) and ``params.sensitive_magnitude``.
        - ``madrid``: ``_madrid``.
    """
    meta, census, vote, placement = parts['meta'], parts['census'], parts['vote'], parts['placement']
    params = meta.get('params') or {}
    latam, rest = figures.get('latam_rate'), figures.get('rest_rate')
    ecological = vote.get('ecological') or {}
    pairs = _regional_pairs(vote)
    years = [pair[key] for pair in pairs for key in ('from', 'to') if pair[key]]
    article = _get(placement, 'sensitive', 'article') or {}
    together, countries = article.get('all') or {}, article.get('countries') or {}
    stock, growth = together.get('stock_share'), together.get('growth_share')

    return {
        'processes': len({row.get('process') for row in parts['participation'].get('origin') or []}),
        'half': bool(latam) and rest is not None and abs(rest / latam - 2) <= HALF_TOLERANCE,
        'lower': latam is not None and rest is not None and latam < rest,
        'ecological': {'n': ecological.get('n'), 'r': ecological.get('r'), 'min_voters': ecological.get('min_voters'),
                       'year': _year(params.get('split_source'))},
        'communities': pairs,
        'years': [min(years), max(years)] if years else None,
        'margins_fall': _every(pairs, lambda before, after: after < before, 'margin_from', 'margin_to'),
        'left_falls': _every(pairs, lambda before, after: after < before, 'left_from', 'left_to'),
        'absorbs': _every(pairs, lambda psoe, other: psoe > 0 > other, 'd_psoe', 'd_other_left'),
        'block_steady': _every(pairs, lambda block, psoe: abs(block) < abs(psoe), 'd_block', 'd_psoe'),
        'sensitive': {'magnitude': params.get('sensitive_magnitude'), 'period': article.get('period'),
                      'stock': stock, 'growth': growth, 'stock_countries': countries.get('stock_share'),
                      'growth_countries': countries.get('growth_share'),
                      'down': stock is not None and growth is not None and growth < stock},
        'madrid': _madrid(placement, census, parts['prediction'], params),
    }


def _table_row(label: str, before, after, change=None) -> dict:
    """
    A row of the table of a scenario: 2023, the projection and the change (``after − before`` by default).

    Parameters
    ----------
    label : str
        Header of the row.
    before, after : int or None
        The values of 2023 and of the projection.
    change : int, optional
        The change, when the part gives it.

    Returns
    -------
    dict
        ``{'label', 'before', 'after', 'change'}``; ``change`` is ``None`` when it cannot be computed.
    """
    if change is None and before is not None and after is not None:
        change = after - before

    return {'label': label, 'before': before, 'after': after, 'change': change}


def _scenario_table(scenario: dict) -> list:
    """
    The rows of the table of a scenario of the prediction: the census, the voters, the votes to candidatures and
    the seats of the left, of the right and of the other blocks (only when the part has other blocks).

    Parameters
    ----------
    scenario : dict
        A scenario of ``prediction.scenarios``.

    Returns
    -------
    list of dict
        ``_table_row`` per row.
    """
    census, new = scenario.get('census'), scenario.get('new_electors')
    rows = [_table_row('Censo CERA', census - new if census is not None and new is not None else None, census, new)]
    rows += [_table_row(label, scenario.get(before), scenario.get(after)) for label, before, after in PREDICTION_ROWS]
    blocks = {row.get('block'): row for row in scenario.get('by_block') or []}

    for block, label in BLOCK_LABELS.items():
        row = blocks.get(block) or {}
        rows.append(_table_row(label, row.get('before'), row.get('after'), row.get('change')))

    others = [row for block, row in blocks.items() if block not in BLOCK_LABELS]

    if others:
        rows.append(_table_row('Otros', *(sum(row.get(key) or 0 for row in others)
                                          for key in ('before', 'after', 'change'))))

    return rows


def _grid(grid: Optional[dict], rows_key: str, cols_key: str, uniform: bool = False) -> Optional[dict]:
    """
    A sensitivity grid as its table renders it.

    Parameters
    ----------
    grid : dict or None
        Grid with the participations under ``rows_key``, the shares of the vote under ``cols_key`` and
        ``values``.
    rows_key, cols_key : str
        Keys of the rows and of the columns.
    uniform : bool, optional
        Whether to mark the observed cell of the uniform split (the smallest participation at ``UNIFORM_SPLIT``).

    Returns
    -------
    dict or None
        ``None`` without rows or columns, else ``cols`` (the headers, ``_pct_label``), ``rows`` (``{'label',
        'observed', 'cells'}`` with ``cells`` ``{'value', 'mark'}``; ``observed`` is the smallest participation),
        ``lo`` (0) and ``hi`` (the largest value, at least 0) for ``shade_scale`` and ``max`` (``{'value',
        'participation', 'frac'}`` of the largest cell, or ``None`` without values).
    """
    rows, cols, values = _get(grid, rows_key) or [], _get(grid, cols_key) or [], _get(grid, 'values') or []

    if not rows or not cols:
        return None

    observed = rows.index(min(rows))
    table, top = [], None

    for i, participation in enumerate(rows):
        cells = []

        for j, frac in enumerate(cols):
            value = values[i][j] if i < len(values) and j < len(values[i] or []) else None
            cells.append({'value': value, 'mark': uniform and i == observed and abs(frac - UNIFORM_SPLIT) < 1e-9})

            if value is not None and (top is None or value > top['value']):
                top = {'value': value, 'participation': _pct_label(participation), 'frac': _pct_label(frac)}

        table.append({'label': _pct_label(participation), 'observed': i == observed, 'cells': cells})

    return {'cols': [_pct_label(frac) for frac in cols], 'rows': table, 'lo': 0,
            'hi': max(top['value'], 0) if top else 0, 'max': top}


def _scenario_name(row: dict) -> str:
    """
    Name of a row of ``sensitivity.scenarios`` from what it changes: «Central», «Participación ×2», «Sesgo 90%»
    or, with several changes, «Censo ×2, part. ×3, 90%».

    Parameters
    ----------
    row : dict
        ``{'census_x', 'part_x', 'bias'}``.

    Returns
    -------
    str
        The name.
    """
    pieces = []

    if row.get('census_x') is not None:
        pieces.append(('Censo ' + _times(row['census_x']), 'censo ' + _times(row['census_x'])))

    if row.get('part_x') is not None and row['part_x'] != 1:
        pieces.append(('Participación ' + _times(row['part_x']), 'part. ' + _times(row['part_x'])))

    if isinstance(row.get('bias'), (int, float)):
        pieces.append(('Sesgo ' + fmt_pct(100 * row['bias'], 0), fmt_pct(100 * row['bias'], 0)))

    if not pieces:
        return 'Central'

    if len(pieces) == 1:
        return pieces[0][0]

    name = ', '.join(short for _, short in pieces)

    return name[0].upper() + name[1:]


def _scenario_rows(sensitivity: dict, figures: dict) -> dict:
    """
    The table of the three dimensions (``sensitivity.scenarios``).

    Parameters
    ----------
    sensitivity : dict
        Data of the ``sensitivity`` part.
    figures : dict
        ``_figures`` of the page (``census_total`` and ``electors_2023``, the factor of the real census).

    Returns
    -------
    dict
        ``rows`` (``{'name', 'census', 'participation', 'bias', 'd_left', 'd_right'}``, the texts of the cells:
        «real (×1,19)» for the census of the election, «observada» for the observed participation, «empírico» or
        «90% izq.» for the bias) and ``span`` (the largest change in absolute value, at least 1, for
        ``shade_around``).
    """
    total, before = figures.get('census_total'), figures.get('electors_2023')
    real = 'real ({})'.format(_times(total / before)) if total is not None and before else 'real'
    rows, changes = [], []

    for row in sensitivity.get('scenarios') or []:
        bias = row.get('bias')
        rows.append({
            'name': _scenario_name(row),
            'census': real if row.get('census_x') is None else _times(row['census_x']),
            'participation': 'observada' if row.get('part_x') == 1 else _times(row.get('part_x')),
            'bias': 'empírico' if bias == 'empirico' else fmt_pct(100 * bias, 0) + ' izq.'
            if isinstance(bias, (int, float)) else bias,
            'd_left': row.get('d_left'),
            'd_right': row.get('d_right'),
        })
        changes += [abs(row[key]) for key in ('d_left', 'd_right') if row.get(key) is not None]

    return {'rows': rows, 'span': max(changes + [1])}


def _question_3(parts: dict, figures: dict, gap: list, last_seat_csv: Optional[str] = None) -> dict:
    """
    View model of the card of question 3 (``#p3``): the table of each census scenario, the facts of its notes,
    of the gap chart, of the three grids and of its summary.

    Parameters
    ----------
    parts : dict
        The data of every part by name (``{}`` for a missing one).
    figures : dict
        ``_figures`` of the page (``litigation``, ``grid_observed``, ``census_total``, ``electors_2023``).
    gap : list of dict
        The rows of the gap chart of the ``convocatoria`` scenario (``initial.gap.convocatoria``).
    last_seat_csv : str, optional
        Link to the CSV of the last seat of every province.

    Returns
    -------
    dict
        - ``scenarios``: per scenario of ``SCENARIOS``, ``{'key', 'label', 'rows', 'changed', 'unchanged',
          'votes'}``: ``rows`` from ``_scenario_table``, ``changed`` the names of the provinces whose seats
          change, ``unchanged`` whether nothing changes (no province, no block) and ``votes`` the projected votes.
        - ``votes_less``: the votes the scenario without the litigation has fewer.
        - ``real_left``: the change of the left block's seats in the ``convocatoria`` scenario (the real split).
        - ``seats_unchanged``: whether no candidature and no block changes seats in that scenario.
        - ``gap``: ``{'shown', 'total', 'max_net', 'min_gap', 'flipped', 'csv'}``: the provinces of the chart and
          of ``last_seat``, the largest net effect and the smallest gap among the shown ones, the provinces whose
          seats flip and ``last_seat_csv``.
        - ``grid`` and ``placement``: ``_grid`` of ``sensitivity.grid`` (with the uniform cell marked) and of
          ``sensitivity.placement``; ``contrast`` and ``pool`` of the placement and ``placement_countries`` (the
          countries of its pool, from ``params``, in the case of the prose).
        - ``three``: ``_scenario_rows``; ``uniform``: ``UNIFORM_SPLIT`` as a percentage.
    """
    prediction, sensitivity = parts['prediction'], parts['sensitivity']
    params = _get(parts['meta'], 'params') or {}
    scenarios = []

    for key in SCENARIOS:
        scenario = _get(prediction, 'scenarios', key) or {}
        changed = [row.get('name') for row in scenario.get('changed') or []]
        scenarios.append({'key': key, 'label': SCENARIO_LABELS[key], 'rows': _scenario_table(scenario),
                          'changed': changed, 'votes': scenario.get('votes_projected'),
                          'unchanged': bool(scenario) and not changed
                          and not any(row.get('change') for row in scenario.get('by_block') or [])})

    first, second = scenarios[0]['votes'], scenarios[1]['votes']
    central = _get(prediction, 'scenarios', SCENARIOS[0]) or {}
    left = next((row for row in central.get('by_block') or [] if row.get('block') == LEFT), {})
    last_seat = prediction.get('last_seat') or []
    nets = [row['net'] for row in gap if row.get('net') is not None]
    gaps = [row['gap'] for row in gap if row.get('gap') is not None]
    placement = sensitivity.get('placement') or {}

    return {
        'scenarios': scenarios,
        'votes_less': first - second if first is not None and second is not None else None,
        'real_left': left.get('change'),
        'seats_unchanged': bool(central) and not any(row.get('change') for row in (central.get('by_sigla') or [])
                                                     + (central.get('by_block') or [])),
        'gap': {'shown': len(gap), 'total': len(last_seat), 'max_net': max(nets) if nets else None,
                'min_gap': min(gaps) if gaps else None,
                'flipped': [row.get('name') for row in last_seat if row.get('flipped')], 'csv': last_seat_csv},
        'grid': _grid(sensitivity.get('grid'), 'participations', 'left_fracs', uniform=True),
        'placement': _grid(placement, 'participations', 'fracs'),
        'contrast': placement.get('contrast'),
        'pool': placement.get('pool'),
        'placement_countries': [_country_name(name) for name in params.get('placement_countries') or []],
        'three': _scenario_rows(sensitivity, figures),
        'uniform': _pct_label(UNIFORM_SPLIT),
    }
