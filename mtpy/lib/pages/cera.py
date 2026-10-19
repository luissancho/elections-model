"""
View models of the cards of the CERA study page (``/analisis/voto-cera``), one per question, and their helpers.

``analisis.cera_context`` assembles them; each takes the data of the parts it reads and returns what its card
renders, with ``None`` for what the parts do not have (the templates guard it), never an error.
"""
from typing import Optional

from .filters import fmt_num

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
          all the countries, for the summary.
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
        'p_slope': _p_value(figures.get('p_slope')),
        'slope_significant': figures.get('p_slope') is not None and figures['p_slope'] < SIGNIFICANCE,
    }
