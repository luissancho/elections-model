"""
Formatting filters of the web pages, following the rules of ``web/dist/js/format.js``.
"""
import functools
import math
import re
import warnings
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

MONTHS = ('ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sept', 'oct', 'nov', 'dic')
DASH = '–'
_DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')


def _is_missing(x) -> bool:
    """
    Tell whether a value has nothing to format.

    Parameters
    ----------
    x : object
        Value to check.

    Returns
    -------
    bool
        True for ``None`` and for any float that is NaN or infinite.
    """
    if x is None:
        return True

    try:
        return not math.isfinite(x)
    except TypeError:
        return True


def fmt_num(x, digits: int = 0) -> str:
    """
    Format a number in Spanish notation (``.`` thousands, ``,`` decimals).

    Ties round half away from zero, as ``Intl.NumberFormat`` does in ``format.js``.

    Parameters
    ----------
    x : int or float or None
        Number to format.
    digits : int, optional
        Number of decimals.

    Returns
    -------
    str
        The formatted number, or ``DASH`` when ``x`` is missing or not a finite number.
    """
    if isinstance(x, bool) or _is_missing(x):
        return DASH

    try:
        value = Decimal(x).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError, ValueError):
        return DASH

    text = '{:,.{}f}'.format(value, digits)

    return text.replace(',', '§').replace('.', ',').replace('§', '.')


def fmt_int(x) -> str:
    """
    Format a number without decimals.

    Parameters
    ----------
    x : int or float or None
        Number to format.

    Returns
    -------
    str
        The formatted number, or ``DASH``.
    """
    return fmt_num(x, 0)


def fmt_pct(x, digits: int = 1) -> str:
    """
    Format a percentage already expressed in the 0-100 scale.

    Parameters
    ----------
    x : int or float or None
        Percentage to format.
    digits : int, optional
        Number of decimals.

    Returns
    -------
    str
        ``'32,7 %'``, or ``DASH`` when ``x`` is missing.
    """
    text = fmt_num(x, digits)

    return text if text == DASH else text + ' %'


def fmt_prob(p, digits: int = 0) -> str:
    """
    Format a probability in [0, 1] as a percentage that never rounds a near-certainty away.

    Parameters
    ----------
    p : float or None
        Probability.
    digits : int, optional
        Number of decimals of the percentage.

    Returns
    -------
    str
        ``'0 %'`` and ``'100 %'`` only when exact; a near-certainty is capped below the top value the
        decimals can show (``'> 99 %'`` from 0.995 with no decimals, ``'> 99,9 %'`` from 0.9995 with one) and
        a near-impossibility floored above zero (``'< 1 %'`` up to 0.005, ``'< 0,1 %'`` up to 0.0005), else
        the percentage with ``digits`` decimals; ``DASH`` when ``p`` is missing.
    """
    if isinstance(p, bool) or _is_missing(p):
        return DASH

    if p == 0:
        return '0 %'

    if p == 1:
        return '100 %'

    unit = 10 ** (-digits)  # smallest percentage the decimals can show
    edge = unit / 2 / 100  # as a probability

    if p >= 1 - edge:
        return '> ' + fmt_pct(100 - unit, digits)

    if 0 < p <= edge:
        return '< ' + fmt_pct(unit, digits)

    return fmt_pct(p * 100, digits)


def fmt_range(lo, hi, digits: int = 1) -> str:
    """
    Format an interval as ``lo–hi``.

    Parameters
    ----------
    lo, hi : int or float or None
        Bounds of the interval.
    digits : int, optional
        Number of decimals of both bounds.

    Returns
    -------
    str
        ``'27,3–38,0'``, or ``DASH`` when a bound is missing.
    """
    if _is_missing(lo) or _is_missing(hi):
        return DASH

    return '{}{}{}'.format(fmt_num(lo, digits), DASH, fmt_num(hi, digits))


@functools.lru_cache(maxsize=1)
def _madrid_tz():
    """
    Return the Europe/Madrid time zone, or a fixed UTC+1 offset (with a warning) without tzdata.

    Returns
    -------
    datetime.tzinfo
        ``ZoneInfo('Europe/Madrid')``, or ``timezone(timedelta(hours=1))`` when the time zone
        database is missing (slim Docker images may lack ``tzdata``).
    """
    try:
        return ZoneInfo('Europe/Madrid')
    except ZoneInfoNotFoundError:
        warnings.warn('Europe/Madrid time zone not found (install tzdata); using a fixed UTC+1 offset',
                      stacklevel=2)
        return timezone(timedelta(hours=1))


def _parse_datetime(iso) -> Optional[datetime]:
    """
    Parse an ISO date or date-time.

    Parameters
    ----------
    iso : str or datetime.date or None
        ``YYYY-MM-DD`` (local midnight) or an ISO date-time (a trailing ``Z`` is UTC); an aware
        value is converted to Europe/Madrid.

    Returns
    -------
    datetime.datetime or None
        The parsed value, or ``None`` when it is missing or invalid.
    """
    if isinstance(iso, datetime):
        value = iso
    elif isinstance(iso, date):
        value = datetime(iso.year, iso.month, iso.day)
    elif isinstance(iso, str) and iso:
        try:
            if _DATE_RE.fullmatch(iso):
                value = datetime.combine(date.fromisoformat(iso), datetime.min.time())
            else:
                value = datetime.fromisoformat(iso[:-1] + '+00:00' if iso.endswith('Z') else iso)
        except ValueError:
            return None
    else:
        return None

    if value.tzinfo is not None:
        value = value.astimezone(_madrid_tz())

    return value


def fmt_date(iso) -> str:
    """
    Format a date as ``'13 oct 2026'``.

    Parameters
    ----------
    iso : str or datetime.date or None
        ISO date or date-time.

    Returns
    -------
    str
        The formatted date, or ``DASH`` when it is missing or invalid.
    """
    value = _parse_datetime(iso)

    if value is None:
        return DASH

    return '{} {} {}'.format(value.day, MONTHS[value.month - 1], value.year)


def fmt_datetime(iso) -> str:
    """
    Format an instant as ``'8 oct 2026, 20:11'`` in Europe/Madrid time.

    Parameters
    ----------
    iso : str or datetime.datetime or None
        ISO date-time; aware values are converted to Europe/Madrid (UTC+1 without tzdata).

    Returns
    -------
    str
        The formatted date and time, or ``DASH`` when it is missing or invalid.
    """
    value = _parse_datetime(iso)

    if value is None:
        return DASH

    return '{}, {:02d}:{:02d}'.format(fmt_date(value), value.hour, value.minute)


def _fraction(x, start, end) -> Optional[float]:
    """
    Position of a value in a range, as a fraction clipped to [0, 1].

    Parameters
    ----------
    x : int or float or None
        Value to place.
    start, end : int or float
        Bounds of the range; it may run downwards (``start > end``), so the fraction grows towards ``end``.

    Returns
    -------
    float or None
        ``(x - start) / (end - start)`` clipped to [0, 1], or ``None`` when ``x`` is missing or the range is
        empty.
    """
    if isinstance(x, bool) or _is_missing(x) or _is_missing(start) or _is_missing(end) or start == end:
        return None

    return min(1.0, max(0.0, (x - start) / (end - start)))


def bar_width(x, vmin=0, vmax=100) -> float:
    """
    Width of the bar of a table cell, as a percentage of the cell (the bars of ``Computer.print_ratings``).

    Parameters
    ----------
    x : int or float or None
        Value of the cell.
    vmin, vmax : int or float, optional
        Values that give an empty and a full bar.

    Returns
    -------
    float
        The percentage, 0 to 100 with one decimal; 0 when ``x`` is missing or ``vmin == vmax``.
    """
    fraction = _fraction(x, vmin, vmax)

    return 0.0 if fraction is None else round(100 * fraction, 1)


def bar_sides(x, limit) -> dict:
    """
    Geometry of a bar centred at zero inside a table cell: it grows to the left with a negative value and to the
    right with a positive one, up to ``limit`` in either direction (the diverging bars of ``print_ratings``).

    Parameters
    ----------
    x : int or float or None
        Value of the cell.
    limit : int or float
        Value (in absolute terms) that fills a whole side.

    Returns
    -------
    dict
        ``{'left': start of the bar, 'width': its width, 'side': 'neg' | 'pos' | 'zero'}``, ``left`` and
        ``width`` as percentages of the cell with one decimal; an empty bar at the centre (``'zero'``) when
        ``x`` is zero, missing or ``limit`` is not positive.
    """
    if isinstance(x, bool) or _is_missing(x) or _is_missing(limit) or limit <= 0 or x == 0:
        return {'left': 50.0, 'width': 0.0, 'side': 'zero'}

    width = round(50 * min(abs(x), limit) / limit, 1)

    if x < 0:
        return {'left': round(50 - width, 1), 'width': width, 'side': 'neg'}

    return {'left': 50.0, 'width': width, 'side': 'pos'}


def shade(x, start, end, color: str, alpha: float = 0.55) -> str:
    """
    Background of a shaded table cell: ``color`` with an opacity that grows with the value from ``start``
    (transparent) to ``end`` (``alpha``), like the gradients of ``Computer.print_ratings``.

    Parameters
    ----------
    x : int or float or None
        Value of the cell.
    start, end : int or float
        Values of the transparent and the fullest cells; the range may run downwards (``start > end``).
    color : str
        Hex colour ``#rrggbb``.
    alpha : float, optional
        Opacity of the fullest cell.

    Returns
    -------
    str
        An ``rgba()`` colour, or ``'transparent'`` when ``x`` is missing, the range is empty or the colour is
        not a six-digit hex colour.
    """
    fraction = _fraction(x, start, end)

    if fraction is None or not isinstance(color, str) or not re.fullmatch(r'#[0-9A-Fa-f]{6}', color):
        return 'transparent'

    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))

    return 'rgba({}, {}, {}, {})'.format(r, g, b, round(alpha * fraction, 3))
