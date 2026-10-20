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
# The green of ``--color-ok`` and its lightest and strongest opacities in ``shade_scale``
SCALE_COLOR = '#16a34a'
SCALE_ALPHA = (0.08, 0.55)


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
        ``'32,7%'`` (the sign attached, Luis, 2026-10-15), or ``DASH`` when ``x`` is missing.
    """
    text = fmt_num(x, digits)

    return text if text == DASH else text + '%'


def fmt_prob(p, digits: int = 1) -> str:
    """
    Format a probability in [0, 1] as a percentage that never rounds a near-certainty away.

    Parameters
    ----------
    p : float or None
        Probability.
    digits : int, optional
        Number of decimals of the percentage (one everywhere, Luis, 2026-10-15).

    Returns
    -------
    str
        ``'0,0%'`` and ``'100,0%'`` only when exact; a near-certainty is capped below the top value the
        decimals can show (``'> 99,9%'`` from 0.9995 with one decimal, ``'> 99%'`` from 0.995 with none) and
        a near-impossibility floored above zero (``'< 0,1%'`` up to 0.0005, ``'< 1%'`` up to 0.005), else
        the percentage with ``digits`` decimals; ``DASH`` when ``p`` is missing.
    """
    if isinstance(p, bool) or _is_missing(p):
        return DASH

    if p == 0:
        return fmt_pct(0, digits)

    if p == 1:
        return fmt_pct(100, digits)

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


def fmt_date_short(iso) -> str:
    """
    Format a date as ``'3 oct'`` (day and month, no year).

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

    return '{} {}'.format(value.day, MONTHS[value.month - 1])


def delta_text(x, digits: int = 0) -> str:
    """
    Format a difference with its sign: ``'+3'``, ``'−14'`` (typographic minus), ``'+0,0'``.

    Parameters
    ----------
    x : int or float or None
        The difference.
    digits : int, optional
        Number of decimals.

    Returns
    -------
    str
        The signed number, or ``DASH`` when ``x`` is missing.
    """
    text = fmt_num(abs(x), digits) if not (isinstance(x, bool) or _is_missing(x)) else DASH

    if text == DASH:
        return DASH

    negative = round(x, digits) < 0

    return ('−' if negative else '+') + text


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


def shade_scale(x, lo, hi) -> str:
    """
    Background of a cell of a sensitivity grid: the green of ``--color-ok`` (``SCALE_COLOR``) growing from ``lo``
    (the lightest, ``SCALE_ALPHA[0]``) to ``hi`` (the strongest, ``SCALE_ALPHA[1]``), clipped at both ends.

    Parameters
    ----------
    x : int or float or None
        Value of the cell.
    lo, hi : int or float
        Values of the lightest and the strongest cells; with an empty range every value takes the lightest.

    Returns
    -------
    str
        An ``rgba()`` colour, or ``''`` when ``x`` is missing (the cell goes without a style).
    """
    if isinstance(x, bool) or _is_missing(x):
        return ''

    fraction = _fraction(x, lo, hi) or 0.
    alpha = SCALE_ALPHA[0] + (SCALE_ALPHA[1] - SCALE_ALPHA[0]) * fraction
    r, g, b = (int(SCALE_COLOR[i:i + 2], 16) for i in (1, 3, 5))

    return 'rgba({}, {}, {}, {})'.format(r, g, b, round(alpha, 3))


def delta_badge(x, digits: int = 0) -> dict:
    """
    Arrow and direction of a difference, for the small badges of the tables and cards: ``▲ 6``, ``▼ 1,0`` or a dash.

    Parameters
    ----------
    x : int or float or None
        The difference.
    digits : int, optional
        Number of decimals of the magnitude.

    Returns
    -------
    dict
        ``{'dir': 'up' | 'down' | 'flat', 'text': '▲ 6' | '▼ 1,0' | DASH}``: the arrow with the magnitude (no sign),
        rounded like ``fmt_num``; ``flat`` with ``DASH`` when the magnitude rounds to zero or ``x`` is missing.
    """
    if isinstance(x, bool) or _is_missing(x):
        return {'dir': 'flat', 'text': DASH}

    magnitude = fmt_num(abs(x), digits)

    if magnitude == DASH or set(magnitude) <= set('0,.'):
        return {'dir': 'flat', 'text': DASH}

    return {'dir': 'up' if x > 0 else 'down', 'text': ('▲ ' if x > 0 else '▼ ') + magnitude}


def shade_around(x, centre, span, below: str, above: str, alpha: float = 0.55) -> str:
    """
    Background of a cell shaded both ways from a centre (the double gradients of the tables): ``below`` grows from
    ``centre`` (transparent) down to ``centre - span`` and ``above`` from ``centre`` up to ``centre + span``.

    Parameters
    ----------
    x : int or float or None
        Value of the cell.
    centre : int or float
        Value of the transparent cell.
    span : int or float
        Distance from ``centre`` that gives the fullest cell, either way; it must be positive.
    below, above : str
        Hex colours ``#rrggbb`` of the values below and above (or at) the centre.
    alpha : float, optional
        Opacity of the fullest cell.

    Returns
    -------
    str
        An ``rgba()`` colour (``shade``), or ``'transparent'`` when ``x`` is missing or ``span`` is not positive.
    """
    if isinstance(x, bool) or _is_missing(x) or _is_missing(centre) or _is_missing(span) or span <= 0:
        return 'transparent'

    if x < centre:
        return shade(x, centre, centre - span, below, alpha)

    return shade(x, centre, centre + span, above, alpha)


def gradient(x, start, end, colors) -> str:
    """
    Colour of a value on a scale of several stops (the rating badge of the polls tables): ``colors`` are spread
    evenly from ``start`` to ``end`` and the value takes the mix of the two stops it falls between.

    Parameters
    ----------
    x : int or float or None
        Value to colour.
    start, end : int or float
        Values of the first and the last colour; the range may run downwards (``start > end``) and a value
        outside it takes the colour of the nearest end.
    colors : sequence of str
        Two or more hex colours ``#rrggbb``, in order from ``start`` to ``end``.

    Returns
    -------
    str
        A hex colour ``#rrggbb`` (each channel rounded half up, as the mirror of ``format.js`` does), or
        ``'transparent'`` when ``x`` is missing, the range is empty or there are not two valid colours.
    """
    fraction = _fraction(x, start, end)
    stops = list(colors) if isinstance(colors, (list, tuple)) else []

    valid = all(isinstance(c, str) and re.fullmatch(r'#[0-9A-Fa-f]{6}', c) for c in stops)

    if fraction is None or len(stops) < 2 or not valid:
        return 'transparent'

    position = fraction * (len(stops) - 1)
    index = min(int(position), len(stops) - 2)
    t = position - index
    lo = [int(stops[index][i:i + 2], 16) for i in (1, 3, 5)]
    hi = [int(stops[index + 1][i:i + 2], 16) for i in (1, 3, 5)]

    return '#' + ''.join('{:02x}'.format(math.floor(a + (b - a) * t + 0.5)) for a, b in zip(lo, hi))


def rng_cell(lo, hi, median, total, majority=None) -> Optional[dict]:
    """
    Geometry of the range cell of a seats table: the 95 % interval as a band, the median as a mark and the
    absolute majority as a line, all as percentages of the chamber.

    Parameters
    ----------
    lo, hi : int or float or None
        Bounds of the interval, in seats.
    median : int or float or None
        Central value, in seats.
    total : int or float or None
        Seats of the chamber (the full width of the cell).
    majority : int or float, optional
        Absolute majority, in seats.

    Returns
    -------
    dict or None
        ``{'left': start of the band, 'width': its width, 'mark': the median, 'line': the majority or None}``,
        percentages of the cell with one decimal, each value clipped to [0, 100] before the width is taken; ``None``
        when a bound, the median or the total is missing, the total is not positive or the interval is inverted.
    """
    if any(isinstance(x, bool) or _is_missing(x) for x in (lo, hi, median, total)) or total <= 0 or lo > hi:
        return None

    def place(x) -> float:
        return round(100 * min(1.0, max(0.0, x / total)), 1)

    left = place(lo)

    return {
        'left': left,
        'width': round(place(hi) - left, 1),
        'mark': place(median),
        'line': None if _is_missing(majority) else place(majority),
    }
