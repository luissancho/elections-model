"""
Server-rendered web pages: routes, the Jinja2 environment, formatting filters and page contexts.

The pages are rendered by the controllers of ``mtpy/controllers`` (``Page`` and its subclasses)
from the templates of ``web/templates``. Every context is built from the published bundle through
``webapi.site()``, reading all the parts of a page with the same run, so a page never mixes runs.
The formatting filters follow the rules of ``web/dist/js/format.js``. Like ``webapi``, this module
does not import the model nor ``publish``, so the web workers stay light.
"""
import functools
import math
import os
import re
import warnings
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Optional
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from ..core.api import HttpError
from ..core.app import App
from . import bundle
from .webapi import check_mode, check_run, check_scope, site

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TEMPLATES_DIR = os.path.join(REPO_ROOT, 'web', 'templates')
SITE_TITLE = 'Pronóstico electoral'
PAGES = ({'href': '/', 'label': 'Portada'}, {'href': '/promedio', 'label': 'Promedio'})
ROUTES = [('/promedio', 'promedio', 'index', ['GET'])]
NOT_FOUND = [('/api/', 'base'), ('/', 'page')]
MONTHS = ('ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sept', 'oct', 'nov', 'dic')
DASH = '–'
MESSAGES = {
    'no bundle published yet': 'Todavía no hay ningún pronóstico publicado.',
    'scope not published': 'Este ámbito no tiene pronóstico publicado.',
    'run not found': 'Ese run no existe.',
    'no runs published': 'Este ámbito no tiene runs publicados.',
    'invalid scope': 'Ámbito no válido.',
    'invalid run': 'Run no válido.',
    'invalid mode': 'Modo no válido.',
}
TABLE_ROWS = 40
OTHERS_COLOR = '#9e9e9e'
MODE_LABELS = {'nowcast': 'Hoy', 'forecast': 'Elección'}

_DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')
_COLOR_RE = re.compile(r'#[0-9A-Fa-f]{3,8}')


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
        ``'0 %'`` and ``'100 %'`` only when exact, ``'> 99 %'`` from 0.995, ``'< 1 %'`` up to
        0.005 (0 excluded), else the percentage; ``DASH`` when ``p`` is missing.
    """
    if isinstance(p, bool) or _is_missing(p):
        return DASH

    if p == 0:
        return '0 %'

    if p == 1:
        return '100 %'

    if p >= 0.995:
        return '> 99 %'

    if 0 < p <= 0.005:
        return '< 1 %'

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


def environment() -> Environment:
    """
    Jinja2 environment of the pages, created on first use and cached in the App.

    It loads ``TEMPLATES_DIR`` with autoescape on for ``.html``, ``StrictUndefined`` and the
    filters ``num``, ``int``, ``pct``, ``prob``, ``range``, ``date`` and ``datetime``. Without
    a booted App the environment is built but not cached.

    Returns
    -------
    jinja2.Environment
        The shared environment (``App.get_().templates``).
    """
    app = App.get_() if App.has_() else None
    env = app.templates if app is not None else None

    if env is None:
        env = Environment(
            loader=FileSystemLoader(TEMPLATES_DIR),
            autoescape=select_autoescape(('html',)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.policies['json.dumps_kwargs'] = {'sort_keys': False}
        env.filters.update({
            'num': fmt_num,
            'int': fmt_int,
            'pct': fmt_pct,
            'prob': fmt_prob,
            'range': fmt_range,
            'date': fmt_date,
            'datetime': fmt_datetime,
        })

        if app is not None:
            app.set('templates', env)

    return env


def parse_state(query: dict) -> dict:
    """
    Validate the page parameters of a query string.

    Parameters
    ----------
    query : dict
        Query string parameters (``scope``, ``mode``, ``run``; others are ignored).

    Returns
    -------
    dict
        ``{'scope': str or None, 'mode': str, 'run': str or None}``; ``mode`` defaults to
        ``'forecast'`` and an empty value counts as missing.

    Raises
    ------
    HttpError
        400 when a given parameter is invalid.
    """
    scope = query.get('scope') or None

    if scope is not None:
        scope = check_scope(scope)

    return {
        'scope': scope,
        'mode': check_mode(query.get('mode') or 'forecast'),
        'run': check_run(query.get('run')),
    }


def published_scopes() -> list:
    """
    Scopes with a published forecast, in catalogue order.

    Returns
    -------
    list of dict
        Rows of ``site().scopes()`` whose ``simulable`` is true (``code``, ``name``...).
    """
    return [row for row in site().scopes()['scopes'] if row['simulable']]


def resolve_scope(scope: Optional[str]) -> str:
    """
    Scope a page shows.

    Parameters
    ----------
    scope : str or None
        Validated scope of the request, or ``None`` when it was not given.

    Returns
    -------
    str
        ``scope`` when it is published; without it, the first published scope in catalogue
        order (manifest order for scopes outside the catalogue).

    Raises
    ------
    HttpError
        503 when nothing is published; 404 when the given scope is not published.
    """
    manifest = site().manifest_data()

    if scope is None:
        codes = [row['code'] for row in published_scopes()] or list(manifest['scopes'])

        if not codes:
            raise HttpError(503, 'no bundle published yet')

        return codes[0]

    if scope not in manifest['scopes']:
        raise HttpError(404, 'scope not published')

    return scope


def part(scope: str, run: str, name: str, mode: Optional[str] = None) -> dict:
    """
    Data of a part of a published run.

    Parameters
    ----------
    scope : str
        Scope code.
    run : str
        Run id.
    name : str
        Part name.
    mode : str, optional
        Simulation mode, for the mode parts.

    Returns
    -------
    dict
        The ``data`` member of the part envelope.

    Raises
    ------
    HttpError
        404 when the run or the part does not exist.
    """
    return bundle.loads(site().run_file(scope, run, name, mode))['data']


def query_string(state: dict) -> str:
    """
    Query string that keeps the state of a page in its links.

    Parameters
    ----------
    state : dict
        ``scope``, ``mode`` and ``run``; ``run`` is kept only when ``pinned`` is true.

    Returns
    -------
    str
        ``'?scope=es&mode=forecast'``, plus ``&run=...`` for a pinned run.
    """
    params = [('scope', state.get('scope')), ('mode', state.get('mode'))]

    if state.get('pinned'):
        params.append(('run', state.get('run')))

    return '?' + urlencode([(key, value) for key, value in params if value])


def api_url(scope: str, part: str, mode: Optional[str] = None, run: Optional[str] = None,
            fmt: Optional[str] = None) -> str:
    """
    URL of a part in the ``/api/v1`` forecast routes.

    Parameters
    ----------
    scope : str
        Scope code.
    part : str
        Part name; ``'meta'`` is the scope route itself.
    mode : str, optional
        Simulation mode, for the mode parts.
    run : str, optional
        Run id.
    fmt : str, optional
        Output format (``'json'`` or ``'csv'``).

    Returns
    -------
    str
        ``'/api/v1/forecast/{scope}[/{mode}]/{part}[?run=...&format=...]'``.
    """
    url = '/api/v1/forecast/' + scope

    if mode:
        url += '/' + mode

    if part != 'meta':
        url += '/' + part

    params = [(key, value) for key, value in (('run', run), ('format', fmt)) if value]

    return url + ('?' + urlencode(params) if params else '')


class Catalog(dict):
    """
    Parties of a run by name (``{name: {fullname, color, block}}``).

    A party missing from the run metadata gets its name as full name, the grey of the
    others and no block, so the templates can look up any name.
    """

    def __missing__(self, name):
        """
        Entry of a party outside the catalogue.

        Parameters
        ----------
        name : str
            Party name.

        Returns
        -------
        dict
            ``{'fullname': name, 'color': OTHERS_COLOR, 'block': None}``.
        """
        return {'fullname': name, 'color': OTHERS_COLOR, 'block': None}


def catalog(meta: dict) -> Catalog:
    """
    Party catalogue of a run.

    Parameters
    ----------
    meta : dict
        Data of the meta part.

    Returns
    -------
    Catalog
        Entries from ``meta['parties']``; a missing full name falls back to the name and a
        missing or malformed colour to ``OTHERS_COLOR``.
    """
    entries = Catalog()

    for party in meta.get('parties') or []:
        color = party.get('color')
        entries[party['name']] = {
            'fullname': party.get('fullname') or party['name'],
            'color': color if isinstance(color, str) and _COLOR_RE.fullmatch(color) else OTHERS_COLOR,
            'block': party.get('block'),
        }

    return entries


def block_color(block: str, meta: dict, parties: Catalog) -> str:
    """
    Colour of a block: that of its first party in ``bmaps.vs``, then in ``bmaps.blocks``.

    Parameters
    ----------
    block : str
        Block name.
    meta : dict
        Data of the meta part.
    parties : Catalog
        Party catalogue of the run.

    Returns
    -------
    str
        The colour, or ``OTHERS_COLOR`` when the block has no party in either map.
    """
    bmaps = meta.get('bmaps') or {}

    for key in ('vs', 'blocks'):
        groups = bmaps.get(key)
        names = groups.get(block) if isinstance(groups, dict) else None

        if names:
            return parties[names[0]]['color']

    return OTHERS_COLOR


def common_context(state: dict, active: str) -> dict:
    """
    Template context shared by every page.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    active : str
        ``href`` of the current page in the navigation.

    Returns
    -------
    dict
        ``site_title``, ``pages`` (``href``, ``label``, ``active``), ``state`` (``scope``,
        ``mode``, ``run``, ``pinned``), ``query``, ``scopes``, ``manifest``, ``meta``,
        ``catalog``, ``mode_label`` and ``title_suffix``.

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope or a missing run.
    """
    scope = resolve_scope(state['scope'])
    pinned = state['run'] is not None
    run = state['run'] or site().latest_run(scope)
    mode = state['mode']
    meta = part(scope, run, 'meta')
    full = {'scope': scope, 'mode': mode, 'run': run, 'pinned': pinned}

    if mode == 'nowcast':
        title_suffix = 'Estimación a {}'.format(fmt_date(meta.get('as_of')))
    else:
        title_suffix = 'Pronóstico para el {}'.format(fmt_date(meta.get('event_date')))

    return {
        'site_title': SITE_TITLE,
        'pages': [dict(page, active=page['href'] == active) for page in PAGES],
        'state': full,
        'query': query_string(full),
        'scopes': published_scopes(),
        'manifest': site().manifest_data(),
        'meta': meta,
        'catalog': catalog(meta),
        'mode_label': MODE_LABELS[mode],
        'title_suffix': title_suffix,
    }


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


def index_context(state: dict, active: str = '/') -> dict:
    """
    Template context of the home page.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    active : str, optional
        ``href`` of the page in the navigation.

    Returns
    -------
    dict
        The common context plus ``headline`` (the headline of the mode), ``majority``
        (``majority_rows``) and ``initial`` (``state``, ``meta``, ``headline``, ``vote``,
        ``summary`` and ``runs``, the data the charts are drawn from).

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope, a missing run or a missing part.
    """
    context = common_context(state, active)
    full = context['state']
    scope, mode, run = full['scope'], full['mode'], full['run']
    headline = part(scope, run, 'headline')
    current = headline[mode]

    context['headline'] = current
    context['majority'] = majority_rows(current.get('p_majority'), context['meta'], context['catalog'])
    context['initial'] = {
        'state': full,
        'meta': context['meta'],
        'headline': headline,
        'vote': part(scope, run, 'vote', mode=mode),
        'summary': part(scope, run, 'summary', mode=mode),
        'runs': bundle.loads(site().history(scope))['data'],
    }

    return context


def table_rows(polls: list, parties: list, n: int = TABLE_ROWS) -> list:
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
        Maximum number of rows.

    Returns
    -------
    list of list of str
        The last ``n`` polls by date (ties keep the later published first) as ``date``,
        pollster, sponsor, sample size and one percentage per party, all formatted.
    """
    latest = sorted(reversed(polls), key=lambda poll: poll['date'], reverse=True)[:n]

    return [
        [
            fmt_date(poll['date']),
            poll.get('pollster') or DASH,
            poll.get('sponsor') or DASH,
            fmt_int(poll.get('sample_size')),
        ] + [fmt_num(poll.get(name), 1) for name in parties]
        for poll in latest
    ]


def promedio_context(state: dict, active: str = '/promedio') -> dict:
    """
    Template context of the poll average page.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    active : str, optional
        ``href`` of the page in the navigation.

    Returns
    -------
    dict
        The common context plus ``table_parties`` (the main-bloc parties present in the
        series, or all of them), ``table_rows`` (``table_rows``), ``n_polls``, ``csv_href``,
        ``series_subtitle`` and ``initial`` (``state``, ``meta``, ``series``, ``polls``,
        ``projection`` and ``vote``, the data the chart is drawn from).

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope, a missing run or a missing part.
    """
    context = common_context(state, active)
    full, meta = context['state'], context['meta']
    scope, mode, run = full['scope'], full['mode'], full['run']
    series = part(scope, run, 'series')
    polls = part(scope, run, 'polls')
    vote = part(scope, run, 'vote', mode=mode)
    names = series.get('parties') or []
    main = (meta.get('bmaps') or {}).get('main') or []

    if mode == 'nowcast':
        subtitle = 'a ' + fmt_date(meta.get('as_of'))
    else:
        subtitle = '{} → {}'.format(fmt_date(meta.get('as_of')), fmt_date(vote.get('when')))

    table_parties = [name for name in main if name in names] or list(names)
    context['table_parties'] = table_parties
    context['table_rows'] = table_rows(polls.get('polls') or [], table_parties)
    context['n_polls'] = meta.get('n_polls')
    context['csv_href'] = api_url(scope, 'polls', run=run, fmt='csv')
    context['series_subtitle'] = subtitle
    context['initial'] = {
        'state': full,
        'meta': meta,
        'series': series,
        'polls': polls,
        'projection': part(scope, run, 'projection', mode=mode),
        'vote': vote,
    }

    return context
