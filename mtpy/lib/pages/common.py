"""
Common parts of the web pages: routes, messages, the Jinja2 environment and the shared context.
"""
import os
import re
from datetime import date, datetime
from typing import Optional
from urllib.parse import urlencode, urlsplit

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from ...core.api import HttpError
from ...core.app import App
from .. import bundle
from ..webapi import check_mode, check_run, check_scope, scope_codes, settings, site
from .filters import (_madrid_tz, _parse_datetime, bar_sides, bar_width, delta_badge, delta_text, fmt_date, fmt_date_short, gradient,
                      fmt_datetime, fmt_int, fmt_num, fmt_pct, fmt_prob, fmt_range, rng_cell, shade, shade_around)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
TEMPLATES_DIR = os.path.join(REPO_ROOT, 'web', 'templates')
SITE_TITLE = 'Demoscopio'
# Under the wordmark in the header, and the default description of the pages
SITE_TAGLINE = 'Promedio de encuestas, previsión de escaños y ranking de encuestadoras'
# The author, linked from the footer next to the model attribution
AUTHOR_HANDLE = '@luis_sancho'
AUTHOR_URL = 'https://x.com/luis_sancho'
# Main menu: the global pages. Promedios and Escaños are the tabs of the event bar (``EVENT_TABS``).
PAGES = (
    {'href': '/', 'label': 'Inicio'},
    {'href': '/elecciones', 'label': 'Histórico Errores'},
    {'href': '/casas', 'label': 'Ranking Encuestadoras'},
    {'href': '/metodo', 'label': 'Método'},
)
EVENT_TABS = (
    {'suffix': '', 'label': 'Promedios'},
    {'suffix': '/escanos', 'label': 'Escaños'},
)
# Bottom tab bar of the phones: five entries with an icon of the sprite (``_svg.html``); the method goes in the
# round button of the header
TABBAR = (
    {'key': 'inicio', 'label': 'Inicio', 'icon': 'home'},
    {'key': 'promedios', 'label': 'Promedios', 'icon': 'line'},
    {'key': 'escanos', 'label': 'Escaños', 'icon': 'hemi'},
    {'key': 'historico', 'label': 'Histórico', 'icon': 'target'},
    {'key': 'ranking', 'label': 'Ranking', 'icon': 'trophy'},
)
# Entry of the tab bar lit by each global page (``PAGES`` href); the method lights none
TABBAR_ACTIVE = {'/': 'inicio', '/elecciones': 'historico', '/casas': 'ranking'}
# Blocks of the hemicycle from left to right (the mirror of ``BLOCK_ORDER`` of ``catalog.js``)
HEMICYCLE_BLOCKS = ('Izquierda', 'Separatista', 'Regionalista', 'Derecha')
# Chamber of a scope, for the event bar and the home ("Congreso, 350 escaños")
CHAMBERS = {'es': 'Congreso'}
DEFAULT_CHAMBER = 'Parlamento'
# ``Router.handle`` returns the first route that matches: the fixed routes go before ``/{scope}``.
ROUTES = [
    ('/promedio', 'redirect', 'promedio', ['GET']),
    ('/escanos', 'redirect', 'escanos', ['GET']),
    ('/modelo', 'redirect', 'modelo', ['GET']),
    ('/casas', 'casas', 'index', ['GET']),
    ('/casas/{pollster}', 'casas', 'profile', ['GET']),
    ('/elecciones', 'elecciones', 'index', ['GET']),
    ('/elecciones/{country:loc}/{scope}/{date}', 'elecciones', 'detail', ['GET']),
    ('/metodo', 'metodo', 'index', ['GET']),
    ('/metodo/{section:str}', 'metodo', 'section', ['GET']),
    ('/{scope}', 'promedios', 'index', ['GET']),
    ('/{scope}/escanos', 'escanos', 'index', ['GET']),
]
NOT_FOUND = [('/api/', 'base'), ('/', 'page')]
MESSAGES = {
    'no bundle published yet': 'Todavía no hay ningún pronóstico publicado.',
    'scope not published': 'Este ámbito no tiene pronóstico publicado.',
    'run not found': 'Ese run no existe.',
    'no runs published': 'Este ámbito no tiene runs publicados.',
    'invalid scope': 'Ámbito no válido.',
    'invalid run': 'Run no válido.',
    'invalid mode': 'Modo no válido.',
    'invalid region': 'Circunscripción no válida.',
    'region not found': 'Circunscripción no encontrada.',
    'no database configured': 'La base de datos no está disponible.',
    'database unavailable': 'La base de datos no está disponible.',
    'pollster not found': 'Encuestadora no encontrada.',
    'invalid pollster': 'Encuestadora no válida.',
    'invalid date': 'Fecha no válida.',
    'event not published': 'Elección no publicada.',
    'backtest not published': 'Backtest no publicado.',
    'unknown scope': 'Ámbito desconocido.',
    'section not found': 'Página del método no encontrada.',
}
OTHERS_COLOR = '#9e9e9e'
_COLOR_RE = re.compile(r'#[0-9A-Fa-f]{3,8}')
DARK_INK = '#111'
LIGHT_INK = '#fff'


def ink_color(color: Optional[str]) -> str:
    """
    Text colour readable on a background of ``color``: dark on light colours, white on dark ones.

    Parameters
    ----------
    color : str or None
        Hex colour ``#rgb``, ``#rrggbb`` or ``#rrggbbaa`` (the alpha is ignored).

    Returns
    -------
    str
        ``'#111'`` when the relative luminance (WCAG, linear sRGB) of the colour is above 0.5, ``'#fff'``
        otherwise or when the colour is not a valid hex colour (``OTHERS_COLOR`` gives white).
    """
    if not isinstance(color, str) or not re.fullmatch(r'#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})', color):
        return LIGHT_INK

    digits = color[1:]
    if len(digits) == 3:
        digits = ''.join(char * 2 for char in digits)

    def linear(channel):
        value = int(channel, 16) / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(digits[i:i + 2]) for i in (0, 2, 4))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return DARK_INK if luminance > 0.5 else LIGHT_INK


def environment() -> Environment:
    """
    Jinja2 environment of the pages, created on first use and cached in the App.

    It loads ``TEMPLATES_DIR`` with autoescape on for ``.html``, ``StrictUndefined``, the
    filters ``num``, ``int``, ``pct``, ``prob``, ``range``, ``date``, ``datetime``, ``ink``, ``rng``
    (``ink_color``), ``bar`` (``bar_width``), ``sides`` (``bar_sides``) and ``shade`` (``shade``: the bars and
    shades of the cells of the ranking) and the globals ``method_href`` (``metodo.method_href``: link to a
    subpage of the method, or to the method itself while the subpage has no text) and ``pollster_href``
    (``slugs.pollster_href``: link to the profile of a pollster). Without a booted App the
    environment is built but not cached.

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
            'prob': lambda p, digits=1: fmt_prob(p, digits),  # tables show one decimal (Luis, 2026-10-14)
            'range': fmt_range,
            'date': fmt_date,
            'datetime': fmt_datetime,
            'ink': ink_color,
            'bar': bar_width,
            'sides': bar_sides,
            'shade': shade,
            'shade_around': shade_around,
            'gradient': gradient,
            'rng': rng_cell,
            'date_short': fmt_date_short,
            'delta': delta_text,
            'badge': delta_badge,
        })
        # imported here: ``metodo`` and ``slugs`` import this module
        from .metodo import method_href
        from .slugs import pollster_href
        env.globals['method_href'] = method_href
        env.globals['pollster_href'] = pollster_href

        if app is not None:
            app.set('templates', env)

    return env


def parse_state(query: dict, scope: Optional[str] = None) -> dict:
    """
    Validate the page parameters of a request.

    Parameters
    ----------
    query : dict
        Query string parameters (``scope``, ``mode``, ``run``; others are ignored).
    scope : str, optional
        Scope of the route of an event page (already checked by ``resolve_event_scope``); when
        given it wins over the ``scope`` of the query, which is then ignored.

    Returns
    -------
    dict
        ``{'scope': str or None, 'mode': str or None, 'run': str or None}``; an empty value counts
        as missing and a missing ``mode`` stays ``None`` (``common_context`` resolves it with
        ``default_mode``).

    Raises
    ------
    HttpError
        400 when a given parameter is invalid.
    """
    if scope is None:
        scope = query.get('scope') or None

        if scope is not None:
            scope = check_scope(scope)

    mode = query.get('mode') or None

    return {
        'scope': scope,
        'mode': check_mode(mode) if mode is not None else None,
        'run': check_run(query.get('run')),
    }


def default_mode(frozen: bool) -> str:
    """
    Mode an event page shows when the request does not choose one.

    Parameters
    ----------
    frozen : bool
        Whether the publication is frozen (``manifest.freeze.active``).

    Returns
    -------
    str
        ``'forecast'`` while frozen (the final forecast is what matters and "today" no longer
        gets polls), otherwise ``'nowcast'``.
    """
    return 'forecast' if frozen else 'nowcast'


def analysis_part(scope: str, name: str) -> Optional[dict]:
    """
    Data of a part of the published houses analysis of a scope.

    Parameters
    ----------
    scope : str
        Scope code.
    name : str
        Part name, one of ``bundle.ANALYSIS_PARTS``.

    Returns
    -------
    dict or None
        The ``data`` member of the part envelope, or ``None`` when the part is not published.

    Raises
    ------
    HttpError
        Any error of the bundle other than the 404 of an unpublished analysis.
    """
    return bundle_part(site().analysis, scope, name)


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
        order.

    Raises
    ------
    HttpError
        503 when nothing is published; 404 when the given scope is not published.
    """
    manifest = site().manifest_data()

    if scope is None:
        codes = [row['code'] for row in published_scopes()]

        if not codes:
            raise HttpError(503, 'no bundle published yet')

        return codes[0]

    if scope not in manifest['scopes']:
        raise HttpError(404, 'scope not published')

    return scope


def resolve_event_scope(raw: str) -> str:
    """
    Scope of the route of an event page (``/{scope}``, ``/{scope}/escanos``).

    Parameters
    ----------
    raw : str
        Path segment captured by the route (lower case letters, digits, ``_`` and ``-``).

    Returns
    -------
    str
        The scope, when it is in the catalogue and published.

    Raises
    ------
    HttpError
        404 ``unknown scope`` outside the catalogue; 404 ``scope not published`` for a catalogue
        scope without a published forecast; 503 without a bundle.
    """
    if raw not in scope_codes():
        raise HttpError(404, 'unknown scope')

    return resolve_scope(raw)


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


def event_query(state: dict) -> str:
    """
    Query string that keeps the state of an event page in its links (the scope is in the path).

    Parameters
    ----------
    state : dict
        ``mode``, ``run`` and ``pinned``; ``run`` is kept only when ``pinned`` is true.

    Returns
    -------
    str
        ``'?mode=nowcast'``, plus ``&run=...`` for a pinned run.
    """
    params = [('mode', state.get('mode'))]

    if state.get('pinned'):
        params.append(('run', state.get('run')))

    return '?' + urlencode([(key, value) for key, value in params if value])


def scope_query(scope: str) -> str:
    """
    Query string of a global page filtered by a scope.

    Parameters
    ----------
    scope : str
        Scope code.

    Returns
    -------
    str
        ``'?scope=es'``.
    """
    return '?' + urlencode([('scope', scope)])


def days_left(event_date: Optional[str], today: Optional[date] = None) -> Optional[int]:
    """
    Days from today to an election.

    Parameters
    ----------
    event_date : str or None
        ISO date of the election.
    today : datetime.date, optional
        Reference day; today in Europe/Madrid by default.

    Returns
    -------
    int or None
        Days left (0 on election day, negative after it), or ``None`` when the date is missing
        or invalid.
    """
    when = _parse_datetime(event_date)

    if when is None:
        return None

    if today is None:
        today = datetime.now(_madrid_tz()).date()

    return (when.date() - today).days


def days_text(days: Optional[int]) -> Optional[str]:
    """
    Sentence of the days left to an election.

    Parameters
    ----------
    days : int or None
        Days left (``days_left``).

    Returns
    -------
    str or None
        ``'Faltan {n} días'``, ``'Falta 1 día'``, ``'Jornada electoral'`` (0) or
        ``'Celebradas hace {n} días'`` (negative); ``None`` without a date.
    """
    if days is None:
        return None

    if days > 1:
        return 'Faltan {} días'.format(days)

    if days == 1:
        return 'Falta 1 día'

    if days == 0:
        return 'Jornada electoral'

    return 'Celebradas hace {} día{}'.format(-days, '' if days == -1 else 's')


def event_title(scope: str, scopes: list) -> str:
    """
    Name of the event (the next election) of a scope.

    Parameters
    ----------
    scope : str
        Scope code.
    scopes : list of dict
        Rows of ``published_scopes()`` (``code``, ``name``).

    Returns
    -------
    str
        ``'Elecciones generales'`` for ``es``, otherwise ``'Elecciones autonómicas · {name}'`` with
        the catalogue name (the code when the scope is not in ``scopes``).
    """
    if scope == 'es':
        return 'Elecciones generales'

    names = {row['code']: row['name'] for row in scopes}

    return 'Elecciones autonómicas · {}'.format(names.get(scope, scope))


def chamber_text(scope: str, meta: dict) -> Optional[str]:
    """
    Name of the chamber of a scope with its seats.

    Parameters
    ----------
    scope : str
        Scope code.
    meta : dict
        Data of the meta part of the run (``n_seats``).

    Returns
    -------
    str or None
        ``'Congreso, 350 escaños'`` for ``es``, ``'Parlamento, {n} escaños'`` for the rest; ``None`` without
        ``n_seats``.
    """
    seats = meta.get('n_seats')

    if seats is None:
        return None

    return '{}, {} escaños'.format(CHAMBERS.get(scope, DEFAULT_CHAMBER), fmt_int(seats))


def tabbar(scope: str, active: Optional[str], active_suffix: str, event: bool, query: str) -> list:
    """
    Entries of the bottom tab bar.

    Parameters
    ----------
    scope : str
        Scope of the page (the event shown, or the scope the global page reads).
    active : str or None
        ``href`` of the current global page in ``PAGES``; ``None`` on an event page.
    active_suffix : str
        Suffix of the current tab of the event bar (``''`` or ``'/escanos'``), on an event page.
    event : bool
        Whether the page is an event page: its links keep ``query``.
    query : str
        ``event_query`` of the page state (``''`` outside an event).

    Returns
    -------
    list of dict
        One entry per row of ``TABBAR``: ``key``, ``label``, ``icon``, ``href`` and ``active``. Promedios and
        Escaños link the event of ``scope`` (with ``query`` on an event page), Histórico keeps the scope in its
        query and Ranking goes to ``/casas``; the active one is the tab of an event page or the global page of
        ``TABBAR_ACTIVE`` (none for the method).
    """
    keep = query if event else ''
    hrefs = {
        'inicio': '/',
        'promedios': '/' + scope + keep,
        'escanos': '/' + scope + '/escanos' + keep,
        'historico': '/elecciones' + scope_query(scope),
        'ranking': '/casas',
    }

    if event:
        lit = 'escanos' if active_suffix == '/escanos' else 'promedios'
    else:
        lit = TABBAR_ACTIVE.get(active)

    return [dict(row, href=hrefs[row['key']], active=row['key'] == lit) for row in TABBAR]


def event_info(scope: str, meta: dict, active_suffix: str, state: dict, today: Optional[date] = None,
               scopes: Optional[list] = None) -> dict:
    """
    Data of the event bar of an event page.

    Parameters
    ----------
    scope : str
        Scope code.
    meta : dict
        Data of the meta part of the run shown (``event_date``).
    active_suffix : str
        Suffix of the current tab in ``EVENT_TABS`` (``''`` or ``'/escanos'``).
    state : dict
        Resolved page state (``mode``, ``run``, ``pinned``).
    today : datetime.date, optional
        Reference day for ``days_left``.
    scopes : list of dict, optional
        Rows of ``published_scopes()``, read when not given.

    Returns
    -------
    dict
        ``title`` (``event_title``), ``date`` (``event_date``), ``days_left``, ``days_text``
        (``days_text``), ``chamber`` (``chamber_text``), ``run_at`` and ``n_polls`` (of the run), ``href``
        (``'/{scope}'``) and ``tabs``: ``{'href', 'label', 'active'}`` per entry of ``EVENT_TABS``, the links
        keeping the state (``event_query``).
    """
    if scopes is None:
        scopes = published_scopes()

    event_date = meta.get('event_date')
    days = days_left(event_date, today)
    query = event_query(state)

    return {
        'title': event_title(scope, scopes),
        'date': event_date,
        'days_left': days,
        'days_text': days_text(days),
        'chamber': chamber_text(scope, meta),
        'run_at': meta.get('run_at'),
        'n_polls': meta.get('n_polls'),
        'href': '/' + scope,
        'tabs': [{'href': '/' + scope + tab['suffix'] + query, 'label': tab['label'],
                  'active': tab['suffix'] == active_suffix} for tab in EVENT_TABS],
    }


def title_of(mode: str, meta: dict, frozen: bool, pinned: bool, is_latest: bool,
             today: Optional[date] = None) -> str:
    """
    Title of the estimate a page shows.

    Parameters
    ----------
    mode : str
        ``'nowcast'`` or ``'forecast'``.
    meta : dict
        Data of the meta part of the run shown (``event_date``, ``run_at``).
    frozen : bool
        Whether the publication is frozen.
    pinned : bool
        Whether the run was chosen with ``?run=``.
    is_latest : bool
        Whether the run shown is the latest of the scope in the manifest.
    today : datetime.date, optional
        Date of the nowcast title of the latest run; today in Europe/Madrid by default.

    Returns
    -------
    str
        In nowcast mode ``'Estimación a {today}'`` for the latest run and ``'Estimación del {run_at}'`` for
        an older pinned run or while frozen (never today's date over a run read before the veda); in forecast mode ``'Pronóstico definitivo para el {event_date}'`` only while
        frozen and showing the latest run, ``'Pronóstico publicado el {run_at}'`` for an older pinned run and
        ``'Pronóstico para el {event_date}'`` otherwise.
    """
    if mode == 'nowcast':
        if not is_latest or frozen:
            return 'Estimación del {}'.format(fmt_date(meta.get('run_at')))

        if today is None:
            today = datetime.now(_madrid_tz()).date()

        return 'Estimación a {}'.format(fmt_date(today.isoformat()))

    if frozen and is_latest:
        return 'Pronóstico definitivo para el {}'.format(fmt_date(meta.get('event_date')))

    if pinned and not is_latest:
        return 'Pronóstico publicado el {}'.format(fmt_date(meta.get('run_at')))

    return 'Pronóstico para el {}'.format(fmt_date(meta.get('event_date')))


def api_url(scope: str, part: str, mode: Optional[str] = None, run: Optional[str] = None,
            fmt: Optional[str] = None, kind: str = 'forecast', table: Optional[str] = None,
            date: Optional[str] = None) -> str:
    """
    URL of a part in the ``/api/v1`` routes.

    Parameters
    ----------
    scope : str
        Scope code.
    part : str
        Part name; ``'meta'`` is the scope route itself in the forecast routes; the table name
        (``'ratings'``...) in the database routes.
    mode : str, optional
        Simulation mode, for the mode parts of the forecast.
    run : str, optional
        Run id, for the forecast.
    fmt : str, optional
        Output format (``'json'`` or ``'csv'``).
    kind : str, optional
        ``'forecast'`` (the default), ``'analysis'`` (the houses analysis), ``'data'`` (the
        tables of the database), ``'events'`` (the list of elections of a scope), ``'event'``
        (a part of a past election, needs ``date``) or ``'backtest'``.
    table : str, optional
        CSV table of an analysis part (``'events'`` for the herding by cycle) or of the model of
        an event (``'shares'``, ``'seats'`` or ``'blocks'``).
    date : str, optional
        Election date, for ``kind='event'``.

    Returns
    -------
    str
        ``'/api/v1/forecast/{scope}[/{mode}]/{part}[?run=...&format=...]'``,
        ``'/api/v1/analysis/{scope}/{part}[?format=...&table=...]'``,
        ``'/api/v1/{part}?scope={scope}[&format=...]'``, ``'/api/v1/events/{scope}[?format=...]'``,
        ``'/api/v1/events/{scope}/{date}/{part}[?format=...&table=...]'`` or
        ``'/api/v1/backtest/{scope}/{part}[?format=...]'``.
    """
    if kind == 'analysis':
        url = '/api/v1/analysis/{}/{}'.format(scope, part)
        params = (('format', fmt), ('table', table))
    elif kind == 'events':
        url = '/api/v1/events/' + scope
        params = (('format', fmt),)
    elif kind == 'event':
        url = '/api/v1/events/{}/{}/{}'.format(scope, date, part)
        params = (('format', fmt), ('table', table))
    elif kind == 'backtest':
        url = '/api/v1/backtest/{}/{}'.format(scope, part)
        params = (('format', fmt),)
    elif kind == 'data':
        url = '/api/v1/' + part
        params = (('scope', scope), ('format', fmt))
    else:
        url = '/api/v1/forecast/' + scope

        if mode:
            url += '/' + mode

        if part != 'meta':
            url += '/' + part

        params = (('run', run), ('format', fmt))

    params = [(key, value) for key, value in params if value]

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


def hemicycle_order(names: list, parties: Catalog) -> list:
    """
    Party names in the order of the hemicycle: by block (``HEMICYCLE_BLOCKS``, unknown blocks last) and, within a
    block, by the order of the catalogue (``Catalog.order`` of ``catalog.js`` does the same in the browser).

    Parameters
    ----------
    names : list of str
        Party names.
    parties : Catalog
        Party catalogue of the run (its insertion order is the catalogue order).

    Returns
    -------
    list of str
        The names, sorted.
    """
    positions = {name: index for index, name in enumerate(parties)}

    def rank(name):
        block = parties[name]['block']
        block_rank = HEMICYCLE_BLOCKS.index(block) if block in HEMICYCLE_BLOCKS else len(HEMICYCLE_BLOCKS)
        return block_rank, positions.get(name, len(positions))

    return sorted(names, key=rank)


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


def common_context(state: dict, active: Optional[str], event: bool = False, active_suffix: str = '') -> dict:
    """
    Template context shared by every page.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``); on an event page ``scope`` comes from the
        route (``resolve_event_scope``).
    active : str or None
        ``href`` of the current page in the main menu; ``None`` for the event pages, which are
        not in it.
    event : bool, optional
        Whether the page is an event page (Promedios or Escaños): it gets the event bar and links
        that keep ``mode`` and ``run``; a global page gets links that keep ``scope``.
    active_suffix : str, optional
        Suffix of the current tab of the event bar (``EVENT_TABS``).

    Returns
    -------
    dict
        ``site_title``, ``site_tagline``, ``site_host`` (the host of ``settings()['site_url']``, for the signature of the
        downloadable images), ``author_handle``, ``author_url``, ``pages`` (``href``, ``label``, ``active``), ``state`` (``scope``,
        ``mode`` resolved with ``default_mode``, ``run``, ``pinned``), ``query`` (``event_query``
        on an event page, ``scope_query`` on a global one), ``scope_query``, ``scopes`` and
        ``scope_filter`` (``published_scopes()``, the options of the scope filter),
        ``filter_action`` (``active``, target of the scope filter), ``is_event``, ``event_bar``
        (``event_info``, or ``None`` on a global page), ``manifest``, ``meta``, ``catalog``,
        ``freeze`` (the manifest switch),
        ``frozen`` (whether it is active), ``title_suffix`` (``title_of``) and ``tabbar`` (``tabbar``, the
        bottom bar of the phones).

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope or a missing run.
    """
    scope = resolve_scope(state['scope'])
    pinned = state['run'] is not None
    latest = site().latest_run(scope)
    run = state['run'] or latest
    meta = part(scope, run, 'meta')

    manifest = site().manifest_data()
    # a manifest written before phase 6 has no ``since``: complete the switch for StrictUndefined
    freeze = {'active': False, 'message': None, 'since': None, **(manifest.get('freeze') or {})}
    frozen = bool(freeze.get('active'))
    mode = state['mode'] or default_mode(frozen)
    full = {'scope': scope, 'mode': mode, 'run': run, 'pinned': pinned}
    scopes = published_scopes()
    info = event_info(scope, meta, active_suffix, full, scopes=scopes) if event else None

    return {
        'site_title': SITE_TITLE,
        'site_tagline': SITE_TAGLINE,
        'site_host': urlsplit(settings()['site_url']).hostname or '',
        'author_handle': AUTHOR_HANDLE,
        'author_url': AUTHOR_URL,
        'pages': [dict(page, active=page['href'] == active) for page in PAGES],
        'state': full,
        'query': event_query(full) if event else scope_query(scope),
        'scope_query': scope_query(scope),
        'scopes': scopes,
        'scope_filter': scopes,
        'filter_action': active,
        'is_event': event,
        'event_bar': info,
        'manifest': manifest,
        'freeze': freeze,
        'frozen': frozen,
        'meta': meta,
        'catalog': catalog(meta),
        'title_suffix': title_of(mode, meta, frozen, pinned, run == latest),
        'tabbar': tabbar(scope, active, active_suffix, event, event_query(full) if event else ''),
    }


def bundle_part(read, *args) -> Optional[dict]:
    """
    Data of a published part, or ``None`` when it is not published.

    Parameters
    ----------
    read : callable
        Reader of the site that returns the bytes of the part envelope (``site().analysis``,
        ``site().event`` or ``site().backtest``).
    *args
        Arguments of the reader.

    Returns
    -------
    dict or None
        The ``data`` member of the part envelope, or ``None`` when the reader answers 404.

    Raises
    ------
    HttpError
        Any error of the reader other than the 404 of an unpublished part.
    """
    try:
        return bundle.loads(read(*args))['data']
    except HttpError as error:
        if error.status == 404:
            return None

        raise
