"""
Common parts of the web pages: routes, messages, the Jinja2 environment and the shared context.
"""
import os
import re
from typing import Optional
from urllib.parse import urlencode

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from ...core.api import HttpError
from ...core.app import App
from .. import bundle
from ..webapi import check_mode, check_run, check_scope, site
from .filters import fmt_date, fmt_datetime, fmt_int, fmt_num, fmt_pct, fmt_prob, fmt_range

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
TEMPLATES_DIR = os.path.join(REPO_ROOT, 'web', 'templates')
SITE_TITLE = 'Pronóstico electoral'
PAGES = (
    {'href': '/', 'label': 'Portada'},
    {'href': '/promedio', 'label': 'Promedio'},
    {'href': '/escanos', 'label': 'Escaños'},
    {'href': '/casas', 'label': 'Casas'},
    {'href': '/elecciones', 'label': 'Elecciones'},
    {'href': '/modelo', 'label': 'Modelo'},
    {'href': '/metodo', 'label': 'Método'},
)
ROUTES = [
    ('/promedio', 'promedio', 'index', ['GET']),
    ('/escanos', 'escanos', 'index', ['GET']),
    ('/casas', 'casas', 'index', ['GET']),
    ('/elecciones', 'elecciones', 'index', ['GET']),
    ('/modelo', 'modelo', 'index', ['GET']),
    ('/metodo', 'metodo', 'index', ['GET']),
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
    'pollster not found': 'Casa no encontrada.',
    'invalid pollster': 'Casa no válida.',
    'invalid date': 'Fecha no válida.',
    'event not published': 'Elección no publicada.',
    'backtest not published': 'Backtest no publicado.',
}
OTHERS_COLOR = '#9e9e9e'
MODE_LABELS = {'nowcast': 'Hoy', 'forecast': 'Elección'}
_COLOR_RE = re.compile(r'#[0-9A-Fa-f]{3,8}')


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
        ``catalog``, ``mode_label``, ``freeze`` (the manifest switch), ``frozen`` (whether it is active)
        and ``title_suffix`` (``Pronóstico definitivo`` in forecast mode while frozen).

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

    manifest = site().manifest_data()
    # a manifest written before phase 6 has no ``since``: complete the switch for StrictUndefined
    freeze = {'active': False, 'message': None, 'since': None, **(manifest.get('freeze') or {})}
    frozen = bool(freeze.get('active'))

    if mode == 'nowcast':
        title_suffix = 'Estimación a {}'.format(fmt_date(meta.get('as_of')))
    elif frozen:
        title_suffix = 'Pronóstico definitivo para el {}'.format(fmt_date(meta.get('event_date')))
    else:
        title_suffix = 'Pronóstico para el {}'.format(fmt_date(meta.get('event_date')))

    return {
        'site_title': SITE_TITLE,
        'pages': [dict(page, active=page['href'] == active) for page in PAGES],
        'state': full,
        'query': query_string(full),
        'scopes': published_scopes(),
        'manifest': manifest,
        'freeze': freeze,
        'frozen': frozen,
        'meta': meta,
        'catalog': catalog(meta),
        'mode_label': MODE_LABELS[mode],
        'title_suffix': title_suffix,
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
