"""
Method page contexts: the landing page (``/metodo``) and its subpages (``/metodo/{section}``), with the live
figures of the general election (simulations and backtest coverage) and the thresholds of the scopes catalogue.
"""
from typing import Optional

from ...core.api import HttpError
from ..webapi import catalogue, site
from .backtest import backtest_context, backtest_part
from .common import common_context, part
from .filters import _is_missing

# Scope whose figures the method texts quote ("en las generales"): the texts have no scope filter.
NATIONAL_SCOPE = 'es'
# Cross-cutting sections of the landing page, which belong to no question of the method.
SECTIONS = (
    ('datos', 'Los datos'),
    ('publicacion', 'La publicación'),
    ('limitaciones', 'Limitaciones'),
)
# Subpages of the method (``/metodo/{id}``), in menu order. A subpage without ``template`` has no text yet: it
# is not linked and its route is a 404.
METODO_SECTIONS = {
    'promedio': {'title': '¿Cómo promediamos las encuestas?', 'template': 'metodo_promedio.html'},
    'escanos': {'title': '¿Cómo simulamos los escaños?', 'template': 'metodo_escanos.html'},
    'rating': {'title': '¿Cómo calculamos el rating de las encuestadoras?', 'template': 'metodo_rating.html'},
    'backtest': {'title': '¿Cuánto acierta el modelo?', 'template': 'metodo_backtest.html'},
}


def subpages(active: Optional[str]) -> list:
    """
    Subnavigation of the method.

    Parameters
    ----------
    active : str or None
        Id of the current subpage, ``None`` on the method landing page.

    Returns
    -------
    list of dict
        ``{'id', 'title', 'href', 'active', 'available'}`` per entry of ``METODO_SECTIONS``, in
        order; ``available`` tells whether the subpage has a text (``template``).
    """
    return [
        {'id': section, 'title': entry['title'], 'href': '/metodo/' + section, 'active': section == active,
         'available': entry['template'] is not None}
        for section, entry in METODO_SECTIONS.items()
    ]


def method_href(section: str) -> str:
    """
    Link to a subpage of the method, for the closing cards and the event bar.

    Parameters
    ----------
    section : str
        Id of the subpage in ``METODO_SECTIONS``.

    Returns
    -------
    str
        ``'/metodo/{section}'`` when the subpage has a text, otherwise ``'/metodo'``.
    """
    entry = METODO_SECTIONS.get(section)

    return '/metodo/' + section if entry and entry['template'] is not None else '/metodo'


def thresholds_rows() -> list:
    """
    Thresholds and rating weight of every scope of the catalogue, for the method texts.

    They are not bundle data: the web reads them from the scopes catalogue (``data/es-scopes.csv``).

    Returns
    -------
    list of dict
        ``{'code', 'name', 'threshold', 'threshold_scope', 'rating_weight', 'seats'}`` per scope, in catalogue
        order; ``threshold`` (the barrier in each district, in %) and ``threshold_scope`` (the barrier over the
        whole scope) are ``None`` where the catalogue leaves them empty.
    """
    def number(value) -> Optional[float]:
        return None if _is_missing(value) else float(value)

    return [
        {'code': code, 'name': row['name'], 'threshold': number(row['threshold']),
         'threshold_scope': number(row['threshold_scope']), 'rating_weight': number(row['rating_weight']),
         'seats': None if _is_missing(row['seats']) else int(row['seats'])}
        for code, row in catalogue().iterrows()
    ]


def national_figures() -> dict:
    """
    Live figures of the general election that the method texts quote.

    Returns
    -------
    dict
        ``n_sim`` (simulations of the latest published run of ``NATIONAL_SCOPE``, ``None`` without one),
        ``coverage`` (``[{'horizon', 'cov95'}]``: the share of results inside the 95 % interval by horizon in
        its published backtest, shortest horizon first; empty without a backtest), ``events`` and ``horizons``
        (those of the backtest, empty without one).

    Raises
    ------
    HttpError
        Any error of the bundle other than the 404 of an unpublished scope, run or backtest.
    """
    try:
        n_sim = part(NATIONAL_SCOPE, site().latest_run(NATIONAL_SCOPE), 'meta').get('n_sim')
    except HttpError as error:
        if error.status != 404:
            raise

        n_sim = None

    meta = backtest_part(NATIONAL_SCOPE, 'meta') or {}
    by_horizon = backtest_part(NATIONAL_SCOPE, 'by-horizon') or {}
    rows = sorted((row for row in by_horizon.get('rows') or [] if not _is_missing(row.get('cov_shares95'))),
                  key=lambda row: row['horizon'])

    return {
        'n_sim': n_sim,
        'coverage': [{'horizon': row['horizon'], 'cov95': row['cov_shares95']} for row in rows],
        'events': meta.get('events') or [],
        'horizons': meta.get('horizons') or [],
    }


def metodo_context(state: dict) -> dict:
    """
    Template context of the method landing page.

    The text lives in ``metodo.html``; the page has no scope filter and no ``initial`` data (its module,
    ``metodo.js``, only wires the header controls): its live figures are those of ``NATIONAL_SCOPE``.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).

    Returns
    -------
    dict
        The common context plus ``subpages`` (``subpages(None)``, the index of the four questions), ``figures``
        (``national_figures``), ``thresholds`` (``thresholds_rows``) and ``sections`` (the cross-cutting
        sections, ``{'id', 'title'}`` in ``SECTIONS`` order).

    Raises
    ------
    HttpError
        503 without a bundle; 404 for an unpublished scope or a missing run.
    """
    context = common_context(state, '/metodo')
    context['subpages'] = subpages(None)
    context['figures'] = national_figures()
    context['thresholds'] = thresholds_rows()
    context['sections'] = [{'id': section_id, 'title': title} for section_id, title in SECTIONS]

    return context


def metodo_section_context(state: dict, section: str) -> dict:
    """
    Template context of a subpage of the method.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    section : str
        Id of the subpage (a key of ``METODO_SECTIONS``).

    Returns
    -------
    dict
        The common context (for ``backtest``, the whole ``backtest_context``) plus ``subpages``
        (``subpages(section)``), ``figures`` (``national_figures``), ``thresholds`` (``thresholds_rows``) and
        ``section``.

    Raises
    ------
    HttpError
        404 ``section not found`` for an unknown subpage or one without text yet; 503 without a bundle; 404 for
        an unpublished scope or a missing run.
    """
    entry = METODO_SECTIONS.get(section)

    if entry is None or entry['template'] is None:
        raise HttpError(404, 'section not found')

    # only the backtest has its own data; the texts share the common context and the live figures
    context = backtest_context(state, '/metodo') if section == 'backtest' else common_context(state, '/metodo')
    context['subpages'] = subpages(section)
    context['figures'] = national_figures()
    context['thresholds'] = thresholds_rows()
    context['section'] = section

    return context
