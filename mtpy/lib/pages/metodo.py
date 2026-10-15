"""
Method page contexts: the landing page (``/metodo``) and its subpages (``/metodo/{section}``), with the live
figures of the general election (simulations and backtest coverage), the thresholds of the scopes catalogue and,
on the ``promedio`` subpage, the live example of the average (the ``weights`` part of the latest national run).
"""
from typing import Optional

from ...core.api import HttpError
from ..webapi import catalogue, site
from .backtest import backtest_context, backtest_part
from .common import api_url, common_context, part
from .filters import _is_missing

# Scope whose figures the method texts quote ("en las generales"): the texts have no scope filter.
NATIONAL_SCOPE = 'es'
# Rows of the weights table of the example shown at first; "Cargar más" reveals the rest in batches of this size.
EXAMPLE_SHOWN = 20
# Keys of a row of the example that the chart needs (the island carries no more).
ISLAND_ROW_KEYS = ('date', 'pollster', 'sponsor', 'sample_size', 'pct', 'pct_adj', 'weight', 'weight_kernel',
                   'weight_over', 'weight_sample', 'weight_rating')
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


def weights_example() -> Optional[dict]:
    """
    The live example of the average: the polls that weigh on the average of the party leading the vote of the
    latest national run, on the day that run read its estimate (the ``weights`` part).

    Returns
    -------
    dict or None
        ``name``, ``fullname``, ``color`` (from ``meta``), ``date`` (the day read), ``run``, ``stats`` (those of
        the leader), ``rows`` (the rows of the leader, by date), ``n_rows``, ``weight_max`` (scale of the weight
        bar: 3 unless a weight exceeds it), ``line`` and ``kernel``. ``None`` without a national run, when the run
        has no ``weights`` part (published before the part existed), when the headline has no parties or when the
        leader has no fit in the part.

    Raises
    ------
    HttpError
        Any error of the bundle other than a 404.
    """
    try:
        run = site().latest_run(NATIONAL_SCOPE)
        parties = (part(NATIONAL_SCOPE, run, 'headline').get('nowcast') or {}).get('parties') or []
        if not parties:
            return None
        leader = max(parties, key=lambda row: row['pct'] if row.get('pct') is not None else float('-inf'))['name']
        weights = part(NATIONAL_SCOPE, run, 'weights')
    except HttpError as error:
        if error.status != 404:
            raise

        return None

    if leader not in (weights.get('parties') or []):
        return None

    catalogue = next((row for row in part(NATIONAL_SCOPE, run, 'meta').get('parties') or [] if row['name'] == leader), {})
    rows = sorted((row for row in weights['rows'] if row['name'] == leader), key=lambda row: row['date'])

    return {
        'name': leader,
        'fullname': catalogue.get('fullname'),
        'color': catalogue.get('color'),
        'date': weights['date'],
        'run': run,
        'stats': weights['stats'][leader],
        'rows': rows,
        'n_rows': len(rows),
        'weight_max': max([3.0] + [row['weight'] for row in rows if row.get('weight') is not None]),
        'line': weights['line'][leader],
        'kernel': weights['kernel'][leader],
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
        ``section``. The ``promedio`` subpage adds ``example`` (``weights_example``, ``None`` without one),
        ``example_shown`` (``EXAMPLE_SHOWN``), ``example_hidden`` (whether rows beyond those are hidden),
        ``example_csv`` (the CSV of the ``weights`` part) and, with an example, the ``initial`` island
        (``state`` and the ``example`` slice the chart draws: ``name``, ``color``, ``date``, ``stats``, ``rows``
        with ``ISLAND_ROW_KEYS``, ``line``, ``kernel``).

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

    if section == 'promedio':
        example = weights_example()
        context['example'] = example
        context['example_shown'] = EXAMPLE_SHOWN
        context['example_hidden'] = example is not None and example['n_rows'] > EXAMPLE_SHOWN
        context['example_csv'] = api_url(NATIONAL_SCOPE, 'weights', fmt='csv')

        if example is not None:
            context['initial'] = {
                'state': context['state'],
                'example': {
                    'name': example['name'], 'color': example['color'], 'date': example['date'],
                    'stats': example['stats'],
                    'rows': [{key: row.get(key) for key in ISLAND_ROW_KEYS} for row in example['rows']],
                    'line': example['line'], 'kernel': example['kernel'],
                },
            }

    return context
