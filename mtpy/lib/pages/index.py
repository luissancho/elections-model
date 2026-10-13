"""
Context of the headline page: until phase 7 merges it with the average page, the Promedios tab of an event and the
home page.
"""
from typing import Optional

from .. import bundle
from ..webapi import site
from .common import Catalog, block_color, common_context, part
from .filters import _is_missing


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


def index_context(state: dict, active: Optional[str] = '/', event: bool = False, active_suffix: str = '') -> dict:
    """
    Template context of the headline page.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    active : str or None, optional
        ``href`` of the page in the navigation (``None`` on an event page).
    event : bool, optional
        Whether it is rendered as an event page (see ``common_context``).
    active_suffix : str, optional
        Suffix of the current tab of the event bar.

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
    context = common_context(state, active, event=event, active_suffix=active_suffix)
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
