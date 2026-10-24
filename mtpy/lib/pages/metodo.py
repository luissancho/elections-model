"""
Method page contexts: the landing page (``/metodo``) and its subpages (``/metodo/{section}``), with the live
figures of the general election (simulations and backtest coverage), the thresholds of the scopes catalogue, on the
``promedio`` subpage the live examples of the average (the ``weights`` part of the latest national run) and of the
house effects (the last poll of one pollster before every analysed election since a date, from the ``poll-errors``
analyses) and, on the ``rating`` subpage, the public rubric of the prior quality
(``data/pollsters-quality-criteria.json``).
"""
from typing import Optional

from ...core.api import HttpError
from ..quality import criteria_keys, load_criteria, max_points
from ..studies.lean.stats import sign_chance
from ..webapi import catalogue, site
from .backtest import backtest_context, backtest_part
from .casas import election_label
from .common import analysis_part, api_url, bundle_part, common_context, part
from .elecciones import error_blocks, history_events, history_rows, history_scopes, poll_error_parts
from .filters import _is_missing
from .slugs import PROFILE_PATH, slugify

# Scope whose figures the method texts quote ("en las generales"): the texts have no scope filter.
NATIONAL_SCOPE = 'es'
# Keys of a row of the example that the chart needs (the island carries no more).
ISLAND_ROW_KEYS = ('date', 'pollster', 'sponsor', 'sample_size', 'pct', 'pct_adj', 'weight', 'weight_kernel',
                   'weight_over', 'weight_sample', 'weight_rating')
# The live example of the house effects: the last poll of this pollster before every analysed election since this
# date (the CIS from September 2018, when its vote estimate changed), against the result, in these two blocks
# (``left`` is counted as overestimated, ``right`` as underestimated).
HOUSE_EXAMPLE = {'pollster': 'CIS', 'since': '2018-09-01', 'left': 'Izquierda', 'right': 'Derecha'}
# Keys of a row of the house effects example that its bullseye needs, plus the two blocks.
HOUSE_ROW_KEYS = ('event', 'event_date', 'days', 'scope', 'name')
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
        (those of the backtest, empty without one), ``years`` (the distinct years of the events, ascending) and
        ``backtest`` (the row of the shortest horizon: ``horizon``, ``n_cases``, ``mae`` (of the model),
        ``mae_last_poll``, ``mae_mean_4w`` and ``cov95``; ``None`` without a backtest).

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
    first = rows[0] if rows else None
    events = meta.get('events') or []

    return {
        'n_sim': n_sim,
        'coverage': [{'horizon': row['horizon'], 'cov95': row['cov_shares95']} for row in rows],
        'events': events,
        'horizons': meta.get('horizons') or [],
        'years': sorted({str(event)[:4] for event in events}),
        'backtest': None if first is None else {
            'horizon': first['horizon'], 'n_cases': first.get('n_cases'), 'mae': first.get('mae_shares'),
            'mae_last_poll': first.get('mae_last_poll'), 'mae_mean_4w': first.get('mae_mean_4w'),
            'cov95': first['cov_shares95'],
        },
    }


def weights_example() -> Optional[dict]:
    """
    The live example of the average: the polls that weigh on the average of the party leading the vote of the
    latest national run, on the day that run read its estimate (the ``weights`` part).

    Returns
    -------
    dict or None
        ``name``, ``fullname``, ``color`` (from ``meta``), ``date`` (the day the average is read, the centre of
        the kernel), ``published`` (the day of the run, ``run_at``, the date the texts show), ``run``, ``stats``
        (those of the leader), ``rows`` (the rows of the leader, newest first), ``n_rows``, ``line``, ``kernel``
        and ``top`` (``pollster``, ``sponsor``, ``date`` and ``weight`` of its heaviest poll, the newest on a tie;
        ``None`` without rows). ``None`` without a national run, when the run
        has no ``weights`` part (published before the part existed), when the headline has no parties or when the
        leader has no fit in the part.

    Raises
    ------
    HttpError
        Any error of the bundle other than a 404.
    """
    try:
        run = site().latest_run(NATIONAL_SCOPE)
        headline = part(NATIONAL_SCOPE, run, 'headline')
        parties = (headline.get('nowcast') or {}).get('parties') or []
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
    # Newest first: the reader looks for the latest polls
    rows = sorted((row for row in weights['rows'] if row['name'] == leader), key=lambda row: row['date'], reverse=True)

    # The heaviest poll of the leader (the newest on a tie), the one the text names
    heaviest = max(rows, key=lambda row: (row.get('weight') or 0, row['date']), default=None)
    top = {key: heaviest.get(key) for key in ('pollster', 'sponsor', 'date', 'weight')} if heaviest else None

    return {
        'name': leader,
        'fullname': catalogue.get('fullname'),
        'color': catalogue.get('color'),
        'date': weights['date'],
        'published': (headline.get('run_at') or '')[:10] or weights['date'],
        'run': run,
        'stats': weights['stats'][leader],
        'rows': rows,
        'n_rows': len(rows),
        'line': weights['line'][leader],
        'kernel': weights['kernel'][leader],
        'top': top,
    }


def cycle_counts() -> Optional[dict]:
    """
    The polls and pollsters of the current cycle of ``NATIONAL_SCOPE`` that enter the average: the rows of the
    ``polls`` part of its latest run (the polls that pass the inclusion filters, those that weigh 0 for overlapping
    with a later delivery included).

    Returns
    -------
    dict or None
        ``n_polls`` (the rows with a pollster) and ``n_pollsters`` (their distinct pollsters). ``None`` without a
        national run, without the part or without polls.

    Raises
    ------
    HttpError
        Any error of the bundle other than a 404.
    """
    try:
        run = site().latest_run(NATIONAL_SCOPE)
        polls = part(NATIONAL_SCOPE, run, 'polls').get('polls') or []
    except HttpError as error:
        if error.status != 404:
            raise

        return None

    pollsters = [row['pollster_id'] for row in polls if row.get('pollster_id') is not None]

    if not pollsters:
        return None

    return {'n_polls': len(pollsters), 'n_pollsters': len(set(pollsters))}


def house_effects_example() -> Optional[dict]:
    """
    The live example of the house effects: the error of the last poll of the pollster of ``HOUSE_EXAMPLE`` before
    every analysed election of every scope since its ``since`` date, by block (the rows of the bullseye of its
    profile), and its effects in the current cycle of ``NATIONAL_SCOPE``.

    Returns
    -------
    dict or None
        ``pollster``, ``since``, ``href`` (its profile), ``blocks`` (the pair of the bullseye axes, the first two
        of ``error_blocks``), ``left`` and ``right`` (the blocks the text counts, ``HOUSE_EXAMPLE``), ``rows``
        (its ``history_rows`` since the date, newest first and, on the same date, in catalogue order, each with
        ``name``, the name of its election, ``election_label``), ``n`` (their number), ``over`` and ``under``
        (``{block: n}`` for ``left`` and ``right``: the elections in which the poll gave the block more, or less,
        vote than the result), ``chance`` (the ``sign_chance`` of ``over`` for ``left`` and of ``under`` for
        ``right``, ``{block: dict or None}``), ``first`` and ``last`` (the dates of the oldest and the newest rows),
        ``cycle``
        (``{name: effect}``, the ``cycle`` effect of the parties and blocks of the ``house-effects`` analysis of
        ``NATIONAL_SCOPE``; ``None`` without one or without a row of the pollster) and ``n_cycle`` (its polls in
        that cycle, ``None`` without ``cycle``). ``None`` without an analysed scope, when the axes are not the two
        blocks of the example or when the pollster has no row since the date.

    Raises
    ------
    HttpError
        Any error of the bundle other than a 404.
    """
    scopes = history_scopes()
    parts = poll_error_parts(scopes)
    blocks = error_blocks(scopes, parts)[:2]
    name, since = HOUSE_EXAMPLE['pollster'], HOUSE_EXAMPLE['since']
    left, right = HOUSE_EXAMPLE['left'], HOUSE_EXAMPLE['right']

    if not parts or set(blocks) != {left, right}:
        return None

    events = history_events(scopes)
    names = {(event['scope'], event['date']): event['name'] for event in events if event.get('name')}
    scope_names = {scope['code']: scope['name'] for scope in scopes}
    order = {scope['code']: index for index, scope in enumerate(scopes)}
    rows = [dict(row, name=election_label(row['scope'], row['event_date'], names, scope_names))
            for row in history_rows(scopes, events, parts)
            if row.get('pollster') == name and (row.get('event_date') or '') >= since]
    rows.sort(key=lambda row: order[row['scope']])
    rows.sort(key=lambda row: row['event_date'], reverse=True)

    if not rows:
        return None

    def count(block, sign):
        """Elections in which the poll erred in the direction of `sign` (1 over, -1 under) in `block`."""
        return sum(1 for row in rows if row.get(block) is not None and sign * row[block] > 0)

    def chance(block, sign):
        """`sign_chance` of the elections with an error in `block` erring in the direction of `sign`."""
        return sign_chance(sum(1 for row in rows if row.get(block) is not None), count(block, sign))

    effects = analysis_part(NATIONAL_SCOPE, 'house-effects') or {}
    mine = [row for row in effects.get('rows') or [] if row.get('pollster') == name]
    cycle = {row['name']: row.get('cycle') for row in mine} or None

    return {
        'pollster': name,
        'since': since,
        'href': '{}/{}'.format(PROFILE_PATH, slugify(name)),
        'blocks': blocks,
        'left': left,
        'right': right,
        'rows': rows,
        'n': len(rows),
        'over': {block: count(block, 1) for block in (left, right)},
        'under': {block: count(block, -1) for block in (left, right)},
        'chance': {left: chance(left, 1), right: chance(right, -1)},
        'first': rows[-1]['event_date'],
        'last': rows[0]['event_date'],
        'cycle': cycle,
        'n_cycle': max(row.get('n_cycle') or 0 for row in mine) if cycle else None,
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


def quality_rubric() -> dict:
    """
    The public rubric of the prior quality, as the ``rating`` subpage shows it.

    Returns
    -------
    dict
        ``rows`` (one ``{'key', 'name', 'max', 'levels', 'note'}`` per criterion, in the order of the catalogue, with
        ``note`` the text the catalogue adds to the criterion or ``None``, and ``levels`` a list of
        ``{'points': int, 'text': str}`` from the highest level to 0), ``max_points`` (the sum of the maxima), the ``per_point`` and ``floor`` of the scale and ``cap`` (``per_point · max_points``, the
        highest prior quality a pollster can get).

    Raises
    ------
    ValueError
        If the catalogue of criteria is not valid (``quality.load_criteria``).
    """
    criteria = load_criteria()
    scale = criteria['scale']
    rows = []
    for key in criteria_keys(criteria):
        entry = criteria['criteria'][key]
        levels = [{'points': points, 'text': entry['levels'][str(points)]} for points in range(entry['max'], -1, -1)]
        rows.append({'key': key, 'name': entry['name'], 'max': entry['max'], 'levels': levels,
                     'note': entry.get('note')})
    total = max_points(criteria)

    return {'rows': rows, 'max_points': total, 'per_point': scale['per_point'], 'floor': scale['floor'],
            'cap': scale['per_point'] * total}


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
        ``section``. The ``rating`` subpage adds ``rubric`` (``quality_rubric``). The ``promedio`` subpage adds
        ``example`` (``weights_example``, ``None`` without one), ``example_csv`` (the CSV of the ``weights`` part),
        ``house_example`` (``house_effects_example``, ``None`` without one), ``cycle`` (``cycle_counts``) and,
        with either example, the
        ``initial`` island (``state``, the ``example`` slice the weights chart draws: ``name``, ``color``, ``date``,
        ``stats``, ``rows`` with ``ISLAND_ROW_KEYS``, ``line``, ``kernel``; and the ``house`` slice its bullseye
        draws: ``blocks`` and ``rows`` with ``HOUSE_ROW_KEYS`` plus the blocks).

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

    if section == 'rating':
        context['rubric'] = quality_rubric()

    if section == 'promedio':
        example = weights_example()
        house = house_effects_example()
        context['example'] = example
        context['example_csv'] = api_url(NATIONAL_SCOPE, 'weights', fmt='csv')
        context['house_example'] = house
        context['cycle'] = cycle_counts()
        context['lean_published'] = bundle_part(site().study, 'sesgo-cis', 'meta') is not None
        initial = {'state': context['state']}

        if example is not None:
            initial['example'] = {
                'name': example['name'], 'color': example['color'], 'date': example['date'],
                'stats': example['stats'],
                'rows': [{key: row.get(key) for key in ISLAND_ROW_KEYS} for row in example['rows']],
                'line': example['line'], 'kernel': example['kernel'],
            }

        if house is not None:
            keys = HOUSE_ROW_KEYS + tuple(house['blocks'])
            initial['house'] = {'blocks': house['blocks'], 'rows': [{key: row.get(key) for key in keys} for row in house['rows']]}

        if len(initial) > 1:
            context['initial'] = initial

    return context
