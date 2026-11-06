"""
Method page contexts: the landing page (``/metodo``) and its subpages (``/metodo/{section}``), with the live
figures of the general election (simulations and backtest coverage), the thresholds of the scopes catalogue, on the
``promedio`` subpage the live examples of the average (the ``weights`` part of the latest national run) and of the
house effects (the last poll of one pollster before every analysed election since a date, from the ``poll-errors``
analyses) and, on the ``rating`` subpage, the public rubric of the prior quality
(``data/pollsters-quality-criteria.json``).
"""
import math
from typing import Optional

from ...core.api import HttpError
from ..quality import criteria_keys, load_criteria, max_points
from ..studies.lean.stats import sign_chance
from ..webapi import catalogue, site
from .backtest import backtest_context, backtest_part
from .casas import election_label
from .common import analysis_part, api_url, bundle_part, catalog, common_context, part
from .elecciones import error_blocks, history_events, history_rows, history_scopes, poll_error_parts
from .filters import DASH, _is_missing, delta_text, fmt_int, fmt_num
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
# Age (years since the first poll listing it) below which a party is young for the drift of the opinion: its
# drift is multiplied (``DriftEstimator.age_max``), and the live example of the seats subpage names them.
YOUNG_PARTY_YEARS = 4
# Keys of a row of the house effects example that its bullseye needs, plus the two blocks.
HOUSE_ROW_KEYS = ('event', 'event_date', 'days', 'scope', 'name')
# The rating check of the ``rating`` subpage: its part (published only for ``NATIONAL_SCOPE``), the populations of its
# tables in order with their public names, the step the limits of the cloud round to and the number of ticks of a
# half axis.
RATING_CHECK_PART = 'rating-check'
RATING_CHECK_POPULATIONS = (('all', 'Todas'), ('generales', 'Generales'), ('autonomicas', 'Autonómicas'))
RATING_CHECK_AXIS_STEP = 5
RATING_CHECK_AXIS_TICKS = 5
# Keys of a point of the cloud that its chart needs, plus its ``label`` (pollster and election, for the tooltip).
RATING_CHECK_ROW_KEYS = ('pollster', 'd_prev', 'd_obs')
# Small whole numbers as the texts of the method write them (``number_word``).
NUMBER_WORDS = ('cero', 'una', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve', 'diez')
# Cross-cutting sections of the landing page, which belong to no question of the method (the cards of the data
# and of the publication were removed on 2026-10-28).
SECTIONS = (
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


def error_example() -> Optional[dict]:
    """
    The live example of the three errors: the pieces of the margin of each main party of the latest national
    run, at the anchor of the run (``Hoy``) and at its election (``Elección``), from the ``fan`` part.

    Returns
    -------
    dict or None
        ``published`` (``run_at``), ``event_date``, ``horizon`` (days from the anchor to the election: the
        largest horizon of the fan, 0 on election day), ``n_events``, ``first_year`` and ``last_year`` (of the
        elections whose polls fit the polling-error estimator, ``meta.diagnostics.error_events``), ``scope`` (the
        estimator scope) and ``rows``, one per party of ``bmaps.main`` with a fit, by vote descending: ``name``,
        ``fullname``, ``color``, ``pct``, ``err`` (``s``), ``pct_err`` (``e``), ``dof``, ``sd_now`` (``σ`` at
        horizon 0), ``drift`` (``d`` at ``horizon``), ``sd_election``, ``lo`` and ``hi`` (the 95 % interval at
        ``horizon``), plus the diagnostics of the drift of the run (``meta.diagnostics``): ``drift_k`` (the
        constant of the relative random walk), ``multiplier`` (the drift multiplier of the young parties) and
        ``young`` (the simulated parties younger than four years at the anchor, by ``ages``; empty without
        ages). ``None`` without a national run, when its fan lacks the pieces (published before 2026-10-25)
        or when its meta lacks ``error_events``.

    Raises
    ------
    HttpError
        Any error of the bundle other than a 404.
    """
    try:
        run = site().latest_run(NATIONAL_SCOPE)
        meta = part(NATIONAL_SCOPE, run, 'meta')
        fan = part(NATIONAL_SCOPE, run, 'fan')
    except HttpError as error:
        if error.status != 404:
            raise

        return None

    rows = fan.get('rows') or []
    events = (meta.get('diagnostics') or {}).get('error_events')
    if not rows or not all('pct_err' in row for row in rows) or not events:
        return None

    horizon = max(int(h) for h in fan.get('horizons') or [0])
    at_now = {row['name']: row for row in rows if int(row['horizon']) == 0}
    at_election = {row['name']: row for row in rows if int(row['horizon']) == horizon}
    parties = catalog(meta)
    out = []
    for name in (meta.get('bmaps') or {}).get('main') or []:
        now, then = at_now.get(name), at_election.get(name)
        if now is None or then is None:
            continue  # no fit for the party: no row in the fan
        out.append({
            'name': name, 'fullname': parties[name]['fullname'], 'color': parties[name]['color'], 'pct': now['mean'],
            'err': now['err'], 'pct_err': now['pct_err'], 'drift': then['drift'], 'sd_now': now['sd'],
            'sd_election': then['sd'], 'lo': then['lo'], 'hi': then['hi'], 'dof': now['dof'],
        })
    out.sort(key=lambda row: -(row['pct'] or 0))

    diagnostics = meta.get('diagnostics') or {}
    ages = diagnostics.get('ages') or {}

    return {
        'published': meta.get('run_at'), 'event_date': meta.get('event_date'), 'horizon': horizon,
        'n_events': len(events), 'first_year': str(events[0])[:4], 'last_year': str(events[-1])[:4],
        'scope': diagnostics.get('error_scope'), 'rows': out,
        'drift_k': diagnostics.get('drift_k'), 'multiplier': diagnostics.get('multiplier'),
        'young': [name for name, age in ages.items() if age is not None and age < YOUNG_PARTY_YEARS],
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
        (its ``history_rows`` since the date with an error in both blocks, newest first and, on the same date, in
        catalogue order, each with ``name``, the name of its election, ``election_label``; the rows of a scope
        whose ``poll-errors`` part has no blocks lack their columns and stay out, as in the bullseye), ``n``
        (their number), ``over`` and ``under`` (``{block: n}`` for ``left`` and ``right``: the elections in which
        the poll gave the block more, or less, vote than the result), ``chance`` (the ``sign_chance`` of ``over``
        for ``left`` and of ``under`` for ``right``, ``{block: dict or None}``), ``first`` and ``last`` (the dates
        of the oldest and the newest rows), ``cycle``
        (``{name: effect}``, the ``cycle`` effect of the parties and blocks of the ``house-effects`` analysis of
        ``NATIONAL_SCOPE``; ``None`` without one or without a row of the pollster) and ``n_cycle`` (its polls in
        that cycle, ``None`` without ``cycle``). ``None`` without an analysed scope, when the axes are not the two
        blocks of the example or when the pollster has no row with both block errors since the date.

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
            if row.get('pollster') == name and (row.get('event_date') or '') >= since
            and all(row.get(block) is not None for block in (left, right))]
    rows.sort(key=lambda row: order[row['scope']])
    rows.sort(key=lambda row: row['event_date'], reverse=True)

    if not rows:
        return None

    def count(block, sign):
        """Elections in which the poll erred in the direction of `sign` (1 over, -1 under) in `block`."""
        return sum(1 for row in rows if sign * row[block] > 0)

    def chance(block, sign):
        """`sign_chance` of the elections erring in the direction of `sign` in `block`."""
        return sign_chance(len(rows), count(block, sign))

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


def interval_text(entry: Optional[dict], digits: int = 2, signed: bool = False) -> str:
    """
    The 95% interval of a measure of the rating check, as the support line of a KPI or a cell writes it.

    Parameters
    ----------
    entry : dict or None
        ``{'value', 'lo', 'hi', ...}`` of the part.
    digits : int, optional
        Decimals of both bounds.
    signed : bool, optional
        Whether the bounds carry their sign (``+0,05``), as a difference does.

    Returns
    -------
    str
        ``'IC 95%: −0,24 a −0,08'``, or ``''`` without both bounds.
    """
    entry = entry or {}

    if _is_missing(entry.get('lo')) or _is_missing(entry.get('hi')):
        return ''

    write = delta_text if signed else fmt_without_plus

    return 'IC 95%: {} a {}'.format(write(entry['lo'], digits), write(entry['hi'], digits))


def fmt_without_plus(x, digits: int = 2) -> str:
    """
    A number as ``delta_text`` writes it but without the plus: a typographic minus when negative and no sign
    otherwise (``−0,18``, ``0,44``).

    Parameters
    ----------
    x : float or None
        The number.
    digits : int, optional
        Decimals.

    Returns
    -------
    str
        The formatted number, or the dash of ``fmt_num`` when missing.
    """
    text = delta_text(x, digits)

    return text[1:] if text.startswith('+') else text


def number_word(n) -> str:
    """
    A small whole number in words, as the method writes counts below eleven («cuatro o más»); larger numbers,
    and anything that is not a whole number, as ``fmt_int`` writes them.

    Parameters
    ----------
    n : int or None
        The number.

    Returns
    -------
    str
        ``'cuatro'`` for 4 (``'una'`` for 1: the texts count elections and pollsters), ``'12'`` for 12.
    """
    if isinstance(n, int) and not isinstance(n, bool) and 0 <= n < len(NUMBER_WORDS):
        return NUMBER_WORDS[n]

    return fmt_int(n)


def rating_check_kpis(summary: dict) -> list:
    """
    The KPIs of the card «Cómo lo comprobamos», for ``_macros.kpis``.

    Parameters
    ----------
    summary : dict
        The ``summary`` of the ``rating-check`` part (``{population: {...}}``).

    Returns
    -------
    list of dict
        ``{'label', 'value', 'unit', 'support', 'ok', 'color'}``: the pollster-election pairs (with the pollsters),
        the elections as «generales + autonómicas» (only when both populations have elections), the rank
        correlation and the calibration slope with their 95% intervals and the skill of the rating against
        predicting nothing, in percent (its support line adds the skill of the prior quality alone, when present).
    """
    every = summary.get('all') or {}
    general = (summary.get('generales') or {}).get('n_events')
    regional = (summary.get('autonomicas') or {}).get('n_events')
    spearman, slope, skill = every.get('spearman') or {}, every.get('slope') or {}, every.get('skill') or {}
    skill_rating, skill_prior = skill.get('skill_rating'), skill.get('skill_prior')
    skill_text = DASH if _is_missing(skill_rating) else fmt_without_plus(skill_rating * 100, 1) + '%'
    skill_support = 'frente a prever una desviación nula'
    if not _is_missing(skill_prior):
        skill_support += '; con la calidad previa sola, {}%'.format(fmt_without_plus(skill_prior * 100, 1))
    elections = [] if any(_is_missing(n) or not n for n in (general, regional)) else [
        {'label': 'Elecciones', 'value': '{} + {}'.format(fmt_int(general), fmt_int(regional)), 'unit': '',
         'support': 'generales + autonómicas', 'ok': False, 'color': None}]

    return [
        {'label': 'Pares encuestadora-elección', 'value': fmt_int(every.get('n_pairs')), 'unit': '',
         'support': '{} encuestadoras'.format(fmt_int(every.get('n_pollsters'))), 'ok': False, 'color': None},
        *elections,
        {'label': 'Correlación de rango', 'value': fmt_without_plus(spearman.get('value')), 'unit': '',
         'support': interval_text(spearman), 'ok': False, 'color': None},
        {'label': 'Pendiente de calibración', 'value': fmt_without_plus(slope.get('value')), 'unit': '',
         'support': interval_text(slope), 'ok': False, 'color': None},
        {'label': 'Acierto relativo', 'value': skill_text, 'unit': '',
         'support': skill_support, 'ok': False, 'color': None},
    ]


def rating_check_axes(rows: list) -> dict:
    """
    Limits of the cloud of the rating check: the same on both axes, symmetric around 0 and rounded up to a multiple
    of ``RATING_CHECK_AXIS_STEP``, so the diagonal is the 45° line and both differentials read on one scale.

    Parameters
    ----------
    rows : list of dict
        The points, with ``d_prev`` and ``d_obs``.

    Returns
    -------
    dict
        ``{'xMin', 'xMax', 'yMin', 'yMax', 'interval'}`` for ``scatterOption``; the bound is at least one step and
        the interval leaves ``RATING_CHECK_AXIS_TICKS`` ticks on each half axis.
    """
    largest = max([abs(row[key]) for row in rows for key in ('d_prev', 'd_obs')] or [0])
    bound = max(1, math.ceil(largest / RATING_CHECK_AXIS_STEP)) * RATING_CHECK_AXIS_STEP

    return {'xMin': -bound, 'xMax': bound, 'yMin': -bound, 'yMax': bound, 'interval': bound / RATING_CHECK_AXIS_TICKS}


def rating_check_polled(every: dict) -> dict:
    """
    The check of the calibration slope on the elections with ``min_event_polls`` polls or more, as the note of the
    card writes it.

    Parameters
    ----------
    every : dict
        The ``all`` population of the ``summary`` of the part.

    Returns
    -------
    dict
        ``same`` (whether the check holds exactly the pairs of the slope, so it separates nothing), ``value`` (the
        slope written without the plus, or ``None``) and ``interval`` (``interval_text``).
    """
    polled, fitted = every.get('slope_polled') or {}, every.get('slope') or {}
    value = polled.get('value')

    return {
        'same': bool(polled.get('n')) and polled.get('n') == fitted.get('n'),
        'value': None if _is_missing(value) else fmt_without_plus(value),
        'interval': interval_text(polled),
    }


def rating_check_summary() -> Optional[dict]:
    """
    The value of ``rating_check_context``, read from the bundle without the cache.

    Returns
    -------
    dict or None
        See ``rating_check_context``.
    """
    data = analysis_part(NATIONAL_SCOPE, RATING_CHECK_PART)

    if data is None:
        return None

    summary = data.get('summary') or {}

    if not summary.get('all'):
        return None

    scope_names = {code: row['name'] for code, row in catalogue().iterrows()}

    def election(row):
        """Public name of the election of a point: «Generales 2023», «Andalucía 2022»."""
        scope, year = row.get('scope'), (row.get('event_date') or '')[:4]
        name = 'Generales' if scope == NATIONAL_SCOPE else scope_names.get(scope, scope)
        return '{} {}'.format(name, year).strip()

    def finite(x):
        """Whether ``x`` is a finite number."""
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    rows = [dict({key: row.get(key) for key in RATING_CHECK_ROW_KEYS},
                 label='{} · {}'.format(row.get('pollster'), election(row)))
            for row in data.get('scatter') or [] if finite(row.get('d_prev')) and finite(row.get('d_obs'))]
    quarters = {}
    for row in data.get('quarters') or []:
        quarters.setdefault(row.get('population'), {})[row.get('quarter')] = row
    last = (data.get('params') or {}).get('quarters') or max(
        [row.get('quarter') or 0 for row in data.get('quarters') or []] or [0])
    quarter_rows, average_rows = [], []
    for population, name in RATING_CHECK_POPULATIONS:
        entry = summary.get(population)
        if not entry:
            continue
        first, bottom = quarters.get(population, {}).get(1), quarters.get(population, {}).get(last)
        if first or bottom:
            quarter_rows.append({
                'population': population, 'name': name,
                'first': fmt_num((first or {}).get('error_avg'), 2), 'n_first': fmt_int((first or {}).get('n')),
                'last': fmt_num((bottom or {}).get('error_avg'), 2), 'n_last': fmt_int((bottom or {}).get('n')),
            })
        average = entry.get('average') or {}
        if average.get('n'):
            average_rows.append({
                'population': population, 'name': name, 'n': fmt_int(average.get('n')),
                'mae_flat': fmt_num(average.get('mae_flat'), 2), 'mae_rated': fmt_num(average.get('mae_rated'), 2),
                'diff': delta_text(average.get('diff'), 3), 'interval': interval_text(average, 3, signed=True),
                'better': fmt_int(average.get('better')), 'worse': fmt_int(average.get('worse')),
            })

    return {
        'kpis': rating_check_kpis(summary),
        'summary': summary,
        'quarters': quarter_rows,
        'average': average_rows,
        'dates': data.get('dates') or {},
        'params': data.get('params') or {},
        'quarters_min': number_word(last or None),
        'slope_polled': rating_check_polled(summary['all']),
        'csv': {'pairs': api_url(NATIONAL_SCOPE, RATING_CHECK_PART, fmt='csv', kind='analysis'),
                'elections': api_url(NATIONAL_SCOPE, RATING_CHECK_PART, fmt='csv', kind='analysis', table='elections')},
        'initial': {'scatter': rows, 'fit': data.get('fit'), 'axes': rating_check_axes(rows)},
    }


def rating_check_context() -> Optional[dict]:
    """
    The live figures of the card «Cómo lo comprobamos» of the ``rating`` subpage: the ``rating-check`` part of the
    analysis of ``NATIONAL_SCOPE`` (the rating before each election against the deviation of each pollster in it,
    and the simple average of the last campaign with and without the rating).

    The value is computed once and cached under the TTL of the site (``Site.cached``), as ``reliability_context``:
    the part is read at most once per TTL, also when it is missing.

    Returns
    -------
    dict or None
        ``None`` without the part or when its ``summary`` has no ``all`` population (the card then shows the
        paragraph of a missing part). Otherwise ``kpis`` (``rating_check_kpis``), ``summary`` (the one of the part, by
        population), ``quarters`` (one row per population with quarters: ``population``, ``name``, ``first`` and
        ``last``, the mean error per party in points of the first and the last quarter of the rating formatted with
        two decimals, and their pairs ``n_first`` and ``n_last``), ``average`` (one row per population with
        elections: ``population``, ``name``, ``n``, ``mae_flat``, ``mae_rated``, ``diff`` (with rating minus flat,
        signed), ``interval`` and the elections ``better`` and ``worse`` with the rating, all formatted), ``dates``,
        ``params``, ``quarters_min`` (the pollsters an election needs to be split in quarters, in words: «cuatro»),
        ``slope_polled`` (``rating_check_polled``), ``csv`` (``{'pairs': url, 'elections': url}``) and ``initial`` (the slice of the island the cloud
        draws: ``scatter``, the points with finite ``d_prev`` and ``d_obs`` with ``RATING_CHECK_ROW_KEYS`` and
        ``label``; ``fit``, the ends of the fitted line or ``None``; ``axes``, ``rating_check_axes``). The cached
        value is shared by the requests: do not modify it.

    Raises
    ------
    HttpError
        Any error of the bundle other than a 404.
    """
    return site().cached(('rating-check',), rating_check_summary)


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
        sections, ``{'id', 'title'}`` in ``SECTIONS`` order) and ``andorra_published`` (whether the ``andorra``
        study is published, so the «Veda» limit links to it).

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
    context['andorra_published'] = bundle_part(site().study, 'andorra', 'meta') is not None

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
        ``section``. The ``rating`` subpage adds ``rubric`` (``quality_rubric``),
        ``herd_published`` (whether the ``rebano`` study is published, so the herding card links to it) and
        ``rating_check`` (``rating_check_context``, ``None`` without the part) and, with it, the ``initial`` island
        (``state`` and the ``rating_check`` slice the cloud draws). The ``escanos`` subpage adds
        ``errors`` (``error_example``, ``None`` without one). The ``promedio`` subpage adds
        ``example`` (``weights_example``, ``None`` without one), ``example_csv`` (the CSV of the ``weights`` part),
        ``house_example`` (``house_effects_example``, ``None`` without one), ``cycle`` (``cycle_counts``), ``lean_published`` (whether the ``sesgo-cis`` study is published, so
        the card links to it), ``shy_published`` (the same for ``voto-oculto``) and, with either example, the
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
        check = rating_check_context()
        context['rating_check'] = check
        context['herd_published'] = bundle_part(site().study, 'rebano', 'meta') is not None

        if check is not None:
            context['initial'] = {'state': context['state'], 'rating_check': check['initial']}

    if section == 'escanos':
        context['errors'] = error_example()

    if section == 'promedio':
        example = weights_example()
        house = house_effects_example()
        context['example'] = example
        context['example_csv'] = api_url(NATIONAL_SCOPE, 'weights', fmt='csv')
        context['house_example'] = house
        context['cycle'] = cycle_counts()
        context['lean_published'] = bundle_part(site().study, 'sesgo-cis', 'meta') is not None
        context['shy_published'] = bundle_part(site().study, 'voto-oculto', 'meta') is not None
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
