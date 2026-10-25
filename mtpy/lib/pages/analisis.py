"""
Contexts of the Análisis section: the index of the studies (``/analisis``), their band in the home and the page of
each study (``/analisis/{study}``).

Everything is read from the published parts of the study (``site().study``): the figures that the texts quote, the
three KPI rows, the CSV downloads and the ``initial`` island with only what the charts draw. A part or a key that is
missing gives ``None`` (the templates guard it), never an error.
"""
from typing import Optional

from markupsafe import Markup

from ...core.api import HttpError
from .. import bundle
from ..webapi import site
from .common import api_url, bundle_part, common_context
from .cera import (MONTHS_LONG, SCENARIOS, UNIFORM_SPLIT, _census_version, _check, _debate, _get, _question_1,
                   _question_2, _question_3, _sigla, _two_periods, _version)
from . import lean
from .filters import delta_badge, fmt_int, fmt_pct

# Page template of each study with a page; a published study missing here has no page (404)
STUDIES = {'voto-cera': {'template': 'analisis_voto_cera.html'},
           'sesgo-cis': {'template': 'analisis_sesgo_cis.html'}}
CERA = 'voto-cera'
# Verdict of each of the three questions of the CERA study and the colour class of its card
VERDICTS = [('SÍ', 'q1'), ('SIN EVIDENCIA', 'q2'), ('NO', 'q3')]
QUESTIONS = ('¿Se dispara el censo de los españoles en el exterior?',
             '¿Sesgo de izquierdas o reparto en provincias «estratégicas»?',
             '¿Pueden alterar el reparto de escaños el 29-N?')
ARTICLE = {'url': 'https://medium.com/@luis_sancho/el-voto-cera-en-espa%C3%B1a-y-el-efecto-de-la-ley-de-nietos-ae2cd1b063a5',
           'date': '2026-06-30'}
# First auto of the Supreme Court that suspended the inscriptions in litigation: a fact of the context, not of the
# data (spec D7); the second one is the ``as_of`` of ``params.litigation``
SUSPENDED_ON = '2026-09-10'
# Net effect of each scenario in the rows of ``prediction.last_seat``
SCENARIO_NET = {'convocatoria': 'net', 'sin_litigio': 'net_b'}
# Rows of the gap chart: the provinces with the smallest gap of the last seat
GAP_SHOWN = 10
REPORT_KEYS = ('n_mean', 'n_max', 'votes_mean', 'votes_max')
CSV_LABELS = {
    'voto-cera': {
        'census-series': 'Serie mensual del censo CERA',
        'census-provinces': 'Censo CERA por provincia',
        'census-countries': 'Censo CERA por país',
        'participation': 'Participación CERA por origen y en las generales',
        'vote': 'Voto CERA por proceso electoral',
        'ecological': 'Inferencia ecológica por provincia',
        'last-seat': 'Último escaño por provincia',
        'sensitivity-grid': 'Rejilla de participación y voto a la izquierda',
        'sensitivity-scenarios': 'Escenarios de censo, participación y sesgo',
        'placement-grid': 'Rejilla de colocación óptima',
        'check': 'Comprobación con y sin voto CERA por provincia',
    },
    'sesgo-cis': {
        'elections': 'Las elecciones del CIS, con y sin Tezanos',
        'houses': 'Ranking de encuestadoras desde 2018',
        'matrix': 'Matriz casa por elección',
        'election-houses': 'Últimos sondeos de las generales de 2023 por bloques',
    },
}


def study_part(study: str, part: str) -> Optional[dict]:
    """
    Data of a part of a published study.

    Parameters
    ----------
    study : str
        Study slug, one of ``bundle.STUDIES``.
    part : str
        Part name, one of ``bundle.STUDY_PARTS[study]``.

    Returns
    -------
    dict or None
        The ``data`` of the part, or ``None`` when it is not published.
    """
    return bundle_part(site().study, study, part)


def verdict_rows() -> list:
    """
    The three questions of the CERA study with their verdict.

    Returns
    -------
    list of dict
        ``{'label', 'tone', 'question'}`` per question, in order: ``label`` the verdict (``VERDICTS``), ``tone`` the
        colour class (``q1``...) and ``question`` its short title (``QUESTIONS``).
    """
    return [{'label': label, 'tone': tone, 'question': question}
            for (label, tone), question in zip(VERDICTS, QUESTIONS)]


def study_cards() -> list:
    """
    One card per published study with a page, for the index and the band of the home.

    Returns
    -------
    list of dict
        ``{'study', 'href', 'title', 'subtitle'}`` per study of ``site().studies()`` with a published ``meta`` and
        an entry in ``STUDIES``, in that order; ``title`` and ``subtitle`` come from the ``meta``.
    """
    cards = []

    for study in site().studies():
        meta = study_part(study, 'meta') if study in STUDIES else None

        if meta is None:
            continue

        cards.append({
            'study': study,
            'href': '/analisis/' + study,
            'title': meta.get('title'),
            'subtitle': meta.get('subtitle'),
        })

    return cards


def analisis_context(state: dict) -> dict:
    """
    Template context of the index of the studies.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).

    Returns
    -------
    dict
        ``common_context`` of ``/analisis`` plus ``studies`` (``study_cards``).
    """
    context = common_context(state, '/analisis')
    context['studies'] = study_cards()

    return context


def study_context(state: dict, study: str) -> dict:
    """
    Template context of the page of a study.

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).
    study : str
        Study slug of the route.

    Returns
    -------
    dict
        The context of the study (``CONTEXTS[study]``: ``cera_context`` or ``lean_context``).

    Raises
    ------
    HttpError
        404 ``study not found`` when the study has no page (not in ``STUDIES``) or is not published (no ``meta``).
    """
    if study not in STUDIES or study_part(study, 'meta') is None:
        raise HttpError(404, 'study not found')

    return CONTEXTS[study](state)


def _day_month(iso: Optional[str], year: bool = False) -> Optional[str]:
    """
    Compact date of the KPI notes: ``'10-09'`` or, with the year, ``'1-8-2026'``.

    Parameters
    ----------
    iso : str or None
        ISO date ``YYYY-MM-DD``.
    year : bool, optional
        Whether to add the year (then the day and the month go without leading zeros, as in the infographic).

    Returns
    -------
    str or None
        The date, or ``None`` without a valid date.
    """
    try:
        y, m, d = (int(piece) for piece in (iso or '').split('-'))
    except ValueError:
        return None

    return '{}-{}-{}'.format(d, m, y) if year else '{:02d}-{:02d}'.format(d, m)


def _previous_election(check: dict, meta: dict) -> dict:
    """
    Check of the last general election before the one of the study (2023 for the 29-N).

    Parameters
    ----------
    check : dict
        Data of the ``check`` part.
    meta : dict
        Data of the ``meta`` part (``election.date``).

    Returns
    -------
    dict
        The entry of ``check.elections`` with the latest date before the election of the study, or ``{}``.
    """
    day = _get(meta, 'election', 'date') or '9999-12-31'
    rows = [row for row in check.get('elections') or [] if (row.get('date') or '') < day]

    return max(rows, key=lambda row: row['date']) if rows else {}


def _grid_cell(grid: Optional[dict], rows_key: str, cols_key: str, column: float = UNIFORM_SPLIT):
    """
    Cell of a sensitivity grid at its smallest participation and at a given column.

    Parameters
    ----------
    grid : dict or None
        Grid with the row values under ``rows_key``, the column values under ``cols_key`` and ``values``.
    rows_key : str
        Key of the participations (the rows).
    cols_key : str
        Key of the shares of the vote (the columns).
    column : float, optional
        Column value (``UNIFORM_SPLIT``: the uniform split).

    Returns
    -------
    int or None
        The cell, or ``None`` when the grid or the column is missing.
    """
    rows, cols, values = _get(grid, rows_key) or [], _get(grid, cols_key) or [], _get(grid, 'values') or []
    j = next((index for index, value in enumerate(cols) if value is not None and abs(value - column) < 1e-9), None)

    if not rows or j is None:
        return None

    i = rows.index(min(rows))

    try:
        return values[i][j]
    except (IndexError, TypeError):
        return None


def _latest_generales(vote: dict) -> dict:
    """
    Row of the latest general election of ``vote.processes``.

    Parameters
    ----------
    vote : dict
        Data of the ``vote`` part.

    Returns
    -------
    dict
        The row, or ``{}`` without general elections.
    """
    rows = [row for row in vote.get('processes') or [] if row.get('kind') == 'generales' and row.get('date')]

    return max(rows, key=lambda row: row['date']) if rows else {}


def _long_date(iso: Optional[str]) -> Optional[str]:
    """
    Date in words, as the pill of the election says it: ``'29 de noviembre de 2026'``.

    Parameters
    ----------
    iso : str or None
        ISO date ``YYYY-MM-DD``.

    Returns
    -------
    str or None
        The date, or ``None`` without a valid date.
    """
    try:
        y, m, d = (int(piece) for piece in (iso or '').split('-'))
    except ValueError:
        return None

    return '{} de {} de {}'.format(d, MONTHS_LONG[m - 1], y) if 1 <= m <= len(MONTHS_LONG) else None


def _figures(parts: dict) -> dict:
    """
    The figures that the texts of the CERA page quote, all taken from the parts.

    Parameters
    ----------
    parts : dict
        The data of every part by name (``{}`` for a missing one).

    Returns
    -------
    dict
        ``census_total``, ``electors_2023`` (the monthly close of ``baseline_date``, the base of the increment),
        ``electors_official`` (the CERA electors of the baseline election in ``participation.generales``, from
        Infoelectoral) and ``electors_gap`` (``electors_official - electors_2023``), ``baseline_date``,
        ``growth_abs``, ``growth_pct``, ``baseline_year``, ``net_before``,
        ``net_after``, ``g1``, ``g2`` and ``ratio`` (the additions of the two periods the page shows and their
        pace: ``periods_version`` says which, ``now`` or, when the part lacks it, ``article``), ``g1_article``,
        ``g2_article``, ``ratio_article`` (the replication of the article with its window), ``latam_rate``,
        ``rest_rate``, ``p_slope``, ``r``, ``n_ecological``, ``n_over_gap_2023``, ``decided_2023``,
        ``net_psoe_2023``, ``voters_projected``, ``votes_projected``, ``voters_2023``, ``participation_projected``,
        ``left_share``, ``litigation``, ``max_gain``, ``grid_observed``, ``placement_observed``, ``pool``,
        ``report``, ``sensitive_all`` and ``sensitive_countries`` (the shares of the sensitive provinces in the
        version of ``placement.sensitive`` the page shows, ``sensitive_version``), ``changed`` and ``n_changed``
        (the provinces whose seats change in the ``convocatoria`` scenario); ``None`` for what the parts do not
        have.
    """
    meta, census, vote = parts['meta'], parts['census'], parts['vote']
    placement, prediction, sensitivity = parts['placement'], parts['prediction'], parts['sensitivity']
    params = meta.get('params') or {}
    scenario = _get(prediction, 'scenarios', 'convocatoria') or {}
    census_new, new_electors = scenario.get('census'), scenario.get('new_electors')
    electors_2023 = census_new - new_electors if census_new is not None and new_electors is not None else None
    provinces = census.get('provinces') or []
    version = _census_version(census)
    shown = [_two_periods(row, version) for row in provinces]
    g1 = sum(row['g1'] or 0 for row in shown) if provinces else None
    g2 = sum(row['g2'] or 0 for row in shown) if provinces else None
    g1_article = sum(row.get('g1') or 0 for row in provinces) if provinces else None
    g2_article = sum(row.get('g2') or 0 for row in provinces) if provinces else None
    sensitive = placement.get('sensitive')
    sensitive_version = _version(sensitive)
    previous = _previous_election(parts['check'], meta)
    rates = params.get('participation') or {}
    voters = scenario.get('voters_projected')
    max_gain = _get(params, 'check', 'max_gain')
    report = prediction.get('report') or {}
    baseline = _get(params, 'election', 'baseline') or previous.get('date')
    official = next((row.get('electors') for row in parts['participation'].get('generales') or []
                     if baseline and row.get('date') == baseline), None)

    return {
        'census_total': _get(meta, 'election', 'census_total'),
        'electors_2023': electors_2023,
        'electors_official': official,
        'electors_gap': official - electors_2023 if official is not None and electors_2023 is not None else None,
        'baseline_date': _get(meta, 'election', 'baseline_date'),
        'growth_abs': new_electors,
        'growth_pct': 100 * new_electors / electors_2023 if electors_2023 else None,
        'baseline_year': (_get(params, 'election', 'baseline') or previous.get('date') or '')[:4] or None,
        'net_before': _get(meta, 'regression', 'net_before'),
        'net_after': _get(meta, 'regression', 'net_after'),
        'g1': g1,
        'g2': g2,
        'ratio': g2 / g1 if g1 else None,
        'periods_version': version,
        'g1_article': g1_article,
        'g2_article': g2_article,
        'ratio_article': g2_article / g1_article if g1_article else None,
        'latam_rate': round(100 * rates['latam'], 2) if rates.get('latam') is not None else None,
        'rest_rate': round(100 * rates['rest'], 2) if rates.get('rest') is not None else None,
        'p_slope': _get(meta, 'regression', 'p_slope'),
        'r': _get(vote, 'ecological', 'r'),
        'n_ecological': _get(vote, 'ecological', 'n'),
        'n_over_gap_2023': previous.get('n_over_gap'),
        'decided_2023': [row.get('name') for row in previous.get('decided') or []],
        'net_psoe_2023': previous.get('net_psoe'),
        'voters_projected': voters,
        'votes_projected': scenario.get('votes_projected'),
        'voters_2023': scenario.get('voters_2023'),
        'participation_projected': round(100 * voters / census_new, 1) if voters is not None and census_new else None,
        'left_share': _latest_generales(vote).get('left_share'),
        'litigation': _get(params, 'litigation', 'count'),
        'max_gain': max_gain if max_gain is not None else _get(meta, 'registered', 'max_gain'),
        'grid_observed': _grid_cell(sensitivity.get('grid'), 'participations', 'left_fracs'),
        'placement_observed': _grid_cell(sensitivity.get('placement'), 'participations', 'fracs'),
        'pool': _get(sensitivity, 'placement', 'pool'),
        'report': {key: report.get(key) for key in REPORT_KEYS},
        'sensitive_all': _get(sensitive, sensitive_version, 'all'),
        'sensitive_countries': _get(sensitive, sensitive_version, 'countries'),
        'sensitive_version': sensitive_version,
        'changed': bool(scenario.get('changed')) or any(row.get('change') for row in scenario.get('by_block') or []),
        'n_changed': len(scenario.get('changed') or []),
    }


def _kpis(figures: dict, meta: dict) -> list:
    """
    The three KPI rows of the head of the CERA page, for ``_macros.kpis``.

    Parameters
    ----------
    figures : dict
        ``_figures`` of the page.
    meta : dict
        Data of the ``meta`` part (``params.litigation.as_of``, ``election.census_closed``).

    Returns
    -------
    list of dict
        ``{'label', 'value', 'unit', 'support', 'ok', 'color'}``: the census of the election (with the growth
        against the previous general election as a ``.dlt`` arrow, so ``support`` is ``Markup``), the inscriptions
        in litigation (with the dates of the autos and the close of the census in force) and the expected CERA
        voters (with the participation rates by origin and the voters of the previous general election). A missing
        figure shows a dash and drops its piece of the note.
    """
    year = figures['baseline_year']
    census_note = ''

    if figures['growth_pct'] is not None:
        badge = delta_badge(figures['growth_pct'])
        census_note = Markup('<span class="dlt {}">{}</span> frente a {} ({})').format(
            badge['dir'], badge['text'] + ('%' if badge['dir'] != 'flat' else ''), year or 'la anterior',
            fmt_int(figures['electors_2023']))

    litigation = []

    if figures['litigation'] is not None:
        maintained = _day_month(_get(meta, 'params', 'litigation', 'as_of'))
        litigation.append('suspendidas por el Supremo el {}'.format(_day_month(SUSPENDED_ON))
                          + (', mantenido el {}'.format(maintained) if maintained else ''))

    closed = _day_month(_get(meta, 'election', 'census_closed'), year=True)

    if closed:
        litigation.append('censo vigente: cerrado el {}'.format(closed))

    voters = []

    if figures['latam_rate'] is not None:
        voters.append('~{} Latam'.format(fmt_pct(figures['latam_rate'], 0)))

    if figures['rest_rate'] is not None:
        voters.append('~{} resto'.format(fmt_pct(figures['rest_rate'], 0)))

    if figures['voters_2023'] is not None and year:
        voters.append('{}: {}'.format(year, fmt_int(figures['voters_2023'])))

    row = {'unit': '', 'ok': False, 'color': None}

    return [
        dict(row, label='Censo CERA 29-N', value=fmt_int(figures['census_total']), support=census_note),
        dict(row, label='Inscripciones en litigio', value=fmt_int(figures['litigation']), support=' · '.join(litigation)),
        dict(row, label='Votantes CERA esperados', value=fmt_int(figures['voters_projected']), support=' · '.join(voters)),
    ]


def _csv_links(study: str) -> list:
    """
    The CSV downloads of a study (``CSV_LABELS[study]``).

    Parameters
    ----------
    study : str
        Study slug.

    Returns
    -------
    list of dict
        ``{'label', 'href'}`` per CSV of ``bundle.STUDY_CSV[study]``, in that order; ``href`` is the ``api_url`` of its part
        with ``format=csv`` and its ``table``.
    """
    part_of = {name: part for part, names in bundle.STUDY_CSV_OF[study].items() for name in names}

    return [{'label': CSV_LABELS[study][name], 'href': api_url(study, part_of[name], kind='study', fmt='csv', table=name)}
            for name in bundle.STUDY_CSV[study]]


def _island(state: dict, parts: dict, figures: dict) -> dict:
    """
    The ``initial`` island of the CERA page: only what the charts draw.

    Parameters
    ----------
    state : dict
        ``state`` of the common context.
    parts : dict
        The data of every part by name.
    figures : dict
        ``_figures`` of the page.

    Returns
    -------
    dict
        ``state``; ``census`` (``series``, ``marks`` and ``periods``, the two ``[start, end]`` periods the page
        shows: ``now``, or the article's when the part lacks them); ``origins`` (``name``, ``g1``, ``g2``,
        ``ratio`` of those two periods per origin);
        ``participation`` (``process``, ``origin``, ``rate``); ``ecological`` (``rows`` with ``name``,
        ``latam_share``, ``left_share``, ``census``, plus ``r`` and ``slope``); ``slope`` (the block margin of each
        regional election, ``community``, ``previous``, ``last``); ``cannibal`` (``community``, ``d_psoe``,
        ``d_other_left``); ``sensitive`` (the version of ``placement.sensitive`` the page shows: ``period``,
        ``countries`` and ``all``); ``gap`` (per scenario, the ``GAP_SHOWN``
        provinces with the smallest gap, ``name``, ``winner`` and ``runner_up`` by their siglas, ``gap``, ``net``,
        for labels like «Girona (JxCat/PP)») and ``radar``
        (``today``: the expected participation, no directed placement and the left share of the last general
        election, in %; ``max``: 100 on every axis).
    """
    census, vote, participation = parts['census'], parts['vote'], parts['participation']
    version = _census_version(census)
    sensitive = parts['placement'].get('sensitive') or {}
    ecological = vote.get('ecological') or {}
    regional = participation.get('autonomicas') or []
    last_seat = sorted(parts['prediction'].get('last_seat') or [],
                       key=lambda row: (row.get('gap') is None, row.get('gap') or 0))[:GAP_SHOWN]

    return {
        'state': state,
        'census': {'series': census.get('series'), 'marks': census.get('marks'),
                   'periods': _get(census, 'periods', version)},
        'origins': [{'name': row.get('name'), **{key: value for key, value in _two_periods(row, version).items()
                                                 if key != 'c2_start'}} for row in census.get('origins') or []],
        'participation': [{key: row.get(key) for key in ('process', 'origin', 'rate')}
                          for row in participation.get('origin') or []],
        'ecological': {
            'rows': [{key: row.get(key) for key in ('name', 'latam_share', 'left_share', 'census')}
                     for row in ecological.get('rows') or []],
            'r': ecological.get('r'),
            'slope': ecological.get('slope'),
        },
        'slope': [{'community': row.get('community'), 'previous': _get(row, 'previous', 'margin'),
                   'last': _get(row, 'last', 'margin')} for row in regional],
        'cannibal': [{key: row.get(key) for key in ('community', 'd_psoe', 'd_other_left')} for row in regional],
        'sensitive': sensitive.get(_version(sensitive)),
        'gap': {scenario: [{'name': row.get('name'), 'winner': _sigla(row, 'winner'),
                            'runner_up': _sigla(row, 'runner_up'), 'gap': row.get('gap'),
                            'net': row.get(SCENARIO_NET[scenario])}
                           for row in last_seat] for scenario in SCENARIOS},
        'radar': {'today': [figures['participation_projected'], 0, figures['left_share']], 'max': [100, 100, 100]},
    }


def cera_context(state: dict) -> dict:
    """
    Template context of the page of the CERA study (``/analisis/voto-cera``).

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).

    Returns
    -------
    dict
        ``common_context`` of ``/analisis`` plus the data of the seven parts after ``meta`` under their names
        (``census``, ``participation``, ``vote``, ``placement``, ``prediction``, ``sensitivity``, ``check``) and the
        one of ``meta`` as ``study_meta`` (``meta`` stays the run metadata of the common context, which the footer
        reads), ``verdicts`` (``verdict_rows``), ``figures``
        (``_figures``), ``kpis`` (``_kpis``), ``registered`` (``{'at', 'claim'}`` of the registration, or ``None``
        while it is a draft), ``checked`` (whether ``check.verdict`` is set), ``sources`` (``meta.sources``),
        ``csv_links`` (``_csv_links``), ``article`` (``ARTICLE``), ``initial`` (``_island``), ``election_day``
        (the date of the election in words, ``_long_date``), the view models of the three questions, ``q1``,
        ``q2`` and ``q3`` (``cera._question_1``, ``_question_2`` and ``_question_3``), and those of the debate and
        of the check, ``debate`` (``cera._debate``) and ``comprobacion`` (``cera._check``).

    Raises
    ------
    HttpError
        503 without a bundle.
    """
    context = common_context(state, '/analisis')
    parts = {part: study_part(CERA, part) or {} for part in bundle.STUDY_PARTS[CERA]}
    meta = parts['meta']
    figures = _figures(parts)
    context.update({part: data for part, data in parts.items() if part != 'meta'})
    context.update(
        study_meta=meta,
        verdicts=verdict_rows(),
        figures=figures,
        kpis=_kpis(figures, meta),
        registered={'at': meta['registered_at'], 'claim': _get(meta, 'registered', 'claim')}
        if meta.get('registered_at') else None,
        checked=parts['check'].get('verdict') is not None,
        sources=meta.get('sources') or [],
        csv_links=_csv_links(CERA),
        article=ARTICLE,
        initial=_island(context['state'], parts, figures),
        election_day=_long_date(_get(meta, 'election', 'date')),
        q1=_question_1(parts['census'], figures),
        q2=_question_2(parts, figures),
        debate=_debate(parts, _previous_election(parts['check'], meta)),
        comprobacion=_check(parts),
    )
    context['q3'] = _question_3(parts, figures, context['initial']['gap'][SCENARIOS[0]],
                                api_url(CERA, 'prediction', kind='study', fmt='csv', table='last-seat'))

    return context


def _published_at(study: str) -> Optional[str]:
    """``published_at`` of a study in the manifest, or ``None`` without a manifest or without the entry."""
    try:
        return _get(site().manifest_data(), 'studies', study, 'published_at')
    except HttpError:
        return None


def lean_context(state: dict) -> dict:
    """
    Template context of the page of the CIS bias study (``/analisis/sesgo-cis``).

    Parameters
    ----------
    state : dict
        Validated page parameters (``parse_state``).

    Returns
    -------
    dict
        ``common_context`` of ``/analisis`` plus ``study_meta`` (the ``meta`` part), ``summary``, ``periods``
        (``meta.periods``), ``verdicts`` (``lean.verdict_rows``), ``kpis``, ``tests``, ``tech`` (``lean.tech_note``),
        ``compare`` (``lean.compare_boxes``), ``elections`` (the rows of the period with Tezanos, newest first),
        ``houses`` (the rows of the ranking with the ``href`` of the profile), ``runner_up`` (``lean.runner_up``: the
        house that follows the CIS, or ``None``), ``cis_leads`` (``lean.cis_leads``: whether the CIS tops the ranking,
        the condition of the strong titles of question 2), ``election`` (the part, or ``None``),
        ``sources``, ``csv_links``, ``initial`` (``lean.island`` plus the common ``state``) and ``published_at``
        (``None`` without the entry in the manifest).

    Raises
    ------
    HttpError
        503 without a bundle.
    """
    context = common_context(state, '/analisis')
    parts = {part: study_part(lean.LEAN, part) or {} for part in bundle.STUDY_PARTS[lean.LEAN]}
    meta, summary = parts['meta'], parts['summary']
    periods = _get(meta, 'periods')
    context.update(
        study_meta=meta,
        summary=summary,
        periods=periods,
        verdicts=lean.verdict_rows(),
        kpis=lean.kpis(summary, periods),
        tests=lean.tests(summary),
        tech=lean.tech_note(summary),
        compare=lean.compare_boxes(summary, periods),
        elections=lean.election_rows(parts),
        houses=lean.house_rows(parts),
        runner_up=lean.runner_up(_get(parts['houses'], 'rows') or []),
        cis_leads=lean.cis_leads(_get(parts['houses'], 'rows') or []),
        election=parts['election'] or None,
        sources=meta.get('sources') or [],
        csv_links=_csv_links(lean.LEAN),
        initial=dict(lean.island(parts), state=context['state']),
        published_at=_published_at(lean.LEAN),
    )

    return context


# Context builder of each study of ``STUDIES``
CONTEXTS = {CERA: cera_context, lean.LEAN: lean_context}
