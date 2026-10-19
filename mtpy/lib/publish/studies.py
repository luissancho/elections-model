"""Exporters and ``publish_study`` for the studies of the bundle (today one, the CERA vote study ``voto-cera``).

The study package (``mtpy.lib.studies.cera``, with its regression libraries) is imported inside the functions that
need it, never at module level: the web imports ``publish`` and must stay light.
"""
import math
import os
import re
import time
from typing import Callable, Optional

import pandas as pd

from .. import bundle
from ..bundle import BundleWriter
from ...core.io import FileSystem
from .common import PCT_DECIMALS, PROB_DECIMALS, provenance, PublishRefused, REPO_ROOT, round_cols, _iso_dates, _today
from .pointers import update_manifest


STUDY_TITLES = {'voto-cera': ('El voto CERA en España y el efecto de la «Ley de Nietos»',
                              '¿Es posible orquestar con éxito un fraude electoral significativo a través del censo '
                              'exterior?')}

REGISTRATION_KEYS = ('registered_at', 'registered_commit', 'registered', 'registered_changed_at')

SCENARIOS = ('convocatoria', 'sin_litigio')

# The candidature columns of the check, with and without the CERA vote, that the page names by their sigla
WINNERS = ('winner_with', 'winner_without')

PROCESS_COLUMNS = ['kind', 'community', 'step', 'date', 'electors', 'voters', 'rate', 'left_share', 'margin',
                   'psoe_pct', 'pp_pct', 'vox_pct', 'other_left_pct', 'd_psoe', 'd_other_left', 'd_block', 'premium',
                   'psoe_pp_cera', 'psoe_pp_total']

MONTHS = ('enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre',
          'noviembre', 'diciembre')

MONTHLY_URL = 'https://www.ine.es/oficina_censo/censo_cerrado/cerapropais_{yy:02d}.xlsx'

ELECTION_TABLE = re.compile(r'elecgen(\d{4})_tab(\d+)\.xlsx$')

ELECTION_TABLE_URL = 'https://www.ine.es/oficina_censo/elecgen{year}/Elecgral_{year}_cc_tab{n}.xlsx'

INFOELECTORAL_URL = 'https://infoelectoral.interior.gob.es/'

REGIONAL_SOURCES = [
    ('Junta Electoral de Castilla y León: voto CERA de las autonómicas y por consulado (BOCYL)',
     'https://bocyl.jcyl.es/'),
    ('Junta Electoral de Andalucía: voto CERA de las autonómicas (BOJA)',
     'https://www.juntadeandalucia.es/eboja.html'),
    ('Junta Electoral de Aragón: voto CERA de las autonómicas y por consulado (BOA)', 'https://www.boa.aragon.es/'),
    ('Junta Electoral de Extremadura: voto CERA por consulado (DOE)', 'https://doe.juntaex.es/'),
]


def _value(value, decimals: Optional[int] = None):
    """Return a scalar as a plain number (``None`` for missing or infinite), rounded to ``decimals`` if given."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(value, decimals) if decimals is not None and isinstance(value, float) else value


def _rounded(obj, decimals: int = PCT_DECIMALS):
    """Round every float of a nested structure of dicts and lists (``None`` for NaN and infinite)."""
    if isinstance(obj, dict):
        return {key: _rounded(value, decimals) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_rounded(value, decimals) for value in obj]
    return _value(obj, decimals)


def _ints(frame: pd.DataFrame, cols) -> pd.DataFrame:
    """Cast the present columns to Python ints, keeping missing values as ``None`` (object columns)."""
    out = frame.copy()
    for col in [c for c in cols if c in out.columns]:
        out[col] = pd.Series([None if pd.isna(v) else int(round(v)) for v in out[col]], index=out.index, dtype=object)
    return out


def _table(frame: pd.DataFrame, pct=(), prob=(), ints=(), decimals: Optional[dict] = None) -> pd.DataFrame:
    """Round a frame for export: percentages, probabilities, other columns by ``decimals`` and integer counts."""
    out = round_cols(frame, pct=pct, prob=prob)
    for col, places in (decimals or {}).items():
        if col in out.columns:
            out[col] = [_value(v, places) for v in out[col]]
    return _ints(out, ints)


def _rows(frame: pd.DataFrame) -> list:
    """Records of a frame, with ``None`` for every missing or infinite value."""
    return [{key: _value(value) for key, value in row.items()} for row in frame.to_dict(orient='records')]


def _iso(day) -> str:
    """A date (string or timestamp) as ``YYYY-MM-DD``."""
    return pd.Timestamp(day).strftime('%Y-%m-%d')


def _long_date(day: str) -> str:
    """A ``YYYY-MM-DD`` date in long Spanish form (``29 de noviembre de 2026``)."""
    stamp = pd.Timestamp(day)
    return f'{stamp.day} de {MONTHS[stamp.month - 1]} de {stamp.year}'


def _two_periods(study, group: str) -> pd.DataFrame:
    """The two-period rows of a group with both versions: the article's columns and the current ones (``_now``)."""
    article, now = study.growth[group]['article'], study.growth[group]['now']
    key = article.columns[0]
    merged = article.merge(now, on=[key, 'name'], how='outer', suffixes=('', '_now'))
    counts = [c for c in merged.columns if c.startswith(('c1_', 'c2_', 'g1', 'g2'))]
    return _table(merged, prob=['ratio', 'ratio_now'], ints=counts)


def _election_growth(study, group: str) -> pd.DataFrame:
    """The growth from the baseline election to the convocation census of a group, rounded."""
    return _table(study.election_growth[group], pct=['pct'], ints=['baseline', 'election', 'growth'])


def export_study_census(study) -> tuple:
    """Export the census series, its marks and the growth of the CERA census (question 1).

    Parameters
    ----------
    study : CeraStudy
        The computed study (``compute_cera``).

    Returns
    -------
    tuple
        ``(data, frames)``: ``data`` for ``study-census`` with ``series`` (``dates``, ``values``: national
        electors by month), ``marks`` (``date``, ``label``), ``periods`` (``article`` and ``now``, two
        ``[start, end]`` pairs each), ``provinces``, ``countries``, ``origins`` and ``exposure`` (the two-period
        rows: ``c1_*``, ``c2_*``, ``g1``, ``g2``, ``ratio`` of the article and the same with ``_now``), and
        ``election_provinces`` and ``election_countries`` (``baseline``, ``election``, ``growth``, ``pct``);
        ``frames`` the ``census-series``, ``census-provinces`` and ``census-countries`` twins (the two-period
        rows with the election growth of the key merged in).
    """
    groups = {group: _two_periods(study, group) for group in ('provinces', 'countries', 'origins', 'exposure')}
    election = {group: _election_growth(study, group) for group in ('provinces', 'countries')}
    dates = _iso_dates(study.series.index).tolist()
    values = [int(v) for v in study.series.values]
    data = {
        'series': {'dates': dates, 'values': values},
        'marks': [{'date': day, 'label': label} for day, label in study.marks],
        'periods': {'article': [[_iso(a), _iso(b)] for a, b in study.periods_article],
                    'now': [[_iso(a), _iso(b)] for a, b in study.periods_now]},
        **{group: _rows(frame) for group, frame in groups.items()},
        'election_provinces': _rows(election['provinces']),
        'election_countries': _rows(election['countries']),
    }
    frames = {
        'census-series': pd.DataFrame({'date': dates, 'electors': values}),
        'census-provinces': groups['provinces'].merge(election['provinces'].drop(columns='name'), on='region_id',
                                                      how='outer'),
        'census-countries': groups['countries'].merge(election['countries'].drop(columns='name'),
                                                      on='country_code', how='outer'),
    }
    return data, frames


def export_study_participation(study) -> tuple:
    """Export the CERA participation by origin, in the general elections and in the regional ones (question 2).

    Parameters
    ----------
    study : CeraStudy
        The computed study.

    Returns
    -------
    tuple
        ``(data, frames)``: ``data`` for ``study-participation`` with ``origin`` (``process``, ``origin``,
        ``electors``, ``voters``, ``rate`` of each process with data by consulate), ``generales`` (``date``,
        ``electors``, ``voters``, ``rate``, ``period``) and ``autonomicas`` (``behaviour.autonomic_processes``:
        ``community``, ``previous`` and ``last`` processes, ``d_psoe``, ``d_other_left``, ``d_block``);
        ``frames`` the ``participation`` twin, the origin and generales rows in long form with a ``table`` column.
    """
    origin = pd.concat([frame.assign(process=stem) for stem, frame in study.participation['origin'].items()],
                       ignore_index=True)[['process', 'origin', 'electors', 'voters', 'rate']]
    origin = _table(origin, pct=['rate'], ints=['electors', 'voters'])
    generales = _table(study.participation['generales'], pct=['rate'], ints=['electors', 'voters'])
    data = {'origin': _rows(origin), 'generales': _rows(generales), 'autonomicas': _rounded(study.autonomics)}
    frame = pd.concat([origin.assign(table='origin'), generales.assign(table='generales')], ignore_index=True)
    columns = ['table', 'process', 'origin', 'date', 'period', 'electors', 'voters', 'rate']
    return data, {'participation': frame[columns]}


def _processes(study) -> pd.DataFrame:
    """One row per process of the vote: the general elections and both processes of each regional one."""
    turnout = study.participation['generales'].set_index('date')
    rows = []
    for process in study.generales:
        extra = turnout.loc[process['date']] if process['date'] in turnout.index else {}
        rows.append({'kind': 'generales', 'community': 'España', **process,
                     **{key: extra[key] for key in ('electors', 'voters', 'rate') if key in extra}})
    for process in study.autonomics:
        changes = {key: process[key] for key in ('d_psoe', 'd_other_left', 'd_block')}
        rows.append({'kind': 'autonomicas', 'community': process['community'], 'step': 'previous',
                     **process['previous']})
        rows.append({'kind': 'autonomicas', 'community': process['community'], 'step': 'last', **process['last'],
                     **changes})
    frame = pd.DataFrame(rows).reindex(columns=PROCESS_COLUMNS)
    return _table(frame, pct=PROCESS_COLUMNS[6:], ints=['electors', 'voters'])


def export_study_vote(study) -> tuple:
    """Export the vote of the CERA electors: blocks by process, representation ratios and the ecological inference.

    Parameters
    ----------
    study : CeraStudy
        The computed study.

    Returns
    -------
    tuple
        ``(data, frames)``: ``data`` for ``study-vote`` with ``processes`` (``PROCESS_COLUMNS``: the general
        elections, ``step`` null, and the ``previous`` and ``last`` processes of each regional one, the changes
        on the ``last`` row), ``ratios`` (``sigla``, ``share_cera``, ``share_total``, ``ratio``) and
        ``ecological`` (``rows`` by province, ``n``, ``min_voters``, ``r``, ``slope`` and ``quartiles``);
        ``frames`` the ``vote`` (processes) and ``ecological`` (rows) twins.
    """
    processes = _processes(study)
    ratios = round_cols(study.ratios, pct=['share_cera', 'share_total', 'ratio'])
    eco = study.ecological
    rows = _table(pd.DataFrame(eco['rows']), pct=['latam_share', 'left_share'], ints=['census'])
    ecological = {'rows': _rows(rows), 'n': int(eco['n']), 'min_voters': study.params['ecological_min_voters'],
                  'r': _value(eco['r'], PROB_DECIMALS), 'slope': _value(eco['slope'], PROB_DECIMALS),
                  'quartiles': _rounded(eco['quartiles'])}
    data = {'processes': _rows(processes), 'ratios': _rows(ratios), 'ecological': ecological}
    return data, {'vote': processes, 'ecological': rows}


def _versions(rows: dict, keys: list) -> list:
    """Merge the rows of the ``article`` and ``now`` versions on ``keys``, the current columns with ``_now``."""
    merged = pd.DataFrame(rows['article']).merge(pd.DataFrame(rows['now']), on=keys, how='outer', suffixes=('', '_now'))
    return _rows(merged)


def export_study_placement(study) -> tuple:
    """Export where the new electors registered: sensitive provinces, acceleration and the Madrid magnet.

    Parameters
    ----------
    study : CeraStudy
        The computed study.

    Returns
    -------
    tuple
        ``(data, {})``: ``data`` for ``study-placement`` with ``sensitive`` (by version, ``article`` and
        ``now``: ``period``, and ``stock_share`` and ``growth_share`` of the sensitive provinces for the three
        countries, ``countries``, and for every country, ``all``), ``acceleration`` (by province: ``g1``,
        ``g2``, ``ratio`` of the article and ``_now``, with ``magnitude`` and ``gap``), ``correlations`` (by
        version: ``r_size_growth``, ``r_size_ratio``, ``r_gap_growth``) and ``madrid`` (by ``label``: ``share_p1``,
        ``share_p2``, ``stock_share`` and ``_now``). No CSV twin.
    """
    sensitive = {version: {'period': [part['start'], part['end']], 'countries': _rounded(part['countries']),
                           'all': _rounded(part['all'])} for version, part in study.sensitive.items()}
    acceleration = {version: [_rounded(row, PROB_DECIMALS) for row in part['rows']]
                    for version, part in study.acceleration.items()}
    correlations = {version: {key: _value(part[key], PROB_DECIMALS)
                              for key in ('r_size_growth', 'r_size_ratio', 'r_gap_growth')}
                    for version, part in study.acceleration.items()}
    madrid = {version: _rounded(rows) for version, rows in study.madrid.items()}
    data = {'sensitive': sensitive, 'acceleration': _versions(acceleration, ['region_id', 'name', 'magnitude', 'gap']),
            'correlations': correlations, 'madrid': _versions(madrid, ['label'])}
    return data, {}


def _by_block(model: dict) -> list:
    """The seats of each block before and after the increment (``block``, ``before``, ``after``, ``change``)."""
    return [{'block': block, **{key: int(value) for key, value in seats.items()}}
            for block, seats in model['by_block'].items()]


def _scenario(study, name: str) -> dict:
    """The figures of one census scenario of the increment model."""
    scenario, baseline = study.scenarios[name], study.params['election']['baseline']
    model, new = scenario['model'], scenario['new']
    rates = new['origin'].map(study.params['participation']).astype(float)
    voters_2023 = int(study.voters[baseline].sum())
    votes_2023 = int(round(float(study.cera[baseline].values.sum())))
    siglas = [{'sigla': sigla, **{key: int(value) for key, value in seats.items()}}
              for sigla, seats in model['by_sigla'].items() if seats['before'] or seats['after']]
    return {
        'census': int(scenario['census']['electors'].sum()),
        'new_electors': int(new['electors'].sum()),
        'voters_2023': voters_2023,
        'voters_projected': int(round(voters_2023 + float((new['electors'] * rates).sum()))),
        'votes_2023': votes_2023,
        'votes_projected': int(round(votes_2023 + model['votes_extra'])),
        'by_block': _by_block(model),
        'by_sigla': sorted(siglas, key=lambda row: (-row['before'], row['sigla'])),
        'changed': [{'region_id': int(region), 'name': study.province_names.get(region)}
                    for region in model['changed']],
    }


def _siglas(study, day: str) -> dict:
    """The sigla of each candidature full name of an election's Infoelectoral results (``{}`` without results)."""
    results = study.results.get(day)
    return dict(results.siglas) if results is not None else {}


def _with_siglas(frame: pd.DataFrame, siglas: dict, columns) -> pd.DataFrame:
    """Add ``{column}_sigla`` after each candidature column: its sigla, or the full name when it has none."""
    out = frame.copy()
    for col in [c for c in columns if c in out.columns]:
        values = [None if pd.isna(name) else (siglas.get(name) or name) for name in out[col]]
        out.insert(out.columns.get_loc(col) + 1, col + '_sigla', pd.Series(values, index=out.index, dtype=object))
    return out


def export_study_prediction(study) -> tuple:
    """Export the increment model in both census scenarios, the last seat of each province and the replication.

    Parameters
    ----------
    study : CeraStudy
        The computed study.

    Returns
    -------
    tuple
        ``(data, frames)``: ``data`` for ``study-prediction`` with ``scenarios`` (``convocatoria`` and
        ``sin_litigio``: ``census``, ``new_electors``, ``voters_2023``, ``voters_projected``, ``votes_2023``,
        ``votes_projected``, ``by_block``, ``by_sigla`` (the candidatures with seats) and ``changed``),
        ``last_seat`` (52 rows of ``seats.net_effect`` of ``convocatoria`` plus ``net_b`` of ``sin_litigio``, with
        ``winner_sigla`` and ``runner_up_sigla``, the siglas of the baseline results, next to the full names)
        and ``report`` (``votes_mean``, ``votes_max``, ``n_mean``, ``n_max``, ``participation_mean``,
        ``participation_max``); ``frames`` the ``last-seat`` twin.
    """
    other = study.scenarios['sin_litigio']['net'][['region_id', 'net']].rename(columns={'net': 'net_b'})
    last = study.scenarios['convocatoria']['net'].merge(other, on='region_id', how='left')
    last = _with_siglas(last, _siglas(study, study.params['election']['baseline']), ('winner', 'runner_up'))
    last = _table(last, decimals={'d_runner': 1, 'd_winner': 1, 'net': 1, 'net_b': 1, 'participation_needed': 2},
                  ints=['gap'])
    report = study.report
    data = {
        'scenarios': {name: _scenario(study, name) for name in SCENARIOS},
        'last_seat': _rows(last),
        'report': {'votes_mean': int(round(report['votes_mean'])), 'votes_max': int(round(report['votes_max'])),
                   'n_mean': int(report['n_mean']), 'n_max': int(report['n_max']),
                   'participation_mean': _value(report['participation_mean'], PCT_DECIMALS),
                   'participation_max': _value(report['participation_max'], PCT_DECIMALS)},
    }
    return data, {'last-seat': last}


def _grid(frame: pd.DataFrame) -> tuple:
    """The axes and the integer values of a grid indexed by participation."""
    return [float(v) for v in frame.index], [float(v) for v in frame.columns], frame.astype(int).values.tolist()


def export_study_sensitivity(study) -> tuple:
    """Export the three sensitivity grids of question 3.

    Parameters
    ----------
    study : CeraStudy
        The computed study.

    Returns
    -------
    tuple
        ``(data, frames)``: ``data`` for ``study-sensitivity`` with ``grid`` (``participations``,
        ``left_fracs``, ``values``: seats the left gains, one row per participation), ``scenarios`` (the rows of
        ``stress.scenario_table``: ``census_x``, ``part_x``, ``bias``, ``d_left``, ``d_right``) and ``placement``
        (``participations``, ``fracs``, ``values``: seats the PSOE takes, ``contrast`` and ``pool``); ``frames``
        the ``sensitivity-grid``, ``sensitivity-scenarios`` and ``placement-grid`` twins (the grids in long form).
    """
    participations, fracs, values = _grid(study.grid)
    p_participations, p_fracs, p_values = _grid(study.placement['grid'])
    scenarios = study.scenario_rows.astype(object).where(study.scenario_rows.notna(), None)
    data = {
        'grid': {'participations': participations, 'left_fracs': fracs, 'values': values},
        'scenarios': _rows(scenarios),
        'placement': {'participations': p_participations, 'fracs': p_fracs, 'values': p_values,
                      'contrast': int(study.placement['contrast']), 'pool': int(study.placement['pool'])},
    }
    frames = {
        'sensitivity-grid': study.grid.stack().rename('d_left').reset_index(),
        'sensitivity-scenarios': study.scenario_rows,
        'placement-grid': study.placement['grid'].stack().rename('seats').reset_index(),
    }
    return data, frames


def _gap_key(gap) -> tuple:
    """Sort key of a gap of the last seat: the tightest first, a missing one last."""
    missing = gap is None or pd.isna(gap)
    return missing, 0 if missing else gap


def _election_check(study, day: str) -> dict:
    """The summary of the check with and without the CERA vote of one election."""
    from ..studies.cera.study import VERDICT_BLOCK, VERDICT_SIGLA

    check, rows = study.checks[day], study.checks[day]['rows']
    voters, electors = int(study.voters[day].sum()), int(study.scrutinies[day]['electors'].sum())
    decided = rows.loc[rows['decided'], ['region_id', 'name', 'winner_with', 'winner_without', 'crosses_block']]
    decided = _with_siglas(decided, _siglas(study, day), WINNERS)
    return {'date': day, 'voters': voters, 'rate': _value(100. * voters / electors, PCT_DECIMALS),
            'n_over_gap': int(check['n_over_gap']), 'n_decided': int(check['n_decided']), 'decided': _rows(decided),
            'net_psoe': int(check['net_by_sigla'].get(VERDICT_SIGLA, 0)),
            'net_left': int(check['net_by_block'].get(VERDICT_BLOCK, 0)),
            'net_by_block': {block: int(net) for block, net in check['net_by_block'].items()}}


def export_study_check(study) -> tuple:
    """Export the check with and without the CERA vote of every election with a CERA scrutiny, and the verdict.

    Parameters
    ----------
    study : CeraStudy
        The computed study.

    Returns
    -------
    tuple
        ``(data, frames)``: ``data`` for ``study-check`` with ``elections`` (by date: ``date``, ``voters``,
        ``rate``, ``n_over_gap``, ``n_decided``, ``decided`` (``region_id``, ``name``, ``winner_with``,
        ``winner_with_sigla``, ``winner_without``, ``winner_without_sigla``, ``crosses_block``), ``net_psoe``,
        ``net_left``, ``net_by_block``), ``rows`` (the provinces of the latest election, by ``gap``, the ones
        without a gap last, with the ``_sigla`` of both winners) and ``verdict`` (``check.verdict`` of the study's
        election, ``'acierto'`` or ``'fallo'``, or ``None`` while it has no scrutiny); ``frames`` the ``check``
        twin (the rows with a ``date`` column).
    """
    from ..studies.cera.check import verdict

    days = sorted(study.checks)
    rows = study.checks[days[-1]]['rows']
    rows = rows.iloc[sorted(range(len(rows)), key=lambda i: _gap_key(rows['gap'].iloc[i]))].reset_index(drop=True)
    rows = _with_siglas(rows, _siglas(study, days[-1]), WINNERS)
    inputs = study.verdict_input
    data = {
        'elections': [_election_check(study, day) for day in days],
        'rows': _rows(rows),
        'verdict': None if inputs is None else verdict(max_gain=study.params['check']['max_gain'], **inputs),
    }
    return data, {'check': rows.assign(date=days[-1])[['date', *rows.columns]]}


def study_sources(study) -> list:
    """The official sources of the CERA study, with their links (``label``, ``url``).

    Parameters
    ----------
    study : CeraStudy
        The computed study.

    Returns
    -------
    list of dict
        The INE monthly workbook of the last close, the INE tables of the convocation census, the CERA
        scrutiny and the results of the Ministerio del Interior (Infoelectoral) and the regional electoral boards.
    """
    sources = [{'label': 'INE, Oficina del Censo Electoral: censo CERA cerrado cada mes por provincia y país',
                'url': MONTHLY_URL.format(yy=study.series_last.year % 100)}]
    for table in study.params['election']['census_tables']:
        match = ELECTION_TABLE.search(table)
        if match:
            year, n = match.groups()
            sources.append({'label': f'INE, Oficina del Censo Electoral: censo de consulta de las generales de {year} '
                                     f'(tabla {n})', 'url': ELECTION_TABLE_URL.format(year=year, n=n)})
    sources.append({'label': 'Ministerio del Interior: escrutinio del voto CERA por provincia y candidatura',
                    'url': INFOELECTORAL_URL})
    sources.append({'label': 'Ministerio del Interior, Infoelectoral: resultados del Congreso por provincia',
                    'url': INFOELECTORAL_URL})
    return sources + [{'label': label, 'url': url} for label, url in REGIONAL_SOURCES]


def registered_claim(study) -> dict:
    """The registered prediction of the CERA study: the checkable claim, its threshold and the seats it rests on.

    Parameters
    ----------
    study : CeraStudy
        The computed study.

    Returns
    -------
    dict
        ``claim`` (the sentence), ``max_gain`` (``params['check']['max_gain']``) and ``seats`` (the seats by
        block of each scenario, as in the ``prediction`` part).
    """
    max_gain = study.params['check']['max_gain']
    seats = 'escaño' if max_gain == 1 else 'escaños'
    claim = (f'El voto CERA de las generales del {_long_date(study.params["election"]["date"])} no dará al PSOE ni a '
             f'la izquierda un saldo neto de más de {max_gain} {seats}')
    return {'claim': claim, 'max_gain': max_gain,
            'seats': {name: _by_block(study.scenarios[name]['model']) for name in SCENARIOS}}


def export_study_meta(study, study_id: str = 'voto-cera', registration: Optional[dict] = None,
                      checked_at: Optional[str] = None, prov: Optional[dict] = None,
                      seconds: Optional[dict] = None) -> tuple:
    """Export the description of a study: titles, election, registration, parameters, sources and provenance.

    Parameters
    ----------
    study : CeraStudy
        The computed study.
    study_id : str
        The study slug (a key of ``STUDY_TITLES``).
    registration : dict, optional
        ``REGISTRATION_KEYS`` (missing keys are null).
    checked_at : str, optional
        Date of the first publication with the scrutiny of the study's election.
    prov : dict, optional
        ``commit``, ``dirty`` and ``versions`` (``provenance``); null commit, clean and empty by default.
    seconds : dict, optional
        Seconds by step.

    Returns
    -------
    tuple
        ``(data, {})``: ``data`` for ``study-meta`` with ``study``, ``title``, ``subtitle``, ``election``
        (``scope``, ``date``, ``census_closed``, ``census_total``, ``census_source``), the registration keys,
        ``checked_at``, ``params``, ``sources``, ``series_last`` (``YYYY-MM-DD``), ``regression`` (``p_slope``
        and ``p_level`` with three significant digits, the slopes and mean net additions with two decimals),
        ``commit``, ``dirty``, ``versions`` and ``seconds``. No CSV twin.
    """
    title, subtitle = STUDY_TITLES[study_id]
    election = study.params['election']
    table = next((t for t in election['census_tables'] if t.endswith('_tab8.xlsx')), election['census_tables'][-1])
    prov = prov or {'commit': None, 'dirty': False, 'versions': {}}
    regression = {key: (float(f'{value:.3g}') if key.startswith('p_') else _value(value, 2))
                  for key, value in study.regression.items()}
    data = {
        'study': study_id, 'title': title, 'subtitle': subtitle,
        'election': {'scope': election['scope'], 'date': election['date'], 'census_closed': election['census_closed'],
                     'census_total': int(study.census['electors'].sum()),
                     'census_source': f'INE, Oficina del Censo Electoral: {os.path.basename(table)}'},
        **{key: (registration or {}).get(key) for key in REGISTRATION_KEYS},
        'checked_at': checked_at, 'params': study.params, 'sources': study_sources(study),
        'series_last': _iso(study.series_last), 'regression': regression,
        'commit': prov['commit'], 'dirty': bool(prov['dirty']), 'versions': prov['versions'], 'seconds': seconds or {},
    }
    return data, {}


STUDY_EXPORTERS = {'census': export_study_census, 'participation': export_study_participation,
                   'vote': export_study_vote, 'placement': export_study_placement,
                   'prediction': export_study_prediction, 'sensitivity': export_study_sensitivity,
                   'check': export_study_check}


def publish_study(study_id: str, writer: BundleWriter, register: bool = False, force: bool = False,
                  data: Optional[FileSystem] = None, compute: Optional[Callable] = None,
                  prov: Optional[Callable] = None, today=None) -> dict:
    """Compute a study and write it into the bundle, overwriting the previous one, and update the manifest.

    The parts are written in the order of ``bundle.STUDY_PARTS[1:]`` (each JSON with its CSV twins) and ``meta``
    last, so a failure halfway leaves the previous ``meta.json`` in place; the manifest entry of the study
    (``published_at``, ``registered_at``, ``checked``) is updated after it. Runs, analyses and the other studies
    are not touched. The registration of the prediction is explicit: only ``register`` writes it, and a later
    publication without ``register`` keeps it (it reads the previous ``meta``).

    Parameters
    ----------
    study_id : str
        The study slug (one of ``bundle.STUDIES``).
    writer : BundleWriter
        Bundle destination.
    register : bool, default False
        Register the prediction (``registered_at``, ``registered_commit``, ``registered``); it needs a clean tree.
    force : bool, default False
        With ``register``, replace an existing registration, keeping its ``registered_at`` and writing
        ``registered_changed_at``.
    data : FileSystem, optional
        The data folder the study reads (``data/`` of the repository by default).
    compute : callable, optional
        ``compute(data)`` returning the study; ``studies.cera.compute_cera`` by default (imported here).
    prov : callable, optional
        Replacement of ``provenance``.
    today : str, datetime.date or datetime.datetime, optional
        Date of the registration, of its change and of the check; the date of the writer's clock by default.

    Returns
    -------
    dict
        ``{'study', 'registered_at', 'checked', 'seconds'}``.

    Raises
    ------
    ValueError
        If ``study_id`` is not one of ``bundle.STUDIES``.
    PublishRefused
        With ``register``, if the tree is dirty or the study is already registered and ``force`` is not given;
        nothing is written then.
    """
    if study_id not in bundle.STUDIES:
        raise ValueError(f'unknown study {study_id!r} (expected one of {", ".join(bundle.STUDIES)})')
    start = mark = time.perf_counter()

    def lap():
        """Seconds since the previous lap (or the start); restart the lap."""
        nonlocal mark
        now = time.perf_counter()
        elapsed, mark = now - mark, now
        return elapsed

    info = (prov or (lambda: provenance(cwd=REPO_ROOT)))()
    day = (_today(today) if today is not None else writer.now().date()).isoformat()
    meta_path = bundle.path_study(study_id, 'meta')
    previous = writer.read_json(meta_path)['data'] if writer.study_exists(study_id) else {}
    registration = {key: previous.get(key) for key in REGISTRATION_KEYS}
    if register and info['dirty']:
        raise PublishRefused('study: register needs a clean tree')
    if register and registration['registered_at'] and not force:
        raise PublishRefused(f'study: already registered on {registration["registered_at"]}')

    if compute is None:
        from ..studies.cera import compute_cera
        compute = compute_cera
    study = compute(data if data is not None else FileSystem(os.path.join(REPO_ROOT, 'data')))
    seconds = {'compute': lap()}
    if register:
        registration = {'registered_at': registration['registered_at'] or day, 'registered_commit': info['commit'],
                        'registered': registered_claim(study),
                        'registered_changed_at': day if registration['registered_at'] else None}

    def write(part, content, frames):
        """Write the JSON of a part and its CSV twins (``{csv name: frame}``)."""
        writer.write_json(bundle.path_study(study_id, part), bundle.STUDY_SCHEMAS[part], content, None)
        for name, frame in frames.items():
            writer.write_csv(bundle.path_study_csv(study_id, name), frame)

    checked = False
    for part in bundle.STUDY_PARTS[1:]:
        content, frames = STUDY_EXPORTERS[part](study)
        write(part, content, frames)
        seconds[part] = lap()
        if part == 'check':
            checked = content['verdict'] is not None
    checked_at = (previous.get('checked_at') or day) if checked else None
    seconds['total'] = time.perf_counter() - start
    seconds = {step: round(value, 1) for step, value in seconds.items()}
    meta, _ = export_study_meta(study, study_id, registration, checked_at, info, seconds)
    write('meta', meta, {})
    update_manifest(writer, studies={study_id: {'published_at': bundle.iso_utc(writer.now()),
                                                'registered_at': registration['registered_at'],
                                                'checked': checked}})
    return {'study': study_id, 'registered_at': registration['registered_at'], 'checked': checked,
            'seconds': seconds}
