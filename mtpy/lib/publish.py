"""Pure exporters of a finished simulation into JSON-ready dicts and CSV twin frames.

Every ``export_*`` function takes an already executed ``Simulator`` and returns ``(data, frame)``:
``data`` follows the bundle schema of the same name and ``frame`` is its tabular twin for the CSV.
Nothing here keeps state or touches storage.
"""
import os
import platform
import subprocess
import time
from datetime import date, datetime, timezone
from importlib import metadata
from typing import Callable, Optional, Sequence

import pandas as pd

from . import bundle
from .bundle import BundleReader, BundleWriter
from .data import get_next_event_date, get_scopes
from .simulator import Simulator
from ..models.elections import Polls

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

PCT_DECIMALS = 2
PROB_DECIMALS = 3
SEATS_DECIMALS = 1

PCT_COLS = ('pct', 'pct_mean', 'pct_lo', 'pct_hi', 'mean', 'lo', 'hi', 'sd')
PROB_COLS = ('p_seats', 'p_majority', 'p_first')
SEATS_COLS = ('seats_mean', 'seats_median', 'seats_lo', 'seats_hi', 'seats_min', 'seats_max')
POLL_COLUMNS = ['date', 'pollster_id', 'pollster', 'sponsor', 'start_date', 'end_date', 'sample_size', 'mtype',
                'rating', 'weight']
HE_COLUMNS = ['pollster_id', 'name', 'pollster', 'n', 'w', 'level', 'dev', 'dev_err', 'prior', 'prior_err',
              'effect', 'effect_err', 'center']
DISP_COLUMNS = ['pollster_id', 'pollster', 'n', 'ss_obs', 'ss_exp', 'ratio_raw', 'ratio', 'herd_ratio', 'factor']
DEFAULT_CORRECTORS = {'house_effects': True, 'dispersion': True, 'regional_noise': True, 'industry_bias': False,
                      'composition': None}
PROVENANCE_PACKAGES = ('numpy', 'pandas', 'scipy', 'statsmodels')
SUMMARY_COLS = ('pct', 'pct_lo', 'pct_hi', 'seats', 'seats_mean', 'seats_lo', 'seats_hi', 'p_seats')


def records(frame, index_name='name'):
    """Convert a frame into a list of row dicts, with its index as a leading column.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to convert.
    index_name : str
        Name given to the index column.

    Returns
    -------
    list of dict
        One dict per row; NaN values are kept (the bundle writer turns them into null).
    """
    return frame.rename_axis(index_name).reset_index().to_dict(orient='records')


def round_cols(frame, pct=(), prob=(), seats=()):
    """Round the present columns of each group to its number of decimals.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to round (not modified).
    pct, prob, seats : iterable of str
        Column names rounded to ``PCT_DECIMALS``, ``PROB_DECIMALS`` and ``SEATS_DECIMALS``.

    Returns
    -------
    pandas.DataFrame
        A rounded copy.
    """
    out = frame.copy()
    for cols, decimals in ((pct, PCT_DECIMALS), (prob, PROB_DECIMALS), (seats, SEATS_DECIMALS)):
        present = [c for c in cols if c in out.columns]
        if present:
            out[present] = out[present].round(decimals)
    return out


def _summary_table(sim, names=None):
    """Return ``sim.summary(names)`` rounded for publication, with ``seats`` as nullable integer."""
    table = round_cols(sim.summary(names), pct=PCT_COLS, prob=PROB_COLS, seats=SEATS_COLS)
    table['seats'] = table['seats'].astype('Int64')  # keeps ints; an empty block becomes NA (null)
    return table


def _p_majority(sim):
    """Return the probability of an absolute majority per block, rounded."""
    return sim.probabilities('vs').round(PROB_DECIMALS)


def export_vote(sim):
    """Export the vote share forecast.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``vote`` schema and its CSV twin.
    """
    table = round_cols(sim.vote_forecast(), pct=PCT_COLS)[['pct', 'sd', 'lo', 'hi']]
    rows = records(table, 'name')
    data = {'horizon': int(sim.horizon), 'when': sim.when.strftime('%Y-%m-%d'), 'rows': rows}
    return data, pd.DataFrame(rows, columns=['name', 'pct', 'sd', 'lo', 'hi'])


def export_summary(sim):
    """Export the seat summary for parties, vote-share blocks and seat blocks.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``summary`` schema and its CSV twin (tables stacked with a ``group`` column).
    """
    tables = {'parties': _summary_table(sim), 'vs': _summary_table(sim, 'vs'),
              'blocks': _summary_table(sim, 'blocks')}
    data = {
        'n_seats': int(sim.n_seats),
        'majority': int(sim.n_seats) // 2 + 1,
        **{group: records(table) for group, table in tables.items()},
        'p_majority': _p_majority(sim).to_dict(),
        'totals': {name: int(seats) for name, seats in sim.totals().items()},
    }
    frame = pd.concat([table.rename_axis('name').reset_index().assign(group=group)
                       for group, table in tables.items()], ignore_index=True)
    frame.insert(0, 'group', frame.pop('group'))
    return data, frame


def export_dist(sim):
    """Export the distribution of seats per simulation.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``dist`` schema and its CSV twin (simulations by parties).
    """
    frame = sim.dist()
    data = {'n_seats': int(sim.n_seats), 'parties': list(frame.columns), 'seats': frame.values.tolist()}
    return data, frame


def export_districts(sim):
    """Export votes and seats per district, leaving out the scope total.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``districts`` schema and its CSV twin.
    """
    region_ids = [r for r in sim.params['regions'] if r != sim.default_region]
    regions = [{'id': r, 'name': sim.region_names.get(r, r), 'seats': int(sim.reg_totals.loc[r, 'seats'])}
               for r in region_ids]
    rows = []
    for r in region_ids:
        table = round_cols(sim.unit_summary(r), pct=PCT_COLS, prob=PROB_COLS, seats=(*SEATS_COLS, 'seats'))
        for row in records(table[list(SUMMARY_COLS)]):
            rows.append({'region_id': r, 'region': sim.region_names.get(r, r), **row})
    columns = ['region_id', 'region', 'name', *SUMMARY_COLS]
    data = {'parties': list(sim.params['names']), 'regions': regions, 'rows': rows}
    return data, pd.DataFrame(rows, columns=columns)


def export_scenario(sim):
    """Export one randomly chosen simulation as a coherent seat scenario.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``scenario`` schema and its CSV twin.
    """
    i = sim.scenario()
    table = sim.result(i)
    parties = list(table.columns)
    region_ids = list(sim.params['regions'])
    rows = [{'region_id': region_id, 'region': region, 'seats': [int(s) for s in values]}
            for region_id, region, values in zip(region_ids, table.index, table.values)]
    frame = table.copy()
    frame.insert(0, 'region', list(table.index))
    frame.insert(0, 'region_id', region_ids)
    frame = frame.reset_index(drop=True)
    return {'simulation': int(i), 'parties': parties, 'rows': rows}, frame


def export_projection(sim):
    """Export the projected vote share over time for parties and both block groupings.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``projection`` schema and its long-format CSV twin.
    """
    groups, dates, long = {}, None, []
    for group, names in (('parties', None), ('vs', 'vs'), ('blocks', 'blocks')):
        out = sim.projection(names).round(PCT_DECIMALS)
        if dates is None:
            dates = list(out.index.strftime('%Y-%m-%d'))
        columns = list(out['mean'].columns)
        groups[group] = {'names': columns,
                         **{stat: {c: out[(stat, c)].tolist() for c in columns} for stat in ('mean', 'lo', 'hi')}}
        for c in columns:
            long.append(pd.DataFrame({'date': dates, 'group': group, 'name': c, 'mean': out[('mean', c)].values,
                                      'lo': out[('lo', c)].values, 'hi': out[('hi', c)].values}))
    frame = pd.concat(long, ignore_index=True)
    return {'dates': dates, 'groups': groups}, frame


def headline_mode(sim):
    """Build the headline of one mode: vote share, seats and win probability per party.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    dict
        ``{'parties': [...], 'p_majority': {block: p}}`` with parties in the order of the vote forecast.
    """
    vote = round_cols(sim.vote_forecast(), pct=PCT_COLS)
    summary = round_cols(sim.summary(), prob=PROB_COLS, seats=SEATS_COLS)
    parties = [{'name': name, 'pct': vote.loc[name, 'pct'], 'lo': vote.loc[name, 'lo'], 'hi': vote.loc[name, 'hi'],
                'seats': int(summary.loc[name, 'seats']), 'seats_lo': summary.loc[name, 'seats_lo'],
                'seats_hi': summary.loc[name, 'seats_hi'], 'p_first': summary.loc[name, 'p_first']}
               for name in vote.index]
    return {'parties': parties, 'p_majority': _p_majority(sim).to_dict()}


def _stat_bound(cell, key):
    """Return the rounded ``key`` of a ``fc_stat`` cell, or ``None`` when the cell holds no statistics."""
    return round(cell[key], PCT_DECIMALS) if isinstance(cell, dict) else None


def export_series(fc, names):
    """Export the fitted daily series of each party with its confidence bounds.

    Parameters
    ----------
    fc : Forecaster
        Fitted forecaster (``sim.model``).
    names : list of str
        Party names to export.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``series`` schema and its long-format CSV twin.
    """
    end = fc.date_fit_last if fc.date_fit_last is not None else fc.forecast.index.max()
    table = fc.forecast.loc[:end, names].dropna(how='all')
    dates = [d.strftime('%Y-%m-%d') for d in table.index]
    mean, lo, hi = {}, {}, {}
    for name in names:
        cells = [fc.fc_stat.loc[d, name] for d in table.index]
        mean[name] = list(table[name].round(PCT_DECIMALS))
        lo[name] = [_stat_bound(c, 'cmin') for c in cells]
        hi[name] = [_stat_bound(c, 'cmax') for c in cells]
    data = {'dates': dates, 'parties': list(names), 'mean': mean, 'lo': lo, 'hi': hi}
    rows = [{'date': d, 'name': name, 'mean': mean[name][i], 'lo': lo[name][i], 'hi': hi[name][i]}
            for name in names for i, d in enumerate(dates)]
    return data, pd.DataFrame(rows, columns=['date', 'name', 'mean', 'lo', 'hi'])


def export_polls(fc, names):
    """Export the published polls and the previous election result.

    Parameters
    ----------
    fc : Forecaster
        Fitted forecaster (``sim.model``).
    names : list of str
        Party names to export.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``polls`` schema and the polls table as its CSV twin.
    """
    polls = fc.fc_series_raw.reset_index()
    for column in POLL_COLUMNS:
        if column not in polls.columns:
            polls[column] = None
    polls = polls[POLL_COLUMNS + list(names)]
    polls[list(names)] = polls[list(names)].round(PCT_DECIMALS)
    polls['weight'] = polls['weight'].astype(float).round(PROB_DECIMALS)
    results = fc.nfc_series.reset_index()[['date', *names]]
    results = round_cols(results, pct=names)
    data = {'parties': list(names), 'columns': list(POLL_COLUMNS), 'polls': polls.to_dict(orient='records'),
            'results': results.to_dict(orient='records')}
    return data, polls


def export_fan(sim):
    """Export the fan of vote share per party and horizon.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``fan`` schema and its CSV twin.
    """
    fan = round_cols(sim.fan(), pct=('mean', 'sd', 'lo', 'hi'))
    horizons = sorted(int(h) for h in fan['horizon'].unique())
    return {'horizons': horizons, 'rows': fan.to_dict(orient='records')}, fan


def export_house_effects(fc):
    """Export the house effect of each pollster on each series.

    Parameters
    ----------
    fc : Forecaster
        Fitted forecaster (``sim.model``).

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``house-effects`` schema and its CSV twin; empty when none was fitted.
    """
    if fc.house_effects is None:
        return {'rows': []}, pd.DataFrame(columns=HE_COLUMNS)
    table = fc.house_effects.reset_index().reindex(columns=HE_COLUMNS)
    keep = ('pollster_id', 'name', 'pollster', 'n', 'w')
    table = round_cols(table, pct=[c for c in HE_COLUMNS if c not in keep])
    return {'rows': table.to_dict(orient='records')}, table


def export_dispersion(fc):
    """Export the dispersion and herding diagnostics of each pollster.

    Parameters
    ----------
    fc : Forecaster
        Fitted forecaster (``sim.model``).

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``dispersion`` schema and its CSV twin; empty when none was fitted.
    """
    if fc.dispersion is None:
        return {'rows': []}, pd.DataFrame(columns=DISP_COLUMNS)
    table = fc.dispersion.reset_index().reindex(columns=DISP_COLUMNS)
    keep = ('pollster_id', 'pollster', 'n')
    table = round_cols(table, prob=[c for c in DISP_COLUMNS if c not in keep])
    return {'rows': table.to_dict(orient='records')}, table


def n_polls(fc):
    """Count the published polls and the distinct pollsters.

    Parameters
    ----------
    fc : Forecaster
        Fitted forecaster (``sim.model``).

    Returns
    -------
    tuple of (int, int)
        Number of polls and number of pollsters.
    """
    return int(fc.fc_series_raw.shape[0]), int(fc.fc_series_raw['pollster_id'].nunique())


def _git(args, cwd):
    """Run a git command and return its decoded output, or ``None`` if it fails."""
    try:
        return subprocess.check_output(['git', *args], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def provenance(cwd: Optional[str] = None) -> dict:
    """Collect the code version and the library versions of the run.

    Parameters
    ----------
    cwd : str, optional
        Directory where git commands run.

    Returns
    -------
    dict
        ``commit`` (short SHA, ``GIT_COMMIT`` env var or ``None``), ``dirty`` (uncommitted changes;
        ``False`` if git is unavailable) and ``versions`` (Python and main libraries).
    """
    sha, porcelain = _git(['rev-parse', '--short', 'HEAD'], cwd), _git(['status', '--porcelain'], cwd)
    versions = {'python': platform.python_version()}
    for package in PROVENANCE_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except Exception:
            versions[package] = None
    return {'commit': sha or os.getenv('GIT_COMMIT') or None, 'dirty': bool(porcelain),
            'versions': versions}


def _meta_parties(sim):
    """Return the catalogue rows of the simulated parties, in simulation order."""
    catalogue = sim.parties.set_index('name') if 'name' in sim.parties.columns else sim.parties
    return [{'name': name, 'id': int(catalogue.loc[name, 'id']), 'fullname': catalogue.loc[name, 'fullname'],
             'color': catalogue.loc[name, 'color'], 'block': catalogue.loc[name, 'block'],
             'regional': int(sim.forecast.loc[name, 'regional'])} for name in sim.names]


def export_meta(sim, run_id: str, run_at: str, n_sim: int, max_fc: int, correctors: dict, seconds: dict,
                clip_rate: dict, db_polls: Optional[int], db_last_poll: Optional[str], prov: dict,
                freeze: bool) -> dict:
    """Build the ``meta`` document describing how and when a run was produced.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.
    run_id, run_at : str
        Identifier and ISO timestamp of the run.
    n_sim, max_fc : int
        Simulations per mode and maximum number of forecast days.
    correctors : dict
        Correctors enabled in the run.
    seconds : dict
        Seconds spent in each phase.
    clip_rate : dict
        Share of clipped simulations per mode.
    db_polls, db_last_poll : int or str, optional
        Number of polls in the database and date of the latest one.
    prov : dict
        Result of ``provenance``.
    freeze : bool
        Whether the run is a frozen result.

    Returns
    -------
    dict
        Data for the ``meta`` schema.
    """
    fc = sim.model
    n_seats = int(sim.n_seats)
    polls, pollsters = n_polls(fc)
    drift = sim.v2drift
    regions = [{'id': int(r), 'name': sim.region_names.get(r, r), 'seats': int(sim.reg_totals.loc[r, 'seats'])}
               for r in sim.regions]
    return {
        'run_id': run_id, 'run_at': run_at, 'commit': prov['commit'], 'dirty': bool(prov['dirty']),
        'versions': prov['versions'], 'scope': sim.scope, 'event_date': sim.event_date,
        'as_of': sim.as_of.strftime('%Y-%m-%d'), 'date_last': fc.date_last.strftime('%Y-%m-%d'),
        'date_fit_last': fc.date_fit_last.strftime('%Y-%m-%d') if fc.date_fit_last is not None else None,
        'horizon_max': int(sim.horizon_max), 'n_sim': int(n_sim),
        'seed': None if sim.seed is None else int(sim.seed), 'drange': list(sim.drange), 'max_fc': int(max_fc),
        'alpha': float(sim.alpha), 'correctors': correctors, 'n_polls': polls, 'n_pollsters': pollsters,
        'db_polls': db_polls, 'db_last_poll': db_last_poll, 'n_seats': n_seats, 'majority': n_seats // 2 + 1,
        'parties': _meta_parties(sim), 'bmaps': fc.bmaps, 'smap': sim.smap, 'regions': regions,
        'diagnostics': {'drift_k': float(drift.k) if drift is not None else None,
                        'multiplier': float(drift.multiplier) if drift is not None else None,
                        'ages': sim.ages.round(PCT_DECIMALS).to_dict(), 'composition': float(sim.composition_ratio),
                        'clip_rate': clip_rate},
        'seconds': seconds, 'freeze': bool(freeze),
    }


def export_headline(sim, run_id: str, run_at: str, nowcast: dict, forecast: dict) -> dict:
    """Build the ``headline`` document: the run summary shown first on the site.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.
    run_id, run_at : str
        Identifier and ISO timestamp of the run.
    nowcast, forecast : dict
        Results of ``headline_mode`` for each mode.

    Returns
    -------
    dict
        Data for the ``headline`` schema.
    """
    return {'run_id': run_id, 'run_at': run_at, 'event_date': sim.event_date,
            'as_of': sim.as_of.strftime('%Y-%m-%d'), 'date_last': sim.model.date_last.strftime('%Y-%m-%d'),
            'n_polls': n_polls(sim.model)[0], 'nowcast': nowcast, 'forecast': forecast}


ATTRIBUTION = {'polls': 'Sondeos: Wikipedia, CC BY-SA 4.0', 'results': 'Origen de los datos: Ministerio del Interior',
               'model': 'Modelo y curación: Luis Sancho'}


class PublishRefused(Exception):
    """The publication was refused by a guard (for instance the LOREG window)."""


def loreg_guard(scope: str, event_date: str, today=None, force: bool = False) -> bool:
    """Refuse to publish national forecasts in the five days before the election (LOREG art. 69.7).

    Parameters
    ----------
    scope : str
        Scope code; only ``'es'`` is guarded.
    event_date : str
        Election date, ``YYYY-MM-DD``.
    today : str or datetime.date, optional
        Current date; defaults to today in UTC.
    force : bool
        Publish even inside the window.

    Returns
    -------
    bool
        Whether the date falls inside the window.

    Raises
    ------
    PublishRefused
        Inside the window and not forced.
    """
    if today is None:
        today = datetime.now(timezone.utc).date()
    elif isinstance(today, str):
        today = date.fromisoformat(today)
    event = date.fromisoformat(event_date)
    in_window = scope == 'es' and 0 <= (event - today).days <= 5
    if in_window and not force:
        raise PublishRefused('{} {}: inside the LOREG window (art. 69.7); pass force=true to publish'.format(
            scope, event_date))
    return in_window


def resolve_scopes(scopes, catalogue: Optional[pd.DataFrame] = None) -> list:
    """Expand the requested scopes into a list of scope codes.

    Parameters
    ----------
    scopes : str or sequence of str
        ``'all'`` (``es`` plus every scope with a parent), one scope code or a sequence of them.
    catalogue : pandas.DataFrame, optional
        Scope catalogue indexed by ``scode`` with a ``parent`` column; defaults to ``get_scopes()``.

    Returns
    -------
    list of str
        Scope codes to publish.

    Raises
    ------
    ValueError
        Some scope is not in the catalogue.
    """
    catalogue = get_scopes() if catalogue is None else catalogue
    if isinstance(scopes, str):
        if scopes == 'all':
            scopes = ['es'] + [s for s in catalogue.index if s != 'es' and pd.notnull(catalogue.loc[s, 'parent'])]
        else:
            scopes = [scopes]
    else:
        scopes = list(scopes)
    unknown = [s for s in scopes if s not in catalogue.index]
    if unknown:
        raise ValueError('publish: unknown scopes {}'.format(sorted(unknown)))
    return scopes


def next_event_date(scope: str) -> Optional[str]:
    """Return the date of the next event of the scope, or ``None`` when there is none.

    Parameters
    ----------
    scope : str
        Scope code.

    Returns
    -------
    str or None
        ``YYYY-MM-DD`` date.
    """
    return get_next_event_date(scope)


def db_stats(scope: str, event_date: str) -> tuple:
    """Count the polls of the event in the database and find the date of the latest one.

    Parameters
    ----------
    scope, event_date : str
        Event scope and date.

    Returns
    -------
    tuple
        ``(number of polls, 'YYYY-MM-DD' of the latest)``; ``(0, None)`` when there are none.
    """
    polls = Polls().get_results(query=dict(filters=["event_scope = '{}'".format(scope),
                                                    "event_date = '{}'".format(event_date)]), formatted=True)
    if polls.shape[0] == 0:
        return 0, None
    return int(polls.shape[0]), polls['date'].max().strftime('%Y-%m-%d')


MODE_EXPORTERS = (('vote', export_vote), ('summary', export_summary), ('dist', export_dist),
                  ('districts', export_districts), ('scenario', export_scenario), ('projection', export_projection))


def publish_mode(sim, writer: BundleWriter, scope: str, run_id: str) -> dict:
    """Write the files of the current mode of the simulator (JSON and CSV per part).

    Parameters
    ----------
    sim : Simulator
        Simulator already run in ``sim.mode``.
    writer : BundleWriter
        Bundle destination.
    scope, run_id : str
        Scope and run being written.

    Returns
    -------
    dict
        ``headline_mode`` of the simulator.
    """
    mode = sim.mode
    for part, exporter in MODE_EXPORTERS:
        data, frame = exporter(sim)
        writer.write_json(bundle.path_part(scope, run_id, part, mode), part, data, scope, run_id, mode)
        writer.write_csv(bundle.path_csv(scope, run_id, part, mode), frame)
    return headline_mode(sim)


def latest_entry(headline: dict) -> dict:
    """Build the manifest entry of a scope from the headline of its latest run.

    Parameters
    ----------
    headline : dict
        Data of ``headline.json``.

    Returns
    -------
    dict
        Entry with ``latest``, ``run_at``, ``event_date``, ``as_of``, ``date_last`` and ``n_polls``.
    """
    return {'latest': headline['run_id'], 'run_at': headline['run_at'], 'event_date': headline['event_date'],
            'as_of': headline['as_of'], 'date_last': headline['date_last'], 'n_polls': headline['n_polls']}


def publish_forecast(scope: str, writer: BundleWriter, run_id: str, event_date: str, n_sim: int = 1000,
                     seed: int = 42, drange=6, max_fc: int = 10, alpha: float = 0.05,
                     correctors: Optional[dict] = None, freeze: bool = False, verbose: int = 0,
                     simulator: Optional[Callable] = None, stats: Optional[Callable] = None,
                     prov: Optional[Callable] = None) -> dict:
    """Run the model for one scope and write an immutable run into the bundle.

    Pointers (history and manifest) are not touched. ``headline.json`` is written last, so a run
    without it is incomplete.

    Parameters
    ----------
    scope : str
        Scope code.
    writer : BundleWriter
        Bundle destination.
    run_id : str
        Run identifier (``YYYYMMDD-HHMMSS``).
    event_date : str
        Election date, ``YYYY-MM-DD``.
    n_sim : int
        Number of simulations per mode.
    seed, drange, max_fc, alpha
        Model parameters.
    correctors : dict, optional
        Overrides of ``DEFAULT_CORRECTORS``.
    freeze : bool
        Whether the site is frozen (recorded in ``meta``).
    verbose : int
        Simulator verbosity.
    simulator, stats, prov : callable, optional
        Injectable replacements of ``Simulator``, ``db_stats`` and ``provenance``.

    Returns
    -------
    dict
        ``{'scope', 'entry', 'seconds'}``.

    Raises
    ------
    FileExistsError
        The run already exists.
    """
    writer.begin_run(scope, run_id)
    run_at = bundle.iso_utc(writer.now())
    correctors = {**DEFAULT_CORRECTORS, **(correctors or {})}
    seconds, heads, clip = {}, {}, {}
    t0 = t = time.perf_counter()
    sim = (simulator or Simulator)(scope=scope, event_date=event_date, drange=drange, alpha=alpha, seed=seed,
                                   mode='nowcast', verbose=verbose, path='.', **correctors)
    seconds['init'], t = time.perf_counter() - t, time.perf_counter()
    names = sim.params['names']
    sim.fit_forecast(names=names, max_fc=max_fc, fillna=True)
    seconds['fit'], t = time.perf_counter() - t, time.perf_counter()
    sim.run(split=True, random=True, n_sim=n_sim)
    clip['nowcast'] = float(sim.clip_rate())
    seconds['nowcast'], t = time.perf_counter() - t, time.perf_counter()
    heads['nowcast'] = publish_mode(sim, writer, scope, run_id)
    sim.mode = 'forecast'
    sim.run(split=True, random=True, n_sim=n_sim, horizon='deadline')
    clip['forecast'] = float(sim.clip_rate())
    seconds['forecast'], t = time.perf_counter() - t, time.perf_counter()
    heads['forecast'] = publish_mode(sim, writer, scope, run_id)
    fc = sim.model
    cycle = (('series', export_series(fc, names)), ('polls', export_polls(fc, names)), ('fan', export_fan(sim)),
             ('house-effects', export_house_effects(fc)), ('dispersion', export_dispersion(fc)))
    for part, (data, frame) in cycle:
        writer.write_json(bundle.path_part(scope, run_id, part), part, data, scope, run_id)
        writer.write_csv(bundle.path_csv(scope, run_id, part), frame)
    db_polls, db_last_poll = (stats or db_stats)(scope, event_date)
    prov = prov or (lambda: provenance(cwd=REPO_ROOT))
    seconds['export'] = time.perf_counter() - t
    seconds['total'] = time.perf_counter() - t0
    meta = export_meta(sim, run_id, run_at, n_sim, max_fc, correctors, seconds, clip, db_polls, db_last_poll,
                       prov(), freeze)
    writer.write_json(bundle.path_part(scope, run_id, 'meta'), 'meta', meta, scope, run_id)
    headline = export_headline(sim, run_id, run_at, heads['nowcast'], heads['forecast'])
    writer.write_json(bundle.path_part(scope, run_id, 'headline'), 'headline', headline, scope, run_id)
    return {'scope': scope, 'entry': latest_entry(headline), 'seconds': seconds}


def rebuild_history(writer: BundleWriter, scope: str) -> dict:
    """Rebuild ``history.json`` of a scope from the headlines of its complete runs.

    Parameters
    ----------
    writer : BundleWriter
        Bundle to read and write.
    scope : str
        Scope code.

    Returns
    -------
    dict
        The history data, ``{'scope', 'runs'}``.
    """
    runs = [writer.read_json(bundle.path_part(scope, rid, 'headline'))['data'] for rid in writer.list_runs(scope)]
    data = {'scope': scope, 'runs': runs}
    writer.write_json(bundle.path_history(scope), 'history', data, scope, run_id=None)
    return data


def read_manifest(reader: BundleReader) -> dict:
    """Read the manifest data, or the default one when the bundle has none yet.

    Parameters
    ----------
    reader : BundleReader
        Bundle to read.

    Returns
    -------
    dict
        Manifest data.
    """
    if reader.exists(bundle.path_manifest()):
        return reader.read_json(bundle.path_manifest())['data']
    return {'contract': bundle.CONTRACT, 'updated_at': None, 'scopes': {},
            'freeze': {'active': False, 'message': None}, 'attribution': dict(ATTRIBUTION)}


def _write_manifest(writer: BundleWriter, data: dict) -> dict:
    """Stamp and write the manifest data; return it."""
    data['updated_at'] = bundle.iso_utc(writer.now())
    data['contract'] = bundle.CONTRACT
    writer.write_json(bundle.path_manifest(), 'manifest', data, scope=None)
    return data


def update_manifest(writer: BundleWriter, scopes: Optional[dict] = None, freeze: Optional[dict] = None) -> dict:
    """Merge scope entries and/or the freeze switch into the manifest and write it.

    Parameters
    ----------
    writer : BundleWriter
        Bundle to update.
    scopes : dict, optional
        Entries by scope code; each replaces the existing entry of its scope.
    freeze : dict, optional
        ``{'active': bool, 'message': str | None}``.

    Returns
    -------
    dict
        The manifest data written.

    Raises
    ------
    ValueError
        ``freeze`` has no boolean ``active``.
    """
    if freeze is not None and not (isinstance(freeze, dict) and isinstance(freeze.get('active'), bool)):
        raise ValueError('publish: freeze needs a boolean "active"')
    data = read_manifest(writer)
    data['scopes'].update(scopes or {})
    if freeze is not None:
        data['freeze'] = {'active': freeze['active'], 'message': freeze.get('message')}
    return _write_manifest(writer, data)


def point(writer: BundleWriter, scope: str, run_id: str) -> dict:
    """Point the manifest of a scope to an existing run.

    Parameters
    ----------
    writer : BundleWriter
        Bundle to update.
    scope, run_id : str
        Scope and run to publish as the latest.

    Returns
    -------
    dict
        The manifest data written.

    Raises
    ------
    ValueError
        The run does not exist or is incomplete.
    """
    name = bundle.path_part(scope, run_id, 'headline')
    if not writer.exists(name):
        raise ValueError('{}: run {} not found'.format(scope, run_id))
    return update_manifest(writer, {scope: latest_entry(writer.read_json(name)['data'])})


def unpublish(writer: BundleWriter, scope: str, run_id: str) -> dict:
    """Delete a run, rebuild the history and repoint the manifest if it pointed to that run.

    Parameters
    ----------
    writer : BundleWriter
        Bundle to update.
    scope, run_id : str
        Run to delete.

    Returns
    -------
    dict
        ``{'removed': run_id, 'latest': run now published for the scope or None}``.

    Raises
    ------
    ValueError
        The run does not exist.
    """
    if not writer.exists(bundle.path_run(scope, run_id)):
        raise ValueError('{}: run {} not found'.format(scope, run_id))
    writer.remove(bundle.path_run(scope, run_id))
    rebuild_history(writer, scope)
    manifest = read_manifest(writer)
    latest = manifest['scopes'].get(scope, {}).get('latest')
    if latest == run_id:
        runs = writer.list_runs(scope)
        if runs:
            point(writer, scope, runs[-1])
            latest = runs[-1]
        else:
            del manifest['scopes'][scope]
            _write_manifest(writer, manifest)
            latest = None
    return {'removed': run_id, 'latest': latest}
