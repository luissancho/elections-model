"""Exporters of one forecast run, the backfill helpers and ``publish_forecast``."""
import time
from datetime import date, datetime, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .. import bundle
from ..bundle import BundleWriter
from ...core.app import App
from ..data import get_district_communities
from ..simulator import Simulator
from ..utils import REST_BLOCK, norm_range, web_blocks
from .common import (
    db_stats, ISO_DAY_RE, latest_entry, NothingToPublish, PCT_DECIMALS, PROB_DECIMALS, provenance, records,
    REPO_ROOT, round_cols, _today
)


PCT_COLS = ('pct', 'pct_mean', 'pct_lo', 'pct_hi', 'mean', 'lo', 'hi', 'sd')
# Columns of the ``fan`` part and its CSV: the six historical ones first, then the pieces of the margin (2026-10-25)
FAN_COLUMNS = ['name', 'horizon', 'mean', 'sd', 'lo', 'hi', 'err', 'pct_err', 'drift', 'dof']


PROB_COLS = ('p_seats', 'p_majority', 'p_first')


SEATS_COLS = ('seats_mean', 'seats_median', 'seats_lo', 'seats_hi', 'seats_min', 'seats_max')


POLL_COLUMNS = ['date', 'pollster_id', 'pollster', 'sponsor', 'start_date', 'end_date', 'sample_size', 'mtype',
                'rating', 'weight']


HE_COLUMNS = ['pollster_id', 'name', 'pollster', 'n', 'w', 'level', 'dev', 'dev_err', 'prior', 'prior_err',
              'effect', 'effect_err', 'center']


DISP_COLUMNS = ['pollster_id', 'pollster', 'n', 'ss_obs', 'ss_exp', 'ratio_raw', 'ratio', 'herd_ratio', 'factor']
WEIGHTS_COLUMNS = ['name', 'date', 'pollster_id', 'pollster', 'sponsor', 'sample_size', 'pct', 'pct_adj', 'weight',
                   'weight_kernel', 'weight_over', 'weight_sample', 'weight_rating']


DEFAULT_CORRECTORS = {'house_effects': True, 'dispersion': True, 'regional_noise': True, 'industry_bias': False,
                      'composition': None}


SUMMARY_COLS = ('pct', 'pct_lo', 'pct_hi', 'seats', 'seats_median', 'seats_mean', 'seats_lo', 'seats_hi', 'p_seats')


def _summary_table(sim, names=None):
    """Return ``sim.summary(names)`` rounded for publication, with ``seats`` as nullable integer."""
    table = round_cols(sim.summary(names), pct=PCT_COLS, prob=PROB_COLS, seats=SEATS_COLS)
    table['seats'] = table['seats'].astype('Int64')  # keeps ints; an empty block becomes NA (null)
    return table


def _web_bmap(sim):
    """Return the Izquierda / Derecha / Otros map of the scope, or ``None`` if its ``vs`` lacks a side."""
    return web_blocks(sim.model.bmaps.get('vs') or {}, list(sim.params['names']))


def _blocks_summary_table(sim, bmap):
    """Summary of Izquierda, Derecha and Otros (the rest) for publication.

    Otros takes the vote as the complement of Izquierda + Derecha (the residual ``'-'`` is not in the simulated
    shares, so a member sum would miss it) and the seats of its members, which are the complement of the chamber
    in a split run. Without members its seats are 0 and its probabilities null. It has no ``p_majority``.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.
    bmap : dict
        Output of ``web_blocks``.

    Returns
    -------
    pandas.DataFrame
        One row per block (Izquierda, Derecha, Otros), as ``_summary_table``.
    """
    sides = {k: bmap[k] for k in ('Izquierda', 'Derecha')}
    union = _summary_table(sim, {'union': bmap['Izquierda'] + bmap['Derecha']}).loc['union']
    table = _summary_table(sim, {**sides, **({REST_BLOCK: bmap[REST_BLOCK]} if bmap[REST_BLOCK] else {})})
    if REST_BLOCK not in table.index:
        table = table.reindex([*table.index, REST_BLOCK])
        zeros = [c for c in table.columns if c.startswith('seats')]
        table.loc[REST_BLOCK, zeros] = 0
        table['seats'] = table['seats'].astype('Int64')
    table.loc[REST_BLOCK, ['pct', 'pct_mean']] = (100 - union[['pct', 'pct_mean']]).round(PCT_DECIMALS).values
    table.loc[REST_BLOCK, ['pct_lo', 'pct_hi']] = (100 - union[['pct_hi', 'pct_lo']]).round(PCT_DECIMALS).values
    table.loc[REST_BLOCK, 'p_majority'] = np.nan
    return table.loc[list(bmap)]


def _blocks_projection(sim, bmap):
    """Projected vote of Izquierda, Derecha and Otros, the last as the complement of the first two.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.
    bmap : dict
        Output of ``web_blocks``.

    Returns
    -------
    pandas.DataFrame
        Same layout as ``sim.projection``; the band of Otros is clipped to 0-100.
    """
    union_key = '__vs__'
    out = sim.projection({'Izquierda': bmap['Izquierda'], 'Derecha': bmap['Derecha'],
                          union_key: bmap['Izquierda'] + bmap['Derecha']})
    rest = pd.DataFrame({'mean': 100 - out[('mean', union_key)], 'lo': (100 - out[('hi', union_key)]).clip(lower=0),
                         'hi': (100 - out[('lo', union_key)]).clip(upper=100)})
    for stat in ('mean', 'lo', 'hi'):
        out[(stat, REST_BLOCK)] = rest[stat]
    columns = [(stat, name) for stat in ('mean', 'lo', 'hi') for name in (*bmap,)]
    return out[columns]


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
    web = _web_bmap(sim)
    tables = {'parties': _summary_table(sim), 'vs': _summary_table(sim, 'vs'),
              'blocks': _blocks_summary_table(sim, web) if web is not None else _summary_table(sim, 'blocks')}
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


def export_districts(sim, communities: Optional[dict] = None):
    """Export votes and seats per district, leaving out the scope total.

    The ``votes`` of each district are the valid votes of the previous election (``sim.prev_totals``,
    rounded to an integer): the totals of the upcoming election (``sim.reg_totals``) carry its seats but
    no votes, since the loader cannot know them yet.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.
    communities : dict, optional
        ``{region_id: {'code', 'name'} | None}``, the autonomous community of each district
        (``data.get_district_communities``). Without it every ``community`` is ``None``.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``districts`` schema and its CSV twin.
    """
    region_ids = [r for r in sim.params['regions'] if r != sim.default_region]
    regions = [{'id': r, 'name': sim.region_names.get(r, r), 'seats': int(sim.reg_totals.loc[r, 'seats']),
                'community': communities.get(r) if communities else None,
                'votes': int(round(float(sim.prev_totals.loc[r, 'votes'])))}
               for r in region_ids]
    rows = []
    for r in region_ids:
        table = round_cols(sim.unit_summary(r), pct=PCT_COLS, prob=PROB_COLS, seats=SEATS_COLS)
        for row in records(table[list(SUMMARY_COLS)]):
            row['seats'] = int(row['seats'])
            rows.append({'region_id': r, 'region': sim.region_names.get(r, r), **row})
    columns = ['region_id', 'region', 'name', *SUMMARY_COLS]
    data = {'parties': list(sim.params['names']), 'regions': regions, 'rows': rows}
    return data, pd.DataFrame(rows, columns=columns)


def export_scenario(sim):
    """Export the central seat allocation of the total region and every district.

    The table is ``sim.central_result()``, the allocation of the central vote shares. It is the same in
    nowcast and horizon modes and is the one the headline of the seats shows.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``scenario`` schema and its CSV twin.
    """
    table = sim.central_result()
    parties = list(table.columns)
    region_ids = list(sim.params['regions'])
    rows = [{'region_id': region_id, 'region': region, 'seats': [int(s) for s in values]}
            for region_id, region, values in zip(region_ids, table.index, table.values)]
    frame = table.copy()
    frame.insert(0, 'region', list(table.index))
    frame.insert(0, 'region_id', region_ids)
    frame = frame.reset_index(drop=True)
    return {'parties': parties, 'rows': rows}, frame


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
    web = _web_bmap(sim)
    for group, names in (('parties', None), ('vs', 'vs'), ('blocks', 'blocks')):
        out = (_blocks_projection(sim, web) if group == 'blocks' and web is not None
               else sim.projection(names)).round(PCT_DECIMALS)
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


def export_series(fc, names, end=None):
    """Export the fitted daily series of each party with its confidence bounds.

    Parameters
    ----------
    fc : Forecaster
        Fitted forecaster (``sim.model``).
    names : list of str
        Party names to export.
    end : str or pandas.Timestamp, optional
        Last day exported, inclusive. By default the last fitted day (``fc.date_fit_last``, or the end of the
        series without it); an ``end`` beyond it changes nothing, an earlier one cuts the series there (the
        backtest cuts at the anchor of each case, so the series stops where the model was frozen, and since
        2026-10-26 the publish cuts at ``sim.as_of`` too, where the projection starts).

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``series`` schema and its long-format CSV twin.
    """
    last = fc.date_fit_last if fc.date_fit_last is not None else fc.forecast.index.max()
    end = last if end is None else min(pd.Timestamp(end), pd.Timestamp(last))
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


def poll_seats(fc) -> dict:
    """Seat projections of the polls, from ``fc.load_poll_seats()`` when the forecaster has it.

    Parameters
    ----------
    fc : Forecaster
        Fitted forecaster (``sim.model``). A forecaster without ``load_poll_seats`` (a test double) publishes
        no projections.

    Returns
    -------
    dict
        ``{(date, pollster_id, sponsor_id): {party: [min, max]}}`` (see ``Forecaster.load_poll_seats``).
    """
    loader = getattr(fc, 'load_poll_seats', None)
    return loader() if callable(loader) else {}


def export_poll_rows(polls, names, seats=None):
    """Rows of a polls table as published: the ``POLL_COLUMNS``, the share of each party and the seat projections.

    Parameters
    ----------
    polls : pandas.DataFrame
        Polls with the columns of ``POLL_COLUMNS`` that exist (``date``, ``pollster_id`` and ``sponsor_id`` as
        columns, not index levels; a missing column is exported null) and one share per party.
    names : list of str
        Party names to export.
    seats : dict, optional
        Seat projections by ``(date, pollster_id, sponsor_id)`` (``poll_seats``); none by default.

    Returns
    -------
    tuple of (list of dict, pandas.DataFrame)
        The records of the ``polls`` member of the ``polls`` and ``event-cycle`` schemas and the table of their
        CSV twin. A poll with projections carries ``seats`` (``{party: [min, max]}``, only the parties that have
        one); the table adds ``seats_min_{party}`` and ``seats_max_{party}`` columns (nullable integers).
    """
    polls = polls.copy()
    for column in POLL_COLUMNS:
        if column not in polls.columns:
            polls[column] = None
    seats = seats or {}
    names = list(names)
    keys = [
        (pd.Timestamp(d).strftime('%Y-%m-%d'), int(p), int(s) if pd.notnull(s) else 0)
        for d, p, s in zip(polls['date'], polls['pollster_id'], polls['sponsor_id'] if 'sponsor_id' in polls.columns else [None] * len(polls))
    ]
    # Both tables are assembled at once with ``concat``: assigning the rounded shares column by column would leave
    # one block per party and, past a hundred of them, every seat column inserted afterwards would warn about
    # fragmentation (pandas ``PerformanceWarning``).
    polls = pd.concat([polls[POLL_COLUMNS], polls[names].round(PCT_DECIMALS)], axis=1)
    polls['weight'] = polls['weight'].astype(float).round(PROB_DECIMALS)
    records = polls.to_dict(orient='records')
    for key, record in zip(keys, records):
        found = {name: seats[key][name] for name in names if name in seats.get(key, {})}
        if found:
            record['seats'] = found
    projections = {}
    for name in names:
        projections['seats_min_' + name] = pd.array([seats.get(k, {}).get(name, [None])[0] for k in keys], dtype='Int64')
        projections['seats_max_' + name] = pd.array([seats.get(k, {}).get(name, [None, None])[-1] for k in keys], dtype='Int64')
    return records, pd.concat([polls, pd.DataFrame(projections, index=polls.index)], axis=1)


def export_polls(fc, names, seats=None):
    """Export the published polls, their seat projections and the previous election result.

    Parameters
    ----------
    fc : Forecaster
        Fitted forecaster (``sim.model``).
    names : list of str
        Party names to export.
    seats : dict, optional
        Seat projections by ``(date, pollster_id, sponsor_id)`` (``poll_seats``); none by default.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``polls`` schema and the polls table as its CSV twin (see ``export_poll_rows``).
    """
    records, polls = export_poll_rows(fc.fc_series_raw.reset_index(), names, seats)
    results = fc.nfc_series.reset_index()[['date', *names]]
    results = round_cols(results, pct=names)
    data = {'parties': list(names), 'columns': list(POLL_COLUMNS), 'polls': records,
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
        Data for the ``fan`` schema and its CSV twin; rows are ``name, horizon, mean, sd, lo, hi`` plus, since
        2026-10-25, the three pieces of the margin in points, ``err`` (standard error of the average), ``pct_err``
        (terminal polling error) and ``drift`` (drift at that horizon, 0 at horizon 0), with
        ``sd² = err² + pct_err² + drift²``, and ``dof`` (degrees of freedom of the t, integer).
    """
    fan = round_cols(sim.fan(), pct=('mean', 'sd', 'lo', 'hi', 'err', 'pct_err', 'drift')).rename(columns={'party': 'name'})
    fan['dof'] = fan['dof'].astype(int)
    fan = fan[FAN_COLUMNS]
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


def export_weights(fc, names, date):
    """Export the polls that weigh on the average of each party on ``date`` and the local line of that day.

    Parameters
    ----------
    fc : Forecaster
        Fitted forecaster (``sim.model``), with ``weights_at``.
    names : list of str
        Party names to export; a party ``weights_at`` cannot fit (fewer than three usable polls, or no poll
        before ``date``) is left out of ``parties``.
    date : str or Timestamp
        Day on which the average is read: ``sim.as_of``, the day of the published vote.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``weights`` schema (``date``, ``parties``, ``rows`` in long format with
        ``WEIGHTS_COLUMNS``, ``line`` and ``kernel`` per party over the window and ``stats`` per party) and the
        rows as its CSV twin.
    """
    def pct(values):
        return [None if pd.isnull(v) else round(float(v), PCT_DECIMALS) for v in values]

    def isodates(index):
        return [d.strftime('%Y-%m-%d') for d in index]

    rows, line, kernel, stats = [], {}, {}, {}
    for name in names:
        out = fc.weights_at(name, date=date)
        if out is None:
            continue
        polls = out['polls'].reset_index()
        polls.insert(0, 'name', name)
        polls['date'] = isodates(polls['date'])
        polls['sample_size'] = pd.array(polls['sample_size'], dtype='Int64')
        polls = round_cols(polls, pct=('pct', 'pct_adj'),
                           prob=('weight', 'weight_kernel', 'weight_over', 'weight_sample', 'weight_rating'))
        rows.extend(polls[WEIGHTS_COLUMNS].to_dict(orient='records'))
        line[name] = {'dates': isodates(out['line'].index), 'mean': pct(out['line']['mean']),
                      'lo': pct(out['line']['cmin']), 'hi': pct(out['line']['cmax'])}
        kernel[name] = {'dates': isodates(out['kernel'].index),
                        'values': [round(float(v), PROB_DECIMALS) for v in out['kernel']]}
        st = out['stats']
        stats[name] = {'date': pd.Timestamp(st['date']).strftime('%Y-%m-%d'), 'mean': round(float(st['mean']), PCT_DECIMALS),
                       'lo': round(float(st['cmin']), PCT_DECIMALS), 'hi': round(float(st['cmax']), PCT_DECIMALS),
                       'err': round(float(st['err']), PCT_DECIMALS), 'nobs': int(st['nobs']),
                       'neff': round(float(st['neff']), 1), 'bandwidth': round(float(st['bandwidth']), 1)}
    data = {'date': pd.Timestamp(date).strftime('%Y-%m-%d'), 'parties': list(line), 'rows': rows, 'line': line,
            'kernel': kernel, 'stats': stats}
    return data, pd.DataFrame(rows, columns=WEIGHTS_COLUMNS)


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


def _meta_parties(sim):
    """Return the catalogue rows of the simulated parties, in simulation order."""
    catalogue = sim.parties.set_index('name') if 'name' in sim.parties.columns else sim.parties
    return [{'name': name, 'id': int(catalogue.loc[name, 'id']), 'fullname': catalogue.loc[name, 'fullname'],
             'color': catalogue.loc[name, 'color'], 'block': catalogue.loc[name, 'block'],
             'regional': int(sim.forecast.loc[name, 'regional'])} for name in sim.names]


def export_meta(sim, run_id: str, run_at: str, n_sim: int, max_fc: int, correctors: dict, seconds: dict,
                clip_rate: dict, db_polls: Optional[int], db_last_poll: Optional[str], prov: dict,
                freeze: bool, limit_date: Optional[str] = None) -> dict:
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
    limit_date : str, optional
        Poll cutoff date (``YYYY-MM-DD``) of a retrospective run, ``None`` for a normal one.

    Returns
    -------
    dict
        Data for the ``meta`` schema.
    """
    fc = sim.model
    n_seats = int(sim.n_seats)
    polls, pollsters = n_polls(fc)
    drift = sim.v2drift
    web = _web_bmap(sim)
    bmaps = {**fc.bmaps, 'blocks': web} if web is not None else fc.bmaps
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
        'parties': _meta_parties(sim), 'bmaps': bmaps, 'smap': sim.smap, 'regions': regions,
        'diagnostics': {'drift_k': float(drift.k) if drift is not None else None,
                        'multiplier': float(drift.multiplier) if drift is not None else None,
                        'ages': sim.ages.round(PCT_DECIMALS).to_dict(), 'composition': float(sim.composition_ratio),
                        'clip_rate': clip_rate,
                        # the elections whose polls fit the polling-error estimator, and the scope they belong to
                        'error_events': sim.error_events(), 'error_scope': sim.est_scope},
        'seconds': seconds, 'freeze': bool(freeze), 'limit_date': limit_date,
    }


def export_headline(sim, run_id: str, run_at: str, nowcast: dict, forecast: dict, backfill: bool = False) -> dict:
    """Build the ``headline`` document: the run summary shown first on the site.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.
    run_id, run_at : str
        Identifier and ISO timestamp of the run.
    nowcast, forecast : dict
        Results of ``headline_mode`` for each mode.
    backfill : bool, default False
        Whether the run is retrospective (produced later with the polls known on its day).

    Returns
    -------
    dict
        Data for the ``headline`` schema.
    """
    return {'run_id': run_id, 'run_at': run_at, 'event_date': sim.event_date,
            'as_of': sim.as_of.strftime('%Y-%m-%d'), 'date_last': sim.model.date_last.strftime('%Y-%m-%d'),
            'n_polls': n_polls(sim.model)[0], 'backfill': bool(backfill), 'nowcast': nowcast, 'forecast': forecast}


def backfill_days(spec: dict, today: Optional[str | date | datetime] = None) -> list:
    """Expand a retrospective publication range into its days.

    Parameters
    ----------
    spec : dict
        ``{'from': 'YYYY-MM-DD', 'to': 'YYYY-MM-DD'}``; ``to`` is optional and defaults to ``from``.
    today : str, datetime.date or datetime.datetime, optional
        Current date, as in ``loreg_guard``; defaults to today in Europe/Madrid.

    Returns
    -------
    list of str
        ISO days from ``from`` to ``to``, both included, ascending.

    Raises
    ------
    ValueError
        ``spec`` is not a dict with ``from``, a bound is not an ISO date, ``from`` is after ``to``
        or ``to`` is not before today (today's forecast is the normal publish). The message starts
        with ``'publish: backfill'``.
    """
    if not isinstance(spec, dict) or 'from' not in spec:
        raise ValueError('publish: backfill must be {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}')
    raw = {'from': spec['from'], 'to': spec['from'] if spec.get('to') is None else spec['to']}
    bounds = []
    for key, value in raw.items():
        try:
            if not (isinstance(value, str) and ISO_DAY_RE.match(value)):
                raise ValueError(value)
            bounds.append(date.fromisoformat(value))
        except ValueError:
            raise ValueError('publish: backfill "{}" is not a YYYY-MM-DD date: {!r}'.format(key, value)) from None
    start, end = bounds
    today = _today(today)
    if start > end:
        raise ValueError('publish: backfill "from" {} is after "to" {}'.format(start, end))
    if end >= today:
        raise ValueError('publish: backfill "to" must be before today ("to" {}, today {})'.format(end, today))
    return [(start + timedelta(days=n)).isoformat() for n in range((end - start).days + 1)]


def backfill_run_id(day: str) -> str:
    """Return the run id of a retrospective run: noon of its day.

    Parameters
    ----------
    day : str
        ``YYYY-MM-DD`` day.

    Returns
    -------
    str
        ``YYYYMMDD-120000``.
    """
    return day.replace('-', '') + '-120000'


def backfill_run_at(day: str) -> str:
    """Return the ``run_at`` of a retrospective run: noon UTC of its day.

    Parameters
    ----------
    day : str
        ``YYYY-MM-DD`` day.

    Returns
    -------
    str
        ``YYYY-MM-DDT12:00:00Z``.
    """
    return day + 'T12:00:00Z'


MODE_EXPORTERS = (('vote', export_vote), ('summary', export_summary), ('dist', export_dist),
                  ('districts', export_districts), ('scenario', export_scenario), ('projection', export_projection))


def publish_mode(sim, writer: BundleWriter, scope: str, run_id: str, communities: Optional[dict] = None) -> dict:
    """Write the files of the current mode of the simulator (JSON and CSV per part).

    Parameters
    ----------
    sim : Simulator
        Simulator already run in ``sim.mode``.
    writer : BundleWriter
        Bundle destination.
    scope, run_id : str
        Scope and run being written.
    communities : dict, optional
        Community of each district, for ``export_districts``.

    Returns
    -------
    dict
        ``headline_mode`` of the simulator.
    """
    mode = sim.mode
    for part, exporter in MODE_EXPORTERS:
        data, frame = exporter(sim, communities=communities) if exporter is export_districts else exporter(sim)
        writer.write_json(bundle.path_part(scope, run_id, part, mode), part, data, scope, run_id, mode)
        writer.write_csv(bundle.path_csv(scope, run_id, part, mode), frame)
    return headline_mode(sim)


def district_communities(scope: str) -> dict:
    """Community of each district of a scope, or nothing when the catalogue is not available.

    Parameters
    ----------
    scope : str
        Scope code.

    Returns
    -------
    dict
        ``data.get_district_communities(scope)``; empty only when the app has no ``data`` filesystem
        (the fake apps of the unit tests), so the districts carry ``community: None``. A missing or
        malformed catalogue raises.
    """
    if getattr(App.get_(), 'data', None) is None:
        return {}

    return get_district_communities(scope)


def publish_forecast(scope: str, writer: BundleWriter, run_id: str, event_date: str, n_sim: int = 1000,
                     seed: int = 42, drange=6, max_fc: int = 10, alpha: float = 0.05,
                     correctors: Optional[dict] = None, freeze: bool = False, verbose: int = 0,
                     simulator: Optional[Callable] = None, stats: Optional[Callable] = None,
                     prov: Optional[Callable] = None, limit_date: Optional[str] = None,
                     run_at: Optional[str] = None,
                     progress: Optional[Callable[[str, float], None]] = None,
                     today: Optional[str] = None) -> dict:
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
    limit_date : str, optional
        Poll cutoff date (``YYYY-MM-DD``) of a retrospective run: only the polls published up to that
        day, inclusive, enter the average, because the lower bound of ``drange`` (days before the
        event) is raised to the days from ``limit_date`` to ``event_date``, as the backtest does; the
        simulator also gets ``limit_date`` to read ``as_of`` that day. The run is marked ``backfill``
        and ``meta.limit_date`` records it (``meta.drange`` holds the effective window). ``None``
        (default) runs with ``drange`` as given.
    run_at : str, optional
        ISO UTC timestamp recorded as the run moment; defaults to the writer clock.
    today : str, optional
        The day taken as today (``YYYY-MM-DD``), given to the simulator as the cap of its anchor: the average
        is read on the day of the run, never in the future (2026-10-26); the date in Europe/Madrid by default.
    progress : callable, optional
        Called as ``progress(step, seconds)`` when each phase ends, with ``step`` in order ``init``
        (simulator built), ``fit`` (average fitted), ``nowcast`` (simulation of today),
        ``export-nowcast`` (its parts written), ``forecast`` (simulation of the election), ``export``
        (its parts and the run parts written) and ``meta`` (``meta`` and ``headline`` written), and the
        seconds of the phase. ``None`` (default) reports nothing. The ``export`` of ``seconds`` is still
        the sum of both exports.

    Returns
    -------
    dict
        ``{'scope', 'entry', 'seconds'}``.

    Raises
    ------
    FileExistsError
        The run already exists.
    NothingToPublish
        Building the simulator or fitting the average raised ``ValueError`` (no polls, or an average
        that cannot be fitted). Any later error, ``ValueError`` included, propagates as is.
    """
    writer.begin_run(scope, run_id)
    run_at = run_at or bundle.iso_utc(writer.now())
    cutoff = {}
    if limit_date is not None:
        # The polls are cut by `drange` (days before the event), not by `limit_date`, which only anchors
        # `as_of`: `days >= cut` keeps the polls published up to `limit_date`, inclusive
        lo, hi = norm_range(drange)
        cut = (date.fromisoformat(event_date) - date.fromisoformat(limit_date)).days
        drange = (max(int(lo or 0), cut), hi)
        cutoff = {'limit_date': limit_date}
    correctors = {**DEFAULT_CORRECTORS, **(correctors or {})}
    heads, clip = {}, {}
    start = mark = time.perf_counter()

    def lap():
        """Seconds since the previous lap (or the start); restart the lap."""
        nonlocal mark
        now = time.perf_counter()
        elapsed, mark = now - mark, now
        return elapsed

    def report(step, seconds):
        """Report the end of a phase through ``progress``, if any; return its seconds."""
        if progress is not None:
            progress(step, seconds)
        return seconds

    try:
        sim = (simulator or Simulator)(scope=scope, event_date=event_date, drange=drange, alpha=alpha, seed=seed,
                                       mode='nowcast', verbose=verbose, path='.', today=today, **cutoff, **correctors)
    except ValueError as e:
        raise NothingToPublish(str(e)) from e
    init = report('init', lap())
    try:
        names = sim.params['names']
        sim.fit_forecast(names=names, max_fc=max_fc, fillna=True)
    except ValueError as e:
        raise NothingToPublish(str(e)) from e
    fit = report('fit', lap())
    sim.run(split=True, random=True, n_sim=n_sim)
    clip['nowcast'] = float(sim.clip_rate())
    nowcast = report('nowcast', lap())
    communities = district_communities(scope)
    heads['nowcast'] = publish_mode(sim, writer, scope, run_id, communities)
    export = report('export-nowcast', lap())
    sim.mode = 'forecast'
    sim.run(split=True, random=True, n_sim=n_sim, horizon='deadline')
    clip['forecast'] = float(sim.clip_rate())
    forecast = report('forecast', lap())
    heads['forecast'] = publish_mode(sim, writer, scope, run_id, communities)
    fc = sim.model
    # the published series ends at the anchor, where the projection starts (the fit may reach further)
    cycle = (('series', export_series(fc, names, end=sim.as_of)), ('polls', export_polls(fc, names, seats=poll_seats(fc))),
             ('fan', export_fan(sim)),
             ('house-effects', export_house_effects(fc)), ('dispersion', export_dispersion(fc)),
             ('weights', export_weights(fc, names, sim.as_of)))
    for part, (data, frame) in cycle:
        writer.write_json(bundle.path_part(scope, run_id, part), part, data, scope, run_id)
        writer.write_csv(bundle.path_csv(scope, run_id, part), frame)
    db_polls, db_last_poll = (stats or db_stats)(scope, event_date)
    prov = prov or (lambda: provenance(cwd=REPO_ROOT))
    export += report('export', lap())
    seconds = {'init': init, 'fit': fit, 'nowcast': nowcast, 'forecast': forecast, 'export': export,
               'total': time.perf_counter() - start}
    seconds = {step: round(value, 1) for step, value in seconds.items()}
    meta = export_meta(sim, run_id, run_at, n_sim, max_fc, correctors, seconds, clip, db_polls, db_last_poll,
                       prov(), freeze, limit_date=limit_date)
    writer.write_json(bundle.path_part(scope, run_id, 'meta'), 'meta', meta, scope, run_id)
    headline = export_headline(sim, run_id, run_at, heads['nowcast'], heads['forecast'],
                               backfill=limit_date is not None)
    writer.write_json(bundle.path_part(scope, run_id, 'headline'), 'headline', headline, scope, run_id)
    report('meta', lap())
    return {'scope': scope, 'entry': latest_entry(headline), 'seconds': seconds}
