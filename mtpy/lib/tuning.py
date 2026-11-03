"""
Tuning of the polling average on future polls.

The average is tuned by how well a cut of the model predicts the polls published in the days after it: the
average read at the date of each future poll, plus the house effect of its pollster, against the figure the
poll published. This module holds the pure functions of that harness and its orchestration: the pure functions
read no database; `future_polls`, `run_cut`, `search_cycles`, `poll_dates` and the orchestration per scope
(`run_tuning`, `run_search`, `tuning_scope`) read it (read-only). The tables of a scope are written like those
of the backtest, under `TUNING_PREFIX` instead of `backtest/results`.
"""
import copy
import functools
import itertools
import json
import multiprocessing
import os
import re
import signal
import time
import warnings

from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from psycopg2 import InterfaceError as DriverInterfaceError, OperationalError as DriverOperationalError
from sqlalchemy.exc import InterfaceError as EngineInterfaceError, OperationalError as EngineOperationalError

from .backtest import describe_source, matched_overrides, poll_stats, results_path
from .computer import Computer
from .data import get_event_dates, get_event_params, get_next_event_date, get_poll_series
from .forecaster import Forecaster
from .mtypes import DROP_MTYPES
from .simulator import Simulator
from .utils import group_results

PREDICTION_COLUMNS = ['date', 'pollster_id', 'pollster', 'sample_size', 'party', 'published', 'predicted', 'residual']
# Columns of a future poll that identify and describe it, besides its published figures (see `published_blocks`)
POLL_COLUMNS = ['date', 'pollster_id', 'sponsor_id', 'pollster', 'sample_size']

# Search space of the average (spec A5 after the review), fixed before the first run: `bw` is the bandwidth as
# (`bw_type`, `bw` for a fixed one or the iterations of an adaptive one), `bw_cap` the cap of the adaptive bandwidth
# in pilots, `poly_deg` the degree of the local fit, `max_fc` the prolongation, `wspan` the over-publication window,
# `tau` and `he_n_iter` the prior deviation and backfitting iterations of the house effects and `herd_prior` the
# shrinkage of the herding effect
GRID = {
    'bw': [('fixed', 'isj'), ('fixed', 30), ('fixed', 45), ('fixed', 60), ('fixed', 90), ('adaptive', 1),
           ('adaptive', 2), ('adaptive', 3)],
    'bw_cap': [None, 1.0, 1.5, 2.0],
    'poly_deg': [0, 1],
    'max_fc': [0, 3, 10],
    'wspan': [2, 4, 7, 14],
    'tau': [0.04, 0.08, 0.16],
    'he_n_iter': [1, 3],
    'herd_prior': [2., 5., 10.]
}

# Order of the keys in each phase of the search: the smoothing first, then the shrinkages
ORDER_SMOOTHING = ['bw', 'bw_cap', 'poly_deg', 'max_fc', 'wspan']
ORDER_SHRINKAGE = ['tau', 'he_n_iter', 'herd_prior']

# Live default of each key of the grid, what the model uses when the configuration leaves it implicit:
# `Forecaster.set_reg_params` (adaptive bandwidth with 3 iterations, no cap, degree 1), `fit_forecast` (`max_fc`
# 10), `Computer` (`wspan` 4), `set_he_params` (`tau` 0.08, `n_iter` 3) and `set_disp_params` (`herd_prior` =
# `n_prior` = 5)
DEFAULTS = {
    'bw': ('adaptive', 3), 'bw_cap': None, 'poly_deg': 1, 'max_fc': 10, 'wspan': 4, 'tau': 0.08, 'he_n_iter': 3,
    'herd_prior': 5.
}

TRIAL_COLUMNS = ['round', 'key', 'value', 'label', 'rmse7', 'rmse28', 'roughness', 'chosen']

# Folder of the tables of the tuning in the file system of the app (local `files/` or S3) when no local folder is
# given: `es` at its root, the other scopes in a subfolder each (see `tuning_fs`, the twin of `backtest.results_fs`)
TUNING_PREFIX = 'backtest/tuning'

# Repository copy of the tables of the tuning (`backtest/tuning`)
TUNING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backtest', 'tuning'))

# Files of the tuning of a scope in its folder (the search modes also write `best.json`, `trials-{phase}.csv` and
# `scores-{phase}.csv`)
TUNING_FILES = ('cuts.csv', 'polls.csv', 'scores.csv', 'trials.csv', 'run.json')

# Name of every file of the tuning in the folder of a scope: `TUNING_FILES`, `best.json`, the tables of a phase
# (`trials-smoothing.csv`) and those of a part of the shrinkage phase (`trials-shrinkage-2019-11-10.csv`,
# `run-shrinkage-2019-11-10.json`); what `copy_tuning` copies
TUNING_NAME = re.compile(r'^(cuts|polls|scores|trials|run|best)(-[A-Za-z0-9_.-]+)?\.(csv|json)$')

# Tables a part of the shrinkage phase writes, as `{table}-shrinkage-{part}.csv`, and those `merge` joins
PART_TABLES = ('cuts', 'scores', 'trials')

# Phases of the search (spec A6): the keys tried and in which order, the rounds over them, the days between two
# cuts, whether the house effects, the effective sample and the herding are on, and the partition of the cycles
# it is scored on
PHASES = {
    'smoothing': {'order': ORDER_SMOOTHING, 'rounds': 2, 'cut_days': 7, 'effects': False, 'partition': 'search'},
    'shrinkage': {'order': ORDER_SHRINKAGE, 'rounds': 1, 'cut_days': 14, 'effects': True, 'partition': 'search'},
    'check': {'order': [], 'rounds': 0, 'cut_days': 7, 'effects': True, 'partition': 'check'}
}

# Cycles held out of the search to check its result (spec A7): the elections from this date on (2023 and the
# running cycle of the generales, the autonómicas since 2020); the shrinkage phase only searches from `SHRINKAGE_SINCE`
CHECK_SINCE = '2020-01-01'
SHRINKAGE_SINCE = '2015-01-01'

# Database errors a cut is retried on (a dropped or refused connection, a cancelled statement): the driver's, which
# `mtpy.core.data` re-raises from the failed queries, and the engine's, raised when connecting. Programming errors and
# the model's own errors are not retried
CONNECTION_ERRORS = (DriverOperationalError, DriverInterfaceError, EngineOperationalError, EngineInterfaceError)
# Retries of a cut after a connection error, and the seconds waited before each
CUT_RETRIES = 3
RETRY_SECONDS = 10.
# Seconds a cut may run in a process of the pool before its watchdog stops it (`timed_call`): a cut takes 2-3 minutes
# at most, so one past this hangs on a dead database link
CUT_TIMEOUT = 15 * 60.

CUT_COLUMNS = ['config', 'partition', 'event_date', 'cut', 'n_polls', 'n_overrides', 'n_unpredicted', 'seconds',
               'autocorr', 'roughness', 'error']
SCORE_COLUMNS = ['config', 'partition', 'event_date', 'k', 'n_cuts', 'n_polls', 'n_cells', 'rmse', 'mae', 'skill',
                 'autocorr', 'roughness']


def predict_polls(fc: Forecaster, polls: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """
    Predict the future polls of a cut with its fitted average and house effects.

    The average of a party is read at the date of the poll (`fit_forecast` leaves the straight prolongation up
    to `date_fit_last`, which already is the last poll plus `max_fc`; after that day the last value is held) and
    the house effect of the pollster is added.

    Parameters
    ----------
    fc : Forecaster
        Forecaster with the average fitted (`fit_forecast`) on the polls of the cut.
    polls : pd.DataFrame
        Polls published after the cut: `date`, `pollster_id`, `sponsor_id`, `pollster`, `sample_size` and one
        column per party with the published figure.
    names : list of str
        Parties or blocks to predict.

    Returns
    -------
    pd.DataFrame
        One row per poll and party with a published figure: `date`, `pollster_id`, `pollster`, `sample_size`,
        `party`, `published`, `predicted` and `residual` (published minus predicted).
    """
    # The dates are capped `max_fc` days past `date_fit_last` (2 * `max_fc` past the last poll), beyond the end of the
    # prolongation: the forward-filled average holds there the value of `date_fit_last`, so the cap changes nothing
    last = fc.date_fit_last + pd.Timedelta(days=fc.max_fc)
    average = {name: fc.forecast[name].ffill() for name in names}

    rows = []
    for poll in polls.to_dict('records'):
        date = min(pd.Timestamp(poll['date']), last)
        for name in names:
            published = poll[name]
            if pd.isnull(published):
                continue

            predicted = float(average[name].asof(date)) + fc.house_effect_of(poll['pollster_id'], name)
            rows.append({
                'date': poll['date'], 'pollster_id': poll['pollster_id'], 'pollster': poll['pollster'],
                'sample_size': poll['sample_size'], 'party': name, 'published': float(published),
                'predicted': predicted, 'residual': float(published) - predicted
            })

    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def score_polls(pred: pd.DataFrame, k: int) -> dict:
    """
    Score the predictions of the future polls of a horizon.

    The skill compares the mean squared error with the floor of pure sampling noise, the mean over the cells
    of `p (1 - p) / n` in points squared: 1 is a perfect prediction and 0 is as bad as the noise of the polls
    themselves. A missing sample size takes the median of the present ones; without any, the skill is NaN.

    Parameters
    ----------
    pred : pd.DataFrame
        Output of `predict_polls` already filtered to the horizon `k`.
    k : int
        Horizon in days, echoed in the result.

    Returns
    -------
    dict
        `k`, `n_polls` (distinct polls), `n_cells`, `rmse` and `mae` (equal weight per cell, in points),
        `skill` and `rmse_by_party` (party to RMSE). Everything is NaN with no cells.
    """
    n_cells = len(pred)
    if n_cells == 0:
        return {'k': k, 'n_polls': 0, 'n_cells': 0, 'rmse': np.nan, 'mae': np.nan, 'skill': np.nan,
                'rmse_by_party': {}}

    residual = pred['residual'].astype(float)
    mse = float((residual ** 2).mean())

    sample = pred['sample_size'].astype(float)
    if sample.notna().any():
        sample = sample.fillna(sample.median())
        p = pred['published'].astype(float) / 100
        floor = float((p * (1 - p) / sample).mean()) * 100 ** 2
    else:
        floor = np.nan
    skill = 1 - mse / floor if floor > 0 else np.nan

    keys = [column for column in ['date', 'pollster_id', 'sponsor_id'] if column in pred.columns]
    return {
        'k': k,
        'n_polls': len(pred[keys].drop_duplicates()),
        'n_cells': n_cells,
        'rmse': float(np.sqrt(mse)),
        'mae': float(residual.abs().mean()),
        'skill': skill,
        'rmse_by_party': {
            party: float(np.sqrt((group ** 2).mean())) for party, group in residual.groupby(pred['party'])
        }
    }


def autocorrelation(pred: pd.DataFrame) -> float:
    """
    Lag-1 autocorrelation of the residuals, ordered by date within each party and pooled over parties.

    A positive value means the residuals persist from one poll to the next: the average reacts too slowly.

    Parameters
    ----------
    pred : pd.DataFrame
        Output of `predict_polls`: `party`, `date` and `residual`.

    Returns
    -------
    float
        Pearson correlation between each residual and the next one of the same party; NaN with fewer than
        3 such pairs.
    """
    current, following = [], []
    for _, group in pred.sort_values('date', kind='stable').groupby('party'):
        values = group['residual'].astype(float).to_numpy()
        current.extend(values[:-1])
        following.extend(values[1:])

    if len(current) < 3 or np.std(current) == 0 or np.std(following) == 0:
        return np.nan

    return float(np.corrcoef(current, following)[0, 1])


def roughness(forecast: pd.DataFrame, names: list[str]) -> float:
    """
    Mean absolute day-to-day change of the fitted series, in points per day.

    Only consecutive days with a fitted value count; NaN gaps are not bridged.

    Parameters
    ----------
    forecast : pd.DataFrame
        Daily series, one column per party or block.
    names : list of str
        Columns to average over.

    Returns
    -------
    float
        Mean over `names` of the mean absolute daily change; 0 for flat series, NaN without any pair of
        consecutive fitted days.
    """
    changes = [forecast[name].astype(float).diff().abs().mean() for name in names]
    changes = [change for change in changes if not np.isnan(change)]

    return float(np.mean(changes)) if changes else np.nan


def default_config() -> dict[str, Any]:
    """
    Configuration of the live average: the defaults of the model.

    Returns
    -------
    dict
        `reg_params`, `he_params` and `disp_params` empty (each `set_*_params` keeps its defaults),
        `house_effects`, `dispersion` and `herding` on, `weights` None (the weights stored with the polls),
        `max_fc` 10 and `label` `'default'`. A new dict on every call.
    """
    return {
        'reg_params': {}, 'house_effects': True, 'he_params': {}, 'dispersion': True, 'herding': True,
        'disp_params': {}, 'weights': None, 'max_fc': 10, 'label': 'default'
    }


def cycle_cuts(
    event_date: str,
    poll_dates: pd.Series,
    cut_days: int = 7,
    window: int = 365,
    min_polls: int = 5,
    lookback: int = 90,
    today: Optional[str] = None
) -> list[pd.Timestamp]:
    """
    Cut dates of a cycle: one every `cut_days` from `window` days before the election.

    Parameters
    ----------
    event_date : str
        Date of the election.
    poll_dates : pd.Series
        Dates of the useful polls of the cycle (one entry per poll).
    cut_days : int, optional
        Days between two cuts.
    window : int, optional
        Days before the election of the first cut.
    min_polls : int, optional
        Polls a cut needs in its `lookback` days (`(cut - lookback, cut]`) to be kept.
    lookback : int, optional
        Days before a cut where its polls are counted.
    today : str, optional
        Last possible cut of a running cycle; the current date by default (no cut is ever in the future).

    Returns
    -------
    list of pd.Timestamp
        The cuts from `event_date - window` up to `min(event_date - 1, today)`, every `cut_days`, without those
        with fewer than `min_polls` polls in their lookback.
    """
    event = pd.Timestamp(event_date)
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    last = min(event - pd.Timedelta(days=1), today)
    cuts = pd.date_range(event - pd.Timedelta(days=window), last, freq='{}D'.format(int(cut_days)))
    if len(cuts) == 0:
        return []

    dates = np.sort(pd.to_datetime(pd.Series(poll_dates)).dropna().to_numpy())
    upper = np.searchsorted(dates, cuts.to_numpy(), side='right')
    lower = np.searchsorted(dates, (cuts - pd.Timedelta(days=lookback)).to_numpy(), side='right')

    return [cut for cut, count in zip(cuts, upper - lower) if count >= min_polls]


def future_polls(
    scope: str,
    event_date: str,
    cut: pd.Timestamp,
    k: int,
    drop_mtypes: list[str]
) -> pd.DataFrame:
    """
    The polls of a cycle published in the `k` days after a cut that a cut has to predict (spec A3).

    The targets are the polls that enter the model (`computed`, methodology type out of `drop_mtypes`) with a
    stored `weight_over` above 0, i.e. without the overlapped waves of the trackings: a set that does not depend
    on the configuration being tuned.

    Parameters
    ----------
    scope : str
        Scope of the election.
    event_date : str
        Date of the election.
    cut : pd.Timestamp
        Date of the cut.
    k : int
        Horizon in days: the polls dated in `(cut, cut + k]`.
    drop_mtypes : list of str
        Methodology types left out of the model.

    Returns
    -------
    pd.DataFrame
        The columns of `get_poll_series` (published figure per party), one row per poll, with a new index; empty
        when no poll falls in the window.
    """
    cut = pd.Timestamp(cut)
    days_to_event = (pd.Timestamp(event_date) - cut).days
    # `days` counts from the poll to the election: the window is `days` in [days_to_event - k, days_to_event - 1].
    # A bound of 0 is read as no bound by `get_poll_series`, so the dates are filtered again below
    polls = get_poll_series(scope, [event_date], drange=(max(days_to_event - k, 0), days_to_event - 1),
                            drop_mtypes=drop_mtypes)
    if polls.empty:
        return polls.reset_index(drop=True)

    dates = pd.to_datetime(polls['date'])
    useful = (
        (dates > cut) & (dates <= cut + pd.Timedelta(days=k))
        & polls['computed'].fillna(False).astype(bool) & (polls['weight_over'] > 0)
        & ~polls['mtype'].isin(drop_mtypes or [])
    )

    return polls.loc[useful].reset_index(drop=True)


def published_blocks(polls: pd.DataFrame, bmap: dict[str, list[str]], names: list[str]) -> pd.DataFrame:
    """
    Published figures of some polls grouped like the series of the average.

    Parameters
    ----------
    polls : pd.DataFrame
        Polls with the `POLL_COLUMNS` and one column per party.
    bmap : dict
        Blocks of the average: each party with the parties it inherits (`Simulator.poll_blocks`).
    names : list of str
        Blocks to keep.

    Returns
    -------
    pd.DataFrame
        The `POLL_COLUMNS` and one column per block of `names` (the sum of the published figures of its
        parties, NaN when the poll lists none).
    """
    if polls.empty:
        return pd.DataFrame(columns=POLL_COLUMNS + list(names))

    grouped = group_results(polls, bmap=bmap)

    return pd.concat([polls[POLL_COLUMNS], grouped[list(names)]], axis=1)


def reweighted_overrides(polls: pd.DataFrame, weights: dict[str, Any]) -> pd.DataFrame:
    """
    Weights of the polls of a cut recomputed in memory with `Computer.reweight`, as `weight_overrides`.

    Both stored weights, `weight_over` and `weight_sample`, are replaced: computed over the polls up to the cut
    only, they carry no look-ahead (the stored ones depend on later waves and on full-cycle sample medians).
    `proc_sample` is not overridable and stays the stored one.

    Parameters
    ----------
    polls : pd.DataFrame
        Polls of the cut as `get_poll_series` returns them.
    weights : dict
        `wspan` and, optionally, `ol_max`.

    Returns
    -------
    pd.DataFrame
        The recomputed columns indexed by (`date`, `pollster_id`, `sponsor_id`), with `date` as a timestamp.
    """
    columns = ['weight_over', 'weight_sample']
    computed = Computer.reweight(polls, weights['wspan'], ol_max=weights.get('ol_max', 5000))
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(polls['date']), polls['pollster_id'], polls['sponsor_id']],
        names=['date', 'pollster_id', 'sponsor_id']
    )

    return computed.loc[polls.index, columns].set_axis(index, axis=0)


def run_cut(
    scope: str,
    event_date: str,
    cut: pd.Timestamp,
    config: dict[str, Any],
    horizons: tuple[int, ...] = (7, 28),
    max_fc: Optional[int] = None,
    verbose: int = 0,
    event_params: Optional[dict] = None
) -> dict[str, Any]:
    """
    Freeze the average at a cut and score how it predicts the polls published after it.

    The Forecaster is frozen like a case of the backtest (`drange` = days from the cut to the election, the
    blocks of the `Simulator`: each party with the parties it inherits) and fitted on the `main` parties of the
    election. The future polls are read once, for the largest horizon, grouped into the same blocks and
    predicted (`predict_polls`); each prediction is labelled with the smallest horizon that reaches it, and the
    score of a horizon `k` covers every poll in `(cut, cut + k]`.

    Parameters
    ----------
    scope : str
        Scope of the election.
    event_date : str
        Date of the election.
    cut : pd.Timestamp
        Date of the cut.
    config : dict
        Configuration of the average (see `default_config`): `reg_params`, `he_params` and `disp_params` go to
        the Forecaster, `house_effects` and `dispersion` are its switches, `herding` goes into `disp_params`,
        `max_fc` to `fit_forecast` and `weights` (`wspan`, optionally `ol_max`), when set, recomputes the
        `weight_over` and `weight_sample` of the polls of the cut on those polls alone (`reweighted_overrides`;
        `proc_sample` stays the stored one).
    horizons : tuple of int, optional
        Horizons in days.
    max_fc : int, optional
        Overrides `config['max_fc']`.
    verbose : int, optional
        Level of verbosity of the Forecaster.
    event_params : dict, optional
        The params of the election (`get_event_params(scope, event_date, path='.')`), read once per cycle by
        `run_tuning` and passed to the Forecaster; None lets the Forecaster derive them.

    Returns
    -------
    dict
        `meta` (`scope`, `event_date`, `cut`, `drange`, `n_polls` in the series of the average, `n_overrides`
        polls of the series with recomputed weights, `n_unpredicted` cells without an average, `label`, `max_fc`,
        `names`), `polls` (the predictions with `days_ahead` and `k`), `scores` (`score_polls` of each horizon,
        in the order of `horizons`), `autocorr` (`autocorrelation` of all the predictions), `roughness` (of the
        fitted series) and `seconds`.
    """
    t0 = time.time()
    cut = pd.Timestamp(cut)
    drange = (pd.Timestamp(event_date) - cut).days
    max_fc = config['max_fc'] if max_fc is None else max_fc
    horizons = [int(k) for k in horizons]
    drop_mtypes = list(DROP_MTYPES)

    disp_params = dict(config['disp_params']) | {'herding': bool(config['herding'])}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        overrides = None
        if config.get('weights'):
            cut_polls = get_poll_series(scope, [event_date], drange=(drange, None), drop_mtypes=drop_mtypes)
            overrides = reweighted_overrides(cut_polls, config['weights'])

        fc = Forecaster(
            scope=scope, event_date=event_date, drop_mtypes=drop_mtypes, drange=drange,
            reg_params=dict(config['reg_params']), house_effects=bool(config['house_effects']),
            he_params=dict(config['he_params']), dispersion=bool(config['dispersion']), disp_params=disp_params,
            weight_overrides=overrides, verbose=verbose, event_params=event_params
        )
        # The blocks of the Simulator (each party counts the polls of its predecessors), taken from the params the
        # Forecaster has just read: setting them before `build_series` equals passing them as `bmap`, without
        # deriving the params of the scope a second time
        params = fc.event_params
        fc.bmap = Simulator.poll_blocks(list(params['parties']['event']), params.get('smap') or {})
        names = [name for name in params['bmaps']['main'] if name in fc.bmap]
        fc.build_series()
        fc.fit_forecast(names=names, max_fc=max_fc)

    raw = fc.series_raw if fc.series_raw is not None else fc.series
    series_polls = raw.index[raw['pollster'].notnull().to_numpy()]
    n_overrides = matched_overrides(fc, overrides)

    targets = published_blocks(future_polls(scope, event_date, cut, max(horizons), drop_mtypes), fc.bmap, names)
    pred = predict_polls(fc, targets, names)
    unpredicted = pred['predicted'].isnull()
    pred = pred.loc[~unpredicted].reset_index(drop=True)

    ordered = sorted(horizons)
    pred['days_ahead'] = (pd.to_datetime(pred['date']) - cut).dt.days.astype(int)
    pred['k'] = np.array(ordered, dtype=int)[np.searchsorted(ordered, pred['days_ahead'].to_numpy(), side='left')]

    return {
        'meta': {
            'scope': scope, 'event_date': event_date, 'cut': cut, 'drange': drange, 'n_polls': len(series_polls),
            'n_overrides': n_overrides, 'n_unpredicted': int(unpredicted.sum()), 'label': config.get('label'),
            'max_fc': max_fc, 'names': names
        },
        'polls': pred,
        'scores': [score_polls(pred.loc[pred['days_ahead'] <= k], k) for k in horizons],
        'autocorr': autocorrelation(pred),
        'roughness': roughness(fc.forecast, names),
        'seconds': time.time() - t0
    }


def apply_setting(config: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """
    Copy of a configuration with one value of the grid applied where the parameter lives.

    Parameters
    ----------
    config : dict
        Configuration of the average (see `default_config`). Not modified.
    key : str
        Key of `GRID`: `bw` goes to `reg_params` as `bw_type` plus `bw` (fixed) or `bw_kwargs['n_iter']` (adaptive,
        the pilot `bw` keeps its default), replacing any previous bandwidth; `bw_cap` and `poly_deg` go to
        `reg_params`; `wspan` sets `weights` to `{'wspan': value}`; `tau` and `he_n_iter` (as `n_iter`) go to
        `he_params`; `herd_prior` to `disp_params`; `max_fc` to the configuration itself.
    value : Any
        One of the values of `GRID[key]`.

    Returns
    -------
    dict
        A deep copy of `config` with the setting applied and `label` = `'{key}={value}'`.

    Raises
    ------
    ValueError
        If `key` is not in `GRID`, `value` is not in `GRID[key]`, `bw_cap` is below 1 (other than None) or
        `wspan` is not positive.
    """
    if key not in GRID:
        raise ValueError(f'Unknown tuning parameter: {key}')
    if value not in GRID[key]:
        raise ValueError(f'Value {value!r} of {key} is not in the grid {GRID[key]}')
    if key == 'bw_cap' and value is not None and value < 1:
        raise ValueError(f'bw_cap must be None or at least 1, got {value!r}')
    if key == 'wspan' and value <= 0:
        raise ValueError(f'wspan must be positive, got {value!r}')

    out = copy.deepcopy(config)
    if key == 'bw':
        bw_type, setting = value
        reg_params = {k: v for k, v in (out.get('reg_params') or {}).items() if k not in ('bw_type', 'bw', 'bw_kwargs')}
        if bw_type == 'fixed':
            reg_params.update({'bw_type': 'fixed', 'bw': setting})
        else:
            reg_params.update({'bw_type': 'adaptive', 'bw_kwargs': {'n_iter': setting, 'min_delta': None}})
        out['reg_params'] = reg_params
    elif key in ('bw_cap', 'poly_deg'):
        out['reg_params'] = {**(out.get('reg_params') or {}), key: value}
    elif key == 'wspan':
        out['weights'] = {'wspan': value}
    elif key in ('tau', 'he_n_iter'):
        out['he_params'] = {**(out.get('he_params') or {}), ('n_iter' if key == 'he_n_iter' else key): value}
    elif key == 'herd_prior':
        out['disp_params'] = {**(out.get('disp_params') or {}), key: value}
    else:
        out[key] = value
    out['label'] = f'{key}={value}'

    return out


def is_implicit(config: dict[str, Any], key: str) -> bool:
    """
    Whether a key of the grid is left implicit in a configuration (the model then uses its live default).

    Parameters
    ----------
    config : dict
        Configuration of the average.
    key : str
        Key of `GRID`.

    Returns
    -------
    bool
        True when the configuration does not set the key: `bw` when `reg_params` has none of `bw_type`, `bw` and
        `bw_kwargs`; `wspan` when `weights` is None; `herd_prior` when `disp_params` has neither `herd_prior` nor
        `n_prior` (with `n_prior` set, the implicit `herd_prior` follows it and is not a default to fill); the
        other keys when they are absent from the dict where they live.
    """
    reg_params = config.get('reg_params') or {}
    he_params = config.get('he_params') or {}
    disp_params = config.get('disp_params') or {}
    if key == 'bw':
        return not any(name in reg_params for name in ('bw_type', 'bw', 'bw_kwargs'))
    if key in ('bw_cap', 'poly_deg'):
        return key not in reg_params
    if key == 'wspan':
        return config.get('weights') is None
    if key in ('tau', 'he_n_iter'):
        return ('n_iter' if key == 'he_n_iter' else key) not in he_params
    if key == 'herd_prior':
        return 'herd_prior' not in disp_params and 'n_prior' not in disp_params
    return key not in config


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Copy of a configuration with every key of the grid it leaves implicit set to its live default.

    The model is the same, but the configuration is written as the search writes its candidates, so a candidate
    equal to the live default is recognised as the configuration in force (see `coordinate_descent`). Normalising
    `wspan` turns `weights` None (the weights stored with the polls) into `{'wspan': 4}`, the weights recomputed
    at the cut with the live window: the baseline of the search.

    Parameters
    ----------
    config : dict
        Configuration of the average (see `default_config`). Not modified.

    Returns
    -------
    dict
        A copy with `apply_setting(config, key, DEFAULTS[key])` for every implicit key (`is_implicit`); explicit
        values and the `label` are kept as they are.
    """
    out = copy.deepcopy(config)
    for key in GRID:
        if is_implicit(out, key):
            out = apply_setting(out, key, DEFAULTS[key])
    if 'label' in config:
        out['label'] = config['label']
    else:
        out.pop('label', None)

    return out


def setting_of(config: dict[str, Any], key: str) -> Any:
    """
    Value of a key of the grid in a configuration whose keys are all explicit (`normalize_config`), as `GRID` writes it.

    Parameters
    ----------
    config : dict
        Normalised configuration of the average.
    key : str
        Key of `GRID`.

    Returns
    -------
    Any
        The value, read where `apply_setting` puts it: `bw` as `(bw_type, bw)` or `('adaptive', n_iter)`.
    """
    reg_params = config.get('reg_params') or {}
    if key == 'bw':
        if reg_params.get('bw_type') == 'fixed':
            return 'fixed', reg_params.get('bw')
        return 'adaptive', (reg_params.get('bw_kwargs') or {}).get('n_iter')
    if key in ('bw_cap', 'poly_deg'):
        return reg_params.get(key)
    if key == 'wspan':
        return (config.get('weights') or {}).get('wspan')
    if key in ('tau', 'he_n_iter'):
        return (config.get('he_params') or {}).get('n_iter' if key == 'he_n_iter' else key)
    if key == 'herd_prior':
        disp_params = config.get('disp_params') or {}
        return disp_params.get('herd_prior', disp_params.get('n_prior'))
    return config.get(key)


def best_label(config: dict[str, Any]) -> str:
    """
    Label of a configuration kept by the search: every key of the grid whose value differs from the live default.

    Parameters
    ----------
    config : dict
        Configuration of the average.

    Returns
    -------
    str
        `'key=value'` of each setting that differs from `DEFAULTS`, in the order of `GRID`, joined by `'; '` (the cap
        of the adaptive bandwidth is left out under a fixed bandwidth, where it does nothing); `'default'` when none
        differs.
    """
    config = normalize_config(effective_config(config))
    settings = []
    for key in GRID:
        value = setting_of(config, key)
        if value != DEFAULTS[key]:
            settings.append('{}={}'.format(key, value))

    return '; '.join(settings) or 'default'


def config_key(config: dict[str, Any]) -> str:
    """
    Canonical key of a configuration, for the memo of the search: the configuration without its `label`.

    Parameters
    ----------
    config : dict
        Configuration.

    Returns
    -------
    str
        JSON dump with sorted keys (values that are not JSON by their `repr`), so the order in which the settings
        were applied and the label naming the last of them do not make a new configuration.
    """
    return json.dumps({k: v for k, v in config.items() if k != 'label'}, sort_keys=True, default=repr)


def coordinate_descent(
    evaluate: Callable[[dict], dict],
    start: dict[str, Any],
    grid: dict[str, list],
    order: list[str],
    rounds: int = 2,
    tolerance: float = 0.02,
    roughness_ratio: float = 1.5,
    setter: Callable[[dict, str, Any], dict] = apply_setting,
    normalize: Optional[Callable[[dict], dict]] = None
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Coordinate descent over a grid: one key at a time, keeping the best allowed value of each.

    The start is evaluated first (the roughness reference). In each round and for each key of `order`, every value
    of the key is applied to the configuration in force and evaluated; the value kept is the one with the lowest
    `rmse7` among those with `rmse28` at most `(1 + tolerance)` times the `rmse28` of the configuration in force
    and `roughness` at most `roughness_ratio` times the roughness of the start. Only a strictly lower `rmse7` than
    the configuration in force replaces it (ties keep the current value, and so does a key with no candidate
    passing the guards); between candidates tied on `rmse7` the first in the grid wins. Every configuration is
    evaluated once: a repeated one (the configuration in force, or one reached again in a later round) takes its
    memoised figures. The memo compares configurations as written, so a start that leaves a parameter implicit
    must be normalised (`normalize`, `normalize_config` for the configurations of the average) for a candidate
    equal to the live default to be recognised as the configuration in force. No key is skipped here (for
    example `bw_cap` under a fixed bandwidth): pruning the grid is the caller's business.

    Parameters
    ----------
    evaluate : callable
        `evaluate(config)` returns a dict with `rmse7`, `rmse28` and `roughness`.
    start : dict
        Starting configuration. Not modified.
    grid : dict
        Values to try of each key.
    order : list of str
        Keys of `grid` in the order they are tried in each round.
    rounds : int, optional
        Number of rounds over `order`.
    tolerance : float, optional
        Largest relative worsening of `rmse28` allowed against the configuration in force.
    roughness_ratio : float, optional
        Largest roughness allowed, in times the roughness of the start.
    setter : callable, optional
        `setter(config, key, value)` returns a new configuration with the value applied (`apply_setting`).
    normalize : callable, optional
        `normalize(config)` applied to `start` before its evaluation (`normalize_config` for the configurations
        of the average; note it turns `weights` None into `{'wspan': 4}`). None leaves `start` as it is.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        The final configuration and the table of the trials (`round`, `key`, `value` as given in the grid,
        `label`, `rmse7`, `rmse28`, `roughness`, `chosen`): a row of round 0 for the start and one per value of
        each key in each round, with `chosen` True for the values that leave the configuration kept for that key in
        that round (the current one included when it is in the grid).

    Raises
    ------
    ValueError
        The start has a NaN `rmse7`, `rmse28` or `roughness`: no candidate could pass the guards against it.
    """
    memo: dict[str, dict] = {}

    def figures(config: dict[str, Any]) -> dict:
        """Figures of a configuration, evaluated once."""
        key = config_key(config)
        if key not in memo:
            result = evaluate(config)
            memo[key] = {name: float(result[name]) for name in ('rmse7', 'rmse28', 'roughness')}
        return memo[key]

    current = normalize(start) if normalize is not None else start
    current_figures = figures(current)
    if any(np.isnan(value) for value in current_figures.values()):
        # every guard against a NaN is False: no candidate could ever be kept, and the descent would say nothing
        raise ValueError('coordinate_descent: the start {} has no figures to compare with ({})'.format(
            current.get('label'), ', '.join('{} {}'.format(name, value) for name, value in current_figures.items())))
    roughness_ref = current_figures['roughness']
    rows = [{'round': 0, 'key': None, 'value': None, 'label': current.get('label'), **current_figures,
             'chosen': True}]

    for round_number in range(1, rounds + 1):
        for key in order:
            rmse28_limit = (1 + tolerance) * current_figures['rmse28']
            roughness_limit = roughness_ratio * roughness_ref
            candidates = []
            for value in grid[key]:
                config = setter(current, key, value)
                candidates.append((value, config, figures(config)))

            best, best_figures = current, current_figures
            for value, config, values in candidates:
                allowed = values['rmse28'] <= rmse28_limit and values['roughness'] <= roughness_limit
                if allowed and values['rmse7'] < best_figures['rmse7']:
                    best, best_figures = config, values

            kept = config_key(best)
            for value, config, values in candidates:
                rows.append({'round': round_number, 'key': key, 'value': value, 'label': config.get('label'),
                             **values, 'chosen': config_key(config) == kept})
            current, current_figures = best, best_figures

    trials = pd.DataFrame(rows, columns=TRIAL_COLUMNS)
    trials['value'] = pd.Series([row['value'] for row in rows], dtype=object)

    return current, trials


# --- Orchestration per scope ----------------------------------------------------------------------------------------

class NothingToTune(Exception):
    """A scope with no cycle to tune, or no cut with enough polls in its cycles: nothing is written."""


class TuningFailed(Exception):
    """The cuts of a scope exist but the tuning cannot be trusted: every cut failed, or a cut of the start of a search
    or of the check failed (a configuration scored on fewer cuts than another is not comparable). Nothing is
    written."""


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """
    Mean of the values weighted by the weights, leaving out the NaN values and the null weights.

    Parameters
    ----------
    values, weights : pd.Series
        Aligned values and weights.

    Returns
    -------
    float
        The weighted mean; NaN when no value has a positive weight.
    """
    values = values.astype(float)
    weights = weights.astype(float)
    keep = values.notna() & (weights > 0)
    if not keep.any():
        return np.nan

    return float((values[keep] * weights[keep]).sum() / weights[keep].sum())


def aggregate_scores(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Scores of each configuration, partition, cycle and horizon over the cuts of the cycle.

    Parameters
    ----------
    rows : pd.DataFrame
        One row per configuration, partition, cycle, cut and horizon: `config`, `partition`, `event_date`, `k`,
        `n_polls`, `n_cells`, `rmse`, `mae`, `skill`, `autocorr` and `roughness` of the cut.

    Returns
    -------
    pd.DataFrame
        `SCORE_COLUMNS`: `n_cuts`, and `n_polls` and `n_cells` summed over the cuts; `rmse` the RMSE over every
        cell of the cycle (the square root of the mean of the squared RMSE of the cuts weighted by their cells,
        so each cell weighs the same, spec A4); `mae` and `skill` the means weighted by the cells of each cut;
        `autocorr` and `roughness` the plain means over the cuts (one value per cut, whatever the horizon: the
        definition of `search_figures`).
    """
    if rows.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    out = []
    for (label, partition, event_date, k), group in rows.groupby(['config', 'partition', 'event_date', 'k'],
                                                                 sort=False, dropna=False):
        cells = group['n_cells']
        mse = weighted_mean(group['rmse'].astype(float) ** 2, cells)
        out.append({
            'config': label, 'partition': partition, 'event_date': event_date, 'k': k, 'n_cuts': len(group),
            'n_polls': int(group['n_polls'].sum()), 'n_cells': int(cells.sum()), 'rmse': float(np.sqrt(mse)),
            'mae': weighted_mean(group['mae'], cells), 'skill': weighted_mean(group['skill'], cells),
            'autocorr': float(group['autocorr'].astype(float).mean()),
            'roughness': float(group['roughness'].astype(float).mean())
        })

    return pd.DataFrame(out, columns=SCORE_COLUMNS)


def search_cycles(
    scope: str,
    since: str = '2008-01-01',
    min_polls: int = 10,
    today: Optional[str] = None
) -> dict[str, list[str]]:
    """
    Cycles of a scope to tune the average on, split into the search and the check partitions (spec A1, A7).

    Parameters
    ----------
    scope : str
        Scope code.
    since : str, optional
        First possible election.
    min_polls : int, optional
        Polls an election needs (`get_event_dates(..., min_polls=min_polls, featured=True)`).
    today : str, optional
        The current date by default: the running cycle is the first election after it.

    Returns
    -------
    dict
        `search`: the featured elections with enough polls before `CHECK_SINCE`; `check`: those from `CHECK_SINCE`
        on plus the next election of the scope after `today` (the running cycle, `get_next_event_date`; asked
        from `today` so that a held election that is not featured cannot hide it). Dates as `YYYY-MM-DD`, in
        order.
    """
    dates = list(get_event_dates(scope, date_from=since, min_polls=min_polls, featured=True))
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    following = get_next_event_date(scope, date_from=today.strftime('%Y-%m-%d'))
    if following is not None and following not in dates:
        dates.append(following)

    return {
        'search': [date for date in dates if date < CHECK_SINCE],
        'check': [date for date in dates if date >= CHECK_SINCE]
    }


def poll_dates(scope: str, event_date: str) -> pd.Series:
    """
    Dates of the polls of a cycle that count for its cuts (`cycle_cuts`), in one query.

    The polls are those of the targets (`future_polls`): `computed`, with a `weight_over` above 0 and a
    methodology type out of `DROP_MTYPES`.

    Parameters
    ----------
    scope : str
        Scope code.
    event_date : str
        Date of the election.

    Returns
    -------
    pd.Series
        One date per poll.
    """
    from ..models.elections import Polls

    filters = [
        "event_scope = '{}'".format(scope), "event_date = '{}'".format(event_date), 'computed', 'weight_over > 0',
        "(mtype IS NULL OR mtype NOT IN ('{}'))".format("', '".join(DROP_MTYPES))
    ]
    polls = Polls().get_results(query=dict(filters=filters), formatted=True)

    return pd.to_datetime(polls['date']).reset_index(drop=True)


def tuning_fs(root: Optional[str], scope: str) -> tuple:
    """
    File system and folder where the tables of the tuning of a scope live (the twin of `backtest.results_fs`).

    Parameters
    ----------
    root : str or None
        Local folder. Without it the file system of the app (`App.get_().fs`: local or S3) and the
        `TUNING_PREFIX` folder are used.
    scope : str
        Scope code.

    Returns
    -------
    tuple of (FileSystem, str)
        The file system and the folder in it: `''` (the root of the file system) for `es` with `root`, the
        scope code for the others; `TUNING_PREFIX` (plus `/{scope}` for the regional scopes) without `root`.

    Raises
    ------
    RuntimeError
        There is no `root` and the app has no file system.
    """
    from ..core.io import FileSystem

    if root:
        return FileSystem(os.path.abspath(root)), '' if scope == 'es' else scope
    from ..core.app import App

    fs = App.get_().fs
    if fs is None:
        raise RuntimeError('tuning: no file system configured (set S3_BUCKET or run with a local files/ root)')

    return fs, TUNING_PREFIX if scope == 'es' else TUNING_PREFIX + '/' + scope


def plan_cycles(
    scope: str,
    cycles: dict[str, list[str]],
    cut_days: int = 7,
    read_params: bool = True,
    verbose: int = 0
) -> list[dict[str, Any]]:
    """
    Cuts of every cycle, and the params of its election, read once for every configuration evaluated on them.

    Parameters
    ----------
    scope : str
        Scope code.
    cycles : dict
        `{'search': [...], 'check': [...]}` (either may be missing).
    cut_days : int, optional
        Days between two cuts (`cycle_cuts`, up to today in a running cycle).
    read_params : bool, optional
        Read the params of each election (`get_event_params(scope, event_date, path='.')`, as the Forecaster of
        `run_cut` does), to pass them to every cut; False with a replacement runner.
    verbose : int, optional
        1 prints the cycles without any cut.

    Returns
    -------
    list of dict
        One entry per cycle with at least one cut, search partition first: `partition`, `event_date`, `cuts`
        and `event_params` (None without `read_params`).
    """
    plan = []
    for partition in ('search', 'check'):
        for event_date in cycles.get(partition) or []:
            cuts = cycle_cuts(event_date, poll_dates(scope, event_date), cut_days=cut_days)
            if len(cuts) == 0:
                if verbose > 0:
                    print('Tuning {} {}: no cut with enough polls'.format(scope, event_date))
                continue
            params = get_event_params(scope, event_date, path='.') if read_params else None
            plan.append({'partition': partition, 'event_date': event_date, 'cuts': cuts, 'event_params': params})

    return plan


class CutTimeout(Exception):
    """A cut ran past `CUT_TIMEOUT` seconds in a process of the pool (a query hung on a dead database link). It is
    retried as a connection error, on a fresh connection (`reset_connection`)."""


def reset_connection() -> None:
    """
    Forget the database engine of this process without touching its sockets, so the next query opens a fresh one.

    The engine is disposed with `close=False` (closing a connection on a dead link could hang again) and the adapter
    drops it (`dispose`): its next `connect` builds a new engine. Nothing happens without an app or a database.
    """
    from ..core.app import App

    if not App.has_():
        return
    db = App.get_().db
    if db is None:
        return
    engine = getattr(db, '_client', None)
    if engine is not None and hasattr(engine, 'dispose'):
        engine.dispose(close=False)
    db.dispose()


def raise_cut_timeout(signum, frame):
    """Handler of SIGALRM in `timed_call`."""
    raise CutTimeout('the cut did not finish in {:g} s'.format(CUT_TIMEOUT))


def timed_call(run: Callable, *args, **kwargs) -> Any:
    """
    Run one attempt of a cut under a watchdog of `CUT_TIMEOUT` seconds (SIGALRM, so only in the main thread of a
    process: the processes of the pool run their tasks there).

    Parameters
    ----------
    run : callable
        `run_cut` or its replacement.
    *args, **kwargs
        Its arguments.

    Returns
    -------
    Any
        Its result.

    Raises
    ------
    CutTimeout
        The attempt ran past `CUT_TIMEOUT` seconds. The timer and the previous handler are restored in any case. An
        alarm that lands after `run` returned (before the timer is stopped) is only warned about: the finished cut
        keeps its result instead of being retried.
    """
    finished = False

    def watchdog(signum, frame):
        """Stop the cut, unless it has already returned (the alarm landed before the timer was stopped)."""
        if finished:
            warnings.warn('Tuning cut: the watchdog fired after the cut finished; its result is kept')
            return
        raise_cut_timeout(signum, frame)

    previous = signal.signal(signal.SIGALRM, watchdog)
    signal.setitimer(signal.ITIMER_REAL, CUT_TIMEOUT)
    try:
        result = run(*args, **kwargs)
        finished = True
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_with_retries(run: Callable, *args, **kwargs) -> dict[str, Any]:
    """
    Run one cut, again after a database connection error or a `CutTimeout` (after `reset_connection`), up to
    `CUT_RETRIES` times.

    Parameters
    ----------
    run : callable
        `run_cut` or its replacement.
    *args, **kwargs
        Its arguments.

    Returns
    -------
    dict
        Its result.

    Raises
    ------
    Exception
        The last connection error after the retries, or any other error at once.
    """
    for attempt in range(CUT_RETRIES + 1):
        try:
            return run(*args, **kwargs)
        except CONNECTION_ERRORS + (CutTimeout,) as e:
            if isinstance(e, CutTimeout):
                reset_connection()
            if attempt == CUT_RETRIES:
                raise
            warnings.warn('Tuning cut: connection error, retry {} of {}: {}'.format(attempt + 1, CUT_RETRIES, e))
            time.sleep(RETRY_SECONDS)


def boot_worker() -> None:
    """
    Boot the app in a process of the pool of `run_tuning` (its default `initializer`).

    Every process of the pool owns its app and its SQLAlchemy engine: no connection is shared between processes.
    The queries of psycopg2 wait in Python (`psycopg2.extras.wait_select`) instead of inside libpq, so the watchdog
    of a cut (`timed_call`) can stop a query hung on a dead link. That green mode refuses COPY and large objects:
    the workers only read with plain queries, and must stay that way. It also connects asynchronously, so libpq's
    `connect_timeout` is not honoured in the workers: the watchdog of the cut is what bounds a hung connect.
    """
    import psycopg2.extensions
    import psycopg2.extras

    from .. import mtpy as app

    app.run()
    psycopg2.extensions.set_wait_callback(psycopg2.extras.wait_select)


def pool_cut(task: tuple) -> dict[str, Any]:
    """
    Run one cut in a process of the pool of `run_tuning`, each attempt under the watchdog of `timed_call`, with the
    retries of `run_with_retries`.

    Parameters
    ----------
    task : tuple
        `(runner, scope, event_date, cut, config, horizons, max_fc, verbose, event_params)`: picklable values;
        `runner` None runs the module-level `run_cut` with `event_params`, otherwise the replacement (a
        module-level callable with the signature of `run_cut` without `event_params`, for tests).

    Returns
    -------
    dict
        `result` (that of the cut, None when it failed), `error` (the message of the exception, None when it ran)
        and `warnings` (`(message, category)` of each warning raised by the cut, for the parent to raise again).
    """
    runner, scope, event_date, cut, config, horizons, max_fc, verbose, event_params = task
    run = runner or functools.partial(run_cut, event_params=event_params)
    result, error = None, None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = run_with_retries(functools.partial(timed_call, run), scope, event_date, cut, config,
                                      horizons=horizons, max_fc=max_fc, verbose=verbose)
        except Exception as e:
            error = str(e)

    return {'result': result, 'error': error, 'warnings': [(str(w.message), w.category) for w in caught]}


def pooled_outcomes(tasks: list[tuple], workers: int, initializer: Optional[Callable]):
    """
    Run the tasks of `pool_cut` on a pool of processes and yield their outcomes in the order of the tasks.

    The pool uses the `spawn` context (as macOS does by default): every process starts clean and boots with
    `initializer`. The warnings of each cut are raised again here, in the parent, so the callers collect them as in
    the sequential run.

    Parameters
    ----------
    tasks : list of tuple
        Arguments of `pool_cut`.
    workers : int
        Processes of the pool (never more than the tasks).
    initializer : callable or None
        Run once by each process before its first task.

    Yields
    ------
    dict
        The outcome of `pool_cut` of each task, in the order of `tasks`.
    """
    context = multiprocessing.get_context('spawn')
    with context.Pool(processes=min(workers, len(tasks)), initializer=initializer) as pool:
        for outcome in pool.imap(pool_cut, tasks):
            for message, category in outcome['warnings']:
                warnings.warn(message, category)
            yield outcome


def record_cuts(cuts: list[tuple], outcomes, scope: str, runner: Optional[Callable], horizons: tuple[int, ...],
                max_fc: Optional[int], verbose: int, cut_rows: list, prediction_frames: list, score_rows: list) -> None:
    """
    Run (or take from the pool) every cut of `run_tuning` and append its rows to the tables, in the order of `cuts`.

    Parameters
    ----------
    cuts : list of tuple
        `(config, cycle, cut)` of each cut, in order.
    outcomes : iterator or None
        The outcomes of the pool (`pooled_outcomes`), in the order of `cuts`; None runs each cut here.
    scope, runner, horizons, max_fc, verbose
        Those of `run_tuning`.
    cut_rows, prediction_frames, score_rows : list
        Rows of `cuts`, frames of `polls` and rows of the scores, appended to.
    """
    for config, cycle, cut in cuts:
        label = config.get('label')
        partition, event_date = cycle['partition'], cycle['event_date']
        keys = {'config': label, 'partition': partition, 'event_date': event_date, 'cut': pd.Timestamp(cut)}
        if outcomes is not None:
            outcome = next(outcomes)
            result, error = outcome['result'], outcome['error']
        else:
            run = runner or functools.partial(run_cut, event_params=cycle['event_params'])
            try:
                result, error = run_with_retries(run, scope, event_date, cut, config, horizons=horizons,
                                                 max_fc=max_fc, verbose=max(verbose - 1, 0)), None
            except Exception as e:
                result, error = None, str(e)
        if error is not None:
            warnings.warn('Tuning {} {} {} cut {:%Y-%m-%d} skipped: {}'.format(scope, label, event_date,
                                                                               pd.Timestamp(cut), error))
            cut_rows.append(keys | {'error': error})
            continue

        meta = result['meta']
        cut_rows.append(keys | {
            'n_polls': meta.get('n_polls'), 'n_overrides': meta.get('n_overrides', 0),
            'n_unpredicted': meta.get('n_unpredicted', 0), 'seconds': result['seconds'],
            'autocorr': result['autocorr'], 'roughness': result['roughness'], 'error': None
        } | {'rmse{}'.format(score['k']): score['rmse'] for score in result['scores']})
        prediction_frames.append(result['polls'].assign(**keys)[list(keys) + list(result['polls'].columns)])
        for score in result['scores']:
            score_rows.append(keys | {name: score[name] for name in ('k', 'n_polls', 'n_cells', 'rmse', 'mae',
                                                                      'skill')}
                              | {'autocorr': result['autocorr'], 'roughness': result['roughness']})
        if verbose > 0:
            print('Tuning {} {} {} cut {:%Y-%m-%d} ({} days): {:.1f} s'.format(
                scope, label, event_date, pd.Timestamp(cut), meta.get('drange'), result['seconds']), flush=True)


def run_tuning(
    scope: str,
    configs: list[dict],
    cycles: Optional[dict] = None,
    cut_days: int = 7,
    horizons: tuple[int, ...] = (7, 28),
    max_fc: Optional[int] = None,
    runner: Optional[Callable] = None,
    verbose: int = 0,
    plan: Optional[list[dict]] = None,
    workers: int = 1,
    initializer: Optional[Callable] = boot_worker
) -> dict[str, pd.DataFrame]:
    """
    Run every configuration on every cut of every cycle and aggregate its scores.

    A cut that fails on a database connection (`CONNECTION_ERRORS`) is retried up to `CUT_RETRIES` times,
    `RETRY_SECONDS` apart. A cut that still fails, or fails otherwise, is warned about and recorded in `cuts` with
    its `error` (as the backtest records a case it skips); it scores nothing. The callers decide what an errored
    cut means (`search_figures`, `tuning_scope`).

    With `workers` above 1 the cuts are spread over a pool of processes (`pooled_outcomes`), each with its own app
    and database engine; the outcomes are taken in the order of the cuts (configuration, cycle, cut), so the tables,
    the warnings and the lines printed are those of the sequential run (the `seconds` of each cut are its own,
    measured in its process).

    Parameters
    ----------
    scope : str
        Scope code.
    configs : list of dict
        Configurations of the average (see `default_config`), run as given; their labels must be unique.
    cycles : dict, optional
        `{'search': [...], 'check': [...]}`; `search_cycles(scope)` by default. Ignored with `plan`.
    cut_days : int, optional
        Days between two cuts. Ignored with `plan`.
    horizons : tuple of int, optional
        Horizons in days.
    max_fc : int, optional
        Overrides the `max_fc` of every configuration.
    runner : callable, optional
        Replacement of `run_cut`, with its signature without `event_params` (for tests). With the default
        `run_cut`, the params of each election are read once (`plan_cycles`) and passed to all its cuts.
    verbose : int, optional
        1 prints one line per cut with its seconds; higher levels are passed, minus 1, to the Forecaster.
    plan : list of dict, optional
        The cuts of the cycles as `plan_cycles` returns them, computed once by a caller that runs several times
        on the same cycles (`run_search`).
    workers : int, optional
        Processes running the cuts; 1 (the default) runs them here, one after the other.
    initializer : callable, optional
        Run once by each process of the pool before its first cut (`workers` above 1): `boot_worker` (the app and
        its database engine) by default; None for none (tests with a runner that needs no app).

    Returns
    -------
    dict
        `cuts`: one row per configuration, cycle and cut (`CUT_COLUMNS` plus `rmse{k}` of each horizon);
        `polls`: the predictions of every cut (`run_cut`) with the `config`, `partition`, `event_date` and `cut`
        columns in front; `scores`: one row per configuration, partition, cycle and horizon (`aggregate_scores`).

    Raises
    ------
    ValueError
        Two configurations share a label, or `workers` is not a positive integer.
    """
    labels = [config.get('label') for config in configs]
    if len(set(labels)) < len(labels):
        raise ValueError('Configurations need unique labels, got {}'.format(labels))
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer, got {!r}'.format(workers))
    if plan is None:
        plan = plan_cycles(scope, search_cycles(scope) if cycles is None else cycles, cut_days=cut_days,
                           read_params=runner is None, verbose=verbose)
    horizons = tuple(int(k) for k in horizons)

    cuts = [(config, cycle, cut) for config in configs for cycle in plan for cut in cycle['cuts']]
    if workers > 1 and len(cuts) > 1:
        tasks = [(runner, scope, cycle['event_date'], cut, config, horizons, max_fc, max(verbose - 1, 0),
                  None if runner is not None else cycle['event_params']) for config, cycle, cut in cuts]
        outcomes = pooled_outcomes(tasks, workers, initializer)
    else:
        outcomes = None

    cut_rows, prediction_frames, score_rows = [], [], []
    try:
        record_cuts(cuts, outcomes, scope, runner, horizons, max_fc, verbose, cut_rows, prediction_frames, score_rows)
    finally:
        if outcomes is not None:
            outcomes.close()   # stops the pool also when the loop raises

    rmse_columns = ['rmse{}'.format(k) for k in horizons]
    cuts = pd.DataFrame(cut_rows, columns=CUT_COLUMNS + rmse_columns)
    frames = [frame for frame in prediction_frames if not frame.empty]
    polls = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['config', 'partition',
                                                                                        'event_date', 'cut'])
    scores = aggregate_scores(pd.DataFrame(score_rows))

    return {'cuts': cuts, 'polls': polls, 'scores': scores}


def effective_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    The configuration as the model runs it: without the cap of the adaptive bandwidth under a fixed bandwidth.

    Parameters
    ----------
    config : dict
        Configuration of the average. Not modified.

    Returns
    -------
    dict
        A copy without `reg_params['bw_cap']` when `reg_params['bw_type']` is `'fixed'` (the cap only acts on the
        adaptive bandwidth); the configuration itself otherwise.
    """
    reg_params = config.get('reg_params') or {}
    if reg_params.get('bw_type') != 'fixed' or 'bw_cap' not in reg_params:
        return config

    return {**config, 'reg_params': {k: v for k, v in reg_params.items() if k != 'bw_cap'}}


def search_figures(out: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Figures of a configuration that the search compares, with every cycle weighing the same.

    Parameters
    ----------
    out : dict
        `run_tuning` of one configuration.

    Returns
    -------
    dict
        `rmse7` and `rmse28`: the mean over the cycles of the RMSE of each cycle at k = 7 and 28 (each cycle
        weighs the same, whatever its number of cuts and polls); `roughness`: the mean over the cycles of the
        mean roughness of their cuts (a property of the fitted series, each cut weighing the same). All NaN when
        any cut failed (the configuration would be scored on fewer cuts than the others: `coordinate_descent`
        never allows NaN figures) or without any value.
    """
    scores, cuts = out['scores'], out['cuts']
    if not cuts.empty and cuts['error'].notna().any():
        return {'rmse7': np.nan, 'rmse28': np.nan, 'roughness': np.nan}

    def by_cycle(k: int) -> float:
        """Mean over the cycles of the RMSE at `k`."""
        return float(scores.loc[scores['k'] == k, 'rmse'].astype(float).mean())

    roughness_by_cycle = cuts.groupby('event_date', sort=False)['roughness'].mean() if not cuts.empty else pd.Series()

    return {'rmse7': by_cycle(7), 'rmse28': by_cycle(28), 'roughness': float(roughness_by_cycle.astype(float).mean())}


def phase_cycles(cycles: dict[str, list[str]], phase: str) -> dict[str, list[str]]:
    """
    Cycles a phase of the search is scored on.

    Parameters
    ----------
    cycles : dict
        `{'search': [...], 'check': [...]}`.
    phase : str
        Key of `PHASES`.

    Returns
    -------
    dict
        The partition of the phase alone: `search` (from `SHRINKAGE_SINCE` in the shrinkage phase) or `check`.
    """
    partition = PHASES[phase]['partition']
    dates = list(cycles.get(partition) or [])
    if phase == 'shrinkage':
        dates = [date for date in dates if date >= SHRINKAGE_SINCE]

    return {partition: dates}


def run_search(
    scope: str,
    phase: str = 'smoothing',
    start: Optional[dict] = None,
    grid: dict = GRID,
    rounds: Optional[int] = None,
    cut_days: Optional[int] = None,
    product: bool = False,
    **kwargs
) -> dict[str, Any]:
    """
    One phase of the search of the configuration of the average on the future polls (spec A6, A7).

    The start is `normalize_config(start or default_config())`: the live default with every implicit key written,
    and `weights` None turned into `{'wspan': 4}`, the weights recomputed at each cut with the live window (the
    stored ones carry look-ahead). Each configuration is scored on the cycles of its phase, every cycle weighing
    the same (`search_figures`).

    - `'smoothing'` (phase 1): coordinate descent over `ORDER_SMOOTHING`, two rounds, a cut every 7 days on the
      `search` cycles, with `house_effects`, `dispersion` and `herding` off in every configuration (the returned
      `best` too; `tuning_scope` saves it with the three on).
    - `'shrinkage'` (phase 2): `ORDER_SHRINKAGE`, one round, a cut every 14 days on the `search` cycles from
      `SHRINKAGE_SINCE`, with everything on, from `start` (the `best` of phase 1).
    - `'check'`: no search; `start` with everything on and the normalised default (label `'default'`) are scored
      on the `check` cycles. When they are the same configuration only the default runs; a `start` labelled
      `'default'` (or unlabelled) that differs from it is relabelled `'best'`.

    With `product` (a part of a search phase, see `tuning_scope`) there is no descent: every configuration of the
    product of the values of the keys of the phase is evaluated from `start` (the values of `grid` plus the value of
    `start` in its own form when it is not among them or `start` writes it otherwise, so every configuration a
    descent from `start` can visit is there, see `product_configs`; repeated configurations once, by `config_key`),
    and `best` is `start`. Each trial is then one configuration (`key`
    `'product'`), so `merge_parts` can repeat the descent of the whole phase on the joined parts.

    An unlabelled start of a search is labelled `'start'`. A configuration with a failed cut gets NaN figures and
    is never kept (`search_figures`); a failed cut of the start of a search, or of either configuration of the
    check, stops the phase (`TuningFailed`): nothing could be compared with it.

    Only the keys of the phase present in `grid` are tried. Every distinct configuration runs once: the
    descent memoises its figures, and a candidate that only changes the cap of the adaptive bandwidth under a
    fixed bandwidth (`effective_config`, a no-op in the model) takes the figures of the configuration it equals
    without running; its trial is kept, with those figures and the same `evaluation` number.

    Parameters
    ----------
    scope : str
        Scope code.
    phase : str, optional
        `'smoothing'`, `'shrinkage'` or `'check'`.
    start : dict, optional
        Starting configuration; `default_config()` by default.
    grid : dict, optional
        Values of each key (`GRID`).
    rounds : int, optional
        Rounds of the descent; 2 in phase 1 and 1 in phase 2 by default.
    cut_days : int, optional
        Days between two cuts; 7 in phase 1 and in the check, 14 in phase 2 by default.
    product : bool, optional
        Evaluate the product of the values of the keys of the phase instead of descending (not in the check).
    **kwargs
        `cycles` (`{'search': [...], 'check': [...]}`, `search_cycles(scope)` by default), and `horizons`
        (must include 7 and 28), `max_fc`, `runner`, `verbose`, `workers` and `initializer` of `run_tuning`.

    Returns
    -------
    dict
        `best` (the configuration kept), `trials` (the table of `coordinate_descent` plus `evaluation`, the
        number of the run behind each trial, and `config_key`, the `config_key` of its effective configuration,
        which `merge_parts` joins on; empty in the check), `scores`, `cuts` and `polls` (the tables of
        `run_tuning` of every configuration run, with `evaluation`), `configs` (the label of each run, in order),
        `cycles` (those scored), `cut_days` and `rounds`.

    Raises
    ------
    ValueError
        Unknown phase, horizons without 7 or 28, or a search phase with none of its keys in `grid`.
    NothingToTune
        The phase has no cycle, or no cycle has a cut.
    TuningFailed
        A cut of the start of a search, or of a configuration of the check, failed.
    """
    if phase not in PHASES:
        raise ValueError('Unknown phase {!r}, expected one of {}'.format(phase, list(PHASES)))
    if product and phase == 'check':
        raise ValueError('The check has no product to evaluate')
    setup = PHASES[phase]
    rounds = setup['rounds'] if rounds is None else rounds
    cut_days = setup['cut_days'] if cut_days is None else cut_days
    cycles = kwargs.pop('cycles', None)
    cycles = phase_cycles(search_cycles(scope) if cycles is None else cycles, phase)
    if not {7, 28} <= {int(k) for k in kwargs.get('horizons', (7, 28))}:
        raise ValueError('The search scores the horizons 7 and 28, got {}'.format(kwargs.get('horizons')))
    order = [key for key in setup['order'] if key in grid]
    if phase != 'check' and len(order) == 0:
        raise ValueError('The grid has none of the keys of the {} phase {}, got {}'.format(
            phase, setup['order'], list(grid)))
    if not any(cycles.values()):
        raise NothingToTune('no {} cycle to tune'.format(setup['partition']))
    plan = plan_cycles(scope, cycles, cut_days=cut_days, read_params=kwargs.get('runner') is None,
                       verbose=kwargs.get('verbose', 0))
    if len(plan) == 0:
        raise NothingToTune('no cut with enough polls in the {} cycles'.format(setup['partition']))

    switches = {name: setup['effects'] for name in ('house_effects', 'dispersion', 'herding')}
    start = normalize_config(start or default_config()) | switches
    if phase != 'check' and start.get('label') is None:
        start['label'] = 'start'

    runs: dict[str, tuple[int, dict]] = {}
    failed: dict[str, str] = {}   # effective key of a run with a failed cut -> its first error
    tables: dict[str, list] = {'cuts': [], 'polls': [], 'scores': []}
    labels = []

    def evaluate_many(configs: list[dict]) -> dict[str, dict]:
        """Run the configurations not run yet (by their effective configuration) and record their tables."""
        pending = [config for config in configs if config_key(effective_config(config)) not in runs]
        if pending:
            out = run_tuning(scope, pending, plan=plan, **kwargs)
            for config in pending:
                number = len(runs)
                label = config.get('label')
                own = {name: frame.loc[frame['config'] == label] if not frame.empty else frame
                       for name, frame in out.items()}
                for name, frame in own.items():
                    tables[name].append(frame.assign(evaluation=number))
                runs[config_key(effective_config(config))] = (number, search_figures(own))
                errors = own['cuts']['error'].dropna() if not own['cuts'].empty else pd.Series(dtype=object)
                if len(errors) > 0:
                    failed[config_key(effective_config(config))] = '{} of {} cuts failed, first: {}'.format(
                        len(errors), len(own['cuts']), errors.iloc[0])
                labels.append(label)
        return {config_key(config): runs[config_key(effective_config(config))][1] for config in configs}

    if phase == 'check':
        default = normalize_config(default_config())
        configs = [default]
        if config_key(start) != config_key(default):
            if start.get('label') in (None, 'default'):
                start['label'] = 'best'
            configs = [start, default]
        evaluate_many(configs)
        for config in configs:
            reason = failed.get(config_key(effective_config(config)))
            if reason is not None:
                raise TuningFailed('check of {} on {}: {}'.format(config.get('label'), scope, reason))
        best = start
        trials = pd.DataFrame(columns=TRIAL_COLUMNS + ['evaluation', 'config_key'])
    elif product:
        configs = product_configs(start, grid, order)
        figures = evaluate_many(configs)
        start_key = config_key(effective_config(start))
        if start_key in failed:
            raise TuningFailed('start {} of the {} product on {}: {}'.format(start.get('label'), phase, scope,
                                                                            failed[start_key]))
        best = start
        trials = pd.DataFrame([
            {'round': 0, 'key': 'product', 'value': None, 'label': config.get('label'),
             **figures[config_key(config)], 'chosen': config_key(effective_config(config)) == start_key,
             'evaluation': runs[config_key(effective_config(config))][0],
             'config_key': config_key(effective_config(config))} for config in configs
        ], columns=TRIAL_COLUMNS + ['evaluation', 'config_key'])
    else:
        trial_keys = [config_key(effective_config(start))]

        def setter(config: dict, key: str, value: Any) -> dict:
            """`apply_setting`, recording the configuration of each trial in order."""
            out = apply_setting(config, key, value)
            trial_keys.append(config_key(effective_config(out)))
            return out

        start_key = config_key(effective_config(start))

        def evaluate(config: dict) -> dict:
            """Figures of one configuration on the cycles of the phase; a failed start stops the search."""
            figures = evaluate_many([config])[config_key(config)]
            if config_key(effective_config(config)) == start_key and start_key in failed:
                raise TuningFailed('start {} of the {} search on {}: {}'.format(
                    start.get('label'), phase, scope, failed[start_key]))
            return figures

        best, trials = coordinate_descent(evaluate, start, grid, order, rounds=rounds, setter=setter,
                                          normalize=normalize_config)
        trials['evaluation'] = [runs[key][0] for key in trial_keys]
        trials['config_key'] = trial_keys

    concat = {name: pd.concat(frames, ignore_index=True) for name, frames in tables.items()}

    return {'best': best, 'trials': trials, **concat, 'configs': labels, 'cycles': cycles, 'cut_days': cut_days,
            'rounds': rounds if phase != 'check' and not product else None}


def product_configs(start: dict[str, Any], grid: dict, order: list[str]) -> list[dict]:
    """
    Every configuration of the product of the values of some keys, applied on a start.

    A descent from `start` writes each value of the grid with `apply_setting` (`coordinate_descent` applies its
    candidates even when the value is the one in force), while a key it has not replaced keeps the form `start`
    gives it. Both forms are in the product: the values of the grid as `apply_setting` writes them, plus the value
    of `start` as `start` writes it when it is not in the grid or `start` writes it in another form (`herd_prior` 5
    for 5.0, `bw_kwargs` without `min_delta`, a pilot `bw` the adaptive setting drops). So every candidate of the
    descent has its `config_key` here, the one `merge_parts` looks up; a normalised start (whose values are written
    as `apply_setting` writes them) adds nothing but the values outside the grid.

    Parameters
    ----------
    start : dict
        Normalised starting configuration (`normalize_config`).
    grid : dict
        Values of each key.
    order : list of str
        Keys of the product, in the order they are applied.

    Returns
    -------
    list of dict
        `start` first, then each configuration of the product not seen yet (by the `config_key` of its effective
        configuration), labelled `'key=value, ...'` (`'key=value (start)'` for a value of the grid kept in the form
        of `start`): every configuration a coordinate descent from `start` over `order` can visit.
    """
    options = {}
    for key in order:
        own = setting_of(start, key)
        options[key] = [(value, True) for value in grid[key]]
        if own not in grid[key]:
            options[key].append((own, False))
        else:
            written = next(value for value in grid[key] if value == own)   # 5.0 of the grid for a 5 of the start
            if config_key(apply_setting(start, key, written)) != config_key(start):
                options[key].append((own, False))

    out, seen = [start], {config_key(effective_config(start))}
    for combination in itertools.product(*(options[key] for key in order)):
        config = start
        for key, (value, applied) in zip(order, combination):
            if applied:
                config = apply_setting(config, key, value)
        config = {**config, 'label': ', '.join(
            '{}={}{}'.format(key, value, ' (start)' if not applied and value in grid[key] else '')
            for key, (value, applied) in zip(order, combination))}
        key = config_key(effective_config(config))
        if key not in seen:
            seen.add(key)
            out.append(config)

    return out


def part_figures(trials: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Figures of every configuration of a part of the shrinkage phase on each of its cycles.

    Parameters
    ----------
    trials : pd.DataFrame
        `trials-shrinkage-{part}.csv`: `evaluation` and `config_key` of each trial.
    scores : pd.DataFrame
        `scores-shrinkage-{part}.csv`: `evaluation`, `event_date`, `k`, `rmse` and `roughness`.

    Returns
    -------
    pd.DataFrame
        `config_key`, `event_date`, `rmse7`, `rmse28` and `roughness` (the RMSE of the cycle at k = 7 and 28 and
        the mean roughness of its cuts, the figures `search_figures` averages), one row per configuration and
        cycle. A configuration whose trial has no figures (a failed cut: `search_figures` gave it NaN) is left out.
    """
    scored = trials.dropna(subset=['rmse7', 'rmse28', 'roughness'])
    keys = scored.drop_duplicates('evaluation').set_index('evaluation')['config_key']
    rows = []
    for (evaluation, event_date), group in scores.loc[scores['evaluation'].isin(keys.index)].groupby(
            ['evaluation', 'event_date'], sort=False):
        rmse = group.set_index('k')['rmse'].astype(float)
        rows.append({'config_key': keys[evaluation], 'event_date': str(event_date), 'rmse7': rmse.get(7, np.nan),
                     'rmse28': rmse.get(28, np.nan), 'roughness': float(group['roughness'].astype(float).iloc[0])})

    return pd.DataFrame(rows, columns=['config_key', 'event_date', 'rmse7', 'rmse28', 'roughness'])


def merge_parts(
    parts: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    start: dict[str, Any],
    grid: dict = GRID,
    rounds: Optional[int] = None
) -> dict[str, Any]:
    """
    The descent of the shrinkage phase repeated on the joined tables of its parts (run, for example, one per cycle in
    separate tasks).

    The figures of a configuration are those `run_search` would give it on the cycles of every part together: the
    mean over the cycles of the RMSE of each cycle at k = 7 and 28 and of its roughness, every cycle weighing the
    same. Parts that evaluated the product of the keys of the phase (`run_search` with `product`, what
    `tuning_scope` runs for a part) hold every configuration the descent can visit, so the result is that of the
    whole phase run at once. A configuration some cycle has not run (parts from a smaller grid, or from their own
    descents) has NaN figures and is never kept; it is warned about.

    Parameters
    ----------
    parts : dict
        `{part: (trials, scores)}`, the `trials-shrinkage-{part}.csv` and `scores-shrinkage-{part}.csv` of each part.
    start : dict
        The start of the parts (the `best.json` of the smoothing phase).
    grid : dict, optional
        Values of each key (`GRID`).
    rounds : int, optional
        Rounds of the descent; that of the shrinkage phase by default.

    Returns
    -------
    dict
        `best` (the configuration kept), `trials` (the table of `coordinate_descent` plus `config_key`), `cycles`
        (`{'search': [...]}`, those of the parts), `n_configs` (configurations with figures in the descent) and
        `missing` (labels of the trials with configurations missing from some cycle).

    Raises
    ------
    NothingToTune
        No part has figures.
    TuningFailed
        The start is missing from some cycle (a part from another start, or with a failed cut of its start).
    """
    figures = pd.concat([part_figures(trials, scores) for trials, scores in parts.values()], ignore_index=True)
    if figures.empty:
        raise NothingToTune('no figures in the shrinkage parts {}'.format(sorted(parts)))
    cycles = sorted(figures['event_date'].unique())
    by_key = {key: group.set_index('event_date') for key, group in figures.groupby('config_key', sort=False)}

    setup = PHASES['shrinkage']
    order = [key for key in setup['order'] if key in grid]
    switches = {name: setup['effects'] for name in ('house_effects', 'dispersion', 'herding')}
    start = normalize_config(start) | switches
    if start.get('label') is None:
        start['label'] = 'start'
    start_key = config_key(effective_config(start))

    def complete(key: str) -> bool:
        """Whether a configuration has figures on every cycle of the parts."""
        return key in by_key and set(by_key[key].index) == set(cycles)

    if not complete(start_key):
        raise TuningFailed('the start {} is missing from some cycle of the shrinkage parts'.format(start.get('label')))

    trial_keys, missing = [start_key], []

    def setter(config: dict, key: str, value: Any) -> dict:
        """`apply_setting`, recording the configuration of each trial in order."""
        out = apply_setting(config, key, value)
        trial_keys.append(config_key(effective_config(out)))
        return out

    def evaluate(config: dict) -> dict:
        """Mean figures of a configuration over the cycles, NaN when some cycle has not run it."""
        key = config_key(effective_config(config))
        if not complete(key):
            missing.append(config.get('label'))
            return {'rmse7': np.nan, 'rmse28': np.nan, 'roughness': np.nan}
        rows = by_key[key]
        return {name: float(rows[name].astype(float).mean()) for name in ('rmse7', 'rmse28', 'roughness')}

    rounds = setup['rounds'] if rounds is None else rounds
    best, trials = coordinate_descent(evaluate, start, grid, order, rounds=rounds, setter=setter,
                                      normalize=normalize_config)
    trials['config_key'] = trial_keys
    if missing:
        warnings.warn('merge: {} configurations of the descent are missing from some part and are never kept: {}'
                      .format(len(missing), ', '.join(str(label) for label in missing)))

    return {'best': best, 'trials': trials, 'cycles': {'search': cycles},
            'n_configs': len({key for key in trial_keys if complete(key)}), 'missing': missing}


def tuning_names(fs, folder: str) -> list[str]:
    """
    Names of the files of the tuning in the folder of a scope (`TUNING_NAME`; not those of other scopes nor others).

    Parameters
    ----------
    fs : FileSystem
        File system of the folder (`S3` or local).
    folder : str
        Folder in the file system (`''` for its root).

    Returns
    -------
    list of str
        The file names, sorted.
    """
    names = [path.rstrip('/').rsplit('/', 1)[-1] for path in fs.glob(results_path(folder, '*'))]

    return sorted(name for name in names if TUNING_NAME.match(name))


def copy_tuning(scope: str, src: tuple, dst: tuple) -> dict:
    """
    Copy the tuning tables of a scope from one folder to another, mirroring the source (the twin of
    `backtest.copy_results`).

    Parameters
    ----------
    scope : str
        Scope code (only for messages).
    src, dst : tuple[FileSystem, str]
        File system and folder of the source and the destination, as `tuning_fs` returns them.

    Returns
    -------
    dict
        `copied`: names of the tuning files written to the destination; `removed`: tuning files the source does not
        have that were deleted from the destination. Other files, and the folders of other scopes, are not touched.

    Raises
    ------
    NothingToTune
        If the source has no tuning file (nothing is touched).
    """
    src_fs, src_folder = src
    dst_fs, dst_folder = dst
    names = tuning_names(src_fs, src_folder)
    if not names:
        raise NothingToTune('no tuning tables for {} in {}'.format(scope, describe_source(src_fs, src_folder)))
    for name in names:
        dst_fs.write_bytes(src_fs.read_bytes(results_path(src_folder, name)), results_path(dst_folder, name))
    removed = [name for name in tuning_names(dst_fs, dst_folder) if name not in names]
    for name in removed:
        dst_fs.remove(results_path(dst_folder, name))

    return {'copied': names, 'removed': removed}


def read_parts(fs, folder: str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    The tables of the parts of the shrinkage phase in a folder.

    Parameters
    ----------
    fs : FileSystem
        File system of the folder.
    folder : str
        Folder of the scope.

    Returns
    -------
    dict
        `{part: (trials, scores)}` of every `trials-shrinkage-{part}.csv` with its `scores-shrinkage-{part}.csv`.
    """
    parts = {}
    for name in tuning_names(fs, folder):
        match = re.match(r'^trials-shrinkage-(.+)\.csv$', name)
        if match is None:
            continue
        part = match.group(1)
        scores = results_path(folder, 'scores-shrinkage-{}.csv'.format(part))
        if fs.exists(scores):
            parts[part] = (fs.read_csv(results_path(folder, name)), fs.read_csv(scores))

    return parts


def tuning_scope(
    scope: str,
    out: Optional[str] = None,
    mode: str = 'smoothing',
    configs: Optional[list[dict]] = None,
    start: Optional[dict] = None,
    grid: Optional[dict] = None,
    cycles: Optional[dict] = None,
    cut_days: Optional[int] = None,
    horizons: tuple[int, ...] = (7, 28),
    rounds: Optional[int] = None,
    verbose: int = 0,
    runner: Optional[Callable] = None,
    stats: Optional[Callable] = None,
    prov: Optional[Callable] = None,
    workers: int = 1,
    part: Optional[str] = None
) -> dict[str, Any]:
    """
    Run one mode of the tuning of a scope and write its tables and `run.json` to `tuning_fs(out, scope)`.

    Each run rewrites `TUNING_FILES` in the folder of the scope. The search modes also write `trials-{mode}.csv`
    and `scores-{mode}.csv` (kept per phase, so the scores of the search survive the check) and, in `smoothing`
    and `shrinkage`, `best.json`: the configuration kept with `house_effects`, `dispersion` and `herding` on and
    labelled with every setting that differs from the default (`best_label`), as the next phase and the check read
    it (the shrinkage one overwrites the smoothing one).

    The shrinkage phase can be split into parts (one per cycle, for example, in separate tasks): a run with `part`
    does not descend but evaluates the whole product of the values of `tau`, `he_n_iter` and `herd_prior` from the
    start (`run_search` with `product`: 18 configurations with the grid of the spec, three times the 6 of the
    descent of the whole phase), writes only `cuts-`, `scores-` and `trials-shrinkage-{part}.csv` and
    `run-shrinkage-{part}.json` (with `search` `'product'` and its `start`), and never `best.json`. The `merge`
    mode then joins the parts of the folder and repeats the descent of the phase on the joined table
    (`merge_parts`): the same result as the whole phase run at once, since every configuration that descent can
    visit is in every part. It starts from the start the parts recorded (`best.json` only for parts without
    one), so a second merge gives the same result, and writes `best.json`, `trials-shrinkage.csv` and
    `trials.csv` (the joined descent), `scores-shrinkage.csv` and `scores.csv` (the scores of every part, with its
    `part`), `cuts.csv` (the cuts of every part) and `run.json`, and removes the `polls.csv` of a previous run (the
    parts keep no predictions).

    Parameters
    ----------
    scope : str
        Scope code.
    out : str, optional
        Local folder: the national scope writes at its root, the others in `out/{scope}`. By default the file
        system of the app (local `files/` or S3) under `TUNING_PREFIX` (`backtest/tuning`).
    mode : str, optional
        `'smoothing'`, `'shrinkage'` or `'check'` (`run_search` with that phase, from `start` or, in the last
        two, from the `best.json` of the folder), `'evaluate'` (`run_tuning` of `configs` as given, without
        normalising them: `default_config()` there is the live model with the weights stored with the polls) or
        `'merge'` (the parts of the shrinkage phase in the folder, from `start` or its `best.json`; no cut runs).
    configs : list of dict, optional
        Configurations of the `'evaluate'` mode.
    start : dict, optional
        Start of the search modes.
    grid : dict, optional
        Values of each key of the search (`GRID` by default).
    cycles : dict, optional
        `{'search': [...], 'check': [...]}`; `search_cycles(scope)` by default.
    cut_days : int, optional
        Days between two cuts; the default of the phase (7 in `'evaluate'`).
    horizons : tuple of int, optional
        Horizons in days.
    rounds : int, optional
        Rounds of the descent; the default of the phase.
    verbose : int, optional
        Verbosity of `run_tuning` (1 prints one line per cut).
    runner : callable, optional
        Replacement of `run_cut` (for tests).
    stats : callable, optional
        Replacement of `backtest.poll_stats` (for tests).
    prov : callable, optional
        Replacement of `publish.provenance`, whose `commit` and `dirty` go to `run.json` (for tests).
    workers : int, optional
        Processes running the cuts (`run_tuning`); 1 runs them here.
    part : str, optional
        Name of a part of the shrinkage phase (only in `'shrinkage'`; see above).

    Returns
    -------
    dict
        `n_configs` (configurations run), `n_cuts` (distinct cuts), `cycles` (those run, as in `run.json`),
        `best` (in the search modes and `merge`; not in a part), `folder` written and `seconds` (`run`, `write`
        and `total`).

    Raises
    ------
    ValueError
        Unknown mode, `'evaluate'` without configurations, or `part` outside `'shrinkage'`.
    FileNotFoundError
        `'shrinkage'`, `'check'` or `'merge'` without `start` nor a `best.json` in the folder.
    NothingToTune
        No cycle to tune, or no cut with enough polls in its cycles, or no part to merge; nothing is written.
    TuningFailed
        Cuts exist but every one failed, or a search could not trust its start or the check a configuration
        (`run_search`); nothing is written.
    """
    began = time.perf_counter()
    modes = list(PHASES) + ['evaluate', 'merge']
    if mode not in modes:
        raise ValueError('Unknown mode {!r}, expected one of {}'.format(mode, modes))
    if mode == 'evaluate' and not configs:
        raise ValueError('The evaluate mode needs configurations')
    if part is not None and mode != 'shrinkage':
        raise ValueError('A part only applies to the shrinkage mode, got mode {!r}'.format(mode))

    fs, folder = tuning_fs(out, scope)
    if start is None and mode in ('shrinkage', 'check', 'merge'):
        path = results_path(folder, 'best.json')
        if not fs.exists(path):
            raise FileNotFoundError('no best.json in {}: run the previous phase or pass start'.format(
                describe_source(fs, folder)))
        start = json.loads(fs.read_bytes(path).decode('utf-8'))
    if mode == 'merge':
        return merge_scope(scope, fs, folder, start, grid=grid or GRID, rounds=rounds, stats=stats, prov=prov,
                           began=began)

    cycles = search_cycles(scope) if cycles is None else cycles
    if not any(cycles.get(partition) for partition in ('search', 'check')):
        raise NothingToTune('no cycle to tune')
    if mode == 'evaluate':
        cut_days = 7 if cut_days is None else cut_days
        result = run_tuning(scope, configs, cycles=cycles, cut_days=cut_days, horizons=horizons, runner=runner,
                            verbose=verbose, workers=workers)
        result |= {'trials': pd.DataFrame(columns=TRIAL_COLUMNS), 'configs': [c.get('label') for c in configs],
                   'cycles': cycles, 'cut_days': cut_days, 'rounds': None}
    else:
        result = run_search(scope, phase=mode, start=start, grid=grid or GRID, rounds=rounds, cut_days=cut_days,
                            cycles=cycles, horizons=horizons, runner=runner, verbose=verbose, workers=workers,
                            product=part is not None)
    if result['cuts'].empty:
        raise NothingToTune('no cut with enough polls in the cycles')
    evaluated = result['cuts'].loc[result['cuts']['error'].isnull()]
    if evaluated.empty:
        raise TuningFailed('every cut failed, first error: {}'.format(result['cuts']['error'].iloc[0]))
    seconds = {'run': time.perf_counter() - began}

    best = None
    if part is not None:
        for name in PART_TABLES:
            fs.write_csv(result[name], results_path(folder, '{}-shrinkage-{}.csv'.format(name, part)))
    else:
        for name in ('cuts', 'polls', 'scores', 'trials'):
            fs.write_csv(result[name], results_path(folder, '{}.csv'.format(name)))
        if mode != 'evaluate':
            for name in ('trials', 'scores'):
                fs.write_csv(result[name], results_path(folder, '{}-{}.csv'.format(name, mode)))
            best = result['best']
        if mode in ('smoothing', 'shrinkage'):
            best = best | {'house_effects': True, 'dispersion': True, 'herding': True, 'label': best_label(best)}
            fs.write_bytes(json.dumps(best, indent=1).encode('utf-8'), results_path(folder, 'best.json'))

    if part is not None:
        extra = {'search': 'product', 'part': part, 'start': start}
    else:
        extra = {} if mode == 'evaluate' else {'search': 'descent'}
    record = run_record(scope, mode, prov, stats, configs=result['configs'], cycles=result['cycles'],
                        cut_days=result['cut_days'], horizons=[int(k) for k in horizons], rounds=result['rounds'],
                        seconds=round(seconds['run'], 1), **extra)
    run_name = 'run.json' if part is None else 'run-shrinkage-{}.json'.format(part)
    fs.write_bytes(json.dumps(record, indent=1).encode('utf-8'), results_path(folder, run_name))
    seconds['write'] = time.perf_counter() - began - seconds['run']
    seconds['total'] = time.perf_counter() - began

    figures = {'n_configs': len(result['configs']),
               'n_cuts': int(evaluated[['event_date', 'cut']].drop_duplicates().shape[0]),
               'cycles': result['cycles'], 'folder': describe_source(fs, folder), 'seconds': seconds}
    if best is not None:
        figures['best'] = best

    return figures


def run_record(scope: str, mode: str, prov: Optional[Callable], stats: Optional[Callable], **fields) -> dict:
    """
    The `run.json` of a run of the tuning: when, from which commit, on which state of the polls and with what.

    Parameters
    ----------
    scope : str
        Scope code.
    mode : str
        Mode of the run.
    prov : callable or None
        Replacement of `publish.provenance` (for tests).
    stats : callable or None
        Replacement of `backtest.poll_stats` (for tests).
    **fields
        The figures of the run (`configs`, `cycles`, `cut_days`, `horizons`, `rounds`, `seconds`, ...).

    Returns
    -------
    dict
        `run_at`, `commit`, `dirty`, `scope`, `mode`, the fields given, `db_polls` and `db_last_poll`, with
        `seconds` last.
    """
    if prov is None:
        from .publish.common import provenance as prov
    provenance = prov()
    seconds = fields.pop('seconds', None)

    return {'run_at': datetime.now().isoformat(timespec='seconds'), 'commit': provenance['commit'],
            'dirty': provenance.get('dirty'), 'scope': scope, 'mode': mode, **fields, 'db_polls': None,
            'db_last_poll': None, **(stats or poll_stats)(scope), 'seconds': seconds}


def merge_scope(
    scope: str,
    fs,
    folder: str,
    start: dict[str, Any],
    grid: dict = GRID,
    rounds: Optional[int] = None,
    stats: Optional[Callable] = None,
    prov: Optional[Callable] = None,
    began: Optional[float] = None
) -> dict[str, Any]:
    """
    The `merge` mode of `tuning_scope`: join the parts of the shrinkage phase of a folder and keep the best.

    Parameters
    ----------
    scope : str
        Scope code.
    fs : FileSystem
        File system of the folder.
    folder : str
        Folder of the scope.
    start : dict
        Start of the parts when they recorded none (the `best.json` of the folder); the start recorded in their
        `run-shrinkage-{part}.json` otherwise, so a second merge (after the first rewrote `best.json`) repeats the
        first.
    grid : dict, optional
        Values of each key (`GRID`).
    rounds : int, optional
        Rounds of the descent; that of the shrinkage phase by default.
    stats, prov : callable, optional
        Replacements of `backtest.poll_stats` and `publish.provenance` (for tests).
    began : float, optional
        `time.perf_counter()` at the start of the run.

    Returns
    -------
    dict
        The figures of `tuning_scope`: `n_configs` (configurations with figures in the joined descent), `n_cuts`
        (cuts of the start in the parts), `cycles`, `best`, `folder` and `seconds`.

    Raises
    ------
    NothingToTune
        No part in the folder.
    TuningFailed
        The start is missing from some cycle of the parts, or the parts recorded different starts.
    """
    began = time.perf_counter() if began is None else began
    parts = read_parts(fs, folder)
    if not parts:
        raise NothingToTune('no shrinkage part in {}'.format(describe_source(fs, folder)))
    records = {}
    for name in parts:
        path = results_path(folder, 'run-shrinkage-{}.json'.format(name))
        if fs.exists(path):
            records[name] = json.loads(fs.read_bytes(path).decode('utf-8'))
    starts = {name: record['start'] for name, record in records.items() if record.get('start') is not None}
    if starts:
        keys = {config_key(effective_config(normalize_config(own))) for own in starts.values()}
        if len(keys) > 1:
            raise TuningFailed('the shrinkage parts {} start from different configurations'.format(sorted(starts)))
        start = next(iter(starts.values()))
    result = merge_parts(parts, start, grid=grid, rounds=rounds)
    start_key = result['trials']['config_key'].iloc[0]
    scores = pd.concat([part_scores.assign(part=name) for name, (_, part_scores) in parts.items()], ignore_index=True)
    n_cuts = 0
    for trials, part_scores in parts.values():
        evaluations = trials.loc[trials['config_key'] == start_key, 'evaluation'].unique()
        own = part_scores.loc[part_scores['evaluation'].isin(evaluations)]
        n_cuts += int(own.drop_duplicates('event_date')['n_cuts'].sum())
    seconds = {'run': time.perf_counter() - began}

    best = result['best'] | {'house_effects': True, 'dispersion': True, 'herding': True,
                             'label': best_label(result['best'])}
    fs.write_bytes(json.dumps(best, indent=1).encode('utf-8'), results_path(folder, 'best.json'))
    cuts = []
    for name in parts:
        path = results_path(folder, 'cuts-shrinkage-{}.csv'.format(name))
        if fs.exists(path):
            cuts.append(fs.read_csv(path).assign(part=name))
    for name in ('trials-shrinkage.csv', 'trials.csv'):
        fs.write_csv(result['trials'], results_path(folder, name))
    for name in ('scores-shrinkage.csv', 'scores.csv'):
        fs.write_csv(scores, results_path(folder, name))
    fs.write_csv(pd.concat(cuts, ignore_index=True) if cuts else pd.DataFrame(columns=CUT_COLUMNS + ['part']),
                 results_path(folder, 'cuts.csv'))
    if fs.exists(results_path(folder, 'polls.csv')):
        fs.remove(results_path(folder, 'polls.csv'))
    rounds = PHASES['shrinkage']['rounds'] if rounds is None else rounds
    record = run_record(scope, 'merge', prov, stats, parts=sorted(parts), cycles=result['cycles'], rounds=rounds,
                        missing=result['missing'], seconds=round(seconds['run'], 1))
    fs.write_bytes(json.dumps(record, indent=1).encode('utf-8'), results_path(folder, 'run.json'))
    seconds['write'] = time.perf_counter() - began - seconds['run']
    seconds['total'] = time.perf_counter() - began

    return {'n_configs': result['n_configs'], 'n_cuts': n_cuts, 'cycles': result['cycles'], 'best': best,
            'folder': describe_source(fs, folder), 'seconds': seconds}
