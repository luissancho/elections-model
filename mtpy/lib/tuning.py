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
import json
import os
import time
import warnings

from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .backtest import describe_source, poll_stats, results_path
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

# Files of the tuning of a scope in its folder (the search modes also write `best.json` and `trials-{phase}.csv`)
TUNING_FILES = ('cuts.csv', 'polls.csv', 'scores.csv', 'trials.csv', 'run.json')

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

CUT_COLUMNS = ['config', 'partition', 'event_date', 'cut', 'n_polls', 'n_overrides', 'n_unpredicted', 'seconds',
               'autocorr', 'roughness', 'error']
SCORE_COLUMNS = ['config', 'partition', 'event_date', 'k', 'n_cuts', 'n_polls', 'n_cells', 'rmse', 'mae', 'skill',
                 'autocorr', 'roughness']


def predict_polls(fc: Forecaster, polls: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """
    Predict the future polls of a cut with its fitted average and house effects.

    The average of a party is read at the date of the poll (`fit_forecast` leaves the straight prolongation up
    to `date_fit_last + max_fc`; after that day the last value is held) and the house effect of the pollster
    is added.

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
    n_overrides = int(overrides.index.isin(series_polls).sum()) if overrides is not None else 0

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
    """A scope with no cycle to tune, or none of whose cuts could be evaluated: nothing is written."""


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
        so each cell weighs the same, spec A4); `mae`, `skill`, `autocorr` and `roughness` the means weighted by
        the cells of each cut.
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
            'autocorr': weighted_mean(group['autocorr'], cells), 'roughness': weighted_mean(group['roughness'], cells)
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
        The current date by default: the next election is a running cycle only when it is after it.

    Returns
    -------
    dict
        `search`: the featured elections with enough polls before `CHECK_SINCE`; `check`: those from `CHECK_SINCE`
        on plus the next election of the scope (the running cycle, `get_next_event_date` after the last featured
        one) when it has not been held yet. Dates as `YYYY-MM-DD`, in order.
    """
    dates = list(get_event_dates(scope, date_from=since, min_polls=min_polls, featured=True))
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    following = get_next_event_date(scope, date_from=dates[-1] if dates else since)
    if following is not None and pd.Timestamp(following) > today and following not in dates:
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


def run_tuning(
    scope: str,
    configs: list[dict],
    cycles: Optional[dict] = None,
    cut_days: int = 7,
    horizons: tuple[int, ...] = (7, 28),
    max_fc: Optional[int] = None,
    runner: Optional[Callable] = None,
    verbose: int = 0,
    plan: Optional[list[dict]] = None
) -> dict[str, pd.DataFrame]:
    """
    Run every configuration on every cut of every cycle and aggregate its scores.

    A cut that fails is warned about and recorded in `cuts` with its `error` (as the backtest records a case it
    skips); it scores nothing.

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

    Returns
    -------
    dict
        `cuts`: one row per configuration, cycle and cut (`CUT_COLUMNS` plus `rmse{k}` of each horizon);
        `polls`: the predictions of every cut (`run_cut`) with the `config`, `partition`, `event_date` and `cut`
        columns in front; `scores`: one row per configuration, partition, cycle and horizon (`aggregate_scores`).

    Raises
    ------
    ValueError
        Two configurations share a label.
    """
    labels = [config.get('label') for config in configs]
    if len(set(labels)) < len(labels):
        raise ValueError('Configurations need unique labels, got {}'.format(labels))
    if plan is None:
        plan = plan_cycles(scope, search_cycles(scope) if cycles is None else cycles, cut_days=cut_days,
                           read_params=runner is None, verbose=verbose)
    horizons = tuple(int(k) for k in horizons)

    cut_rows, prediction_frames, score_rows = [], [], []
    for config in configs:
        label = config.get('label')
        for cycle in plan:
            partition, event_date = cycle['partition'], cycle['event_date']
            run = runner or functools.partial(run_cut, event_params=cycle['event_params'])
            for cut in cycle['cuts']:
                keys = {'config': label, 'partition': partition, 'event_date': event_date, 'cut': pd.Timestamp(cut)}
                try:
                    result = run(scope, event_date, cut, config, horizons=horizons, max_fc=max_fc,
                                 verbose=max(verbose - 1, 0))
                except Exception as e:
                    warnings.warn('Tuning {} {} {} cut {:%Y-%m-%d} skipped: {}'.format(scope, label, event_date,
                                                                                       pd.Timestamp(cut), e))
                    cut_rows.append(keys | {'error': str(e)})
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
                        scope, label, event_date, pd.Timestamp(cut), meta.get('drange'), result['seconds']))

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
        mean roughness of their cuts (a property of the fitted series, each cut weighing the same). NaN without
        any value.
    """
    scores, cuts = out['scores'], out['cuts']

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
    **kwargs
        `cycles` (`{'search': [...], 'check': [...]}`, `search_cycles(scope)` by default), and `horizons`
        (must include 7 and 28), `max_fc`, `runner` and `verbose` of `run_tuning`.

    Returns
    -------
    dict
        `best` (the configuration kept), `trials` (the table of `coordinate_descent` plus `evaluation`, the
        number of the run behind each trial; empty in the check), `scores`, `cuts` and `polls` (the tables of
        `run_tuning` of every configuration run, with `evaluation`), `configs` (the label of each run, in order),
        `cycles` (those scored), `cut_days` and `rounds`.

    Raises
    ------
    ValueError
        Unknown phase, or horizons without 7 or 28.
    NothingToTune
        The phase has no cycle, or no cycle has a cut.
    """
    if phase not in PHASES:
        raise ValueError('Unknown phase {!r}, expected one of {}'.format(phase, list(PHASES)))
    setup = PHASES[phase]
    rounds = setup['rounds'] if rounds is None else rounds
    cut_days = setup['cut_days'] if cut_days is None else cut_days
    cycles = kwargs.pop('cycles', None)
    cycles = phase_cycles(search_cycles(scope) if cycles is None else cycles, phase)
    if not {7, 28} <= {int(k) for k in kwargs.get('horizons', (7, 28))}:
        raise ValueError('The search scores the horizons 7 and 28, got {}'.format(kwargs.get('horizons')))
    if not any(cycles.values()):
        raise NothingToTune('no {} cycle to tune'.format(setup['partition']))
    plan = plan_cycles(scope, cycles, cut_days=cut_days, read_params=kwargs.get('runner') is None,
                       verbose=kwargs.get('verbose', 0))
    if len(plan) == 0:
        raise NothingToTune('no cut with enough polls in the {} cycles'.format(setup['partition']))

    switches = {name: setup['effects'] for name in ('house_effects', 'dispersion', 'herding')}
    start = normalize_config(start or default_config()) | switches

    runs: dict[str, tuple[int, dict]] = {}
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
        best = start
        trials = pd.DataFrame(columns=TRIAL_COLUMNS + ['evaluation'])
    else:
        trial_keys = [config_key(effective_config(start))]

        def setter(config: dict, key: str, value: Any) -> dict:
            """`apply_setting`, recording the configuration of each trial in order."""
            out = apply_setting(config, key, value)
            trial_keys.append(config_key(effective_config(out)))
            return out

        def evaluate(config: dict) -> dict:
            """Figures of one configuration on the cycles of the phase."""
            return evaluate_many([config])[config_key(config)]

        order = [key for key in setup['order'] if key in grid]
        best, trials = coordinate_descent(evaluate, start, grid, order, rounds=rounds, setter=setter,
                                          normalize=normalize_config)
        trials['evaluation'] = [runs[key][0] for key in trial_keys]

    concat = {name: pd.concat(frames, ignore_index=True) for name, frames in tables.items()}

    return {'best': best, 'trials': trials, **concat, 'configs': labels, 'cycles': cycles, 'cut_days': cut_days,
            'rounds': rounds if phase != 'check' else None}


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
    prov: Optional[Callable] = None
) -> dict[str, Any]:
    """
    Run one mode of the tuning of a scope and write its tables and `run.json` to `tuning_fs(out, scope)`.

    Each run rewrites `TUNING_FILES` in the folder of the scope. The search modes also write `trials-{mode}.csv`
    (kept per phase) and, in `smoothing` and `shrinkage`, `best.json`: the configuration kept with
    `house_effects`, `dispersion` and `herding` on, as the next phase and the check read it (the shrinkage one
    overwrites the smoothing one).

    Parameters
    ----------
    scope : str
        Scope code.
    out : str, optional
        Local folder: the national scope writes at its root, the others in `out/{scope}`. By default the file
        system of the app (local `files/` or S3) under `TUNING_PREFIX` (`backtest/tuning`).
    mode : str, optional
        `'smoothing'`, `'shrinkage'` or `'check'` (`run_search` with that phase, from `start` or, in the last
        two, from the `best.json` of the folder) or `'evaluate'` (`run_tuning` of `configs` as given, without
        normalising them: `default_config()` there is the live model with the weights stored with the polls).
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

    Returns
    -------
    dict
        `n_configs` (configurations run), `n_cuts` (distinct cuts), `cycles` (those run, as in `run.json`),
        `best` (in the search modes), `folder` written and `seconds` (`run`, `write` and `total`).

    Raises
    ------
    ValueError
        Unknown mode, or `'evaluate'` without configurations.
    FileNotFoundError
        `'shrinkage'` or `'check'` without `start` nor a `best.json` in the folder.
    NothingToTune
        No cycle to tune, or no cut could be evaluated; nothing is written.
    """
    began = time.perf_counter()
    if mode not in list(PHASES) + ['evaluate']:
        raise ValueError('Unknown mode {!r}, expected one of {}'.format(mode, list(PHASES) + ['evaluate']))
    if mode == 'evaluate' and not configs:
        raise ValueError('The evaluate mode needs configurations')
    cycles = search_cycles(scope) if cycles is None else cycles
    if not any(cycles.get(partition) for partition in ('search', 'check')):
        raise NothingToTune('no cycle to tune')

    fs, folder = tuning_fs(out, scope)
    if mode == 'evaluate':
        cut_days = 7 if cut_days is None else cut_days
        result = run_tuning(scope, configs, cycles=cycles, cut_days=cut_days, horizons=horizons, runner=runner,
                            verbose=verbose)
        result |= {'trials': pd.DataFrame(columns=TRIAL_COLUMNS), 'configs': [c.get('label') for c in configs],
                   'cycles': cycles, 'cut_days': cut_days, 'rounds': None}
    else:
        if start is None and mode in ('shrinkage', 'check'):
            path = results_path(folder, 'best.json')
            if not fs.exists(path):
                raise FileNotFoundError('no best.json in {}: run the previous phase or pass start'.format(
                    describe_source(fs, folder)))
            start = json.loads(fs.read_bytes(path).decode('utf-8'))
        result = run_search(scope, phase=mode, start=start, grid=grid or GRID, rounds=rounds, cut_days=cut_days,
                            cycles=cycles, horizons=horizons, runner=runner, verbose=verbose)
    evaluated = result['cuts'].loc[result['cuts']['error'].isnull()] if not result['cuts'].empty else result['cuts']
    if evaluated.empty:
        raise NothingToTune('no cut could be evaluated')
    seconds = {'run': time.perf_counter() - began}

    for name in ('cuts', 'polls', 'scores', 'trials'):
        fs.write_csv(result[name], results_path(folder, '{}.csv'.format(name)))
    best = None
    if mode != 'evaluate':
        fs.write_csv(result['trials'], results_path(folder, 'trials-{}.csv'.format(mode)))
        best = result['best']
    if mode in ('smoothing', 'shrinkage'):
        best = best | {'house_effects': True, 'dispersion': True, 'herding': True}
        fs.write_bytes(json.dumps(best, indent=1).encode('utf-8'), results_path(folder, 'best.json'))

    if prov is None:
        from .publish.common import provenance as prov
    provenance = prov()
    record = {
        'run_at': datetime.now().isoformat(timespec='seconds'), 'commit': provenance['commit'],
        'dirty': provenance.get('dirty'), 'scope': scope, 'mode': mode, 'configs': result['configs'],
        'cycles': result['cycles'], 'cut_days': result['cut_days'], 'horizons': [int(k) for k in horizons],
        'rounds': result['rounds'], 'db_polls': None, 'db_last_poll': None, **(stats or poll_stats)(scope),
        'seconds': round(seconds['run'], 1)
    }
    fs.write_bytes(json.dumps(record, indent=1).encode('utf-8'), results_path(folder, 'run.json'))
    seconds['write'] = time.perf_counter() - began - seconds['run']
    seconds['total'] = time.perf_counter() - began

    figures = {'n_configs': len(result['configs']),
               'n_cuts': int(evaluated[['event_date', 'cut']].drop_duplicates().shape[0]),
               'cycles': result['cycles'], 'folder': describe_source(fs, folder), 'seconds': seconds}
    if best is not None:
        figures['best'] = best

    return figures
