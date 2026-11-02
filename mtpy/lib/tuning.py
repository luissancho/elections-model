"""
Tuning of the polling average on future polls.

The average is tuned by how well a cut of the model predicts the polls published in the days after it: the
average read at the date of each future poll, plus the house effect of its pollster, against the figure the
poll published. This module holds the pure functions of that harness and its orchestration: the pure functions
read no database; `future_polls` and `run_cut` read it (read-only).
"""
import time
import warnings

from typing import Any, Optional

import numpy as np
import pandas as pd

from .computer import Computer
from .data import get_poll_series
from .forecaster import Forecaster
from .mtypes import DROP_MTYPES
from .simulator import Simulator
from .utils import group_results

PREDICTION_COLUMNS = ['date', 'pollster_id', 'pollster', 'sample_size', 'party', 'published', 'predicted', 'residual']
# Columns of a future poll that identify and describe it, besides its published figures (see `published_blocks`)
POLL_COLUMNS = ['date', 'pollster_id', 'sponsor_id', 'pollster', 'sample_size']


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
    verbose: int = 0
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
            weight_overrides=overrides, verbose=verbose
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
