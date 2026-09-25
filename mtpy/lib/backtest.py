"""
Backtest of the model on past elections.

For each election with official results and each horizon `d` (days before the election), the model is
frozen at `limit_date = election - d`: only polls published at least `d` days before the election enter
the average, the pollster ratings and the estimators of the `Computer` use previous elections only (by
construction of `Simulator`), and the forecast and the seat simulation are compared with the official
results. The errors of the individual polls stored by `Computer` are the backtest of the pollsters; this
module is the backtest of the model itself: the poll average, its intervals, the seats and the majority
probabilities. Each case is simulated twice: as a nowcast (the election held at the anchor of the forecast,
only the polling error) and at the true horizon (adding the drift of the opinion up to the election, columns
with the suffix `_h`).
"""
import time
import warnings

from typing import Any, Optional

import numpy as np
import pandas as pd

from .data import get_event_results
from .simulator import Simulator

DEFAULT_EVENTS = ['2015-12-20', '2016-06-26', '2019-04-28', '2019-11-10', '2023-07-23']
DEFAULT_HORIZONS = [6, 14, 30, 60, 90, 180]
LEVELS = [0.5, 0.8, 0.95]


# --- Metrics ----------------------------------------------------------------------------------------

def crps_from_samples(samples: np.ndarray, y: float) -> float:
    """
    Continuous ranked probability score of an empirical distribution: `E|X - y| - E|X - X'| / 2`.
    Lower is better; it equals the absolute error when the distribution is a point mass.
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    if n == 0:
        return np.nan

    term1 = np.mean(np.abs(x - y))
    # E|X - X'| for the empirical distribution, via the sorted sample: 2 / n^2 * sum_i (2i - n - 1) x_i
    i = np.arange(1, n + 1)
    term2 = 2. * np.sum((2 * i - n - 1) * x) / (n * n)

    return float(term1 - term2 / 2.)


def brier(p: float, y: bool | int) -> float:
    """Brier score of a probability `p` for a binary outcome `y`."""
    return float((p - float(y)) ** 2)


def log_score(p: float, y: bool | int, eps: float = 1e-3) -> float:
    """Negative log-likelihood of the outcome (probabilities clipped to `[eps, 1 - eps]`)."""
    p = float(np.clip(p, eps, 1 - eps))

    return float(-np.log(p if y else 1 - p))


def covered(lo: float, hi: float, y: float) -> bool:
    return bool(np.isfinite(lo) and np.isfinite(hi) and lo <= y <= hi)


def official_results(scope: str, event_date: str) -> pd.DataFrame:
    """
    Official national results of an election: `pct` (over valid votes) and `seats` per party.
    """
    df = get_event_results(scope, [event_date])
    df = df.loc[(df['region_id'].astype(int) == 0) & (df['party_id'] > 0)]

    return df.set_index('party')[['pct', 'seats']].astype(float)


# --- One case ---------------------------------------------------------------------------------------

def _quantile_cols(x: np.ndarray, suffix: str = '') -> dict[str, float]:
    """Bounds of the central intervals of `LEVELS` of a sample, as `lo{level}` / `hi{level}` columns."""
    out = {}
    for level in LEVELS:
        lo, hi = np.nanquantile(x, [(1 - level) / 2, 1 - (1 - level) / 2])
        out['lo{}{}'.format(int(level * 100), suffix)] = lo
        out['hi{}{}'.format(int(level * 100), suffix)] = hi

    return out


def _collect(sim: Simulator, vs: dict[str, Any]) -> dict[str, Any]:
    """Outputs of one run of the simulator used by the backtest."""
    return {
        'summary': sim.summary(), 'shares': sim.shares(), 'dist': sim.dist(),
        'probs': sim.probabilities(vs, majority=sim.n_seats // 2 + 1)
    }


def run_case(
    scope: str,
    event_date: str,
    horizon: int,
    n_sim: int = 500,
    seed: int = 42,
    max_fc: int = 10,
    min_polls: int = 5,
    nowcast_only: bool = False,
    house_effects: bool = True,
    industry_bias: bool = False,
    verbose: int = 0
) -> dict[str, pd.DataFrame]:
    """
    Freeze the model `horizon` days before `event_date` and compare it with the official results.
    Cases with fewer than `min_polls` usable polls are rejected (`ValueError`).

    Two runs of the simulator share the fit and the seed: the nowcast (the election held at the anchor
    `as_of`, only the polling error: columns without suffix) and, unless `nowcast_only`, the forecast at
    the true horizon (`horizon='deadline'`: the deadline of a past election is the election itself, so the
    drift of the opinion from `as_of` to the election is added: columns with the suffix `_h`).

    `house_effects` and `industry_bias` are passed to the `Simulator` (M6): the effects of the cycle are
    estimated with the polls up to `limit_date` only, and their prior with the elections before `event_date`;
    the baselines are computed on the raw polls.

    Returns
    -------
    dict
        `shares` (one row per party: forecast, intervals, baselines and official share), `seats` (one row
        per party: headline, intervals, CRPS and official seats), `blocks` (one row per block of the `vs`
        map: probability of absolute majority and outcome) and `meta` (one row).
    """
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sim = Simulator(
            scope=scope, event_date=event_date, drange=horizon, seed=seed, verbose=verbose,
            house_effects=house_effects, industry_bias=industry_bias
        )
        sim.fit_forecast(names=sim.params['names'], max_fc=max_fc, fillna=True)
        vs = sim.event_params['bmaps']['vs']

        sim.run(split=True, random=True, n_sim=n_sim)
        now = _collect(sim, vs)

        fwd = None
        if not nowcast_only:
            sim.run(split=True, random=True, n_sim=n_sim, horizon='deadline')
            fwd = _collect(sim, vs)

    names = sim.params['names']
    official = official_results(scope, event_date).reindex(names)
    # Baselines are computed on the raw polls: the house effects correction must not leak into them
    raw = sim.model.series_raw if sim.model.series_raw is not None else sim.model.series
    polls = raw.loc[raw['pollster'].notnull()].reset_index(raw.index.names[1:])
    if polls.shape[0] < min_polls:
        raise ValueError('Only {} polls published at least {} days before {} (minimum {})'.format(
            polls.shape[0], horizon, event_date, min_polls
        ))
    main = set(sim.event_params['bmaps'].get('main', []))

    # Baselines: last published poll, simple mean of the polls of the last four weeks, previous election
    last_date = polls.index.max()
    last_poll = polls.loc[[last_date], names].mean()
    mean_4w = polls.loc[polls.index >= last_date - pd.Timedelta(days=28), names].mean()
    prev = sim.prev_results['pct'].loc[sim.default_region].reindex(names)

    summary = now['summary']

    rows = []
    for n in names:
        fc_stat = sim.model.fc_stat[n].loc[sim.as_of]
        fc_stat = fc_stat if isinstance(fc_stat, dict) else {}
        row = {
            'event_date': event_date, 'horizon': horizon, 'party': n, 'main': n in main,
            'official': official.loc[n, 'pct'],
            'model': summary.loc[n, 'pct'],
            'fc_lo': fc_stat.get('cmin', np.nan), 'fc_hi': fc_stat.get('cmax', np.nan),
            'last_poll': last_poll.get(n, np.nan), 'mean_4w': mean_4w.get(n, np.nan), 'prev_result': prev.get(n, np.nan)
        }
        row.update(_quantile_cols(now['shares'][n].to_numpy(dtype=float)))
        if fwd is not None:
            row.update(_quantile_cols(fwd['shares'][n].to_numpy(dtype=float), suffix='_h'))
        rows.append(row)
    shares_df = pd.DataFrame(rows)

    # Seats: parties whose vote has nowhere to go (no previous result nor smap rule) make the seat forecast
    # of the event meaningless; flag them
    orphans = [
        n for n in names
        if summary.loc[n, 'pct'] >= 1 and np.isnan(prev.get(n, np.nan)) and len(sim.smap.get(n, [])) == 0
    ]
    rows = []
    for n in names:
        x = now['dist'][n].to_numpy(dtype=float)
        y = official.loc[n, 'seats']
        row = {
            'event_date': event_date, 'horizon': horizon, 'party': n, 'main': n in main,
            'official': y, 'model': summary.loc[n, 'seats'], 'mean': summary.loc[n, 'seats_mean'],
            'crps': crps_from_samples(x, y) if np.isfinite(y) else np.nan,
            'seats_valid': len(orphans) == 0
        }
        row.update(_quantile_cols(x))
        if fwd is not None:
            xh = fwd['dist'][n].to_numpy(dtype=float)
            row['model_h'] = fwd['summary'].loc[n, 'seats']
            row['mean_h'] = fwd['summary'].loc[n, 'seats_mean']
            row['crps_h'] = crps_from_samples(xh, y) if np.isfinite(y) else np.nan
            row.update(_quantile_cols(xh, suffix='_h'))
        rows.append(row)
    seats_df = pd.DataFrame(rows)

    majority = sim.n_seats // 2 + 1
    rows = []
    for block, parties in vs.items():
        parties = [parties] if isinstance(parties, str) else list(parties)
        y_seats = official.loc[[p for p in parties if p in official.index], 'seats'].sum()
        y = bool(y_seats >= majority)
        p = float(now['probs'].get(block, np.nan))
        row = {
            'event_date': event_date, 'horizon': horizon, 'block': block, 'parties': '+'.join(parties),
            'official_seats': y_seats, 'model_seats': float(summary.loc[[p for p in parties if p in summary.index], 'seats'].sum()),
            'p_majority': p, 'majority': y,
            'brier': brier(p, y) if np.isfinite(p) else np.nan, 'log_score': log_score(p, y) if np.isfinite(p) else np.nan,
            'seats_valid': len(orphans) == 0
        }
        if fwd is not None:
            ph = float(fwd['probs'].get(block, np.nan))
            row['model_seats_h'] = float(fwd['summary'].loc[[p for p in parties if p in fwd['summary'].index], 'seats'].sum())
            row['p_majority_h'] = ph
            row['brier_h'] = brier(ph, y) if np.isfinite(ph) else np.nan
            row['log_score_h'] = log_score(ph, y) if np.isfinite(ph) else np.nan
        rows.append(row)
    blocks_df = pd.DataFrame(rows)

    he = sim.model.house_effects
    he_main = he.loc[he.index.get_level_values('name').isin(names)] if he is not None else None
    # Houses informed by the cycle (enough polls: finite deviation error); the others sit at their prior
    he_active = he_main.loc[np.isfinite(he_main['dev_err'].astype(float))] if he_main is not None else None

    meta = pd.DataFrame([{
        'event_date': event_date, 'horizon': horizon, 'limit_date': sim.limit_date, 'prev_date': sim.prev_date,
        'as_of': str(sim.as_of.date()), 'horizon_max': sim.horizon_max,
        'drift_k': sim.v2drift.k if sim.v2drift is not None else np.nan,
        'house_effects': bool(house_effects), 'industry_bias': bool(industry_bias),
        'he_pollsters': int(he_active.index.get_level_values('pollster_id').nunique()) if he_active is not None else 0,
        'he_mean_abs': float(he_active['effect'].abs().mean()) if he_active is not None and len(he_active) else np.nan,
        'he_max_abs': float(he_active['effect'].abs().max()) if he_active is not None and len(he_active) else np.nan,
        'n_polls': int(polls.shape[0]), 'last_poll': str(last_date.date()), 'n_parties': len(names),
        'orphans': '+'.join(orphans), 'n_sim': n_sim, 'seed': seed, 'seconds': round(time.time() - t0, 1)
    }])

    return {'shares': shares_df, 'seats': seats_df, 'blocks': blocks_df, 'meta': meta}


# --- Aggregation ------------------------------------------------------------------------------------

def _mae(df: pd.DataFrame, col: str) -> float:
    d = (df[col] - df['official']).abs()

    return float(d.mean()) if d.notnull().any() else np.nan


def summarize_case(shares: pd.DataFrame, seats: pd.DataFrame, blocks: pd.DataFrame) -> dict[str, Any]:
    """
    Metrics of one (event, horizon) case, over the main parties for the shares and the seats.
    """
    s = shares.loc[shares['main']] if shares['main'].any() else shares
    t = seats.loc[seats['main']] if seats['main'].any() else seats
    valid = bool(seats['seats_valid'].all()) if len(seats) else False
    # Horizon run (columns with the suffix `_h`): optional, see `run_case`
    has_h = 'lo95_h' in shares.columns and 'lo95_h' in seats.columns
    has_h_blocks = 'brier_h' in blocks.columns

    out = {
        'mae_shares': _mae(s, 'model'),
        'mae_last_poll': _mae(s, 'last_poll'),
        'mae_mean_4w': _mae(s, 'mean_4w'),
        'mae_prev_result': _mae(s, 'prev_result'),
        'rmse_shares': float(np.sqrt(((s['model'] - s['official']) ** 2).mean())),
        'bias_sum': float((s['model'] - s['official']).sum()),  # signed error of the sum of the main parties
        'cov_fc95': float(np.mean([covered(a, b, y) for a, b, y in zip(s['fc_lo'], s['fc_hi'], s['official'])])),
    }
    for level in LEVELS:
        lo, hi = 'lo{}'.format(int(level * 100)), 'hi{}'.format(int(level * 100))
        out['cov_shares{}'.format(int(level * 100))] = float(np.mean([covered(a, b, y) for a, b, y in zip(s[lo], s[hi], s['official'])]))
        out['cov_seats{}'.format(int(level * 100))] = float(np.mean([covered(a, b, y) for a, b, y in zip(t[lo], t[hi], t['official'])])) if valid else np.nan

    out['mae_seats'] = _mae(t, 'model') if valid else np.nan
    out['mae_seats_all'] = _mae(seats, 'model') if valid else np.nan
    out['crps_seats'] = float(t['crps'].mean()) if valid else np.nan
    out['brier_vs'] = float(blocks['brier'].mean()) if valid and len(blocks) else np.nan
    out['log_score_vs'] = float(blocks['log_score'].mean()) if valid and len(blocks) else np.nan

    for level in LEVELS:
        lo, hi = 'lo{}_h'.format(int(level * 100)), 'hi{}_h'.format(int(level * 100))
        out['cov_shares{}_h'.format(int(level * 100))] = float(np.mean([covered(a, b, y) for a, b, y in zip(s[lo], s[hi], s['official'])])) if has_h else np.nan
        out['cov_seats{}_h'.format(int(level * 100))] = float(np.mean([covered(a, b, y) for a, b, y in zip(t[lo], t[hi], t['official'])])) if valid and has_h else np.nan
    out['mae_seats_h'] = _mae(t, 'model_h') if valid and has_h else np.nan
    out['crps_seats_h'] = float(t['crps_h'].mean()) if valid and has_h else np.nan
    out['brier_vs_h'] = float(blocks['brier_h'].mean()) if valid and has_h_blocks and len(blocks) else np.nan
    out['log_score_vs_h'] = float(blocks['log_score_h'].mean()) if valid and has_h_blocks and len(blocks) else np.nan
    out['seats_valid'] = valid

    return out


def run_backtest(
    scope: str = 'es',
    events: Optional[list[str]] = None,
    horizons: Optional[list[int]] = None,
    n_sim: int = 500,
    seed: int = 42,
    max_fc: int = 10,
    min_polls: int = 5,
    nowcast_only: bool = False,
    house_effects: bool = True,
    industry_bias: bool = False,
    verbose: int = 0
) -> dict[str, pd.DataFrame]:
    """
    Run every (event, horizon) case and aggregate the metrics (see `run_case` for the two runs per case).

    Returns
    -------
    dict
        `shares`, `seats`, `blocks`, `meta` (concatenated per case), `metrics` (one row per case) and
        `by_horizon` (means over the events, per horizon).
    """
    events = events or DEFAULT_EVENTS
    horizons = horizons or DEFAULT_HORIZONS

    parts = {'shares': [], 'seats': [], 'blocks': [], 'meta': []}
    metrics = []
    for event_date in events:
        for horizon in horizons:
            if verbose > 0:
                print('Backtest {} at {} days...'.format(event_date, horizon))

            try:
                case = run_case(
                    scope, event_date, horizon, n_sim=n_sim, seed=seed, max_fc=max_fc, min_polls=min_polls,
                    nowcast_only=nowcast_only, house_effects=house_effects, industry_bias=industry_bias
                )
            except Exception as e:
                # A case without usable polls (e.g. a 6-month cycle at a 180-day horizon) is recorded, not fatal
                warnings.warn('Backtest {} at {} days skipped: {}'.format(event_date, horizon, e))
                parts['meta'].append(pd.DataFrame([{'event_date': event_date, 'horizon': horizon, 'error': str(e)}]))
                continue

            for key in parts:
                parts[key].append(case[key])
            metrics.append({'event_date': event_date, 'horizon': horizon} | summarize_case(case['shares'], case['seats'], case['blocks']))

    out = {key: pd.concat(frames, ignore_index=True) for key, frames in parts.items()}
    out['metrics'] = pd.DataFrame(metrics)
    numeric = [c for c in out['metrics'].columns if c not in ('event_date', 'horizon', 'seats_valid')]
    by_horizon = out['metrics'].groupby('horizon')[numeric].mean()
    by_horizon.insert(0, 'n_cases', out['metrics'].groupby('horizon')['mae_shares'].count())
    by_horizon.insert(1, 'n_seats_cases', out['metrics'].groupby('horizon')['mae_seats'].count())
    out['by_horizon'] = by_horizon.reset_index()

    return out
