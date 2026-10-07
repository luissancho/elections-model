from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.stats import norm
from tqdm import tqdm
import warnings

from typing import Any, Literal, Optional
from typing_extensions import Self

from ..core.app import Core
from ..core.utils.helpers import apply_agg_func, format_number
from ..core.utils.dates import ts_from_delta
from ..core.utils.stat import LeastSquaresEstimator, LocalKernelEstimator, Stat
from ..core.utils.dataviz import (
    adjust_figure, build_color_seq_map, create_figure, get_color, get_df_styler, get_params, plot_diverging,
    plot_figure, plot_scatter, plot_scores, plot_series, print_styler, set_note, set_title, table_styles
)

from .data import (
    get_event_dates, get_event_series, get_poll_series, get_parties, get_pollsters,
    get_next_event_date, get_ratings, save_model_data, save_ratings_data, get_event_params,
    get_drift, save_drift_data, get_house_effects, save_house_effects_data, get_event_results, get_scopes,
    get_herding, save_herding_data,
    get_scope_parent, get_event_data
)
from .utils import (
    build_blocks, group_results, norm_range, partial_parties, REGIONAL_LIST_ID
)


class DriftEstimator:
    """
    Drift of the poll average with the horizon, modelled as a relative random walk: after `d` days the share of
    a party at level `p` has moved by `sigma(d, p) = p · sqrt(k · d)` percentage points (`var = k · d · p²`),
    the same relative drift for every party. Measured over every party polled in the Spanish cycles, the drift
    is proportional to the level (elasticity of the variance on the level ≈ 2): a regional party at 1 % moves a
    few hundredths of a point in a month, a party at 30 % moves a couple of points.

    `k` is the weighted geometric mean of `rms² / (d · level²)` over the rows of the drift table with
    `horizon >= d_min`, with weights `n · decay^years` (increments, discounted by the age of the cycle relative
    to the most recent one). The geometric mean is the typical established party: the parties born or dissolved
    within a cycle (VOX in 2019, Cs in 2015) drift several times more and dominate the quadratic mean
    (`agg='quadratic'`: the weighted mean of `rms² / (d · level²)`, the variance a random party experiences).
    Shorter horizons are excluded because the increments of a kernel-smoothed series understate the movement
    there (the fitted series is locally linear, so their variance grows like `d²`). The decay matters: the
    cycles of the two-party era (up to 2011) drift less than the fragmented ones since 2015.
    `curve` keeps the weighted geometric mean of `rms / level` by horizon (all rows).

    Young parties (M8): when the table carries the `age` of the party at each election (years since its first
    poll), a second constant `k_young` is fitted on the rows with `age < age_max` and the drift of a young party
    is multiplied by `multiplier = sqrt(k_young / k)` (never below 1): parties less than four years old drift
    about three times more than the established ones, and the effect vanishes afterwards.
    """

    def __init__(
        self,
        k: float,
        d_min: int = 60,
        decay: Optional[float] = None,
        agg: Literal['geometric', 'quadratic'] = 'geometric',
        curve: Optional[pd.Series] = None,
        k_young: float = np.nan,
        age_max: float = 4.,
        n_young: int = 0
    ) -> None:
        self.k = float(k)
        self.d_min = int(d_min)
        self.decay = decay
        self.agg = agg
        self.curve = curve
        self.k_young = float(k_young)
        self.age_max = float(age_max)
        self.n_young = int(n_young)
        ratio = self.k_young / self.k if (np.isfinite(self.k_young) and np.isfinite(self.k) and self.k > 0) else np.nan
        self.multiplier = float(np.sqrt(ratio)) if np.isfinite(ratio) and ratio > 1 else 1.

    @classmethod
    def fit(
        cls,
        data: pd.DataFrame,
        d_min: int = 60,
        decay: Optional[float] = None,
        agg: Literal['geometric', 'quadratic'] = 'geometric',
        age_max: float = 4.,
        min_young: int = 6
    ) -> 'DriftEstimator':
        """
        Fit the relative random-walk constant from a drift table with columns `horizon`, `level`, `n` and
        `rms` (and `event_date` when `decay` is given).

        Parameters
        ----------
        data : pd.DataFrame
            Drift table (see `Computer.get_drift_data`).
        d_min : int, optional
            Minimum horizon (days) of the rows used in the fit.
        decay : float, optional
            Yearly decay of the weight of a cycle with its age, relative to the most recent cycle of the table
            (`None`: every cycle weighs the same).
        agg : {'geometric', 'quadratic'}, optional
            Mean of the relative drift across rows: geometric (the typical established party, default) or
            quadratic (the variance a random party experiences, dominated by the parties born within a cycle).
        age_max : float, optional
            Parties younger than this (years, column `age`) are fitted apart (`k_young`).
        min_young : int, optional
            Young rows (at `d_min` or more) needed to fit `k_young`; otherwise the multiplier is 1 with a warning.
        """
        if agg not in ('geometric', 'quadratic'):
            raise ValueError("`agg` must be 'geometric' or 'quadratic'")

        df = data.dropna(subset=['horizon', 'level', 'n', 'rms'])
        df = df.loc[(df['n'] > 0) & (df['level'] > 0) & (df['rms'] > 0)]

        weight = df['n'].astype(float)
        if decay is not None and df.shape[0] > 0:
            dates = pd.to_datetime(df['event_date'])
            years = (dates.max() - dates).dt.days / 365.25
            weight = weight * np.power(float(decay), years)

        horizon = df['horizon'].astype(int)
        relative = df['rms'].astype(float) / df['level'].astype(float)  # relative drift of each row

        # Transform in which the rows are averaged (log for the geometric mean, square for the quadratic one)
        fwd, inv = (np.log, np.exp) if agg == 'geometric' else (np.square, np.sqrt)

        if df.shape[0] > 0:
            means = (weight * fwd(relative)).groupby(horizon).sum() / weight.groupby(horizon).sum()
            curve = inv(means).rename('relative')
        else:
            curve = pd.Series(dtype=float, name='relative')
        curve.index.name = 'horizon'

        def fit_k(mask):
            # rows of sqrt(k): relative / sqrt(d), averaged in the transformed scale
            z = fwd(relative[mask] / np.sqrt(horizon[mask].astype(float)))
            return float(inv((weight[mask] * z).sum() / weight[mask].sum()) ** 2)

        # Young parties (age below `age_max` at that election) are fitted apart; rows without age are established
        age = df['age'].astype(float) if 'age' in df.columns else pd.Series(np.nan, index=df.index)
        if 'age' in df.columns and df.shape[0] > 0 and age.isnull().all():
            warnings.warn('The `age` column is missing for every row: no age multiplier (check the party names)')
        young = age.notnull() & (age < age_max)
        rows = (horizon >= d_min) & ~young
        if rows.sum() == 0:
            warnings.warn('No drift data at horizons of {} days or more: the drift is set to 0'.format(d_min))
            k = np.nan
        else:
            k = fit_k(rows)

        rows_young = (horizon >= d_min) & young
        n_young = int(rows_young.sum())
        if young.any() and n_young < min_young:
            warnings.warn('Only {} drift rows of parties younger than {} years at {} days or more: no age multiplier'.format(
                n_young, age_max, d_min
            ))
        k_young = fit_k(rows_young) if n_young >= min_young else np.nan

        return cls(k, d_min=d_min, decay=decay, agg=agg, curve=curve, k_young=k_young, age_max=age_max, n_young=n_young)

    def var(
        self,
        d: int | float | np.ndarray,
        level: float | np.ndarray | pd.Series = 1.,
        age: Optional[float | np.ndarray | pd.Series] = None
    ) -> float | np.ndarray:
        """
        Variance of the drift after `d` days of a share at `level` (0 when the constant could not be fitted);
        parties younger than `age_max` (years, `age`) get `multiplier²` times more. A missing age counts as
        established.
        """
        days = np.clip(np.asarray(d, dtype=float), 0, None)
        var = (self.k if np.isfinite(self.k) else 0.) * days * np.square(np.asarray(level, dtype=float))
        if age is not None and self.multiplier > 1:
            a = np.asarray(age, dtype=float)
            var = var * np.where(np.isfinite(a) & (a < self.age_max), np.square(self.multiplier), 1.)

        return float(var) if np.ndim(var) == 0 else var

    def sigma(
        self,
        d: int | float | np.ndarray,
        level: float | np.ndarray | pd.Series = 1.,
        age: Optional[float | np.ndarray | pd.Series] = None
    ) -> float | np.ndarray:
        """
        Standard deviation of the drift after `d` days of a share at `level`, in percentage points
        (`level · sqrt(k · d)`, times `multiplier` for a party younger than `age_max`).
        """
        return np.sqrt(self.var(d, level, age=age))


class SwingNoise:
    """
    Deviations of the provinces from the proportional swing between two elections (M9). The relative residual
    `log(actual / (previous · national ratio))` of a party in a province splits into the mean of its autonomous
    community (`reg_mean`: the regional swing, 86 % of the variance) and the deviation within it (`within`).
    Both variances decrease with the predicted level `L` of the party in the province as `a + b / L`
    (a sampling-like term plus a floor), fitted by weighted least squares over level bins.

    `sigma_region(level)` and `sigma_province(level)` return the relative standard deviations used to draw the
    multiplicative shocks in `Simulator.build_umat`.
    """

    def __init__(
        self,
        a_r: float = 0.,
        b_r: float = 0.,
        a_p: float = 0.,
        b_p: float = 0.,
        n: int = 0,
        level_floor: float = 1.
    ) -> None:
        self.a_r, self.b_r, self.a_p, self.b_p = float(a_r), float(b_r), float(a_p), float(b_p)
        self.n = int(n)
        self.level_floor = float(level_floor)

    @staticmethod
    def decompose(residuals: pd.DataFrame) -> pd.DataFrame:
        """
        Split the residual `logres` of each (pair, party, province) row into the mean of its autonomous community
        (`reg_mean`, over the provinces of the same `pair`, `party` and `group`) and the deviation within it
        (`within`), and add `k` (provinces of that community for the pair and party) and `n_groups`
        (communities where the party has rows in the pair). Both counts feed the unbiased fit.
        """
        df = residuals.copy()
        if df.shape[0] == 0:
            df['reg_mean'] = pd.Series(dtype=float)
            df['within'] = pd.Series(dtype=float)
            df['k'] = pd.Series(dtype=int)
            df['n_groups'] = pd.Series(dtype=int)
            return df

        g = df.groupby(['pair', 'party', 'group'], dropna=False)['logres']
        df['reg_mean'] = g.transform('mean')
        df['within'] = df['logres'] - df['reg_mean']
        df['k'] = g.transform('size').astype(int)
        df['n_groups'] = df.groupby(['pair', 'party'])['group'].transform('nunique').astype(int)

        return df

    @staticmethod
    def wls_nonneg(L, y, n) -> tuple[float, float]:
        """
        Weighted least squares of `y = a + b / L` with weights `n` and `a, b ≥ 0`. With two parameters the
        constrained optimum is the free fit when feasible, or the best of the two boundary fits (`a = 0` with
        `b` refitted through the origin, `b = 0` with `a` the weighted mean) and `(0, 0)`: the free fit is never
        clipped coefficient by coefficient, which would leave the other one unadjusted.
        """
        x = 1. / np.asarray(L, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(n, dtype=float)
        sw = np.sqrt(w)
        X = np.column_stack([np.ones(x.shape[0]), x])
        a, b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if a >= 0 and b >= 0:
            return float(a), float(b)

        candidates = [
            (0., max(float((w * y * x).sum() / (w * x * x).sum()), 0.)),
            (max(float((w * y).sum() / w.sum()), 0.), 0.),
            (0., 0.)
        ]
        sse = [float((w * (y - (a_ + b_ * x)) ** 2).sum()) for a_, b_ in candidates]

        return candidates[int(np.argmin(sse))]

    @classmethod
    def fit(
        cls,
        residuals: pd.DataFrame,
        bins: tuple[float, ...] = (2., 3., 5., 10., 20.),
        min_pairs: int = 3,
        level_floor: float = 1.
    ) -> 'SwingNoise':
        """
        Fit the level curves from a residuals table with columns `pair`, `level`, `reg_mean` and `within`
        (see `decompose` and `Computer.get_swing_residuals`), plus `k` and `n_groups` when available.

        The deviation within a community of `k` provinces has variance `σ_w² (1 − 1/k)` (zero in the
        uniprovincial ones), so the provincial curve is fitted on the communities with `k ≥ 2` with `within`
        scaled by `√(k / (k − 1))`; the community mean carries `σ_w² / k` of that provincial noise, which is
        subtracted (with the fitted provincial curve) before fitting the regional one. Parties present in a
        single community are left out of the regional fit: their community mean is zero by construction (the
        national ratio is their own). Without `k` the correction is skipped (a table with `reg_mean` and
        `within` drawn directly).

        Parameters
        ----------
        residuals : pd.DataFrame
            One row per election pair, party and province.
        bins : tuple of float, optional
            Inner edges of the level bins (percentage points).
        min_pairs : int, optional
            Election pairs needed; otherwise no noise, with a warning.
        level_floor : float, optional
            Lower bound of the level in the curves.
        """
        df = residuals.dropna(subset=['level', 'reg_mean', 'within'])
        if df['pair'].nunique() < min_pairs:
            warnings.warn('Swing residuals of {} election pairs (minimum {}): no regional noise'.format(df['pair'].nunique(), min_pairs))
            return cls(level_floor=level_floor)

        level = df['level'].to_numpy(dtype=float)
        k = df['k'].to_numpy(dtype=float) if 'k' in df.columns else np.full(df.shape[0], np.inf)
        n_groups = df['n_groups'].to_numpy(dtype=float) if 'n_groups' in df.columns else np.full(df.shape[0], 2.)
        edges = [0.] + list(bins) + [np.inf]
        bin_of = pd.cut(df['level'], bins=edges, labels=False).to_numpy()

        def binned(mask, values, correction=None):
            d = pd.DataFrame({'bin': bin_of[mask], 'L': level[mask], 'v': values[mask], 'c': 0. if correction is None else correction[mask]})
            agg = d.groupby('bin').agg(n=('v', 'size'), L=('L', 'mean'), var=('v', 'var'), c=('c', 'mean'))
            agg['var'] = agg['var'] - agg['c']
            return agg.loc[agg['n'] >= 3].dropna()

        finite_k = np.isfinite(k)
        k_safe = np.where(finite_k, np.maximum(k, 1.), 2.)   # only used where `k` is finite
        within_adj = df['within'].to_numpy(dtype=float) * np.where(finite_k, np.sqrt(k_safe / np.maximum(k_safe - 1., 1.)), 1.)
        agg_p = binned(k >= 2, within_adj)
        if agg_p.shape[0] < 2:
            warnings.warn('Not enough level bins in the swing residuals: no regional noise')
            return cls(level_floor=level_floor)
        a_p, b_p = cls.wls_nonneg(agg_p['L'], agg_p['var'], agg_p['n'])

        var_p = a_p + b_p / np.clip(level, level_floor, None)
        agg_r = binned(n_groups >= 2, df['reg_mean'].to_numpy(dtype=float), np.where(finite_k, var_p / k_safe, 0.))
        if agg_r.shape[0] < 2:
            warnings.warn('Not enough level bins in the regional swing residuals: no regional noise')
            return cls(level_floor=level_floor)
        a_r, b_r = cls.wls_nonneg(agg_r['L'], agg_r['var'], agg_r['n'])

        return cls(a_r, b_r, a_p, b_p, n=int(df.shape[0]), level_floor=level_floor)

    def _var(self, a: float, b: float, level) -> float | np.ndarray:
        lvl = np.clip(np.asarray(level, dtype=float), self.level_floor, None)
        var = np.clip(a + b / lvl, 0., None)

        return float(var) if np.ndim(var) == 0 else var

    def var_region(self, level) -> float | np.ndarray:
        """Variance of the regional (autonomous community) relative shock at a given level."""
        return self._var(self.a_r, self.b_r, level)

    def var_province(self, level) -> float | np.ndarray:
        """Variance of the province relative shock, net of the regional one, at a given level."""
        return self._var(self.a_p, self.b_p, level)

    def sigma_region(self, level) -> float | np.ndarray:
        return np.sqrt(self.var_region(level))

    def sigma_province(self, level) -> float | np.ndarray:
        return np.sqrt(self.var_province(level))


class Computer(Core):

    def __init__(
        self,
        scope: str,
        event_dates: Optional[list[str]] = None,
        drop_mtypes: Optional[list[str]] = ['aggr', 'online'],
        drop_ctypes: Optional[list[str]] = ['wban', 'exit'],
        drange: Optional[tuple[int, int] | int] = None,
        n_last: Optional[int] = None,
        alpha: float = .05,
        reg_params: Optional[dict[str, Any]] = None,
        ol_dev: Optional[int] = 3,
        ol_max: Optional[int] = 5000,
        wspan: Optional[int] = 4,
        pos_decay: Optional[float] = .5,
        week_decay: Optional[float] = .7,
        year_decay: Optional[float] = .9,
        bias_dev_tau: Optional[float] = .01,
        min_polls: Optional[int] = 3,
        ratings_margin: Optional[float] = .05,
        error_weights: Optional[dict[str, float]] = {'blocks': .7, 'within': .3},
        verbose: int = 0,
        path: Optional[str] = None
    ) -> None:
        """
        Events and polls computer.

        Calculates parameters and errors made by electoral polls in the past, with the goal of
        generating a ranking and weighting system to apply to each poll based on
        its polling firm, sample size, proximity to election date, etc.

        Parameters
        ----------
        scope : str
            Scope of the election events.
        event_dates : list of str, optional
            List of dates of the election events.
        drop_mtypes : list of str, optional
            Drop the polls published by pollsters whose methodology type is in the list.
        drop_ctypes : list of str, optional
            Drop the polls published by pollsters whose context type is in the list.
        drange : tuple of int or int, optional
            Only the polls published within the specified days range before the event will be included.
            If an integer is provided, it will be converted to (`drange`, None), meaning that only polls published
            more than `drange` days before the event will be included.
        n_last : int, optional
            Number of last polls for each pollster and event to use.
        alpha : float, optional
            Confidence interval.
        reg_params : dict of str, optional
            Parameters for the regression estimator.
            If `None`, the default parameters will be used.
        ol_dev : int, optional
            Number of standard deviations used to winsorize the sample size of each poll and prevent outliers.
        ol_max : int, optional
            Maximum sample size permitted for each poll, because once a certain size is reached,
            the sampling error does not decrease significantly with increasing observations.
        wspan : int, optional
            Window span (in days) to use in order to prevent too many polls published by a pollster in a short period.
        pos_decay : float, optional
            Position decay.
        week_decay : float, optional
            Week decay.
        year_decay : float, optional
            Year decay.
        bias_dev_tau : float, optional
            The standard deviation prior used to compute the bias deviation of each poll.
        min_polls : int, optional
            Minimum number of polls to compute the rating.
        ratings_margin : float, optional
            The margin to be applied to the ratings around the [0, 1] range.
        error_weights : dict, optional
            Weights for our custom error metric.
        verbose : int, optional
            Level of verbosity.
        path : str, optional
            Path where the model outputs (forecasts, figures) are stored, relative to the app file system root
            (`files/`). Input data (`params.json`, maps) is always read from the versioned `data/` directory.
        """
        super().__init__()

        self.scope = scope
        self.event_dates = event_dates or get_event_dates(scope=scope)
        self.drop_mtypes = drop_mtypes if drop_mtypes is not None else list()
        self.drop_ctypes = drop_ctypes if drop_ctypes is not None else list()

        self.drange = drange
        self.n_last = n_last
        self.alpha = alpha
        self.reg_params = self.set_reg_params(reg_params)
        self.ol_dev = ol_dev
        self.ol_max = ol_max
        self.wspan = wspan
        self.pos_decay = pos_decay
        self.week_decay = week_decay
        self.year_decay = year_decay
        self.bias_dev_tau = bias_dev_tau
        self.min_polls = min_polls
        self.ratings_margin = ratings_margin
        self.error_weights = error_weights

        self.verbose = verbose  # Print progress
        self.path = path or '.'  # Path to the model files, relative to the app file system root (files/)

        self.event_params = get_event_params(
            scope=self.scope,
            event_dates=self.event_dates,
            path=self.path
        )

        self.params = None  # Parameters for the computer model
        self.names = None  # List of party names included
        self.parties = None  # List of parties included in the polls published for this election event
        self.pollsters = None  # List of pollsters with polls published for this election event

        self.keys = ['event_date', 'date', 'pollster_id', 'sponsor_id']
        self.series = None  # DataFrame to build containing the series of polls, weights and results
        self.errors = None
        self.biases = None
        self.ratings = None
        self.drift = None  # Drift table of the current events (see `get_drift_data`)
        self.house_effects = None  # House effects table of the current events (see `get_house_effects_data`)
        self.herding = None  # Herding table of the current events (see `get_herding_data`)

        self.seats_estimator = None
        self.error_estimator = None
        self.bias_estimator = None

    @property
    def events(self) -> pd.DataFrame:
        """
        Returns all the events final results.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the events final results.
        """
        return self.series.loc[self.series.pollster.isnull()].droplevel(self.keys[1:])

    @property
    def polls(self) -> pd.DataFrame:
        """
        Returns all the polls predictions.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the polls predictions.
        """
        return self.series.loc[self.series.pollster.notnull()]

    def regional_flags(self) -> pd.Series:
        """
        `regional` flag of every party (1: it only runs in part of the territory of the scope), indexed by
        name. In the national scope it is the curated flag of the parties table. Within an autonomous
        community it is derived from the results by district of the current events: a party is regional when,
        in the last election it ran, its districts held less than half of the valid votes of the community
        (see `partial_parties`: UPL or Por Ávila in Castilla y León, the parties of a single island).
        """
        flags = get_parties().set_index('name')['regional'].astype(int)
        if get_scope_parent(self.scope) is None:
            return flags

        flags[:] = 0
        dates = list(self.event_dates)
        res = get_event_results(self.scope, dates)
        res = res.loc[(res['party_id'] > 0) & (res['region_id'] > 0) & (res['region_id'] != REGIONAL_LIST_ID)]
        data = get_event_data(self.scope, dates)
        for date, g in res.groupby('date', sort=True):
            pct = g.pivot_table(index='region_id', columns='party', values='pct', aggfunc='sum', observed=True)
            votes = data.loc[data['date'] == date].set_index('region_id')['votes'].astype(float)
            partial = partial_parties(pct, votes)
            for name in pct.columns:
                if name in flags.index and pct[name].fillna(0).gt(0).any():
                    flags[name] = int(name in partial)

        return flags

    def merge_bmaps(
        self,
        name: str
    ) -> dict:
        """
        Given a bmap name, returns a dictionary with the parties
        that compose each block in every event.

        Parameters
        ----------
        name : str
            Name of the bmap to merge.

        Returns
        -------
        dict
            A dictionary with the merged bmaps.
        """
        bm = {}

        for event_date in self.event_params.keys():
            map = self.event_params[event_date]['bmaps'].get(name)
            if map is None:
                continue
            if isinstance(map, (tuple, list)):
                map = dict(zip(map, map))

            for party, block in map.items():
                # Normalize to a fresh list so that a single party is not iterated character by character
                # and the event params are never mutated
                block = [block] if isinstance(block, str) else list(block)
                if party not in bm:
                    bm[party] = block
                else:
                    for b in block:
                        if b not in bm[party]:
                            bm[party].append(b)

        return bm

    def set_reg_params(
        self,
        reg_params: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Normalize parameters for the LocalKernelEstimator.

        Parameters
        ----------
        reg_params : dict, optional
            Parameters for the LocalKernelEstimator.

        Returns
        -------
        dict
            Parameters for the LocalKernelEstimator.
        """
        reg_params = reg_params if reg_params is not None else dict()
        reg_params = {
            'kernel': reg_params.get('kernel', 'gaussian'),  # Kernel used to ponderate the observations
            'bw_type': reg_params.get('bw_type', 'adaptive'),  # Fixed or adaptive kernel bandwidth
            'bw': reg_params.get('bw', 'isj'),  # Default kernel bandwidth or name of bandwidth selection method
            'bw_kwargs': reg_params.get('bw_kwargs', {  # Adaptive kernel bandwidth selection parameters
                'n_iter': 3,  # Number of iterations
                'min_delta': None  # Minimum change in bandwidth estimate to be achieved in order to stop iterations
            }),
            'poly_deg': reg_params.get('poly_deg', 1),  # Degree of the polynomial used to fit the local data
            'cov_type': reg_params.get('cov_type', 'hac'),  # Type of robust covariance estimator
            'cov_kwargs': reg_params.get('cov_kwargs', {  # Robust covariance estimator parameters
                'hac_lags': 1,  # Number of lags used to compute the HAC estimator
                'kernel': 'bartlett'  # Kernel used to compute the HAC estimator
            })
        }

        return reg_params

    def filter_polls(
        self,
        featured: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Filter polls based on the given parameters.

        Parameters
        ----------
        featured : bool, optional
            Whether to filter the polls in featured events and have been computed.
        **kwargs
            Keyword arguments to overwrite the default filter parameters:
            - drange : tuple of int or int
            - n_last : int
            - drop_mtypes : list of str
            - drop_ctypes : list of str

        Returns
        -------
        pd.DataFrame
            A DataFrame with the filtered polls.
        """
        drange = kwargs['drange'] if 'drange' in kwargs else self.drange
        drange = norm_range(drange, int(self.series.days.max()))

        n_last = kwargs['n_last'] if 'n_last' in kwargs else self.n_last
        n_last = n_last if n_last is not None else 0

        drop_mtypes = kwargs['drop_mtypes'] if 'drop_mtypes' in kwargs else self.drop_mtypes
        drop_mtypes = drop_mtypes if drop_mtypes is not None else list()

        drop_ctypes = kwargs['drop_ctypes'] if 'drop_ctypes' in kwargs else self.drop_ctypes
        drop_ctypes = drop_ctypes if drop_ctypes is not None else list()

        df = self.series.loc[
            (
                self.series.pollster.notnull()
            ) & (
                ~self.series.mtype.isin(drop_mtypes)
            ) & (
                ~self.series.ctype.isin(drop_ctypes)
            ) & (
                self.series.weight_over > 0
            ) & (
                self.series.days.between(*drange)
            )
        ]

        if featured:
            df = df.loc[df['featured'] & df['computed']]

        if n_last > 0:
            df = df.groupby(['event_date', 'pollster_id']).tail(n_last)

        return df

    def load_events(
        self,
        metric: Literal['pct', 'votes', 'seats'] = 'pct'
    ) -> pd.DataFrame:
        """
        Load the events final percentage results for each party.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the events final percentage results for each party.
        """
        events = get_event_series(
            scope=self.scope,
            event_dates=self.event_dates,
            metric=metric
        )
        events['event_date'] = events['date']
        events['sample_size'] = events['votes']
        events['computed'] = events['featured']

        return events.set_index('date').sort_index()

    def load_polls(
        self,
        metric: Literal['pct', 'votes', 'seats'] = 'pct'
    ) -> pd.DataFrame:
        """
        Load the polls predicted percentage results for each party.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the polls predicted percentage results for each party.
        """
        polls = get_poll_series(
            scope=self.scope,
            event_dates=self.event_dates,
            metric=metric
        )

        return polls.set_index(self.keys).sort_index()

    def load_errors(self) -> pd.DataFrame:
        """
        Load the polls predicted errors for each party.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the polls predicted errors for each party.
        """
        errors = get_poll_series(
            scope=self.scope,
            event_dates=self.event_dates,
            metric='error'
        )

        return errors.set_index(self.keys).sort_index()

    def load_biases(self) -> pd.DataFrame:
        """
        Load the polls predicted biases for each party.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the polls predicted biases for each party.
        """
        biases = get_poll_series(
            scope=self.scope,
            event_dates=self.event_dates,
            metric='bias'
        )

        return biases.set_index(self.keys).sort_index()

    def load_ratings(self) -> pd.DataFrame:
        """
        Load all the pollsters ratings.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the pollsters ratings.
        """
        ratings = get_ratings(
            scope=self.scope,
            event_dates=self.event_dates
        )

        return ratings.set_index(['event_date', 'pollster_id']).sort_index()

    def build_series(self) -> Self:
        """
        Build the series of polls, weights and results for the selected election events.
        """
        if self.verbose > 0:
            print('Load events...')

        events = self.load_events()

        if self.verbose > 0:
            print('Load polls...')

        polls = self.load_polls()

        if self.verbose > 0:
            print('Load errors...')

        self.errors = self.load_errors()

        if self.verbose > 0:
            print('Load biases...')

        self.biases = self.load_biases()

        if self.verbose > 0:
            print('Load ratings...')

        self.ratings = self.load_ratings()

        if self.verbose > 0:
            print('Load parties...')

        self.parties = get_parties()

        if self.verbose > 0:
            print('Load pollsters...')

        self.pollsters = get_pollsters()

        if self.verbose > 0:
            print('Build series...')

        data_cols = [
            'start_date', 'end_date', 'pollster', 'sponsor',
            'mtype', 'ctype', 'computed', 'featured',
            'sample_size', 'parties', 'days', 'proc_sample', 'rating',
            'error_avg', 'error_blocks', 'error_within', 'bias_avg', 'bias_blocks', 'bias_within', 'bias',
            'bias_dev_adj', 'bias_dev_err', 'weight_sample', 'weight_over', 'weight_rating'
        ]

        # Initialize series by concatenating events and polls
        self.series = pd.concat([
            polls.reset_index(),
            events.reset_index()
        ], ignore_index=True, sort=False)

        # Filter pollsters and parties that have polls published
        self.pollsters = self.pollsters[self.pollsters.name.isin(self.series.pollster.unique())]
        self.parties = self.parties[self.parties.name.isin(self.series.columns)]
        self.names = self.parties.name.tolist()

        # Sort index and columns
        self.series = self.series.set_index(self.keys).sort_index()[
            data_cols + self.names
        ]

        # Check for parties that don't have any errors computed (not participated in any final event)
        # but are present in the polls, in order to prevent missing names errors
        self.errors = self.errors.reindex(columns=self.names)

        return self

    def compute_weights(
        self,
        save: bool = False,
        overwrite: bool = False
    ) -> pd.DataFrame:
        """
        Compute the weights that later will be used to fit the forecast model.

        Here we compute the following weight components for each poll prediction:
            - mtype
                The pollster's metodology type.
            - proc_sample
                The sample size of each poll, after being processed to prevent missing values and outliers.
                We try to fill missing values using this criteria:
                    - First, fill with the median sample size of the polls performed by the same pollster at the same event.
                    - If no polls found, fill with the median sample size of the polls performed by the same pollster at any event.
                    - If no polls found, fill with the median sample size of all existing polls.
                Then, winsorize the sample sizes to prevent outliers.
            - weight_sample
                The weight of the poll based on its sample size.
                We try to give more weight to the polls with a larger sample size, and to do so we compute
                the square root of the ratio between each poll's sample size and the median sample size of
                all the polls published for the same event.
            - weight_over
                The weight of the poll based on the overlap/overflow of each pollster's polls.
                The goal is to take the following into account:
                    - Polls that overlap with subsequent polls should be discarded.
                        Removes partial results from trackings and incremental polls that are updated every day.
                        In these cases keep only the last results published.
                    - Polls published in a short period of time should be assigned a lower weight.
                        Prevents giving too much weight to pollsters trying to flood the market with their polls.
                        If N polls are published within a `wspan` days range, a weight of 1/N will be assigned to each poll.
            - weight_rating
                The weight of the poll based on the pollster's rating.
                We try to give more weight to the polls from pollsters with a higher rating.
                To see how the rating and its associated weights are computed, check `compute_ratings`.

        Parameters
        ----------
        save : bool, optional
            Whether to save the data to the database.
        overwrite : bool, optional
            Whether to force and overwrite the computation of the parameters for the polls that have already been computed.

        Returns
        -------
        pd.DataFrame
            A table containing the computed values of each poll.
        """
        columns = [
            'mtype', 'proc_sample', 'weight_sample', 'weight_over', 'weight_rating'
        ]
        df = self.polls.sort_index().reset_index().drop(columns=columns)

        if not overwrite:
            df = df.loc[~df['computed']]

        if df.empty:
            return df

        if self.verbose > 0:
            print('Compute mtype...')

        # For each poll, assign its pollster's metodology type
        df['mtype'] = df['pollster'].map(self.pollsters.set_index('name')['mtype'])

        if self.verbose > 0:
            print('Compute sample...')

        # We need to fix the sample size of the polls that didn't report it
        df['sample_size'] = df['sample_size'].where(df['sample_size'] > 0, np.nan)
        # First, we compute the median sample size of the polls published by the same pollster for the same event
        msizes_loc = df.groupby(['event_date', 'pollster'])['sample_size'].median().dropna()
        # If there are no polls published by the same pollster for the same event, we compute the median sample size
        # of the polls published by the same pollster for any event
        msizes_tot = df.groupby('pollster')['sample_size'].median().dropna()
        # Finally, if there are no polls published by the same pollster, we compute the median sample size of all polls
        msizes_all = df.loc[~df['mtype'].isin(self.drop_mtypes)]['sample_size'].median()

        # First, fill missing values with the median sample size computed above
        proc_sample = df['sample_size'].fillna(
            df[['event_date', 'pollster']].apply(tuple, axis=1).map(msizes_loc)
        ).fillna(
            df['pollster'].map(msizes_tot)
        ).fillna(
            msizes_all
        )
        # Then, winsorize the sample sizes to prevent outliers
        df['proc_sample'] = Stat(
            proc_sample.clip(0, self.ol_max),
            dropna=True,
            outliers='group',
            n_dev=self.ol_dev
        ).data.round().astype(int)

        # Compute the weight based on the sample size of the polls, using the square root of the ratio between
        # each poll's sample size and the median sample size of all the polls published for the same event.
        sample_medians = df.groupby('event_date')['proc_sample'].median()
        df['weight_sample'] = df.apply(
            lambda x: np.sqrt(x['proc_sample'] / sample_medians.loc[x['event_date']]),
            axis=1
        ).round(2).astype(float)

        if self.verbose > 0:
            print('Compute overweights...')

        # Compute the days range between the poll start/end date and the final event date
        df['wrange'] = df.apply(lambda dr: (
            (dr['start_date'] - dr['event_date']).days,
            (dr['end_date'] - dr['event_date']).days
        ), axis=1)
        df['wtype'] = pd.factorize(df[['mtype', 'ctype']].astype(str).apply('-'.join, axis=1))[0]
        # Group polls by pollster/sponsor and compute the overlap/overflow weight for each group's polls
        df = df.merge(
            apply_agg_func(
                df,
                by=['event_date', 'pollster_id', 'wtype'],
                func=self.poll_overweights,
                columns='wrange',
                sort=['date', 'sponsor_id']
            ).round(2).astype(float).rename('weight_over').reset_index(),
            on=['event_date', 'pollster_id', 'wtype', 'date', 'sponsor_id'],
            how='left'
        )

        if self.verbose > 0:
            print('Merge ratings...')

        # Set the weight based on the pollster's rating
        df = df.merge(
            self.ratings['weight_rating'].round(2).astype(float),
            left_on=['event_date', 'pollster_id'],
            right_index=True,
            how='left'
        )

        if self.verbose > 0:
            print('Process data...')

        # Set the flag to indicate that the computation is done for each poll
        df['computed'] = True
        df['featured'] = df['event_date'].map(self.events['featured'])

        df = df.set_index(self.keys)[columns + ['computed', 'featured']].sort_index()

        if save:
            if self.verbose > 0:
                print('Save polls data...')

            # Add the common missing `event_scope` index to the polls DataFrame and save data
            data = pd.concat([df], keys=[self.scope], names=['event_scope'] + self.keys)
            nrows = save_model_data('Polls', data)

            if self.verbose > 0:
                print('{} rows updated...'.format(nrows))

        # Update series
        self.series[columns + ['computed', 'featured']] = df.reindex(self.series.index)

        return df

    def compute_errors(
        self,
        save: bool = False
    ) -> pd.DataFrame:
        """
        Compute the error/bias of each poll prediction against the final event results.

        For each poll, the following errors/deviations are computed:
            - error_avg
                The average error over the most important parties for each event (usually predicted by all pollsters).
                These are defined at the event parameters `main` bmap.
            - error_blocks
                The error/bias of the poll prediction over the margin gap between the two main blocks of parties.
                These are defined at the event parameters `vs` bmap.
            - error_within
                The part of the error of the parties of the `vs` blocks that only redistributes votes within a
                block (percentage points, see `block_error_decomposition`).
            - bias_avg
                The average bias over the most important parties for each event (usually predicted by all pollsters).
                These are defined at the event parameters `main` bmap.
            - bias_blocks
                The bias of the poll prediction over the two main blocks of parties (log-odds, absolute).
            - bias_within
                The bias in the split of each block between its parties (see `within_block_bias`): the error
                that only redistributes votes between allies.
            - bias
                The mix that feeds the pollster ratings: `error_weights` over `bias_blocks` and `bias_within`
                (`{'blocks': .7, 'within': .3}` by default, M6b).

        Parameters
        ----------
        save : bool, optional
            Whether to save the data to the database.

        Returns
        -------
        pd.DataFrame
            A table containing the computed values of each poll.
        """
        columns = ['error_avg', 'error_blocks', 'error_within', 'bias_avg', 'bias_blocks', 'bias_within', 'bias']
        df = self.polls.drop(columns=columns)
        results = self.events.drop(columns=columns)

        if self.verbose > 0:
            print('Compute percentage errors over all parties...')

        errors_pct = self.poll_errors(polls=df, events=results, error_type='pct')

        if self.verbose > 0:
            print('Compute biases from log odds ratios over all parties...')

        errors_lor = self.poll_errors(polls=df, events=results, error_type='lor')

        if self.verbose > 0:
            print('Compute percentage errors over main parties...')

        errors_pct_main = self.poll_errors(polls=df, events=results, error_type='pct', bmap='main')

        if self.verbose > 0:
            print('Compute biases from log odds ratios over main parties...')

        errors_lor_main = self.poll_errors(polls=df, events=results, error_type='lor', bmap='main')

        if self.verbose > 0:
            print('Compute percentage errors over main blocks gap...')

        errors_gap_vs = self.poll_errors(polls=df, events=results, error_type='gap', bmap='vs')

        if self.verbose > 0:
            print('Compute biases from log odds ratios over main blocks...')

        errors_lor_vs = self.poll_errors(polls=df, events=results, error_type='lor', bmap='vs')

        if self.verbose > 0:
            print('Build individual biases...')

        pmap = self.parties.set_index('name').id.to_dict()
        rkeys = self.keys + ['party_id']

        error = errors_pct['error'].rename(columns=pmap).stack().rename_axis(rkeys).rename('error').mul(100)
        bias = errors_lor['error'].rename(columns=pmap).stack().rename_axis(rkeys).rename('bias').apply(self.lor_to_bias)

        dr = pd.concat([error, bias], axis=1).sort_index().round(2).astype(float)

        if self.verbose > 0:
            print('Build aggregated biases...')

        error_avg = errors_pct_main.apply(lambda x: Stat(
            x['error'].dropna().abs(),
            weights=x['event'].loc[x['error'].dropna().index]
        ).mean(), axis=1).rename('error_avg')
        error_blocks = errors_gap_vs['error', 'gap'].rename('error_blocks')
        bias_avg = errors_lor_main.apply(lambda x: Stat(
            x['error'].dropna().abs(),
            weights=x['event'].loc[x['error'].dropna().index]
        ).mean(), axis=1).rename('bias_avg')
        bias_blocks = errors_lor_vs.apply(lambda x: Stat(
            x['error'].dropna().abs(),
            weights=x['event'].loc[x['error'].dropna().index]
        ).mean(), axis=1).rename('bias_blocks')

        # Bias in the split of each `vs` block between its parties (log-odds, see `within_block_bias`), so
        # that a poll that gets the blocks right but swaps votes between allies is told apart
        bias_within = self.within_block_bias(
            errors_pct['poll'].mul(100), errors_pct['event'].mul(100), self.merge_bmaps('vs')
        )
        # The same split in percentage points: the error that redistributes votes within a block
        error_within = self.block_error_decomposition(errors_pct['error'].mul(100), self.merge_bmaps('vs'))['error_within']

        dp = pd.concat([error_avg, error_blocks, error_within, bias_avg, bias_blocks, bias_within], axis=1)

        if self.verbose > 0:
            print('Set computed biases...')

        bias_weights = pd.Series(self.error_weights).add_prefix('bias_')
        dp['bias'] = dp[bias_weights.index].apply(lambda x: Stat(x, weights=bias_weights).mean(), axis=1)

        if self.verbose > 0:
            print('Process data...')

        dp[['error_avg', 'error_blocks']] = dp[['error_avg', 'error_blocks']].mul(100)
        dp[['bias_avg', 'bias_blocks', 'bias_within', 'bias']] = dp[['bias_avg', 'bias_blocks', 'bias_within', 'bias']].apply(self.lor_to_bias)

        df = df.merge(
            dp,
            left_index=True, right_index=True, how='left'
        )[columns].sort_index().round(2).astype(float)

        if save:
            if self.verbose > 0:
                print('Save results data...')

            # Add the common missing `event_scope` index to the polls DataFrame and save data
            data = pd.concat([dr], keys=[self.scope], names=['event_scope'] + self.keys + ['party_id'])
            nrows = save_model_data('PollsResults', data)

            if self.verbose > 0:
                print('{} rows updated...'.format(nrows))

            if self.verbose > 0:
                print('Save polls data...')

            # Add the common missing `event_scope` index to the polls DataFrame and save data
            data = pd.concat([df], keys=[self.scope], names=['event_scope'] + self.keys)
            nrows = save_model_data('Polls', data)

            if self.verbose > 0:
                print('{} rows updated...'.format(nrows))

        # Update series
        self.series[columns] = df.reindex(self.series.index)

        return df
    
    def compute_deviations(
        self,
        save: bool = False
    ) -> pd.DataFrame:
        """
        Compute the deviations of each poll error/bias from the mean error/bias of all other polls
        with similar sample and time to the event.

        For each poll, we compute the difference between its bias and the mean bias of all other polls
        conducted for the same event and similar sample size and time before the event. This is done by
        fitting a local kernel regression, which generates a mean bias for each day and
        its corresponding standard error due to regression. The more polls are available for a given
        day, the more confident we are about the estimated mean bias and the smaller the standard error.

        Parameters
        ----------
        save : bool, optional
            Whether to save the data to the database.

        Returns
        -------
        pd.DataFrame
            A table containing the computed values of each poll.
        """
        columns = ['bias_dev_adj', 'bias_dev_err']
        df = self.polls.drop(columns=columns)

        if self.verbose > 0:
            print('Fit bias estimator...')

        # Convert bias to log odds ratio
        df['bias'] = df['bias'].apply(self.bias_to_lor)

        # Fit bias estimator
        dreg = self.fit_bias_estimator()

        if self.verbose > 0:
            print('Set bias deviations...')

        df = df.merge(
            dreg[['mean', 'err']].add_prefix('bias_'),
            left_index=True, right_index=True, how='left'
        )
        df['bias_dev'] = df['bias'] - df['bias_mean']

        if self.verbose > 0:
            print('Adjust bias deviations...')

        df[['bias_dev_adj', 'bias_dev_err']] = self.bayes_adjust(
            df[['bias_dev', 'bias_err']],
            params={
                'mean': 0,
                'std': self.bias_dev_tau
            }
        )

        if self.verbose > 0:
            print('Process data...')

        df = df[columns].sort_index().apply(self.lor_to_bias).round(2).astype(float)

        if save:
            if self.verbose > 0:
                print('Save polls data...')

            # Add the common missing `event_scope` index to the polls DataFrame and save data
            data = pd.concat([df], keys=[self.scope], names=['event_scope'] + self.keys)
            nrows = save_model_data('Polls', data)

            if self.verbose > 0:
                print('{} rows updated...'.format(nrows))

        # Update series
        self.series[columns] = df.reindex(self.series.index)

        return df

    def compute_ratings(
        self,
        save: bool = False,
        scopes: Optional[list[str] | str] = None,
        rating_weights: Optional[dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Compute the ratings of all the pollsters that have polls performed in the current events.

        Based on FiveThirtyEight's "Predictive Plus-Minus" measure, explained here:
        https://fivethirtyeight.com/methodology/how-our-pollster-ratings-work/

        Three main factors are considered in the rating computation for each pollster:
            - Quality
                A scoring value of the pollster's metodology and historical standards.
                It is set to be a number between 0 and 100, the greater this value, the more reliable the pollster.
            - Polls published
                The total number of polls published by each pollster, with weights exponentially decreasing with time,
                so we give more importance to the most recent polls.
            - Adjusted deviation
                The difference between the pollster's average error and the adjusted error, which is a weighted
                average of the average error of other polls published for the same event/week and the expected error
                of other polls with similar sample size and weeks before the event.

        The idea is to rate pollsters based on their historical accuracy compared to other pollsters. In order to find
        the relative error between pollsters, we compute a deviation from an average. For each poll, we compare
        its error with all the other polls published for the same event/week. If there are not enough polls to compare,
        we run a regression to estimate the expected error based on the sample size and the weeks before the event.

        error = poll error (the error in the margin gap between the two main blocks of parties)
        num_related = number of polls published for the same event/week
        error_related = average error of the polls published for the same event/week
        error_expected = expected error based on the sample size and the weeks before the event

        error_adjusted = (num_related * error_related + min_related * error_expected) / (num_related + min_related)
        dev_adjusted = WAVG(error) - WAVG(error_adjusted)

        Once we have the adjusted deviation of each pollster, we need to account for pollster with few polls published
        or with a low quality and methodology standards. In order to do so, we use bayesian statistics to estimate
        a prior mean and revert the adjusted deviation towards it. The lower the quality of the pollster and the fewer
        the polls published, the more weight we give to this prior mean.

        prior_weight = quality * num_polls / (num_polls + num_polls.mean())
        prior_mean = np.average(prior_weight, weights=num_polls) - prior_weight
        dev_reverted = prior_mean + (dev_adjusted - prior_mean) * prior_weight

        The resulting rating is the mean reverted deviation scaled to the range [5-95].

        The rating of an event can also learn from the polls of other scopes (M11): with `scopes`, the polls
        of every election held strictly before the event, in any of those scopes, enter the algorithm with
        their weight multiplied by the `rating_weight` of their scope (1 for the national one, 0.5 for the
        regional ones). The events rated are always those of the scope of this computer.

        Parameters
        ----------
        save : bool, optional
            Whether to save the data to the database.
        scopes : list of str or 'all', optional
            Scopes whose polls feed the ratings. `None` uses only the scope of this computer (weight 1);
            `'all'` takes every scope of the catalogue (`data/es-scopes.csv`) with a positive weight.
        rating_weights : dict, optional
            Weight of the polls of each scope, replacing those of the catalogue. Scopes with weight 0 are
            left out.

        Returns
        -------
        pd.DataFrame
            A table containing the computed values of each pollster.
        """
        columns = ['rating', 'weight_rating']
        rparams = [
            'quality', 'num_events', 'num_polls', 'num_polls_w',
            'error_avg', 'error_blocks', 'error_within', 'bias_avg', 'bias_blocks', 'bias_within', 'bias',
            'bias_dev_adj', 'bias_dev_err', 'rating_adj', 'rating', 'weight_rating'
        ]
        rkeys = ['event_date', 'pollster_id']

        if scopes is None:
            weights = {self.scope: 1.}
        else:
            weights = rating_weights if rating_weights is not None else get_scopes()['rating_weight'].to_dict()
            names = list(weights) if scopes == 'all' else list(scopes)
            weights = {s: float(weights.get(s, 0.)) for s in names}
            weights.setdefault(self.scope, 1.)
        weights = {s: w for s, w in weights.items() if w > 0 or s == self.scope}

        if self.verbose > 0:
            print('Compute ratings...')

        pool = self.rating_pool(list(weights))

        # Pollsters rated: those of this scope plus those with polls in the scopes that weigh
        pollsters = self.pollsters
        if len(weights) > 1:
            every = get_pollsters()
            rated = set(pollsters['name']) | set(pool['pollster'].dropna().astype(str).unique())
            pollsters = every.loc[every['name'].isin(rated)]

        # Events rated: those of this scope with polls, and its next one
        own = pool.xs(self.scope, level='event_scope').index.get_level_values('event_date').unique().sort_values()
        targets = list(own)
        if len(own) > 0:
            next_date = get_next_event_date(self.scope, date_from=own[-1])
            targets.append(pd.to_datetime(next_date) if next_date is not None else pd.NaT)

        dr = self.rate_events(pool, targets, weights, pollsters)

        if self.verbose > 0:
            print('Process data...')

        # Rescale absolute percentage errors to the range [0, 100]
        dr[[
            'quality', 'error_avg', 'error_blocks', 'error_within', 'rating_adj', 'rating'
        ]] = dr[[
            'quality', 'error_avg', 'error_blocks', 'error_within', 'rating_adj', 'rating'
        ]].mul(100)
        # Rescale log odds ratio deviations to the bias scale
        dr[[
            'bias_avg', 'bias_blocks', 'bias_within', 'bias', 'bias_dev_adj', 'bias_dev_err'
        ]] = dr[[
            'bias_avg', 'bias_blocks', 'bias_within', 'bias', 'bias_dev_adj', 'bias_dev_err'
        ]].apply(self.lor_to_bias)

        dr = dr.set_index(rkeys).sort_index().round(2).astype(float)

        df = self.polls.reset_index().drop(columns=columns).join(dr[columns], on=rkeys).set_index(self.keys)[columns].sort_index()

        if save:
            if self.verbose > 0:
                print('Save ratings data...')

            # Add the common missing `event_scope` index to the polls DataFrame and save data.
            # `pollsters.rating` keeps the last rating of the national scope only
            data = pd.concat([dr], keys=[self.scope], names=['event_scope'] + rkeys)
            nrows = save_ratings_data(data, update_pollsters=(self.scope == 'es'))

            if self.verbose > 0:
                print('{} rows updated...'.format(nrows))

            if self.verbose > 0:
                print('Save polls data...')

            # Add the common missing `event_scope` index to the polls DataFrame and save data
            data = pd.concat([df], keys=[self.scope], names=['event_scope'] + self.keys)
            nrows = save_model_data('Polls', data)

            if self.verbose > 0:
                print('{} rows updated...'.format(nrows))

        # Update pollsters with their last rating
        rlast = dr.groupby('pollster_id')['rating'].last()
        self.pollsters['rating'] = self.pollsters['id'].map(rlast).fillna(0).astype(int)

        # Update series
        self.series[columns] = df.reindex(self.series.index)
        # Update ratings: every row computed, also those of pollsters not stored yet for these events
        new_rows = dr.index.difference(self.ratings.index)
        self.ratings = self.ratings.reindex(self.ratings.index.union(dr.index))
        self.ratings[rparams] = dr.reindex(self.ratings.index)
        if len(new_rows) > 0 and 'pollster' in self.ratings.columns:
            names = pollsters.set_index('id')['name']
            self.ratings.loc[new_rows, 'pollster'] = new_rows.get_level_values('pollster_id').map(names)

        return df

    def rating_pool(
        self,
        scopes: list[str]
    ) -> pd.DataFrame:
        """
        Polls that feed the ratings: the filtered polls (`filter_polls`) of each scope, without the party
        columns, indexed by `event_scope` plus the keys of the series. The scale of the errors is the internal
        one of the rating algorithm ([0, 1] and log odds ratio).

        Parameters
        ----------
        scopes : list of str
            Scopes to gather; the one of this computer uses its own series, the others a computer of that
            scope with the same filters. Scopes without polls are skipped.
        """
        columns = ['rating', 'weight_rating']
        bias_cols = ['bias_avg', 'bias_blocks', 'bias_within', 'bias', 'bias_dev_adj', 'bias_dev_err']
        frames = {}

        for scope in scopes:
            if scope == self.scope:
                comp = self
            else:
                if len(get_event_dates(scope=scope, min_polls=1)) == 0:
                    continue
                comp = Computer(
                    scope=scope, drop_mtypes=self.drop_mtypes, drop_ctypes=self.drop_ctypes, drange=self.drange,
                    n_last=self.n_last, verbose=0, path=self.path
                ).build_series()

            polls = comp.filter_polls().drop(columns=columns + comp.names + ['-'], errors='ignore')
            # Scale absolute percentage errors to the range [0, 1]
            polls[['error_avg', 'error_blocks', 'error_within']] = polls[['error_avg', 'error_blocks', 'error_within']].div(100)
            # Scale bias deviations to the log odds ratio scale
            polls[bias_cols] = polls[bias_cols].apply(self.bias_to_lor)
            frames[scope] = polls

        return pd.concat(frames, names=['event_scope'] + self.keys)

    def rate_events(
        self,
        pool: pd.DataFrame,
        targets: list[pd.Timestamp],
        rating_weights: dict[str, float],
        pollsters: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Pollster ratings before each event of `targets`: the rating algorithm (`pollster_ratings`) over the
        polls of `pool` whose election was held strictly before the event, whatever their scope. Elections
        held on the same day do not see each other. Polls of scopes with weight 0 are dropped before any
        weight is computed (they would otherwise move the reference year of the time decay).

        Parameters
        ----------
        pool : pd.DataFrame
            Polls indexed by `event_scope` plus the keys of the series (see `rating_pool`).
        targets : list of pd.Timestamp
            Dates of the events to rate; `NaT` rates with every poll (no next event known).
        rating_weights : dict
            Weight of the polls of each scope.
        pollsters : pd.DataFrame
            Pollsters to rate (`id`, `name`, `quality`).

        Returns
        -------
        pd.DataFrame
            One row per event and pollster: `event_date`, `pollster_id` and the rating columns, in the
            internal scale. Events without previous polls are skipped.
        """
        rparams = [
            'quality', 'num_events', 'num_polls', 'num_polls_w',
            'error_avg', 'error_blocks', 'error_within', 'bias_avg', 'bias_blocks', 'bias_within', 'bias',
            'bias_dev_adj', 'bias_dev_err', 'rating_adj', 'rating', 'weight_rating'
        ]
        rkeys = ['event_date', 'pollster_id']
        frames = []

        scope_weights = pool.index.get_level_values('event_scope').map(lambda s: rating_weights.get(s, 1.))
        pool = pool.loc[np.asarray(scope_weights, dtype=float) > 0]
        dates = pool.index.get_level_values('event_date')

        for target in targets:
            if self.verbose > 0:
                print('Compute ratings for {}...'.format(target))

            iter_polls = pool if pd.isnull(target) else pool.loc[dates < pd.Timestamp(target)]
            if iter_polls.shape[0] == 0 or iter_polls['bias'].isnull().any():
                if self.verbose > 0:
                    print('No polls found, skipping...')

                continue

            ratings = self.pollster_ratings(iter_polls, rating_weights=rating_weights, pollsters=pollsters).reset_index()
            ratings['event_date'] = pd.to_datetime(target) if not pd.isnull(target) else pd.NaT
            ratings['pollster_id'] = ratings['pollster'].map(pollsters.set_index('name')['id'])
            frames.append(ratings[rkeys + rparams])

        if len(frames) == 0:
            return pd.DataFrame(columns=rkeys + rparams)

        return pd.concat(frames, ignore_index=True)

    def poll_errors(
        self,
        polls: pd.DataFrame,
        events: pd.DataFrame,
        error_type: Literal['pct', 'gap', 'lor'] = 'pct',
        bmap: Optional[dict[str, Any] | str] = None
    ) -> pd.DataFrame:
        """
        Poll errors evaluation.

        Parameters
        ----------
        polls : pd.DataFrame
            Polls to be evaluated.
        events : pd.DataFrame
            Actual results of the election events.
        error_type : Literal['pct', 'gap', 'lor'], optional
            The type of error to compute.
            - 'pct' : Percentage error
            - 'gap' : Error over the margin gap between the two main blocks of parties
            - 'lor' : The log odds ratio between the polls and the events
        bmap : dict, optional
            A dictionary mapping the parties and blocks to be included in the analysis.

        Returns
        -------
        pd.DataFrame
            A table containing the computed values of each poll.
        """
        bname = None
        if bmap is None:
            bmap = self.names
        elif isinstance(bmap, str):
            bname = bmap
            bmap = self.merge_bmaps(bname)
        parties = self.parties['name'].tolist()

        # In case we are evaluating the error over grouped blocks of parties, we need to build the blocks properly
        blocks = build_blocks(bmap, parties)
        names = blocks.index.tolist()

        # Group polls and events using the blocks built before
        polls = group_results(polls, blocks=blocks)
        events = group_results(events, blocks=blocks)

        if bname is not None:
            for dt in events.index.get_level_values(0).astype(str).unique().tolist():
                missing = [n for n in names if n not in list(self.event_params[dt]['bmaps'][bname])]
                events.loc[dt, missing] = np.nan
        
        # In case we are evaluating the error over the margin gap between the two main blocks of parties,
        # we need to compute the difference between the last and the first block of parties
        if error_type == 'gap':
            polls = polls.diff(axis=1).dropna(axis=1, how='all').rename(columns={names[-1]: 'gap'})
            events = events.diff(axis=1).dropna(axis=1, how='all').rename(columns={names[-1]: 'gap'})
            cols = ['gap']
        else:
            cols = names

        # Set the correct MultiIndex to the polls and events DataFrames
        polls = polls.set_axis(
            pd.MultiIndex.from_product([['poll'], cols]), axis=1
        )
        events = events.set_axis(
            pd.MultiIndex.from_product([['event'], cols]), axis=1
        )

        # Merge polls and events
        dmat = pd.merge(polls, events, left_index=True, right_index=True, how='left').astype(float).div(100)

        # Compute errors between polls and events for each poll and party
        if error_type == 'lor':
            # Compute the log odds ratio between the polls and the events
            derr = ((dmat['poll'] / (1 - dmat['poll'])) / (dmat['event'] / (1 - dmat['event']))).apply(np.log)
        else:
            # Compute the difference between the polls and the events
            derr = dmat['poll'] - dmat['event']

        # Merge these errors with the polls and events matrix
        dmat = pd.concat([
            dmat,
            derr.set_axis(pd.MultiIndex.from_product([['error'], cols]), axis=1)
        ], axis=1)

        return dmat

    def pollster_ratings(
        self,
        polls: pd.DataFrame,
        rating_weights: Optional[dict[str, float]] = None,
        pollsters: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Pollster rating algorithm.

        See `compute_ratings` for more details about the rating algorithm.

        Parameters
        ----------
        polls : pd.DataFrame
            Polls to be used in the rating algorithm.
        rating_weights : dict, optional
            Weight of the polls of each scope (`weight_scope`), when `polls` gathers several scopes (index
            level `event_scope`); 1 for every poll by default.
        pollsters : pd.DataFrame, optional
            Pollsters to rate (`name`, `quality`); those of the series by default.

        Returns
        -------
        pd.DataFrame
            A table containing the computed values of each pollster.
        """
        pollsters = self.pollsters if pollsters is None else pollsters
        df = polls[polls['bias'].notnull()]

        # Compute additional weights of each poll
        df[['weight_pos', 'weight_week', 'weight_year', 'weight_scope']] = self.poll_rating_weights(df, rating_weights)

        # Compute the final weight of each poll by multiplying all the other weights computed
        df['weight'] = df[[
            'weight_over', 'weight_sample', 'weight_pos', 'weight_week', 'weight_year', 'weight_scope'
        ]].prod(axis=1)

        # Remove polls with weight below the threshold
        df = df[df['weight'] >= 1e-2].reset_index()

        # Create a DataFrame to store the results, using the pollster names as index
        dr = pd.DataFrame(
            index=pollsters['name'].rename('pollster')
        )

        # Get the quality of each pollster, a scoring value of the pollster's metodology and historical standards
        # It is set to be a number between 0 and 100, the greater this value, the more reliable the pollster
        dr['quality'] = pollsters.set_index('name').quality.astype(float).div(100)

        # Get the total number of events concurred and polls published by each pollster (an event is a scope
        # and a date: two regional elections held on the same day count twice)
        # Pollsters without evaluable polls contribute zero evidence, so their rating equals their prior quality
        if 'event_scope' in df.columns:
            df['event_key'] = df['event_scope'].astype(str) + '|' + df['event_date'].astype(str)
        else:
            df['event_key'] = df['event_date']
        dr['num_events'] = df.groupby('pollster', observed=True)['event_key'].nunique().reindex(dr.index).fillna(0)
        dr['num_polls'] = df.groupby('pollster', observed=True)['event_date'].count().reindex(dr.index).fillna(0)
        # Weighted number of polls, giving more importance to the most recent polls
        dr['num_polls_w'] = df.groupby('pollster', observed=True)['weight'].sum().reindex(dr.index).fillna(0)

        # Compute the weighted mean of each pollster's poll errors
        for col in ['error_avg', 'error_blocks', 'error_within', 'bias_avg', 'bias_blocks', 'bias_within', 'bias']:
            dr[col] = df.groupby('pollster', observed=True).apply(
                lambda x: np.sum(x[col] * x['weight']) / np.sum(x['weight'])
            ).reindex(dr.index)

        # Compute the weighted mean of each pollster's poll deviations
        dr['bias_dev_adj'] = df.groupby('pollster', observed=True).apply(
            lambda x: np.sum(x['weight'] * x['bias_dev_adj']) / np.sum(x['weight'])
        ).reindex(dr.index)
        # Compute the weighted standard error of each pollster's poll deviations
        dr['bias_dev_err'] = df.groupby('pollster', observed=True).apply(
            lambda x: np.sqrt(np.sum(np.square(x['weight']) * np.square(x['bias_dev_err'])) / np.square(np.sum(x['weight'])))
        ).reindex(dr.index) * np.sqrt(
            (dr['num_polls_w'] + 1) / dr['num_polls_w']
        )

        # Adjust the bias deviations using bayesian statistics.
        # The goal of this adjustment is to shrink these deviations in cases where the error is high
        # due to the lack of polls available for a given day. In order to do so, we select a prior distribution
        # of the deviations and update it with the polls data to compute a posterior distribution.
        # By default, the prior distribution is assumed to be a normal distribution with a mean of 0
        # and a standard deviation of `bias_dev_tau` (see class parameters).
        dr[['bias_dev_adj', 'bias_dev_err']] = self.bayes_adjust(
            dr[['bias_dev_adj', 'bias_dev_err']],
            params={
                'mean': 0,
                'std': self.bias_dev_tau
            }
        )

        # Scale the bias deviations to the range [0, 1]
        dr['rating_adj'] = dr['bias_dev_adj'].mul(-1).apply(norm.cdf, scale=self.bias_dev_tau).fillna(dr['quality'])
        # Compute each pollster's final rating, using a weighted average of the adjusted bias deviations
        # and the prior quality assigned to the pollster.
        # The more polls published by the pollster, the more weight is given to the adjusted bias deviations,
        # while pollsters with a low number of polls are given more weight to the prior quality.
        dr['rating'] = (
            (dr['num_polls_w'] * dr['rating_adj']) + (self.min_polls * dr['quality'])
        ) / (
            dr['num_polls_w'] + self.min_polls
        )
        # Compute the weight that each pollster's predictions will have in the polling average forecaster.
        # The weight is computed using the logarithm of the rating, which gives more weight to pollsters with a
        # rating close to 1 and less weight to pollsters with a rating close to 0.
        dr['weight_rating'] = np.log1p(dr['rating']) / np.log1p(dr['rating'].mean())

        dr = dr.sort_values('rating', ascending=False)

        return dr
    
    def poll_rating_weights(
        self,
        polls: pd.DataFrame,
        rating_weights: Optional[dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Compute additional weights needed to compute pollster ratings and an estimation of their relative deviation
        from the expected bias of polls published by other pollsters in the same event and week.

        Parameters
        ----------
        polls : pd.DataFrame, optional
            The polls to get the rating for. When they gather several scopes (index level `event_scope`), an
            event is a scope and a date.
        rating_weights : dict, optional
            Weight of the polls of each scope (`weight_scope`); 1 by default.

        Returns
        -------
        pd.DataFrame
            The polls data with the new weights and deviations.
        """
        df = polls.reset_index()
        if 'event_scope' not in df.columns:
            df['event_scope'] = self.scope
        rating_weights = rating_weights if rating_weights is not None else {}

        # We use all polls published by each pollster, but give more weight to the ones published closer to the event
        df['seq_pos'] = df.groupby(['event_date', 'event_scope', 'pollster_id'], observed=True)['date'].cumcount(ascending=False)
        df['weight_pos'] = df['seq_pos'].apply(lambda x: np.power(self.pos_decay, x))

        # We give more weight to the polls published closer to the event
        df['event_dlimits'] = df.groupby(['event_date', 'event_scope'], observed=True)['days'].transform('min')
        df['weeks'] = ((df['days'] - df['event_dlimits'] + 1) // 7).clip(0)
        df['weight_week'] = np.power(self.week_decay, df['weeks'])

        # Compute the weight of each poll based on the number of years between the poll and the most recent event
        df['event_year'] = df['event_date'].dt.strftime('%Y').astype(int)
        df['years'] = df['event_year'].max() - df['event_year']
        df['weight_year'] = np.power(self.year_decay, df['years'])

        # Weight of the scope of the poll: regional elections inform the rating less than national ones
        df['weight_scope'] = df['event_scope'].map(lambda x: rating_weights.get(x, 1.)).astype(float)

        df = df.set_index(list(polls.index.names))[[
            'weight_pos', 'weight_week', 'weight_year', 'weight_scope'
        ]].round(2).astype(float)

        return df

    def lor_to_bias(
        self,
        x: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Convert from log odds ratio to bias.

        Parameters
        ----------
        x : float | np.ndarray
            The log odds ratio to convert.

        Returns
        -------
        float
            The bias.
        """
        x = np.array(x)

        return (np.exp(x) - 1) * 100

    def bias_to_lor(
        self,
        x: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Convert from bias to log odds ratio.

        Parameters
        ----------
        x : float | np.ndarray
            The bias to convert.

        Returns
        -------
        float
            The log odds ratio.
        """
        x = np.array(x)

        return np.log((x / 100) + 1)

    def poll_overweights(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Compute the overlap/overflow weight of a set of polls.

        Parameters
        ----------
        x : np.ndarray
            Polls series to be computed.

        Returns
        -------
        np.ndarray
            The overlap/overflow weight of the polls.
        """
        x = np.array(x)
        n = len(x)

        # All polls are initially assigned a weight of 1
        w = np.ones(n)

        # Sort polls by their end date
        sind = np.argsort(x[:, 1])
        sx = x[sind]

        # Discard polls that overlap with subsequent polls
        for i in range(n):
            end_i = sx[i, 1]
            for j in range(i + 1, n):
                start_j = sx[j, 0]
                if start_j <= end_i:
                    w[sind[i]] = 0
                    break

        # Polls published in a short period of time should be assigned a lower weight
        if self.wspan is not None:
            for i in range(n):
                if w[sind[i]] > 0:
                    start, end = sx[i]
                    # Find the polls within the window span before and after the current poll
                    n_prev = np.sum((sx[:i, 1] >= end - self.wspan) & (w[sind[:i]] > 0))
                    n_post = np.sum((sx[i + 1:, 1] <= end + self.wspan) & (sx[i + 1:, 0] > end) & (w[sind[i + 1:]] > 0))
                    # Assign a weight of 1 / (N + 1) to the current poll
                    w[sind[i]] = 1 / (n_prev + n_post + 1)

        return w

    def bayes_adjust(
        self,
        x: np.ndarray,
        params: Optional[dict[str, Any] | list[float]] = None
    ) -> np.ndarray:
        """
        Adjust the bias deviations of each poll from the mean bias of all other polls using bayesian statistics.

        The goal of this adjustment is to shrink these deviations in cases where the error is high
        due to the lack of polls available for a given day. In order to do so, we select a prior distribution
        of the deviations and update it with the polls data to compute a posterior distribution.

        By default, the prior distribution is assumed to be a normal distribution with a mean of 0
        and a standard deviation of `bias_dev_tau` (see class parameters).

        Parameters
        ----------
        data : np.ndarray
            Each row represents a poll and contains the mean (estimation) and standard deviation (error)
            of the bias deviation from the estimated mean bias.
        params : dict or list, optional
            Parameters for the bayesian prior distribution.

        Returns
        -------
        np.ndarray
            The adjusted mean (updated estimation) and standard deviation (updated error) of the polls.
        """
        x = np.array(x)

        # Get the mean, standard deviation and lambda parameters of the polls
        data_mean = x[:, 0]
        data_std = x[:, 1]
        data_lambda = 1 / np.square(data_std)

        # Get the parameters of the prior distribution
        dparams = {
            'mean': 0,
            'std': self.bias_dev_tau
        }
        if params is None:
            prior_mean = dparams['mean']
            prior_std = dparams['std']
        elif isinstance(params, dict):
            prior_mean = params.get('mean', dparams['mean'])
            prior_std = params.get('std', dparams['std'])
        elif isinstance(params, list):
            prior_mean = params[0] if len(params) > 0 else dparams['mean']
            prior_std = params[1] if len(params) > 1 else dparams['std']

        # Get the lambda parameter of the prior distribution
        prior_lambda = 1 / np.square(prior_std)

        # Get the lambda parameter of the posterior distribution
        if not isinstance(prior_mean, np.ndarray):
            prior_mean = np.full(data_mean.shape, prior_mean)
        if not isinstance(prior_std, np.ndarray):
            prior_std = np.full(data_std.shape, prior_std)
        prior_lambda = 1 / np.square(prior_std)

        post_lambda = prior_lambda + data_lambda
        post_std = np.sqrt(1 / post_lambda)
        post_mean = (prior_lambda * prior_mean + data_lambda * data_mean) / post_lambda

        return np.stack([post_mean, post_std], axis=1)

    def get_polls_metric(
        self,
        metric: Literal['pct', 'seats', 'error', 'bias'] = 'pct',
        bmap: Optional[list[str] | str] = None,
        pollster: Optional[int | str] = None,
        featured: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get the polls features and results for each party using a specific metric.

        Parameters
        ----------
        metric : Literal['pct', 'seats', 'error', 'bias'], optional
            The metric to get.
        bmap : list of str or str, optional
            The blocks of parties to use to group the data.
        pollster : int or str, optional
            The pollster to use to filter the data.
        featured : bool, optional
            Whether to filter the polls in featured events and have been computed.
        **kwargs
            Keyword arguments to overwrite the default filter parameters:
            - drange : tuple of int or int
            - n_last : int
            - drop_mtypes : list of str
            - drop_ctypes : list of str

        Returns
        -------
        pd.DataFrame
            The polls data with the specified metric.
        """
        df = self.filter_polls(featured, **kwargs).dropna(axis=1, how='all')

        names = [n for n in self.names if n in df.columns]
        if metric == 'error':
            df = df.drop(columns=names).merge(
                self.errors[names], left_index=True, right_index=True, how='left'
            )
        elif metric == 'bias':
            df = df.drop(columns=names).merge(
                self.biases[names], left_index=True, right_index=True, how='left'
            )

        if bmap is None:
            bmap = names
        elif isinstance(bmap, str):
            bmap = self.merge_bmaps(bmap)

        block_results = group_results(df, bmap=bmap)

        df = df.drop(columns=names).merge(
            block_results, left_index=True, right_index=True, how='left'
        ).reset_index()
        names = block_results.dropna(axis=1, how='all').columns.tolist()

        if isinstance(pollster, str):
            df = df.loc[df.pollster == pollster]
        elif isinstance(pollster, int):
            df = df.loc[df.pollster_id == pollster]

        df['event'] = ['{}-{}{}'.format(dt.year, dt.day, dt.strftime('%b')[0]) for dt in df.event_date]
        df[['days', 'sample_size', 'proc_sample']] = df[['days', 'sample_size', 'proc_sample']].fillna(0).astype(int)
        df['weight'] = df.weight_over * df.weight_sample
        df = df.set_index(['event', 'pollster', 'days']).sort_index(ascending=(True, True, False))[[
            'sample_size', 'proc_sample', 'weight', 'error_avg', 'bias_avg', 'error_blocks', 'bias_blocks', 'error_within', 'bias_within', 'bias'
        ] + names]

        return df
    
    def fit_bias_estimator(
        self,
        featured: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fit the local kernel estimator for the error/bias of the polls conducted for each event.

        Returns
        -------
        pd.DataFrame
            A frame containing the estimation of the mean bias and its error over time.
        """
        df = self.filter_polls(featured, **kwargs)

        # Convert bias to log-odds ratio
        df['lor'] = df['bias'].apply(self.bias_to_lor)
        # Compute the weight of the polls, which is the product of the the overlap/overflow and the sampling weights
        df['weight'] = df[['weight_over', 'weight_sample']].prod(axis=1).round(2)

        # Get the unique event dates
        event_dates = df.index.get_level_values(0).unique()
        # Initialize an empty dataframe to store the results
        dreg = pd.DataFrame()

        # Iterate over the event dates
        iter_ = tqdm(event_dates) if self.verbose > 0 else event_dates
        for dt in iter_:
            # Get the current event polls
            d = df.loc[dt].droplevel(df.index.names[2:])
            # Find the date of the first poll
            dt_start = self.polls.loc[dt].index.get_level_values(0).min()
            # Create a date range from the start date to the event date
            px = pd.date_range(start=dt_start, end=dt, freq='D')

            # Fit the local kernel estimator
            reg = LocalKernelEstimator(
                d['lor'],
                weights=d['weight'],
                **self.reg_params
            ).fit(px, alpha=self.alpha).set_index(
                pd.MultiIndex.from_product([[dt], px], names=['event_date', 'date'])
            )

            # Concatenate the results
            dreg = pd.concat([dreg, reg])

        return dreg

    def get_error_estimator_data(
        self,
        featured: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get the data used to fit the error estimator.

        Parameters
        ----------
        featured : bool, optional
            Whether to filter the polls in featured events and have been computed.
        **kwargs
            Keyword arguments to overwrite the default filter parameters:
            - drange : tuple of int or int
            - n_last : int
            - drop_mtypes : list of str
            - drop_ctypes : list of str

        Returns
        -------
        pd.DataFrame
            A table containing the data.
        """
        polls = self.filter_polls(featured, **kwargs)
        polls['weight'] = polls[['weight_over', 'weight_sample']].prod(axis=1).round(2)
        errors = self.errors.loc[polls.index]

        names = [n for n in self.names if n in polls.columns]
        d_pct = polls.set_index('weight', append=True).melt(
            value_vars=names,
            var_name='party',
            value_name='pct',
            ignore_index=False
        ).dropna().reset_index('weight').sort_index().set_index('party', append=True)
        d_err = errors.melt(
            value_vars=names,
            var_name='party',
            value_name='error',
            ignore_index=False
        ).dropna().sort_index().set_index('party', append=True)

        df = pd.concat([d_pct, d_err], axis=1)
        df = df[
            (df.pct > 0) & (df.error.abs() > 0)
            ].dropna().merge(
            self.polls[['days']], left_index=True, right_index=True, how='left'
        ).reset_index(['date', 'pollster_id', 'party']).reset_index(drop=True)

        df['error'] = df.error.abs()
        df['pollster'] = df.pollster_id.map(self.pollsters.set_index('id').name)
        df['regional'] = df.party.map(self.regional_flags()).astype(int)
        df['color'] = df.party.map(self.parties.set_index('name').color)
        df['weeks'] = (df.days + 1) // 7

        df = df[[
            'date', 'pollster', 'party', 'regional', 'weeks', 'weight', 'color', 'pct', 'error'
        ]].sort_values('pct', ignore_index=True)

        return df

    def get_error_estimator(
        self,
        featured: bool = True,
        **kwargs
    ) -> LeastSquaresEstimator:
        """
        Build an estimator of the error in the percentage of votes for each party.

        Runs a regression analysis that predicts polling error based on this inputs:
            - Percentage of votes of the corresponding party.
            - Is the party a regional party (only representing a certain region)?
            - Number of weeks between the poll and the event.

        Parameters
        ----------
        computed : bool, optional
            Whether to filter the polls that have been computed.
        **kwargs
            Keyword arguments to overwrite the default filter parameters:
            - drange : tuple of int or int
            - n_last : int
            - drop_mtypes : list of str
            - drop_contexts : list of str

        Returns
        -------
        stat.LeastSquaresEstimator
            A LeastSquaresEstimator object fitted to the data.
        """
        df = self.get_error_estimator_data(featured, **kwargs)

        return LeastSquaresEstimator(
            x=df[['pct', 'regional', 'weeks']],
            y=df['error'],
            weights=df['weight']
        ).fit()

    def get_drift_data(
        self,
        horizons: tuple[int, ...] = (7, 14, 30, 60, 90, 180, 365),
        min_polls: int = 10,
        bmap: Optional[str] = None,
        max_fc: int = 0
    ) -> pd.DataFrame:
        """
        Measure the drift of the poll average in the past election cycles: for each featured event with
        enough polls, the daily series of each party (`bmap` blocks) is fitted with the `Forecaster` and the
        increments `mu(t + d) - mu(t)` over every day `t` of the cycle are summarised for each horizon `d`.

        Parameters
        ----------
        horizons : tuple of int, optional
            Horizons in days.
        min_polls : int, optional
            Minimum number of usable polls of an event to be included.
        bmap : str, optional
            Block map of the event params used to name the series (`main`, ...); every party polled in the
            cycle by default, so that the drift can be measured against the level of the party.
        max_fc : int, optional
            Days forecast after the last poll of each cycle (0: only the fitted range).

        Returns
        -------
        pd.DataFrame
            One row per event, party and horizon: `event_date`, `event_scope`, `party_id`, `party`, `horizon`,
            `level` (mean share of the party over the cycle), `n` (number of increments), `rms` (root mean
            square of the increments, percentage points) and `bias` (mean increment).
        """
        from .forecaster import Forecaster

        polls = self.filter_polls(featured=True, drange=None, n_last=None)
        if 'event_date' in polls.index.names:
            events = pd.Series(polls.index.get_level_values('event_date'))
        else:
            events = polls['event_date']
        counts = events.value_counts()
        dates = sorted(pd.to_datetime(counts[counts >= min_polls].index).strftime('%Y-%m-%d'))

        parties = get_parties().set_index('name')
        columns = ['event_date', 'event_scope', 'party_id', 'party', 'horizon', 'level', 'n', 'rms', 'bias']

        rows = []
        dates_ = tqdm(dates) if self.verbose > 0 else dates
        for event_date in dates_:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fc = Forecaster(
                    scope=self.scope,
                    event_date=event_date,
                    drop_mtypes=self.drop_mtypes,
                    drange=None,
                    alpha=self.alpha,
                    bmap=bmap if (bmap is not None and bmap in self.event_params[event_date]['bmaps']) else None,
                    reg_params=self.reg_params,
                    verbose=0,
                    path=self.path
                ).build_series()
                fc.fit_forecast(max_fc=max_fc)

            for name in fc.names:
                if name not in parties.index:
                    continue

                mu = fc.forecast[name].astype(float)
                if mu.notnull().sum() == 0:
                    continue

                for d in horizons:
                    delta = (mu.shift(-int(d)) - mu).dropna()
                    if delta.shape[0] == 0:
                        continue

                    rows.append({
                        'event_date': pd.Timestamp(event_date),
                        'event_scope': self.scope,
                        'party_id': int(parties.loc[name, 'id']),
                        'party': name,
                        'horizon': int(d),
                        'level': float(mu.mean()),
                        'n': int(delta.shape[0]),
                        'rms': float(np.sqrt(np.mean(np.square(delta)))),
                        'bias': float(delta.mean())
                    })

        df = pd.DataFrame(rows, columns=columns)

        return df.sort_values(['event_date', 'party_id', 'horizon'], ignore_index=True)

    def compute_drift(
        self,
        save: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute the drift table of the current events (see `get_drift_data`) and, optionally, save it into
        the `drift` table of the database, replacing the rows of the same events.

        Parameters
        ----------
        save : bool, optional
            Whether to save the data into the database.
        **kwargs
            Passed to `get_drift_data`.
        """
        if self.verbose > 0:
            print('Compute drift...')

        df = self.get_drift_data(**kwargs)

        if save:
            if self.verbose > 0:
                print('Save drift data...')

            nrows = save_drift_data(df)

            if self.verbose > 0:
                print('{} rows updated...'.format(nrows))

        self.drift = df

        return df

    def load_drift(self) -> pd.DataFrame:
        """
        Load the drift table of the current events from the database (empty if it was never computed).
        """
        self.drift = get_drift(scope=self.scope, event_dates=self.event_dates)

        return self.drift

    def party_first_polls(self) -> pd.Series:
        """
        Birth date of each party in the data: the first poll listing it (over the polls of the current events
        that enter the model: pollster types not dropped, positive weight), indexed by party name. The age of a
        party is counted from it. The first official result is deliberately not used: a party that ran
        marginally for years (Cs in 2008, VOX in 2015) starts to drift when pollsters start listing it.
        """
        polls = self.filter_polls(featured=False, drange=None, n_last=None)
        dates = polls.index.get_level_values('date')
        first = {}
        for name in [n for n in self.names if n in polls.columns and n != '-']:
            mask = polls[name].notnull().to_numpy()
            if mask.any():
                first[name] = dates[mask].min()

        return pd.Series(first, dtype='datetime64[ns]', name='first_poll')

    def party_ages(
        self,
        date: str | pd.Timestamp
    ) -> pd.Series:
        """
        Age of each party at `date`, in years since its first poll (NaN for unknown parties).
        """
        first = self.party_first_polls()

        return ((pd.Timestamp(date) - first).dt.days / 365.25).rename('age')

    def get_drift_estimator(
        self,
        d_min: int = 60,
        agg: Literal['geometric', 'quadratic'] = 'geometric',
        age_max: float = 4.
    ) -> DriftEstimator:
        """
        Build the drift estimator from the drift table of the current events, discounting the older cycles
        with `year_decay` (the same decay used for the pollster ratings). When the table is empty the data
        is computed on the fly (slower, not saved); when no event has rows at `d_min` days or more (the
        first cycles), the whole table is used with a warning (in-sample constant).

        Parameters
        ----------
        d_min : int, optional
            Minimum horizon (days) of the rows used in the fit (see `DriftEstimator.fit`).
        agg : {'geometric', 'quadratic'}, optional
            Mean of the relative drift across parties (see `DriftEstimator.fit`).
        age_max : float, optional
            Parties younger than this (years since their first poll, at each election) get their own constant
            and a drift multiplier (M8, see `DriftEstimator`).
        """
        df = self.load_drift()

        if df.shape[0] == 0:
            warnings.warn('No drift data saved for these events: computing it on the fly (see `compute_drift`)')
            df = self.get_drift_data()

        if (df['horizon'] >= d_min).sum() == 0:
            warnings.warn('No drift data at {} days or more for these events: using every event'.format(d_min))
            df = get_drift(scope=self.scope)

        # Age of the party at each election: years since its first poll in the cycles of this Computer
        first = self.party_first_polls()
        df = df.copy()
        df['age'] = (pd.to_datetime(df['event_date']) - df['party'].map(first)).dt.days / 365.25

        return DriftEstimator.fit(df, d_min=d_min, decay=self.year_decay, agg=agg, age_max=age_max)


    def get_house_effects_data(
        self,
        he_params: Optional[dict[str, Any]] = None,
        he_min_polls: int = 3
    ) -> pd.DataFrame:
        """
        Measure the house effects in the past election cycles: for each featured event with results, pollster
        and party, the deviation of the pollster from the official result (its polls of the last `n_days`
        days, precision-weighted) and its deviation from the consensus of the cycle (`Forecaster.fit_house_effects`
        without prior). The deviations from the result are centred across pollsters (weights: number of
        polls) so that they measure the tilt of the house relative to the industry; the industry-wide mean
        is kept apart (`industry`).

        Parameters
        ----------
        he_params : dict, optional
            Parameters of the house effects estimation (see `Forecaster.set_he_params`); `n_days` is the
            window before the election of the deviation from the result.
        he_min_polls : int, optional
            Polls a pollster needs in the window to get a deviation from the result.

        Returns
        -------
        pd.DataFrame
            One row per event, pollster and party: `event_date`, `event_scope`, `pollster_id`, `party_id`,
            `pollster`, `party`, `level` (official share), `n_result`, `dev_result`, `dev_result_err`,
            `dev_result_c`, `industry`, `n_cycle`, `dev_cycle`, `dev_cycle_err` (percentage points).
        """
        from .forecaster import Forecaster

        params = Forecaster.set_he_params(None, he_params)
        parties = get_parties().set_index('name')
        pollsters = self.pollsters.set_index('id')['name']
        keys = ['event_date', 'pollster_id', 'party']

        # (a) Deviation from the official result over the last `n_days` days of each cycle
        polls = self.filter_polls(featured=True, drange=(0, params['n_days']), n_last=None)
        weight = (polls['weight_over'] * polls['weight_sample']).astype(float).rename('w')
        errors = self.errors.reindex(polls.index)
        names = [n for n in errors.columns if n in parties.index]
        long = errors[names].stack(future_stack=True).rename('e').reset_index()
        long = long.rename(columns={long.columns[-2]: 'party'}) if 'party' not in long.columns else long
        long['w'] = weight.reindex(pd.MultiIndex.from_frame(long[self.keys])).to_numpy()
        long = long.dropna(subset=['e', 'w'])
        long = long.loc[long['w'] > 0]

        def agg_result(g):
            w, e = g['w'].to_numpy(), g['e'].to_numpy()
            n, sw = len(e), w.sum()
            dev = float((w * e).sum() / sw)
            if n > 1:
                denom = sw - np.square(w).sum() / sw
                var = float((w * np.square(e - dev)).sum() / denom) if denom > 0 else 0.
                err = float(np.sqrt(max(var, 0.) / (sw ** 2 / np.square(w).sum())))
            else:
                err = np.nan
            return pd.Series({'n_result': n, 'dev_result': dev, 'dev_result_err': err})

        if long.shape[0] > 0:
            result = long.groupby(keys, observed=True)[['w', 'e']].apply(agg_result).reset_index()
        else:
            # No poll with a computed error in the window (e.g. the first load of a scope)
            result = pd.DataFrame(columns=keys + ['n_result', 'dev_result', 'dev_result_err'])
        result = result.loc[result['n_result'] >= he_min_polls]

        # Centre across pollsters, per event and party: the industry-wide deviation is kept apart
        result = self.centre_house_results(result)

        columns = [
            'event_date', 'event_scope', 'pollster_id', 'party_id', 'pollster', 'party', 'level',
            'n_result', 'dev_result', 'dev_result_err', 'dev_result_c', 'industry', 'n_cycle', 'dev_cycle', 'dev_cycle_err'
        ]
        if result.shape[0] == 0:
            # No pollster with enough polls in the campaign of any event: no house effects for this scope
            return pd.DataFrame(columns=columns)

        # (b) Deviation from the consensus of the cycle, all polls, no prior
        rows = []
        events = sorted(pd.to_datetime(result['event_date']).dt.strftime('%Y-%m-%d').unique())
        events_ = tqdm(events) if self.verbose > 0 else events
        for event_date in events_:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fc = Forecaster(
                    scope=self.scope,
                    event_date=event_date,
                    drop_mtypes=self.drop_mtypes,
                    drange=None,
                    alpha=self.alpha,
                    bmap=None,
                    reg_params=self.reg_params,
                    house_effects=True,
                    he_params=dict(params, prior=None),
                    verbose=0,
                    path=self.path
                ).build_series()
                effects = fc.fit_house_effects(prior=None)
            for (pid, name), r in effects.iterrows():
                rows.append({
                    'event_date': pd.Timestamp(event_date), 'pollster_id': int(pid), 'party': name,
                    'n_cycle': int(r['n']), 'dev_cycle': float(r['dev']),
                    'dev_cycle_err': float(r['dev_err']) if np.isfinite(r['dev_err']) else np.nan
                })
        cycle = pd.DataFrame(rows, columns=keys + ['n_cycle', 'dev_cycle', 'dev_cycle_err'])

        result['event_date'] = pd.to_datetime(result['event_date'])
        df = result.merge(cycle, on=keys, how='outer')
        df = df.loc[df['party'].isin(parties.index)]
        df['event_scope'] = self.scope
        df['party_id'] = df['party'].map(parties['id']).astype(int)
        df['pollster'] = df['pollster_id'].map(pollsters)
        events = self.events
        df['level'] = [
            float(events.loc[dt, party]) if (dt in events.index and party in events.columns) else np.nan
            for dt, party in zip(df['event_date'], df['party'])
        ]

        return df[columns].sort_values(['event_date', 'pollster_id', 'party_id'], ignore_index=True)

    @staticmethod
    def centre_house_results(
        result: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Centre the deviations of the pollsters from the official result across pollsters, per event and party:
        `industry` is the mean deviation of the industry (weighted by the polls of each pollster, `n_result`)
        and `dev_result_c` the deviation of each pollster from it.

        Parameters
        ----------
        result : pd.DataFrame
            One row per event, pollster and party with `n_result` and `dev_result`. It may be empty (a scope
            where no pollster has enough polls in the campaign).

        Returns
        -------
        pd.DataFrame
            `result` with the columns `industry` and `dev_result_c`.
        """
        if result.shape[0] == 0:
            return result.assign(industry=pd.Series(dtype=float), dev_result_c=pd.Series(dtype=float))

        def centre(g):
            industry = float((g['dev_result'] * g['n_result']).sum() / g['n_result'].sum())
            return g.assign(industry=industry, dev_result_c=g['dev_result'] - industry)

        return result.groupby(['event_date', 'party'], observed=True, group_keys=False)[result.columns].apply(centre)

    def compute_house_effects(
        self,
        save: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute the house effects table of the current events (see `get_house_effects_data`) and, optionally,
        save it into the `pollsters_parties` table of the database, replacing the rows of the same events.
        """
        if self.verbose > 0:
            print('Compute house effects...')

        df = self.get_house_effects_data(**kwargs)

        if save:
            if self.verbose > 0:
                print('Save house effects data...')

            nrows = save_house_effects_data(df)

            if self.verbose > 0:
                print('{} rows updated...'.format(nrows))

        self.house_effects = df

        return df

    # --- Herding (M13) ---------------------------------------------------------------------------------------

    def get_herding_data(
        self,
        n_series: int = 4,
        min_polls: int = 5
    ) -> pd.DataFrame:
        """
        Measure the herding of each pollster in every election cycle with polls, the current one included (it
        does not need the result): the dispersion of its published figures around the average fitted without
        its own polls, relative to the sampling error of its samples, over the `n_series` main series of the
        cycle (`Forecaster.measure_herding` on the raw polls, without house effects nor effective sample).

        Returns
        -------
        pd.DataFrame
            One row per event and pollster: `event_date`, `event_scope`, `pollster_id`, `pollster`, `n`,
            `n_series`, `ss_obs`, `ss_exp`, `dof`, `ratio`, `p_value`.
        """
        from .forecaster import Forecaster

        columns = ['event_date', 'event_scope', 'pollster_id', 'pollster', 'n', 'n_series', 'ss_obs', 'ss_exp', 'dof', 'ratio', 'p_value']
        events = sorted(self.polls.index.get_level_values('event_date').unique().strftime('%Y-%m-%d'))
        frames = []

        events_ = tqdm(events) if self.verbose > 0 else events
        for event_date in events_:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    fc = Forecaster(
                        scope=self.scope, event_date=event_date, drop_mtypes=self.drop_mtypes, drange=None,
                        alpha=self.alpha, bmap=None, house_effects=False, dispersion=False, verbose=0, path=self.path
                    ).build_series()
                    herd = fc.measure_herding(n_series=n_series, min_polls=min_polls)
                except Exception as e:  # A cycle without enough polls to fit an average
                    if self.verbose > 0:
                        print('Herding {} skipped: {}'.format(event_date, e))
                    continue

            if herd.shape[0] == 0:
                continue
            herd = herd.reset_index()
            herd['event_date'] = pd.Timestamp(event_date)
            herd['event_scope'] = self.scope
            frames.append(herd)

        if len(frames) == 0:
            return pd.DataFrame(columns=columns)

        df = pd.concat(frames, ignore_index=True)
        df['pollster_id'] = df['pollster_id'].astype(int)

        return df[columns].sort_values(['event_date', 'pollster_id'], ignore_index=True)

    def compute_herding(
        self,
        save: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute the herding table of the current events (see `get_herding_data`) and, optionally, save it into
        the `pollsters_herding` table of the database, replacing the rows of the same events.
        """
        if self.verbose > 0:
            print('Compute herding...')

        df = self.get_herding_data(**kwargs)

        if save:
            if self.verbose > 0:
                print('Save herding data...')

            nrows = save_herding_data(df)

            if self.verbose > 0:
                print('{} rows updated...'.format(nrows))

        self.herding = df

        return df

    def load_herding(self) -> pd.DataFrame:
        """
        Load the herding table of the scope from the database (empty if it was never computed).
        """
        self.herding = get_herding(scope=self.scope)

        return self.herding

    def herding_summary(
        self,
        event_date: Optional[str] = None,
        data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Herding of each pollster up to an election: its sums of squares pooled over the cycles held up to
        `event_date` (included: the herding of a cycle is known before its result), each cycle weighed by
        `year_decay` per year of age, as in the ratings. Informative: it does not enter the rating.

        Parameters
        ----------
        event_date : str, optional
            Last cycle counted; every cycle by default.
        data : pd.DataFrame, optional
            Herding table (`get_herding_data`); the one of the computer, or the database, by default.

        Returns
        -------
        pd.DataFrame
            Indexed by `pollster`: `herding` (pooled ratio), `herding_last` (ratio of its last cycle),
            `herding_p` (p-value of its last cycle), `herding_n` (polls) and `herding_events` (cycles).
        """
        if data is None:
            data = self.herding if self.herding is not None else self.load_herding()

        columns = ['herding', 'herding_last', 'herding_p', 'herding_n', 'herding_events']
        df = data.copy()
        if df.shape[0] == 0:
            return pd.DataFrame(columns=columns, index=pd.Index([], name='pollster'))

        df['event_date'] = pd.to_datetime(df['event_date'])
        target = df['event_date'].max() if event_date is None else pd.Timestamp(event_date)
        df = df.loc[df['event_date'] <= target]
        df['d'] = np.power(self.year_decay, target.year - df['event_date'].dt.year)
        df['pollster'] = df['pollster'].astype(str)

        def pool(g):
            g = g.sort_values('event_date')
            return pd.Series({
                'herding': float(np.sqrt((g['d'] * g['ss_obs']).sum() / (g['d'] * g['ss_exp']).sum())),
                'herding_last': float(g['ratio'].iloc[-1]),
                'herding_p': float(g['p_value'].iloc[-1]),
                'herding_n': int(g['n'].sum()),
                'herding_events': int(g.shape[0])
            })

        return df.groupby('pollster')[['event_date', 'd', 'ss_obs', 'ss_exp', 'ratio', 'p_value', 'n']].apply(pool)[columns]

    # --- House effects and error decomposition (M6): pure helpers -------------------------------------------

    @staticmethod
    def block_error_decomposition(
        errors: pd.DataFrame,
        blocks: dict[str, list[str]]
    ) -> pd.DataFrame:
        """
        Split the error of a poll over the parties of the blocks into the part that crosses the blocks and
        the part that only redistributes votes within a block: `error_main = Σ|e_p|` (parties of the blocks),
        `error_between = Σ_blocks |Σ_p e_p|`, `error_within = error_main − error_between` (≥ 0). Percentage
        points, one row per poll; parties outside the blocks do not count.

        Parameters
        ----------
        errors : pd.DataFrame
            Signed errors (poll − result) per party, one row per poll.
        blocks : dict
            `{block: [parties]}`.
        """
        members = {b: [p for p in ps if p in errors.columns] for b, ps in blocks.items()}
        parties = [p for ps in members.values() for p in ps]
        main = errors[parties].abs().sum(axis=1, min_count=1)
        between = pd.concat(
            [errors[ps].sum(axis=1, min_count=1).abs() for ps in members.values() if len(ps) > 0], axis=1
        ).sum(axis=1, min_count=1)

        return pd.DataFrame({'error_main': main, 'error_between': between, 'error_within': main - between})

    @staticmethod
    def within_block_bias(
        polls: pd.DataFrame,
        events: pd.DataFrame,
        blocks: dict[str, list[str]]
    ) -> pd.Series:
        """
        Bias of a poll in the split of each block between its parties, in log-odds: for each party, the odds
        of its share within its block in the poll against the same share in the result, in absolute value,
        weighted by the share of the party in the result. It is 0 when every block is split as in the result,
        whatever the size of the blocks: the error between blocks is measured apart (`bias_blocks`).

        Parameters
        ----------
        polls, events : pd.DataFrame
            Percentages per party (0-100), one row per poll (the result row repeated per poll).
        blocks : dict
            `{block: [parties]}`.
        """
        num = pd.Series(0., index=polls.index)
        den = pd.Series(0., index=polls.index)
        for ps in blocks.values():
            ps = [p for p in ps if p in polls.columns and p in events.columns]
            if len(ps) < 2:
                continue  # A block of one party has no split to get wrong
            # Both totals over the parties the poll reports: a poll that omits a party of the block is not
            # penalised for it (its split is judged among the parties it publishes)
            mask = polls[ps].notnull()
            pb, eb = polls[ps].sum(axis=1, min_count=1), events[ps].where(mask).sum(axis=1, min_count=1)
            for p in ps:
                sp, se = polls[p] / pb, events[p] / eb
                with np.errstate(divide='ignore', invalid='ignore'):
                    lor = np.abs(np.log((sp / (1 - sp)) / (se / (1 - se))))
                ok = np.isfinite(lor) & events[p].notnull() & mask[p]
                num = num + np.where(ok, lor * events[p], 0.)
                den = den + np.where(ok, events[p], 0.)

        return (num / den.where(den > 0)).rename('bias_within')

    @staticmethod
    def house_effects_summary(
        data: pd.DataFrame,
        blocks: dict[str, list[str]],
        current: Optional[str] = None,
        year_decay: float = 0.9,
        n_cap: int = 10
    ) -> pd.DataFrame:
        """
        Summary of the house effects of each pollster by party and by block (the sum of its parties): the
        decayed mean of its centred deviations from the results of the past elections (`hist`), the trend of
        those deviations (`trend`, points per year, weighted slope, NaN with fewer than two elections) and its
        deviation from the consensus in the current cycle (`cycle`).

        Parameters
        ----------
        data : pd.DataFrame
            Rows of `pollsters_parties` (see `get_house_effects_data`); the current cycle may be present with
            `dev_cycle` only.
        blocks : dict
            `{block: [parties]}`.
        current : str, optional
            Date of the current event (rows at that date give `cycle`; the others are the history). By default
            the last event of the data.
        year_decay : float, optional
            Yearly decay of the weight of a past election, counted from `current`.
        n_cap : int, optional
            Cap of the polls of a past election counted in its weight.

        Returns
        -------
        pd.DataFrame
            Indexed by `(pollster, name)`: `n_events`, `hist`, `trend`, `cycle`, `n_cycle`.
        """
        df = data.copy()
        df['event_date'] = pd.to_datetime(df['event_date'])
        current = pd.Timestamp(current) if current is not None else df['event_date'].max()
        hist = df.loc[(df['event_date'] < current) & df['dev_result_c'].notnull() & (df['n_result'] > 0)]
        cyc = df.loc[df['event_date'] == current]

        rows = []
        for (pollster, party), g in hist.groupby(['pollster', 'party'], observed=True):
            years = ((current - g['event_date']).dt.days / 365.25).to_numpy()
            w = np.power(year_decay, years) * np.minimum(g['n_result'].astype(float).to_numpy(), n_cap)
            y = g['dev_result_c'].astype(float).to_numpy()
            mean = float((w * y).sum() / w.sum())
            if len(y) > 1 and np.ptp(years) > 0:
                x = -years  # time runs forward
                xm = (w * x).sum() / w.sum()
                trend = float((w * (x - xm) * (y - mean)).sum() / (w * np.square(x - xm)).sum())
            else:
                trend = np.nan
            rows.append({'pollster': pollster, 'name': party, 'n_events': int(len(y)), 'hist': mean, 'trend': trend})
        out = pd.DataFrame(rows, columns=['pollster', 'name', 'n_events', 'hist', 'trend'])

        # Current cycle rows (pollsters without history included)
        cycle = cyc[['pollster', 'party', 'dev_cycle', 'n_cycle']].rename(columns={'party': 'name', 'dev_cycle': 'cycle'})
        out = out.merge(cycle, on=['pollster', 'name'], how='outer')
        out['n_events'] = out['n_events'].fillna(0).astype(int)

        # Blocks: the sum of the parties of the block
        parts = []
        for block, ps in blocks.items():
            sub = out.loc[out['name'].isin(ps)]
            if sub.shape[0] == 0:
                continue
            agg = sub.groupby('pollster', observed=True).agg(
                n_events=('n_events', 'max'), hist=('hist', lambda s: s.sum(min_count=1)),
                trend=('trend', lambda s: s.sum(min_count=1)), cycle=('cycle', lambda s: s.sum(min_count=1)),
                n_cycle=('n_cycle', 'max')
            ).reset_index()
            agg['name'] = block
            parts.append(agg)
        if parts:
            out = pd.concat([out] + parts, ignore_index=True)

        return out.set_index(['pollster', 'name']).sort_index()[['n_events', 'hist', 'trend', 'cycle', 'n_cycle']]

    def print_house_effects(
        self,
        names: Optional[list[str]] = None,
        current: Optional[str] = None,
        min_events: int = 1
    ) -> pd.DataFrame:
        """
        Table of house effects by pollster, party and block (`vs` blocks of the current event): history,
        trend and current cycle (see `house_effects_summary`). The history comes from `house_effects` (or the
        database); the current cycle is measured with a `Forecaster` on the polls of `current` (the next
        event by default), without prior.

        Parameters
        ----------
        names : list of str, optional
            Parties to show (the `main` parties of the current event by default); blocks are always shown.
        current : str, optional
            Current event date.
        min_events : int, optional
            Pollsters with fewer past elections and no current polls are left out.
        """
        from .forecaster import Forecaster

        data = self.house_effects if self.house_effects is not None else get_house_effects(scope=self.scope)
        current = current or get_next_event_date(scope=self.scope, date_from=self.event_dates[-1]) or self.event_dates[-1]
        params = get_event_params(scope=self.scope, event_dates=[current], path=self.path)[current]
        blocks = params['bmaps']['vs']
        names = names or params['bmaps'].get('main', [])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fc = Forecaster(scope=self.scope, event_date=current, drop_mtypes=self.drop_mtypes, drange=None, alpha=self.alpha,
                            bmap=None, reg_params=self.reg_params, house_effects=True, he_params={'prior': None},
                            verbose=0, path=self.path).build_series()
            effects = fc.fit_house_effects(prior=None).reset_index()
        cycle = pd.DataFrame({
            'event_date': pd.Timestamp(current), 'pollster': effects['pollster'], 'party': effects['name'],
            'dev_cycle': effects['dev'], 'n_cycle': effects['n']
        })
        history = data.loc[pd.to_datetime(data['event_date']) < pd.Timestamp(current)]
        frames = [f for f in (history, cycle) if f.shape[0] > 0]
        data = pd.concat(frames, ignore_index=True) if len(frames) > 0 else cycle

        table = self.house_effects_summary(data, blocks, current=current, year_decay=self.year_decay)
        keep = [n for n in names if n in table.index.get_level_values('name')] + list(blocks)
        table = table.loc[table.index.get_level_values('name').isin(keep)]
        table = table.loc[(table['n_events'] >= min_events) | table['n_cycle'].fillna(0).gt(0)]
        pollsters = table.index.get_level_values('pollster').unique()

        return table.reindex(pd.MultiIndex.from_product([pollsters, keep], names=['pollster', 'name'])).dropna(how='all')

    @staticmethod
    def composition_ratio(
        industry: pd.DataFrame,
        ref_date: pd.Timestamp,
        year_decay: float = 0.9,
        min_events: int = 3,
        floor: float = 0.05
    ) -> float:
        """
        Ratio between the variance of the sum of the errors of the main parties and the sum of their
        variances, `r = Σ_e w_e S_e² / Σ_e w_e Q_e` with `S_e = Σ_p e_p`, `Q_e = Σ_p e_p²` and `w_e =
        year_decay^years`, over the past elections. It is 1 when the errors are independent and 0 when they
        cancel out exactly (a fixed total); the Spanish elections give about 0.25. Clipped to `[floor, 1]`.

        Parameters
        ----------
        industry : pd.DataFrame
            Columns `event_date`, `party`, `industry` (industry-wide error per party and election, percentage
            points), already restricted to the main parties of each election.
        ref_date : pd.Timestamp
            Date the ages are counted from.
        year_decay : float, optional
            Yearly decay of the weight of an election.
        min_events : int, optional
            Elections (with at least two parties) needed; otherwise 1 with a warning.
        floor : float, optional
            Lower bound of the ratio.
        """
        df = industry.dropna(subset=['industry']).copy()
        df['event_date'] = pd.to_datetime(df['event_date'])
        per = df.groupby('event_date')['industry'].agg(S='sum', Q=lambda x: float(np.square(x).sum()), k='size')
        per = per.loc[per['k'] >= 2]
        if per.shape[0] < min_events:
            warnings.warn('Composition ratio needs {} elections with two or more main parties ({} available): using 1'.format(
                min_events, per.shape[0]
            ))
            return 1.0

        years = ((pd.Timestamp(ref_date) - per.index) / pd.Timedelta(days=365.25)).to_numpy(dtype=float)
        w = np.power(float(year_decay), np.clip(years, 0, None))
        q = float((w * per['Q'].to_numpy()).sum())
        if q <= 0:
            return 1.0  # No error at all: nothing to learn from
        r = float((w * np.square(per['S'].to_numpy())).sum() / q)

        return float(np.clip(r, floor, 1.0))

    def get_composition_ratio(
        self,
        min_events: int = 3
    ) -> float:
        """
        Composition ratio of the current events (see `composition_ratio`), from the industry-wide errors
        stored in `pollsters_parties` (`Computer.compute_house_effects`) of the **national** parties: the same
        population the correlation is imposed on in `Simulator.build_frame`.
        """
        he = get_house_effects(scope=self.scope, event_dates=self.event_dates)
        if he.shape[0] == 0:
            warnings.warn('No house effects data for these events: composition ratio set to 1 (independent draws)')
            return 1.0

        he = he.dropna(subset=['industry']).copy()
        he['event_date'] = pd.to_datetime(he['event_date'])
        ind = he.groupby(['event_date', 'party'], observed=True)['industry'].first().reset_index()

        regional = self.regional_flags()
        ind = ind.loc[ind['party'].map(regional).fillna(1).astype(int) == 0]

        return self.composition_ratio(ind, pd.Timestamp(self.event_dates[-1]), year_decay=self.year_decay, min_events=min_events)

    def get_swing_residuals(
        self,
        min_level: float = 1.
    ) -> pd.DataFrame:
        """
        Residuals of the proportional swing between consecutive elections of the current events (M9): for each
        pair, party (national share above 1 % in both) and province, `logres = log(actual / (previous ·
        national ratio))`, the predicted level, the autonomous community of the province (`reg_code` of
        `data/es-provinces.csv`), the mean residual of the community for that pair and party (`reg_mean`) and
        the deviation within it (`within`), with the counts `k` and `n_groups` of `SwingNoise.decompose`.

        Parameters
        ----------
        min_level : float, optional
            Minimum previous share of the party in the province to keep the cell.
        """
        groups = self.app.data.read_csv('es-provinces.csv').set_index('code')['reg_code']
        dates = list(self.event_dates)
        res = get_event_results(self.scope, dates)
        res = res.loc[res['party_id'] > 0].copy()
        res['region_id'] = res['region_id'].astype(int)
        pct = res.pivot_table(index=['date', 'region_id'], columns='party', values='pct', aggfunc='sum', observed=True)
        available = pct.index.get_level_values('date').unique()

        rows = []
        for a, b in zip(dates[:-1], dates[1:]):
            ta, tb = pd.Timestamp(a), pd.Timestamp(b)
            if ta not in available or tb not in available:
                continue
            pa, pb = pct.loc[ta], pct.loc[tb]
            if 0 not in pa.index or 0 not in pb.index:
                continue
            parties = [p for p in pa.columns if p in pb.columns and pa.loc[0, p] > 1 and pb.loc[0, p] > 1]
            for p in parties:
                ratio = pb.loc[0, p] / pa.loc[0, p]
                for r in pa.index:
                    if r == 0 or r not in pb.index:
                        continue
                    prev, act = pa.loc[r, p], pb.loc[r, p]
                    if not (np.isfinite(prev) and np.isfinite(act)) or prev < min_level or act <= 0:
                        continue
                    pred = prev * ratio
                    rows.append({'pair': b, 'party': p, 'region': int(r), 'group': groups.get(int(r), np.nan),
                                 'level': float(pred), 'logres': float(np.log(act / pred))})

        df = pd.DataFrame(rows, columns=['pair', 'party', 'region', 'group', 'level', 'logres'])

        return SwingNoise.decompose(df)

    def get_swing_noise(self) -> SwingNoise:
        """
        Estimator of the regional and provincial deviations from the proportional swing (see `SwingNoise`),
        fitted on the elections of the current events.
        """
        return SwingNoise.fit(self.get_swing_residuals())

    def get_seats_estimator_data(self) -> pd.DataFrame:
        """
        Load data used to fit the seats estimator.

        Returns
        -------
        pd.DataFrame
            A table containing the data.
        """
        df = pd.DataFrame()
        seats_total = None

        for metric in ['pct', 'seats']:
            d = self.load_events(metric)
            if metric == 'seats':
                seats_total = d['seats'].astype(float)  # Seats of the chamber in each election
                d = d.drop(columns=['seats'])

            names = [n for n in self.names if n in d.columns]
            d = d.melt(
                value_vars=names,
                var_name='party',
                value_name=metric,
                ignore_index=False
            ).dropna().sort_index().set_index('party', append=True)

            df = pd.concat([df, d], axis=1)

        df = df[
            (df.pct >= 0.1) & (df.seats > 0)
        ].dropna().reset_index().sort_values(['date', 'pct'], ascending=[True, False], ignore_index=True)

        df['regional'] = df.party.map(self.regional_flags().to_dict()).astype(int)
        df['color'] = df.party.map(self.parties.set_index('name').color.to_dict())
        df['year'] = df.date.dt.year.astype(int)
        df['years'] = df.year.max() - df.year
        df['weight'] = np.power(0.97, df.years).round(2)
        df['ratio'] = (df.seats / df.pct).fillna(0).round(2)
        df['pos'] = df.groupby('date').cumcount() + 1
        # Share of the seats of the chamber: comparable between parliaments of different sizes
        df['seats_total'] = df.date.map(seats_total)
        df['share'] = df.seats / df.seats_total

        df = df.sort_values('pct', ignore_index=True)[[
            'party', 'regional', 'pct', 'seats', 'pos', 'color', 'years', 'weight', 'ratio', 'seats_total', 'share'
        ]]

        return df

    def get_seats_estimator(
        self,
        **kwargs
    ) -> LeastSquaresEstimator:
        """
        Build an estimator of the seats based on the percentage of votes.

        Runs a regression analysis that predicts the share of the seats of the chamber (`seats / seats_total`,
        so that parliaments of 33 and 350 seats are comparable) based on the percentage of votes and whether
        the party is regional or not. Multiply the prediction by the seats of the chamber to get seats.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments to pass to the underlying regression model.

        Returns
        -------
        stat.LeastSquaresEstimator
            A LeastSquaresEstimator object fitted to the data.
        """
        df = self.get_seats_estimator_data()

        return LeastSquaresEstimator(
            x=df[['pct', 'regional']],
            y=df['share'],
            weights=df['weight'],
            **kwargs
        ).fit()

    def plot_deviations(
        self,
        data: Optional[pd.DataFrame] = None,
        bmap: Optional[list[str] | str] = None,
        pollster: Optional[int | str] = None,
        computed: bool = True,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        path: Optional[str] = None,
        **kwargs
    ):
        """
        Plot the errors of each poll over the two main blocks of parties (usually the right-wing and left-wing blocks)
        against the final event results.

        The result is a bulls-eye plot, where each point is located at the 2D representation
        of the poll's error over each of the two blocks.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The data to plot.
            If `None`, the errors of the current polls against the final event results will be plotted.
        bmap : list of str or str, optional
            The blocks of parties to use to group the errors.
            If `None`, the errors will be plotted for all the parties.
        pollster : int or str, optional
            The pollster to use to filter the errors.
            If `None`, the errors will be plotted for all the pollsters.
        computed : bool, optional
            Whether to filter the polls that have been computed.
        ax : plt.Axes, optional
            Axes object to plot the figure on.
            If not provided, a new figure will be created.
        show : bool, optional
            Whether to show the figure or not.
        path : str, optional
            If provided, the figure will be saved to a file at this path.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the underlying plotting function.
            Also accepts keyword arguments to overwrite the default filter parameters:
            - drange : tuple of int or int
            - n_last : int
            - drop_mtypes : list of str
            - drop_contexts : list of str
        """
        _plt_params = dict(
            show_days=True, vmax=8, s=50, annot=False, fmt=None, cm=None, cm_err=None,
            grid=True, title=None, note=None
        )
        _fig_params = dict(
            figsize=(12, 12)
        )

        plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
        fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

        if bmap is None:
            bmap = names
        elif isinstance(bmap, str):
            bmap = self.merge_bmaps(bmap)
        parties = self.parties.set_index('name')['color'].to_dict()

        if data is not None:
            df = data.copy()
        else:
            df = self.get_polls_metric(
                metric='error',
                bmap=bmap,
                pollster=pollster,
                computed=computed,
                **kwargs
            )

        blocks = build_blocks(bmap, parties)
        names = [n for n in blocks.index.tolist() if n in df.columns]

        if 'event' in df.index.names and len(df.index.get_level_values('event').unique()) == 1:
            df = df.droplevel('event')
        if 'pollster' in df.index.names and len(df.index.get_level_values('pollster').unique()) == 1:
            df = df.droplevel('pollster')
        if not plt_params['show_days']:
            df = df.droplevel('days')
        df.index = [
            '{} [{}]'.format(' | '.join(i[:-1]), i[-1])
            if isinstance(i, tuple) else str(i)
            for i in df.index.values
        ]

        col_x, col_y = df.columns[(df.shape[1] - len(names)):(df.shape[1] - len(names) + 2)]
        pollsters = df.index
        x = df[col_x]
        y = df[col_y]

        fig, ax = create_figure(ax=ax, **fig_params)

        vmax = plt_params['vmax']
        vlim = (-vmax, vmax)
        vticks = list(np.arange(-vmax, vmax + 1))

        plot_scatter(
            x, y,
            xlim=vlim,
            xticks=vticks,
            xfmt=plt_params['fmt'],
            ylim=vlim,
            yticks=vticks,
            yfmt=plt_params['fmt'],
            cm=plt_params['cm'],
            s=plt_params['s'],
            vlines=[0],
            hlines=[0],
            ax=ax
        )

        if plt_params['annot']:
            annot = [
                ax.annotate(pollsters[j], (x[j], y[j]))
                for j in np.arange(len(pollsters))
            ]
            adjust_text(
                annot,
                ax=ax,
                expand_points=(2, 2),
                arrowprops=dict(arrowstyle='-', color='k', lw=0.5)
            )

        if plt_params['cm_err'] is None:
            plt_params['cm_err'] = {
                2: 'green',
                4: 'orange',
                6: 'red'
            }

        for k, v in plt_params['cm_err'].items():
            theta = np.linspace(0, 2 * np.pi, 360)
            t_ = k * np.cos(theta)
            r_ = k * np.sin(theta)
            ax.plot(t_, r_, get_color(v), lw=2)

        set_title(plt_params['title'], ax=ax)
        set_note(plt_params['note'], ax=ax)

        plot_figure(show=show, path=self.get_path(path), fig=fig)

    def plot_errors(
        self,
        data: Optional[pd.DataFrame] = None,
        bmap: Optional[list[str] | str] = None,
        pollster: Optional[int | str] = None,
        drange: Optional[tuple[int, int] | int] = None,
        n_last: int = None,
        show: bool = True,
        path: Optional[str] = None,
        **kwargs
    ):
        """
        Plot the errors of each poll over the two main blocks of parties (usually the right-wing and left-wing blocks)
        against the final event results.

        The result is a set of two plots:
            - A bar plot representing the average of the errors over these blocks.
            - A diverging bar plot representing the absolute errors over each of the two blocks.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The data to plot.
            If `None`, the errors of the current polls against the final event results will be plotted.
        bmap : list of str or str, optional
            The blocks of parties to use to group the errors.
            If `None`, the errors will be plotted for all the parties.
        pollster : int or str, optional
            The pollster to use to filter the errors.
            If `None`, the errors will be plotted for all the pollsters.
        drange : tuple of int or int, optional
            Only the polls published within the specified days range before the event will be included.
            If an integer is provided, it will be converted to (`drange`, None), meaning that only polls published
            more than `drange` days before the event will be included.
        n_last : int, optional
            Number of last polls for each pollster and event to use.
        show : bool, optional
            Whether to show the figure or not.
        path : str, optional
            If provided, the figure will be saved to a file at this path.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the underlying plotting function.
        """
        _plt_params = dict(
            show_days=True, vmax=8, annot=False, fmt=None, cm=None,
            grid=True, title=None, note=None
        )
        _fig_params = dict(
            figsize=None
        )

        plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
        fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

        if bmap is None:
            bmap = names
        elif isinstance(bmap, str):
            bmap = self.merge_bmaps(bmap)
        parties = self.parties.set_index('name')['color'].to_dict()

        if data is not None:
            df = data.copy()
        else:
            df = self.get_polls_metric(
                metric='error',
                bmap=bmap,
                pollster=pollster,
                n_last=n_last,
                drange=drange
            )

        blocks = build_blocks(bmap, parties)
        names = [n for n in blocks.index.tolist() if n in df.columns]

        if 'event' in df.index.names and len(df.index.get_level_values('event').unique()) == 1:
            df = df.droplevel('event')
        if 'pollster' in df.index.names and len(df.index.get_level_values('pollster').unique()) == 1:
            df = df.droplevel('pollster')
        if not plt_params['show_days']:
            df = df.droplevel('days')
        df.index = [
            '{} [{}]'.format(' | '.join(i[:-1]), i[-1])
            if isinstance(i, tuple) else str(i)
            for i in df.index.values
        ]
        df['error_blocks'] = df['error_blocks'].abs()
        df = df.sort_values('error_blocks')

        col_x, col_y = df.columns[(df.shape[1] - len(names)):(df.shape[1] - len(names) + 2)]
        x = df[col_x][::-1]
        y = df[col_y][::-1]

        fig_params['n_cols'] = 2
        if fig_params['figsize'] is None:
            fig_params['figsize'] = (20, np.ceil(df.shape[0] / 2))

        fig, axs = create_figure(n=2, sharey=True, **fig_params)

        vmax = plt_params['vmax']
        vlim = (0, vmax)
        vticks = list(np.arange(0, vmax + 1))

        plot_scores(
            df['error_blocks'].rename('Error'),
            vmin=vlim[0],
            vmax=vlim[1],
            vticks=vticks,
            fmt=plt_params['fmt'],
            cm=plt_params['cm'],
            grid=plt_params['grid'],
            legend=True,
            ax=axs[0]
        )

        if plt_params['annot']:
            axs[0].bar_label(axs[0].containers[0], labels=[
                '{}%'.format(format_number(t, 1))
                for t in df['error_blocks']
            ], padding=10)

        plot_diverging(
            x.abs(),
            y.abs(),
            yaxis='right',
            xlim=vlim,
            xticks=vticks,
            xfmt=plt_params['fmt'],
            ylim=vlim,
            yticks=vticks,
            yfmt=plt_params['fmt'],
            cm=blocks.color.tolist(),
            ax=axs[1]
        )

        if plt_params['annot']:
            axs[1].bar_label(
                axs[1].containers[0],
                labels=['{}%'.format(format_number(t, 1)) for t in y],
                padding=10
            )
            axs[1].bar_label(
                axs[1].containers[1],
                labels=['{}%'.format(format_number(t, 1)) for t in x],
                padding=10
            )

        set_title(plt_params['title'])
        set_note(plt_params['note'], size='large')

        adjust_figure(fig=fig, sharey=True, n=2, **fig_params)
        plot_figure(show=show, path=self.get_path(path), fig=fig)
    
    def print_rating_weights(
        self,
        pollster: Optional[int | str] = None
    ):
        polls = self.filter_polls()
        df = polls[polls['bias'].notnull()]
        if pollster is not None:
            df = df[df['pollster'] == pollster]

        # Compute additional weights of each poll
        df[['weight_pos', 'weight_week', 'weight_year']] = self.poll_rating_weights(df)

        # Compute the final weight of each poll by multiplying all the other weights computed
        df['weight'] = df[[
            'weight_over', 'weight_sample', 'weight_pos', 'weight_week', 'weight_year'
        ]].prod(axis=1).round(2).astype(float)
        df = df[df['weight'] > 0]

        df = df.reset_index()

        df['event_date'] = df['event_date'].dt.strftime('%b-%y')
        df[['days', 'parties', 'proc_sample']] = df[['days', 'parties', 'proc_sample']].astype(int)

        weight_cols = ['weight_over', 'weight_sample', 'weight_pos', 'weight_week', 'weight_year']
        df = df.set_index(['event_date', 'days'])[[
            'parties', 'proc_sample', 'error_avg', 'error_blocks', 'error_within', 'bias_avg', 'bias_blocks', 'bias_within',
            'bias', 'bias_dev_adj', 'weight'
        ] + weight_cols].rename(columns={i: i.replace('weight_', 'w_') for i in weight_cols})

        bars = [
            {
                'color': 'red-light',
                'subset': ['bias'],
                'vmin': 0,
                'vmax': 40
            },
            {
                'color': ['red-light', 'green-light'],
                'subset': ['bias_dev_adj'],
                'align': 'zero',
                'vmin': -20,
                'vmax': 20
            },
            {
                'color': 'blue-light',
                'subset': ['weight'],
                'vmin': 0,
                'vmax': 2
            },
            {
                'color': 'purple-light',
                'subset': [i.replace('weight_', 'w_') for i in weight_cols],
                'vmin': 0,
                'vmax': 2
            }
        ]

        dfs = get_df_styler(
            df,
            bars=bars,
            styles=table_styles
        )

        print_styler(dfs=dfs)

    def print_ratings(
        self,
        data: Optional[pd.DataFrame] = None,
        event_date: Optional[str] = None
    ):
        """
        Print the pollsters computed ratings into a table of its main components.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The data to print.
            If `None`, the current ratings will be printed.
        event_date : str, optional
            The date of the event to print the ratings for.
            If `None`, the current ratings will be printed.
        """
        if data is None:
            if event_date is None:
                event_date = self.ratings.index.get_level_values('event_date').max()
            df = self.ratings.loc[event_date].set_index('pollster')
        else:
            df = data.copy()

        df = df[[
            'quality', 'num_polls', 'num_polls_w', 'bias_dev_adj', 'rating_adj', 'rating', 'weight_rating'
        ]].sort_values('rating', ascending=False)
        df['rating'] = df['rating'].round()

        # Herding of each pollster up to the event (M13): informative, it does not enter the rating
        herding = self.herding_summary(event_date=event_date)
        if herding.shape[0] > 0:
            df['herding'] = herding['herding'].reindex(df.index).round(2)

        bars = [
            {
                'color': 'green-light',
                'subset': ['rating'],
                'vmin': 0,
                'vmax': 100
            },
            {
                'color': ['red-light', 'green-light'],
                'subset': ['bias_dev_adj'],
                'align': 'zero',
                'vmin': -2,
                'vmax': 2
            }
        ]
        gradients = [
            {
                'cmap': 'purple',
                'subset': ['num_polls_w'],
                'vmin': 0
            },
            {
                'cmap': 'blue',
                'subset': ['rating_adj'],
                'vmin': 0
            }
        ]
        if 'herding' in df.columns:
            # Centred at 1: herding (below) in red, over-dispersion (above, already penalized by M12) in grey
            bars.append({
                'color': ['red-light', 'grey-alpha'],
                'subset': ['herding'],
                'align': 1.,
                'vmin': 0,
                'vmax': 2
            })

        dfs = get_df_styler(
            df,
            bars=bars,
            gradients=gradients,
            styles=table_styles
        )

        print_styler(dfs=dfs)

    def plot_ratings(
        self,
        data: Optional[pd.DataFrame | pd.Series] = None,
        metric: Optional[Literal['rating', 'bias_dev_adj', 'rating_adj', 'weight_rating']] = 'rating',
        event_date: Optional[str] = None,
        show: bool = True,
        path: Optional[str] = None,
        **kwargs
    ):
        """
        Plot the pollsters computed ratings (or a specific component metric) into an inversely ordered bar plot.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series, optional
            The data to plot.
            If `None`, the ratings of the current polls will be plotted.
        metric : Literal['rating', 'bias_dev_adj', 'rating_adj', 'weight_rating'], optional
            The metric to plot.
            By default, the main `rating` metric is plotted.
        event_date : str, optional
            The date of the event to plot the ratings for.
        show : bool, optional
            Whether to show the figure or not.
        path : str, optional
            If provided, the figure will be saved to a file at this path.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the underlying plotting function.
        """
        _plt_params = dict(
            title=None, note=None
        )
        _fig_params = dict(
            figsize=None
        )

        plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
        fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

        if data is None:
            if event_date is None:
                event_date = self.ratings.index.get_level_values('event_date').max()
            ds = self.ratings.loc[event_date].set_index('pollster')[metric]
        elif isinstance(data, pd.DataFrame):
            ds = data[metric].copy()
        else:
            ds = data.copy()

        ds = ds.loc[ds > 0]

        if metric == 'weight_rating':
            vlim = (np.ceil((ds - 1).abs().max() * 10) / 10 + .1) * np.array([-1, 1])
            vticks = .1
        elif metric == 'bias_dev_adj' or metric == 'rating_adj':
            vlim = 6 * self.bias_dev_tau * np.array([-1, 1])
            vticks = .1
        else:
            vlim = (0, 100)
            vticks = 10

        if fig_params['figsize'] is None:
            fig_params['figsize'] = (20, ds.shape[0] // 3)

        fig, ax = create_figure(**fig_params)

        vmean = ds.mean()
        plot_scores(
            ds,
            sort=True,
            vmin=vlim[0],
            vmax=vlim[1],
            vticks=vticks,
            vlines=[vmean],
            title='Ratings',
            ax=ax
        )

        set_title(plt_params['title'], ax=ax)
        set_note(plt_params['note'], ax=ax)

        plot_figure(show=show, path=self.get_path(path), fig=fig)

    def plot_ratings_grid(
        self,
        data: Optional[pd.DataFrame] = None,
        event_date: Optional[str] = None,
        show: bool = True,
        path: Optional[str] = None,
        **kwargs
    ):
        """
        Plot the pollsters computed ratings into a grid of subplots.

        The result is a set of four subplots:
            - A plot representing the main `rating` metric.
            - A plot representing the `weight_rating` metric.
            - A plot representing the `rating_adjusted` metric.
            - A plot representing the `rating_prior` metric.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The data to plot.
            If `None`, the ratings of the current polls will be plotted.
        event_date : str, optional
            The date of the event to plot the ratings for.
        show : bool, optional
            Whether to show the figure or not.
        path : str, optional
            If provided, the figure will be saved to a file at this path.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the underlying plotting function.
        """
        _plt_params = dict(
            title=None, note=None
        )
        _fig_params = dict(
            figsize=None
        )

        plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
        fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

        if data is None:
            if event_date is None:
                event_date = self.ratings.index.get_level_values('event_date').max()
            df = self.ratings.loc[event_date].set_index('pollster')
        else:
            df = data.copy()

        df = df.loc[df.rating > 0][[
            'rating', 'weight_rating', 'bias_dev_adj', 'rating_adj'
        ]].sort_values('rating', ascending=False)

        if fig_params['figsize'] is None:
            fig_params['figsize'] = (20, df.shape[0] // 3)

        fig, axs = create_figure(n=4, n_cols=2, sharey=True, **fig_params)

        vmean = df.rating.mean()
        plot_scores(
            df.rating,
            vmin=0,
            vmax=100,
            vticks=10,
            vlines=[vmean],
            title='Rating',
            ax=axs[0]
        )

        vlim = 6 * self.bias_dev_tau
        plot_scores(
            df.weight_rating - 1,
            vmin=-vlim,
            vmax=vlim,
            vticks=1,
            title='Weight',
            ax=axs[1]
        )
        axs[1].set_xticklabels(['{}'.format(format_number(float(i) + 1)) for i in axs[1].get_xticks()])

        vlim = 6 * self.bias_dev_tau
        plot_scores(
            df.bias_dev_adj,
            vmin=-vlim,
            vmax=vlim,
            vticks=1,
            title='Bias Deviation',
            ax=axs[2]
        )

        vlim = 6 * self.bias_dev_tau
        plot_scores(
            df.rating_adj,
            vmin=-vlim,
            vmax=vlim,
            vticks=1,
            title='Rating Adjusted',
            ax=axs[3]
        )

        set_title(plt_params['title'])
        set_note(plt_params['note'], size='large')

        adjust_figure(fig=fig, n=4, n_cols=2, **fig_params)
        plot_figure(show=show, path=self.get_path(path), fig=fig)

    def plot_ratings_prior(
        self,
        data: Optional[pd.DataFrame] = None,
        event_date: Optional[str] = None,
        show: bool = True,
        path: Optional[str] = None,
        **kwargs
    ):
        """
        Plot the pollsters computed prior ratings, which helps understanding the
        process of the ratings estimation.

        The prior rating is an estimation of the Bayesian prior of the pollsters'
        mean adjusted deviation based on the quality and the number of polls published
        by the pollster. We use it to account for pollsters with few polls published
        or with a low quality and methodology standards.

        The result is a bubble plot with the pollster's quality on the x-axis and
        the number of polls published on the y-axis. The size of the bubbles is
        proportional to the pollster's prior rating computed.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The data to plot.
            If `None`, the prior ratings of the current polls will be plotted.
        event_date : str, optional
            The date of the event to plot the prior ratings for.
        show : bool, optional
            Whether to show the figure or not.
        path : str, optional
            If provided, the figure will be saved to a file at this path.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the underlying plotting function.
        """
        _plt_params = dict(
            title=None, note=None
        )
        _fig_params = dict(
            figsize=(20, 10)
        )

        plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
        fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

        if data is None:
            if event_date is None:
                event_date = self.ratings.index.get_level_values('event_date').max()
            df = self.ratings.loc[event_date].set_index('pollster')
        else:
            df = data.copy()

        df = df.loc[df.rating_prior.notnull()][[
            'quality', 'num_polls_w', 'rating_prior'
        ]]
        s = Stat(df.rating_prior).scale_minmax(20, 2000).data.round(2).astype(float)

        fig, ax = create_figure(**fig_params)

        plot_scatter(
            x=df.quality,
            y=df.num_polls_w,
            annot=True,
            annot_adjust=True,
            s=s,
            ax=ax
        )

        set_title(plt_params['title'], ax=ax)
        set_note(plt_params['note'], ax=ax)

        adjust_figure(fig=fig)
        plot_figure(show=show, path=self.get_path(path), fig=fig)
    
    def plot_bias_weights(
        self,
        date: Optional[str] = None,
        unit: Optional[Literal['bias', 'lor']] = 'bias',
        pollster: Optional[str | int] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Plot the weights for each poll that affects the estimation at a specific date.
        """
        _plt_params = dict(
            freq='D', margin=0.05, legend=True, title=None, note=None
        )
        _fig_params = dict(
            figsize=(20, 10)
        )

        df = self.filter_polls(
            featured=True,
            drange=None,
            n_last=None,
            drop_ctypes=None
        )

        event_date = df.xs(date, level=1).index.get_level_values(0)[0]
        df = df.loc[event_date].reset_index(['pollster_id', 'sponsor_id'])
        ix = pd.date_range(start=df.index.min(), end=event_date, freq='D')

        df['lor'] = df['bias'].apply(self.bias_to_lor)
        df['weight'] = df[['weight_over', 'weight_sample']].prod(axis=1).round(2)
        weights = df['weight'].values

        if date is None:
            date = ix.max()

        plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
        fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

        # Get the estimator instance
        est = LocalKernelEstimator(
            df['lor'],
            weights=weights,
            **self.reg_params
        )

        # Compute the pairwise kernel weights for each day in the index
        kws = est.build_pred(ix).get_kernel().get_weights(est.pred)
        # This kernel weights are combined with the poll weights in order to be used in the model
        weights = kws * weights

        # Get the index position of the specified date and assign the corresponding individual weights
        pos = (pd.to_datetime(date) - ix.min()).days
        df['weight_kernel'] = kws[pos]
        df['weight'] = weights[pos]

        dt_from, dt_to = df.loc[df.weight_kernel >= 1e-2].index[[0, -1]]
        df = df.loc[df.weight >= 1e-2].loc[dt_from:dt_to]

        loc_est = est.get_local_estimator(kws, pos)
        dr = loc_est.fit(np.arange(loc_est.ranges[0][0], loc_est.ranges[0][1] + 1), alpha=self.alpha)
        dr['cint'] = dr['cmax'] - dr['mean']
        dr.index = ts_from_delta(dr.index, dt_from=est.ranges[0][0], freq=est.ts_freq)
        dr = dr.loc[dt_from:dt_to]

        cmap = build_color_seq_map('blue', beta=0.1)
        sw_scaled = (df.weight / df.weight.max(axis=0)).values
        s = (200 * (1 + sw_scaled)).round().astype(int)

        if pollster is not None:
            cmap_p = build_color_seq_map('yellow', beta=0.1)
            if isinstance(pollster, str):
                is_p = (df['pollster'] == pollster).astype(int).values
            elif isinstance(pollster, int):
                is_p = (df['pollster_id'] == pollster).astype(int).values
            color = [cmap_p(i) if j else cmap(i) for i, j in zip(sw_scaled, is_p)]
        else:
            color = [cmap(i) for i in sw_scaled]

        if unit == 'bias':
            dr[['mean', 'cmin', 'cmax', 'err', 'cint']] = dr[['mean', 'cmin', 'cmax', 'err', 'cint']].apply(self.lor_to_bias)

        lbl_reg = 'Mean: {} | CI 95%: {}'.format(
            format_number(dr['mean'].loc[date]),
            format_number(dr['cint'].loc[date])
        )
        lbl_score = 'Error: {}'.format(format_number(dr['err'].loc[date]))

        fig, ax = create_figure(ax=ax, **fig_params)

        col_params = {
            'mean': dict(cm=get_color('blue'), label=lbl_reg),
            'cmin': dict(cm=get_color('yellow'), lw=2, ls='--', label=lbl_score),
            'cmax': dict(cm=get_color('yellow'), lw=2, ls='--', label=None)
        }

        ax.axvline(pd.to_datetime(date), color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)
        ax.axhline(dr['mean'].loc[date], color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)

        ax.scatter(
            df.index.values,
            df[unit].values,
            s=s,
            marker='.',
            color=color,
            label=None
        )

        plot_series(
            dr[['mean', 'cmin', 'cmax']],
            dt_from=dt_from,
            dt_to=dt_to,
            freq=plt_params['freq'],
            ymin=0,
            fmt='{}%',
            legend=False,
            col_params=col_params,
            ax=ax
        )

        ax2 = ax.twinx()
        ax2.plot(
            df.index.values,
            df.weight_kernel.values,
            color=get_color('purple-light'),
            lw=2,
            label='Kernel'
        )
        ax2.set_ylim([0, 1])
        ax2.set_yticks([])

        lh = {
            label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
        }
        ax.legend(
            handles=lh.values(),
            labels=lh.keys(),
            loc='upper left'
        )

        set_title(plt_params['title'], ax=ax)
        set_note(plt_params['note'], ax=ax)

        plot_figure(show=show, path=self.get_path(path), fig=fig)
    
    def print_bias_weights(
        self,
        date: Optional[str] = None,
        unit: Optional[Literal['bias', 'lor']] = 'bias',
        show: bool = True
    ) -> None:
        df = self.filter_polls(
            featured=True,
            drange=None,
            n_last=None,
            drop_ctypes=None
        )

        event_date = df.xs(date, level=1).index.get_level_values(0)[0]
        df = df.loc[event_date].reset_index(['pollster_id', 'sponsor_id'])
        ix = pd.date_range(start=df.index.min(), end=event_date, freq='D')

        df['lor'] = df['bias'].apply(self.bias_to_lor)
        df['weight'] = df[['weight_over', 'weight_sample']].prod(axis=1).round(2)
        weights = df['weight'].values

        if date is None:
            date = ix.max()

        # Get the estimator instance
        est = LocalKernelEstimator(
            df['lor'],
            weights=weights,
            **self.reg_params
        )

        # Compute the pairwise kernel weights for each day in the index
        kws = est.build_pred(ix).get_kernel().get_weights(est.pred)
        # This kernel weights are combined with the poll weights in order to be used in the model
        weights = kws * weights

        # Get the index position of the specified date and assign the corresponding individual weights
        pos = (pd.to_datetime(date) - ix.min()).days
        df['weight_kernel'] = kws[pos]
        df['weight'] = weights[pos]

        dt_from, dt_to = df.loc[df.weight_kernel >= 1e-2].index[[0, -1]]
        df = df.loc[df.weight >= 1e-2].loc[dt_from:dt_to].reset_index()

        loc_est = est.get_local_estimator(kws, pos)
        dr = loc_est.fit(np.arange(loc_est.ranges[0][0], loc_est.ranges[0][1] + 1))
        rmean = dr[pos]

        # Format dates
        df['date'] = df['date'].dt.strftime('%d-%b')
        df['dev'] = df['lor'] - rmean

        if unit == 'bias':
            df['dev'] = df['dev'].apply(self.lor_to_bias)

        # Set result
        df = df[[
            'date', 'pollster', 'bias', 'dev', 'weight', 'days', 'sample_size', 'weight_kernel'
        ]]

        if not show:
            return df
        else:
            # Get the DataFrame styler and print result
            bars = [
                {
                    'color': ['red-light', 'green-light'],
                    'subset': ['dev'],
                    'align': 'zero',
                    'vmin': -10,
                    'vmax': 10
                },
                {
                    'color': 'blue-light',
                    'subset': ['weight'],
                    'vmin': 0,
                    'vmax': 3
                },
                {
                    'color': 'purple-light',
                    'subset': ['weight_kernel'],
                    'vmin': 0,
                    'vmax': 1
                }
            ]

            dfs = get_df_styler(
                df,
                bars=bars,
                styles=table_styles
            )

            print_styler(dfs=dfs)

    def get_path(
        self,
        name: Optional[str] = None
    ) -> str:
        """
        Get the path to the file.

        Parameters
        ----------
        name : str, optional
            The name of the file.
            If `None`, the path to the current polls will be returned.
        """
        if name:
            return '{}/{}'.format(self.path, name)
