import json
import numpy as np
import warnings
import pandas as pd
import os
from scipy.stats import norm, t as student_t
from tqdm import tqdm

from typing import Any, Callable, Literal, Optional
from typing_extensions import Self

from ..core.app import Core
from ..core.utils.stat import Stat
from ..core.utils.helpers import unset_categorical
from ..core.utils.dataviz import plot_kde_1d

from .forecaster import Forecaster
from .computer import Computer
from .data import (
    get_event_dates, get_event_params, get_event_results, get_event_data, get_parties
)
from .utils import (
    build_blocks, group_results, norm_range
)


class Simulator(Core):

    OTHERS = '-'  # Residual category: other candidatures and blank votes (same label as the Forecaster)
    TERMINAL_WEEKS = 1  # The polling error is always the terminal one (last week); the drift is added apart
    DEFAULT_FAN = (0, 7, 14, 30, 60, 90, 180)  # Default horizons (days) of the fan of intervals, see `fan`

    def __init__(
        self,
        scope: str,
        event_date: str,
        drop_mtypes: Optional[list[str]] = ['aggr', 'online'],
        drange: Optional[tuple[int, int] | int] = None,
        alpha: float = 0.05,
        limit_date: Optional[str] = None,
        as_of: Optional[str] = None,
        n_last: int = 1,
        add_errors: bool = True,
        reg_params: Optional[dict[str, Any]] = None,
        smap: Optional[dict[str, str]] = None,
        threshold: Optional[float] = 3.0,
        house_effects: bool = True,
        industry_bias: bool = False,
        he_params: Optional[dict[str, Any]] = None,
        composition: Optional[float | str] = None,
        seed: Optional[int] = None,
        verbose: int = 0,
        path: str = None
    ) -> None:
        """
        Initialize the simulator.

        Parameters
        ----------
        scope : str
            Scope of the election event.
        event_date : str
            Date of the election event.
        drop_mtypes : list of str, optional
            Drop the polls published by pollsters whose methodology type is in the list.
        drange : tuple of int or int, optional
            Only the polls published within the specified days range before the event will be included.
            If an integer is provided, it will be converted to (`drange`, None), meaning that only polls published
            more than `drange` days before the event will be included.
            If `None`, all the polls will be included.
        alpha : float, optional
            Confidence interval.
        limit_date: str, optional
            The date to be used as cut-off date for the polls and the forecast.
            If not provided, the date will be set to the last day of the campaign period (defined by `drange`).
        as_of: str, optional
            The day on which the forecast is read: the anchor of the nowcast. By default the last day actually
            fitted (`last poll + max_fc`) or `limit_date`, whichever comes first. The horizons of the simulation
            (see `run`) are counted from this day up to `event_date`, treated as the deadline of the legislature.
        n_last: int, optional
            The number of last polls to include by each pollster.
            If not provided, only each pollster's last poll will be included.
        add_errors: bool, optional
            Whether to include error margins and confidence intervals in the simulation.
        reg_params : dict of str, optional
            Parameters for the regression estimator.
            If `None`, the default parameters will be used.
        smap: dict[str, str], optional
            The party source map.
        threshold : float, optional
            Legal threshold, as a percentage of the valid votes of each district, below which a candidature
            is excluded from the seat allocation (art. 163.1.a LOREG: 3 %). Applied when `split=True`;
            `None` disables it.
        house_effects : bool, optional
            Subtract the house effect of each pollster (its systematic deviation on each series, estimated in
            the cycle with a prior from its past elections) from its polls before averaging. See
            `Forecaster.fit_house_effects`.
        industry_bias : bool, optional
            Shift the poll average of each party by the industry-wide bias measured in the past elections
            (see `Forecaster.industry_bias`), adding its uncertainty to the error. Off by default: the bias is
            inconsistent between elections for most parties.
        he_params : dict, optional
            Parameters of the house effects estimation, see `Forecaster.set_he_params`.
        composition : float or 'auto', optional
            Joint noise of the national parties: the ratio between the variance of the sum of their errors and
            the sum of their variances (`Computer.composition_ratio`). `'auto'` estimates it from the past
            elections (about 0.2); a number fixes it; `None` (default) or 1 draws every party independently.
            The marginal intervals are the same in every case (see `build_frame`); the backtest is neutral
            between the two, slightly better independent at short horizons and joint at long ones (M7).
        seed : int, optional
            Base random seed.
        verbose : int, optional
            Level of verbosity.
        path : str, optional
            Path where the model outputs (forecasts, figures) are stored, relative to the app file system root
            (`files/`). Input data (`params.json`, maps) is always read from the versioned `data/` directory.
        """
        super().__init__()

        self.scope = scope
        self.event_date = event_date
        self.house_effects = bool(house_effects)
        self.industry_bias = bool(industry_bias)
        self.he_params = he_params
        self.composition = composition
        self.industry_bias_table = None  # Bias applied to each party when `industry_bias` is on, see `build_forecast`
        self.drop_mtypes = drop_mtypes

        self.drange = norm_range(drange)
        self.alpha = alpha
        self.n_last = n_last
        self.add_errors = add_errors
        self.reg_params = reg_params
        self.smap = smap
        self.threshold = threshold

        self.seed = seed  # Base random seed
        self.verbose = verbose  # Print progress
        self.path = path or '.'  # Path to the model files, relative to the app file system root (files/)

        # Event params: derived from the data and overridden by `data/params.json` when the event is listed
        self.event_params = get_event_params(self.scope, self.event_date, path=self.path)

        self.names = self.event_params['parties']['event']

        if limit_date is not None:
            self.limit_date = limit_date
        else:
            self.limit_date = (
                pd.to_datetime(self.event_date) - pd.DateOffset(days=self.drange[0])
            ).strftime('%Y-%m-%d')

        # `event_date` is the deadline of the legislature, not necessarily the election day: the forecast is
        # anchored at `as_of` (set by `build_forecast`) and `horizon_max` days remain up to the deadline
        self.deadline = pd.Timestamp(self.event_date)
        self._as_of = pd.Timestamp(as_of) if as_of is not None else None
        self.as_of = None
        self.horizon_max = None

        if self.smap is None:
            self.smap = self.event_params.get('smap', {})

        for key, rules in self.smap.items():
            if isinstance(rules, dict):
                self.smap[key] = [rules]

        # The poll average of each party also counts the polls of the parties it inherits (smap `agg` rules):
        # e.g. the polls of UP and MP before SUMAR existed, or of Cs for PP in 2023
        self.model = Forecaster(
            scope=self.scope,
            event_date=self.event_date,
            drange=self.drange,
            bmap=self.poll_blocks(self.names, self.smap),
            drop_mtypes=self.drop_mtypes,
            reg_params=self.reg_params,
            alpha=self.alpha,
            house_effects=self.house_effects,
            he_params=self.he_params,
            verbose=self.verbose,
            path=self.path
        ).build_series()

        # Previous election (base of the provincial projection): first non-poll row of the Forecaster series
        self.prev_date = self.model.nfc_series.index[0].strftime('%Y-%m-%d')

        event_dates = get_event_dates(
            scope=self.scope,
            date_from='1980-01-01',
            date_to=self.event_date,
            skip=1
        )
        # Durations (days) of the past legislatures: the empirical prior of the election date (see `horizon_candidates`)
        all_dates = pd.to_datetime(get_event_dates(scope=self.scope, date_to=self.event_date, skip=1))
        self.durations = [int(d) for d in np.diff(all_dates.values).astype('timedelta64[D]').astype(int)]

        if len(event_dates) > 0:
            self.computer = Computer(
                scope=self.scope,
                event_dates=event_dates,
                drop_mtypes=self.drop_mtypes,
                verbose=self.verbose,
                path=self.path
            ).build_series()
        else:
            self.computer = None

        self.parties = get_parties()
        self.cols_forecast = ['mean', 'regional', 'err', 'nobs', 'error']
        self.cols_frame = self.cols_forecast + ['pct', 'pct_err', 'drift', 'std_err', 'rand', 'vpred']
        self.cols_unit = ['prev_pct', 'vpred_pct']

        if self.verbose > 0:
            print('Load seats estimator...')

        if self.computer is not None:
            self.v2seats = self.computer.get_seats_estimator()
        else:
            self.v2seats = None

        if self.verbose > 0:
            print('Load errors estimator...')

        if self.add_errors and self.computer is not None:
            self.v2err = self.computer.get_error_estimator(
                drange=self.drange,
                n_last=self.n_last
            )
        else:
            self.v2err = None

        if self.verbose > 0:
            print('Load drift estimator...')

        # Drift of the opinion with the horizon (random walk fitted on the past cycles), see `Computer.get_drift_estimator`
        if self.add_errors and self.computer is not None:
            self.v2drift = self.computer.get_drift_estimator()
        else:
            self.v2drift = None

        # Composition of the national errors: a common negative correlation between the national parties,
        # set by the ratio between the variance of their sum and the sum of their variances (M7)
        if composition is None or (not isinstance(composition, str) and float(composition) >= 1):
            self.composition_ratio = 1.0
        elif isinstance(composition, str):
            if composition != 'auto':
                raise ValueError("`composition` must be 'auto', a number in (0, 1] or None")
            if self.computer is None:
                warnings.warn('No past elections to estimate the composition ratio: independent draws')
                self.composition_ratio = 1.0
            else:
                self.composition_ratio = self.computer.get_composition_ratio()
        else:
            if float(composition) <= 0:
                raise ValueError('`composition` must be positive (1: independent draws)')
            self.composition_ratio = float(np.clip(float(composition), 0.05, 1.0))
        self.composition_rho = 0.  # Correlation imposed in the last draw (see `build_frame`)

        if self.verbose > 0:
            print('Load region totals...')

        self.default_region = 0  # `region_id` of the national total (see `get_reg_totals`)
        self.reg_totals = self.get_reg_totals()

        self.regions = self.reg_totals.index.tolist()
        self.default_params = {
            'n_sim': 1,
            'split': False,
            'random': False,
            'names': None,
            'regions': None,
            'horizon': None,
            'date_prior': 'historical'
        }

        if self.verbose > 0:
            print('Load previous results...')

        self.prev_results = self.get_prev_results()
        self.prev_totals = self.get_prev_totals()

        self.params = None
        self.rng = None
        self.horizons = None  # Horizon (days from `as_of`) of each simulation, see `build_horizons`
        self.frames = None
        self.units = None
        self.results = None

        if self.verbose > 0:
            print('Set params...')

        self.set_params()

    def set_params(self, reset: bool = False, **kwargs) -> Self:
        if reset or self.params is None:
            self.params = self.default_params.copy()
        elif len(kwargs) == 0:
            return self

        for k, v in self.params.items():
            self.params[k] = kwargs.get(k, v)

        self.params['n_sim'] = int(self.params['n_sim'])
        self.params['split'] = bool(self.params['split'])
        self.params['random'] = bool(self.params['random'])

        if self.params['names'] is not None:
            self.params['names'] = list(self.params['names'])
        else:
            self.params['names'] = self.names

        if not self.params['split']:
            self.params['regions'] = [self.default_region]
        elif self.params['regions'] is not None:
            self.params['regions'] = list(self.params['regions'])
        else:
            self.params['regions'] = self.regions

        horizon = self.params['horizon']
        if horizon is not None and horizon not in ('deadline', 'random'):
            if isinstance(horizon, (str, bool)) or int(horizon) < 0:
                raise ValueError("`horizon` must be None, a non-negative number of days, 'deadline' or 'random'")
            self.params['horizon'] = int(horizon)

        prior = self.params['date_prior']
        if isinstance(prior, str):
            if prior not in ('historical', 'uniform'):
                raise ValueError("`date_prior` must be 'historical', 'uniform' or an array of days")
        else:
            self.params['date_prior'] = np.asarray(prior, dtype=int)

        if self.params['random']:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = None

        return self

    def get_reg_totals(self) -> pd.DataFrame:
        """
        Get each region's total votes and seats available for the event.

        Returns
        -------
        pd.DataFrame
            A DataFrame with total votes and seats available for each region.
        """
        df = get_event_data(self.scope, self.event_date)
        # Regions are indexed by `region_id` (the official province code; 0 = national total), which is
        # what the `smap` rules refer to. The names are kept aside for display purposes.
        df['region_id'] = df['region_id'].astype(int)
        self.region_names = dict(zip(df['region_id'], unset_categorical(df['region']).fillna(self.scope)))

        return df.set_index('region_id')[['votes', 'seats']]
    
    def get_prev_results(self) -> pd.DataFrame:
        """
        Get the results obtained by each party in the last election.

        Returns
        -------
        pd.DataFrame
            A DataFrame with previous results for each party at each region.
        """
        df = get_event_results(self.scope, self.prev_date)
        df = df.loc[df['party_id'] > 0]
        df['region_id'] = df['region_id'].astype(int)

        names = []
        for n in df['party'].unique():
            if n in self.names and n not in names:
                names.append(n)

        for key, rules in self.smap.items():
            if key not in names:
                names.append(key)

            for rule in rules:
                for name in rule['names']:
                    if name not in names:
                        names.append(name)

        df = df.sort_values(['region_id', 'party_id']).groupby(['region_id', 'party'], sort=False, observed=True, dropna=False)[[
            'votes', 'pct', 'seats'
        ]].sum()
        df = df.reset_index().pivot_table(columns='party', index='region_id', sort=False, observed=True, dropna=False)

        ix = pd.Index(self.regions, name='region_id')
        cols = pd.MultiIndex.from_product([['votes', 'pct', 'seats'], names], names=[None, 'party'])

        for c in cols:
            if c not in df.columns:
                df.loc[:, c] = .0

        df = df.where(df > 0, np.nan).loc[ix, cols]

        return df

    def get_prev_totals(self) -> pd.DataFrame:
        """
        Valid votes (candidatures + blank) and blank share of the previous election, per `region_id`.

        `votes` is the base of `events_results.pct` and of the legal threshold; `blank_pct` is assumed
        to stay unchanged in the simulated election.

        Returns
        -------
        pd.DataFrame
            Indexed by `region_id`, with columns `votes` and `blank_pct`.
        """
        df = get_event_data(self.scope, self.prev_date)
        df['region_id'] = df['region_id'].astype(int)
        df = df.set_index('region_id').reindex(self.regions)

        # Stored as int32 in the database: cast before any arithmetic to prevent overflows
        votes = df['votes'].astype('float64')
        blank = df['blank'].astype('float64').fillna(0.)

        if votes.isnull().any():
            warnings.warn('Valid votes missing for {} in some regions: using the sum of party votes'.format(self.prev_date))
            votes = votes.fillna(self.prev_results['votes'].fillna(0).sum(axis=1).astype('float64'))

        totals = pd.DataFrame({'votes': votes, 'blank_pct': (100. * blank / votes).round(2)})
        totals.index.name = 'region_id'

        return totals

    def categories(self) -> list[str]:
        """
        Parties being simulated plus the residual category `'-'` (other candidatures and blank votes).
        """
        return self.params['names'] + [self.OTHERS]

    @property
    def n_seats(self) -> int:
        """
        Seats of the chamber (national total of `reg_totals`).
        """
        return int(self.reg_totals.loc[self.default_region, 'seats'])

    def fit_forecast(self, **kwargs) -> pd.DataFrame:
        self.model.fit_forecast(**kwargs)
        self.forecast = self.build_forecast()

        return self.forecast
    
    def save_forecast(self, prefix: str) -> Self:
        self.model.save_forecast(prefix)

        return self
    
    def load_forecast(self, prefix: Optional[str] = None) -> pd.DataFrame:
        self.model.load_forecast(prefix)
        self.forecast = self.build_forecast()

        return self.forecast

    def _set_anchor(self) -> None:
        """
        Set `as_of`, the day the forecast is read, and `horizon_max`, the days from `as_of` to the deadline.
        By default `as_of` is the last day actually fitted (`Forecaster.date_fit_last`, or the last poll when
        the forecast was loaded from a file) or `limit_date`, whichever comes first.
        """
        forecast = self.model.forecast
        if self.model.date_fit_last is not None:
            fit_last = pd.Timestamp(self.model.date_fit_last)
        else:
            fit_last = pd.Timestamp(self.model.date_last)

        as_of = self._as_of if self._as_of is not None else min(pd.Timestamp(self.limit_date), fit_last)

        if as_of not in forecast.index:
            raise ValueError('`as_of` {} is outside the forecast range ({} to {})'.format(
                as_of.date(), forecast.index.min().date(), forecast.index.max().date()
            ))
        if as_of > self.deadline:
            raise ValueError('`as_of` {} is after the deadline {}'.format(as_of.date(), self.deadline.date()))
        if self._as_of is not None and as_of < pd.Timestamp(self.model.date_last):
            warnings.warn(
                '`as_of` {} precedes the last poll: the kernel average is two-sided, so the value read there '
                'uses later polls and is not a freeze (use `limit_date` for that)'.format(as_of.date())
            )

        self.as_of = as_of
        self.horizon_max = int((self.deadline - as_of).days)

    def build_forecast(self) -> pd.DataFrame:
        """
        Read the forecast of each party at `as_of` (see `_set_anchor`) and add the residual `'-'`, the
        regional flag and the terminal polling error (`error`, the historical error of the polls of the
        last week, from the `Computer` estimator).
        """
        self._set_anchor()
        weeks = self.TERMINAL_WEEKS

        names = self.params['names']

        def stat(name, key):
            # Parties that could not be fitted (too few polls) have no statistics: NaN
            value = self.model.fc_stat[name].loc[self.as_of]

            return value.get(key, np.nan) if isinstance(value, dict) else np.nan

        fc = pd.DataFrame({
            n: (
                self.model.forecast[n].loc[self.as_of],
                stat(n, 'err'),
                stat(n, 'nobs')
            ) for n in names
        }, index=['mean', 'err', 'nobs']).T

        if fc['mean'].isnull().all():
            warnings.warn('No fitted value at `as_of` {}: the forecast is empty'.format(self.as_of.date()))

        # Industry-wide bias (optional): shift the average by the bias of the polls of the past elections
        if self.industry_bias:
            p = self.model.he_params
            table = self.model.industry_bias(
                self.model.load_house_history(), fc.loc[names, 'mean'], pd.Timestamp(self.event_date),
                year_decay=p['year_decay'], prior_events=p['prior_events'], level_floor=p['level_floor'], rel_cap=p['rel_cap']
            )
            fc.loc[names, 'mean'] = fc.loc[names, 'mean'] - table['bias'].reindex(names).fillna(0.)
            fc.loc[names, 'err'] = np.sqrt(np.square(fc.loc[names, 'err']) + np.square(table['bias_err'].reindex(names).fillna(0.)))
            self.industry_bias_table = table

        # Residual of the poll average (other candidatures and blank votes): never fitted, no statistics
        fc.loc[self.OTHERS] = (max(0., 100. - fc['mean'].sum()), np.nan, np.nan)

        regional = self.parties.set_index('name').loc[names].regional.astype(int).to_dict()
        regional[self.OTHERS] = 0
        fc['regional'] = fc.index.map(regional).astype(int)

        fc['error'] = np.nan
        if self.v2err is not None:
            p = np.column_stack([
                fc.loc[names, ['mean', 'regional']].values,
                np.repeat(weeks, len(names))
            ])
            fc.loc[names, 'error'] = self.v2err.predict(p).values
        else:
            fc.loc[names, 'error'] = fc.loc[names, 'err']

        fc = fc[self.cols_forecast]

        return fc

    def frame(self, loc: int = 0) -> pd.DataFrame:
        """
        Get the feature frame of one simulation: forecast, error and simulated national share (`vpred`)
        of each party, plus the residual `'-'` (other candidatures and blank votes, `100 - sum`).

        Parameters
        ----------
        loc : int, optional
            The simulation index.

        Returns
        -------
        pd.DataFrame
            One row per category (`categories()`), columns `cols_frame`.
        """

        return pd.DataFrame(
            self.frames[loc],
            columns=self.cols_frame,
            index=self.categories(),
            dtype=float
        )

    def region_id(self, region: int | str) -> int:
        """
        Resolve a region given either its `region_id` (province code, 0 = national) or its name.
        """
        if isinstance(region, str) and not region.isdigit():
            ids = [i for i, n in self.region_names.items() if n == region]
            if len(ids) == 0:
                raise KeyError(region)

            return ids[0]

        return int(region)

    def unit(self, loc: int = 0, region: Optional[int | str] = None) -> pd.DataFrame:
        """
        Get the provincial shares of one simulation for a region: previous election (`prev_pct`) and
        simulated (`vpred_pct`), both over valid votes and summing 100 with the residual `'-'`.
        `vpred_pct` are exactly the shares fed to the seat allocation.

        Parameters
        ----------
        loc : int, optional
            The simulation index.
        region : int or str, optional
            `region_id` or region name; the national total (0) by default.
        """
        rloc = self.params['regions'].index(self.region_id(region)) if region is not None else 0

        return pd.DataFrame(
            self.units[loc, rloc],
            columns=self.cols_unit,
            index=self.categories(),
            dtype=float
        )

    def result(self, loc: int = 0) -> pd.DataFrame:
        if self.results is None:
            return

        return pd.DataFrame(
            self.results[loc],
            columns=self.params['names'],
            index=pd.Index([self.region_names.get(r, r) for r in self.params['regions']], name='region'),
            dtype=int
        )

    def dist(self) -> pd.DataFrame:
        if self.results is None:
            return

        if self.params['split']:
            rloc = [i for i in range(len(self.params['regions'])) if self.params['regions'][i] != self.default_region]
            results = self.results[:, rloc]
        else:
            results = self.results

        return pd.DataFrame(
            results.sum(axis=1).round(),
            columns=self.params['names'],
            dtype=int
        )

    def shares(self) -> pd.DataFrame:
        """
        Simulated national vote shares (`vpred`) of each party, one row per simulation. The residual `'-'`
        (other candidatures and blank votes) is available through `frame(loc)`.
        """
        if self.frames is None:
            return

        return pd.DataFrame(
            self.frames[:, :, self.cols_frame.index('vpred')],
            columns=self.categories(),
            dtype=float
        )[self.params['names']]

    def totals(
        self,
        sort: bool = False,
        method: Literal['median', 'mean'] = 'median'
    ) -> pd.Series:
        """
        Headline seats of each party: the central statistic of the simulated distribution (`median` by
        default, or `mean`) expanded proportionally to the size of the chamber and rounded by largest
        remainders (see `round_proportional`). Parties without seats in any simulation stay at 0.

        With `split=True` every simulation sums the chamber size, so `method='mean'` is an exact rounding
        and the medians only need a small expansion. In the regression modes (`split=False`) the seats
        estimator is not constrained and the expansion is large: those headlines are only indicative.

        Parameters
        ----------
        sort : bool, optional
            Sort by descending seats (stable: ties keep the order of `names`).
        method : {'median', 'mean'}, optional
            Central statistic.
        """
        if self.results is None:
            return

        ds = self.round_proportional(self.central(self.dist(), method), self.n_seats)

        if sort:
            return ds.sort_values(ascending=False, kind='stable')

        return ds.loc[self.params['names']]

    def scenario(self, method: Literal['median', 'mean'] = 'median') -> int:
        """
        Index of the simulation closest to the central statistic of the seats (L1 distance; ties by L2 and
        then the lowest index). `result(sim.scenario())` is a coherent, publishable province-by-party
        table: a real simulation whose rows are D'Hondt allocations summing the chamber size.
        """
        dist = self._require_dist()

        return self.closest_simulation(dist, self.central(dist, method))

    def blocks(self, names: str | dict[str, Any] | list | tuple) -> pd.DataFrame:
        """
        Blocks of parties: a `bmap` name of the event params (`blocks`, `vs`, `main`, ...), a dict or a
        list, with the party colors (same resolution as `plot_forecast_output`).
        """
        if isinstance(names, str):
            names = self.model.bmaps[names]

        return build_blocks(names, self.model.colors)

    def summary(
        self,
        names: Optional[str | dict[str, Any] | list | tuple] = None,
        alpha: Optional[float] = None,
        method: Literal['median', 'mean'] = 'median'
    ) -> pd.DataFrame:
        """
        Summary table of the simulation, by party or by block of parties.

        Columns: `pct` (forecast point estimate), `pct_mean`, `pct_lo`, `pct_hi` (simulated national
        shares), `seats` (headline, see `totals`), `seats_mean`, `seats_median`, `seats_lo`, `seats_hi`,
        `seats_min`, `seats_max`, `p_seats`, `p_majority`, `p_first` (see `summarize_seats`). Intervals
        are empirical quantiles at `alpha / 2` and `1 - alpha / 2`.

        Parameters
        ----------
        names : str, dict or list, optional
            Blocks of parties (a `bmap` name, a dict or a list, as in `plot_forecast_output`); one row per
            party if `None`.
        alpha : float, optional
            Confidence level of the intervals; the one given to the constructor by default.
        method : {'median', 'mean'}, optional
            Central statistic of the headline seats.
        """
        alpha = self.alpha if alpha is None else alpha
        parties = self.params['names']

        dist = self._require_dist()
        shares = self.shares()
        seats = self.totals(method=method)
        pct = self.forecast.loc[parties, 'mean']

        if names is not None:
            blocks = self.blocks(names)
            dist = group_results(dist, blocks=blocks)
            shares = group_results(shares, blocks=blocks)
            seats = group_results(seats.to_frame().T, blocks=blocks).iloc[0]
            pct = group_results(pct.to_frame().T, blocks=blocks).iloc[0]

        table = self.summarize_seats(dist, self.n_seats, alpha=alpha, method=method, headline=seats)
        desc = self.describe_dist(shares, alpha=alpha)

        return pd.concat([
            pd.DataFrame({'pct': pct, 'pct_mean': desc['mean'], 'pct_lo': desc['lo'], 'pct_hi': desc['hi']}),
            table
        ], axis=1)

    def probabilities(
        self,
        groups: str | dict[str, Any] | list | tuple,
        majority: Optional[int] = None
    ) -> pd.Series:
        """
        Probability that each block or coalition of parties reaches `majority` seats (the absolute
        majority, `n_seats // 2 + 1`, by default).

        Parameters
        ----------
        groups : str, dict or list
            A `bmap` name (`vs`, `blocks`, ...) or a dict `{name: [parties]}`; coalitions may overlap.
        majority : int, optional
            Seats needed.
        """
        majority = self.n_seats // 2 + 1 if majority is None else int(majority)
        groups = self.model.bmaps[groups] if isinstance(groups, str) else groups

        return self.majority_probs(self._require_dist(), majority, groups=groups)

    def fan(
        self,
        horizons: Optional[list[int] | tuple[int, ...]] = None,
        alpha: Optional[float] = None,
        wide: bool = False
    ) -> pd.DataFrame:
        """
        Fan of intervals of the national vote share of each party by horizon: the analytic counterpart of
        the draw of `build_frame`, `mean ± t(1 - alpha / 2, nobs - 2) · sqrt(err² + pct_err² + drift(h)²)`,
        before the seat allocation. Horizon 0 is the nowcast; the drift grows with the horizon and with
        the level of the party (`mean · sqrt(k · h)`).

        Parameters
        ----------
        horizons : list of int, optional
            Horizons in days (`DEFAULT_FAN` plus `horizon_max` by default), clipped to `horizon_max`.
        alpha : float, optional
            Confidence level of the intervals; the one given to the constructor by default.
        wide : bool, optional
            One row per party with the `lo` and `hi` bounds by horizon as columns.

        Returns
        -------
        pd.DataFrame
            `party`, `horizon`, `mean`, `sd`, `lo`, `hi` (long) or the wide table.
        """
        if self.forecast is None or self.horizon_max is None:
            raise ValueError('No forecast available: call `fit_forecast()` first.')

        alpha = self.alpha if alpha is None else alpha
        if horizons is None:
            horizons = sorted(set(self.DEFAULT_FAN) | {self.horizon_max})
        horizons = [int(h) for h in horizons if 0 <= int(h) <= self.horizon_max]

        names = self.params['names']
        fc = self.forecast.loc[names]

        rows = []
        for n in fc.index[fc['mean'] > 0]:
            mean, err, nobs, regional = [float(fc.loc[n, c]) for c in ['mean', 'err', 'nobs', 'regional']]
            if self.v2err is not None:
                p = np.array([[mean, regional, self.TERMINAL_WEEKS]])
                pct_err = float(np.asarray(self.v2err.predict(p)).ravel()[0])
            else:
                pct_err = 0.

            dof = max((nobs if np.isfinite(nobs) else 3.) - 2, 3.)  # same degrees of freedom as `build_frame`
            q = float(student_t.ppf(1 - alpha / 2, dof))
            base = np.nan_to_num(err) ** 2 + pct_err ** 2

            for h in horizons:
                drift_var = self.v2drift.var(h, mean) if self.v2drift is not None else 0.
                sd = float(np.sqrt(base + drift_var))
                rows.append({
                    'party': n, 'horizon': h, 'mean': mean, 'sd': sd,
                    'lo': max(mean - q * sd, 0.), 'hi': mean + q * sd
                })

        df = pd.DataFrame(rows, columns=['party', 'horizon', 'mean', 'sd', 'lo', 'hi'])

        if wide:
            order = [n for n in names if n in set(df['party'])]

            return df.pivot(index='party', columns='horizon', values=['lo', 'hi']).loc[order]

        return df

    def _require_dist(self) -> pd.DataFrame:
        if self.results is None:
            raise ValueError('No simulations available: call `run()` first.')

        return self.dist()

    def plot_forecast_output(
        self,
        data: Optional[pd.DataFrame] = None,
        names: Optional[list[str] | dict[str, Any] | str] = None,
        **kwargs
    ) -> None:
        if data is None:
            data = self.totals(sort=True)

        if names is None:
            names = self.params['names']

        self.model.plot_forecast_output(
            data=data,
            names=names,
            **kwargs
        )

    def plot_dist_kde(
        self,
        regional: bool = False,
        **kwargs
    ) -> None:
        # Ridges ordered by headline seats, then by mean (deterministic for the small parties)
        key = pd.DataFrame({'seats': self.totals(), 'mean': self.dist().mean()})
        names = [
            n for n in key.sort_values(['seats', 'mean'], ascending=False, kind='stable').index
            if self.forecast.loc[n].regional == int(regional)
        ]

        # The interval drawn is the empirical one, the same as in `summary()`
        kwargs.setdefault('ci_method', 'quantile')
        kwargs['cm'] = np.vectorize(self.parties.set_index('name').color.get)(names).tolist()
        if kwargs.get('path') is not None:
            kwargs['path'] = self.get_path(kwargs['path'])

        plot_kde_1d(
            self.dist()[names],
            **kwargs
        )

    @staticmethod
    def alloc_dhondt(
        d_votes: dict[str, float],
        n_seats: int
    ) -> dict[str, int]:
        seats = {n: 0 for n in d_votes.keys()}
        votes_rem = d_votes.copy()

        while sum(seats.values()) < n_seats:
            next_seat = max(votes_rem, key=votes_rem.get)
            seats[next_seat] += 1
            votes_rem[next_seat] = d_votes[next_seat] / (seats[next_seat] + 1)

        return seats

    @staticmethod
    def alloc_seats(
        d_votes: dict[str, float],
        n_seats: int,
        valid_votes: Optional[float] = None,
        threshold: Optional[float] = None
    ) -> dict[str, int]:
        """
        D'Hondt allocation with the legal threshold (art. 163.1.a LOREG): candidatures with less than
        `threshold` percent of the valid votes of the district are excluded from the allocation.

        Parameters
        ----------
        d_votes : dict
            Votes of each candidature.
        n_seats : int
            Seats of the district.
        valid_votes : float, optional
            Base of the threshold (votes to candidatures plus blank votes). The sum of `d_votes` if `None`.
        threshold : float, optional
            Percentage of the valid votes needed to take part in the allocation. `None` disables it.

        Returns
        -------
        dict
            Seats of each candidature (0 for the excluded ones). If no candidature qualifies, no seat is
            allocated.
        """
        base = valid_votes if valid_votes else sum(d_votes.values())
        eligible = {
            n: v for n, v in d_votes.items()
            if v > 0 and (threshold is None or base <= 0 or 100. * v / base >= threshold)
        }

        seats = {n: 0 for n in d_votes.keys()}
        if n_seats > 0 and len(eligible) > 0:
            seats.update(Simulator.alloc_dhondt(eligible, n_seats))

        return seats

    @staticmethod
    def poll_blocks(
        names: list[str],
        smap: dict[str, Any]
    ) -> dict[str, list[str]]:
        """
        Blocks used to build the poll average of the simulated parties: each party plus the poll labels of
        the parties it inherits according to the `agg` rules of the source map (`smap`) that have no
        regional restriction. The polls of a predecessor (UP and MP before SUMAR, Cs for PP in 2023,
        PDeCAT for JxCat in 2019) then count for the successor instead of falling into the residual `'-'`.

        Parameters
        ----------
        names : list of str
            Parties being simulated.
        smap : dict
            Source map of the event: `{party: [{'type': 'agg' | 'sub' | 'split', 'names': [...], 'regions': ...}]}`.

        Returns
        -------
        dict
            `{party: [party, predecessor, ...]}`, in the order of `names`.
        """
        blocks = {n: [n] for n in names}

        for key, rules in smap.items():
            if key not in blocks:
                continue

            for rule in (rules if isinstance(rules, list) else [rules]):
                if rule.get('type') != 'agg' or rule.get('regions') is not None:
                    continue

                for source in rule.get('names', []):
                    # A source that is itself simulated keeps its own polls
                    if source not in blocks and source not in blocks[key]:
                        blocks[key].append(source)

        return blocks

    # --- Summaries of the simulated distributions (pure functions) ---------------------------------

    @staticmethod
    def round_proportional(
        values: pd.Series,
        total: int
    ) -> pd.Series:
        """
        Scale a vector of non-negative values so that it sums `total` and round it to integers by largest
        remainders (Hamilton): the "proportional expansion" of a central statistic to the size of the
        chamber.

        Zeros stay zero (a party without seats in any simulation never receives one), the sum is exact for
        any base (e.g. 326 for the linear seats estimator, 347 for medians, 350 for means) and ties are
        broken by descending remainder, descending scaled value and original position.

        Parameters
        ----------
        values : pd.Series
            Non-negative values (NaN counts as 0).
        total : int
            Target sum.

        Returns
        -------
        pd.Series
            Integers with the same index, summing `total`.
        """
        v = pd.Series(values, dtype=float).fillna(0.).clip(lower=0.)
        total = int(total)

        if total <= 0 or v.sum() <= 0:
            return pd.Series(0, index=v.index, dtype=int)

        scaled = v * total / v.sum()
        floors = np.floor(scaled).astype(int)
        remainders = np.round(scaled - floors, 9)
        missing = total - int(floors.sum())

        order = np.lexsort((np.arange(len(v)), -scaled.values, -remainders.values))
        out = floors.copy()
        out.iloc[order[:missing]] += 1

        return out

    @staticmethod
    def central(
        dist: pd.DataFrame,
        method: Literal['median', 'mean'] = 'median'
    ) -> pd.Series:
        """
        Central statistic of each column of `dist`: the median (weighted inverted CDF of `Stat`, the same
        estimator used by the plots) or the mean. Columns without finite values give NaN.
        """
        if method == 'median':
            return dist.apply(
                lambda x: Stat(x, dropna=True).median() if np.isfinite(x.to_numpy(dtype=float)).any() else np.nan,
                axis=0
            )
        if method == 'mean':
            return dist.mean()

        raise ValueError('Method `{}` does not exist (use `median` or `mean`).'.format(method))

    @staticmethod
    def describe_dist(
        df: pd.DataFrame,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Mean, median, empirical quantiles (`alpha / 2` and `1 - alpha / 2`), minimum and maximum of each
        column of `df`. Columns without finite values get a row of NaN.
        """
        columns = ['mean', 'median', 'lo', 'hi', 'min', 'max']
        rows = {}

        for col in df.columns:
            x = df[col].to_numpy(dtype=float)
            if not np.isfinite(x).any():
                rows[col] = dict.fromkeys(columns, np.nan)
                continue

            samp = Stat(x, dropna=True)
            rows[col] = {
                'mean': samp.mean(), 'median': samp.median(),
                'lo': samp.quantile(alpha / 2), 'hi': samp.quantile(1 - alpha / 2),
                'min': samp.min(), 'max': samp.max()
            }

        return pd.DataFrame.from_dict(rows, orient='index', columns=columns).astype(float)

    @staticmethod
    def summarize_seats(
        dist: pd.DataFrame,
        n_seats: int,
        alpha: float = 0.05,
        method: Literal['median', 'mean'] = 'median',
        headline: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Summary of the simulated seats of each column of `dist` (parties or blocks).

        Columns: `seats` (headline: the central statistic expanded proportionally to `n_seats`, or the
        `headline` given), `seats_mean`, `seats_median`, `seats_lo`, `seats_hi`, `seats_min`, `seats_max`,
        `p_seats` (probability of at least one seat), `p_majority` (probability of an absolute majority,
        `n_seats // 2 + 1`) and `p_first` (probability of being the largest; ties go to the first column).
        """
        desc = Simulator.describe_dist(dist, alpha=alpha)
        valid = dist.notnull().any(axis=0)
        majority = int(n_seats) // 2 + 1

        if headline is None:
            headline = Simulator.round_proportional(Simulator.central(dist, method), n_seats)
        headline = pd.Series(headline).reindex(dist.columns)

        out = pd.DataFrame(index=dist.columns)
        out['seats'] = headline
        out['seats_mean'] = desc['mean']
        out['seats_median'] = desc['median']
        out['seats_lo'] = desc['lo']
        out['seats_hi'] = desc['hi']
        out['seats_min'] = desc['min']
        out['seats_max'] = desc['max']
        out['p_seats'] = (dist > 0).mean().where(valid)
        out['p_majority'] = (dist >= majority).mean().where(valid)

        present = dist.loc[:, valid]
        if present.shape[1] > 0:
            p_first = present.idxmax(axis=1).value_counts(normalize=True)
        else:
            p_first = pd.Series(dtype=float)
        out['p_first'] = p_first.reindex(dist.columns).fillna(0.).where(valid)

        return out

    @staticmethod
    def majority_probs(
        dist: pd.DataFrame,
        majority: int,
        groups: Optional[pd.DataFrame | dict[str, Any] | list | tuple] = None
    ) -> pd.Series:
        """
        Probability that the seats of each group (blocks or coalitions of parties, or the columns of `dist`
        if `groups` is `None`) reach `majority`. Groups may overlap (each one is summed independently);
        groups without any party present give NaN.
        """
        if groups is not None:
            if isinstance(groups, pd.DataFrame):
                members = groups['parties'].to_dict()
            elif isinstance(groups, dict):
                members = {k: ([v] if isinstance(v, str) else list(v)) for k, v in groups.items()}
            else:
                members = {n: [n] for n in groups}

            dist = pd.DataFrame({
                name: dist[[p for p in parties if p in dist.columns]].sum(axis=1, min_count=1)
                if any(p in dist.columns for p in parties) else np.nan
                for name, parties in members.items()
            }, index=dist.index)

        valid = dist.notnull().any(axis=0)

        return (dist >= majority).mean().where(valid).rename('p_majority')

    @staticmethod
    def closest_simulation(
        dist: pd.DataFrame,
        target: pd.Series
    ) -> int:
        """
        Index of the simulation (row of `dist`) closest to `target`: L1 distance, ties broken by L2 and then
        by the lowest index.
        """
        diff = np.nan_to_num(dist[target.index].to_numpy(dtype=float) - target.to_numpy(dtype=float))
        l1 = np.abs(diff).sum(axis=1)
        l2 = np.square(diff).sum(axis=1)

        return int(np.lexsort((np.arange(len(l1)), l2, l1))[0])

    @staticmethod
    def equicorrelation(
        sigmas: np.ndarray,
        r: float
    ) -> tuple[float, np.ndarray]:
        """
        Common correlation between the national parties that makes the variance of the sum of their errors
        equal to `r` times the sum of their variances: `rho = (r - 1) · Σσ² / ((Σσ)² - Σσ²)`, exact whatever
        the sizes of the parties. `rho` is floored at `-1 / (k - 1)` (plus a margin) so that the matrix stays
        positive definite. Returns `rho` and the `k × k` correlation matrix.

        Parameters
        ----------
        sigmas : np.ndarray
            Standard deviation of the error of each party.
        r : float
            Ratio of the variance of the sum to the sum of the variances (1: independent).
        """
        s = np.asarray(sigmas, dtype=float)
        k = len(s)
        if k < 2 or r >= 1:
            return 0., np.eye(k)

        cross = np.square(s.sum()) - np.square(s).sum()
        rho = float((r - 1.) * np.square(s).sum() / cross) if cross > 0 else 0.
        floor = -1. / (k - 1) + 1e-6
        if rho < floor:
            warnings.warn('Composition ratio {:.3f} needs a correlation below -1/(k-1) with {} parties: floored'.format(r, k))
            rho = floor

        return rho, (1. - rho) * np.eye(k) + rho * np.ones((k, k))

    @staticmethod
    def draw_correlated_t(
        rng: np.random.Generator,
        dof: np.ndarray,
        corr: np.ndarray,
        size: Optional[int] = None
    ) -> np.ndarray:
        """
        Draw Student t variables with the given degrees of freedom per coordinate and the correlation
        structure `corr`, through a Gaussian copula: the marginals are exactly `t(dof)` and the correlation
        of the underlying normals is `corr`.

        Parameters
        ----------
        rng : np.random.Generator
            Random generator.
        dof : np.ndarray
            Degrees of freedom of each coordinate.
        corr : np.ndarray
            Correlation matrix (positive definite).
        size : int, optional
            Number of draws (`(size, k)`); a single draw (`(k,)`) by default.
        """
        k = corr.shape[0]
        chol = np.linalg.cholesky(corr)
        z = rng.standard_normal((size, k) if size is not None else k) @ chol.T
        u = np.clip(norm.cdf(z), 1e-12, 1 - 1e-12)

        return student_t.ppf(u, np.asarray(dof, dtype=float))

    def clip_rate(self) -> float:
        """
        Fraction of the simulations in which the residual `'-'` was clipped to 0 (the drawn shares of the
        parties summed more than 100): a diagnostic of the joint noise.
        """
        if self.frames is None:
            raise ValueError('No simulations available: call `run()` first.')

        vpred = self.frames[:, self.categories().index(self.OTHERS), self.cols_frame.index('vpred')]

        return float(np.mean(vpred <= 0))

    @staticmethod
    def horizon_candidates(
        kind: str | np.ndarray | list | tuple,
        horizon_max: int,
        elapsed_days: Optional[int] = None,
        durations: Optional[list[int]] = None
    ) -> np.ndarray:
        """
        Candidate horizons (days until the election) of a prior on the election date.

        Parameters
        ----------
        kind : str or array-like
            `'deadline'`: the legislature runs to its end (`horizon_max`); `'uniform'`: every day from 0 to
            `horizon_max`; `'historical'`: the past legislatures that lasted at least `elapsed_days`, each one
            giving the candidate `duration - elapsed_days` capped at `horizon_max` (equally likely); or an
            explicit array of days, clipped to `[0, horizon_max]`.
        horizon_max : int
            Days from the anchor to the deadline.
        elapsed_days : int, optional
            Days elapsed since the previous election (`'historical'` only).
        durations : list of int, optional
            Durations (days) of the past legislatures (`'historical'` only).
        """
        horizon_max = int(horizon_max)

        if isinstance(kind, str):
            if kind == 'deadline':
                return np.array([horizon_max], dtype=int)
            if kind == 'uniform':
                return np.arange(horizon_max + 1, dtype=int)
            if kind == 'historical':
                cands = [
                    min(int(d) - int(elapsed_days), horizon_max)
                    for d in (durations or []) if int(d) >= int(elapsed_days)
                ]
                if len(cands) == 0:
                    warnings.warn('No past legislature lasted {} days or more: using a uniform prior'.format(elapsed_days))
                    return np.arange(horizon_max + 1, dtype=int)

                return np.array(sorted(cands), dtype=int)

            raise ValueError("Unknown date prior '{}'".format(kind))

        return np.clip(np.asarray(kind, dtype=int), 0, horizon_max)

    @staticmethod
    def horizon_sampler(
        kind: str | np.ndarray | list | tuple,
        horizon_max: int,
        elapsed_days: Optional[int] = None,
        durations: Optional[list[int]] = None
    ) -> Callable[[np.random.Generator, Optional[int]], np.ndarray]:
        """
        Sampler of horizons: a function `(rng, size) -> days` drawing uniformly from `horizon_candidates`.
        """
        cands = Simulator.horizon_candidates(kind, horizon_max, elapsed_days=elapsed_days, durations=durations)

        return lambda rng, size=None: rng.choice(cands, size=size)

    def build_horizons(self) -> np.ndarray:
        """
        Horizon (days from `as_of`) of each of the `n_sim` simulations, from the `horizon` param: `None` (0,
        nowcast), an integer, `'deadline'` (`horizon_max`) or `'random'` (drawn once from the `date_prior`
        with the simulation generator, before any other draw).
        """
        n, horizon = self.params['n_sim'], self.params['horizon']

        if horizon is None:
            return np.zeros(n, dtype=int)

        if self.horizon_max is None:
            raise ValueError('No forecast available: call `fit_forecast()` first.')

        if horizon == 'deadline':
            return np.full(n, self.horizon_max, dtype=int)

        if horizon == 'random':
            if self.rng is None:
                raise ValueError("`horizon='random'` requires `random=True`")

            elapsed = int((self.as_of - pd.Timestamp(self.prev_date)).days)
            sampler = self.horizon_sampler(self.params['date_prior'], self.horizon_max, elapsed, self.durations)

            return np.asarray(sampler(self.rng, n), dtype=int)

        return np.full(n, int(horizon), dtype=int)

    def build_frame(self, horizon: Optional[int] = None) -> pd.DataFrame:
        """
        Build a feature frame to be used as input for the simulation.

        Each row represents the feature space for a single party, with its forecast result
        and an estimation of vote percentage for each simulation.

        Parameters
        ----------
        horizon : int, optional
            Days from `as_of` to the simulated election: the drift of the opinion over that period
            (`drift`, proportional to the level of the party, see `Computer.get_drift_estimator`) is added
            in quadrature to the polling error.
            `None` or 0 is the nowcast.

        Returns
        -------
        pd.DataFrame
            The feature frame for the simulation.
        """

        weeks = self.TERMINAL_WEEKS

        cats = self.categories()
        names = self.params['names']

        # Initialize the feature frame with the forecasted results for each party and the residual '-'
        df = pd.DataFrame(columns=(self.cols_frame), index=cats, dtype=float)
        df.loc[cats, self.cols_forecast] = self.forecast.loc[cats].values

        df['pct'] = df['mean']  # Estimated global vote percentage calculated by the forcaster for this party
        df['regional'] = df['regional'].astype(int)  # Is the party a regional party (only representing a certain region)?
        vind = df['pct'] > 0  # Only parties with a forecasted percentage greater than 0 are considered
        vind[self.OTHERS] = False  # The residual is never drawn: it absorbs the deviation of the parties' draws

        # If random is set to True, add noise to the forecasted percentage, based on the forcasted error
        if self.params['random']:
            # Drift of the opinion over the horizon (relative to the level of each party), added in quadrature
            if self.v2drift is not None and horizon:
                drift = np.asarray(self.v2drift.sigma(horizon, df.loc[vind, 'pct'].fillna(0)), dtype=float)
            else:
                drift = 0.
            df.loc[vind, 'drift'] = drift

            if self.v2err is not None:
                # If an error estimator is available, use it to estimate the error of the forecasted percentage
                # This estimator uses historical data to estimate the error of the forecasted percentage
                # See Computer.get_error_estimator() for more details
                p = np.column_stack([
                    df.loc[vind, ['pct', 'regional']].fillna(0).values,
                    np.repeat(weeks, df.loc[vind].shape[0])
                ])
                df.loc[vind, 'pct_err'] = self.v2err.predict(p).values
                df.loc[vind, 'std_err'] = np.sqrt(
                    np.square(df.loc[vind, 'err']) + np.square(df.loc[vind, 'pct_err']) + np.square(drift)
                )
            else:
                # If no error estimator available, use the purely statistical error provided by the forcaster
                df.loc[vind, 'std_err'] = np.sqrt(np.square(df.loc[vind, 'err']) + np.square(drift))

            # Add noise to the forecasted percentage, based on the calculated error: a Student t per party
            # (`nobs - 2` degrees of freedom), drawn jointly for the national parties with the common negative
            # correlation set by `composition_ratio` (Gaussian copula, so the marginals are the same), or
            # independently when the ratio is 1
            dof = np.clip(df.loc[vind, 'nobs'].fillna(3) - 2, 3, None).to_numpy(dtype=float)
            sig = df.loc[vind, 'std_err'].to_numpy(dtype=float)
            national = (df.loc[vind, 'regional'].to_numpy() == 0) & np.isfinite(sig)
            if self.composition_ratio < 1 and national.sum() >= 2:
                rho, corr_nat = self.equicorrelation(sig[national], self.composition_ratio)
                corr = np.eye(len(sig))
                idx = np.flatnonzero(national)
                corr[np.ix_(idx, idx)] = corr_nat
                self.composition_rho = rho
                noise = self.draw_correlated_t(self.rng, dof, corr)
            else:
                self.composition_rho = 0.
                noise = self.rng.standard_t(dof)
            df.loc[vind, 'rand'] = np.clip(df.loc[vind, 'pct'] + df.loc[vind, 'std_err'] * noise, 0, None)

            df['vpred'] = df['rand']
        else:
            df['vpred'] = df['pct']

        # Other candidatures and blank votes: the remainder of the simulated national shares
        df.loc[self.OTHERS, 'vpred'] = max(0., 100. - df.loc[names, 'vpred'].sum())

        return df[self.cols_frame]
    
    def build_umat(self, frame: pd.DataFrame) -> pd.DataFrame:
        prev_pcts = self.prev_results['pct'].fillna(0)

        for key, rules in self.smap.items():
            for rule in rules:
                type = rule.get('type')
                names = rule.get('names')
                regions = rule.get('regions')

                if names is not None:
                    if type == 'agg':
                        prev_pcts[key] = prev_pcts[[key] + names].sum(axis=1)
                    elif type == 'sub':
                        if key not in frame.index or any(n not in frame.index for n in names):
                            # The rule needs forecasts for every party involved: skip it otherwise
                            continue

                        fc_k = frame.loc[key]['vpred']
                        fc_v = frame.loc[names]['vpred'].sum()
                        prev_v = prev_pcts[names].sum(axis=1)

                        # Share of the previous vote of the source parties inherited by the new party,
                        # in proportion to the current forecasts: prev_v * fc_k / (fc_k + fc_v)
                        if fc_k + fc_v > 0:
                            prev_pcts[key] = np.round(prev_v * fc_k / (fc_k + fc_v), 2)
                        else:
                            prev_pcts[key] = 0.
                        unit_factor = prev_pcts[key] / len(names)
                        prev_pcts[names] = prev_pcts[names].sub(unit_factor, axis=0)
                    elif type == 'split':
                        prev_k = prev_pcts[key]
                        for name in names:
                            prev_pcts[name] += (prev_k / len(names))
                
                if regions is not None:
                    # Regions are province codes (`region_id`), given as strings or ints in params.json
                    if 'exclude' in regions:
                        excluded = [int(r) for r in regions['exclude']]
                        rix = [r == self.default_region or r not in excluded for r in prev_pcts.index]
                    else:
                        included = [int(r) for r in regions]
                        rix = [r == self.default_region or r in included for r in prev_pcts.index]

                    prev_pcts[key] = prev_pcts[key].where(rix, .0)

                    # Keep the national share consistent with the provinces kept by the rule, so that the
                    # proportional swing reproduces the national forecast within those provinces
                    reg_votes = self.prev_totals['votes']
                    provinces = [r for r in prev_pcts.index if r != self.default_region]
                    prev_pcts.loc[self.default_region, key] = np.round(
                        (prev_pcts.loc[provinces, key] * reg_votes.loc[provinces]).sum() / reg_votes.loc[self.default_region], 2
                    )

        names = self.params['names']
        prev_pcts = prev_pcts.where(prev_pcts > 0, np.nan)[names]

        # Base: valid votes (candidatures + blank) of the previous election, per region
        valid = self.prev_totals['votes']
        blank = self.prev_totals['blank_pct']
        # Other candidatures: everything not simulated (nor moved by the smap rules), net of blank votes
        prev_otros = (100. - prev_pcts.fillna(0).sum(axis=1) - blank).clip(lower=0.).round(2)

        n_votes = valid.loc[self.default_region]
        prev_votes = prev_pcts.mul(valid / 100, axis=0).round()
        fc_votes = (frame['vpred'] * n_votes / 100).round()

        # Proportional swing: each party keeps its previous geography, scaled by its national ratio
        fmul = fc_votes[names] / prev_votes.loc[self.default_region][names]
        vpred_pcts = prev_pcts.mul(fmul)

        orphans = [n for n in names if fc_votes[n] > 0 and not np.isfinite(fmul[n])]
        if len(orphans) > 0 and self.verbose > 0:
            warnings.warn('Parties without previous results nor an applicable smap rule get no seats: {}'.format(orphans))

        # The residual of the poll average, net of the (constant) blank share, swings like any other party
        otros_fc = max(0., frame.loc[self.OTHERS, 'vpred'] - blank.loc[self.default_region])
        if prev_otros.loc[self.default_region] > 0:
            vpred_otros = prev_otros * (otros_fc / prev_otros.loc[self.default_region])
        else:
            vpred_otros = pd.Series(otros_fc, index=prev_otros.index)
        prev_pcts[self.OTHERS] = prev_otros + blank
        vpred_pcts[self.OTHERS] = vpred_otros + blank

        # Shares fed to the allocation: renormalized over all the categories, so that only the small
        # inconsistencies of the swing are corrected (not the votes to other candidatures)
        vpred_pcts = vpred_pcts.mul(100. / vpred_pcts.fillna(0).sum(axis=1).replace(0., np.nan), axis=0)

        cats = self.categories()
        ix = pd.Index(self.params['regions'], name='region_id')
        cols = pd.MultiIndex.from_product([self.cols_unit, cats], names=[None, 'party'])

        df = pd.concat([prev_pcts, vpred_pcts], axis=1, keys=self.cols_unit).loc[ix, cols].round(2)

        return df

    def simulate(
        self,
        frame: Optional[pd.DataFrame] = None,
        umat: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Allocate the seats of one simulation. With `split=True`, D'Hondt with the legal threshold in each
        province over the provincial shares of `umat`; otherwise the national seats estimator.
        The residual `'-'` never gets seats.
        """
        if frame is None:
            frame = self.build_frame()

        result = pd.DataFrame(index=self.params['regions'], columns=self.params['names'], dtype=float)

        if self.params['split']:
            if umat is None:
                umat = self.build_umat(frame)

            for region in self.params['regions']:
                if region == self.default_region:
                    continue

                # Shares over valid votes (already summing 100 with the residual, see `build_umat`)
                shares = umat.loc[region]['vpred_pct'].fillna(0)
                valid = float(self.prev_totals.loc[region, 'votes'])
                d_votes = (shares[self.params['names']] * valid / 100).round().to_dict()
                n_seats = int(self.reg_totals.loc[region]['seats'])

                result.loc[region] = self.alloc_seats(d_votes, n_seats, valid_votes=valid, threshold=self.threshold)

            result.loc[self.default_region] = result.loc[result.index != self.default_region].sum(axis=0)
        else:
            p = frame.loc[self.params['names'], ['vpred', 'regional']].fillna(0).values

            result.loc[self.default_region] = self.v2seats.predict(p).clip(0).values

        return result

    def run(self, reset: bool = True, **kwargs) -> Self:
        """
        Run `n_sim` simulations. Each one draws the national shares (`build_frame`), projects them to the
        provinces (`build_umat`) and allocates the seats (`simulate`), storing `frames`, `units` and `results`.

        Params (`set_params`): `n_sim`, `split`, `random`, `names`, `regions`, and the horizon of the
        simulated election: `horizon=None` (nowcast: the election is held now, only the polling error),
        an integer (days from `as_of`, adding the drift of the opinion), `'deadline'` (the legislature runs
        to its end) or `'random'` (each simulation draws its horizon from `date_prior`: `'historical'`,
        `'uniform'` or an array of days, see `horizon_candidates`). The horizons are stored in `horizons`.
        """
        self.set_params(reset=reset, **kwargs)

        cats = self.categories()
        self.horizons = self.build_horizons()

        self.frames = np.zeros((self.params['n_sim'], len(cats), len(self.cols_frame)))
        self.units = np.zeros((self.params['n_sim'], len(self.params['regions']), len(cats), len(self.cols_unit)))
        self.results = np.zeros((self.params['n_sim'], len(self.params['regions']), len(self.params['names'])))

        if not self.params['split'] and self.v2seats is None:
            if self.verbose > 0:
                print('Seats estimator not available')

            return self

        _iters = tqdm(np.arange(self.params['n_sim'])) if self.verbose > 0 else np.arange(self.params['n_sim'])
        for i in _iters:
            frame = self.build_frame(horizon=int(self.horizons[i]))
            umat = self.build_umat(frame)
            units = np.stack([umat[m].loc[self.params['regions'], cats].values for m in self.cols_unit], axis=-1)

            result = self.simulate(frame, umat)

            self.frames[i] = frame.loc[cats, self.cols_frame].values
            self.units[i] = units
            self.results[i] = np.array(result)

        self.frames = self.frames.round(2).astype(float)
        self.units = self.units.round(2).astype(float)
        self.results = self.results.round().astype(int)

        return self

    def get_path(self, name=None):
        if name:
            return '{}/{}'.format(self.path, name)
