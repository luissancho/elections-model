import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from typing import Any, Optional
from typing_extensions import Self

from ..core.app import Core
from ..core.utils.stat import Stat
from ..core.utils.helpers import unset_categorical
from ..core.utils.dataviz import plot_kde_1d

from .forecaster import Forecaster
from .computer import Computer
from .data import (
    get_event_dates, get_event_results, get_event_data, get_parties
)
from .utils import (
    norm_range
)


class Simulator(Core):

    def __init__(
        self,
        scope: str,
        event_date: str,
        drop_mtypes: Optional[list[str]] = ['aggr', 'online'],
        drange: Optional[tuple[int, int] | int] = None,
        alpha: float = 0.05,
        limit_date: Optional[str] = None,
        n_last: int = 1,
        add_errors: bool = True,
        reg_params: Optional[dict[str, Any]] = None,
        smap: Optional[dict[str, str]] = None,
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
        seed : int, optional
            Base random seed.
        verbose : int, optional
            Level of verbosity.
        path : str, optional
            Path to store model files.
        """
        super().__init__()

        self.scope = scope
        self.event_date = event_date
        self.drop_mtypes = drop_mtypes

        self.drange = norm_range(drange)
        self.alpha = alpha
        self.n_last = n_last
        self.add_errors = add_errors
        self.reg_params = reg_params
        self.smap = smap

        self.seed = seed  # Base random seed
        self.verbose = verbose  # Print progress
        self.path = path or os.getcwd()  # Path to store model files

        self.event_params = json.loads(
            self.app.fs.read(f'{self.path}/params.json')
        )[self.scope][self.event_date]

        self.names = self.event_params['parties']['event']

        if limit_date is not None:
            self.limit_date = limit_date
        else:
            self.limit_date = (
                pd.to_datetime(self.event_date) - pd.DateOffset(days=self.drange[0])
            ).strftime('%Y-%m-%d')

        if self.smap is None:
            self.smap = self.event_params.get('smap', {})

        for key, rules in self.smap.items():
            if isinstance(rules, dict):
                self.smap[key] = [rules]

        self.model = Forecaster(
            scope=self.scope,
            event_date=self.event_date,
            drange=self.drange,
            bmap=self.names,
            drop_mtypes=self.drop_mtypes,
            reg_params=self.reg_params,
            alpha=self.alpha,
            verbose=self.verbose,
            path=self.path
        ).build_series()

        event_dates = get_event_dates(
            scope=self.scope,
            date_from='1980-01-01',
            date_to=self.event_date,
            skip=1
        )
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
        self.cols_frame = self.cols_forecast + ['pct', 'pct_err', 'std_err', 'rand', 'vpred']
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
            print('Load region totals...')

        self.default_region = self.scope
        self.reg_totals = self.get_reg_totals()

        self.regions = self.reg_totals.index.tolist()
        self.default_params = {
            'n_sim': 1,
            'split': False,
            'random': False,
            'names': None,
            'regions': None
        }

        if self.verbose > 0:
            print('Load previous results...')

        self.prev_results = self.get_prev_results()

        self.params = None
        self.rng = None
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
        df['region'] = unset_categorical(df['region']).fillna(self.default_region)

        return df.set_index('region')[['votes', 'seats']]
    
    def get_prev_results(self) -> pd.DataFrame:
        """
        Get the results obtained by each party in the last election.

        Returns
        -------
        pd.DataFrame
            A DataFrame with previous results for each party at each region.
        """
        prev_date = self.model.nfc_series.index[0].strftime('%Y-%m-%d')

        df = get_event_results(self.scope, prev_date)
        df = df.loc[df['party_id'] > 0]
        df['region'] = unset_categorical(df['region']).fillna(self.default_region)

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

        df = df.sort_values(['region_id', 'party_id']).groupby(['region', 'party'], sort=False, observed=True, dropna=False)[[
            'votes', 'pct', 'seats'
        ]].sum()
        df = df.reset_index().pivot_table(columns='party', index='region', sort=False, observed=True, dropna=False)

        ix = pd.Index(self.regions, name='region')
        cols = pd.MultiIndex.from_product([['votes', 'pct', 'seats'], names], names=[None, 'party'])

        for c in cols:
            if c not in df.columns:
                df.loc[:, c] = .0

        df = df.where(df > 0, np.NaN).loc[ix, cols]

        return df
    
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

    def build_forecast(self) -> pd.DataFrame:
        weeks = (self.model.drange[0] + 1) // 7

        fc = pd.DataFrame({
            n: (
                self.model.forecast[n].loc[self.limit_date],
                self.model.fc_stat[n].loc[self.limit_date]['err'],
                self.model.fc_stat[n].loc[self.limit_date]['nobs']
            ) for n in self.params['names']
        }, index=['mean', 'err', 'nobs']).T

        fc['regional'] = fc.index.map(
            self.parties.set_index('name').loc[fc.index].regional.to_dict()
        ).astype(int)

        if self.v2err is not None:
            p = np.column_stack([
                fc[['mean', 'regional']].values,
                np.repeat(weeks, fc.shape[0])
            ])
            fc['error'] = self.v2err.predict(p).values
        else:
            fc['error'] = fc['err']

        fc = fc[self.cols_forecast]

        return fc

    def frame(self, loc: int = 0) -> pd.DataFrame:
        """
        Get a parameter frame for a given location.

        Parameters
        ----------
        loc : int, optional
            The location index.

        Returns
        -------
        pd.DataFrame
            The simulation results for the given location.
        """

        return pd.DataFrame(
            self.frames[loc],
            columns=self.cols_frame,
            index=self.params['names'],
            dtype=float
        )

    def unit(self, loc: int = 0, region: str = None) -> pd.DataFrame:
        rloc = self.params['regions'].index(region) if region is not None else 0

        return pd.DataFrame(
            self.units[loc, rloc],
            columns=self.cols_unit,
            index=self.params['names'],
            dtype=float
        )

    def result(self, loc: int = 0) -> pd.DataFrame:
        if self.results is None:
            return

        return pd.DataFrame(
            self.results[loc],
            columns=self.params['names'],
            index=self.params['regions'],
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

    def totals(self, sort: bool = False) -> pd.Series:
        if self.results is None:
            return

        n_seats = self.reg_totals.loc[self.default_region]['seats']
        medians = self.dist().apply(lambda x: Stat(x).median(), axis=0).sort_values(ascending=False)

        ds = medians.apply(np.floor).astype(int)
        diff = n_seats - ds.sum()
        remainders = medians - ds

        if diff > 0:
            idx_sorted = np.argsort(-remainders)

            # Calculate how many times we need to cycle through all parties
            cycles = int(diff // len(idx_sorted))
            remaining = int(diff % len(idx_sorted))
            
            # Add full cycles first
            if cycles > 0:
                ds += cycles
            
            # Then distribute any remaining seats
            for i in idx_sorted[:remaining]:
                ds[i] += 1
        elif diff < 0:
            idx_sorted = np.argsort(remainders)

            # Similar logic for negative diff
            cycles = int(abs(diff) // len(idx_sorted))
            remaining = int(abs(diff) % len(idx_sorted))
            
            # Subtract full cycles first
            if cycles > 0:
                ds -= cycles
            
            # Then remove any remaining seats
            for i in idx_sorted[:remaining]:
                ds[i] -= 1

        if sort:
            ds = ds.sort_values(ascending=False)
        else:
            ds = ds.loc[self.params['names']]

        return ds
    
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
        names = [
            n for n in self.totals().sort_values(ascending=False).index
            if self.forecast.loc[n].regional == int(regional)
        ]

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

    def build_frame(self) -> pd.DataFrame:
        """
        Build a feature frame to be used as input for the simulation.

        Each row represents the feature space for a single party, with its forecast result
        and an estimation of vote percentage for each simulation.

        Returns
        -------
        pd.DataFrame
            The feature frame for the simulation.
        """

        weeks = (self.model.drange[0] + 1) // 7

        # Initialize the feature frame with the forecasted results for each party
        df = pd.DataFrame(columns=(self.cols_frame), index=self.params['names'], dtype=float)
        df[self.cols_forecast] = self.forecast.loc[self.params['names']].values

        df['pct'] = df['mean']  # Estimated global vote percentage calculated by the forcaster for this party
        df['regional'] = df['regional'].astype(int)  # Is the party a regional party (only representing a certain region)?
        vind = df['pct'] > 0  # Only parties with a forecasted percentage greater than 0 are considered

        # If random is set to True, add noise to the forecasted percentage, based on the forcasted error
        if self.params['random']:
            if self.v2err is not None:
                # If an error estimator is available, use it to estimate the error of the forecasted percentage
                # This estimator uses historical data to estimate the error of the forecasted percentage
                # See Computer.get_error_estimator() for more details
                p = np.column_stack([
                    df.loc[vind, ['pct', 'regional']].fillna(0).values,
                    np.repeat(weeks, df.loc[vind].shape[0])
                ])
                df.loc[vind, 'pct_err'] = self.v2err.predict(p).values
                df.loc[vind, 'std_err'] = np.sqrt(np.square(df.loc[vind, 'err']) + np.square(df.loc[vind, 'pct_err']))
            else:
                # If no error estimator available, use the purely statistical error provided by the forcaster
                df.loc[vind, 'std_err'] = df.loc[vind, 'err']

            # Add noise to the forecasted percentage, based on the calculated error
            df.loc[vind, 'rand'] = np.clip(
                df.loc[vind, 'pct'] + df.loc[vind, 'std_err'] * self.rng.standard_t(df.loc[vind, 'nobs'] - 2),
                0, None
            )

            df['vpred'] = df['rand']
        else:
            df['vpred'] = df['pct']

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
                        fc_k = frame.loc[key]['vpred']
                        fc_v = frame.loc[names]['vpred'].sum()
                        prev_v = prev_pcts[names].sum(axis=1)

                        prev_pcts[key] = np.round(prev_v / ((fc_v / fc_k) + 1), 2)
                        unit_factor = prev_pcts[key] / len(names)
                        prev_pcts[names] = prev_pcts[names].sub(unit_factor, axis=0)
                    elif type == 'split':
                        prev_k = prev_pcts[key]
                        for name in names:
                            prev_pcts[name] += (prev_k / len(names))
                
                if regions is not None:
                    if 'exclude' in regions:
                        rix = [r == self.default_region or r not in regions['exclude'] for r in prev_pcts.index]
                    else:
                        rix = [r == self.default_region or r in regions for r in prev_pcts.index]

                    prev_pcts[key] = prev_pcts[key].where(rix, .0)

        prev_pcts = prev_pcts.where(prev_pcts > 0, np.NaN)[self.params['names']]

        total_votes = self.prev_results['votes'].fillna(0).sum(axis=1).astype(int)
        n_votes = total_votes.loc[self.default_region]
        prev_votes = prev_pcts.mul(total_votes / 100, axis=0).round()
        fc_votes = (frame['vpred'] * n_votes / 100).round().astype(int)

        fmul = fc_votes[self.params['names']] / prev_votes.loc[self.default_region][self.params['names']]
        vpred_pcts = prev_pcts.mul(fmul)

        metrics = ['prev_pct', 'vpred_pct']
        ix = pd.Index(self.params['regions'], name='region')
        cols = pd.MultiIndex.from_product([metrics, self.params['names']], names=[None, 'party'])

        df = pd.concat([prev_pcts, vpred_pcts], axis=1, keys=metrics).loc[ix, cols].round(2)

        return df

    def simulate(
        self,
        frame: Optional[pd.DataFrame] = None,
        umat: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if frame is None:
            frame = self.build_frame()

        result = pd.DataFrame(index=self.params['regions'], columns=self.params['names'], dtype=float)

        if self.params['split']:
            if umat is None:
                umat = self.build_umat(frame)

            for region in self.params['regions']:
                if region == self.default_region:
                    continue

                d_pcts = umat.loc[region]['vpred_pct'].fillna(0)
                d_adj = d_pcts * 100 / d_pcts.sum()

                total_votes = self.prev_results['votes'].fillna(0).sum(axis=1).astype(int)
                d_votes = d_adj.mul(total_votes.loc[region] / 100, axis=0).round().to_dict()
                n_seats = int(self.reg_totals.loc[region]['seats'])

                result.loc[region] = self.alloc_dhondt(d_votes, n_seats)

            result.loc[self.default_region] = result.loc[result.index != self.default_region].sum(axis=0)
        else:
            p = frame[['vpred', 'regional']].fillna(0).values

            result.loc[self.default_region] = self.v2seats.predict(p).clip(0).values

        return result

    def run(self, reset: bool = True, **kwargs) -> Self:
        self.set_params(reset=reset, **kwargs)

        self.frames = np.zeros((self.params['n_sim'], len(self.params['names']), len(self.cols_frame)))
        self.units = np.zeros((self.params['n_sim'], len(self.params['regions']), len(self.params['names']), len(self.cols_unit)))
        self.results = np.zeros((self.params['n_sim'], len(self.params['regions']), len(self.params['names'])))

        if not self.params['split'] and self.v2seats is None:
            if self.verbose > 0:
                print('Seats estimator not available')

            return self

        _iters = tqdm(np.arange(self.params['n_sim'])) if self.verbose > 0 else np.arange(self.params['n_sim'])
        for i in _iters:
            frame = self.build_frame()
            umat = self.build_umat(frame)
            units = umat.stack(level=0, future_stack=True).unstack(level=1).values.reshape(
                len(self.params['regions']), len(self.params['names']), len(self.cols_unit)
            )

            result = self.simulate(frame, umat)

            self.frames[i] = np.array(frame)
            self.units[i] = np.array(units)
            self.results[i] = np.array(result)

        self.frames = self.frames.round(2).astype(float)
        self.units = self.units.round(2).astype(float)
        self.results = self.results.round().astype(int)

        return self

    def get_path(self, name=None):
        if name:
            return '{}/{}'.format(self.path, name)
