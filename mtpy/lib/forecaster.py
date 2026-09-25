from ast import literal_eval
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
from tqdm import tqdm

from typing import Any, Optional
from typing_extensions import Self

from ..core.app import Core
from ..core.utils.dates import add_delta, ts_from_delta
from ..core.utils.helpers import format_number
from ..core.utils.stat import LocalKernelEstimator, Kernel
from ..core.utils.dataviz import (
    build_color_seq_map, create_figure, get_color, get_df_styler, get_params, get_text_color,
    plot_figure, plot_series, print_styler, set_note, set_num_locator,
    set_title, table_styles
)

from .utils import normal_update
from .data import (
    get_event_dates, get_event_params, get_event_series, get_poll_series, get_parties, get_pollsters,
    get_house_effects
)
from .utils import (
    build_blocks, group_results, norm_range
)

class Forecaster(Core):

    def __init__(
        self,
        scope: str,
        event_date: str,
        drop_mtypes: Optional[list[str]] = ['aggr', 'online'],
        drange: Optional[tuple[int, int] | int] = None,
        alpha: float = 0.05,
        bmap: Optional[dict[str, Any] | str] = None,
        reg_params: Optional[dict[str, Any]] = None,
        house_effects: bool = False,
        he_params: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        path: Optional[str] = None
    ) -> None:
        """
        Polls corresponding to a single election event.

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
        bmap : dict or str, optional
            A dictionary mapping the parties and blocks to be included in the analysis.
            Instead of analizing the results of each party individually, we might want to group them into blocks.
            If a string is provided, it will select the corresponding map from the event params.
            If `None`, all the parties will be independently analized.

            Examples:
            {
                'PSOE': 'PSOE',
                'PP': 'PP'
            } -> Only the most relevant parties.
            {
                'Derecha': ['PP', 'VOX', 'Cs'],
                'Izquierda': ['PSOE', 'UP', 'SUMAR', 'MP'],
                'Separatistas': ['ERC', 'JxCat', 'EHB', 'CUP'],
                'Regionalistas': ['PNV', 'CC', 'BNG', 'PRC', 'TE', 'EV', 'UPN']
            } -> Main blocks of parties.
            {
                'PP': 'PP',
                'PSOE': 'PSOE',
                'VOX': 'VOX',
                'SUMAR': ['SUMAR', 'UP', 'MP']
            } -> Includes a coalition block that aggregates the results of several parties.
        reg_params : dict of str, optional
            Parameters for the regression estimator.
            If `None`, the default parameters will be used.
        house_effects : bool, optional
            Estimate the house effect of each pollster on each series (backfitting against the average, with a
            prior from its past elections) and subtract it from its polls before averaging. See `fit_house_effects`.
        he_params : dict, optional
            Parameters of the house effects estimation, see `set_he_params`.
        verbose : int, optional
            Level of verbosity.
        path : str, optional
            Path where the model outputs (forecasts, figures) are stored, relative to the app file system root
            (`files/`). Input data (`params.json`, maps) is always read from the versioned `data/` directory.
        """
        super().__init__()

        self.scope = scope
        self.event_date = event_date
        self.drop_mtypes = drop_mtypes

        self.drange = norm_range(drange)
        self.alpha = alpha
        self.bmap = bmap
        self.reg_params = self.set_reg_params(reg_params)
        self.he_enabled = bool(house_effects)  # Estimate and subtract the house effects before averaging (M6)
        self.he_params = self.set_he_params(he_params)

        self.verbose = verbose
        self.path = path or '.'  # Path to the model files, relative to the app file system root (files/)

        # Event params: derived from the data (parties with results, blocks, new parties) and overridden by
        # the entries of `data/params.json` when the event is listed there
        self.event_params = get_event_params(self.scope, self.event_date, path=self.path)

        if isinstance(self.bmap, str):
            self.bmap = self.event_params['bmaps'][self.bmap]
        elif not isinstance(self.bmap, dict):
            self.bmap = self.event_params['parties']['polls']

        self.blocks = None  # Map of blocks and their corresponding party members
        self.names = None  # List of block names included
        self.parties = None  # List of parties included in the polls published for this election event
        self.pollsters = None  # List of pollsters with polls published for this election event
        self.colors = None  # Colors of the parties included in the polls published for this election event

        self.series = None  # DataFrame to build containing the series of polls, weights and results
        self.series_raw = None  # The same series before subtracting the house effects (see `fit_house_effects`)
        self.house_effects = None  # House effect of each pollster on each series, once fitted
        self.forecast = None  # Fitted estimation of the percentage of votes for each party in the election event
        self.fc_stat = None  # Standard error and confidence interval of the forecast for each estimation

        self.date_start = None  # Date of the last election event
        self.date_first = None  # Date of the first poll published for the current election event
        self.date_last = None  # Date of the last poll published for the current election event
        self.date_end = None  # Date of the current election event
        self.date_fit_last = None  # Last day with a fitted value (before any forward fill), see `fit_forecast`

    @property
    def bmaps(self) -> dict[str, Any]:
        return self.event_params['bmaps']

    @property
    def nfc_series(self) -> pd.DataFrame:
        return self.series.loc[self.series.pollster.isnull()].reset_index(self.series.index.names[1:])

    @property
    def fc_series(self) -> pd.DataFrame:
        return self.series.loc[self.series.pollster.notnull()].reset_index(self.series.index.names[1:])

    @property
    def fc_index(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.date_start, end=self.date_end, freq='D')

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

    def set_he_params(
        self,
        he_params: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Normalize the parameters of the house effects estimation (see `fit_house_effects`).

        Parameters
        ----------
        he_params : dict, optional
            - n_iter : backfitting iterations (3).
            - min_polls : polls a pollster needs in the cycle to inform its effect (5).
            - tau : prior standard deviation of an effect, relative to the level of the series (0.08).
            - prior_events : shrinkage of the historical prior toward 0, in elections of `n_cap` polls (1).
            - n_cap : cap of the polls of a past election counted in the prior weight (10).
            - year_decay : yearly decay of the weight of a past election, and of a poll within the cycle (0.9).
            - n_days : window before a past election used to measure the deviation from its result (90).
            - level_floor : floor of the level in the relative deviations, percentage points (2).
            - rel_cap : cap of the absolute relative deviation of a past election (0.5).
            - err_floor : floor of the standard error of a measured deviation, as a fraction of the prior sd (0.25).
            - prior : `'auto'` (history from the database) or `None` (prior 0), see `fit_house_effects`.
        """
        he_params = he_params if he_params is not None else dict()

        return {
            'n_iter': int(he_params.get('n_iter', 3)),
            'min_polls': int(he_params.get('min_polls', 5)),
            'tau': float(he_params.get('tau', 0.08)),
            'prior_events': float(he_params.get('prior_events', 1.)),
            'n_cap': int(he_params.get('n_cap', 10)),
            'year_decay': float(he_params.get('year_decay', 0.9)),
            'n_days': int(he_params.get('n_days', 90)),
            'level_floor': float(he_params.get('level_floor', 2.)),
            'rel_cap': float(he_params.get('rel_cap', 0.5)),
            'err_floor': float(he_params.get('err_floor', 0.25)),
            'prior': he_params.get('prior', 'auto')
        }

    def load_events(self) -> pd.DataFrame:
        event_dates = get_event_dates(scope=self.scope, date_to=self.event_date)

        if len(event_dates) >= 2:
            self.date_end = pd.Timestamp(event_dates[-1])
            self.date_start = pd.Timestamp(event_dates[-2])
        elif len(event_dates) == 1:
            self.date_end = pd.Timestamp(event_dates[0])
            self.date_start = self.date_end - pd.DateOffset(days=6) - pd.DateOffset(months=12)
        else:
            raise ValueError('No event dates found')

        events = get_event_series(
            scope=self.scope,
            event_dates=[
                self.date_start.strftime('%Y-%m-%d'),
                self.date_end.strftime('%Y-%m-%d')
            ]
        )
        events['sample_size'] = events['votes']
        events['computed'] = True

        return events.set_index('date').sort_index()

    def load_polls(self) -> pd.DataFrame:
        polls = get_poll_series(
            scope=self.scope,
            event_dates=[
                self.date_end.strftime('%Y-%m-%d')
            ],
            drange=self.drange,
            drop_mtypes=self.drop_mtypes
        )

        self.date_first, self.date_last = polls.date.agg(['min', 'max']).tolist()

        return polls.set_index(['date', 'pollster_id', 'sponsor_id']).sort_index()

    def build_series(self) -> Self:
        """
        Build the series of polls, weights and results for the election event.
        """
        if self.verbose > 0:
            print('Load events...')

        events = self.load_events()

        if self.verbose > 0:
            print('Load polls...')

        polls = self.load_polls()

        if self.verbose > 0:
            print('Load parties...')

        parties = get_parties()
        self.parties = parties[parties.name.isin(polls.columns)]
        party_names = self.parties.name.tolist()
        self.colors = self.parties.set_index('name')['color'].to_dict()

        if self.verbose > 0:
            print('Load pollsters...')

        pollsters = get_pollsters()
        self.pollsters = pollsters[pollsters.name.isin(polls.pollster.unique())]

        if self.verbose > 0:
            print('Build series...')

        data_cols = [
            'tfs', 'tte', 'start_date', 'end_date', 'pollster', 'sponsor', 'computed',
            'sample_size', 'parties', 'days', 'mtype', 'proc_sample', 'rating',
            'error_avg', 'error_blocks', 'error_within', 'bias_avg', 'bias_blocks', 'bias_within', 'bias', 'bias_dev_adj', 'bias_dev_err',
            'weight_over', 'weight_sample', 'weight_rating', 'weight'
        ]

        # Initialize series by concatenating events and polls
        series = pd.concat([
            polls.reset_index(),
            events.reset_index()
        ], ignore_index=True, sort=False)

        # Set days between start and end dates for each poll
        series['tfs'] = [(dt - self.date_start).days for dt in series.date]  # tfs: time from start
        series['tte'] = [(self.date_end - dt).days for dt in series.date]  # tte: time to end

        # Set poll weights as the product of the all computed weights
        # Pollsters without a computed rating fall back to their prior `quality` (0-100), mapped onto the
        # same scale as `weight_rating` (see `Computer.pollster_ratings`)
        quality = self.pollsters.set_index('id').quality.astype(float).div(100)
        weight_quality = np.log1p(quality) / np.log1p(quality.mean())
        series['weight_rating'] = series.weight_rating.fillna(series.pollster_id.map(weight_quality))
        series['weight'] = series.weight_over * series.weight_sample * series.weight_rating

        # Sort index and columns
        series = series.set_index([
            'date', 'pollster_id', 'sponsor_id'
        ]).sort_index()[
            data_cols + party_names
        ]

        # Build blocks using the defined mapping and aggregate the results of each group of parties
        if self.bmap is None:
            self.bmap = party_names
        blocks = build_blocks(self.bmap, self.colors)
        block_results = group_results(series, blocks=blocks)
        self.blocks = blocks.loc[blocks.index.isin(block_results.columns)]
        self.names = self.blocks.index.tolist()

        # Replace the parties results with the blocks results, then sort index and columns
        series = pd.concat([
            series[data_cols],
            block_results
        ], axis=1)[
            data_cols + self.names
        ].sort_index()

        self.series_raw = None
        self.house_effects = None
        self._set_series(series)

        return self

    def _set_series(self, series: pd.DataFrame) -> Self:
        """
        Set the working series (blocks): drop the polls with incomplete results, recompute the residual
        `'-'` and reset the forecast frames. The first series set is kept as `series_raw` (uncorrected).
        """
        series = series.loc[series.pollster.isnull() | series.computed.fillna(False).astype(bool)].copy()
        # Assign the remaining percentage to a new 'others' block
        series['-'] = 100. - series[self.names].sum(axis=1, min_count=1)
        self.series = series
        if self.series_raw is None:
            self.series_raw = series.copy()

        # Initialize the forecast and statistics frames
        self.forecast = pd.DataFrame(
            columns=self.names + ['-'],
            index=pd.date_range(start=self.date_start, end=self.date_end, freq='D', name='date'),
            dtype=float
        )
        self.fc_stat = pd.DataFrame(
            columns=self.names,
            index=self.forecast.index
        )

        return self

    def fit(
        self,
        name: str,
        max_fc: Optional[int] = 0,
        ret_stat: bool = False
    ) -> pd.Series | tuple[pd.Series, pd.Series]:
        """
        Fit the local kernel estimator for a single party or block.

        Parameters
        ----------
        name : str
            Name of the party or block to be fitted.
        max_fc : int, optional
            Maximum number of days to forecast after last poll date.
        ret_stat : bool, optional
            Return the standard error and confidence interval of the forecast for each estimation.

        Returns
        -------
        pd.Series or tuple
            Fitted estimation of the percentage of votes for the party or block in the election event.
            If `ret_stat` is `True`, returns a tuple with the standard error and confidence interval of the forecast.
        """
        # Polls with a value for the party and a positive weight (zero-weight polls do not enter the fit and
        # would only break the bandwidth selection); a local linear fit needs at least three of them
        df = self.fc_series[self.fc_series[name].notnull() & (self.fc_series['weight'] > 0)]

        if df.shape[0] < 3:
            warnings.warn('Fewer than three usable polls for `{}`: not fitted'.format(name))
            return

        ix = self.fc_index
        px = ix[(ix >= df.index.min()) & (ix <= add_delta(df.index.max(), f'P{max_fc}D'))]

        weights = df.weight.values
        alpha = self.alpha if ret_stat else None

        reg = LocalKernelEstimator(
            df[name],
            weights=weights,
            **self.reg_params
        ).fit(px, alpha=alpha).reindex(ix)

        if ret_stat:
            dreg = reg['mean']
            dstat = pd.Series(reg.to_dict(orient='index')).where(reg['mean'].notnull(), None)

            return dreg, dstat

        return reg

    def fit_forecast(
        self,
        names: Optional[list[str]] = None,
        max_fc: Optional[int] = 0,
        fillna: bool = False
    ) -> pd.DataFrame:
        """
        Fit the local kernel estimator for each party or block.

        Parameters
        ----------
        names : list of str, optional
            List of parties or blocks to be fitted.
            If `None`, all the parties and blocks will be fitted.
            If a party or block is not included in the analysis, it will be ignored.
        max_fc : int, optional
            Maximum number of days to forecast after last poll date.
        fillna : bool, optional
            Fill the missing values with the last available estimation.

        Returns
        -------
        pd.DataFrame
            Fitted estimation of the percentage of votes for each party or block in the election event.

        Notes
        -----
        `date_fit_last` records the last day with a fitted value for any party (`last poll + max_fc`) before the
        forward fill, so that a forecast read at a later date can be traced back to the day it was actually
        estimated. Parties fitted on fewer polls may end earlier and keep their last value from that day on.
        """
        names = names or self.names

        # House effects (M6): estimated once on every series and subtracted from the polls before averaging
        if self.he_enabled and self.house_effects is None:
            self.fit_house_effects()

        names_ = tqdm(names) if self.verbose > 0 else names  # Show progress bar if verbose
        for name in names_:
            res = self.fit(name, max_fc=max_fc, ret_stat=True)
            if res is None:
                warnings.warn('No polls available for `{}`: skipped'.format(name))
                continue

            dreg, dstat = res
            self.forecast.loc[dreg.index, name] = dreg
            self.fc_stat.loc[dstat.index, name] = dstat

        # Last day actually estimated for any party: the anchor of a nowcast read after the forward fill
        fitted = self.forecast[self.names].notnull().any(axis=1)
        self.date_fit_last = fitted[fitted].index.max() if fitted.any() else None

        # Assign the remaining percentage to the 'others' block
        self.forecast['-'] = 100. - self.forecast[self.names].sum(axis=1, min_count=1)

        # Populate the forecast and statistics frames, filling the missing values with the last available estimation
        if fillna:
            self.forecast = self.forecast.ffill()
            # `fc_stat` holds a dict per cell (object dtype): forward-fill column by column, skipping the
            # columns that were not fitted, to avoid pandas' object downcasting on all-NaN columns
            self.fc_stat = self.fc_stat.apply(lambda col: col.ffill() if col.notnull().any() else col)

        return self.forecast

    def get_forecast(
        self,
        sort: bool = False,
        date: Optional[str] = None,
        prefix: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get the fitted estimation of the percentage of votes for each party or block at a specific date.

        Parameters
        ----------
        sort : bool
            Sort the results by descending percentage.
        date : str or datetime, optional
            Date of the forecast.
            If `None`, the last available date will be used.
        prefix : str, optional
            If specified, the forecast will be loaded from the specified prefix file.

        Returns
        -------
        pd.DataFrame
            Fitted estimation at the specified date.
        """
        if prefix is not None:
            if self.verbose > 1:
                print('Load forecast...')

            self.load_forecast(prefix)
        elif self.forecast.isnull().all().all():
            if self.verbose > 1:
                print('Fit forecast...')

            self.fit_forecast()

        # Get the last available date if no date is specified
        if date is None:
            date = self.forecast.dropna(how='all', axis=0).index[-1]

        forecast = self.forecast.loc[date].dropna().rename('pct').round(2)

        if sort:
            forecast = forecast.sort_values(ascending=False)

        return forecast

    def save_forecast(self, prefix: str) -> Self:
        self.app.fs.write_csv(
            self.forecast.reset_index(),
            self.get_path('fc/{}.csv'.format(prefix))
        )
        self.app.fs.write_csv(
            self.fc_stat.reset_index(),
            self.get_path('fc/{}-stat.csv'.format(prefix))
        )

        return self

    def load_forecast(self, prefix: Optional[str] = None) -> pd.DataFrame:
        if prefix is not None:
            self.forecast = self.app.fs.read_csv(
                self.get_path('fc/{}.csv'.format(prefix))
            ).set_index('date').astype(float)
            self.forecast.index = pd.DatetimeIndex(self.forecast.index)

            self.fc_stat = self.app.fs.read_csv(
                self.get_path('fc/{}-stat.csv'.format(prefix))
            ).set_index('date').map(lambda x: literal_eval(x) if isinstance(x, str) else x)
            self.fc_stat.index = pd.DatetimeIndex(self.fc_stat.index)
            self.date_fit_last = None  # A saved forecast is already forward filled: the anchor falls back to `date_last`
        else:
            self.fit_forecast()

        return self.forecast

    # --- House effects (M6) ------------------------------------------------------------------------------

    def load_house_history(self) -> pd.DataFrame:
        """
        Historical deviations of the pollsters in the elections before this one (table `pollsters_parties`,
        see `Computer.compute_house_effects`): the source of the prior of each house effect.
        """
        dates = get_event_dates(scope=self.scope, date_to=self.event_date, skip=1)

        return get_house_effects(scope=self.scope, event_dates=dates)

    def fit_house_effects(
        self,
        n_iter: Optional[int] = None,
        min_polls: Optional[int] = None,
        prior: Optional[str | pd.DataFrame] = 'default'
    ) -> pd.DataFrame:
        """
        Estimate the house effect of each pollster on each series and subtract it from its polls.

        Backfitting: the average of each series is fitted on the corrected polls of the previous iteration
        (the raw polls the first time); the deviation of each pollster is the weighted mean of the residuals
        of its raw polls against that average (`house_deviations`, precision weights `weight_over ·
        weight_sample` decayed with the age of the poll); it is combined with the prior of the pollster on
        that series (`house_prior`: its centred deviations from the results of past elections, or 0) by a
        normal-normal update (`normal_update`); the effects are re-centred so that their weighted mean is 0
        (`center_effects`: the level of the average never depends on the correction) and subtracted from the
        raw polls (`apply_house_effects`). Rows of official results are never touched; `series_raw` keeps
        the uncorrected series and `house_effects` the table of effects.

        Parameters
        ----------
        n_iter : int, optional
            Backfitting iterations (`he_params['n_iter']` by default).
        min_polls : int, optional
            Polls a pollster needs in the cycle to inform its effect (`he_params['min_polls']` by default).
        prior : {'auto'}, pd.DataFrame or None, optional
            `'auto'` loads the history from the database (`load_house_history`); a frame with the columns of
            `pollsters_parties` uses it directly; `None` uses a prior of 0 for every pollster (the effect is
            the deviation of the cycle alone). By default, `he_params['prior']`.

        Returns
        -------
        pd.DataFrame
            Indexed by `(pollster_id, name)`: `pollster`, `n`, `w`, `level`, `dev`, `dev_err`, `prior`,
            `prior_err`, `effect`, `effect_err`, `center`.
        """
        p = self.he_params
        n_iter = p['n_iter'] if n_iter is None else int(n_iter)
        min_polls = p['min_polls'] if min_polls is None else int(min_polls)
        if isinstance(prior, str) and prior == 'default':
            prior = p['prior']

        if isinstance(prior, str) and prior == 'auto':
            history = self.load_house_history()
        elif isinstance(prior, pd.DataFrame):
            history = prior
        else:
            history = None
        if history is not None and history.shape[0] > 0:
            history = history.loc[history['dev_result_c'].notnull()]
            history_groups = {k: g for k, g in history.groupby(['pollster_id', 'party'], observed=True)}
        else:
            history_groups = {}

        raw = self.series_raw
        is_poll = raw['pollster'].notnull()
        polls = raw.loc[is_poll].reset_index(raw.index.names[1:])
        pollster_names = polls.groupby('pollster_id', observed=True)['pollster'].first()

        # Precision weights of the polls, decayed with their age (the current methodology of a pollster
        # matters more than the one of the beginning of the cycle)
        age = (pd.Timestamp(self.date_last) - polls.index).days / 365.25
        weights = (
            polls['weight_over'].astype(float) * polls['weight_sample'].astype(float)
            * np.power(p['year_decay'], np.clip(age, 0, None))
        ).fillna(0.)

        ref_date = pd.Timestamp(self.date_end)
        names = list(self.names)
        effects = None
        current = raw

        for _ in range(max(n_iter, 1)):
            self._set_series(current)

            fitted = {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for name in names:
                    reg = self.fit(name, max_fc=0, ret_stat=False)
                    if reg is not None:
                        fitted[name] = reg
            if len(fitted) == 0:
                break
            fitted = pd.DataFrame(fitted)

            dev = self.house_deviations(polls, fitted, list(fitted.columns), weights, min_polls=min_polls)
            if dev.shape[0] == 0:
                break

            # Level of each series: mean of the fitted average over its last 30 days
            level = {n: float(np.nanmean(fitted[n].dropna().iloc[-30:])) if fitted[n].notnull().any() else 0. for n in fitted.columns}

            priors = np.array([
                self.house_prior(
                    history_groups.get((pid, name)), level[name], ref_date,
                    year_decay=p['year_decay'], he_tau=p['tau'], he_prior_events=p['prior_events'],
                    n_cap=p['n_cap'], level_floor=p['level_floor'], rel_cap=p['rel_cap']
                ) for pid, name in dev.index
            ])
            dev['level'] = [level[name] for _, name in dev.index]
            dev['prior'], dev['prior_err'] = priors[:, 0], priors[:, 1]
            # A handful of near-identical polls must not produce an overconfident deviation
            dev_err = np.maximum(dev['dev_err'].to_numpy(dtype=float), p['err_floor'] * dev['prior_err'].to_numpy(dtype=float))
            dev['effect'], dev['effect_err'] = normal_update(
                dev['dev'].to_numpy(dtype=float), dev_err, dev['prior'].to_numpy(dtype=float), dev['prior_err'].to_numpy(dtype=float)
            )

            effects = self.center_effects(dev)  # weighted by the weight of each pollster on each series
            current = self.apply_house_effects(raw, effects, names)

        self._set_series(current)

        if effects is None:
            effects = pd.DataFrame(
                columns=['n', 'w', 'dev', 'dev_err', 'level', 'prior', 'prior_err', 'effect', 'effect_err', 'center'],
                index=pd.MultiIndex.from_tuples([], names=['pollster_id', 'name'])
            )
        effects.insert(0, 'pollster', effects.index.get_level_values('pollster_id').map(pollster_names))
        self.house_effects = effects

        return effects

    # --- House effects (M6): pure helpers -----------------------------------------------------------------

    @staticmethod
    def industry_bias(
        history: pd.DataFrame,
        levels: pd.Series,
        ref_date: pd.Timestamp,
        year_decay: float = 0.9,
        prior_events: float = 1.,
        level_floor: float = 2.,
        rel_cap: float = 0.5
    ) -> pd.DataFrame:
        """
        Industry-wide bias of the polls on each series, from the past elections: the decayed mean over the
        elections of the relative industry deviation (`industry / level`, the mean error of every pollster
        against the result), shrunk toward 0, scaled to the current level; its uncertainty is the dispersion
        between elections (the bias itself when there is only one: no evidence of stability).

        Parameters
        ----------
        history : pd.DataFrame
            Rows of `pollsters_parties` (`event_date`, `party`, `industry`, `level`).
        levels : pd.Series
            Current level of each series (percentage points), indexed by name.
        ref_date : pd.Timestamp
            Date the ages are counted from.
        year_decay, prior_events, level_floor, rel_cap : see `house_prior`.

        Returns
        -------
        pd.DataFrame
            Indexed by name: `n_events`, `rel`, `rel_err`, `bias`, `bias_err` (percentage points).
        """
        rows = []
        if history is None or history.shape[0] == 0:
            hist = None
        else:
            hist = history.dropna(subset=['industry', 'level']).groupby(['event_date', 'party'], observed=True)[['industry', 'level']].first().reset_index()

        for name, level in levels.items():
            lvl = max(float(level) if np.isfinite(level) else 0., level_floor)
            h = hist.loc[hist['party'] == name] if hist is not None else None
            if h is None or h.shape[0] == 0:
                rows.append({'name': name, 'n_events': 0, 'rel': 0., 'rel_err': 0., 'bias': 0., 'bias_err': 0.})
                continue

            years = (pd.Timestamp(ref_date) - pd.to_datetime(h['event_date'])).dt.days / 365.25
            w = np.power(year_decay, years.clip(lower=0)).to_numpy()
            r = np.clip(h['industry'].astype(float) / np.maximum(h['level'].astype(float), level_floor), -rel_cap, rel_cap).to_numpy()
            rel = float((w * r).sum() / (w.sum() + prior_events))
            if len(r) > 1:
                rel_err = float(np.sqrt((w * np.square(r - rel)).sum() / w.sum()))
            else:
                rel_err = abs(rel)
            rows.append({'name': name, 'n_events': int(len(r)), 'rel': rel, 'rel_err': rel_err, 'bias': rel * lvl, 'bias_err': rel_err * lvl})

        return pd.DataFrame(rows, columns=['name', 'n_events', 'rel', 'rel_err', 'bias', 'bias_err']).set_index('name')

    @staticmethod
    def house_deviations(
        polls: pd.DataFrame,
        fitted: pd.DataFrame,
        names: list[str],
        weights: pd.Series | np.ndarray,
        min_polls: int = 5
    ) -> pd.DataFrame:
        """
        Deviation of each pollster from the fitted average, per series name: the weighted mean of the
        residuals `poll - average(date)` of its polls and the standard error of that mean.

        Parameters
        ----------
        polls : pd.DataFrame
            Polls indexed by date (duplicates allowed), with a `pollster_id` column and the `names` columns.
        fitted : pd.DataFrame
            Fitted average with a daily index and the `names` columns (NaN where not fitted).
        names : list of str
            Series to evaluate.
        weights : pd.Series or np.ndarray
            Precision weight of each poll (aligned with `polls` rows).
        min_polls : int, optional
            Pollsters with fewer residuals get an infinite standard error (no information).

        Returns
        -------
        pd.DataFrame
            Indexed by `(pollster_id, name)`, columns `n`, `w` (total weight), `dev`, `dev_err`.
        """
        w_all = np.asarray(weights, dtype=float)
        rows = []
        for name in names:
            if name not in polls.columns or name not in fitted.columns:
                continue

            resid = polls[name].to_numpy(dtype=float) - fitted[name].reindex(polls.index).to_numpy(dtype=float)
            frame = pd.DataFrame({'pollster_id': polls['pollster_id'].to_numpy(), 'r': resid, 'w': w_all})
            frame = frame.loc[np.isfinite(frame['r']) & np.isfinite(frame['w']) & (frame['w'] > 0)]

            for pid, g in frame.groupby('pollster_id'):
                w, r = g['w'].to_numpy(), g['r'].to_numpy()
                n = len(r)
                sw = w.sum()
                dev = float((w * r).sum() / sw)
                if n >= min_polls and n > 1:
                    neff = sw ** 2 / np.square(w).sum()
                    denom = sw - np.square(w).sum() / sw  # unbiased weighted variance (reliability weights)
                    var = float((w * np.square(r - dev)).sum() / denom) if denom > 0 else 0.
                    dev_err = float(np.sqrt(max(var, 0.) / neff))
                else:
                    dev_err = np.inf
                rows.append({'pollster_id': pid, 'name': name, 'n': n, 'w': float(sw), 'dev': dev, 'dev_err': dev_err})

        out = pd.DataFrame(rows, columns=['pollster_id', 'name', 'n', 'w', 'dev', 'dev_err'])

        return out.set_index(['pollster_id', 'name']).sort_index()

    @staticmethod
    def apply_house_effects(
        series: pd.DataFrame,
        effects: pd.DataFrame,
        names: list[str]
    ) -> pd.DataFrame:
        """
        Subtract the house effect of each pollster from its polls (rows with a pollster), series by series.
        Rows of official results are never touched. Returns a copy.

        Parameters
        ----------
        series : pd.DataFrame
            Forecaster series with `pollster`, `pollster_id` and the `names` columns.
        effects : pd.DataFrame
            Indexed by `(pollster_id, name)` with an `effect` column (percentage points).
        names : list of str
            Series to correct.
        """
        out = series.copy()
        is_poll = out['pollster'].notnull().to_numpy()
        if 'pollster_id' in out.index.names:
            pids = out.index.get_level_values('pollster_id').to_numpy()
        else:
            pids = out['pollster_id'].to_numpy()

        for name in names:
            if name not in out.columns:
                continue
            eff = effects.xs(name, level='name')['effect'] if name in effects.index.get_level_values('name') else None
            if eff is None or eff.empty:
                continue
            shift = pd.Series(pids).map(eff).fillna(0.).to_numpy(dtype=float)
            out[name] = out[name].to_numpy(dtype=float) - np.where(is_poll, shift, 0.)

        return out

    @staticmethod
    def house_prior(
        history: pd.DataFrame,
        level: float,
        ref_date: pd.Timestamp,
        year_decay: float = 0.9,
        he_tau: float = 0.08,
        he_prior_events: float = 1.,
        n_cap: int = 10,
        level_floor: float = 2.,
        rel_cap: float = 0.5
    ) -> tuple[float, float]:
        """
        Prior of the house effect of one pollster on one series for the current cycle, from its history in
        past elections: the decayed mean of its relative deviation (`dev_result_c / level`, centred across
        pollsters) shrunk toward 0 and scaled to the current level, with a fixed relative uncertainty.

        Parameters
        ----------
        history : pd.DataFrame
            Rows `event_date`, `dev_result_c`, `level`, `n_result` of the pollster and party in past elections.
        level : float
            Current level of the series (percentage points).
        ref_date : pd.Timestamp
            Date the ages are counted from (the current event).
        year_decay : float, optional
            Yearly decay of the weight of a past election.
        he_tau : float, optional
            Prior standard deviation, relative to the level.
        he_prior_events : float, optional
            Shrinkage toward 0, in "elections of `n_cap` polls" worth of weight.
        n_cap : int, optional
            Cap of the number of polls of an election counted in its weight.
        level_floor : float, optional
            Floor of the level used in the relative deviations (percentage points).
        rel_cap : float, optional
            Cap of the absolute relative deviation of an election.

        Returns
        -------
        tuple of float
            Prior mean and standard deviation, in percentage points.
        """
        lvl = max(float(level) if np.isfinite(level) else 0., level_floor)
        prior_err = float(he_tau * lvl)

        if history is None or history.shape[0] == 0:
            return 0., prior_err

        h = history.dropna(subset=['dev_result_c', 'level', 'n_result'])
        h = h.loc[h['n_result'] > 0]
        if h.shape[0] == 0:
            return 0., prior_err

        years = (pd.Timestamp(ref_date) - pd.to_datetime(h['event_date'])).dt.days / 365.25
        w = np.power(year_decay, years.clip(lower=0)) * np.minimum(h['n_result'].astype(float), n_cap)
        rel = np.clip(h['dev_result_c'].astype(float) / np.maximum(h['level'].astype(float), level_floor), -rel_cap, rel_cap)
        rel_mean = float((w * rel).sum() / (w.sum() + n_cap * he_prior_events))

        return float(rel_mean * lvl), prior_err

    @staticmethod
    def center_effects(
        effects: pd.DataFrame,
        totals: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Re-centre the effects of each series so that their mean, weighted by the weight of each pollster on
        that series (`w` column, or `totals` per pollster if given), is 0: subtracting them then leaves the
        level of the average unchanged. Adds the `center` column (the shift applied) and returns a copy.

        Parameters
        ----------
        effects : pd.DataFrame
            Indexed by `(pollster_id, name)` with `effect` and `w` columns.
        totals : pd.Series, optional
            Total weight of each pollster (indexed by `pollster_id`), instead of the per-series `w`.
        """
        out = effects.copy()
        pids = out.index.get_level_values('pollster_id')
        if totals is not None:
            w = pd.Series(pids).map(totals).fillna(0.).to_numpy(dtype=float)
        else:
            w = out['w'].fillna(0.).to_numpy(dtype=float)
        frame = pd.DataFrame({'name': out.index.get_level_values('name'), 'e': out['effect'].to_numpy(dtype=float), 'w': w})
        frame['we'] = frame['e'] * frame['w']
        sums = frame.groupby('name')[['we', 'w']].sum()
        center = (sums['we'] / sums['w'].where(sums['w'] > 0)).fillna(0.)
        out['center'] = pd.Series(frame['name']).map(center).to_numpy(dtype=float)
        out['effect'] = out['effect'] - out['center']

        return out

    def print_weights(
        self,
        name: str,
        date: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Print a table with the weights for each poll that affects the estimation at a specific date.

        Parameters
        ----------
        name : str
            Name of the party or block to be fitted.
        date : str or datetime, optional
            Date to choose as the center of the kernel window.
            If `None`, the last available date will be used.
        show : bool, optional
            Print the table.
        """
        df = self.fc_series[self.fc_series[name].notnull()]
        ix = self.fc_index[(self.fc_index >= df.index.min()) & (self.fc_index <= df.index.max())]
        weights = df.weight.values

        if date is None:
            date = np.min([
                pd.to_datetime('now').floor('d', ambiguous=False),
                ix.max()
            ])

        # Get the estimator instance
        est = LocalKernelEstimator(
            df.tfs,
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

        # Remove polls with a weight lower than 1e-2, in order to speed up the computation without losing accuracy
        df = df[df['weight'] >= 1e-2].reset_index()

        # Format dates
        df['date'] = df.date.dt.strftime('%d-%b')
        df['tfs'] = df.tfs - df.tfs.min()
        df['tte'] = df.tte - df.tte.min()

        # Set result
        df = df[[
            'date', 'pollster', 'weight', 'tfs', 'tte', 'sample_size',
            'weight_kernel', 'weight_over', 'weight_sample', 'weight_rating'
        ]]

        if not show:
            return df
        else:
            # Get the DataFrame styler and print result
            bars = [
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
                },
                {
                    'color': 'yellow-light',
                    'subset': ['weight_over'],
                    'vmin': 0,
                    'vmax': 1
                },
                {
                    'color': 'green-light',
                    'subset': ['weight_sample'],
                    'vmin': 0,
                    'vmax': 2
                },
                {
                    'color': 'red-light',
                    'subset': ['weight_rating'],
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

    def plot_weights(
        self,
        name: str,
        date: Optional[str] = None,
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

        df = self.fc_series[self.fc_series[name].notnull()]
        ix = self.fc_index[(self.fc_index >= df.index.min()) & (self.fc_index <= df.index.max())]
        weights = df.weight.values

        if date is None:
            date = np.min([
                pd.to_datetime('now').floor('d', ambiguous=False),
                ix.max()
            ])

        plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
        fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

        # Get the estimator instance
        est = LocalKernelEstimator(
            df[name],
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
        sw_scaled = df.weight / df.weight.max(axis=0)
        color = [cmap(i) for i in sw_scaled]
        s = (200 * (1 + sw_scaled)).values.round().astype(int)

        lbl_reg = 'Forecast: {} | CI 95%: {}'.format(
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
            df[name].values,
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

    def plot_bws(
        self,
        name: str,
        bw: Optional[float] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Plot kernel adaptive bandwidth selection for each day in the index.
        """
        _plt_params = dict(
            n_iter=None, min_delta=None,
            dt_min=None, dt_max=None, ymax=None, yticks=None, freq='M', grid=True,
            margin=0.05, title=None, note=None
        )
        _fig_params = dict(
            figsize=(20, 10)
        )

        plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
        fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

        df = self.fc_series[self.fc_series[name].notnull()]
        ix = self.fc_index[(self.fc_index >= df.index.min()) & (self.fc_index <= df.index.max())]
        weights = df.weight.values

        x = df.tfs.values
        z = np.arange(np.min(x), np.max(x) + 1)

        kernel = Kernel(
            x, weights=weights,
            bw=bw or self.reg_params['bw'],
            n_iter=plt_params['n_iter'] or self.reg_params['bw_kwargs']['n_iter'],
            min_delta=plt_params['min_delta'] or self.reg_params['bw_kwargs']['min_delta']
        )
        bw = kernel.get_bw_fixed()
        kbw = kernel.get_bw_adaptive(p=z)
        d = pd.Series(kbw, index=ix, name='BW')

        dt_min = plt_params['dt_min']
        if dt_min is None:
            dt_min = self.date_start
        elif not isinstance(dt_min, datetime):
            dt_min = pd.to_datetime(dt_min)

        dt_max = plt_params['dt_max']
        if dt_max is None:
            dt_max = np.min([
                pd.to_datetime('now').floor('d', ambiguous=False),
                self.date_end
            ])
        elif not isinstance(dt_max, datetime):
            dt_max = pd.to_datetime(dt_max)

        fig, ax = create_figure(ax=ax, **fig_params)

        plot_series(
            d,
            dt_min=dt_min,
            dt_max=dt_max,
            freq=plt_params['freq'],
            ymin=0,
            ymax=plt_params['ymax'],
            yticks=plt_params['yticks'],
            cm=get_color('blue'),
            ax=ax
        )

        ax.axhline(
            bw,
            color=get_color('grey-light'), ls='--', lw=2, alpha=0.5,
            label='Init: {}'.format(format_number(bw))
        )

        ax.legend(
            loc='upper right'
        )

        set_title(plt_params['title'], ax=ax)
        set_note(plt_params['note'], ax=ax)

        plot_figure(show=show, path=self.get_path(path), fig=fig)

    def plot_forecast_output(
        self,
        data: Optional[pd.DataFrame] = None,
        names: Optional[list[str] | dict[str, Any] | str] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        path: Optional[str] = None,
        **kwargs
    ) -> None:
        _plt_params = dict(
            dt_max=None, polar=False, show_previous=False, size=0.8, vmax=None, ticks=None,
            fmt=None, cm=None, grid=True, legend=True, title=None, note=None
        )
        _fig_params = dict(
            figsize=None
        )

        if isinstance(names, str):
            names = self.bmaps[names]

        if isinstance(names, dict):
            blocks = build_blocks(names, self.colors)
            names = self.names
            block_map = True
        elif isinstance(names, list):
            blocks = self.blocks.loc[names]
            block_map = False
        else:
            blocks = self.blocks
            names = self.names
            block_map = False

        plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
        fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

        d = {}
        if plt_params['show_previous']:
            d['Anterior'] = self.series.iloc[0]
        if data is not None:
            d['Resultados'] = data.copy()
        else:
            d['Resultados'] = self.get_forecast(date=plt_params['dt_max'])

        for k, v in d.items():
            for n in names:
                if n not in v.index:
                    v.loc[n] = 0

            d[k] = v[names]

        names = d['Resultados'].sort_values(ascending=False).index.tolist()

        df = pd.DataFrame.from_dict(d, orient='index')[names]
        if block_map:
            df = group_results(df, blocks=blocks)
        if plt_params['vmax'] and plt_params['vmax'] > df.sum(axis=1).min():
            df['Otros'] = plt_params['vmax'] - df.sum(axis=1)

        idx = df.index.values
        labels = list(df.columns)

        cm = blocks.color.to_dict() | {'Otros': get_color('grey-alpha')}
        cm = df.columns.map(cm).tolist()

        is_polar = plt_params['polar']

        if is_polar:
            fig_params['projection'] = 'polar'

        if fig_params['figsize'] is None:
            if is_polar:
                fig_params['figsize'] = (20, 14)
            else:
                fig_params['figsize'] = (20, len(idx) * 2)

        fig, ax = create_figure(ax=ax, **fig_params)

        g = None
        for i, k in enumerate(idx):
            x = df.loc[k].values.astype(float)
            b = df.loc[k].cumsum().shift().fillna(0).values.astype(float)
            h = plt_params['size']

            if is_polar:
                x = 2 * np.pi * x / 200
                b = 2 * np.pi * b / 200

            g = ax.barh(
                k, width=x, left=b, height=h, color=cm, edgecolor='w', linewidth=1
            )

            annot = np.array([format_number(v, 1) for v in df.loc[k].values.astype(float)])
            if plt_params['fmt'] is not None:
                annot = np.array([plt_params['fmt'].format(v) for v in annot])

            for j in np.arange(len(labels)):
                values = x[j] / 2. + b[j]
                y = list(idx).index(k)

                ax.text(
                    values, y, annot[j],
                    ha='center', va='center', color=get_text_color(cm[j]), weight=600
                )

        if is_polar:
            ax.set_thetalim((0, np.pi))
            ax.set_theta_zero_location('W')
            ax.set_theta_direction(-1)

            ax.set_xticks([])
            ax.set_xticklabels([])
            rorigin = np.abs(ax.get_rmin() * 2)
            if df.shape[0] < 2:
                rorigin *= 2
                ax.set_yticks([])
                ax.set_yticklabels([])

            ax.set_rorigin(-rorigin)
            ax.spines[:].set_color('none')
        else:
            ax.set_xlim([0, df.loc['Resultados'].sum()])
            if isinstance(plt_params['ticks'], (list, tuple)):
                ax.set_xticks(plt_params['ticks'])
            elif plt_params['ticks'] is not None:
                set_num_locator(n=plt_params['ticks'], axis='x', ax=ax)

            xticklabels = [format_number(t) for t in ax.get_xticks()]
            if plt_params['fmt'] is not None:
                xticklabels = [plt_params['fmt'].format(t) for t in xticklabels]
            ax.set_xticklabels(xticklabels)

            if df.shape[0] < 2:
                ax.set_yticks([])
                ax.set_yticklabels([])

            ax.spines[['top', 'right', 'left']].set_color('none')

        ax.grid(False)
        if plt_params['legend'] and g is not None:
            ax.legend(
                handles=g.get_children(),
                labels=labels,
                loc='lower right',
                ncol=len(labels),
                bbox_to_anchor=(1, 1.05)
            )

        set_title(plt_params['title'], y=1.05, ax=ax, loc='left')
        set_note(plt_params['note'], ax=ax)

        plot_figure(show=show, path=self.get_path(path), fig=fig)

    def plot_forecast_series(
        self,
        names: Optional[list[str] | dict[str, Any] | str] = None,
        pollster: Optional[str | int] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        path: Optional[str] = None,
        **kwargs
    ) -> None:
        _col_params = dict(
            ylim=None, ymin=None, ymax=None, yticks=5,
            fmt=None, s=50, lw=3, ls='-'
        )
        _plt_params = dict(
            show_forecast=True, show_ci=False, show_events=True, show_polls=True, show_start=False, show_end=False,
            dt_min=None, dt_max=None, freq='M', grid=True, hlines=None,
            margin=0.05, legend=True, leg_cols=7, title=None, note=None
        )
        _fig_params = dict(
            figsize=(20, 10)
        )

        if isinstance(names, str):
            names = self.bmaps[names]

        if isinstance(names, dict):
            blocks = build_blocks(names, self.colors)
            names = self.names
            series = blocks.index.tolist()
            block_map = True
        elif isinstance(names, list):
            blocks = self.blocks.loc[names]
            series = names
            block_map = False
        else:
            blocks = self.blocks
            names = self.names
            series = self.names
            block_map = False

        col_params = get_params(_col_params, series, 'col_params', **kwargs)
        plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
        fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

        dt_min = plt_params['dt_min']
        if dt_min is None:
            dt_min = self.date_start
        elif not isinstance(dt_min, datetime):
            dt_min = pd.Timestamp(dt_min)

        if plt_params['show_start']:
            plt_params['dt_min'] = self.date_start
        else:
            plt_params['dt_min'] = dt_min

        dt_max = plt_params['dt_max']
        if dt_max is None:
            dt_max = np.min([
                pd.Timestamp('now').floor('d', ambiguous=False),
                self.date_end
            ])
        elif not isinstance(dt_max, datetime):
            dt_max = pd.Timestamp(dt_max)

        if plt_params['show_end']:
            plt_params['dt_max'] = self.date_end
        else:
            plt_params['dt_max'] = dt_max

        ymin = 100
        ymax = 0
        cmin = None
        cmax = None
        vlines = []

        if plt_params['show_forecast']:
            if self.forecast[names].isnull().all().all():
                if self.verbose > 0:
                    print('Fit forecast...')

                self.fit_forecast(names)

            forecast = self.forecast.loc[dt_min:dt_max][names]
            if plt_params['show_ci']:
                fc_stat = self.fc_stat.loc[dt_min:dt_max][names]
                cmin = fc_stat.applymap(lambda x: x['cmin'] if isinstance(x, dict) else x)
                cmax = fc_stat.applymap(lambda x: x['cmax'] if isinstance(x, dict) else x)

            if block_map is not None:
                forecast = group_results(forecast, blocks=blocks)
                if plt_params['show_ci']:
                    cmin = group_results(cmin, blocks=blocks)
                    cmax = group_results(cmax, blocks=blocks)

            ymin_ = forecast.min().min()
            if ymin_ < ymin:
                ymin = ymin_
            ymax_ = forecast.max().max()
            if ymax_ > ymax:
                ymax = ymax_

            if plt_params['show_ci']:
                ymin_ = pd.concat([cmin, cmax], axis=1).min().min()
                if ymin_ < ymin:
                    ymin = ymin_
                ymax_ = pd.concat([cmin, cmax], axis=1).max().max()
                if ymax_ > ymax:
                    ymax = ymax_
        else:
            forecast = None

        if plt_params['show_events']:
            events = self.nfc_series[names]
            if not plt_params['show_start']:
                events = events.loc[dt_min:]
            if not plt_params['show_end']:
                events = events.loc[:dt_max]
            if block_map is not None:
                events = group_results(events, blocks=blocks)

            if events.shape[0] > 0:
                vlines += events.index.tolist()

            ymin_ = events.min().min()
            if ymin_ < ymin:
                ymin = ymin_
            ymax_ = events.max().max()
            if ymax_ > ymax:
                ymax = ymax_
        else:
            events = None

        if plt_params['show_polls']:
            polls = self.fc_series[
                (
                    self.fc_series.pollster.notnull()
                ) & (
                    self.fc_series.pollster != pollster
                ) & (
                    self.fc_series.pollster_id != pollster
                )
            ].loc[dt_min:dt_max][names]
            if block_map is not None:
                polls = group_results(polls, blocks=blocks)

            ymin_ = polls.min().min()
            if ymin_ < ymin:
                ymin = ymin_
            ymax_ = polls.max().max()
            if ymax_ > ymax:
                ymax = ymax_
        else:
            polls = None

        if pollster is not None:
            if isinstance(pollster, str):
                pollster = self.fc_series[
                    self.fc_series.pollster == pollster
                ]
            elif isinstance(pollster, int):
                pollster = self.fc_series[
                    self.fc_series.pollster_id == pollster
                ]
            pollster = pollster.loc[dt_min:dt_max][names]
            if block_map is not None:
                pollster = group_results(pollster, blocks=blocks)

            ymin_ = pollster.min().min()
            if ymin_ < ymin:
                ymin = ymin_
            ymax_ = pollster.max().max()
            if ymax_ > ymax:
                ymax = ymax_

        cm = blocks.color.to_dict()

        fig, ax = create_figure(ax=ax, **fig_params)

        for n in series:
            if plt_params['show_events']:
                ax.scatter(
                    events[n].index, events[n].values,
                    marker='*', s=(col_params[n]['s'] * 4), c=cm[n], alpha=1, label=None
                )
            if pollster is not None:
                ax.scatter(
                    pollster[n].index, pollster[n].values,
                    marker='o', s=col_params[n]['s'], c=cm[n], alpha=1, label=None
                )
            if plt_params['show_polls']:
                ax.scatter(
                    polls[n].index, polls[n].values,
                    marker='.', s=col_params[n]['s'], c=cm[n], alpha=0.5, label=None
                )

        if pollster is not None:
            for n in series:
                ax.plot(
                    pollster[n].index, pollster[n].values,
                    color=cm[n], lw=col_params[n]['lw'], ls='--', alpha=0.5, label=None
                )

        if plt_params['show_forecast']:
            item_col_params = dict(col_params)
            item_plt_params = dict(plt_params) | dict(vlines=vlines, title=None, note=None)
            item_fig_params = dict(fig_params) | dict(figsize=None)

            for n in series:
                if item_col_params[n]['ymin'] is None and item_col_params[n]['ylim'] is None:
                    item_col_params[n]['ymin'] = ymin * (1. - plt_params['margin'])
                if item_col_params[n]['ymax'] is None and item_col_params[n]['ylim'] is None:
                    item_col_params[n]['ymax'] = ymax * (1. + plt_params['margin'])

            plot_series(
                forecast[series],
                cm=cm,
                ax=ax,
                col_params=item_col_params,
                plt_params=item_plt_params,
                fig_params=item_fig_params
            )

            if plt_params['show_ci']:
                for n in series:
                    ax.fill_between(
                        forecast.index,
                        cmin[n].values,
                        cmax[n].values,
                        color=cm[n],
                        alpha=0.2
                    )

        set_title(plt_params['title'], ax=ax)
        set_note(plt_params['note'], ax=ax)

        plot_figure(show=show, path=self.get_path(path), fig=fig)

    def get_path(self, name: Optional[str] = None) -> str | None:
        if name:
            return '{}/{}'.format(self.path, name)
