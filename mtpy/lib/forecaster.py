from ast import literal_eval
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
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

from .data import (
    get_event_dates, get_event_series, get_poll_series, get_parties, get_pollsters
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
        self.bmap = bmap
        self.reg_params = self.set_reg_params(reg_params)

        self.verbose = verbose
        self.path = path or os.getcwd()

        self.event_params = json.loads(
            self.app.fs.read(f'{self.path}/params.json')
        )[self.scope][self.event_date]

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
        self.forecast = None  # Fitted estimation of the percentage of votes for each party in the election event
        self.fc_stat = None  # Standard error and confidence interval of the forecast for each estimation

        self.date_start = None  # Date of the last election event
        self.date_first = None  # Date of the first poll published for the current election event
        self.date_last = None  # Date of the last poll published for the current election event
        self.date_end = None  # Date of the current election event

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
            'error_avg', 'error_blocks', 'bias_avg', 'bias_blocks', 'bias', 'bias_dev_adj', 'bias_dev_err',
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
        series['weight_rating'] = series.weight_rating.fillna(series.pollster_id.map(self.pollsters.set_index('id').quality))
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
        self.series = pd.concat([
            series[data_cols],
            block_results
        ], axis=1)[
            data_cols + self.names
        ].sort_index()
        # Remove polls with incomplete results
        self.series = self.series.loc[self.series.pollster.isnull() | self.series.computed]
        # Assign the remaining percentage to a new 'others' block
        self.series['-'] = 100. - self.series[self.names].sum(axis=1, min_count=1)

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
        df = self.fc_series[self.fc_series[name].notnull()]

        if df.empty:
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
        """
        names = names or self.names

        names_ = tqdm(names) if self.verbose > 0 else names  # Show progress bar if verbose
        for name in names_:
            dreg, dstat = self.fit(name, max_fc=max_fc, ret_stat=True)

            self.forecast.loc[dreg.index, name] = dreg
            # Assign the remaining percentage to a new 'others' block
            self.forecast['-'] = 100. - self.forecast[names].sum(axis=1, min_count=1)
            self.fc_stat.loc[dstat.index, name] = dstat

        # Populate the forecast and statistics frames, filling the missing values with the last available estimation
        if fillna:
            self.forecast = self.forecast.fillna(method='ffill')
            self.fc_stat = self.fc_stat.fillna(method='ffill')

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
            ).set_index('date').applymap(lambda x: literal_eval(x) if isinstance(x, str) else x)
            self.fc_stat.index = pd.DatetimeIndex(self.fc_stat.index)
        else:
            self.fit_forecast()

        return self.forecast

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
