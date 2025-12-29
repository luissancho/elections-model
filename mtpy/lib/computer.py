from adjustText import adjust_text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.stats import norm
from tqdm import tqdm

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
    get_next_event_date, get_ratings, save_model_data, save_ratings_data, get_event_params
)
from .utils import (
    build_blocks, group_results, norm_range
)

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
        error_weights: Optional[dict[str, float]] = {'avg': .7, 'blocks': .3},
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
            Path to store model files.
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
        self.path = path or os.getcwd()  # Path to store model files

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
            map = self.event_params[event_date]['bmaps'][name]
            if isinstance(map, (tuple, list)):
                map = dict(zip(map, map))

            for party, block in map.items():
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
            'error_avg', 'error_blocks', 'bias_avg', 'bias_blocks', 'bias',
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
        for n in self.names:
            if n not in self.errors.columns:
                self.errors[n] = np.NaN
        self.errors = self.errors[self.names]

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
        df['sample_size'] = df['sample_size'].where(df['sample_size'] > 0, np.NaN)
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
            - bias_avg
                The average bias over the most important parties for each event (usually predicted by all pollsters).
                These are defined at the event parameters `main` bmap.
            - bias_blocks
                The bias of the poll prediction over the margin gap between the two main blocks of parties.
            - bias
                The bias of the poll prediction over all the parties.

        Parameters
        ----------
        save : bool, optional
            Whether to save the data to the database.

        Returns
        -------
        pd.DataFrame
            A table containing the computed values of each poll.
        """
        columns = ['error_avg', 'error_blocks', 'bias_avg', 'bias_blocks', 'bias']
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

        dp = pd.concat([error_avg, error_blocks, bias_avg, bias_blocks], axis=1)

        if self.verbose > 0:
            print('Set computed biases...')

        bias_weights = pd.Series(self.error_weights).add_prefix('bias_')
        dp['bias'] = dp[bias_weights.index].apply(lambda x: Stat(x, weights=bias_weights).mean(), axis=1)

        if self.verbose > 0:
            print('Process data...')

        dp[['error_avg', 'error_blocks']] = dp[['error_avg', 'error_blocks']].mul(100)
        dp[['bias_avg', 'bias_blocks', 'bias']] = dp[['bias_avg', 'bias_blocks', 'bias']].apply(self.lor_to_bias)

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
        save: bool = False
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

        Parameters
        ----------
        save : bool, optional
            Whether to save the data to the database.

        Returns
        -------
        pd.DataFrame
            A table containing the computed values of each pollster.
        """
        columns = ['rating', 'weight_rating']
        rparams = [
            'quality', 'num_events', 'num_polls', 'num_polls_w',
            'error_avg', 'error_blocks', 'bias_avg', 'bias_blocks', 'bias',
            'bias_dev_adj', 'bias_dev_err', 'rating_adj', 'rating', 'weight_rating'
        ]
        rkeys = ['event_date', 'pollster_id']

        polls = self.filter_polls().drop(columns=columns + self.names + ['-'])

        # Scale absolute percentage errors to the range [0, 1]
        polls[['error_avg', 'error_blocks']] = polls[['error_avg', 'error_blocks']].div(100)
        # Scale bias deviations to the log odds ratio scale
        polls[[
            'bias_avg', 'bias_blocks', 'bias', 'bias_dev_adj', 'bias_dev_err'
        ]] = polls[[
            'bias_avg', 'bias_blocks', 'bias', 'bias_dev_adj', 'bias_dev_err'
        ]].apply(self.bias_to_lor)

        dr = pd.DataFrame(columns=rkeys + rparams)

        if self.verbose > 0:
            print('Compute ratings...')

        edts = polls.index.get_level_values('event_date').unique()
        for i in np.arange(1, len(edts) + 1):
            event_date = edts[i] if i < len(edts) else get_next_event_date(self.scope, date_from=edts[i - 1])

            if self.verbose > 0:
                print('Compute ratings for {}...'.format(event_date))

            iter_polls = polls.loc[:edts[i - 1]]
            if iter_polls.shape[0] == 0 or iter_polls['bias'].isnull().any():
                if self.verbose > 0:
                    print('No polls found, skipping...')

                continue

            ratings = self.pollster_ratings(iter_polls).reset_index()

            
            ratings['event_date'] = pd.to_datetime(event_date) if event_date is not None else pd.NaT
            ratings['pollster_id'] = ratings['pollster'].map(self.pollsters.set_index('name')['id'])
            ratings = ratings[dr.columns]

            dr = pd.concat([dr, ratings], ignore_index=True)

        if self.verbose > 0:
            print('Process data...')

        # Rescale absolute percentage errors to the range [0, 100]
        dr[[
            'quality', 'error_avg', 'error_blocks', 'rating_adj', 'rating'
        ]] = dr[[
            'quality', 'error_avg', 'error_blocks', 'rating_adj', 'rating'
        ]].mul(100)
        # Rescale log odds ratio deviations to the bias scale
        dr[[
            'bias_avg', 'bias_blocks', 'bias', 'bias_dev_adj', 'bias_dev_err'
        ]] = dr[[
            'bias_avg', 'bias_blocks', 'bias', 'bias_dev_adj', 'bias_dev_err'
        ]].apply(self.lor_to_bias)

        dr = dr.set_index(rkeys).sort_index().round(2).astype(float)

        df = self.polls.reset_index().drop(columns=columns).join(dr[columns], on=rkeys).set_index(self.keys)[columns].sort_index()

        if save:
            if self.verbose > 0:
                print('Save ratings data...')

            # Add the common missing `event_scope` index to the polls DataFrame and save data
            data = pd.concat([dr], keys=[self.scope], names=['event_scope'] + rkeys)
            nrows = save_ratings_data(data)

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
        # Update ratings
        self.ratings[rparams] = dr.reindex(self.ratings.index)

        return df

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
                events.loc[dt, missing] = np.NaN
        
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
        polls: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Pollster rating algorithm.

        See `compute_ratings` for more details about the rating algorithm.

        Parameters
        ----------
        polls : pd.DataFrame
            Polls to be used in the rating algorithm.

        Returns
        -------
        pd.DataFrame
            A table containing the computed values of each pollster.
        """
        df = polls[polls['bias'].notnull()]

        # Compute additional weights of each poll
        df[['weight_pos', 'weight_week', 'weight_year']] = self.poll_rating_weights(df)

        # Compute the final weight of each poll by multiplying all the other weights computed
        df['weight'] = df[[
            'weight_over', 'weight_sample', 'weight_pos', 'weight_week', 'weight_year'
        ]].prod(axis=1)

        # Remove polls with weight below the threshold
        df = df[df['weight'] >= 1e-2].reset_index()

        # Create a DataFrame to store the results, using the pollster names as index
        dr = pd.DataFrame(
            index=self.pollsters['name'].rename('pollster')
        )

        # Get the quality of each pollster, a scoring value of the pollster's metodology and historical standards
        # It is set to be a number between 0 and 100, the greater this value, the more reliable the pollster
        dr['quality'] = self.pollsters.set_index('name').quality.astype(float).div(100)

        # Get the total number of events concurred and polls published by each pollster
        dr['num_events'] = df.groupby('pollster')['event_date'].nunique().reindex(dr.index)
        dr['num_polls'] = df.groupby('pollster')['event_date'].count().reindex(dr.index)
        # Weighted number of polls, giving more importance to the most recent polls
        dr['num_polls_w'] = df.groupby('pollster')['weight'].sum().reindex(dr.index)

        # Compute the weighted mean of each pollster's poll errors
        for col in ['error_avg', 'error_blocks', 'bias_avg', 'bias_blocks', 'bias']:
            dr[col] = df.groupby('pollster').apply(
                lambda x: np.sum(x[col] * x['weight']) / np.sum(x['weight'])
            ).reindex(dr.index)

        # Compute the weighted mean of each pollster's poll deviations
        dr['bias_dev_adj'] = df.groupby('pollster').apply(
            lambda x: np.sum(x['weight'] * x['bias_dev_adj']) / np.sum(x['weight'])
        ).reindex(dr.index)
        # Compute the weighted standard error of each pollster's poll deviations
        dr['bias_dev_err'] = df.groupby('pollster').apply(
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
        polls: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute additional weights needed to compute pollster ratings and an estimation of their relative deviation
        from the expected bias of polls published by other pollsters in the same event and week.

        Parameters
        ----------
        polls : pd.DataFrame, optional
            The polls to get the rating for.

        Returns
        -------
        pd.DataFrame
            The polls data with the new weights and deviations.
        """
        df = polls.reset_index()

        # We use all polls published by each pollster, but give more weight to the ones published closer to the event
        df['seq_pos'] = df.groupby(['event_date', 'pollster_id'])['date'].cumcount(ascending=False)
        df['weight_pos'] = df['seq_pos'].apply(lambda x: np.power(self.pos_decay, x))

        # We give more weight to the polls published closer to the event
        event_dlimits = polls.groupby('event_date')['days'].min().to_dict()
        df['event_dlimits'] = df['event_date'].map(event_dlimits)
        df['weeks'] = ((df['days'] - df['event_dlimits'] + 1) // 7).clip(0)
        df['weight_week'] = np.power(self.week_decay, df['weeks'])

        # Compute the weight of each poll based on the number of years between the poll and the most recent event
        df['event_year'] = df['event_date'].dt.strftime('%Y').astype(int)
        df['years'] = df['event_year'].max() - df['event_year']
        df['weight_year'] = np.power(self.year_decay, df['years'])

        df = df.set_index(self.keys)[[
            'weight_pos', 'weight_week', 'weight_year'
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
        x = np.asarray(x)

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
        x = np.asarray(x)

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
        x = np.asarray(x)
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
        x = np.asarray(x)

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
            'sample_size', 'proc_sample', 'weight', 'error_avg', 'bias_avg', 'error_blocks', 'bias_blocks', 'bias'
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
        df['regional'] = df.party.map(self.parties.set_index('name').regional).astype(int)
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

    def get_seats_estimator_data(self) -> pd.DataFrame:
        """
        Load data used to fit the seats estimator.

        Returns
        -------
        pd.DataFrame
            A table containing the data.
        """
        df = pd.DataFrame()

        for metric in ['pct', 'seats']:
            d = self.load_events(metric)
            if metric == 'seats':
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

        df['regional'] = df.party.map(self.parties.set_index('name').regional.to_dict()).astype(int)
        df['color'] = df.party.map(self.parties.set_index('name').color.to_dict())
        df['year'] = df.date.dt.year.astype(int)
        df['years'] = df.year.max() - df.year
        df['weight'] = np.power(0.97, df.years).round(2)
        df['ratio'] = (df.seats / df.pct).fillna(0).round(2)
        df['pos'] = df.groupby('date').cumcount() + 1

        df = df.sort_values('pct', ignore_index=True)[[
            'party', 'regional', 'pct', 'seats', 'pos', 'color', 'years', 'weight', 'ratio'
        ]]

        return df

    def get_seats_estimator(
        self,
        **kwargs
    ) -> LeastSquaresEstimator:
        """
        Build an estimator of the seats based on the percentage of votes.

        Runs a regression analysis that predicts the number of seats based on the
        percentage of votes and whether the party is regional or not.

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
            y=df['seats'],
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
            'parties', 'proc_sample', 'error_avg', 'error_blocks', 'bias_avg', 'bias_blocks',
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
