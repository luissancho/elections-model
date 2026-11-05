import copy
import json
import math

from ..core.worker import SequenceJob
from ..lib.forecaster import check_rating_exponent
from ..lib.publish import resolve_scopes

# Keys of a tuned configuration of the average (`tuning.default_config`, the `best.json` of the search) and the type
# of each: dicts of parameters, switches, `weights` (checked apart), the prolongation and the label
CONFIG_TYPES = {'reg_params': dict, 'house_effects': bool, 'he_params': dict, 'dispersion': bool, 'herding': bool,
                'disp_params': dict, 'weights': None, 'max_fc': int, 'label': str}

# Keys each dict of parameters of a configuration may hold: the keywords the Forecaster's `set_reg_params`,
# `set_he_params` and `set_disp_params` read (anything else would be ignored, running the live value under a tuned
# label); `herding` is left out of `disp_params` because the switch of the configuration overrides it
CONFIG_PARAMS = {
    'reg_params': ('kernel', 'bw_type', 'bw', 'bw_kwargs', 'bw_cap', 'poly_deg', 'cov_type', 'cov_kwargs'),
    'he_params': ('n_iter', 'min_polls', 'tau', 'prior_events', 'n_cap', 'year_decay', 'n_days', 'level_floor',
                  'rel_cap', 'err_floor', 'prior'),
    'disp_params': ('min_polls', 'n_prior', 'max_ratio', 'herd_series', 'herd_prior')
}
# Keys of `reg_params['bw_kwargs']`, the parameters of the adaptive bandwidth
BW_KWARGS = ('n_iter', 'min_delta')

# Loose arguments of the job a configuration also sets, with their defaults: a flag left at its default yields to
# the configuration, a flag set to anything else must agree with it
CONFIG_SWITCHES = {'house_effects': True, 'dispersion': True, 'herding': True, 'max_fc': 10}


class Backtest(SequenceJob):
    """
    Run the model backtest on past elections and write its tables to ``backtest/results`` of the app file system
    (S3 in ECS; the job form of ``backtest/run_backtest.py``), the input of ``publish backtest`` and of the model
    slice and series of ``publish event``.

    Scope by scope (`backtest.backtest_scope`), it evaluates every (election, horizon) case out of sample, writes
    the eight tables (``shares``, ``seats``, ``blocks``, ``provinces``, ``series``, ``meta``, ``metrics`` and
    ``by_horizon``) and ``run.json`` (parameters, elections and horizons evaluated, commit and state of the polls
    in the database) for ``es`` at the root of the results folder and for the other scopes in ``{scope}/``,
    overwriting the previous run. The results folder of ``run`` is ``out`` when given (a local folder, as
    ``run_backtest.py`` does with the repository copy) and ``backtest/results`` of the app file system
    otherwise; ``push`` reads ``out`` (the repository copy by default) and writes the app file system, ``pull``
    the reverse. Scopes are
    isolated: one that fails never stops the others (unless ``stop_on_error``), and the job exits with an error at the end if any failed. A
    scope without elections to evaluate, or none of whose cases could be evaluated, ends ``skipped`` and nothing
    is written for it. The cases the backtest skips (without usable polls) are reported as warnings of the scope.

    The console output has the levels of ``publish``: 0, the header and the final summary; 1, also
    ``[i/n] scope: empezando``, the result line, the warnings as they happen and the ``by_horizon`` table of each
    scope; 2, also the line of each case of the backtest (``Backtest <election> at <d> days...``).

    Examples
    --------
    From the command line::

        python job.py backtest
        python job.py backtest '{"scopes":"all","horizons":[6,30]}'
        python job.py backtest '{"scopes":["es-md"],"events":["2023-05-28"],"n_sim":200,"verbose":2}'
        python job.py backtest '{"scopes":["es"],"composition":"auto","out":"backtest/results_comp"}'
        python job.py backtest '{"scopes":["es"],"error_model":"average","out":"backtest/results_avg"}'
        python job.py backtest '{"scopes":["es"],"drift_scale":"auto","out":"backtest/results_scale"}'
        python job.py backtest '{"scopes":["es"],"tail_dof":5,"out":"backtest/results_dof5"}'
        python job.py backtest '{"scopes":["es"],"config":"backtest/tuning/best.json","out":"backtest/results_tuned"}'

    Copy the tables between the repository and the app file system (S3 in ECS)::

        python job.py backtest '{"action":"push","scopes":"all"}'
        python job.py backtest '{"action":"pull","scopes":"all"}'

    The flow: the backtest runs in ECS and writes its tables to S3; ``pull`` brings them to the repository copy
    (``backtest/results``) to review ``by_horizon.csv`` and commit; ``push`` seeds S3 with the repository copy.
    Both mirror the source and touch nothing for a scope without tables to copy (``skipped``).

    In Docker (JSON without spaces)::

        RUN_JOB='backtest {"scopes":"all"}'
    """

    sequence = {'backtest': ['backtest']}
    default_action = 'run'
    stop_on_error = False
    chunk_noun = 'scopes'

    def run(self, scopes=('es',), events=None, horizons=None, n_sim=1000, seed=42, max_fc=10, nowcast_only=False,
            house_effects=True, industry_bias=False, dispersion=True, herding=True, composition=None,
            regional_noise=True, min_events=None, error_model='polls', drift_scale=None, tail_dof=None, config=None,
            recompute_ratings=False, flat_ratings=False, rating_exponent=1.0, out=None, verbose=1, runner=None, stats=None,
            stop_on_error=None, action='run', **kwargs):
        """
        Run the backtest of the requested scopes.

        Parameters
        ----------
        scopes : str or list of str, default ('es',)
            Scopes to evaluate, or ``'all'`` (``es`` and every regional scope, see ``publish.resolve_scopes``).
        events : list of str, optional
            Elections to evaluate (``YYYY-MM-DD``), the same for every scope; by default the five general
            elections since 2015 for ``es`` and the featured elections held since 2019 for a regional scope.
        horizons : list of int, optional
            Horizons in days (``[6, 14, 30, 60, 90, 180]`` by default).
        n_sim : int, default 1000
            Simulations per case and mode.
        seed : int, default 42
            Random seed.
        max_fc : int, default 10
            Maximum number of forecast steps.
        nowcast_only : bool, default False
            Skip the horizon run of each case (columns ``_h``).
        house_effects : bool, default True
            Subtract the house effects (M6).
        industry_bias : bool, default False
            Shift the average by the industry-wide bias of past elections (M6).
        dispersion : bool, default True
            Weigh each pollster by its effective sample (M12).
        herding : bool, default True
            Reduce the weight of the pollsters less dispersed than their samples allow (M13).
        composition : str or float, optional
            Joint noise of the national parties (M7): ``'auto'`` or a ratio in (0, 1] (a numeric string is
            accepted, as in the script); independent draws by default.
        regional_noise : bool, default True
            Oscillations by community and province (M9); without it, a deterministic proportional swing.
        min_events : int, optional
            Featured elections a regional scope needs to fit the estimators on itself instead of on ``es``
            (M11); by default always ``es``, 1 forces its own.
        error_model : {'polls', 'average'}, default 'polls'
            Terminal error of the simulation: the error of a single poll or that of the poll average itself in
            the past elections (table ``average_errors``, filled by the ``average_errors`` action of the ``compute``
            job); ``average`` falls back to
            ``polls`` with a warning when the table has too few rows (``meta.csv`` records the one used).
        drift_scale : str, float or None, default None
            Multiplier of the drift variance: ``'auto'`` fits it in each case on the errors of the poll average of
            the elections before it (table ``average_errors``; 1 with a warning without rows), a positive number
            (also as a numeric string) fixes it, ``None`` leaves 1 (``meta.csv`` records the one used and the
            rows of its fit).
        tail_dof : int or None, default None
            Degrees of freedom of the Student t of the draw, the same for every party: an integer of at least 2
            (also as an integer string); ``None`` takes ``nobs - 2`` of each party, at least 3 (``meta.csv``
            records it).
        config : dict, str or None, default None
            Tuned configuration of the average (``tuning.default_config``, the ``best.json`` of the ``tune`` job), or
            the path of a JSON file with it (a relative path is taken from the working directory): the keys
            ``reg_params``, ``he_params`` and ``disp_params`` (dicts with only the keys the Forecaster's
            ``set_*_params`` read, ``CONFIG_PARAMS``; ``bw_kwargs`` with ``n_iter`` and ``min_delta``; ``herding``
            never in ``disp_params``), ``house_effects``, ``dispersion`` and
            ``herding`` (booleans), ``weights`` (``None`` or ``{"wspan": n}`` with an optional ``"ol_max"``,
            positive integers), ``max_fc`` (an integer of at least 0) and ``label`` (a string), no other. Every case
            builds its model with it (``backtest.run_case``): the configuration wins over ``house_effects``,
            ``dispersion``, ``herding`` and ``max_fc``, which take its values; one of them set to something other
            than its default and than the configuration is rejected. ``run.json`` records the configuration and
            ``meta.csv`` its label. ``None`` runs the model as it is.
        recompute_ratings : bool, default False
            Recompute the pollster ratings in memory with the current code (once per scope) and use them in
            every case instead of the stored ones; nothing is saved to the database.
        flat_ratings : bool, default False
            Weigh every poll with rating 1 (the average without the rating, phase C); exclusive with
            ``recompute_ratings``.
        rating_exponent : float, default 1.0
            Power the rating weight of every poll is raised to (phase C2a): 1 is the model as it is, 0 the flat
            rating; a finite number of at least 0 (``meta.csv`` and ``run.json`` record it).
        out : str, optional
            Local results folder (a relative path is taken from the working directory). For ``run`` the app
            file system (``backtest/results``, S3 in ECS) by default; for ``push`` (its origin) and ``pull`` (its
            destination) the repository copy (``backtest/results`` of the repository) by default.
        verbose : int, default 1
            Verbosity of the console output (see the class).
        runner, stats : callable, optional
            Replacements of `backtest.run_backtest` and `backtest.poll_stats` (for tests).
        stop_on_error : bool, optional
            Stop at the first failed scope instead of going on with the others; ``False`` by default.
        action : {'run', 'push', 'pull'}, default 'run'
            ``run`` evaluates and writes the tables; ``push`` copies them from ``out`` (the repository copy) to
            the app file system; ``pull`` copies them from the app file system to ``out``.
        **kwargs
            Ignored extra job arguments.

        Returns
        -------
        dict[str, dict]
            Result of each scope: ``status`` (``completed``, ``pushed``, ``pulled``, ``skipped`` or ``failed``);
            when completed, ``n_cases``, ``n_events``, ``n_horizons``, ``events``, ``horizons``, ``folder`` and
            ``seconds``; when pushed or pulled, ``n_files`` (copied), ``n_removed`` (deleted from the destination
            because the source lacks them) and ``folder`` (the destination); otherwise the ``reason``. Every
            result of ``run`` carries the ``warnings`` of the backtest (the cases skipped).

        Raises
        ------
        ValueError
            If ``flat_ratings`` and ``recompute_ratings`` are both true (``backtest: flat_ratings and recompute_ratings
            are exclusive``), ``composition`` is not ``'auto'``, a ratio in (0, 1] or ``None``, ``error_model`` is not ``'polls'``
            or ``'average'``, ``drift_scale`` is not ``'auto'``, a positive number or ``None``, ``tail_dof`` is not
            an integer of at least 2 or ``None``, ``config`` is not a valid configuration nor the path of a JSON file
            with one (``backtest: config ...``), ``rating_exponent`` is not a finite number of at least 0
            (``rating_exponent: expected a finite number >= 0``) or one of ``house_effects``, ``dispersion``, ``herding`` and
            ``max_fc`` contradicts it (``backtest: <flag> contradicts the config ...``), there are no scopes to run
            (``backtest: no scopes to run``), ``action`` is not ``run``, ``push`` or ``pull`` or ``stop_on_error`` is not true or false
            (``stop_on_error must be true or false``).
        RuntimeError
            If any scope failed (raised after the summary and the alert).
        """
        if action not in ('run', 'push', 'pull'):
            raise ValueError('backtest: unknown action "{}" (expected run, push or pull)'.format(action))
        if flat_ratings and recompute_ratings:
            raise ValueError('backtest: flat_ratings and recompute_ratings are exclusive')
        composition = self._composition(composition)
        error_model = self._error_model(error_model)
        drift_scale = self._drift_scale(drift_scale)
        tail_dof = self._tail_dof(tail_dof)
        rating_exponent = check_rating_exponent(rating_exponent)
        config = self._config(config)
        if config is not None:
            house_effects, dispersion, herding, max_fc = self._config_switches(
                config, house_effects=house_effects, dispersion=dispersion, herding=herding, max_fc=max_fc)
        targets = resolve_scopes(scopes)
        if not targets:
            raise ValueError('backtest: no scopes to run')

        super().run(select='backtest', chunks=targets, action=action, stop_on_error=stop_on_error, events=events, horizons=horizons,
                    n_sim=n_sim, seed=seed, max_fc=max_fc, nowcast_only=nowcast_only, house_effects=house_effects,
                    industry_bias=industry_bias, dispersion=dispersion, herding=herding, composition=composition,
                    regional_noise=regional_noise, min_events=min_events, error_model=error_model,
                    drift_scale=drift_scale, tail_dof=tail_dof, config=config, recompute_ratings=recompute_ratings,
                    flat_ratings=flat_ratings, rating_exponent=rating_exponent, out=out, verbose=verbose, runner=runner, stats=stats)

        return {chunk: result for (key, chunk), result in self.results.items()}

    @staticmethod
    def _composition(value):
        """
        Normalise the ``composition`` argument.

        Parameters
        ----------
        value : str, float, int or None
            ``None`` or ``'auto'`` (kept), or a ratio in (0, 1], also as a numeric string (``'0.25'``).

        Returns
        -------
        str, float or None
            ``None``, ``'auto'`` or the ratio as a float.

        Raises
        ------
        ValueError
            Anything else (a boolean included).
        """
        if value is None or value == 'auto':
            return value
        try:
            ratio = None if isinstance(value, bool) else float(value)
        except (TypeError, ValueError):
            ratio = None
        if ratio is None or not 0 < ratio <= 1:
            raise ValueError('backtest: composition must be "auto", a ratio in (0, 1] or null (got {!r})'.format(value))
        return ratio

    @staticmethod
    def _error_model(value):
        """
        Validate the ``error_model`` argument.

        Parameters
        ----------
        value : str
            ``'polls'`` or ``'average'``.

        Returns
        -------
        str
            The same value.

        Raises
        ------
        ValueError
            Anything else (``None`` included).
        """
        if isinstance(value, str) and value in ('polls', 'average'):
            return value
        raise ValueError('backtest: error_model must be "polls" or "average" (got {!r})'.format(value))

    @staticmethod
    def _drift_scale(value):
        """
        Normalise the ``drift_scale`` argument.

        Parameters
        ----------
        value : str, float, int or None
            ``None`` or ``'auto'`` (kept), or a positive number, also as a numeric string (``'1.5'``).

        Returns
        -------
        str, float or None
            ``None``, ``'auto'`` or the number as a float.

        Raises
        ------
        ValueError
            Anything else (a boolean, zero, a negative or non-finite number included).
        """
        if value is None or value == 'auto':
            return value
        try:
            scale = None if isinstance(value, bool) else float(value)
        except (TypeError, ValueError):
            scale = None
        if scale is None or not math.isfinite(scale) or scale <= 0:
            raise ValueError('backtest: drift_scale must be "auto", a positive number or null (got {!r})'.format(value))
        return scale

    @staticmethod
    def _tail_dof(value):
        """
        Normalise the ``tail_dof`` argument.

        Parameters
        ----------
        value : int, str or None
            ``None`` (kept), or an integer of at least 2, also as an integer string (``'5'``).

        Returns
        -------
        int or None
            ``None`` or the integer.

        Raises
        ------
        ValueError
            Anything else (a boolean, a float, 0, 1 or a negative number included).
        """
        if value is None:
            return None
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError('backtest: tail_dof must be an integer of at least 2 or null (got {!r})'.format(value))
        return value

    @staticmethod
    def _config(value):
        """
        Read and validate the ``config`` argument.

        Parameters
        ----------
        value : dict, str or None
            ``None`` (kept), a configuration of the average or the path of a JSON file with one.

        Returns
        -------
        dict or None
            ``None`` or a copy of the configuration.

        Raises
        ------
        ValueError
            The file cannot be read or is not JSON, the value is not a dict, a key is missing or unknown, a
            value has the wrong type, or ``reg_params``, ``he_params``, ``disp_params`` or ``bw_kwargs`` hold a
            key the Forecaster does not read (``CONFIG_PARAMS``, ``BW_KWARGS``; see ``run``).
        """
        if value is None:
            return None
        if isinstance(value, str):
            try:
                with open(value, encoding='utf-8') as handle:
                    value = json.load(handle)
            except OSError as e:
                raise ValueError('backtest: config file {} cannot be read ({})'.format(value, e)) from e
            except json.JSONDecodeError as e:
                raise ValueError('backtest: config file {} is not valid JSON ({})'.format(value, e)) from e
        if not isinstance(value, dict):
            raise ValueError('backtest: config must be a dict or the path of a JSON file with one (got {!r})'.format(
                value))
        missing, unknown = set(CONFIG_TYPES) - set(value), set(value) - set(CONFIG_TYPES)
        if missing or unknown:
            raise ValueError('backtest: config keys must be {} (missing {}, unknown {})'.format(
                ', '.join(CONFIG_TYPES), sorted(missing), sorted(unknown)))
        for key, kind in CONFIG_TYPES.items():
            if kind is None:
                continue
            item = value[key]
            wrong = not isinstance(item, kind) or (kind is int and isinstance(item, bool))
            if wrong or (key == 'max_fc' and item < 0):
                raise ValueError('backtest: config {} must be {} (got {!r})'.format(
                    key, 'an integer of at least 0' if kind is int else 'a ' + kind.__name__, item))
        for key, names in CONFIG_PARAMS.items():
            unknown = sorted(set(value[key]) - set(names))
            if unknown:
                raise ValueError('backtest: config {} has unknown keys {} (expected some of {})'.format(
                    key, unknown, ', '.join(names)))
        bw_kwargs = value['reg_params'].get('bw_kwargs')
        if bw_kwargs is not None:
            if not isinstance(bw_kwargs, dict) or set(bw_kwargs) - set(BW_KWARGS):
                raise ValueError('backtest: config reg_params bw_kwargs must be a dict with some of {} (got {!r})'
                                 .format(', '.join(BW_KWARGS), bw_kwargs))
        weights = value['weights']
        if weights is not None:
            valid = isinstance(weights, dict) and 'wspan' in weights and set(weights) <= {'wspan', 'ol_max'} and all(
                isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in weights.values())
            if not valid:
                raise ValueError('backtest: config weights must be null or {{"wspan": n}} with an optional "ol_max", '
                                 'positive integers (got {!r})'.format(weights))
        return copy.deepcopy(value)

    @staticmethod
    def _config_switches(config, **flags):
        """
        Align the loose flags with a configuration, which wins.

        Parameters
        ----------
        config : dict
            A configuration validated by ``_config``.
        **flags
            ``house_effects``, ``dispersion``, ``herding`` and ``max_fc`` as given to ``run``.

        Returns
        -------
        tuple
            The values of the configuration, in the order of ``CONFIG_SWITCHES``.

        Raises
        ------
        ValueError
            A flag set to something other than its default and than the configuration.
        """
        for key, default in CONFIG_SWITCHES.items():
            given = flags[key]
            if given != default and given != config[key]:
                raise ValueError('backtest: {} contradicts the config ({!r} given, {!r} in the config "{}")'.format(
                    key, given, config[key], config['label']))
        return tuple(config[key] for key in CONFIG_SWITCHES)
