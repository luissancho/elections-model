import os

from ..core.worker import Job
from ..lib import backtest
from ..lib.publish import REPO_ROOT, resolve_scopes
from .progress import ScopeReporter, collect_warnings, counted, notes_suffix


class Backtest(ScopeReporter, Job):
    """
    Run the model backtest on past elections and write its tables to ``backtest/results`` (the job form of
    ``backtest/run_backtest.py``), the input of ``publish backtest`` and of the model slice of ``publish event``.

    Scope by scope (`backtest.backtest_scope`), it evaluates every (election, horizon) case out of sample, writes
    the seven tables (``shares``, ``seats``, ``blocks``, ``provinces``, ``meta``, ``metrics`` and ``by_horizon``)
    and ``run.json`` (parameters, elections and horizons evaluated, commit and state of the polls in the database)
    into ``out`` for ``es`` and ``out/{scope}`` for the other scopes, overwriting the previous run. Scopes are
    isolated: one that fails never stops the others, and the job exits with an error at the end if any failed. A
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

    In Docker (JSON without spaces)::

        RUN_JOB='backtest {"scopes":"all"}'
    """

    verbose = 1

    def run(self, scopes=('es',), events=None, horizons=None, n_sim=1000, seed=42, max_fc=10, nowcast_only=False,
            house_effects=True, industry_bias=False, dispersion=True, herding=True, composition=None,
            regional_noise=True, min_events=None, recompute_ratings=False, out=None, verbose=1, runner=None, stats=None, **kwargs):
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
        recompute_ratings : bool, default False
            Recompute the pollster ratings in memory with the current code (once per scope) and use them in
            every case instead of the stored ones; nothing is saved to the database.
        out : str, optional
            Results folder; ``backtest/results`` of the repository by default (a relative path is taken from
            the working directory).
        verbose : int, default 1
            Verbosity of the console output (see the class).
        runner, stats : callable, optional
            Replacements of `backtest.run_backtest` and `backtest.poll_stats` (for tests).
        **kwargs
            Ignored extra job arguments.

        Returns
        -------
        dict[str, dict]
            Result of each scope: ``status`` (``completed``, ``skipped`` or ``failed``) and, when completed,
            ``n_cases``, ``n_events``, ``n_horizons``, ``events``, ``horizons``, ``folder`` and ``seconds``;
            otherwise the ``reason``. Every result carries the ``warnings`` of the backtest (the cases skipped).

        Raises
        ------
        ValueError
            If ``composition`` is not ``'auto'``, a ratio in (0, 1] or ``None``.
        RuntimeError
            If any scope failed (raised after the summary and the alert).
        """
        self.verbose = verbose
        params = dict(events=events, horizons=horizons, n_sim=n_sim, seed=seed, max_fc=max_fc,
                      nowcast_only=nowcast_only, house_effects=house_effects, industry_bias=industry_bias,
                      dispersion=dispersion, herding=herding, composition=self._composition(composition),
                      regional_noise=regional_noise, min_events=min_events, recompute_ratings=recompute_ratings,
                      verbose=max(0, verbose - 1),
                      runner=runner, stats=stats)
        targets = resolve_scopes(scopes)
        self._say('backtest: {} · {} · {}'.format(counted(len(targets), 'ámbito', 'ámbitos'),
                                                  counted(n_sim, 'simulación', 'simulaciones'), self._shown(out)))
        results = {}
        failures = []
        lines = []

        for index, scope in enumerate(targets, 1):
            progress = self._progress(scope, index, len(targets))
            progress.start()
            result, table = self._scope(scope, out, progress, params)
            results[scope] = result
            if result['status'] == 'failed':
                failures.append(scope)
            lines.append(self._line(scope, result))
            progress.done(lines[-1])
            if table is not None and self.verbose >= 1:
                print(table.round(3).to_string(index=False), flush=True)

        for line in lines:
            print(line)
        if self.app.logger is not None:
            self.alert('\n'.join(lines))

        if failures:
            raise RuntimeError('backtest: failed scopes: ' + ', '.join(failures))

        return results

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
    def _shown(out):
        """Results folder as shown in the header: relative to the repository when inside it, absolute otherwise."""
        path = os.path.abspath(out or backtest.RESULTS_ROOT)
        if path == REPO_ROOT or path.startswith(REPO_ROOT + os.sep):
            return os.path.relpath(path, REPO_ROOT)
        return path

    def _scope(self, scope, out, progress, params):
        """
        Run the backtest of one scope, converting every outcome into a result dict.

        Parameters
        ----------
        scope : str
            Scope to evaluate.
        out : str or None
            Results folder.
        progress : ScopeProgress
            Live progress of the scope (its warnings).
        params : dict
            Arguments of `backtest.backtest_scope` other than the scope and the folder.

        Returns
        -------
        tuple
            ``(result, table)``: ``{'status': 'completed', ...figures}``, ``{'status': 'skipped', 'reason'}`` or
            ``{'status': 'failed', 'reason'}``, each with its ``warnings``, and the ``by_horizon`` table of a
            completed scope (``None`` otherwise).
        """
        notes = []
        try:
            with collect_warnings(notes, progress.note):
                figures = backtest.backtest_scope(scope, out=out, **params)
        except backtest.NothingToEvaluate as e:
            return {'status': 'skipped', 'reason': str(e), 'warnings': notes}, None
        except Exception as e:
            self._log_error(scope)
            return {'status': 'failed', 'reason': '{}: {}'.format(type(e).__name__, e), 'warnings': notes}, None
        table = figures.pop('by_horizon')

        return {'status': 'completed', **figures, 'warnings': notes}, table

    @staticmethod
    def _line(scope, result):
        """Format the summary line of one scope result, with the count of its warnings."""
        if result['status'] != 'completed':
            line = '{}: {} ({})'.format(scope, result['status'], result['reason'])
        else:
            line = '{}: backtest completed ({} cases, {} events, {} horizons) in {:.0f} s'.format(
                scope, result['n_cases'], result['n_events'], result['n_horizons'], result['seconds']['total'])
        return line + notes_suffix(result)
