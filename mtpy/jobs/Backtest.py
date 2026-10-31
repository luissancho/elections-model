from ..core.worker import SequenceJob
from ..lib.publish import resolve_scopes


class Backtest(SequenceJob):
    """
    Run the model backtest on past elections and write its tables to ``backtest/results`` (the job form of
    ``backtest/run_backtest.py``), the input of ``publish backtest`` and of the model slice of ``publish event``.

    Scope by scope (`backtest.backtest_scope`), it evaluates every (election, horizon) case out of sample, writes
    the seven tables (``shares``, ``seats``, ``blocks``, ``provinces``, ``meta``, ``metrics`` and ``by_horizon``)
    and ``run.json`` (parameters, elections and horizons evaluated, commit and state of the polls in the database)
    into ``out`` for ``es`` and ``out/{scope}`` for the other scopes, overwriting the previous run. Scopes are
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

    In Docker (JSON without spaces)::

        RUN_JOB='backtest {"scopes":"all"}'
    """

    sequence = {'backtest': ['backtest']}
    default_action = 'run'
    stop_on_error = False
    chunk_noun = 'scopes'

    def run(self, scopes=('es',), events=None, horizons=None, n_sim=1000, seed=42, max_fc=10, nowcast_only=False,
            house_effects=True, industry_bias=False, dispersion=True, herding=True, composition=None,
            regional_noise=True, min_events=None, recompute_ratings=False, out=None, verbose=1, runner=None, stats=None,
            stop_on_error=None, **kwargs):
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
        stop_on_error : bool, optional
            Stop at the first failed scope instead of going on with the others; ``False`` by default.
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
            If ``composition`` is not ``'auto'``, a ratio in (0, 1] or ``None``, there are no scopes to run
            (``backtest: no scopes to run``) or ``stop_on_error`` is not true or false
            (``stop_on_error must be true or false``).
        RuntimeError
            If any scope failed (raised after the summary and the alert).
        """
        composition = self._composition(composition)
        targets = resolve_scopes(scopes)
        if not targets:
            raise ValueError('backtest: no scopes to run')

        super().run(select='backtest', chunks=targets, stop_on_error=stop_on_error, events=events, horizons=horizons,
                    n_sim=n_sim, seed=seed, max_fc=max_fc, nowcast_only=nowcast_only, house_effects=house_effects,
                    industry_bias=industry_bias, dispersion=dispersion, herding=herding, composition=composition,
                    regional_noise=regional_noise, min_events=min_events, recompute_ratings=recompute_ratings,
                    out=out, verbose=verbose, runner=runner, stats=stats)

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
