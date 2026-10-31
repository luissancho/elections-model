from ...jobs.progress import counted, notes_suffix
from ...lib import backtest
from ..base import ScopePipeline


class Backtest(ScopePipeline):
    """
    Run the model backtest of one scope on past elections and write its tables (`backtest.backtest_scope`), or
    copy them between the repository copy and the app file system (S3 in ECS).

    Actions: ``run`` (default) evaluates and writes to ``out`` (the app file system when ``out`` is not given);
    ``push`` copies the tables of ``out`` (``backtest/results`` of the repository by default) to the app file
    system, to seed S3 with the repository copy; ``pull`` copies them from the app file system to ``out``, to
    review ``by_horizon.csv`` and commit what a backtest run in ECS left in S3. Both mirror the source: files it
    lacks are removed from the destination. From the job: ``{"action":"push","scopes":"all"}`` and
    ``{"action":"pull","scopes":"all"}``.

    The chunk is the scope; every outcome is a result dict (see `evaluate_task`) and a failure never raises. At
    verbose 1 or more, the ``by_horizon`` table of a completed scope is printed after its result line.

    Attributes
    ----------
    table : pandas.DataFrame or None
        ``by_horizon`` table of the last scope, if it completed.
    """

    label = None
    steps = None
    pipes = {'run': {'Setup': ['prepare'], 'Load': ['evaluate']},
             'push': {'Setup': ['prepare'], 'Load': ['push']},
             'pull': {'Setup': ['prepare'], 'Load': ['pull']}}

    def start(self, n_sim=1000, out=None, **kwargs) -> None:
        """Set the run and print its header: ``backtest: n ámbitos · n simulaciones · folder`` or, for push and
        pull, ``backtest push|pull: n ámbitos · origin → destination``."""
        super().start(**kwargs)
        scopes = counted(self.total, 'ámbito', 'ámbitos')
        if self._action == 'run':
            self._say('backtest: {} · {} · {}'.format(scopes, counted(n_sim, 'simulación', 'simulaciones'),
                                                      backtest.describe_source(*backtest.results_fs(out, 'es'))))
            return
        local = backtest.describe_source(*backtest.results_fs(out or backtest.RESULTS_ROOT, 'es'))
        remote = backtest.describe_source(*backtest.results_fs(None, 'es'))
        origin, target = (local, remote) if self._action == 'push' else (remote, local)
        self._say('backtest {}: {} · {} → {}'.format(self._action, scopes, origin, target))

    def prepare_task(self, chunk=None, position=(1, 1), **kwargs):
        """Create and start the progress of the scope."""
        self.start_progress(chunk, position)
        self.table = None

    def evaluate_task(self, chunk=None, out=None, events=None, horizons=None, n_sim=1000, seed=42, max_fc=10,
                      nowcast_only=False, house_effects=True, industry_bias=False, dispersion=True, herding=True,
                      composition=None, regional_noise=True, min_events=None, recompute_ratings=False, runner=None,
                      stats=None, **kwargs):
        """
        Run the backtest of one scope, converting every outcome into a result dict (``self.result``).

        Parameters
        ----------
        chunk : str
            Scope to evaluate.
        out : str, optional
            Results folder.
        events, horizons, n_sim, seed, max_fc, nowcast_only, house_effects, industry_bias, dispersion, herding, \
composition, regional_noise, min_events, recompute_ratings
            Arguments of `backtest.backtest_scope`.
        runner, stats : callable, optional
            Replacements of `backtest.run_backtest` and `backtest.poll_stats` (for tests).

        Notes
        -----
        The result is ``{'status': 'completed', ...figures}``, ``{'status': 'skipped', 'reason'}`` or
        ``{'status': 'failed', 'reason'}``, each with its ``warnings``; the ``by_horizon`` table of a completed
        scope is kept in ``self.table``.
        """
        def work():
            try:
                figures = backtest.backtest_scope(
                    chunk, out=out, events=events, horizons=horizons, n_sim=n_sim, seed=seed, max_fc=max_fc,
                    nowcast_only=nowcast_only, house_effects=house_effects, industry_bias=industry_bias,
                    dispersion=dispersion, herding=herding, composition=composition,
                    regional_noise=regional_noise, min_events=min_events, recompute_ratings=recompute_ratings,
                    verbose=max(0, self.verbose - 1), runner=runner, stats=stats)
            except backtest.NothingToEvaluate as e:
                return {'status': 'skipped', 'reason': str(e)}
            self.table = figures.pop('by_horizon')
            return {'status': 'completed', **figures}

        self.result = self.collect(chunk, work)

    def push_task(self, chunk=None, out=None, **kwargs):
        """Copy the tables of the scope from ``out`` (the repository copy by default) to the app file system."""
        self._copy(chunk, backtest.results_fs(out or backtest.RESULTS_ROOT, chunk), backtest.results_fs(None, chunk),
                   'pushed')

    def pull_task(self, chunk=None, out=None, **kwargs):
        """Copy the tables of the scope from the app file system to ``out`` (the repository copy by default)."""
        self._copy(chunk, backtest.results_fs(None, chunk), backtest.results_fs(out or backtest.RESULTS_ROOT, chunk),
                   'pulled')

    def _copy(self, chunk, source, destination, status) -> None:
        """Run `backtest.copy_results` for one scope and keep its result dict (``skipped`` without source tables)."""
        def work():
            try:
                done = backtest.copy_results(chunk, source, destination)
            except backtest.NothingToEvaluate as e:
                return {'status': 'skipped', 'reason': str(e)}
            return {'status': status, 'n_files': len(done['copied']), 'n_removed': len(done['removed']),
                    'folder': backtest.describe_source(*destination)}

        self.result = self.collect(chunk, work)

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line of one scope result, with the count of its warnings."""
        if result['status'] in ('pushed', 'pulled'):
            line = '{}: backtest {} ({}{}) to {}'.format(
                chunk, result['status'], counted(result['n_files'], 'file', 'files'),
                ', {} removed'.format(result['n_removed']) if result['n_removed'] else '', result['folder'])
        elif result['status'] != 'completed':
            line = '{}: {} ({})'.format(chunk, result['status'], result['reason'])
        else:
            line = '{}: backtest completed ({} cases, {} events, {} horizons) in {:.0f} s'.format(
                chunk, result['n_cases'], result['n_events'], result['n_horizons'], result['seconds']['total'])
        return line + notes_suffix(result)

    def summary(self, chunk, **kwargs):
        """Report the result line and, below it, the ``by_horizon`` table of a completed scope (verbose 1+)."""
        line = super().summary(chunk, **kwargs)
        if self.table is not None and self.verbose >= 1:
            print(self.table.round(3).to_string(index=False), flush=True)
        return line
