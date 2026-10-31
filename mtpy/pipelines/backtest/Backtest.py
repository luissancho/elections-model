import os

from ...jobs.progress import counted, notes_suffix
from ...lib import backtest
from ...lib.publish import REPO_ROOT
from ..base import ScopePipeline


def shown(out):
    """Results folder as shown in the header: relative to the repository when inside it, absolute otherwise."""
    path = os.path.abspath(out or backtest.RESULTS_ROOT)
    if path == REPO_ROOT or path.startswith(REPO_ROOT + os.sep):
        return os.path.relpath(path, REPO_ROOT)
    return path


class Backtest(ScopePipeline):
    """
    Run the model backtest of one scope on past elections and write its tables (`backtest.backtest_scope`).

    The chunk is the scope; every outcome is a result dict (see `evaluate_task`) and a failure never raises. At
    verbose 1 or more, the ``by_horizon`` table of a completed scope is printed after its result line.

    Attributes
    ----------
    table : pandas.DataFrame or None
        ``by_horizon`` table of the last scope, if it completed.
    """

    label = None
    steps = None
    pipes = {'run': {'Setup': ['prepare'], 'Load': ['evaluate']}}

    def start(self, n_sim=1000, out=None, **kwargs) -> None:
        """Set the run and print the header ``backtest: n ámbitos · n simulaciones · folder``."""
        super().start(**kwargs)
        self._say('backtest: {} · {} · {}'.format(counted(self.total, 'ámbito', 'ámbitos'),
                                                  counted(n_sim, 'simulación', 'simulaciones'), shown(out)))

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

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line of one scope result, with the count of its warnings."""
        if result['status'] != 'completed':
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
