import warnings

from ...jobs.progress import counted, notes_suffix
from ...lib import tuning
from ...lib.backtest import describe_source
from ..base import ScopePipeline


class Tune(ScopePipeline):
    """
    Tune the configuration of the poll average of one scope on the polls published after each cut, or evaluate
    given configurations (`tuning.tuning_scope`), writing its tables to ``out`` (the app file system when ``out``
    is not given), or copy the tables between the repository copy and the app file system (S3 in ECS).

    Actions: ``run`` (default) runs the mode; ``push`` copies the tables of ``out`` (``backtest/tuning`` of the
    repository by default) to the app file system; ``pull`` copies them from the app file system to ``out``. Both
    mirror the tuning files of the scope (`tuning.copy_tuning`): those the source lacks are removed from the
    destination.

    The chunk is the scope; every outcome is a result dict (see `evaluate_task`) and a failure never raises: a
    scope with nothing to tune (`tuning.NothingToTune`) or, in ``shrinkage``, ``check`` and ``merge``, without the
    ``best.json`` of the previous phase ends ``skipped``; a tuning that cannot be scored (`tuning.TuningFailed`)
    ends ``failed``; ``push`` and ``pull`` of a scope without tables end ``skipped``.
    """

    label = None
    steps = None
    pipes = {'run': {'Setup': ['prepare'], 'Load': ['evaluate']},
             'push': {'Setup': ['prepare'], 'Load': ['push']},
             'pull': {'Setup': ['prepare'], 'Load': ['pull']}}

    def start(self, mode='smoothing', out=None, **kwargs) -> None:
        """Set the run and print its header: ``tune: n ámbitos · mode · folder`` or, for push and pull,
        ``tune push|pull: n ámbitos · origin → destination``."""
        super().start(**kwargs)
        scopes = counted(self.total, 'ámbito', 'ámbitos')
        if self._action == 'run':
            self._say('tune: {} · {} · {}'.format(scopes, mode, describe_source(*tuning.tuning_fs(out, 'es'))))
            return
        local = describe_source(*tuning.tuning_fs(out or tuning.TUNING_ROOT, 'es'))
        remote = describe_source(*tuning.tuning_fs(None, 'es'))
        origin, target = (local, remote) if self._action == 'push' else (remote, local)
        self._say('tune {}: {} · {} → {}'.format(self._action, scopes, origin, target))

    def prepare_task(self, chunk=None, position=(1, 1), **kwargs):
        """Create and start the progress of the scope."""
        self.start_progress(chunk, position)

    def evaluate_task(self, chunk=None, out=None, mode='smoothing', configs=None, grid=None, cycles=None,
                      cut_days=None, horizons=(7, 28), rounds=None, workers=1, runner=None, stats=None, **kwargs):
        """
        Run one mode of the tuning of one scope, converting every outcome into a result dict (``self.result``).

        Parameters
        ----------
        chunk : str
            Scope to tune.
        out : str, optional
            Local folder of the tables (the app file system by default).
        mode, configs, grid, cut_days, horizons, rounds
            Arguments of `tuning.tuning_scope`, already normalised by the job.
        cycles : list of str, optional
            Elections (``YYYY-MM-DD``) the tuning is restricted to (see `restrict_cycles`); every cycle of the
            scope by default. In ``shrinkage`` they make the run a part of the phase, named after them (sorted,
            joined by ``_``), for ``merge`` to join.
        workers : int, default 1
            Processes running the cuts.
        runner : callable, optional
            Replacement of `tuning.tuning_scope` (for tests).
        stats : callable, optional
            Replacement of `backtest.poll_stats` (for tests).

        Notes
        -----
        The result is ``{'status': 'completed', ...figures}`` (those of `tuning.tuning_scope`),
        ``{'status': 'skipped', 'reason'}`` or ``{'status': 'failed', 'reason'}``, each with its ``warnings``.
        """
        part = '_'.join(sorted(cycles)) if mode == 'shrinkage' and cycles is not None else None

        def work():
            try:
                figures = (runner or tuning.tuning_scope)(
                    chunk, out=out, mode=mode, configs=configs, grid=grid,
                    cycles=None if cycles is None else restrict_cycles(chunk, cycles), cut_days=cut_days,
                    horizons=horizons, rounds=rounds, verbose=max(0, self.verbose - 1), stats=stats,
                    workers=workers, part=part)
            except (tuning.NothingToTune, FileNotFoundError) as e:
                return {'status': 'skipped', 'reason': str(e)}
            return {'status': 'completed', **figures}

        self.result = self.collect(chunk, work)

    def push_task(self, chunk=None, out=None, **kwargs):
        """Copy the tables of the scope from ``out`` (the repository copy by default) to the app file system."""
        self._copy(chunk, tuning.tuning_fs(out or tuning.TUNING_ROOT, chunk), tuning.tuning_fs(None, chunk), 'pushed')

    def pull_task(self, chunk=None, out=None, **kwargs):
        """Copy the tables of the scope from the app file system to ``out`` (the repository copy by default)."""
        self._copy(chunk, tuning.tuning_fs(None, chunk), tuning.tuning_fs(out or tuning.TUNING_ROOT, chunk), 'pulled')

    def _copy(self, chunk, source, destination, status) -> None:
        """Run `tuning.copy_tuning` for one scope and keep its result dict (``skipped`` without source tables)."""
        def work():
            try:
                done = tuning.copy_tuning(chunk, source, destination)
            except tuning.NothingToTune as e:
                return {'status': 'skipped', 'reason': str(e)}
            return {'status': status, 'n_files': len(done['copied']), 'n_removed': len(done['removed']),
                    'folder': describe_source(*destination)}

        self.result = self.collect(chunk, work)

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line of one scope result, with the count of its warnings."""
        if result['status'] in ('pushed', 'pulled'):
            line = '{}: tune {} ({}{}) to {}'.format(
                chunk, result['status'], counted(result['n_files'], 'file', 'files'),
                ', {} removed'.format(result['n_removed']) if result['n_removed'] else '', result['folder'])
        elif result['status'] != 'completed':
            line = '{}: {} ({})'.format(chunk, result['status'], result['reason'])
        else:
            line = '{}: tune completed ({} configs, {} cuts) in {:.0f} s'.format(
                chunk, result['n_configs'], result['n_cuts'], result['seconds']['total'])
        return line + notes_suffix(result)


def restrict_cycles(scope, dates):
    """
    Cycles of a scope restricted to the given elections, keeping their partition.

    Parameters
    ----------
    scope : str
        Scope code.
    dates : list of str
        Elections (``YYYY-MM-DD``) to keep.

    Returns
    -------
    dict
        `tuning.search_cycles(scope)` with only the given elections in each partition. A date that is not a cycle
        of the scope is ignored with a warning.
    """
    found = tuning.search_cycles(scope)
    known = {date for partition in found.values() for date in partition}
    for date in dates:
        if date not in known:
            warnings.warn('{} is not a cycle of {} (ignored)'.format(date, scope))

    return {partition: [date for date in partition_dates if date in dates]
            for partition, partition_dates in found.items()}
