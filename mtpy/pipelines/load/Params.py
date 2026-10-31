from ...jobs.progress import collect_warnings, counted, notes_suffix
from ...lib import load
from ..base import ScopePipeline
from .Polls import STEPS


class Params(ScopePipeline):
    """
    Rewrite the ``parties`` and ``bmaps`` of every election of a scope in ``data/params.json`` (`load.load_params`).

    The chunk is the scope; every outcome is a result dict (see `load_task`) and a failure never raises.
    """

    label = 'params'
    steps = STEPS
    pipes = {'update': {'Setup': ['prepare'], 'Load': ['load']}}

    def start(self, dry_run=False, **kwargs) -> None:
        """Set the run and print the header ``load params: n ámbitos[ · dry run]``."""
        super().start(**kwargs)
        self._say('load params: {}{}'.format(counted(self.total, 'ámbito', 'ámbitos'),
                                              ' · dry run' if dry_run else ''))

    def prepare_task(self, chunk=None, position=(1, 1), **kwargs):
        """Create and start the progress of the scope."""
        self.progress = self.progress_of(chunk, position)
        self.progress.start()

    def load_task(self, chunk=None, dry_run=False, params_loader=None, skip_missing=False, **kwargs):
        """
        Rewrite the parameters of one scope, converting every outcome into a result dict (``self.result``).

        Parameters
        ----------
        chunk : str
            Scope to rewrite.
        dry_run : bool
            Report without writing.
        params_loader : callable, optional
            Replacement of `load.load_params` (for tests).
        skip_missing : bool, default False
            Turn a scope without elections into ``skipped`` instead of ``failed``.

        Notes
        -----
        The result is that of `load.load_params` or ``{'status': 'failed' | 'skipped', 'action': 'params',
        'reason'}``, with the ``warnings`` raised.
        """
        notes = []
        try:
            with collect_warnings(notes, self.progress.note):
                result = (params_loader or load.load_params)(
                    chunk, save=not dry_run, verbose=self.model_verbose, progress=self.progress,
                    report=lambda text, level=1: self._say(text, level))
        except Exception as e:
            result = {**self.failed(chunk, e), 'action': 'params'}

        if skip_missing and result['status'] == 'failed' and result['reason'] == 'ValueError: ' + load.NO_EVENTS.format(
                chunk):
            result['status'], result['reason'] = 'skipped', load.NO_EVENTS.format(chunk)
        result['warnings'] = notes
        self.result = result

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line of one scope result."""
        if result['status'] in ('failed', 'skipped'):
            return '{}: params {} ({})'.format(chunk, result['status'], result['reason']) + notes_suffix(result)

        return '{}: params {} ({} elections, {} changed, {} skipped{}) in {:.0f} s'.format(
            chunk, result['status'], result['n_events'], result['n_changed'], result['n_skipped'],
            ', dry run' if result['status'] == 'checked' else '', result['seconds']['total']) + notes_suffix(result)
