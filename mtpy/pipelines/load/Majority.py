from ...jobs.progress import counted, notes_suffix
from ..base import ScopePipeline
from .Polls import STEPS


class Majority(ScopePipeline):
    """
    Take the frozen snapshot of the published run for the study of the CERA vote (``data/cera/majority.json``,
    `mtpy.lib.studies.cera.snapshot.take_snapshot`).

    The chunk is the scope; every outcome is a result dict (see `snapshot_task`). A refusal of the snapshot is a
    ``refused`` result and any other exception a ``failed`` one, so nothing raises. The action ``rebulk`` (the
    ``overwrite`` of the job) lets the snapshot replace an existing file.
    """

    label = 'majority'
    steps = STEPS
    pipes = {'update': {'Setup': ['prepare'], 'Load': ['snapshot']},
             'rebulk': {'Setup': ['prepare'], 'Load': ['snapshot']}}

    def start(self, dry_run=False, **kwargs) -> None:
        """Set the run and print the header ``load majority: n ámbitos[ · overwrite][ · dry run]``."""
        super().start(**kwargs)
        flags = ''.join(text for active, text in ((self._action == 'rebulk', ' · overwrite'), (dry_run, ' · dry run'))
                        if active)
        self._say('load majority: {}{}'.format(counted(self.total, 'ámbito', 'ámbitos'), flags))

    def prepare_task(self, chunk=None, position=(1, 1), **kwargs):
        """Create and start the progress of the scope."""
        self.start_progress(chunk, position)

    def snapshot_task(self, chunk=None, dry_run=False, run_id=None, snapshotter=None, **kwargs):
        """
        Take the snapshot of one scope, converting every outcome into a result dict (``self.result``).

        Parameters
        ----------
        chunk : str
            Scope to snapshot.
        dry_run : bool
            Rebuild and check the run without writing the file.
        run_id : str, optional
            Published run to freeze; the latest one by default.
        snapshotter : callable, optional
            Replacement of `take_snapshot` (for tests).

        Notes
        -----
        The result is that of `take_snapshot`, ``{'status': 'refused', 'action': 'majority', 'reason'}`` when it
        raises `SnapshotRefused`, or the ``failed`` result of `collect` for any other exception.
        """
        from ...lib.studies.cera.snapshot import SnapshotRefused, take_snapshot

        def work():
            try:
                return (snapshotter or take_snapshot)(
                    chunk, run_id=run_id, save=not dry_run, overwrite=self._action == 'rebulk',
                    verbose=self.model_verbose, progress=self.progress,
                    report=lambda text, level=1: self._say(text, level))
            except SnapshotRefused as e:
                return {'status': 'refused', 'action': 'majority', 'reason': str(e)}

        self.result = self.collect(chunk, work, action='majority')

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line of one scope result."""
        if result['status'] in ('failed', 'refused', 'skipped'):
            return '{}: majority {} ({})'.format(chunk, result['status'], result['reason']) + notes_suffix(result)

        base = result['base']
        return '{}: majority {} (run {}, right {} seats, majority in {} of {}{}) in {:.0f} s'.format(
            chunk, result['status'], result['run_id'], base['right'], base['with_majority'], result['n_sim'],
            ', dry run' if result['status'] == 'checked' else '', result['seconds']['total']) + notes_suffix(result)
