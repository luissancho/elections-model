from ...lib import bundle, publish
from ..base import ScopePipeline


class Backtest(ScopePipeline):
    """
    Publish the model backtest of each scope to the web bundle (``publish.publish_backtest``).

    The chunk is the scope. Each scope ends ``published``, ``skipped`` (no results for the scope:
    ``publish.NothingToPublish``) or ``failed`` (any other error); a failure never raises.

    Attributes
    ----------
    writer : BundleWriter
        Bundle writer (under the dry-run prefix in a dry run).
    """

    label = 'backtest'
    pipes = {'publish': {'Setup': ['prepare'], 'Load': ['publish']}}

    def start(self, fs=None, dry_run=False, **kwargs) -> None:
        """Set the run and create the bundle writer (under the dry-run prefix in a dry run)."""
        super().start(**kwargs)
        self.writer = bundle.BundleWriter(fs, prefix=bundle.DRY_PREFIX if dry_run else bundle.PREFIX)

    def prepare_task(self, chunk=None, position=(1, 1), **kwargs):
        """Start the live progress of the scope."""
        self.start_progress(chunk, position)

    def publish_task(self, chunk=None, source=None, **kwargs):
        """
        Publish the model backtest of one scope, converting every outcome into a result dict (``self.result``).

        Parameters
        ----------
        chunk : str
            Scope to publish.
        source : str or None
            Results folder (see ``publish.backtest_source``).

        Notes
        -----
        The result has a ``status``: ``published`` with ``n_cases``, ``n_events`` and ``seconds``, ``skipped`` for
        ``publish.NothingToPublish`` and ``failed`` for any other exception, both with a ``reason``.
        """
        try:
            out = publish.publish_backtest(chunk, self.writer, source=source)
        except publish.NothingToPublish as e:
            self.result = {'status': 'skipped', 'reason': str(e)}
        except Exception as e:
            self.result = self.failed(chunk, e)
        else:
            self.result = {'status': 'published', 'n_cases': out['n_cases'], 'n_events': out['n_events'],
                           'seconds': out['seconds']}

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line of one scope result."""
        if result['status'] == 'published':
            return '{}: backtest published ({} cases, {} elections) in {:.0f} s'.format(
                chunk, result['n_cases'], result['n_events'], result['seconds']['total'])
        return '{}: backtest {} ({})'.format(chunk, result['status'], result['reason'])
