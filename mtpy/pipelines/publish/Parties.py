from ...jobs.progress import counted
from ...lib import bundle, publish
from ..base import ScopePipeline


class Parties(ScopePipeline):
    """
    Write the party catalogue of the database next to the manifest (``publish.publish_parties``).

    It runs once per job (not per scope), also after ``forecast`` and ``analysis``, so the pages always colour and
    name every party. In a dry run it writes under ``site-dry/v1``; a failure is a failed result, never an
    exception.
    """

    chunked = False
    label = 'parties'
    pipes = {'publish': {'Load': ['write']}}

    def start(self, fs=None, dry_run=False, **kwargs) -> None:
        """Set the run and create the bundle writer (under the dry-run prefix in a dry run)."""
        super().start(**kwargs)
        self.writer = bundle.BundleWriter(fs, prefix=bundle.DRY_PREFIX if dry_run else bundle.PREFIX)

    def write_task(self, **kwargs):
        """
        Write the party catalogue, converting the outcome into a result dict (``self.result``).

        Notes
        -----
        The result is ``{'status': 'published', 'n_parties'}`` or ``{'status': 'failed', 'reason'}``.
        """
        try:
            out = publish.publish_parties(self.writer)
            self.result = {'status': 'published', 'n_parties': out['n_parties']}
        except Exception as e:
            self.result = self.failed('parties', e)

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line of the catalogue result."""
        if result['status'] == 'published':
            return 'parties: {}'.format(counted(result['n_parties'], 'partido', 'partidos'))
        return 'parties: failed ({})'.format(result['reason'])
