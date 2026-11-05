from ...lib import bundle, publish
from .base import PublishPipeline


class Studies(PublishPipeline):
    """
    Republish the live studies to the web bundle (``publish.publish_study`` of each one).

    It runs once per job (not per scope), with no LOREG guard, over the studies of ``bundle.STUDIES`` flagged
    ``live`` in ``publish.STUDY_RUNNERS``, in that order. It never registers a prediction. The exception of a study
    is caught, so the next one still runs; the result is ``failed`` when any study failed, and nothing raises.

    Attributes
    ----------
    writer : BundleWriter
        Bundle writer (under the dry-run prefix in a dry run).
    """

    chunked = False
    label = 'studies'
    pipes = {'publish': {'Setup': ['prepare'], 'Load': ['publish']}}

    def start(self, fs=None, dry_run=False, **kwargs) -> None:
        """Set the run and create the bundle writer (under the dry-run prefix in a dry run)."""
        super().start(**kwargs)
        self.writer = self.writer_for(fs, dry_run)

    def prepare_task(self, **kwargs):
        """Start the live progress of the studies."""
        self.start_progress('studies', (1, 1))

    def publish_task(self, **kwargs):
        """
        Publish every live study, converting the outcomes into a result dict (``self.result``).

        Notes
        -----
        The result is ``{'status': 'published' | 'failed', 'studies': {study: {'status': 'published', 'seconds'} |
        {'status': 'failed', 'reason'}}}``; ``failed`` when any study failed.
        """
        studies = {}
        for study in bundle.STUDIES:
            if not publish.STUDY_RUNNERS[study]['live']:
                continue
            try:
                out = publish.publish_study(study, self.writer)
            except Exception as e:
                studies[study] = self.failed(study, e)
            else:
                studies[study] = {'status': 'published', 'seconds': out['seconds']}
        failed = any(result['status'] == 'failed' for result in studies.values())
        self.result = {'status': 'failed' if failed else 'published', 'studies': studies}

    def failure_name(self, chunk) -> str:
        """Name of the failure: ``'studies'``."""
        return 'studies'

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line: one fragment per study, joined by ``, ``."""
        parts = []
        for study, outcome in result['studies'].items():
            if outcome['status'] == 'published':
                parts.append('{} published in {:.0f} s'.format(study, outcome['seconds']['total']))
            else:
                parts.append('{} failed ({})'.format(study, outcome['reason']))
        return 'studies: ' + ', '.join(parts)
