from ...lib import publish
from .base import PublishPipeline


class Study(PublishPipeline):
    """
    Publish a study to the web bundle (``publish.publish_study``).

    It runs once per job (not per scope), with no LOREG guard. It ends ``published``, ``refused`` (a registration
    refused: ``publish.PublishRefused``) or ``failed`` (any other error); a failure never raises.

    Attributes
    ----------
    writer : BundleWriter
        Bundle writer (under the dry-run prefix in a dry run).
    study : str
        Slug of the study.
    """

    chunked = False
    label = 'study'
    pipes = {'publish': {'Setup': ['prepare'], 'Load': ['publish']}}

    def start(self, fs=None, dry_run=False, study=None, **kwargs) -> None:
        """Set the run: the bundle writer (under the dry-run prefix in a dry run) and the slug of the study."""
        super().start(**kwargs)
        self.writer = self.writer_for(fs, dry_run)
        self.study = study

    def prepare_task(self, **kwargs):
        """Start the live progress of the study."""
        self.start_progress(self.study, (1, 1))

    def publish_task(self, register=False, force=False, **kwargs):
        """
        Publish the study, converting every outcome into a result dict (``self.result``).

        Parameters
        ----------
        register : bool, default False
            Register the prediction of the study.
        force : bool, default False
            Replace an existing registration.

        Notes
        -----
        The result has a ``status``: ``published`` with ``registered_at``, ``checked`` and ``seconds``, ``refused``
        for ``publish.PublishRefused`` (a registration refused) and ``failed`` for any other exception, both with a
        ``reason``.
        """
        try:
            out = publish.publish_study(self.study, self.writer, register=register, force=force)
        except publish.PublishRefused as e:
            self.result = {'status': 'refused', 'reason': str(e)}
        except Exception as e:
            self.result = self.failed(self.study, e)
        else:
            self.result = {'status': 'published', 'registered_at': out['registered_at'], 'checked': out['checked'],
                           'seconds': out['seconds']}

    def failure_name(self, chunk) -> str:
        """Name of the failure of the study: ``'<study> (study)'``."""
        return '{} (study)'.format(self.study)

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line of the result of the study."""
        study = self.study
        if result['status'] == 'published':
            registered = result['registered_at']
            registered = 'registrado el {}'.format(registered) if registered else 'sin registro'
            return '{}: study published ({}, {}) in {:.0f} s'.format(
                study, registered, 'comprobado' if result['checked'] else 'sin comprobar', result['seconds']['total'])
        return '{}: study {} ({})'.format(study, result['status'], result['reason'])
