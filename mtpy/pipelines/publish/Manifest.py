from ...lib import publish
from .base import PublishPipeline


def manifest_line(data):
    """Format the summary line of the manifest data."""
    return 'manifest: {} scopes, freeze {}'.format(
        len(data.get('scopes', {})), 'on' if (data.get('freeze') or {}).get('active') else 'off')


class Manifest(PublishPipeline):
    """
    Write the manifest of the web bundle (``publish.update_manifest``), with the ``freeze`` notice when given.

    It runs once per job (not per scope). With ``dry_run`` nothing is written; a failure to write is a failed
    result, never an exception.
    """

    chunked = False
    label = 'manifest'
    pipes = {'publish': {'Load': ['write']}}

    def start(self, fs=None, dry_run=False, **kwargs) -> None:
        """Set the run and create the bundle writer (under the dry-run prefix in a dry run)."""
        super().start(**kwargs)
        self.writer = self.writer_for(fs, dry_run)

    def write_task(self, dry_run=False, freeze=None, **kwargs):
        """
        Update the manifest, converting the outcome into a result dict (``self.result``).

        Parameters
        ----------
        dry_run : bool, default False
            Leave the manifest alone: ``{'status': 'skipped', 'reason': 'dry run'}``.
        freeze : dict, optional
            Freeze notice of the manifest (``{'active': bool, 'message': str}``).

        Notes
        -----
        The result is ``{'status': 'published', **data}`` (the manifest data written) or
        ``{'status': 'failed', 'reason'}``.
        """
        if dry_run:
            self.result = {'status': 'skipped', 'reason': 'dry run'}
            return
        try:
            data = publish.update_manifest(self.writer, freeze=freeze)
            self.result = {'status': 'published', **data}
        except Exception as e:
            self.result = self.failed('manifest', e)

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line of the manifest result."""
        if result['status'] == 'skipped':
            return 'manifest: not written (dry run)'
        if result['status'] == 'published':
            return manifest_line(result)
        return 'manifest: failed ({})'.format(result['reason'])
