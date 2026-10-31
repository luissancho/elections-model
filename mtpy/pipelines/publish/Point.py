from ...lib import bundle, publish
from ..base import ScopePipeline
from .Forecast import result_line


class Point(ScopePipeline):
    """
    Point the manifest of each scope to a run of its history (``publish.point``).

    The chunk is the scope. Each scope ends ``published`` (with ``latest``) or ``failed`` (the run does not exist,
    or any other error); a failure never raises. There is no live progress, only the summary line.

    Attributes
    ----------
    writer : BundleWriter
        Bundle writer (under the dry-run prefix in a dry run).
    """

    label = 'point'
    pipes = {'publish': {'Load': ['move']}}

    def start(self, fs=None, dry_run=False, **kwargs) -> None:
        """Set the run and create the bundle writer (under the dry-run prefix in a dry run)."""
        super().start(**kwargs)
        self.writer = bundle.BundleWriter(fs, prefix=bundle.DRY_PREFIX if dry_run else bundle.PREFIX)

    def move_task(self, chunk=None, run=None, **kwargs):
        """
        Point the scope to the run, converting the outcome into a result dict (``self.result``).

        Parameters
        ----------
        chunk : str
            Scope to act on.
        run : str
            Run id.
        """
        try:
            publish.point(self.writer, chunk, run)
            self.result = {'status': 'published', 'latest': run}
        except ValueError as e:
            self.result = {'status': 'failed', 'reason': str(e)}
        except Exception as e:
            self.result = self.failed(chunk, e)

    def line_of(self, chunk, result, run=None, **kwargs):
        """Format the summary line of one scope result."""
        return result_line(chunk, result, run)
