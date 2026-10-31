from ...lib import publish
from .Point import Point


class Unpublish(Point):
    """
    Remove a run of each scope from the web bundle and rebuild its history (``publish.unpublish``).

    The chunk is the scope. Each scope ends ``published`` (with ``removed`` and the new ``latest``) or ``failed``;
    a failure never raises.
    """

    label = 'unpublish'

    def move_task(self, chunk=None, run=None, **kwargs):
        """
        Remove the run of the scope, converting the outcome into a result dict (``self.result``).

        Parameters
        ----------
        chunk : str
            Scope to act on.
        run : str
            Run id.
        """
        try:
            self.result = {'status': 'published', **publish.unpublish(self.writer, chunk, run)}
        except ValueError as e:
            self.result = {'status': 'failed', 'reason': str(e)}
        except Exception as e:
            self.result = self.failed(chunk, e)
