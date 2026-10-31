from ...lib import compute
from .Compute import Compute


class Ratings(Compute):
    """
    Compute the cross-scope pollster ratings of the events of a scope (`compute.rate_scope`, with the polls of
    every scope).

    Same chunk, result and lines as `Compute`; only the action and the unit of work change.
    """

    label = 'ratings'
    ACTION = 'ratings'

    def unit(self, chunk, dry_run, computer, featured) -> dict:
        """Run `compute.rate_scope` on the scope and return its figures."""
        return compute.rate_scope(chunk, save=not dry_run, verbose=self.model_verbose, progress=self.progress,
                                  computer=computer)
