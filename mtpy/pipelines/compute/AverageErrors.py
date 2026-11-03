from ...lib import compute
from .Compute import ACTIONS, Compute


class AverageErrors(Compute):
    """
    Compute the errors of the poll average of the past elections of a scope (`compute.average_errors_scope`), the
    table that only the uncertainty options of method 15 read: an action of its own, run after a ``full`` compute
    of the scope (its Forecasters take the prior of the house effects from the stored table).

    Same chunk, result and lines as `Compute`; only the action and the unit of work change. It never skips nor
    picks a mode, and it never marks ``computed_at``.
    """

    label = ACTIONS['average_errors']
    ACTION = 'average_errors'

    def unit(self, chunk, dry_run, computer, featured) -> dict:
        """Run `compute.average_errors_scope` on the scope and return its figures."""
        return {'dry_run': dry_run,
                **compute.average_errors_scope(chunk, save=not dry_run, verbose=self.model_verbose,
                                               progress=self.progress, computer=computer)}
