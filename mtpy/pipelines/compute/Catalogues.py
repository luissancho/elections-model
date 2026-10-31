from ...lib import compute
from ..base import ScopePipeline
from .Compute import ACTIONS, CATALOGUES, Compute


class Catalogues(ScopePipeline):
    """
    Replicate the versioned catalogues (scopes, districts and the prior quality of the pollsters,
    `compute.replicate_catalogues`) once, not per scope.

    The result is keyed ``catalogues`` and a failure never raises (it is a failed result).
    """

    chunked = False
    label = ACTIONS[CATALOGUES]
    pipes = {'update': {'Load': ['replicate']}}

    def start(self, dry_run=False, **kwargs) -> None:
        """Set the run and print the header ``compute catálogos[ · dry run]``."""
        super().start(**kwargs)
        self._say('compute {}{}'.format(ACTIONS[CATALOGUES], ' · dry run' if dry_run else ''))

    def replicate_task(self, dry_run=False, catalogues=None, **kwargs):
        """
        Replicate the catalogues, converting every outcome into a result dict (``self.result``).

        Parameters
        ----------
        dry_run : bool
            Validate the files without writing.
        catalogues : callable, optional
            Replacement of `compute.replicate_catalogues`, ``catalogues(save) -> dict`` (for tests).
        """
        name = ACTIONS[CATALOGUES]
        self.result = self.collect(CATALOGUES, lambda: {
            'status': 'computed', 'action': name, **(catalogues or compute.replicate_catalogues)(save=not dry_run)},
            action=name)

    def failure_name(self, chunk) -> str:
        """Name of the failed action in the job error: ``catalogues (catálogos)``."""
        return '{} ({})'.format(CATALOGUES, ACTIONS[CATALOGUES])

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line, shared with the other actions of the group."""
        return Compute.line_of(self, CATALOGUES, result)
