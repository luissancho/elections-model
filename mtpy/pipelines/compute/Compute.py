from ...jobs.progress import counted, notes_suffix
from ...lib import compute
from ..base import ScopePipeline

# Labels of the phases reported at verbose 2 (keys of `compute.COMPUTE_STEPS` and `compute.RATINGS_STEP`)
STEPS = {'weights': 'pesos', 'featured': 'eventos destacados', 'errors': 'errores', 'deviations': 'desviaciones',
         'drift': 'deriva', 'house_effects': 'efectos de casa', 'herding': 'rebaño', 'ratings': 'ratings'}
# Public name of each action, in the headers and the summary lines
ACTIONS = {'catalogues': 'catálogos', 'compute': 'series', 'ratings': 'ratings'}
# Key of the result of the global `catalogues` action (the others are keyed by scope)
CATALOGUES = 'catalogues'


class Compute(ScopePipeline):
    """
    Compute the poll series of a scope into the database: weights, featured events, errors, deviations, drift,
    house effects and herding (`compute.compute_scope`).

    The chunk is the scope; every outcome is a result dict (see `compute_task`) and a failure never raises. The
    subclass `Ratings` changes the action and the unit of work; this class also formats the lines of all the
    actions of the group (`line_of`).
    """

    label = 'series'
    steps = STEPS
    ACTION = 'compute'
    pipes = {'update': {'Setup': ['prepare'], 'Load': ['compute']}}

    def start(self, dry_run=False, **kwargs) -> None:
        """Set the run and print the header ``compute <action>: n ámbitos[ · dry run]``."""
        super().start(**kwargs)
        self._say('compute {}: {}{}'.format(ACTIONS[self.ACTION], counted(self.total, 'ámbito', 'ámbitos'),
                                             ' · dry run' if dry_run else ''))

    def prepare_task(self, chunk=None, position=(1, 1), **kwargs):
        """Create and start the progress of the scope."""
        self.start_progress(chunk, position)

    def compute_task(self, chunk=None, dry_run=False, computer=None, featured=None, **kwargs):
        """
        Run the action on one scope, converting every outcome into a result dict (``self.result``).

        Parameters
        ----------
        chunk : str
            Scope to compute.
        dry_run : bool
            Compute without writing.
        computer, featured : callable, optional
            Factories passed to the unit of work (for tests).

        Notes
        -----
        The result is ``{'status': 'computed', 'action', ...figures}`` or ``{'status': 'failed', 'action',
        'reason'}``, with the ``warnings`` raised.
        """
        name = ACTIONS[self.ACTION]
        self.result = self.collect(chunk, lambda: {'status': 'computed', 'action': name,
                                                   **self.unit(chunk, dry_run, computer, featured)}, action=name)

    def unit(self, chunk, dry_run, computer, featured) -> dict:
        """Run `compute.compute_scope` on the scope and return its figures."""
        return compute.compute_scope(chunk, save=not dry_run, verbose=self.model_verbose, progress=self.progress,
                                     computer=computer, featured=featured)

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line of one result of any action, with the count of its model warnings."""
        if result['status'] != 'computed':
            line = '{}: {} {} ({})'.format(chunk, result['action'], result['status'], result['reason'])
        elif result['action'] == ACTIONS[CATALOGUES] and 'scopes' in result:
            line = '{}: saved ({} scopes, {} districts, {} quality evaluations, {} pollsters) in {:.0f} s'.format(
                chunk, result['scopes'], result['districts'], result['quality'], result['pollsters'],
                result['seconds']['total'])
        elif result['action'] == ACTIONS[CATALOGUES]:
            line = '{}: checked ({} quality evaluations, dry run) in {:.0f} s'.format(
                chunk, result['quality'], result['seconds']['total'])
        elif result['action'] == 'ratings':
            line = '{}: ratings computed ({} pollsters, {} events) in {:.0f} s'.format(
                chunk, result['n_pollsters'], result['n_events'], result['seconds']['total'])
        else:
            featured = 'dry run' if result['featured'] is None else '{} featured'.format(result['featured'])
            line = '{}: series computed ({} polls, {} events, {}) in {:.0f} s'.format(
                chunk, result['n_polls'], result['n_events'], featured, result['seconds']['total'])
        return line + notes_suffix(result)
