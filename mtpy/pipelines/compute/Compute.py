from ...jobs.progress import counted, notes_suffix
from ...lib import compute
from ..base import ScopePipeline

# Labels of the phases reported at verbose 2 (keys of `compute.COMPUTE_STEPS` and `compute.RATINGS_STEP`)
STEPS = {'weights': 'pesos', 'featured': 'eventos destacados', 'errors': 'errores', 'deviations': 'desviaciones',
         'drift': 'deriva', 'house_effects': 'efectos de casa', 'herding': 'rebaño',
         'average_errors': 'errores del promedio', 'ratings': 'ratings'}
# Public name of each action, in the headers and the summary lines
ACTIONS = {'catalogues': 'catálogos', 'compute': 'series', 'ratings': 'ratings'}
# Key of the result of the global `catalogues` action (the others are keyed by scope)
CATALOGUES = 'catalogues'
# Reason of the skipped result of a scope with nothing to compute
NOTHING_PENDING = 'nothing loaded since the last compute'


class Compute(ScopePipeline):
    """
    Compute the poll series of a scope into the database: weights, featured events, errors, deviations, drift,
    house effects, herding and the errors of the poll average (`compute.compute_scope`).

    The chunk is the scope; every outcome is a result dict (see `compute_task`) and a failure never raises. The
    subclass `Ratings` changes the action and the unit of work; this class also formats the lines of all the
    actions of the group (`line_of`).

    The `compute` action skips a scope with nothing loaded since its last compute (`compute.pending_events`)
    with a ``skipped`` result, unless ``overwrite`` is set; ``pending`` replaces that query (for tests) with a
    callable ``scope -> list of dates``. `Ratings` never skips.
    """

    label = 'series'
    steps = STEPS
    ACTION = 'compute'
    pipes = {'update': {'Setup': ['prepare'], 'Load': ['compute']}}

    def start(self, dry_run=False, overwrite=False, **kwargs) -> None:
        """Set the run and print the header ``compute <action>: n ámbitos[ · overwrite][ · dry run]``."""
        super().start(**kwargs)
        flags = (' · overwrite' if overwrite and self.ACTION == 'compute' else '') + (' · dry run' if dry_run else '')
        self._say('compute {}: {}{}'.format(ACTIONS[self.ACTION], counted(self.total, 'ámbito', 'ámbitos'), flags))

    def prepare_task(self, chunk=None, position=(1, 1), overwrite=False, pending=None, **kwargs):
        """
        Create and start the progress of the scope and end the run as skipped when there is nothing to compute.

        Parameters
        ----------
        overwrite : bool
            Compute even with nothing pending.
        pending : callable, optional
            Replacement of `compute.pending_events` (for tests).
        """
        self.start_progress(chunk, position)
        if self.ACTION != 'compute' or overwrite:
            return

        try:
            has_pending = bool((pending or compute.pending_events)(chunk))
        except Exception as e:
            self.end(self.failed(chunk, e, action=ACTIONS[self.ACTION], warnings=[]))

        if not has_pending:
            self.end({'status': 'skipped', 'action': ACTIONS[self.ACTION], 'reason': NOTHING_PENDING,
                      'warnings': []})

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
