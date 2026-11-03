from ...core.worker import PipelineEnd
from ...jobs.progress import counted, notes_suffix
from ...lib import compute
from ..base import ScopePipeline

# Labels of the phases reported at verbose 2 (keys of `compute.COMPUTE_STEPS` and `compute.RATINGS_STEP`)
STEPS = {'weights': 'pesos', 'featured': 'eventos destacados', 'errors': 'errores', 'deviations': 'desviaciones',
         'drift': 'deriva', 'house_effects': 'efectos de casa', 'herding': 'rebaño',
         'average_errors': 'errores del promedio', 'ratings': 'ratings'}
# Public name of each action, in the headers and the summary lines
ACTIONS = {'catalogues': 'catálogos', 'compute': 'series', 'ratings': 'ratings',
           'average_errors': 'errores del promedio'}
# Key of the result of the global `catalogues` action (the others are keyed by scope)
CATALOGUES = 'catalogues'
# Reason of the skipped result of a scope with nothing to compute
NOTHING_PENDING = 'nothing loaded since the last compute'


class Compute(ScopePipeline):
    """
    Compute the poll series of a scope into the database (`compute.compute_scope`) in the mode the pending work
    asks for: ``light`` (weights and herding, what the polls of the current cycle can change) or ``full`` (weights,
    featured events, errors, deviations, drift, house effects and herding).

    The chunk is the scope; every outcome is a result dict (see `compute_task`) and a failure never raises. The
    subclasses `Ratings` and `AverageErrors` change the action and the unit of work; this class also formats the
    lines of all the actions of the group (`line_of`).

    The `compute` action skips a scope with nothing loaded since its last compute (`compute.pending_events`)
    with a ``skipped`` result, unless ``overwrite`` is set, whatever the ``mode``. Then it picks the mode
    (`compute.resolve_mode`): a requested ``light`` or ``full`` stands, except a ``light`` with a pending election
    that has an official result, refused as ``failed`` (it would stamp the scope as computed without the phases of
    that result); with ``auto`` (the default), ``full`` with ``overwrite`` or when a pending election has a result
    (`compute.result_dates`), ``light`` otherwise. ``pending`` and ``results`` replace those two queries (for tests)
    with callables ``scope -> list of dates``. `Ratings` and `AverageErrors` never skip nor pick a mode.
    """

    label = 'series'
    steps = STEPS
    ACTION = 'compute'
    pipes = {'update': {'Setup': ['prepare'], 'Load': ['compute']}}

    def start(self, dry_run=False, overwrite=False, mode=compute.AUTO, **kwargs) -> None:
        """
        Set the run and print the header ``compute <action>: n ámbitos[ · overwrite][ · light | · full][ · dry run]``
        (the first two flags only for the series; the mode only when requested).
        """
        super().start(**kwargs)
        self.mode = self.mode_reason = None
        series = self.ACTION == 'compute'
        flags = ((' · overwrite' if overwrite and series else '')
                 + (' · ' + mode if series and mode != compute.AUTO else '')
                 + (' · dry run' if dry_run else ''))
        self._say('compute {}: {}{}'.format(ACTIONS[self.ACTION], counted(self.total, 'ámbito', 'ámbitos'), flags))

    def prepare_task(self, chunk=None, position=(1, 1), overwrite=False, mode=compute.AUTO, pending=None,
                     results=None, **kwargs):
        """
        Create and start the progress of the scope, end the run as skipped when there is nothing to compute and
        pick the mode of the series (``self.mode`` and ``self.mode_reason``).

        Parameters
        ----------
        overwrite : bool
            Compute even with nothing pending (and, with ``auto``, in ``full`` mode).
        mode : str
            ``auto``, ``light`` or ``full`` (see `compute.resolve_mode`).
        pending : callable, optional
            Replacement of `compute.pending_events` (for tests).
        results : callable, optional
            Replacement of `compute.result_dates` (for tests); only consulted with pending elections and ``auto``
            without ``overwrite``, or ``light`` requested (to refuse it when a pending election has a result).

        Notes
        -----
        A failure of either query, an unknown mode or a ``light`` requested with a pending election that has a
        result (`compute.resolve_mode`) ends the scope as ``failed`` with its reason.
        """
        self.start_progress(chunk, position)
        if self.ACTION != 'compute':
            return

        try:
            light = mode == 'light'
            dates = None if overwrite and not light else (pending or compute.pending_events)(chunk)
            if not overwrite and len(dates) == 0:
                self.end({'status': 'skipped', 'action': ACTIONS[self.ACTION], 'reason': NOTHING_PENDING,
                          'warnings': []})
            needs_results = bool(dates) and ((mode == compute.AUTO and not overwrite) or light)
            with_results = (results or compute.result_dates)(chunk) if needs_results else None
            self.mode, self.mode_reason = compute.resolve_mode(mode, overwrite, dates, with_results)
        except PipelineEnd:  # the skip is not a failure
            raise
        except Exception as e:
            self.end(self.failed(chunk, e, action=ACTIONS[self.ACTION], warnings=[]))

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
        """Run `compute.compute_scope` on the scope in ``self.mode`` and return its figures, with the reason."""
        return {'dry_run': dry_run, 'mode_reason': self.mode_reason,
                **compute.compute_scope(chunk, save=not dry_run, verbose=self.model_verbose, progress=self.progress,
                                        computer=computer, featured=featured, mode=self.mode)}

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
        elif result['action'] == ACTIONS['average_errors']:
            line = '{}: average errors computed ({} rows, {} events{}) in {:.0f} s'.format(
                chunk, result['n_rows'], result['n_events'], ', dry run' if result['dry_run'] else '',
                result['seconds']['total'])
        else:
            parts = ['{} polls'.format(result['n_polls']), '{} events'.format(result['n_events'])]
            if result['featured'] is not None:
                parts.append('{} featured'.format(result['featured']))
            if result['dry_run']:
                parts.append('dry run')
            line = '{}: series computed ({}, {}: {}) in {:.0f} s'.format(
                chunk, result['mode'], result['mode_reason'], ', '.join(parts), result['seconds']['total'])
        return line + notes_suffix(result)
