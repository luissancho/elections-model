from ..core.worker import Job
from ..lib.compute import compute_scope, rate_scope
from ..lib.publish import resolve_scopes
from .progress import ScopeReporter, collect_warnings, counted, notes_suffix


# Labels of the phases reported at verbose 2 (keys of `compute.COMPUTE_STEPS` and `compute.RATINGS_STEP`)
STEPS = {'weights': 'pesos', 'featured': 'eventos destacados', 'errors': 'errores', 'deviations': 'desviaciones',
         'drift': 'deriva', 'house_effects': 'efectos de casa', 'herding': 'rebaño', 'ratings': 'ratings'}
# Public name of each action, in the headers and the summary lines
ACTIONS = {'compute': 'series', 'ratings': 'ratings'}


class Compute(ScopeReporter, Job):
    """
    Compute the poll series and the pollster ratings of each scope into the database (the sequence of
    ``notebooks/data-load/PollsCompute.ipynb``), the counterpart of ``publish`` for the model tables.

    ``compute`` runs, scope by scope, the weights, the featured events, the errors, the deviations, the drift, the
    house effects and the herding (`compute.compute_scope`). ``ratings`` runs, scope by scope, the cross-scope
    pollster ratings of its events (`compute.rate_scope`, with the polls of every scope). When both are requested,
    the series of every scope go first and the ratings of every scope after, because each rating reads the errors
    of all the scopes. Scopes are isolated: one that fails never stops the others, and the job exits with an error
    at the end if any failed. With ``dry_run`` nothing is written and the series stop after the deviations (drift,
    house effects and herding read the stored errors).

    The console output has the four levels of ``publish``: 0, the header of each action and the final summary; 1,
    also ``[i/n] scope: empezando`` and the result line of each scope; 2, also one line per phase with its seconds;
    3, also the own output of the ``Computer`` (which receives ``verbose - 2``).

    Examples
    --------
    From the command line::

        python job.py compute '{"what":["compute","ratings"],"scopes":"all"}'
        python job.py compute '{"what":["ratings"],"scopes":["es"]}'
        python job.py compute '{"what":["compute"],"scopes":["es-md"],"dry_run":true,"verbose":2}'

    In Docker (JSON without spaces)::

        RUN_JOB='compute {"what":["compute","ratings"],"scopes":"all"}'
    """

    WHAT = ('compute', 'ratings')
    verbose = 1

    def run(self, what=('compute', 'ratings'), scopes=('es',), dry_run=False, verbose=1, computer=None,
            featured=None, **kwargs):
        """
        Run the requested computations.

        Parameters
        ----------
        what : str or list of str, default ('compute', 'ratings')
            Actions to run, among ``compute`` and ``ratings``; the series of every scope run before the ratings.
        scopes : str or list of str, default ('es',)
            Scopes to compute, or ``'all'`` (``es`` and every regional scope, see ``publish.resolve_scopes``).
        dry_run : bool, default False
            Compute without writing anything into the database.
        verbose : int, default 1
            Verbosity of the console output: 0, only the header of each action and the final summary; 1, also
            ``[i/n] scope: empezando`` and the result line of each scope; 2, also the phase lines with their
            seconds; 3, also the own output of the model (the ``Computer`` receives ``verbose - 2``).
        computer : callable, optional
            Factory with the signature ``Computer(scope=..., verbose=...)``; the real ``Computer`` by default
            (for tests).
        featured : callable, optional
            ``update_featured(scope) -> int``; the real one by default (for tests).
        **kwargs
            Ignored extra job arguments.

        Returns
        -------
        dict[str, dict]
            Result of each scope (of the last action that ran on it): ``status`` (``computed`` or ``failed``),
            ``action`` (``series`` or ``ratings``) and, when computed, the figures of `compute.compute_scope` or
            `compute.rate_scope`; when failed, the ``reason``.

        Raises
        ------
        ValueError
            If ``what`` holds an unknown action.
        RuntimeError
            If any scope failed (raised after the summary and the alert).
        """
        self.verbose = verbose
        model_verbose = max(0, verbose - 2)
        whats = [what] if isinstance(what, str) else list(what)
        for item in whats:
            if item not in self.WHAT:
                raise ValueError('compute: unknown what "{}" (expected one of {})'.format(item, ', '.join(self.WHAT)))

        targets = resolve_scopes(scopes)
        n_scopes = counted(len(targets), 'ámbito', 'ámbitos')
        suffix = ' · dry run' if dry_run else ''
        results = {}
        failures = []
        lines = []

        for action in self.WHAT:
            if action not in whats:
                continue
            self._say('compute {}: {}{}'.format(ACTIONS[action], n_scopes, suffix))
            for index, scope in enumerate(targets, 1):
                progress = self._progress(scope, index, len(targets), steps=STEPS)
                progress.start()
                result = self._scope(action, scope, dry_run, model_verbose, progress, computer, featured)
                results[scope] = result
                if result['status'] == 'failed':
                    failures.append('{} ({})'.format(scope, result['action']))
                lines.append(self._line(scope, result))
                progress.done(lines[-1])

        for line in lines:
            print(line)
        if self.app.logger is not None:
            self.alert('\n'.join(lines))

        if failures:
            raise RuntimeError('compute: failed scopes: ' + ', '.join(failures))

        return results

    def _scope(self, action, scope, dry_run, verbose, progress, computer, featured):
        """
        Run one action on one scope, converting every outcome into a result dict.

        Parameters
        ----------
        action : str
            ``compute`` or ``ratings``.
        scope : str
            Scope to compute.
        dry_run : bool
            Compute without writing.
        verbose : int
            Verbosity of the ``Computer``.
        progress : ScopeProgress
            Live progress of the scope (its phases).
        computer, featured : callable or None
            Factories passed to `compute.compute_scope` and `compute.rate_scope`.

        Returns
        -------
        dict
            ``{'status': 'computed', 'action', ...figures}`` or ``{'status': 'failed', 'action', 'reason'}``.
        """
        name = ACTIONS[action]
        notes = []
        try:
            with collect_warnings(notes, progress.note):
                if action == 'compute':
                    figures = compute_scope(scope, save=not dry_run, verbose=verbose, progress=progress,
                                            computer=computer, featured=featured)
                else:
                    figures = rate_scope(scope, save=not dry_run, verbose=verbose, progress=progress,
                                         computer=computer)
        except Exception as e:
            self._log_error(scope)
            return {'status': 'failed', 'action': name, 'reason': '{}: {}'.format(type(e).__name__, e),
                    'warnings': notes}

        return {'status': 'computed', 'action': name, **figures, 'warnings': notes}

    @staticmethod
    def _line(scope, result):
        """Format the summary line of one scope result, with the count of its model warnings."""
        if result['status'] != 'computed':
            line = '{}: {} {} ({})'.format(scope, result['action'], result['status'], result['reason'])
        elif result['action'] == 'ratings':
            line = '{}: ratings computed ({} pollsters, {} events) in {:.0f} s'.format(
                scope, result['n_pollsters'], result['n_events'], result['seconds']['total'])
        else:
            featured = 'dry run' if result['featured'] is None else '{} featured'.format(result['featured'])
            line = '{}: series computed ({} polls, {} events, {}) in {:.0f} s'.format(
                scope, result['n_polls'], result['n_events'], featured, result['seconds']['total'])
        return line + notes_suffix(result)
