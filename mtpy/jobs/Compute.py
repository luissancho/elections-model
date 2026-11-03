from ..core.worker import SequenceJob
from ..lib.compute import AUTO, check_mode
from ..lib.publish import resolve_scopes


class Compute(SequenceJob):
    """
    Compute the poll series and the pollster ratings of each scope into the database (the sequence of
    ``notebooks/data-load/PollsCompute.ipynb``), the counterpart of ``publish`` for the model tables.

    ``compute`` runs, scope by scope, the series of the model (`compute.compute_scope`) in one of two modes: ``full``
    (the weights, the featured events, the errors, the deviations, the drift, the house effects and the herding) or
    ``light`` (the weights and the herding, what the polls of the current cycle can change: the overlap weight of a
    poll depends on its neighbours of the cycle, and the Forecaster drops a poll never weighed). With ``mode``
    ``auto``, the default, the pipeline picks: ``full`` with ``overwrite`` or when a pending election of the scope
    has an official result (its errors, featured flag, drift and house effects have to be computed), ``light`` when
    the pending elections are only cycles without a result (the daily routine). ``"mode": "light"`` or ``"full"``
    forces the phases, but never avoids the skip of a scope with nothing pending (below), and a ``light`` with a
    pending election that has a result ends ``failed`` (it would stamp the scope as computed without the phases of
    that result; use ``full`` or ``auto``). ``ratings`` runs, scope by
    scope, the cross-scope pollster ratings of its events (`compute.rate_scope`, with the polls of every scope).
    ``average_errors`` runs, scope by scope, the errors of the poll average in the past elections
    (`compute.average_errors_scope`, one fit of the average per election and horizon: about 37 minutes in ``es``),
    the table that only the uncertainty options of method 15 read (``error_model='average'``, ``drift_scale='auto'``,
    not the defaults): run it after a ``full`` compute of the scope, whose house effects its fits take as prior.
    ``catalogues`` replicates the versioned catalogues (scopes, districts and the prior quality of the pollsters,
    `compute.replicate_catalogues`) once, before anything else: run it after editing the rubric or the evaluations
    of `data/`, so the ratings take the new quality. The actions always run in this order, whatever the order of
    ``what``: the catalogues once, the series of every scope, the ratings of every scope (each rating reads the
    errors of all the scopes) and the errors of the average of every scope. Scopes are isolated: one that fails
    never stops the others (unless ``stop_on_error``), and the job exits with an error at the end if any failed.
    With ``dry_run`` nothing is written: ``catalogues`` only validates the files, the ``full`` series stops after
    the deviations and the ``light`` one after the weights (drift, house effects and herding read what the previous
    phases stored), and the ratings and the errors of the average compute without saving.

    With ``overwrite`` false, the default, ``compute`` only runs on the scopes with something loaded since their last
    compute, read from ``loaded_at`` and ``computed_at`` of ``events`` (written by the loaders and by the compute
    itself, in both modes); the other scopes end ``skipped`` with that reason, whatever the ``mode``. ``catalogues``,
    ``ratings`` and ``average_errors`` are not affected. After a change of code or parameters pass
    ``"overwrite": true`` (``full`` for every scope). Right after the ``ALTER TABLE`` that adds the marks every
    election counts as pending, with results included, so the first ``compute`` computes everything once in ``full``.

    The console output has the four levels of ``publish``: 0, the header of each action and the final summary; 1,
    also ``[i/n] scope: empezando`` and the result line of each scope (with the mode of the series and its reason);
    2, also one line per phase with its seconds; 3, also the own output of the ``Computer`` (which receives
    ``verbose - 2``).

    Examples
    --------
    From the command line::

        python job.py compute '{"what":["compute","ratings"],"scopes":"all"}'
        python job.py compute '{"what":["catalogues","ratings"],"scopes":"all"}'
        python job.py compute '{"what":["ratings"],"scopes":["es"]}'
        python job.py compute '{"what":["compute"],"scopes":"all","overwrite":true}'
        python job.py compute '{"what":["compute"],"scopes":["es"],"mode":"full"}'
        python job.py compute '{"what":["average_errors"],"scopes":"all"}'
        python job.py compute '{"what":["compute"],"scopes":["es-md"],"dry_run":true,"verbose":2}'

    In Docker (JSON without spaces)::

        RUN_JOB='compute {"what":["compute","ratings"],"scopes":"all"}'
    """

    sequence = {'compute': ['catalogues', 'compute', 'ratings', 'average_errors']}
    default_action = 'update'
    stop_on_error = False
    chunk_noun = 'scopes'
    WHAT = ('catalogues', 'compute', 'ratings', 'average_errors')

    def run(self, what=('compute', 'ratings'), scopes=('es',), overwrite=False, mode=AUTO, dry_run=False, verbose=1,
            computer=None, featured=None, catalogues=None, pending=None, results=None, stop_on_error=None, **kwargs):
        """
        Run the requested computations.

        Parameters
        ----------
        what : str or list of str, default ('compute', 'ratings')
            Actions to run, among ``catalogues``, ``compute``, ``ratings`` and ``average_errors``, always in this
            order: the catalogues once, then the series of every scope, the ratings of every scope and the errors of
            the average of every scope.
        scopes : str or list of str, default ('es',)
            Scopes to compute, or ``'all'`` (``es`` and every regional scope, see ``publish.resolve_scopes``).
        overwrite : bool, default False
            Compute the series of every scope even if nothing was loaded since its last compute (after a change of
            code or parameters), in ``full`` mode with ``mode`` ``auto``; without it, such a scope ends ``skipped``.
        mode : {'auto', 'light', 'full'}, default 'auto'
            Phases of the series: ``auto`` lets the pipeline pick from what is pending (see the class), ``light`` or
            ``full`` forces them. It never avoids the skip of a scope with nothing pending, and ``light`` is refused
            (``failed``) on a scope with a pending election that has a result.
        dry_run : bool, default False
            Compute without writing anything into the database (``catalogues`` only validates the files).
        verbose : int, default 1
            Verbosity of the console output: 0, only the header of each action and the final summary; 1, also
            ``[i/n] scope: empezando`` and the result line of each scope; 2, also the phase lines with their
            seconds; 3, also the own output of the model (the ``Computer`` receives ``verbose - 2``).
        computer : callable, optional
            Factory with the signature ``Computer(scope=..., verbose=...)``; the real ``Computer`` by default
            (for tests).
        featured : callable, optional
            ``update_featured(scope) -> int``; the real one by default (for tests).
        catalogues : callable, optional
            ``replicate_catalogues(save) -> dict``; the real one by default (for tests).
        pending : callable, optional
            ``pending_events(scope) -> list``; the real one by default (for tests).
        results : callable, optional
            ``result_dates(scope) -> list``; the real one by default (for tests).
        stop_on_error : bool, optional
            Stop at the first failed scope instead of going on with the others; ``False`` by default.
        **kwargs
            Ignored extra job arguments.

        Returns
        -------
        dict[str, dict]
            Result of each scope (of the last action that ran on it): ``status`` (``computed``, ``skipped`` or
            ``failed``), ``action`` (``series``, ``ratings`` or ``errores del promedio``) and, when computed, the
            figures of `compute.compute_scope` (with ``mode`` and ``mode_reason``), `compute.rate_scope` or
            `compute.average_errors_scope`; when failed, the ``reason``. The ``catalogues`` action is keyed
            ``'catalogues'``, with the figures of `compute.replicate_catalogues`.

        Raises
        ------
        ValueError
            If ``what`` holds an unknown action, ``mode`` is not ``auto``, ``light`` or ``full``, there are no scopes
            to run (``compute: no scopes to run``) or ``stop_on_error`` is not true or false (``stop_on_error must be
            true or false``).
        RuntimeError
            If any scope failed (raised after the summary and the alert).
        """
        whats = [what] if isinstance(what, str) else list(what)
        for item in whats:
            if item not in self.WHAT:
                raise ValueError('compute: unknown what "{}" (expected one of {})'.format(item, ', '.join(self.WHAT)))
        check_mode(mode)

        targets = resolve_scopes(scopes)
        if not targets:
            raise ValueError('compute: no scopes to run')

        super().run(select=['compute.' + item for item in self.WHAT if item in whats], chunks=targets,
                    stop_on_error=stop_on_error, dry_run=dry_run, verbose=verbose, computer=computer,
                    featured=featured, catalogues=catalogues, overwrite=overwrite, mode=mode, pending=pending,
                    results=results)

        outcome = {}
        for (key, chunk), result in self.results.items():
            outcome['catalogues' if chunk is None else chunk] = result
        return outcome
