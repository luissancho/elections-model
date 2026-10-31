from ..core.worker import SequenceJob
from ..lib.publish import resolve_scopes


class Compute(SequenceJob):
    """
    Compute the poll series and the pollster ratings of each scope into the database (the sequence of
    ``notebooks/data-load/PollsCompute.ipynb``), the counterpart of ``publish`` for the model tables.

    ``compute`` runs, scope by scope, the weights, the featured events, the errors, the deviations, the drift, the
    house effects and the herding (`compute.compute_scope`). ``ratings`` runs, scope by scope, the cross-scope
    pollster ratings of its events (`compute.rate_scope`, with the polls of every scope). When both are requested,
    the series of every scope go first and the ratings of every scope after, because each rating reads the errors
    of all the scopes. ``catalogues`` replicates the versioned catalogues (scopes, districts and the prior quality
    of the pollsters, `compute.replicate_catalogues`) once, before anything else: run it after editing the
    rubric or the evaluations of `data/`, so the ratings take the new quality. Scopes are isolated: one that fails
    never stops the others (unless ``stop_on_error``), and the job exits with an error at the end if any failed. With ``dry_run`` nothing is
    written, ``catalogues`` only validates the files and the series stop after the deviations (drift, house
    effects and herding read the stored errors).

    The console output has the four levels of ``publish``: 0, the header of each action and the final summary; 1,
    also ``[i/n] scope: empezando`` and the result line of each scope; 2, also one line per phase with its seconds;
    3, also the own output of the ``Computer`` (which receives ``verbose - 2``).

    Examples
    --------
    From the command line::

        python job.py compute '{"what":["compute","ratings"],"scopes":"all"}'
        python job.py compute '{"what":["catalogues","ratings"],"scopes":"all"}'
        python job.py compute '{"what":["ratings"],"scopes":["es"]}'
        python job.py compute '{"what":["compute"],"scopes":["es-md"],"dry_run":true,"verbose":2}'

    In Docker (JSON without spaces)::

        RUN_JOB='compute {"what":["compute","ratings"],"scopes":"all"}'
    """

    sequence = {'compute': ['catalogues', 'compute', 'ratings']}
    default_action = 'update'
    stop_on_error = False
    chunk_noun = 'scopes'
    WHAT = ('catalogues', 'compute', 'ratings')

    def run(self, what=('compute', 'ratings'), scopes=('es',), dry_run=False, verbose=1, computer=None,
            featured=None, catalogues=None, stop_on_error=None, **kwargs):
        """
        Run the requested computations.

        Parameters
        ----------
        what : str or list of str, default ('compute', 'ratings')
            Actions to run, among ``catalogues``, ``compute`` and ``ratings``, always in this order: the catalogues
            once, then the series of every scope, then the ratings of every scope.
        scopes : str or list of str, default ('es',)
            Scopes to compute, or ``'all'`` (``es`` and every regional scope, see ``publish.resolve_scopes``).
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
        stop_on_error : bool, optional
            Stop at the first failed scope instead of going on with the others; ``False`` by default.
        **kwargs
            Ignored extra job arguments.

        Returns
        -------
        dict[str, dict]
            Result of each scope (of the last action that ran on it): ``status`` (``computed`` or ``failed``),
            ``action`` (``series`` or ``ratings``) and, when computed, the figures of `compute.compute_scope` or
            `compute.rate_scope`; when failed, the ``reason``. The ``catalogues`` action is keyed ``'catalogues'``,
            with the figures of `compute.replicate_catalogues`.

        Raises
        ------
        ValueError
            If ``what`` holds an unknown action, there are no scopes to run (``compute: no scopes to run``) or
            ``stop_on_error`` is not true or false (``stop_on_error must be true or false``).
        RuntimeError
            If any scope failed (raised after the summary and the alert).
        """
        whats = [what] if isinstance(what, str) else list(what)
        for item in whats:
            if item not in self.WHAT:
                raise ValueError('compute: unknown what "{}" (expected one of {})'.format(item, ', '.join(self.WHAT)))

        targets = resolve_scopes(scopes)
        if not targets:
            raise ValueError('compute: no scopes to run')

        super().run(select=['compute.' + item for item in self.WHAT if item in whats], chunks=targets,
                    stop_on_error=stop_on_error, dry_run=dry_run, verbose=verbose, computer=computer,
                    featured=featured, catalogues=catalogues)

        results = {}
        for (key, chunk), result in self.results.items():
            results['catalogues' if chunk is None else chunk] = result
        return results
