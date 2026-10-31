from ..core.worker import SequenceJob
from ..lib.load import YEARS_WITH_OVERWRITE
from ..lib.publish.common import resolve_scopes


class Load(SequenceJob):
    """
    Load the Wikipedia polls of an election into the database (the sequence of
    ``notebooks/data-load/PollsLoadWikipedia.ipynb``), scope by scope: read the article, check that every name is
    mapped, build and select the series, save them and compute their weights (`load.load_polls`).

    The load stops, without saving anything, when a party, pollster or sponsor of the article has no mapping; the
    result line names them. With ``force`` the unmapped pollsters and sponsors are added to the catalogues (with a
    warning to review their type) and the article is read again; parties are never added. By default only the new
    polls are saved; with ``overwrite`` the whole cycle is replaced. With ``dry_run`` the article is read and
    checked but nothing is written (the line says what would be added). ``years`` cannot be combined with
    ``overwrite``, which would delete the other years of the cycle. The event is the ``event_date`` given or, by
    default, the next election of each scope. With ``scopes`` ``'all'``, a scope without an upcoming event or whose
    event has no polls article is skipped; named explicitly, it fails. Scopes are isolated: one that fails never
    stops the others (unless ``stop_on_error``), and the job exits with an error at the end if any failed.

    The action ``params`` rewrites, for every election of each scope, the ``parties`` (``event``, ``polls``) and
    the ``bmaps`` (``main``, ``blocks``, ``vs``) of ``data/params.json`` from the polls, the results and the block
    of each party (`load.load_params`); the other keys of each election (``smap``, ``threshold``, ``max``,
    ``min``...) are kept as they are and an election without polls nor results is left untouched. With ``dry_run``
    nothing is written and every key that would change is reported at level 1. ``event_date``, ``overwrite``,
    ``force`` and ``years`` do not apply to it.

    The console output has the four levels of ``publish``: 0, the header and the final summary; 1, also
    ``[i/n] scope: empezando``, the names block, the warnings and the result line of each scope; 2, also one line
    per phase with its seconds and the skipped names; 3, also the own output of the loader and the ``Computer``
    (which receive ``verbose - 2``).

    Examples
    --------
    From the command line::

        python job.py load '{"scopes":["es"],"overwrite":true}'
        python job.py load '{"scopes":["es"],"force":true}'
        python job.py load '{"scopes":["es"],"event_date":"2023-07-23","overwrite":true}'
        python job.py load '{"scopes":"all","dry_run":true,"verbose":2}'
        python job.py load '{"what":"params","scopes":"all"}'
        python job.py load '{"what":"params","scopes":"all","dry_run":true}'

    In Docker (JSON without spaces)::

        RUN_JOB='load {"scopes":["es"],"overwrite":true}'
    """

    sequence = {'load': ['polls', 'params']}
    default_action = 'update'
    stop_on_error = False
    chunk_noun = 'scopes'
    WHAT = ('polls', 'params')

    def run(self, what=('polls',), scopes=('es',), event_date=None, overwrite=False, force=False, years=None,
            dry_run=False, verbose=1, loader=None, computer=None, catalogues=None, params_loader=None,
            stop_on_error=None, **kwargs):
        """
        Run the requested loads.

        Parameters
        ----------
        what : str or list of str, default ('polls',)
            Actions to run, in order: ``polls`` (the Wikipedia polls of the next election) and ``params`` (the
            ``parties`` and ``bmaps`` of every election of the scope in ``data/params.json``, `load.load_params`).
        scopes : str or list of str, default ('es',)
            Scopes to load, or ``'all'`` (see ``publish.resolve_scopes``).
        event_date : str, optional
            Election day (``YYYY-MM-DD``) to load; the next election of each scope by default. Only ``polls``.
        overwrite : bool, default False
            Replace the whole cycle instead of saving only the new polls. Only ``polls``.
        force : bool, default False
            Add the unmapped pollsters and sponsors to the catalogues and go on. Only ``polls``.
        years : list of str, optional
            Years of the article to read (as ``'2026'``; numbers are converted), all by default. Not allowed
            together with ``overwrite``. Only ``polls``.
        dry_run : bool, default False
            Read and check without writing anything; ``params`` reports every key that would change.
        verbose : int, default 1
            Verbosity of the console output: 0 to 3, see the class docstring.
        loader, computer, catalogues : callable, optional
            Factories and helper passed to `load.load_polls`; the real ones by default (for tests).
        params_loader : callable, optional
            Replacement of `load.load_params` (for tests).
        stop_on_error : bool, optional
            Stop at the first failed scope instead of going on with the others; ``False`` by default.
        **kwargs
            Ignored extra job arguments.

        Returns
        -------
        dict
            With one action, ``{scope: result}``: the figures of `load.load_polls` or `load.load_params`
            (``status`` ``loaded`` or ``checked``), or ``status`` ``failed`` (``skipped`` with ``'all'``, see the
            class docstring) with the ``reason``; always the ``warnings`` raised. With several actions,
            ``{scope: {action: result}}``.

        Raises
        ------
        ValueError
            If ``what`` holds an unknown action, ``years`` comes with ``overwrite``, there are no scopes to run
            (``load: no scopes to run``) or ``stop_on_error`` is not true or false (``stop_on_error must be true or
            false``).
        RuntimeError
            If any scope failed (raised after the summary and the alert).
        """
        whats = [what] if isinstance(what, str) else list(what)
        for item in whats:
            if item not in self.WHAT:
                raise ValueError('load: unknown what "{}" (expected one of {})'.format(item, ', '.join(self.WHAT)))
        if years and overwrite:
            raise ValueError(YEARS_WITH_OVERWRITE)
        if years is not None:
            years = [str(year) for year in years]

        targets = resolve_scopes(scopes)
        if not targets:
            raise ValueError('load: no scopes to run')

        super().run(select=['load.' + item for item in whats], action='rebulk' if overwrite else 'update',
                    chunks=targets, stop_on_error=stop_on_error, event_date=event_date, force=force, years=years,
                    dry_run=dry_run, verbose=verbose, loader=loader, computer=computer, catalogues=catalogues,
                    params_loader=params_loader, skip_missing=scopes == 'all')

        results = {}
        for (key, scope), result in self.results.items():
            results.setdefault(scope, {})[key.split('.', 1)[1]] = result

        if len(whats) == 1:
            return {scope: by_action[whats[0]] for scope, by_action in results.items()}
        return results
