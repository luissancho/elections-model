from ..core.worker import Job
from ..lib.load import NO_POLLS_ARTICLE, YEARS_WITH_OVERWRITE, load_polls
from ..lib.publish.common import next_event_date, resolve_scopes
from .progress import ScopeReporter, collect_warnings, counted, notes_suffix


# Labels of the phases reported at verbose 2 (keys of `load.LOAD_STEPS`)
STEPS = {'read': 'lectura', 'names': 'nombres', 'series': 'series', 'save': 'guardado', 'weights': 'pesos'}


class Load(ScopeReporter, Job):
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
    stops the others, and the job exits with an error at the end if any failed.

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

    In Docker (JSON without spaces)::

        RUN_JOB='load {"scopes":["es"],"overwrite":true}'
    """

    WHAT = ('polls',)
    verbose = 1

    def run(self, what=('polls',), scopes=('es',), event_date=None, overwrite=False, force=False, years=None,
            dry_run=False, verbose=1, loader=None, computer=None, catalogues=None, **kwargs):
        """
        Run the requested loads.

        Parameters
        ----------
        what : str or list of str, default ('polls',)
            Actions to run; only ``polls`` for now.
        scopes : str or list of str, default ('es',)
            Scopes to load, or ``'all'`` (see ``publish.resolve_scopes``).
        event_date : str, optional
            Election day (``YYYY-MM-DD``) to load; the next election of each scope by default.
        overwrite : bool, default False
            Replace the whole cycle instead of saving only the new polls.
        force : bool, default False
            Add the unmapped pollsters and sponsors to the catalogues and go on.
        years : list of str, optional
            Years of the article to read (as ``'2026'``; numbers are converted), all by default. Not allowed
            together with ``overwrite``.
        dry_run : bool, default False
            Read and check without writing anything.
        verbose : int, default 1
            Verbosity of the console output: 0 to 3, see the class docstring.
        loader, computer, catalogues : callable, optional
            Factories and helper passed to `load.load_polls`; the real ones by default (for tests).
        **kwargs
            Ignored extra job arguments.

        Returns
        -------
        dict[str, dict]
            Result of each scope: the figures of `load.load_polls` (``status`` ``loaded`` or ``checked``), or
            ``status`` ``failed`` (``skipped`` with ``'all'``, see the class docstring) with the ``reason``; always
            the ``warnings`` raised.

        Raises
        ------
        ValueError
            If ``what`` holds an unknown action, or ``years`` comes with ``overwrite``.
        RuntimeError
            If any scope failed (raised after the summary and the alert).
        """
        self.verbose = verbose
        whats = [what] if isinstance(what, str) else list(what)
        for item in whats:
            if item not in self.WHAT:
                raise ValueError('load: unknown what "{}" (expected one of {})'.format(item, ', '.join(self.WHAT)))
        if years and overwrite:
            raise ValueError(YEARS_WITH_OVERWRITE)
        if years is not None:
            years = [str(year) for year in years]

        targets = resolve_scopes(scopes)
        flags = ''.join(text for active, text in ((overwrite, ' · overwrite'), (force, ' · force'),
                                                   (dry_run, ' · dry run')) if active)
        self._say('load polls: {}{}'.format(counted(len(targets), 'ámbito', 'ámbitos'), flags))
        results = {}
        failures = []
        lines = []

        for index, scope in enumerate(targets, 1):
            progress = self._progress(scope, index, len(targets), steps=STEPS)
            progress.start()
            result = self._scope(scope, event_date, overwrite, force, years, dry_run, verbose, progress, loader,
                                 computer, catalogues, skip_missing=scopes == 'all')
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
            raise RuntimeError('load: failed scopes: ' + ', '.join(failures))

        return results

    def _scope(self, scope, event_date, overwrite, force, years, dry_run, verbose, progress, loader, computer,
               catalogues, skip_missing=False):
        """
        Load the polls of one scope, converting every outcome into a result dict.

        Parameters
        ----------
        scope : str
            Scope to load.
        event_date : str or None
            Election day, or ``None`` for the next election of the scope.
        overwrite, force, years, dry_run
            Options of the job (see `run`).
        verbose : int
            Verbosity of the job; the loader and the ``Computer`` receive ``verbose - 2``.
        progress : ScopeProgress
            Live progress of the scope (its phases).
        loader, computer, catalogues : callable or None
            Factories and helper passed to `load.load_polls`.
        skip_missing : bool, default False
            Turn a scope without an upcoming event or without a polls article into ``skipped`` instead of
            ``failed`` (the job was asked for ``'all'``).

        Returns
        -------
        dict
            The result of `load.load_polls` or ``{'status': 'failed' | 'skipped', 'action': 'polls', 'reason'}``,
            with the ``warnings`` raised.
        """
        notes = []
        try:
            event_date = event_date or next_event_date(scope)
            if event_date is None:
                result = {'status': 'failed', 'action': 'polls', 'reason': 'no upcoming event'}
            else:
                with collect_warnings(notes, progress.note):
                    result = load_polls(scope, event_date, overwrite=overwrite, force=force, years=years,
                                        save=not dry_run, verbose=max(0, verbose - 2), progress=progress,
                                        report=lambda text, level=1: self._say(text, level), loader=loader,
                                        computer=computer, catalogues=catalogues)
        except Exception as e:
            self._log_error(scope)
            result = {'status': 'failed', 'action': 'polls', 'reason': '{}: {}'.format(type(e).__name__, e)}

        if skip_missing and result['status'] == 'failed' and result['reason'] in (
                'no upcoming event', NO_POLLS_ARTICLE.format(event_date)):
            result['status'] = 'skipped'
        result['warnings'] = notes
        return result

    @staticmethod
    def _line(scope, result):
        """Format the summary line of one scope result, with the count of its warnings."""
        if result['status'] in ('failed', 'skipped'):
            return '{}: polls {} ({})'.format(scope, result['status'], result['reason']) + notes_suffix(result)

        added = result['added']
        pollsters, sponsors = len(added['pollsters']), len(added['sponsors'])
        seconds = result['seconds']['total']
        if result['status'] == 'loaded':
            line = '{}: polls loaded ({} read, {} saved, {}{}) in {:.0f} s'.format(
                scope, result['n_read'], result['n_saved'], result['event_date'],
                ', overwrite' if result['overwrite'] else '', seconds)
            verb = 'added'
        else:
            line = '{}: polls checked ({} read, {} new, {}{}, dry run) in {:.0f} s'.format(
                scope, result['n_read'], result['n_saved'], result['event_date'],
                ', overwrite' if result['overwrite'] else '', seconds)
            verb = 'to add'
        nouns = [counted(n, one, many) + ' ' + verb for n, one, many in
                 ((pollsters, 'pollster', 'pollsters'), (sponsors, 'sponsor', 'sponsors')) if n]
        if nouns:
            line += ' · ' + ', '.join(nouns)
        return line + notes_suffix(result)
