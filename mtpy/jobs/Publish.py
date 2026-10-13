import contextlib
import traceback
import warnings
from datetime import date, datetime, timedelta

from ..core.worker import Job
from ..lib import bundle, publish


STEPS = {'init': 'simulador listo', 'fit': 'promedio ajustado', 'nowcast': 'simulación de hoy',
         'export-nowcast': 'exportación de hoy', 'forecast': 'simulación de la elección', 'export': 'exportación',
         'meta': 'meta y titular'}


def counted(n, one, many):
    """Return ``n`` with the singular or plural noun (``'1 ámbito'``, ``'2 ámbitos'``)."""
    return '{} {}'.format(n, one if n == 1 else many)


class ScopeProgress:
    """
    Live progress lines of one scope (and day, with ``backfill``) of ``publish forecast``.

    Calling the object reports a phase of ``publish.publish_forecast`` (its ``progress``
    argument); ``start``, ``note`` and ``done`` print the start of the scope, a model warning
    and its result line. Every line goes through ``say`` with the prefix ``[i/n] scope[ day]``.

    Parameters
    ----------
    say : callable
        Printer of one progress line (``Publish._say``).
    scope : str
        Scope being published.
    index, total : int
        Position of the scope (from 1) and number of scopes of the run.
    day : str, optional
        Day of a retrospective run.
    """

    def __init__(self, say, scope, index, total, day=None):
        self.say = say
        self.position = '[{}/{}]'.format(index, total)
        self.prefix = '{} {}{}'.format(self.position, scope, '' if day is None else ' ' + day)

    def __call__(self, step, seconds):
        """Report the end of the phase ``step`` (a key of ``STEPS``) after ``seconds``."""
        self.say('{}: {} ({:.1f} s)'.format(self.prefix, STEPS.get(step, step), seconds))

    def start(self):
        """Report that the scope (or day) starts."""
        self.say('{}: empezando'.format(self.prefix))

    def note(self, text):
        """Report a model warning of the scope."""
        self.say('{} · aviso: {}'.format(self.prefix, text))

    def done(self, line):
        """Report the summary line of the result of the scope (``Publish._line``)."""
        self.say('{} {}'.format(self.position, line))


class Publish(Job):
    """
    Publish the model results to the web bundle (``site/v1``) of the file system.

    Runs the forecast of each scope, writes its run folder and moves the pointers
    (``latest``, ``history`` and the manifest). Scopes are isolated: one that fails
    never stops the others, and the job exits with an error at the end if any failed.

    Each scope ends ``published``, ``skipped`` (no upcoming event, or
    ``publish.NothingToPublish``: no polls or an average that cannot be fitted),
    ``refused`` (LOREG guard) or ``failed`` (any other error, ``ValueError`` included,
    such as a bundle validation failure or a non-ISO ``event_date``). ``forecast`` prints its
    progress live (``[i/n] scope: phase (seconds)`` and the result line of each scope, also to the
    logger) and collects the model warnings of each scope into its result (``warnings``) and the
    count of its summary line instead of letting them reach stderr; with ``verbose``, it prints
    them live too.

    ``analysis`` computes the houses analysis of each scope (house effects, herding, poll
    errors and error data of the ``Computer``) and overwrites ``analysis/{scope}/``. It has no
    LOREG guard, never touches the manifest and, with ``dry_run``, writes under
    ``site-dry/v1/analysis/``. Each scope ends ``published``, ``skipped``
    (``publish.NothingToPublish``) or ``failed``.

    ``backtest`` converts the model backtest tables (``backtest/results[/{scope}]/``, written by
    ``backtest/run_backtest.py``) into ``backtest/{scope}/`` and overwrites it. It reads files, not the
    database, has no LOREG guard, never touches the manifest and, with ``dry_run``, writes under
    ``site-dry/v1/backtest/``. Each scope ends ``published``, ``skipped`` (no results for the scope)
    or ``failed``.

    ``event`` publishes past elections of each scope into ``events/{scope}/{date}/`` (poll errors by house, official
    results and the slice of the model backtest of the election), overwriting them. It needs ``event_date`` (a date,
    a list of dates or ``'all'``: the featured elections of the scope before today), cannot be combined with
    ``forecast``, has no LOREG guard, never touches the manifest and, with ``dry_run``, writes under
    ``site-dry/v1/events/``. Each date of each scope ends ``published``, ``skipped`` (not a featured election of
    the scope, no official results or no polls: ``publish.NothingToPublish``) or ``failed``.

    With ``backfill``, ``forecast`` publishes retrospective runs instead: one per day of the
    range and scope, with the polls published up to that day, the run id and ``run_at`` of
    its noon, and ``backfill`` set in its headline. A day whose run already exists is
    ``skipped``, or ``failed`` when that run is incomplete (it has no ``headline.json``; remove
    it with ``unpublish`` and retry). Every backfill rebuilds the history of the scope, also when
    all its days already existed, so a retry repairs a history or manifest left behind. A
    backfill only moves the manifest ``latest`` forward (to its newest run, when that is newer
    than the current one), so it never moves ``latest`` back nor undoes a ``point``.

    Examples
    --------
    From the command line::

        python job.py publish '{"what":["forecast"],"scopes":["es"]}'
        python job.py publish '{"what":["forecast"],"scopes":"all"}'
        python job.py publish '{"what":["analysis"],"scopes":["es"]}'
        python job.py publish '{"what":["backtest"],"scopes":["es"]}'
        python job.py publish '{"what":["event"],"scopes":["es"],"event_date":"2023-07-23"}'
        python job.py publish '{"what":["event"],"scopes":["es"],"event_date":"all"}'
        python job.py publish '{"what":["manifest"],"freeze":{"active":true,"message":"..."}}'
        python job.py publish '{"what":["point"],"scopes":["es"],"run":"20261007-141503"}'
        python job.py publish '{"what":["forecast"],"scopes":["es"],"n_sim":20,"dry_run":true}'
        python job.py publish '{"what":["forecast"],"scopes":["es"],"backfill":{"from":"2026-10-01","to":"2026-10-07"}}'

    In Docker (JSON without spaces)::

        RUN_JOB='publish {"what":["forecast"],"scopes":["es"]}'
    """

    WHAT = ('forecast', 'analysis', 'event', 'backtest', 'manifest', 'point', 'unpublish')
    RUN_EXISTS = 'run exists'
    RUN_INCOMPLETE = 'incomplete run exists; unpublish it first'

    def run(self, what=('forecast',), scopes=('es',), event_date=None, n_sim=1000, seed=42, drange=6, max_fc=10,
            alpha=0.05, correctors=None, freeze=None, run=None, dry_run=False, force=False, fs=None, today=None,
            backfill=None, source=None, verbose=0, **kwargs):
        """
        Run the requested publication actions.

        Parameters
        ----------
        what : str or list of str, default ('forecast',)
            Actions to run, among ``forecast``, ``analysis``, ``event``, ``backtest``, ``manifest``,
            ``point`` and ``unpublish``. They run in that order when several are given.
        scopes : str or list of str, default ('es',)
            Scopes to act on, or ``'all'``.
        event_date : str or list of str, optional
            Election date (ISO 8601, normalised to ``YYYY-MM-DD``). Defaults to the next
            event of each scope. A value that is not a date fails each scope. For ``event``, it is required and
            can also be a list of dates or ``'all'`` (see ``publish.event_dates_to_publish``).
        n_sim : int, default 1000
            Number of simulations.
        seed : int, default 42
            Random seed.
        drange : int, default 6
            Polls window parameter passed to the forecast.
        max_fc : int, default 10
            Maximum number of forecast steps.
        alpha : float, default 0.05
            Significance level of the intervals.
        correctors : dict, optional
            Correctors passed to the forecast.
        freeze : dict, optional
            Freeze notice of the manifest (``{'active': bool, 'message': str}``).
        run : str, optional
            Run id to point to or remove. Required by ``point`` and ``unpublish``.
        dry_run : bool, default False
            Write under ``site-dry/v1`` (runs, analysis and backtest) and leave the pointers and manifest
            alone.
        force : bool, default False
            Publish inside the LOREG window anyway.
        fs : FileSystem, optional
            File system to write to. Defaults to ``self.app.fs``.
        today : str, optional
            Date (``YYYY-MM-DD``) taken as today by the LOREG guard and the backfill range
            check; for tests and rehearsals only (unlike ``force``, it leaves no mark in
            ``meta.freeze``).
        backfill : dict, optional
            Range ``{'from': 'YYYY-MM-DD', 'to': 'YYYY-MM-DD'}`` (``to`` defaults to ``from``
            and must be before today) of retrospective ``forecast`` runs, one per day; see
            ``publish.backfill_days``. Requires ``forecast`` in ``what``.
        source : str, optional
            Results folder of ``backtest`` and of the model slice of ``event`` (default ``backtest/results`` of
            the repository).
        verbose : int, default 0
            Verbosity passed to the forecast and the analysis.
        **kwargs
            Ignored extra job arguments.

        Returns
        -------
        dict[str, dict or list of dict]
            Result of each scope (of the last action that ran on it), with a ``status`` of
            ``published``, ``skipped``, ``refused`` or ``failed``; with ``backfill``, the
            ``forecast`` result of a scope is the list of the results of its days, each with
            its ``day``, and the ``event`` result is the list of the results of its dates, each with its
            ``event_date``.

        Raises
        ------
        ValueError
            If ``what`` is unknown, ``run`` is missing (or ``event_date`` for ``event``), ``event`` is
            combined with ``forecast``, or ``backfill`` is given without ``forecast`` in ``what`` or its range is invalid.
        RuntimeError
            If no file system is configured, or any scope failed (raised after the
            manifest, the summary and the alert).
        """
        whats = [what] if isinstance(what, str) else list(what)
        for item in whats:
            if item not in self.WHAT:
                raise ValueError('publish: unknown what "{}" (expected one of {})'.format(item, ', '.join(self.WHAT)))
        if ('point' in whats or 'unpublish' in whats) and not run:
            raise ValueError('publish: "run" is required')
        if 'event' in whats and not event_date:
            raise ValueError('publish: "event_date" is required for event')
        if 'event' in whats and 'forecast' in whats:
            raise ValueError('publish: "event" cannot be combined with "forecast"')
        if backfill is not None and 'forecast' not in whats:
            raise ValueError('publish: "backfill" needs "forecast" in what')
        days = None if backfill is None else publish.backfill_days(backfill, today)

        fs = fs if fs is not None else self.app.fs
        if fs is None:
            raise RuntimeError(
                'publish: no file system configured (set S3_BUCKET or run with a local files/ root)'
            )
        writer = bundle.BundleWriter(fs, prefix=bundle.DRY_PREFIX if dry_run else bundle.PREFIX)
        run_id = bundle.run_id()

        results = {}
        failures = []
        lines = []

        national = None
        if 'forecast' in whats or 'analysis' in whats:
            # the day before today: the national window also covers the election day itself
            eve = (publish.common._today(today) - timedelta(days=1)).isoformat()
            national = publish.next_event_date('es', date_from=eve)

        if 'forecast' in whats:
            entries = {}
            if days is None:
                plan = [(None, run_id, {})]
            else:
                plan = [(day, publish.backfill_run_id(day),
                         {'limit_date': day, 'run_at': publish.backfill_run_at(day)}) for day in days]
            params = dict(n_sim=n_sim, seed=seed, drange=drange, max_fc=max_fc, alpha=alpha,
                          correctors=correctors, verbose=verbose)
            targets = publish.resolve_scopes(scopes)
            n_scopes = counted(len(targets), 'ámbito', 'ámbitos')
            if days is None:
                self._say('publish forecast: {} · run {}'.format(n_scopes, run_id))
            else:
                self._say('publish forecast: {} · backfill {}..{} ({})'.format(
                    n_scopes, days[0], days[-1], counted(len(days), 'día', 'días')))
            for index, scope in enumerate(targets, 1):
                outcomes, entry = self._forecast_runs(scope, writer, plan, event_date, today, force, dry_run,
                                                      national=national, index=index, total=len(targets), **params)
                results[scope] = outcomes if days is not None else outcomes[0]
                if any(result['status'] == 'failed' for result in outcomes):
                    failures.append('{} ({})'.format(scope, 'forecast'))
                if entry is not None:
                    entries[scope] = entry
                lines.extend(self._line(scope, result) for result in outcomes)
            if dry_run:
                lines.append('manifest: not written (dry run)')
            elif entries:
                lines.append(self._write_manifest(writer, failures, scopes=entries))

        if 'analysis' in whats:
            for scope in publish.resolve_scopes(scopes):
                result = self._analysis_scope(scope, writer, verbose, today, force, national)
                results[scope] = result
                if result['status'] == 'failed':
                    failures.append('{} ({})'.format(scope, 'analysis'))
                lines.append(self._analysis_line(scope, result))

        if 'event' in whats:
            for scope in publish.resolve_scopes(scopes):
                outcomes = self._event_scope(scope, writer, event_date, today, source, verbose)
                results[scope] = outcomes
                if any(result['status'] == 'failed' for result in outcomes):
                    failures.append('{} ({})'.format(scope, 'event'))
                lines.extend(self._event_line(scope, result) for result in outcomes)

        if 'backtest' in whats:
            for scope in publish.resolve_scopes(scopes):
                result = self._backtest_scope(scope, writer, source)
                results[scope] = result
                if result['status'] == 'failed':
                    failures.append('{} ({})'.format(scope, 'backtest'))
                lines.append(self._backtest_line(scope, result))

        if 'manifest' in whats:
            if dry_run:
                lines.append('manifest: not written (dry run)')
            else:
                lines.append(self._write_manifest(writer, failures, freeze=freeze))

        for action in ('point', 'unpublish'):
            if action not in whats:
                continue
            for scope in publish.resolve_scopes(scopes):
                result = self._pointer_scope(action, writer, scope, run)
                results[scope] = result
                if result['status'] == 'failed':
                    failures.append('{} ({})'.format(scope, action))
                lines.append(self._line(scope, result, run))

        for line in lines:
            print(line)
        if self.app.logger is not None:
            self.alert('\n'.join(lines))

        if failures:
            raise RuntimeError('publish: failed scopes: ' + ', '.join(failures))

        return results

    def _forecast_runs(self, scope, writer, plan, event_date, today, force, dry_run, national=None, index=1, total=1,
                       **params):
        """
        Publish the runs of one scope and rebuild its history.

        The history is rebuilt and the manifest entry chosen when some run was published or,
        with ``backfill``, some day was skipped because its run exists, so a retry repairs the
        pointers of a backfill whose history or manifest write failed.

        Parameters
        ----------
        scope : str
            Scope to publish.
        writer : BundleWriter
            Bundle writer.
        plan : list of tuple
            ``(day, run_id, extra)`` of each run: ``day`` is ``None`` for a normal run and
            ``extra`` holds the additional ``publish_forecast`` parameters of a retrospective
            one (``limit_date`` and ``run_at``).
        event_date : str or None
            Forced election date, or ``None`` for the next event of the scope.
        today : str or None
            Date taken as today by the LOREG guard.
        force : bool
            Publish inside the LOREG window anyway.
        dry_run : bool
            Leave the history alone.
        national : str or None
            Date of the next election of ``es``, for the LOREG guard of every scope.
        index, total : int, default 1
            Position of the scope (from 1) and number of scopes, for the live progress lines
            (``_progress``): the start, the phases and the result of each run print as they happen.
        **params
            Forecast parameters.

        Returns
        -------
        tuple
            ``(outcomes, entry)``: the result of each run of ``plan`` (with its ``day`` when
            there is one) and the new manifest entry of the scope (see ``_manifest_entry``),
            ``None`` when the manifest must not change for the scope, there is nothing to point
            to or this is a dry run. If rebuilding the history or reading the manifest fails,
            every ``published`` result and every day skipped because its run exists turns
            ``failed``.
        """
        outcomes = []
        for day, rid, extra in plan:
            progress = self._progress(scope, index, total, day)
            progress.start()
            result = self._forecast_scope(scope, writer, rid, event_date, today, force, skip_existing=day is not None,
                                          national=national, progress=progress, **params, **extra)
            progress.done(self._line(scope, result if day is None else {'day': day, **result}))
            outcomes.append(result)
        entry = None
        run_ids = [rid for _, rid, _ in plan]
        involved = [i for i, result in enumerate(outcomes) if result['status'] == 'published'
                    or (result['status'] == 'skipped' and result.get('reason') == self.RUN_EXISTS)]
        if involved and not dry_run:
            try:
                runs = {head['run_id']: head for head in publish.rebuild_history(writer, scope)['runs']}
                candidates = [outcomes[i] if outcomes[i]['status'] == 'published' else
                              {'run_id': run_ids[i], 'entry': publish.latest_entry(runs[run_ids[i]])}
                              for i in involved if outcomes[i]['status'] == 'published' or run_ids[i] in runs]
                if candidates:
                    entry = self._manifest_entry(writer, scope, candidates, backfill=plan[0][0] is not None)
            except Exception as e:
                self._log_error(scope)
                failed = {'status': 'failed', 'reason': '{}: {}'.format(type(e).__name__, e)}
                outcomes = [{**failed, 'warnings': result.get('warnings', [])} if i in involved else result
                            for i, result in enumerate(outcomes)]
        days = [day for day, _, _ in plan]
        return [result if day is None else {'day': day, **result} for day, result in zip(days, outcomes)], entry

    @staticmethod
    def _manifest_entry(writer, scope, published, backfill):
        """
        Choose the manifest entry of a scope after publishing (or finding) some runs.

        A normal publish always points to the run just produced. A backfill only moves the
        pointer forward: its newest run becomes ``latest`` when the scope has no entry yet
        or the run id is greater than the current ``latest``; otherwise the manifest keeps
        its entry, so neither a newer run nor a run that ``point`` stepped back from is
        promoted by a backfill.

        Parameters
        ----------
        writer : BundleWriter
            Bundle writer, to read the current manifest.
        scope : str
            Scope published.
        published : list of dict
            ``published`` results of this invocation, in plan order (ascending days); with
            ``backfill``, also ``{'run_id', 'entry'}`` of the days whose run already existed.
        backfill : bool
            Whether the runs are retrospective.

        Returns
        -------
        dict or None
            The new entry, or ``None`` to leave the scope of the manifest untouched.
        """
        newest = max(published, key=lambda result: result['run_id'])
        if not backfill:
            return newest['entry']
        current = (publish.read_manifest(writer)['scopes'].get(scope) or {}).get('latest')
        if current is None or newest['run_id'] > current:
            return newest['entry']
        return None

    def _forecast_scope(self, scope, writer, run_id, event_date, today, force, skip_existing=False, national=None,
                        progress=None, **params):
        """
        Publish the forecast of one scope, converting every outcome into a result dict.

        Parameters
        ----------
        scope : str
            Scope to publish.
        writer : BundleWriter
            Bundle writer.
        run_id : str
            Run id of this invocation.
        event_date : str or None
            Forced election date, or ``None`` for the next event of the scope. It is
            normalised with ``date.fromisoformat``; an invalid value fails the scope.
        today : str or None
            Date taken as today by the LOREG guard.
        force : bool
            Publish inside the LOREG window anyway.
        skip_existing : bool, default False
            Check the run first: a complete one (with ``headline.json``) is ``skipped``
            (``run exists``) and an incomplete one is ``failed`` (``incomplete run exists;
            unpublish it first``), after the event and the LOREG guard are resolved and without
            running the model. Without it, an existing run (``FileExistsError``) is ``failed``.
        national : str or None
            Date of the next election of ``es``, for the LOREG guard of every scope.
        progress : ScopeProgress, optional
            Live progress of the scope: passed to ``publish.publish_forecast`` for its phases and,
            with ``verbose``, used to print each model warning. ``None`` prints nothing.
        **params
            Forecast parameters.

        Returns
        -------
        dict
            Result with a ``status`` and its details: ``skipped`` for no upcoming event,
            ``publish.NothingToPublish`` or (with ``skip_existing``) a complete existing run,
            ``refused`` for the LOREG guard and ``failed`` for an incomplete existing run (with
            ``skip_existing``) or any other exception. Every result carries the ``warnings``
            emitted while the model ran (unique messages, in order of appearance), which never
            reach stderr; with ``verbose``, each one is also printed live as a progress line.
        """
        notes = []
        note = progress.note if params.get('verbose') and progress is not None else None
        try:
            value = event_date or publish.next_event_date(scope)
            if value is None:
                return {'status': 'skipped', 'reason': 'no upcoming event', 'warnings': notes}
            event = date.fromisoformat(value).isoformat()
            in_window = publish.loreg_guard(scope, event, today, force, national)
            if skip_existing:
                if writer.exists(bundle.path_part(scope, run_id, 'headline')):
                    return {'status': 'skipped', 'reason': self.RUN_EXISTS, 'warnings': notes}
                if writer.exists(bundle.path_run(scope, run_id)):
                    return {'status': 'failed', 'reason': self.RUN_INCOMPLETE, 'warnings': notes}
            with self._collect_warnings(notes, note):
                out = publish.publish_forecast(scope, writer, run_id, event, freeze=in_window, progress=progress,
                                               **params)
        except publish.PublishRefused as e:
            return {'status': 'refused', 'reason': str(e), 'warnings': notes}
        except publish.NothingToPublish as e:
            return {'status': 'skipped', 'reason': str(e), 'warnings': notes}
        except Exception as e:
            self._log_error(scope)
            return {'status': 'failed', 'reason': '{}: {}'.format(type(e).__name__, e), 'warnings': notes}
        return {'status': 'published', 'run_id': run_id, 'entry': out['entry'], 'seconds': out['seconds'],
                'warnings': notes}

    @staticmethod
    @contextlib.contextmanager
    def _collect_warnings(notes, note=None):
        """
        Collect every warning raised inside the block instead of letting it reach stderr.

        All categories are recorded (``DeprecationWarning`` included, the filters are set to
        ``always`` inside the block) and the previous filters and ``showwarning`` are restored
        on exit, also when the block raises.

        Parameters
        ----------
        notes : list of str
            Destination of the messages: each new message text is appended once, in order of
            appearance.
        note : callable, optional
            Called with the text of each new message as it is raised (live output).

        Yields
        ------
        list of str
            ``notes``.
        """
        def show(message, category, filename, lineno, file=None, line=None):
            text = str(message)
            if text not in notes:
                notes.append(text)
                if note is not None:
                    note(text)

        with warnings.catch_warnings():
            warnings.simplefilter('always')
            warnings.showwarning = show
            yield notes

    def _progress(self, scope, index, total, day=None):
        """
        Build the live progress printer of one scope (and day) of ``forecast``.

        Parameters
        ----------
        scope : str
            Scope being published.
        index, total : int
            Position of the scope (from 1) and number of scopes of the run.
        day : str, optional
            Day of a retrospective run (``backfill``), added to the prefix.

        Returns
        -------
        ScopeProgress
            Callable ``progress(step, seconds)`` of ``publish.publish_forecast``, with ``start``,
            ``note`` and ``done`` for the other lines of the scope.
        """
        return ScopeProgress(self._say, scope, index, total, day)

    def _say(self, text):
        """Print one progress line now, with the time, and log it through the app logger, if there is one."""
        line = '[{}] {}'.format(datetime.now().strftime('%H:%M:%S'), text)
        print(line, flush=True)
        if self.app.logger is not None:
            self.app.logger.info(line)

    def _analysis_scope(self, scope, writer, verbose, today=None, force=False, national=None):
        """
        Publish the houses analysis of one scope, converting every outcome into a result dict.

        Parameters
        ----------
        scope : str
            Scope to publish.
        writer : BundleWriter
            Bundle writer.
        verbose : int
            Verbosity passed to the analysis.
        today : str or None
            Date taken as today by the LOREG guard.
        force : bool
            Publish inside the LOREG window anyway; the analysis carries no mark of it.
        national : str or None
            Date of the next election of ``es``, for the LOREG guard of every scope.

        Returns
        -------
        dict
            Result with a ``status``: ``published`` with ``event_date``, ``n_pollsters`` and
            ``seconds``, ``skipped`` for ``publish.NothingToPublish``, ``refused`` for the LOREG
            guard and ``failed`` for any other exception, all but ``published`` with a ``reason``.
            A scope without an upcoming event is still published (the analysis then works from
            its last election) and only the national window guards it.
        """
        try:
            publish.loreg_guard(scope, publish.next_event_date(scope), today, force, national)
            out = publish.publish_analysis(scope, writer, verbose=verbose)
        except publish.PublishRefused as e:
            return {'status': 'refused', 'reason': str(e)}
        except publish.NothingToPublish as e:
            return {'status': 'skipped', 'reason': str(e)}
        except Exception as e:
            self._log_error(scope)
            return {'status': 'failed', 'reason': '{}: {}'.format(type(e).__name__, e)}
        return {'status': 'published', 'event_date': out['event_date'], 'n_pollsters': out['n_pollsters'],
                'seconds': out['seconds']}

    def _event_scope(self, scope, writer, event_date, today, source, verbose):
        """
        Publish the past elections of one scope, converting every outcome into a result dict.

        Parameters
        ----------
        scope : str
            Scope to publish.
        writer : BundleWriter
            Bundle writer.
        event_date : str or list of str
            Date, list of dates or ``'all'`` (see ``publish.event_dates_to_publish``).
        today : str or None
            Date taken as today to resolve ``'all'``.
        source : str or None
            Results folder of the backtest (see ``publish.backtest_source``).
        verbose : int
            Verbosity passed to the computer.

        Returns
        -------
        list of dict
            One result per date, each with its ``event_date`` and a ``status``: ``published`` with ``n_polls``,
            ``n_pollsters``, ``has_model`` and ``seconds``, ``skipped`` for ``publish.NothingToPublish`` and
            ``failed`` for any other exception (an invalid date or a failure resolving ``'all'`` included),
            both with a ``reason``. ``'all'`` with no past featured election gives one ``skipped`` result.
        """
        try:
            days = publish.event_dates_to_publish(scope, event_date, today=today)
        except Exception as e:
            self._log_error(scope)
            return [{'status': 'failed', 'event_date': str(event_date), 'reason': '{}: {}'.format(type(e).__name__, e)}]
        if not days:
            return [{'status': 'skipped', 'event_date': str(event_date), 'reason': 'no past featured elections'}]

        outcomes = []
        for day in days:
            try:
                out = publish.publish_event(scope, day, writer, source=source, verbose=verbose)
            except publish.NothingToPublish as e:
                outcomes.append({'status': 'skipped', 'event_date': day, 'reason': str(e)})
            except Exception as e:
                self._log_error(scope)
                outcomes.append({'status': 'failed', 'event_date': day, 'reason': '{}: {}'.format(type(e).__name__, e)})
            else:
                outcomes.append({'status': 'published', 'event_date': day, 'n_polls': out['n_polls'],
                                 'n_pollsters': out['n_pollsters'], 'has_model': out['has_model'],
                                 'seconds': out['seconds']})
        return outcomes

    def _backtest_scope(self, scope, writer, source):
        """
        Publish the model backtest of one scope, converting every outcome into a result dict.

        Parameters
        ----------
        scope : str
            Scope to publish.
        writer : BundleWriter
            Bundle writer.
        source : str or None
            Results folder (see ``publish.backtest_source``).

        Returns
        -------
        dict
            Result with a ``status``: ``published`` with ``n_cases``, ``n_events`` and ``seconds``,
            ``skipped`` for ``publish.NothingToPublish`` and ``failed`` for any other exception,
            both with a ``reason``.
        """
        try:
            out = publish.publish_backtest(scope, writer, source=source)
        except publish.NothingToPublish as e:
            return {'status': 'skipped', 'reason': str(e)}
        except Exception as e:
            self._log_error(scope)
            return {'status': 'failed', 'reason': '{}: {}'.format(type(e).__name__, e)}
        return {'status': 'published', 'n_cases': out['n_cases'], 'n_events': out['n_events'],
                'seconds': out['seconds']}

    def _pointer_scope(self, action, writer, scope, run):
        """
        Point a scope to a run or remove a run, converting the outcome into a result dict.

        Parameters
        ----------
        action : str
            ``point`` or ``unpublish``.
        writer : BundleWriter
            Bundle writer.
        scope : str
            Scope to act on.
        run : str
            Run id.

        Returns
        -------
        dict
            Result with a ``status`` and its details.
        """
        try:
            if action == 'point':
                publish.point(writer, scope, run)
                return {'status': 'published', 'latest': run}
            return {'status': 'published', **publish.unpublish(writer, scope, run)}
        except ValueError as e:
            return {'status': 'failed', 'reason': str(e)}
        except Exception as e:
            self._log_error(scope)
            return {'status': 'failed', 'reason': '{}: {}'.format(type(e).__name__, e)}

    def _write_manifest(self, writer, failures, **kwargs):
        """
        Update the manifest, converting an exception into a failure and a summary line.

        Parameters
        ----------
        writer : BundleWriter
            Bundle writer.
        failures : list of str
            Failures of the run; ``manifest`` is appended when the write fails.
        **kwargs
            Arguments of ``publish.update_manifest`` (``scopes`` or ``freeze``).

        Returns
        -------
        str
            Summary line of the manifest.
        """
        try:
            return self._manifest_line(publish.update_manifest(writer, **kwargs))
        except Exception as e:
            self._log_error('manifest')
            failures.append('manifest')
            return 'manifest: failed ({}: {})'.format(type(e).__name__, e)

    def _log_error(self, scope):
        """Log the traceback being handled through the app logger, if there is one."""
        if self.app.logger is not None:
            self.app.logger.error('publish {}: {}'.format(scope, traceback.format_exc()))

    @staticmethod
    def _manifest_line(data):
        """Format the summary line of the manifest data."""
        return 'manifest: {} scopes, freeze {}'.format(
            len(data.get('scopes', {})), 'on' if (data.get('freeze') or {}).get('active') else 'off')

    @staticmethod
    def _analysis_line(scope, result):
        """Format the summary line of one scope result of ``analysis``."""
        if result['status'] == 'published':
            return '{}: analysis published (event {}, {} pollsters) in {:.0f} s'.format(
                scope, result['event_date'], result['n_pollsters'], result['seconds']['total'])
        return '{}: analysis {} ({})'.format(scope, result['status'], result['reason'])

    @staticmethod
    def _event_line(scope, result):
        """Format the summary line of one date result of ``event``."""
        if result['status'] == 'published':
            return '{}: event {} published ({} polls, {} pollsters) in {:.0f} s'.format(
                scope, result['event_date'], result['n_polls'], result['n_pollsters'], result['seconds']['total'])
        return '{}: event {} {} ({})'.format(scope, result['event_date'], result['status'], result['reason'])

    @staticmethod
    def _backtest_line(scope, result):
        """Format the summary line of one scope result of ``backtest``."""
        if result['status'] == 'published':
            return '{}: backtest published ({} cases, {} elections) in {:.0f} s'.format(
                scope, result['n_cases'], result['n_events'], result['seconds']['total'])
        return '{}: backtest {} ({})'.format(scope, result['status'], result['reason'])

    @staticmethod
    def _line(scope, result, run=None):
        """Format the summary line of one scope result (of one day, with ``backfill``)."""
        status = result['status']
        day = result.get('day')
        if status == 'published' and 'entry' in result:
            entry = result['entry']
            return '{}: published {} (as_of {}, {} polls{}) in {:.0f} s{}'.format(
                scope, result['run_id'], entry.get('as_of'), entry.get('n_polls'),
                '' if day is None else ', backfill {}'.format(day), result['seconds']['total'],
                Publish._notes_suffix(result))
        if status == 'published':
            return '{}: published {}'.format(scope, result.get('latest') or result.get('removed') or run)
        if day is not None:
            line = '{}: {} {} ({})'.format(scope, status, day, result['reason'])
        else:
            line = '{}: {} ({})'.format(scope, status, result['reason'])
        return line + Publish._notes_suffix(result)

    @staticmethod
    def _notes_suffix(result):
        """Return the `` · n avisos`` suffix of a result with model warnings, or an empty string."""
        n = len(result.get('warnings') or ())
        if n == 0:
            return ''
        return ' · {} {}'.format(n, 'aviso' if n == 1 else 'avisos')
