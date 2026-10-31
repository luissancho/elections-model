from datetime import date, timedelta
from typing import Optional

from ...jobs.progress import collect_warnings, counted, notes_suffix
from ...lib import bundle, publish
from ..base import ScopePipeline
from .Manifest import manifest_line


STEPS = {'init': 'simulador listo', 'fit': 'promedio ajustado', 'nowcast': 'simulación de hoy',
         'export-nowcast': 'exportación de hoy', 'forecast': 'simulación de la elección', 'export': 'exportación',
         'meta': 'meta y titular'}


def result_line(scope, result, run=None):
    """
    Format the summary line of one scope result of a forecast (of one day, with ``backfill``), a ``point`` or an
    ``unpublish``.

    Parameters
    ----------
    scope : str
        Scope of the result.
    result : dict
        Result with a ``status`` and its details.
    run : str, optional
        Run id of a ``point`` or ``unpublish`` result without ``latest`` or ``removed``.

    Returns
    -------
    str
        The summary line.
    """
    status = result['status']
    day = result.get('day')
    if status == 'published' and 'entry' in result:
        entry = result['entry']
        return '{}: published {} (as_of {}, {} polls{}) in {:.0f} s{}'.format(
            scope, result['run_id'], entry.get('as_of'), entry.get('n_polls'),
            '' if day is None else ', backfill {}{}'.format(day, ', replaced' if result.get('replaced') else ''),
            result['seconds']['total'],
            notes_suffix(result))
    if status == 'published':
        return '{}: published {}'.format(scope, result.get('latest') or result.get('removed') or run)
    if day is not None:
        line = '{}: {} {} ({})'.format(scope, status, day, result['reason'])
    else:
        line = '{}: {} ({})'.format(scope, status, result['reason'])
    return line + notes_suffix(result)


def national_event(today=None):
    """
    Date of the next general election (``es``) as the LOREG guard of every scope needs it.

    The day before today is the starting point, so the national window also covers the election day itself.

    Parameters
    ----------
    today : str, optional
        Date taken as today.

    Returns
    -------
    str or None
        ISO date of the next election of ``es``, or ``None`` if there is none.
    """
    eve = (publish.common._today(today) - timedelta(days=1)).isoformat()
    return publish.next_event_date('es', date_from=eve)


class Forecast(ScopePipeline):
    """
    Publish the forecast of one scope to the web bundle (``site/v1``): its run folder and history, and (in `finish`,
    once for every scope of the job) the manifest entries of the scopes published.

    The chunk is the scope. Each scope ends ``published``, ``skipped`` (no upcoming event, or
    ``publish.NothingToPublish``: no polls or an average that cannot be fitted), ``refused`` (LOREG guard) or
    ``failed`` (any other error, ``ValueError`` included, such as a bundle validation failure or a non-ISO
    ``event_date``); a failure never raises. The progress is printed live (``[i/n] scope: phase (seconds)`` and the
    result line of each run) and the model warnings are collected into the result (``warnings``).

    With ``days`` (a backfill), one retrospective run is published per day, with the polls published up to that
    day and the run id and ``run_at`` of its noon; the result of the scope is then the list of the results of its
    days (see `publish_task`).

    Attributes
    ----------
    writer : BundleWriter
        Bundle writer (under the dry-run prefix in a dry run).
    run_id : str
        Run id of a normal (not retrospective) run of this job.
    plan : list of tuple
        ``(day, run_id, extra)`` of each run of a scope (see `publish_task`).
    entries : dict
        New manifest entry of each scope published, written by `finish`.
    """

    label = 'forecast'
    steps = STEPS
    RUN_EXISTS = 'run exists'
    RUN_INCOMPLETE = 'incomplete run exists; unpublish it first'
    pipes = {'publish': {'Load': ['publish']}}

    def __init__(self, *args, **kwargs):
        """Create the pipeline with no manifest entries, so `finish` is safe even if `start` failed."""
        super().__init__(*args, **kwargs)
        self.entries = {}

    def start(self, fs=None, dry_run=False, today=None, days=None, replace=False, event_date=None, force=False,
              n_sim=1000, seed=42, drange=6, max_fc=10, alpha=0.05, correctors=None, **kwargs) -> None:
        """
        Set the run: the writer, the run id, the plan of runs of each scope, the forecast parameters, the date of
        the next election of ``es`` and the header ``publish forecast: n ámbitos · run <id>`` (or ``· backfill
        <from>..<to> (n días)``).

        Parameters
        ----------
        fs : FileSystem
            File system to write to.
        dry_run : bool, default False
            Write under ``site-dry/v1`` and leave the history and the manifest alone.
        today : str, optional
            Date taken as today by the LOREG guard and the anchor of the average.
        days : list of str, optional
            Days of a backfill (``publish.backfill_days``); ``None`` for a normal run.
        replace : bool, default False
            Remove the existing run of each retrospective day before computing it again.
        event_date : str, optional
            Forced election date; ``None`` for the next event of each scope.
        force : bool, default False
            Publish inside the LOREG window anyway.
        n_sim, seed, drange, max_fc, alpha, correctors
            Forecast parameters.
        **kwargs
            ``verbose`` and ``position`` (see `ScopePipeline.start`); other arguments of the job are ignored.
        """
        super().start(**kwargs)
        self.writer = bundle.BundleWriter(fs, prefix=bundle.DRY_PREFIX if dry_run else bundle.PREFIX)
        self.run_id = bundle.run_id()
        self.days = days
        self.replace = replace
        self.event_date = event_date
        self.today = today
        self.force = force
        self.dry_run = dry_run
        if days is None:
            self.plan = [(None, self.run_id, {})]
        else:
            self.plan = [(day, publish.backfill_run_id(day),
                          {'limit_date': day, 'run_at': publish.backfill_run_at(day)}) for day in days]
        self.params = dict(n_sim=n_sim, seed=seed, drange=drange, max_fc=max_fc, alpha=alpha,
                           correctors=correctors, verbose=self.model_verbose)
        self.national = national_event(today)
        n_scopes = counted(self.total, 'ámbito', 'ámbitos')
        if days is None:
            self._say('publish forecast: {} · run {}'.format(n_scopes, self.run_id))
        else:
            self._say('publish forecast: {} · backfill {}..{} ({})'.format(
                n_scopes, days[0], days[-1], counted(len(days), 'día', 'días')))

    def publish_task(self, chunk=None, position=(1, 1), **kwargs):
        """
        Publish the runs of one scope and rebuild its history.

        The history is rebuilt and the manifest entry chosen when some run was published or,
        with ``backfill``, some day was skipped because its run exists (so a retry repairs the
        pointers of a backfill whose history or manifest write failed) or had its run removed to
        be replaced (also when computing it again failed, so the history never lists it).

        Parameters
        ----------
        chunk : str
            Scope to publish.
        position : tuple of int, default (1, 1)
            Position of the scope (from 1) and number of scopes, for the live progress lines: the start, the
            phases and the result of each run print as they happen.

        Notes
        -----
        ``self.result`` is the result of the run (without ``days``) or the list of the result of each day, each
        with its ``day``. The new manifest entry of the scope (see `_manifest_entry`) is kept in
        ``self.entries[chunk]``; there is none when the manifest must not change for the scope, there is nothing to
        point to or this is a dry run. If rebuilding the history or reading the manifest fails, every
        ``published`` result and every day skipped because its run exists turns ``failed``.
        """
        scope, writer, plan, dry_run = chunk, self.writer, self.plan, self.dry_run
        outcomes = []
        for day, rid, extra in plan:
            progress = self.progress_of(scope, position, day)
            progress.start()
            result = self._forecast_scope(scope, writer, rid, self.event_date, self.today, self.force,
                                          skip_existing=day is not None, replace=self.replace and day is not None,
                                          dry_run=dry_run, national=self.national, progress=progress,
                                          **self.params, **extra)
            progress.done(self.line_of(scope, result if day is None else {'day': day, **result}))
            outcomes.append(result)
        entry = None
        run_ids = [rid for _, rid, _ in plan]
        involved = [i for i, result in enumerate(outcomes) if result['status'] == 'published' or result.get('replaced')
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
        outcomes = [result if day is None else {'day': day, **result} for day, result in zip(days, outcomes)]
        self.result = outcomes[0] if self.days is None else outcomes
        if entry is not None:
            self.entries[chunk] = entry

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

    def _forecast_scope(self, scope, writer, run_id, event_date, today, force, skip_existing=False, replace=False,
                        dry_run=False, national=None, progress=None, **params):
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
        replace : bool, default False
            With ``skip_existing``, remove an existing run (complete or not) and run the model
            instead: ``publish.unpublish`` rebuilds the history and repoints ``latest`` when it was
            that run, so the pointers stay consistent even if the new run fails; in a dry run only
            the folder is removed. It happens after the event and the LOREG guard are resolved, so
            a refused day keeps its run.
        dry_run : bool, default False
            Whether the bundle is the dry one (``replace`` then removes the folder alone).
        national : str or None
            Date of the next election of ``es``, for the LOREG guard of every scope.
        progress : ScopeProgress, optional
            Live progress of the scope: passed to ``publish.publish_forecast`` for its phases and,
            used to print each model warning (level 1). ``None`` prints nothing.
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
            reach stderr; each one is also printed live as a progress line (level 1). With
            ``skip_existing``, every result also carries ``replaced``: whether an existing run was
            removed first.
        """
        notes = []
        replaced = False
        note = progress.note if progress is not None else None

        def done(result):
            return {**result, 'replaced': replaced} if skip_existing else result

        try:
            value = event_date or publish.next_event_date(scope)
            if value is None:
                return done({'status': 'skipped', 'reason': 'no upcoming event', 'warnings': notes})
            event = date.fromisoformat(value).isoformat()
            in_window = publish.loreg_guard(scope, event, today, force, national)
            if skip_existing and writer.exists(bundle.path_run(scope, run_id)):
                if not replace:
                    if writer.exists(bundle.path_part(scope, run_id, 'headline')):
                        return done({'status': 'skipped', 'reason': self.RUN_EXISTS, 'warnings': notes})
                    return done({'status': 'failed', 'reason': self.RUN_INCOMPLETE, 'warnings': notes})
                self._remove_run(writer, scope, run_id, dry_run)
                replaced = True
            with collect_warnings(notes, note):
                out = publish.publish_forecast(scope, writer, run_id, event, freeze=in_window, progress=progress,
                                               today=today, **params)
        except publish.PublishRefused as e:
            return done({'status': 'refused', 'reason': str(e), 'warnings': notes})
        except publish.NothingToPublish as e:
            return done({'status': 'skipped', 'reason': str(e), 'warnings': notes})
        except Exception as e:
            self._log_error(scope)
            return done({'status': 'failed', 'reason': '{}: {}'.format(type(e).__name__, e), 'warnings': notes})
        return done({'status': 'published', 'run_id': run_id, 'entry': out['entry'], 'seconds': out['seconds'],
                     'warnings': notes})

    @staticmethod
    def _remove_run(writer, scope, run_id, dry_run):
        """
        Remove an existing run that is about to be computed again.

        Parameters
        ----------
        writer : BundleWriter
            Bundle writer.
        scope : str
            Scope of the run.
        run_id : str
            Run to remove.
        dry_run : bool
            In a dry run only the folder goes, since the dry bundle has no pointers; otherwise
            ``publish.unpublish`` also rebuilds the history and repoints ``latest`` when it was
            this run, so nothing points to a run that no longer exists while the model runs.
        """
        if dry_run:
            writer.remove(bundle.path_run(scope, run_id))
        else:
            publish.unpublish(writer, scope, run_id)

    def line_of(self, chunk, result, run=None, **kwargs):
        """Format the summary line of one scope result (of one day, with ``backfill``)."""
        return result_line(chunk, result, run)

    def summary(self, chunk, **kwargs):
        """
        Build the summary line of the scope (the list of the lines of its days, with ``days``).

        The lines were already reported live, run by run, by `publish_task`, so they are not reported again.
        """
        if isinstance(self.result, list):
            return [self.line_of(chunk, result) for result in self.result]
        return self.line_of(chunk, self.result)

    def finish(self, dry_run=False, **kwargs) -> Optional[str]:
        """
        Write the manifest entries of the scopes published, once for the whole job.

        Parameters
        ----------
        dry_run : bool, default False
            Leave the manifest alone (line ``manifest: not written (dry run)``).
        **kwargs
            Other arguments of the run, ignored.

        Returns
        -------
        str or None
            ``'manifest'`` if writing the manifest failed (logged, line ``manifest: failed (Type: message)``);
            otherwise ``None``. Without entries (no scope published, or none that moves the manifest), nothing is
            written and there is no line.
        """
        if dry_run:
            self.line = 'manifest: not written (dry run)'
            return None
        if not self.entries:
            return None
        try:
            self.line = manifest_line(publish.update_manifest(self.writer, scopes=self.entries))
            return None
        except Exception as e:
            self._log_error('manifest')
            self.line = 'manifest: failed ({}: {})'.format(type(e).__name__, e)
            return 'manifest'
