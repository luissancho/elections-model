import traceback
from datetime import date

from ..core.worker import Job
from ..lib import bundle, publish


class Publish(Job):
    """
    Publish the model results to the web bundle (``site/v1``) of the file system.

    Runs the forecast of each scope, writes its run folder and moves the pointers
    (``latest``, ``history`` and the manifest). Scopes are isolated: one that fails
    never stops the others, and the job exits with an error at the end if any failed.

    Each scope ends ``published``, ``skipped`` (no upcoming event, or
    ``publish.NothingToPublish``: no polls or an average that cannot be fitted),
    ``refused`` (LOREG guard) or ``failed`` (any other error, ``ValueError`` included,
    such as a bundle validation failure or a non-ISO ``event_date``).

    With ``backfill``, ``forecast`` publishes retrospective runs instead: one per day of the
    range and scope, with the polls published up to that day, the run id and ``run_at`` of
    its noon, and ``backfill`` set in its headline. A day whose run already exists is
    ``skipped``. The manifest always points to the newest run of the rebuilt history, so a
    backfill never moves ``latest`` back.

    Examples
    --------
    From the command line::

        python job.py publish '{"what":["forecast"],"scopes":["es"]}'
        python job.py publish '{"what":["forecast"],"scopes":"all"}'
        python job.py publish '{"what":["manifest"],"freeze":{"active":true,"message":"..."}}'
        python job.py publish '{"what":["point"],"scopes":["es"],"run":"20261007-141503"}'
        python job.py publish '{"what":["forecast"],"scopes":["es"],"n_sim":20,"dry_run":true}'
        python job.py publish '{"what":["forecast"],"scopes":["es"],"backfill":{"from":"2026-10-01","to":"2026-10-07"}}'

    In Docker (JSON without spaces)::

        RUN_JOB='publish {"what":["forecast"],"scopes":["es"]}'
    """

    WHAT = ('forecast', 'manifest', 'point', 'unpublish')
    DEFERRED = {'analysis': 4, 'event': 5, 'backtest': 5}

    def run(self, what=('forecast',), scopes=('es',), event_date=None, n_sim=1000, seed=42, drange=6, max_fc=10,
            alpha=0.05, correctors=None, freeze=None, run=None, dry_run=False, force=False, fs=None, today=None,
            backfill=None, verbose=0, **kwargs):
        """
        Run the requested publication actions.

        Parameters
        ----------
        what : str or list of str, default ('forecast',)
            Actions to run, among ``forecast``, ``manifest``, ``point`` and ``unpublish``.
            They run in that order when several are given.
        scopes : str or list of str, default ('es',)
            Scopes to act on, or ``'all'``.
        event_date : str, optional
            Election date (ISO 8601, normalised to ``YYYY-MM-DD``). Defaults to the next
            event of each scope. A value that is not a date fails each scope.
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
            Write under ``site-dry/v1`` and leave the pointers and manifest alone.
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
            and cannot be after today) of retrospective ``forecast`` runs, one per day; see
            ``publish.backfill_days``.
        verbose : int, default 0
            Verbosity passed to the forecast.
        **kwargs
            Ignored extra job arguments.

        Returns
        -------
        dict[str, dict or list of dict]
            Result of each scope, with a ``status`` of ``published``, ``skipped``,
            ``refused`` or ``failed``; with ``backfill``, the ``forecast`` result of a scope
            is the list of the results of its days, each with its ``day``.

        Raises
        ------
        ValueError
            If ``what`` is unknown or deferred to a later phase, ``run`` is missing or the
            ``backfill`` range is invalid.
        RuntimeError
            If no file system is configured, or any scope failed (raised after the
            manifest, the summary and the alert).
        """
        whats = [what] if isinstance(what, str) else list(what)
        for item in whats:
            if item in self.DEFERRED:
                raise ValueError('publish: "{}" arrives in phase {}'.format(item, self.DEFERRED[item]))
        for item in whats:
            if item not in self.WHAT:
                raise ValueError('publish: unknown what "{}" (expected one of {})'.format(item, ', '.join(self.WHAT)))
        if ('point' in whats or 'unpublish' in whats) and not run:
            raise ValueError('publish: "run" is required')
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

        if 'forecast' in whats:
            entries = {}
            if days is None:
                plan = [(None, run_id, {})]
            else:
                plan = [(day, publish.backfill_run_id(day),
                         {'limit_date': day, 'run_at': publish.backfill_run_at(day)}) for day in days]
            params = dict(n_sim=n_sim, seed=seed, drange=drange, max_fc=max_fc, alpha=alpha,
                          correctors=correctors, verbose=verbose)
            for scope in publish.resolve_scopes(scopes):
                outcomes, entry = self._forecast_runs(scope, writer, plan, event_date, today, force, dry_run,
                                                      **params)
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

    def _forecast_runs(self, scope, writer, plan, event_date, today, force, dry_run, **params):
        """
        Publish the runs of one scope and rebuild its history.

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
        **params
            Forecast parameters.

        Returns
        -------
        tuple
            ``(outcomes, entry)``: the result of each run of ``plan`` (with its ``day`` when
            there is one) and the manifest entry of the newest run of the rebuilt history,
            ``None`` when nothing was published or this is a dry run. If rebuilding the history
            fails, every ``published`` result turns ``failed``.
        """
        outcomes = [self._forecast_scope(scope, writer, rid, event_date, today, force,
                                         skip_existing=day is not None, **params, **extra)
                    for day, rid, extra in plan]
        entry = None
        if any(result['status'] == 'published' for result in outcomes) and not dry_run:
            try:
                history = publish.rebuild_history(writer, scope)
                entry = publish.latest_entry(history['runs'][-1])
            except Exception as e:
                self._log_error(scope)
                failed = {'status': 'failed', 'reason': '{}: {}'.format(type(e).__name__, e)}
                outcomes = [failed if result['status'] == 'published' else result for result in outcomes]
        days = [day for day, _, _ in plan]
        return [result if day is None else {'day': day, **result} for day, result in zip(days, outcomes)], entry

    def _forecast_scope(self, scope, writer, run_id, event_date, today, force, skip_existing=False, **params):
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
            Turn an existing run (``FileExistsError``) into ``skipped`` instead of ``failed``.
        **params
            Forecast parameters.

        Returns
        -------
        dict
            Result with a ``status`` and its details: ``skipped`` for no upcoming event,
            ``publish.NothingToPublish`` or (with ``skip_existing``) an existing run, ``refused``
            for the LOREG guard and ``failed`` for any other exception.
        """
        try:
            value = event_date or publish.next_event_date(scope)
            if value is None:
                return {'status': 'skipped', 'reason': 'no upcoming event'}
            event = date.fromisoformat(value).isoformat()
            in_window = publish.loreg_guard(scope, event, today, force)
            out = publish.publish_forecast(scope, writer, run_id, event, freeze=in_window, **params)
        except publish.PublishRefused as e:
            return {'status': 'refused', 'reason': str(e)}
        except publish.NothingToPublish as e:
            return {'status': 'skipped', 'reason': str(e)}
        except FileExistsError as e:
            if skip_existing:
                return {'status': 'skipped', 'reason': 'run exists'}
            self._log_error(scope)
            return {'status': 'failed', 'reason': '{}: {}'.format(type(e).__name__, e)}
        except Exception as e:
            self._log_error(scope)
            return {'status': 'failed', 'reason': '{}: {}'.format(type(e).__name__, e)}
        return {'status': 'published', 'run_id': run_id, 'entry': out['entry'], 'seconds': out['seconds']}

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
    def _line(scope, result, run=None):
        """Format the summary line of one scope result (of one day, with ``backfill``)."""
        status = result['status']
        day = result.get('day')
        if status == 'published' and 'entry' in result:
            entry = result['entry']
            return '{}: published {} (as_of {}, {} polls{}) in {:.0f} s'.format(
                scope, result['run_id'], entry.get('as_of'), entry.get('n_polls'),
                '' if day is None else ', backfill {}'.format(day), result['seconds']['total'])
        if status == 'published':
            return '{}: published {}'.format(scope, result.get('latest') or result.get('removed') or run)
        if day is not None:
            return '{}: {} {} ({})'.format(scope, status, day, result['reason'])
        return '{}: {} ({})'.format(scope, status, result['reason'])
