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

    Examples
    --------
    From the command line::

        python job.py publish '{"what":["forecast"],"scopes":["es"]}'
        python job.py publish '{"what":["forecast"],"scopes":"all"}'
        python job.py publish '{"what":["manifest"],"freeze":{"active":true,"message":"..."}}'
        python job.py publish '{"what":["point"],"scopes":["es"],"run":"20261007-141503"}'
        python job.py publish '{"what":["forecast"],"scopes":["es"],"n_sim":20,"dry_run":true}'

    In Docker (JSON without spaces)::

        RUN_JOB='publish {"what":["forecast"],"scopes":["es"]}'
    """

    WHAT = ('forecast', 'manifest', 'point', 'unpublish')
    DEFERRED = {'analysis': 4, 'event': 5, 'backtest': 5}

    def run(self, what=('forecast',), scopes=('es',), event_date=None, n_sim=1000, seed=42, drange=6, max_fc=10,
            alpha=0.05, correctors=None, freeze=None, run=None, dry_run=False, force=False, fs=None, today=None,
            verbose=0, **kwargs):
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
            Date (``YYYY-MM-DD``) taken as today by the LOREG guard; for tests and
            rehearsals only (unlike ``force``, it leaves no mark in ``meta.freeze``).
        verbose : int, default 0
            Verbosity passed to the forecast.
        **kwargs
            Ignored extra job arguments.

        Returns
        -------
        dict[str, dict]
            Result of each scope, with a ``status`` of ``published``, ``skipped``,
            ``refused`` or ``failed``.

        Raises
        ------
        ValueError
            If ``what`` is unknown or deferred to a later phase, or ``run`` is missing.
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
            for scope in publish.resolve_scopes(scopes):
                result = self._forecast_scope(
                    scope, writer, run_id, event_date, today, force, n_sim=n_sim, seed=seed, drange=drange,
                    max_fc=max_fc, alpha=alpha, correctors=correctors, verbose=verbose)
                if result['status'] == 'published' and not dry_run:
                    try:
                        publish.rebuild_history(writer, scope)
                    except Exception as e:
                        self._log_error(scope)
                        result = {'status': 'failed', 'reason': '{}: {}'.format(type(e).__name__, e)}
                results[scope] = result
                if result['status'] == 'failed':
                    failures.append('{} ({})'.format(scope, 'forecast'))
                if result['status'] == 'published':
                    entries[scope] = result['entry']
                lines.append(self._line(scope, result))
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

    def _forecast_scope(self, scope, writer, run_id, event_date, today, force, **params):
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
        **params
            Forecast parameters.

        Returns
        -------
        dict
            Result with a ``status`` and its details: ``skipped`` for no upcoming event or
            ``publish.NothingToPublish``, ``refused`` for the LOREG guard and ``failed`` for
            any other exception.
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
        """Format the summary line of one scope result."""
        status = result['status']
        if status == 'published' and 'entry' in result:
            entry = result['entry']
            return '{}: published {} (as_of {}, {} polls) in {:.0f} s'.format(
                scope, result['run_id'], entry.get('as_of'), entry.get('n_polls'), result['seconds']['total'])
        if status == 'published':
            return '{}: published {}'.format(scope, result.get('latest') or result.get('removed') or run)
        return '{}: {} ({})'.format(scope, status, result['reason'])
