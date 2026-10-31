from ..core.worker import SequenceJob
from ..lib import publish


class Publish(SequenceJob):
    """
    Publish the model results to the web bundle (``site/v1``) of the file system.

    Runs the forecast of each scope, writes its run folder and moves the pointers
    (``latest``, ``history`` and the manifest). Scopes are isolated: one that fails
    never stops the others (unless ``stop_on_error``), and the job exits with an error at the end if any failed.

    Each scope ends ``published``, ``skipped`` (no upcoming event, or
    ``publish.NothingToPublish``: no polls or an average that cannot be fitted),
    ``refused`` (LOREG guard) or ``failed`` (any other error, ``ValueError`` included,
    such as a bundle validation failure or a non-ISO ``event_date``). ``forecast`` prints its
    progress live (``[i/n] scope: phase (seconds)`` and the result line of each scope, also to the
    logger) and collects the model warnings of each scope into its result (``warnings``) and the
    count of its summary line instead of letting them reach stderr; from ``verbose`` 1 (the default), it prints
    them live too. ``verbose`` has four levels (0 header and summary, 1 scope lines and warnings, 2 phases,
    3 model output); see ``run``.

    ``analysis`` computes the houses analysis of each scope (house effects, herding, poll
    errors and error data of the ``Computer``) and overwrites ``analysis/{scope}/``. It has no
    LOREG guard, never touches the manifest and, with ``dry_run``, writes under
    ``site-dry/v1/analysis/``. Each scope ends ``published``, ``skipped``
    (``publish.NothingToPublish``) or ``failed``.

    ``backtest`` converts the model backtest tables (``backtest/results[/{scope}]/``, written by
    the ``backtest`` job) into ``backtest/{scope}/`` and overwrites it. It reads files, not the
    database, has no LOREG guard, never touches the manifest and, with ``dry_run``, writes under
    ``site-dry/v1/backtest/``. Each scope ends ``published``, ``skipped`` (no results for the scope)
    or ``failed``.

    ``study`` computes a study (``voto-cera``, the CERA vote study, or ``sesgo-cis``, the lean of the CIS computed
    from the published analyses and events) and overwrites ``studies/{study}/`` (``publish.publish_study``), updating
    its entry of the manifest. It needs ``study``, has no LOREG guard (it publishes no polls) and registers the
    prediction of the study only with ``register`` (and a clean tree; ``force`` replaces an existing registration;
    ``sesgo-cis`` has nothing to register and refuses it). ``sesgo-cis`` reads the live bundle, so it is republished
    after ``analysis`` and ``event``. With ``dry_run`` it writes under ``site-dry/v1/``. The study ends
    ``published``, ``refused`` (``publish.PublishRefused``: the registration was refused) or ``failed``.

    ``event`` publishes past elections of each scope into ``events/{scope}/{date}/`` (poll errors by house, official
    results and the slice and daily series of the model backtest of the election), overwriting them. It needs
    ``event_date`` (a date, a list of dates or ``'all'``: the elections of the scope with an official result before
    today), cannot be combined with ``forecast``, has no LOREG guard, never touches the manifest and, with
    ``dry_run``, writes under ``site-dry/v1/events/``. Each date of each scope ends ``published``, ``skipped`` (not
    an election of the scope with official results, or no polls: ``publish.NothingToPublish``), ``refused``
    (``publish.PublishRefused``: the live event has the model slice or the series but the backtest tables are
    missing in the file system; ``force`` publishes it anyway, and they are removed) or ``failed``.

    With ``backfill``, ``forecast`` publishes retrospective runs instead: one per day of the
    range and scope, with the polls published up to that day, the run id and ``run_at`` of
    its noon, and ``backfill`` set in its headline. A day whose run already exists is
    ``skipped``, or ``failed`` when that run is incomplete (it has no ``headline.json``; remove
    it with ``unpublish`` and retry), unless ``backfill`` carries ``replace``: then the existing
    run, complete or not, is removed first as ``unpublish`` does (after the LOREG guard, so a
    refused day keeps its run) and the day is computed again, ``replaced`` in its result. Every
    backfill rebuilds the history of the scope, also when all its days already existed, so a
    retry repairs a history or manifest left behind. A backfill only moves the manifest
    ``latest`` forward (to its newest run, when that is newer than the current one), so it never
    moves ``latest`` back nor undoes a ``point``.

    Examples
    --------
    From the command line::

        python job.py publish '{"what":["forecast"],"scopes":["es"]}'
        python job.py publish '{"what":["forecast"],"scopes":"all"}'
        python job.py publish '{"what":["forecast","analysis"],"scopes":["es"],"n_sim":10000}'
        python job.py publish '{"what":["analysis"],"scopes":["es"]}'
        python job.py publish '{"what":["backtest"],"scopes":["es"]}'
        python job.py publish '{"what":["event"],"scopes":["es"],"event_date":"2023-07-23"}'
        python job.py publish '{"what":["event"],"scopes":["es"],"event_date":"all"}'
        python job.py publish '{"what":["study"],"study":"voto-cera"}'
        python job.py publish '{"what":["study"],"study":"voto-cera","register":true}'
        python job.py publish '{"what":["study"],"study":"sesgo-cis"}'
        python job.py publish '{"what":["manifest"],"freeze":{"active":true,"message":"..."}}'
        python job.py publish '{"what":["point"],"scopes":["es"],"run":"20261007-141503"}'
        python job.py publish '{"what":["forecast"],"scopes":["es"],"n_sim":20,"dry_run":true}'
        python job.py publish '{"what":["forecast"],"scopes":["es"],"backfill":{"from":"2026-10-01","to":"2026-10-07"}}'
        python job.py publish '{"what":["forecast"],"scopes":["es"],"backfill":{"from":"2026-10-01","to":"2026-10-07","replace":true}}'

    In Docker (JSON without spaces)::

        RUN_JOB='publish {"what":["forecast"],"scopes":["es"]}'
    """

    WHAT = ('forecast', 'analysis', 'event', 'backtest', 'study', 'parties', 'manifest', 'point', 'unpublish')
    CHUNKED = ('forecast', 'analysis', 'event', 'backtest', 'point', 'unpublish')
    sequence = {'publish': ['forecast', 'analysis', 'event', 'backtest', 'study', 'parties', 'manifest', 'point',
                            'unpublish']}
    default_action = 'publish'
    stop_on_error = False
    chunk_noun = 'scopes'

    def run(self, what=('forecast',), scopes=('es',), event_date=None, n_sim=1000, seed=42, drange=6, max_fc=10,
            alpha=0.05, correctors=None, freeze=None, run=None, dry_run=False, force=False, fs=None, today=None,
            backfill=None, source=None, study=None, register=False, verbose=1, stop_on_error=None, **kwargs):
        """
        Run the requested publication actions.

        Parameters
        ----------
        what : str or list of str, default ('forecast',)
            Actions to run, among ``forecast``, ``analysis``, ``event``, ``backtest``, ``study``, ``parties``,
            ``manifest``, ``point`` and ``unpublish``. They run in that order when several are given. ``parties``
            writes the party catalogue of the database next to the manifest (``publish.publish_parties``);
            ``forecast`` and ``analysis`` rewrite it too, once per run of the job, so the pages always colour and
            name every party.
        scopes : str or list of str, default ('es',)
            Scopes to act on, or ``'all'``. They are resolved (``publish.resolve_scopes``, which reads the scope
            catalogue) only when some action works scope by scope (`CHUNKED`); ``study``, ``parties`` and
            ``manifest`` alone run once and never read it.
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
            Publish inside the LOREG window anyway; with ``event``, publish a date even when the backtest tables
            are missing and the live event had the model slice or the series (which are then removed); with
            ``study`` and ``register``, replace an existing registration.
        fs : FileSystem, optional
            File system to write to. Defaults to ``self.app.fs``.
        today : str, optional
            Date (``YYYY-MM-DD``) taken as today by the LOREG guard, the backfill range check and
            the anchor of the average (the run is read that day, never later: ``meta.as_of``);
            for tests and rehearsals only (unlike ``force``, it leaves no mark in ``meta.freeze``,
            but a past ``today`` publishes a run read in the past).
        backfill : dict, optional
            Range ``{'from': 'YYYY-MM-DD', 'to': 'YYYY-MM-DD'}`` (``to`` defaults to ``from``
            and must be before today) of retrospective ``forecast`` runs, one per day; see
            ``publish.backfill_days``. Requires ``forecast`` in ``what``. With ``'replace': True``,
            a day whose run already exists is removed first (``publish.unpublish``; in a dry run,
            only its folder) and computed again instead of being skipped or failed.
        source : str, optional
            Results folder of ``backtest`` and of the model slice of ``event`` (default ``backtest/results`` of
            the repository).
        study : str, optional
            Slug of the study (one of ``bundle.STUDIES``). Required by ``study``.
        register : bool, default False
            Register the prediction of the study (``publish.publish_study``); with ``force``, replace an existing
            registration.
        verbose : int, default 1
            Verbosity of the console output: 0, only the header of the action and the final summary; 1, also
            ``[i/n] scope: empezando``, the result line and the model warnings of each scope as they happen;
            2, also the phase lines of the forecast; 3, also the own output of the model (``Simulator`` and
            ``Computer`` receive ``verbose - 2``).
        stop_on_error : bool, optional
            Stop at the first failed scope instead of going on with the others (the manifest entries of the
            scopes already published are still written); ``False`` by default.
        **kwargs
            Ignored extra job arguments.

        Returns
        -------
        dict[str, dict or list of dict]
            Result of each scope (of the last action that ran on it), with a ``status`` of
            ``published``, ``skipped``, ``refused`` or ``failed``; with ``backfill``, the
            ``forecast`` result of a scope is the list of the results of its days, each with
            its ``day`` and ``replaced`` (whether an existing run was removed first), and the
            ``event`` result is the list of the results of its dates, each with its ``event_date``.

        Raises
        ------
        ValueError
            If ``what`` is unknown, ``run`` is missing (or ``event_date`` for ``event``, or ``study`` for
            ``study``), ``event`` is
            combined with ``forecast``, or ``backfill`` is given without ``forecast`` in ``what``, its range is
            invalid or its ``replace`` is not a boolean, an action of `CHUNKED` has no scopes to run (``publish:
            no scopes to run``) or ``stop_on_error`` is not true or false.
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
        if 'study' in whats and not study:
            raise ValueError('publish: "study" is required for study')
        if 'event' in whats and 'forecast' in whats:
            raise ValueError('publish: "event" cannot be combined with "forecast"')
        if backfill is not None and 'forecast' not in whats:
            raise ValueError('publish: "backfill" needs "forecast" in what')
        days = None if backfill is None else publish.backfill_days(backfill, today)
        replace = False if backfill is None else backfill.get('replace', False)
        if not isinstance(replace, bool):
            raise ValueError('publish: backfill "replace" must be true or false')

        fs = fs if fs is not None else self.app.fs
        if fs is None:
            raise RuntimeError(
                'publish: no file system configured (set S3_BUCKET or run with a local files/ root)'
            )

        select = ['publish.' + w for w in self.WHAT
                  if w in whats or (w == 'parties' and ('forecast' in whats or 'analysis' in whats))]
        # only the actions that work scope by scope need the scope catalogue (and the database behind it)
        chunked = any(w in self.CHUNKED for w in whats)
        targets = publish.resolve_scopes(scopes) if chunked else []
        if chunked and not targets:
            raise ValueError('publish: no scopes to run')

        super().run(select=select, chunks=targets, stop_on_error=stop_on_error, fs=fs, dry_run=dry_run, today=today,
                    days=days, replace=replace, event_date=event_date, force=force, n_sim=n_sim, seed=seed,
                    drange=drange, max_fc=max_fc, alpha=alpha, correctors=correctors, freeze=freeze, run=run,
                    source=source, study=study, register=register, verbose=verbose)

        results = {}
        for (key, chunk), result in self.results.items():
            what = key.split('.', 1)[1]
            if what in self.CHUNKED:
                results[chunk] = result
            elif what == 'study':
                results[study] = result
        return results
