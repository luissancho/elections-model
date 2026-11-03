from ..core.worker import SequenceJob
from ..lib.load import YEARS_WITH_OVERWRITE
from ..lib.publish import resolve_scopes


class Update(SequenceJob):
    """
    Update the model of the requested scopes in one task: load the new Wikipedia polls, compute the series and publish
    the forecast, the analysis and the party catalogue (``load``, ``compute`` and ``publish`` chained, in that order,
    each group on every scope before the next one starts).

    The groups run ``load.polls``, ``compute.compute`` and ``publish.forecast``, ``publish.analysis`` and
    ``publish.parties`` (once, after the scopes), with the same pipelines as the jobs of those names, so every result
    line, status and level of ``verbose`` is the one of the job it comes from. The compute is skipped on the scopes with
    nothing loaded since their last compute, read from the marks ``loaded_at`` and ``computed_at`` of ``events``
    (``compute.pending_events``): a scope whose load found no new poll ends ``skipped`` in the compute and goes on to
    the publication. ``overwrite`` is the only switch that redoes everything: it replaces the whole cycle of polls
    (``load.rebulk``) and computes the series whether or not anything was loaded; use it after a change of code or
    parameters. Right after the ``ALTER TABLE`` that adds the marks every election counts as pending, so the first
    ``update`` without ``overwrite`` computes everything once.

    Unlike ``load``, ``compute`` and ``publish``, ``update`` stops at the first failure by default: a scope that fails
    in a group stops the whole job before the next group, so a forecast is never published on polls that did not load
    or series that did not compute. With ``stop_on_error`` false every group runs on every scope, the job publishes
    what it has and it exits with an error at the end if anything failed. The alert is sent once, with the summary of
    the whole chain.

    ``force`` reaches the load and the publication at once: in ``update`` it is the publication inside the veda (the
    LOREG freeze) and, in the load, the addition of the unmapped pollsters and sponsors to the catalogues. To add new
    pollsters, run ``load`` with ``force`` on its own first. ``event_date`` is not accepted (it would force the
    election of the forecast and of the load alike; use ``load`` or ``publish``), and ``years`` cannot be combined with
    ``overwrite``. With ``dry_run`` nothing is written: the load reads and checks, the compute stops after the
    deviations and the publication goes under ``site-dry/``. Scopes ``'all'`` skip the scopes without a polls
    article, as ``load`` does.

    Examples
    --------
    From the command line::

        python job.py update '{"scopes":["es"],"n_sim":10000,"verbose":3}'
        python job.py update '{"scopes":"all"}'
        python job.py update '{"scopes":["es"],"overwrite":true}'
        python job.py update '{"scopes":["es"],"n_sim":20,"dry_run":true}'

    In Docker (JSON without spaces)::

        RUN_JOB='update {"scopes":["es"],"n_sim":10000}'
    """

    sequence = {'load': ['polls'], 'compute': ['compute'], 'publish': ['forecast', 'analysis', 'parties']}
    default_action = 'update'
    stop_on_error = True
    chunk_noun = 'scopes'

    def run(self, scopes=('es',), overwrite=False, dry_run=False, verbose=1, fs=None, stop_on_error=None, **kwargs):
        """
        Run the chain load, compute and publish.

        Parameters
        ----------
        scopes : str or list of str, default ('es',)
            Scopes to update, or ``'all'`` (see ``publish.resolve_scopes``).
        overwrite : bool, default False
            Replace the whole cycle of polls and compute the series even where nothing was loaded.
        dry_run : bool, default False
            Write nothing: the load only reads and checks and the publication goes under ``site-dry/``.
        verbose : int, default 1
            Verbosity of the console output, as in ``publish`` (0 to 3).
        fs : FileSystem, optional
            File system of the web bundle; the one of the app by default.
        stop_on_error : bool, optional
            Stop at the first failure; ``True`` by default (``False`` goes on and publishes what it has).
        **kwargs
            Arguments of the pipelines: ``force``, ``years``, ``n_sim``, ``seed``, ``drange``, ``max_fc``, ``alpha``,
            ``today``, ``correctors`` and the factories of the tests (``loader``, ``computer``, ``catalogues``,
            ``featured``, ``pending``).

        Returns
        -------
        dict
            ``{scope: {'polls': ..., 'compute': ..., 'forecast': ..., 'analysis': ...}, 'parties': ...}``: the result
            of each pipeline on each scope, and the one of the party catalogue.

        Raises
        ------
        ValueError
            If ``event_date`` is given, ``years`` is combined with ``overwrite`` or there are no scopes to run.
        RuntimeError
            If there is no file system, or if any scope failed (raised after the summary and the alert).
        """
        if 'event_date' in kwargs:
            raise ValueError('update: "event_date" is not accepted (use load)')

        years = kwargs.get('years')
        if years:
            if overwrite:
                raise ValueError(YEARS_WITH_OVERWRITE)
            kwargs['years'] = [str(year) for year in years]

        fs = fs if fs is not None else self.app.fs
        if fs is None:
            raise RuntimeError('update: no file system configured (set S3_BUCKET or run with a local files/ root)')

        targets = resolve_scopes(scopes)
        if not targets:
            raise ValueError('update: no scopes to run')

        super().run(select=None, action='rebulk' if overwrite else 'update', chunks=targets,
                    stop_on_error=stop_on_error, overwrite=overwrite, dry_run=dry_run, verbose=verbose, fs=fs,
                    skip_missing=scopes == 'all', **kwargs)

        results = {}
        for (key, chunk), result in self.results.items():
            name = key.split('.', 1)[1]
            if chunk is None:
                results['parties'] = result
            else:
                results.setdefault(chunk, {})[name] = result
        return results
