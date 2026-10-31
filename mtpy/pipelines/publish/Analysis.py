from ...lib import bundle, publish
from ..base import ScopePipeline
from .Forecast import national_event


class Analysis(ScopePipeline):
    """
    Publish the houses analysis of each scope to the web bundle (``publish.publish_analysis``).

    The chunk is the scope. Each scope ends ``published``, ``skipped`` (``publish.NothingToPublish``), ``refused``
    (LOREG guard) or ``failed`` (any other error); a failure never raises. The progress of the scope is printed
    live and its result line is the summary.

    Attributes
    ----------
    writer : BundleWriter
        Bundle writer (under the dry-run prefix in a dry run).
    national : str or None
        Date of the next election of ``es``, for the LOREG guard of every scope.
    """

    label = 'analysis'
    pipes = {'publish': {'Setup': ['prepare'], 'Load': ['publish']}}

    def start(self, fs=None, dry_run=False, today=None, **kwargs) -> None:
        """
        Set the run: the writer and the date of the next election of ``es``.

        Parameters
        ----------
        fs : FileSystem
            File system to write to.
        dry_run : bool, default False
            Write under ``site-dry/v1``.
        today : str, optional
            Date taken as today by the LOREG guard.
        **kwargs
            ``verbose`` and ``position`` (see `ScopePipeline.start`); other arguments of the job are ignored.
        """
        super().start(**kwargs)
        self.writer = bundle.BundleWriter(fs, prefix=bundle.DRY_PREFIX if dry_run else bundle.PREFIX)
        self.national = national_event(today)

    def prepare_task(self, chunk=None, position=(1, 1), **kwargs):
        """Start the live progress of the scope."""
        self.start_progress(chunk, position)

    def publish_task(self, chunk=None, today=None, force=False, **kwargs):
        """
        Publish the houses analysis of one scope, converting every outcome into a result dict (``self.result``).

        Parameters
        ----------
        chunk : str
            Scope to publish.
        today : str or None
            Date taken as today by the LOREG guard.
        force : bool
            Publish inside the LOREG window anyway; the analysis carries no mark of it.

        Notes
        -----
        The result has a ``status``: ``published`` with ``event_date``, ``n_pollsters``, ``seconds`` and
        ``house_effects`` (whether they were published), ``skipped`` for ``publish.NothingToPublish``, ``refused``
        for the LOREG guard and ``failed`` for any other exception, all but ``published`` with a ``reason``.
        A scope without an upcoming event is still published (the analysis then works from its last election) and
        only the national window guards it. A scope whose current cycle has no polls is published without house
        effects (``meta.house_effects`` false).
        """
        scope = chunk
        try:
            publish.loreg_guard(scope, publish.next_event_date(scope), today, force, self.national)
            out = publish.publish_analysis(scope, self.writer, verbose=self.model_verbose)
        except publish.PublishRefused as e:
            self.result = {'status': 'refused', 'reason': str(e)}
        except publish.NothingToPublish as e:
            self.result = {'status': 'skipped', 'reason': str(e)}
        except Exception as e:
            self.result = self.failed(scope, e)
        else:
            self.result = {'status': 'published', 'event_date': out['event_date'],
                           'n_pollsters': out['n_pollsters'], 'seconds': out['seconds'],
                           'house_effects': out.get('house_effects', True)}

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line of one scope result."""
        if result['status'] == 'published':
            line = '{}: analysis published (event {}, {} pollsters) in {:.0f} s'.format(
                chunk, result['event_date'], result['n_pollsters'], result['seconds']['total'])
            return line if result.get('house_effects', True) else line + '; sin efectos de casa'
        return '{}: analysis {} ({})'.format(chunk, result['status'], result['reason'])
