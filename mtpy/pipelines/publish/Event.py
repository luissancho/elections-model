from ...lib import bundle, publish
from ..base import ScopePipeline


class Event(ScopePipeline):
    """
    Publish the past elections of each scope to the web bundle (``publish.publish_event``).

    The chunk is the scope and the result of a scope is the list of the results of its dates, each with its
    ``event_date``. The dates come from ``event_date`` (a date, a list of dates or ``'all'``, see
    ``publish.event_dates_to_publish``); when they cannot be resolved, or there are none, the scope ends in the
    setup stage with a single result. Each date ends ``published``, ``skipped`` (``publish.NothingToPublish``) or
    ``failed`` (any other error, an invalid date included); a failure never raises.

    Attributes
    ----------
    writer : BundleWriter
        Bundle writer (under the dry-run prefix in a dry run).
    days : list of str
        Dates of the scope, resolved by `resolve_dates_task`.
    """

    label = 'event'
    pipes = {'publish': {'Setup': ['resolve_dates'], 'Load': ['publish']}}

    def start(self, fs=None, dry_run=False, **kwargs) -> None:
        """Set the run and create the bundle writer (under the dry-run prefix in a dry run)."""
        super().start(**kwargs)
        self.writer = bundle.BundleWriter(fs, prefix=bundle.DRY_PREFIX if dry_run else bundle.PREFIX)

    def resolve_dates_task(self, chunk=None, position=(1, 1), event_date=None, today=None, **kwargs):
        """
        Start the live progress of the scope and resolve its dates (``self.days``).

        Parameters
        ----------
        chunk : str
            Scope to publish.
        position : tuple of int, default (1, 1)
            Position of the scope (from 1) and number of scopes, for the live progress lines.
        event_date : str or list of str
            Date, list of dates or ``'all'``.
        today : str or None
            Date taken as today to resolve ``'all'``.

        Notes
        -----
        An invalid date, a failure resolving ``'all'`` or no past election with official results leave the scope
        with a single ``failed`` or ``skipped`` result (in a list, with the ``event_date`` asked) and `publish_task`
        does nothing. (`end` is not used: it logs its result as a dict and this one is a list.)
        """
        self.start_progress(chunk, position)
        try:
            self.days = publish.event_dates_to_publish(chunk, event_date, today=today)
        except Exception as e:
            self.result = [self.failed(chunk, e, event_date=str(event_date))]
            return
        if not self.days:
            self.result = [{'status': 'skipped', 'event_date': str(event_date),
                            'reason': 'no past elections with official results'}]

    def publish_task(self, chunk=None, source=None, **kwargs):
        """
        Publish the past elections of one scope, converting every outcome into a result dict.

        Parameters
        ----------
        chunk : str
            Scope to publish.
        source : str or None
            Results folder of the backtest (see ``publish.backtest_source``).

        Notes
        -----
        ``self.result`` is a list with one result per date, each with its ``event_date`` and a ``status``:
        ``published`` with ``n_polls``, ``n_pollsters``, ``has_model`` and ``seconds``, ``skipped`` for
        ``publish.NothingToPublish`` and ``failed`` for any other exception, both with a ``reason``.
        """
        if self.result is not None:
            return
        outcomes = []
        for day in self.days:
            try:
                out = publish.publish_event(chunk, day, self.writer, source=source, verbose=self.model_verbose)
            except publish.NothingToPublish as e:
                outcomes.append({'status': 'skipped', 'event_date': day, 'reason': str(e)})
            except Exception as e:
                outcomes.append(self.failed(chunk, e, event_date=day))
            else:
                outcomes.append({'status': 'published', 'event_date': day, 'n_polls': out['n_polls'],
                                 'n_pollsters': out['n_pollsters'], 'has_model': out['has_model'],
                                 'seconds': out['seconds']})
        self.result = outcomes

    def summary(self, chunk, **kwargs):
        """Build the summary lines of the scope, one per date, and report each one on the progress line."""
        lines = [self.line_of(chunk, result) for result in self.result]
        for line in lines:
            self.progress.done(line)
        return lines

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line of one date result."""
        if result['status'] == 'published':
            return '{}: event {} published ({} polls, {} pollsters) in {:.0f} s'.format(
                chunk, result['event_date'], result['n_polls'], result['n_pollsters'], result['seconds']['total'])
        return '{}: event {} {} ({})'.format(chunk, result['event_date'], result['status'], result['reason'])
