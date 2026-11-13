from ...jobs.progress import counted, notes_suffix
from ...lib import load
from ...lib.publish.common import next_event_date
from ..base import ScopePipeline


# Labels of the phases reported at verbose 2 (keys of `load.LOAD_STEPS` and `load.PARAMS_STEPS`)
STEPS = {'read': 'lectura', 'names': 'nombres', 'series': 'series', 'save': 'guardado', 'weights': 'pesos',
         'derive': 'derivación'}


class Polls(ScopePipeline):
    """
    Load the Wikipedia polls of the next election of a scope (`load.load_polls`).

    Actions: ``update`` saves only the new polls; ``rebulk`` replaces the whole cycle (overwrite). The chunk is the
    scope; every outcome is a result dict (see `load_task`) and a failure never raises.
    """

    label = 'polls'
    steps = STEPS
    pipes = {
        'update': {'Setup': ['resolve_event'], 'Load': ['load']},
        'rebulk': {'Setup': ['resolve_event'], 'Load': ['load']},
    }

    def start(self, force=False, dry_run=False, **kwargs) -> None:
        """Set the run and print the header ``load polls: n ámbitos[ · overwrite][ · force][ · dry run]``."""
        super().start(**kwargs)
        flags = ''.join(text for active, text in ((self._action == 'rebulk', ' · overwrite'), (force, ' · force'),
                                                   (dry_run, ' · dry run')) if active)
        self._say('load polls: {}{}'.format(counted(self.total, 'ámbito', 'ámbitos'), flags))

    def resolve_event_task(self, chunk=None, position=(1, 1), event_date=None, skip_missing=False, **kwargs):
        """
        Start the progress of the scope and find its election day.

        Parameters
        ----------
        chunk : str
            Scope to load.
        position : tuple of int
            ``(index, total)`` of the scope.
        event_date : str, optional
            Election day given; the next election of the scope by default (``self.event``).
        skip_missing : bool, default False
            Turn a scope without an upcoming event into ``skipped`` instead of ``failed``.
        """
        self.start_progress(chunk, position)
        try:
            self.event = event_date or next_event_date(chunk)
        except Exception as e:
            self.end(self.failed(chunk, e, action='polls', warnings=[]))

        if self.event is None:
            self.end({'status': 'skipped' if skip_missing else 'failed', 'action': 'polls',
                      'reason': 'no upcoming event', 'warnings': []})

    def load_task(self, chunk=None, force=False, years=None, dry_run=False, loader=None, computer=None,
                  catalogues=None, skip_missing=False, **kwargs):
        """
        Load the polls of one scope, converting every outcome into a result dict (``self.result``).

        Parameters
        ----------
        chunk : str
            Scope to load.
        force, years, dry_run
            Options of the job (see `mtpy.jobs.Load.Load.run`); overwrite is the ``rebulk`` action.
        loader, computer, catalogues : callable or None
            Factories and helper passed to `load.load_polls`.
        skip_missing : bool, default False
            Turn a scope without a polls article into ``skipped`` instead of ``failed``.

        Notes
        -----
        The result is that of `load.load_polls` or ``{'status': 'failed' | 'skipped', 'action': 'polls',
        'reason'}``, with the ``warnings`` raised.
        """
        result = self.collect(chunk, lambda: load.load_polls(
            chunk, self.event, overwrite=self._action == 'rebulk', force=force, years=years, save=not dry_run,
            verbose=self.model_verbose, progress=self.progress, report=lambda text, level=1: self._say(text, level),
            loader=loader, computer=computer, catalogues=catalogues), action='polls')

        if skip_missing and result['status'] == 'failed' and result['reason'] == load.NO_POLLS_ARTICLE.format(
                self.event):
            result['status'] = 'skipped'
        self.result = result

    def summary(self, chunk, **kwargs):
        """
        Build the summary lines of the scope and report them: the result line on the progress line and, below it,
        one line per new poll (`poll_line`), so that they reach the final summary and the alert too.
        """
        lines = self.line_of(chunk, self.result, **kwargs)
        if isinstance(lines, str):
            lines = [lines]
        if self.progress is not None:
            self.progress.done(lines[0])
            for line in lines[1:]:
                self._say(line, 1)
        return lines[0] if len(lines) == 1 else lines

    def line_of(self, chunk, result, **kwargs):
        """
        Format the summary line of one scope result, with the count of its warnings, followed by one line per new
        poll (the polls read that the database did not have: saved or, in a dry run, to save).
        """
        if result['status'] in ('failed', 'skipped'):
            return '{}: polls {} ({})'.format(chunk, result['status'], result['reason']) + notes_suffix(result)

        added = result['added']
        pollsters, sponsors = len(added['pollsters']), len(added['sponsors'])
        seconds = result['seconds']['total']
        if result['status'] == 'loaded':
            line = '{}: polls loaded ({} read, {} saved, {}{}) in {:.0f} s'.format(
                chunk, result['n_read'], result['n_saved'], result['event_date'],
                ', overwrite' if result['overwrite'] else '', seconds)
            verb = 'added'
        else:
            line = '{}: polls checked ({} read, {} new, {}{}, dry run) in {:.0f} s'.format(
                chunk, result['n_read'], result['n_saved'], result['event_date'],
                ', overwrite' if result['overwrite'] else '', seconds)
            verb = 'to add'
        nouns = [counted(n, one, many) + ' ' + verb for n, one, many in
                 ((pollsters, 'pollster', 'pollsters'), (sponsors, 'sponsor', 'sponsors')) if n]
        if nouns:
            line += ' · ' + ', '.join(nouns)
        line += notes_suffix(result)
        polls = result.get('new') or []
        if not polls:
            return line
        return [line] + [poll_line(chunk, poll) for poll in polls]


def poll_line(scope, poll):
    """
    Line of one new poll: ``'es: + 40dB / El País, 2026-11-10 → 2026-11-12'``.

    Parameters
    ----------
    scope : str
        Scope of the poll.
    poll : dict
        ``pollster``, ``sponsor`` (``None`` without one), ``start_date`` and ``end_date`` (`load.new_poll_list`).

    Returns
    -------
    str
        The pollster, the sponsor after ``/`` when there is one and the fieldwork dates (one when they are the
        same day or the start is unknown).
    """
    name = poll['pollster'] or '?'
    if poll['sponsor']:
        name += ' / ' + poll['sponsor']
    start, end = poll['start_date'], poll['end_date']
    dates = end if not start or start == end else '{} → {}'.format(start, end or '?')
    return '{}: + {}, {}'.format(scope, name, dates)
