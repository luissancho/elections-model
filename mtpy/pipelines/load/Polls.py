from ...jobs.progress import collect_warnings, counted, notes_suffix
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
        self.progress = self.progress_of(chunk, position)
        self.progress.start()
        try:
            self.event = event_date or next_event_date(chunk)
        except Exception as e:
            self.end({**self.failed(chunk, e), 'action': 'polls', 'warnings': []})

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
        notes = []
        try:
            with collect_warnings(notes, self.progress.note):
                result = load.load_polls(chunk, self.event, overwrite=self._action == 'rebulk', force=force,
                                         years=years, save=not dry_run, verbose=self.model_verbose,
                                         progress=self.progress,
                                         report=lambda text, level=1: self._say(text, level), loader=loader,
                                         computer=computer, catalogues=catalogues)
        except Exception as e:
            result = {**self.failed(chunk, e), 'action': 'polls'}

        if skip_missing and result['status'] == 'failed' and result['reason'] == load.NO_POLLS_ARTICLE.format(
                self.event):
            result['status'] = 'skipped'
        result['warnings'] = notes
        self.result = result

    def line_of(self, chunk, result, **kwargs):
        """Format the summary line of one scope result, with the count of its warnings."""
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
        return line + notes_suffix(result)
