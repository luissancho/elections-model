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
        poll (the polls read that the database did not have: saved or, in a dry run, to save; `replaced_line` for
        one that takes the place of a poll of the database), per poll that the article gives differently
        (`changed_line`) and per poll that it no longer has (`removed_line`). The summary line counts the last two
        (``· 1 changed, 1 removed``) and only the first `MAX_POLLS` of each have their line, followed by
        ``es: ~ … y 3 sondeos más``.
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
        changed, removed = result.get('changed') or [], result.get('removed') or []
        nouns += ['{} {}'.format(len(polls), noun) for polls, noun in ((changed, 'changed'), (removed, 'removed'))
                  if polls]
        if nouns:
            line += ' · ' + ', '.join(nouns)
        line += notes_suffix(result)

        # A new poll that replaces one of the database is listed once, with the poll it replaces
        replaced = {poll_label(poll): poll for poll in result.get('replaced') or []}
        lines = []
        for poll in result.get('new') or []:
            substitute = replaced.pop(poll_label(poll), None)
            lines.append(poll_line(chunk, poll) if substitute is None else replaced_line(chunk, substitute))
        lines += [replaced_line(chunk, poll) for poll in replaced.values()]
        # The alert is short: beyond `MAX_POLLS` of each kind only the count is given
        for polls, mark, liner in ((changed, '~', changed_line), (removed, '-', removed_line)):
            lines += [liner(chunk, poll) for poll in polls[:MAX_POLLS]]
            if len(polls) > MAX_POLLS:
                lines.append('{}: {} … y {} sondeos más'.format(chunk, mark, len(polls) - MAX_POLLS))
        if not lines:
            return line
        return [line] + lines


# Names of the poll fields in the detail of a changed poll (`change_text`)
FIELD_NAMES = {'start_date': 'inicio', 'sample_size': 'muestra', 'ctype': 'tipo'}
# How a missing value is written in the detail of a changed poll
NULL = '—'
# Most differences written in the line of a changed poll and most changed (or removed) polls listed for a scope:
# the rest are only counted, so the lines of the load do not fill the alert
MAX_CHANGES = 6
MAX_POLLS = 10


def poll_dates(poll):
    """
    Fieldwork of a poll: ``'2026-11-10 → 2026-11-12'``, or one date when both are the same day or the start is
    unknown.

    Parameters
    ----------
    poll : dict
        With ``start_date`` and ``end_date`` (``YYYY-MM-DD`` or ``None``).

    Returns
    -------
    str
        The dates.
    """
    start, end = poll['start_date'], poll['end_date']
    return end if not start or start == end else '{} → {}'.format(start, end or '?')


def poll_label(poll):
    """
    Name of a poll in the lines of the load: ``'40dB / El País, 2026-11-10 → 2026-11-12'``.

    Parameters
    ----------
    poll : dict
        ``pollster``, ``sponsor`` (``None`` without one), ``start_date`` and ``end_date`` (`load.new_poll_list`).

    Returns
    -------
    str
        The pollster, the sponsor after ``/`` when there is one and the fieldwork dates (`poll_dates`).
    """
    name = poll['pollster'] or '?'
    if poll['sponsor']:
        name += ' / ' + poll['sponsor']
    return '{}, {}'.format(name, poll_dates(poll))


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
        The scope, ``+`` and the label of the poll (`poll_label`).
    """
    return '{}: + {}'.format(scope, poll_label(poll))


def replaced_line(scope, poll):
    """
    Line of a new poll that takes the place of one of the database:
    ``'es: + 40dB / Prisa, 2026-11-10 → 2026-11-14 (sustituye a 2026-11-10 → 2026-11-12)'``.

    Parameters
    ----------
    scope : str
        Scope of the poll.
    poll : dict
        A poll of `load.poll_changes` ``replaced``: the new poll with ``old``, the one it replaces.

    Returns
    -------
    str
        The line of the new poll and, in brackets, the fieldwork of the old one, or its whole label when the
        pollster or the sponsor are not the same.
    """
    old = poll['old']
    same_names = (old['pollster'], old['sponsor']) == (poll['pollster'], poll['sponsor'])
    return '{} (sustituye a {})'.format(poll_line(scope, poll), poll_dates(old) if same_names else poll_label(old))


def removed_line(scope, poll):
    """
    Line of a poll of the database that the article no longer has: ``'es: - 40dB / Prisa, 2026-11-10 → 2026-11-14'``.

    Parameters
    ----------
    scope : str
        Scope of the poll.
    poll : dict
        A poll of `load.poll_changes` ``removed``.

    Returns
    -------
    str
        The scope, ``-`` and the label of the poll.
    """
    return '{}: - {}'.format(scope, poll_label(poll))


def changed_line(scope, poll):
    """
    Line of a poll that the article gives differently from the database:
    ``'es: ~ 40dB / Prisa, 2026-11-10 → 2026-11-14: muestra — → 2000; + SALF 1,8'``.

    Parameters
    ----------
    scope : str
        Scope of the poll.
    poll : dict
        A poll of `load.poll_changes` ``changed``, with its ``changes``.

    Returns
    -------
    str
        The scope, ``~``, the label of the poll and its differences (`change_text`) separated by ``; ``: the
        first `MAX_CHANGES` and, if there are more, ``… y 3 más``.
    """
    details = [change_text(change) for change in poll['changes'][:MAX_CHANGES]]
    if len(poll['changes']) > MAX_CHANGES:
        details.append('… y {} más'.format(len(poll['changes']) - MAX_CHANGES))
    return '{}: ~ {}: {}'.format(scope, poll_label(poll), '; '.join(details))


def share_text(value):
    """A share with a decimal comma and no trailing zeros (``31``, ``31,4``, ``1,85``), or `NULL`."""
    if value is None:
        return NULL
    return '{:.2f}'.format(value).rstrip('0').rstrip('.').replace('.', ',')


def seats_text(seats):
    """A seats range ``[min, max]`` as ``125–136``, one number when both are the same, or `NULL`."""
    if seats is None:
        return NULL
    low, high = (NULL if value is None else str(value) for value in seats)
    return low if low == high else '{}–{}'.format(low, high)


def change_text(change):
    """
    One difference of a changed poll, as it is written in its line.

    Parameters
    ----------
    change : dict
        ``field``, ``party``, ``old`` and ``new`` (`load.poll_changes`).

    Returns
    -------
    str
        ``muestra a → b``, ``inicio a → b`` or ``tipo a → b`` for a field of the poll; ``+ PARTY 1,8`` for a party
        added and ``- PARTY`` for one dropped; ``PARTY 31 → 31,4`` for a share and ``PARTY escaños a → b`` for the
        seats. A missing value is `NULL`.
    """
    field, party, old, new = change['field'], change['party'], change['old'], change['new']
    if field == 'party':
        return '- {}'.format(party) if new is None else '+ {} {}'.format(party, share_text(new))
    if field == 'pct':
        return '{} {} → {}'.format(party, share_text(old), share_text(new))
    if field == 'seats':
        return '{} escaños {} → {}'.format(party, seats_text(old), seats_text(new))
    old, new = (NULL if value is None else value for value in (old, new))
    return '{} {} → {}'.format(FIELD_NAMES.get(field, field), old, new)
