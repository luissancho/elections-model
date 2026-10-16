"""
Live progress lines of the jobs that work scope by scope (`publish`, `compute`).

`ScopeReporter` is a mixin for a `Job`: `_say` prints one line now (and logs it) when the run is verbose enough,
`_progress` builds the `ScopeProgress` of one scope and `_log_error` logs the traceback being handled under the
job name. `ScopeProgress` prints the start, the phases, the warnings and the result of one scope with the prefix
``[i/n] scope[ day]``; `counted` writes a number with its noun.

Verbosity levels (``verbose`` of the job): 0, only the header of each action and the final summary; 1, also the
start, the result line and the warnings of each scope; 2, also the phase lines with their seconds; 3, also the own
output of the model (which receives ``verbose - 2``).
"""
import contextlib
import logging
import sys
import traceback
import warnings
from datetime import datetime
from typing import Callable, Optional


def counted(n: int, one: str, many: str) -> str:
    """Return ``n`` with the singular or plural noun (``'1 ámbito'``, ``'2 ámbitos'``)."""
    return '{} {}'.format(n, one if n == 1 else many)


@contextlib.contextmanager
def collect_warnings(notes: list, note: Optional[Callable] = None):
    """
    Collect every warning raised inside the block instead of letting it reach stderr.

    All categories are recorded (``DeprecationWarning`` included, the filters are set to ``always`` inside the
    block) and the previous filters and ``showwarning`` are restored on exit, also when the block raises.

    Parameters
    ----------
    notes : list of str
        Destination of the messages: each new message text is appended once, in order of appearance.
    note : callable, optional
        Called with the text of each new message as it is raised (live output, `ScopeProgress.note`).

    Yields
    ------
    list of str
        ``notes``.
    """
    def show(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if text not in notes:
            notes.append(text)
            if note is not None:
                note(text)

    with warnings.catch_warnings():
        warnings.simplefilter('always')
        warnings.showwarning = show
        yield notes


def notes_suffix(result: dict) -> str:
    """Return the `` · n avisos`` suffix of a result with model ``warnings``, or an empty string."""
    n = len(result.get('warnings') or ())
    if n == 0:
        return ''
    return ' · {} {}'.format(n, 'aviso' if n == 1 else 'avisos')


class ScopeProgress:
    """
    Live progress lines of one scope (and day) of a job action.

    Calling the object reports a phase (``progress(step, seconds)``); ``start``, ``note`` and ``done`` print the
    start of the scope, a model warning and its result line. Every line goes through ``say`` with the prefix
    ``[i/n] scope[ day]`` and its verbosity level: 1 for the start, the warnings and the result, 2 for the phases.

    Parameters
    ----------
    say : callable
        Printer of one progress line, ``say(text, level)`` (`ScopeReporter._say`).
    scope : str
        Scope being worked.
    index, total : int
        Position of the scope (from 1) and number of scopes of the run.
    day : str, optional
        Day of a retrospective run, added to the prefix.
    steps : dict, optional
        Labels of the phases (``{step: label}``); a phase without label prints its key.
    """

    def __init__(self, say, scope, index, total, day=None, steps=None):
        self.say = say
        self.steps = steps or {}
        self.position = '[{}/{}]'.format(index, total)
        self.prefix = '{} {}{}'.format(self.position, scope, '' if day is None else ' ' + day)

    def __call__(self, step, seconds):
        """Report the end of the phase ``step`` after ``seconds``."""
        self.say('{}: {} ({:.1f} s)'.format(self.prefix, self.steps.get(step, step), seconds), 2)

    def start(self):
        """Report that the scope (or day) starts."""
        self.say('{}: empezando'.format(self.prefix), 1)

    def note(self, text):
        """Report a model warning of the scope."""
        self.say('{} · aviso: {}'.format(self.prefix, text), 1)

    def done(self, line):
        """Report the summary line of the result of the scope."""
        self.say('{} {}'.format(self.position, line), 1)


class ScopeReporter:
    """
    Mixin of the jobs that report scope by scope: needs ``self.app`` (with ``logger``), ``self.name`` (the job
    class name, set by `Job`) and ``self.verbose``.
    """

    verbose = 1

    def _progress(self, scope, index, total, day=None, steps=None):
        """
        Build the live progress printer of one scope (and day).

        Parameters
        ----------
        scope : str
            Scope being worked.
        index, total : int
            Position of the scope (from 1) and number of scopes of the run.
        day : str, optional
            Day of a retrospective run, added to the prefix.
        steps : dict, optional
            Labels of the phases (see `ScopeProgress`).

        Returns
        -------
        ScopeProgress
            Callable ``progress(step, seconds)``, with ``start``, ``note`` and ``done`` for the other lines.
        """
        return ScopeProgress(self._say, scope, index, total, day, steps)

    def _say(self, text, level=0):
        """
        Report one progress line now, once on the console, if the run is verbose enough.

        The app logger, if there is one, gets the bare ``text`` (it adds its own timestamp). The line is also
        printed to stdout with ``[hh:mm:ss]`` and flushed, unless the logger already writes its ``info`` to the
        console (`_logger_echoes_console`), so the line never shows twice.

        Parameters
        ----------
        text : str
            Progress line, without time.
        level : int, default 0
            Minimum ``verbose`` of the run for the line to be reported (0: always).
        """
        if self.verbose < level:
            return
        logger = self.app.logger
        if logger is None or not self._logger_echoes_console(logger):
            print('[{}] {}'.format(datetime.now().strftime('%H:%M:%S'), text), flush=True)
        if logger is not None:
            logger.info(text)

    @staticmethod
    def _logger_echoes_console(logger) -> bool:
        """
        Tell whether ``logger.info`` already writes to the console.

        Parameters
        ----------
        logger : logging.Logger or object
            A ``logging.Logger`` or the app ``Log`` (its ``loggers['info']`` is checked); anything else never
            echoes.

        Returns
        -------
        bool
            Whether ``info`` is enabled and some handler of the logger, or of an ancestor reached by propagation,
            is a ``logging.StreamHandler`` (not a ``FileHandler``) writing to ``sys.stdout`` or ``sys.stderr``.
        """
        if not isinstance(logger, logging.Logger):
            loggers = getattr(logger, 'loggers', None)
            logger = loggers.get('info') if isinstance(loggers, dict) else None
        if not isinstance(logger, logging.Logger) or not logger.isEnabledFor(logging.INFO):
            return False
        consoles = (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
        current = logger
        while current is not None:
            for handler in current.handlers:
                if (isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
                        and handler.stream in consoles):
                    return True
            current = current.parent if current.propagate else None
        return False

    def _log_error(self, scope: str) -> None:
        """Log the traceback being handled through the app logger, if there is one, under the job name."""
        if self.app.logger is not None:
            self.app.logger.error('{} {}: {}'.format(self.name.lower(), scope, traceback.format_exc()))
