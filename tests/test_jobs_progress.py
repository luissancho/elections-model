"""Tests de `mtpy/jobs/progress.py`: las líneas de progreso por ámbito que comparten los jobs `publish` y `compute`."""
import logging
import re
import types

from mtpy.jobs.progress import ScopeProgress, ScopeReporter, counted


def test_counted_picks_the_singular_or_the_plural():
    assert counted(1, 'ámbito', 'ámbitos') == '1 ámbito' and counted(2, 'ámbito', 'ámbitos') == '2 ámbitos'


def test_scope_progress_prefixes_every_line_and_names_the_steps_it_knows():
    said = []
    progress = ScopeProgress(lambda text, level: said.append((text, level)), 'es', 1, 3, steps={'fit': 'promedio ajustado'})
    progress.start()
    progress('fit', 2.345)
    progress('other', 1.0)
    progress.note('sin sondeos nuevos')
    progress.done('es: published')
    assert said == [
        ('[1/3] es: empezando', 1),
        ('[1/3] es: promedio ajustado (2.3 s)', 2),
        ('[1/3] es: other (1.0 s)', 2),
        ('[1/3] es · aviso: sin sondeos nuevos', 1),
        ('[1/3] es: published', 1),
    ]
    with_day = ScopeProgress(lambda text, level: said.append((text, level)), 'es', 2, 3, day='2026-10-01')
    with_day.start()
    assert said[-1] == ('[2/3] es 2026-10-01: empezando', 1)


class Reporter(ScopeReporter):
    """Un job mínimo: nombre, app sin logger y verbosidad."""

    def __init__(self, logger=None, verbose=1):
        self.name = 'Compute'
        self.app = types.SimpleNamespace(logger=logger)
        self.verbose = verbose


def test_reporter_prints_only_the_lines_within_its_verbosity(capsys):
    reporter = Reporter(verbose=1)
    reporter._say('cabecera')
    reporter._say('ámbito', 1)
    reporter._say('fase', 2)
    lines = capsys.readouterr().out.splitlines()
    assert [re.sub(r'^\[\d\d:\d\d:\d\d\] ', '', line) for line in lines] == ['cabecera', 'ámbito']
    assert all(re.match(r'^\[\d\d:\d\d:\d\d\] ', line) for line in lines)


def test_reporter_logs_the_traceback_under_the_job_name():
    records = []
    logger = types.SimpleNamespace(error=lambda text: records.append(text), info=lambda text: None)
    reporter = Reporter(logger=logger)
    try:
        raise ValueError('boom')
    except ValueError:
        reporter._log_error('es-md')
    assert len(records) == 1 and records[0].startswith('compute es-md: ') and 'ValueError: boom' in records[0]


def test_reporter_does_not_print_twice_when_the_logger_echoes_the_console(capsys):
    logger = logging.getLogger('test_jobs_progress.echo')
    logger.handlers = [logging.StreamHandler()]  # stderr
    logger.setLevel(logging.INFO)
    logger.propagate = False
    reporter = Reporter(logger=logger)
    reporter._say('una vez')
    out, err = capsys.readouterr()
    assert out == '' and err.strip().endswith('una vez')
    assert ScopeReporter._logger_echoes_console(logger) is True
    assert ScopeReporter._logger_echoes_console(object()) is False
