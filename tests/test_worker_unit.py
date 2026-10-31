"""Tests del worker sin cola ni base de datos."""
import types

import pytest

from mtpy.core.worker import Pipeline, PipelineEnd, Worker


def test_job_class_unwraps_a_module_bound_on_the_namespace():
    """Como `Router.controller_class`: si el paquete `jobs` tiene enlazado el submódulo en vez de la clase."""
    class Publish:
        pass

    module = types.ModuleType('mtpy.jobs.Publish')
    module.Publish = Publish
    worker = Worker.__new__(Worker)
    worker.namespace = types.SimpleNamespace(Publish=module, Other=Publish)
    assert worker.job_class('Publish') is Publish and worker.job_class('Other') is Publish


class Counter(Pipeline):
    """Pipeline sin modelos: anota las tareas, termina antes de tiempo en el chunk `skip` y rompe en `boom`."""
    label = 'count'

    def __init__(self):
        super().__init__()
        self.pipes = {'update': {'Setup': ['check'], 'Load': ['work'], 'Set': ['close']}}
        self.calls, self.started, self.finished, self.kwargs = [], 0, 0, None

    def start(self, **kwargs):
        self.started += 1
        self.kwargs = kwargs

    def check_task(self, chunk=None, **kwargs):
        self.calls.append(('check', chunk))
        if chunk == 'skip':
            self.end({'status': 'skipped', 'reason': 'nothing'})

    def work_task(self, chunk=None, **kwargs):
        self.calls.append(('work', chunk))
        if chunk == 'boom':
            raise KeyError('x')
        self.result = {'status': 'failed' if chunk == 'bad' else 'done', 'chunk': chunk}

    def close_task(self, chunk=None, **kwargs):
        self.calls.append(('close', chunk))

    def summary(self, chunk, **kwargs):
        return '{}: {}'.format(chunk, self.result['status'])

    def finish(self, **kwargs):
        self.finished += 1
        self.line = 'closed'


def test_pipeline_runs_without_config_nor_logger(fresh_app):
    p = Counter()
    p.run('update', chunk='a', position=(1, 2), extra=1)
    assert p.compression is None and p.result == {'status': 'done', 'chunk': 'a'} and p.line == 'a: done'
    assert p.calls == [('check', 'a'), ('work', 'a'), ('close', 'a')]


def test_pipeline_starts_once_with_the_kwargs_of_the_first_run_but_the_chunk(fresh_app):
    p = Counter()
    p.run('update', chunk='a', position=(1, 2), extra=1)
    p.run('update', chunk='b', position=(2, 2), extra=1)
    assert p.started == 1 and p.kwargs == {'position': (1, 2), 'extra': 1}


def test_pipeline_end_keeps_the_result_and_skips_the_remaining_stages(fresh_app):
    p = Counter()
    p.run('update', chunk='skip')
    assert p.result == {'status': 'skipped', 'reason': 'nothing'} and p.line == 'skip: skipped'
    assert p.calls == [('check', 'skip')]


def test_pipeline_resets_result_and_line_on_each_run(fresh_app):
    p = Counter()
    p.run('update', chunk='a')
    with pytest.raises(KeyError):
        p.run('update', chunk='boom')
    assert p.result is None and p.line is None


def test_pipeline_keeps_a_line_set_by_a_task(fresh_app):
    class Lined(Counter):
        def close_task(self, chunk=None, **kwargs):
            self.line = 'mine'
    p = Lined()
    p.run('update', chunk='a')
    assert p.line == 'mine'


def test_failure_name_uses_the_label(fresh_app):
    p = Counter()
    assert p.failure_name('es') == 'es (count)' and p.failure_name(None) == 'count'
    p.label = None
    assert p.failure_name('es') == 'es' and p.failure_name(None) == p.name


def test_default_hooks_do_nothing(fresh_app):
    class Plain(Pipeline):
        def __init__(self):
            super().__init__()
            self.pipes = {'update': {'Setup': []}}
    p = Plain()
    p.run('update')
    assert p.result is None and p.line is None and p.finish() is None and p.chunked is True
