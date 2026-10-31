"""Tests del worker sin cola ni base de datos."""
import json
import types

import pytest

from mtpy.core.worker import Job, JobInterrupted, Pipeline, PipelineEnd, Worker


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


class Recorder:
    """Pushover falso: guarda los mensajes o rompe."""
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def send(self, msg, **kwargs):
        if self.fail:
            raise RuntimeError('pushover down')
        self.sent.append(msg)


def quiet_logger(infos=None, errors=None):
    return types.SimpleNamespace(info=(infos if infos is not None else []).append,
                                 error=(errors if errors is not None else []).append)


class Noop(Job):
    def run(self, **kwargs):
        pass


def test_alert_truncates_pushover_to_1024_and_logs_the_whole_text(fresh_app):
    ps, infos = Recorder(), []
    fresh_app.set('ps', ps)
    fresh_app.set('logger', quiet_logger(infos))
    text = '\n'.join('line {:03d} {}'.format(i, 'x' * 40) for i in range(40))
    job = Noop()
    job.alert(text)
    assert infos == [text] and len(ps.sent[0]) <= Job.ALERT_LIMIT and ps.sent[0].endswith('\n' + Job.TRUNCATED)
    assert ps.sent[0].startswith('line 000') and job.alerted
    job.alert('short')
    assert ps.sent[1] == 'short'


def test_alert_never_raises_when_pushover_fails(fresh_app):
    errors = []
    fresh_app.set('ps', Recorder(fail=True))
    fresh_app.set('logger', quiet_logger(errors=errors))
    Noop().alert('hi')
    assert len(errors) == 1 and 'pushover down' in errors[0]


def test_alert_without_logger_nor_pushover_is_silent(fresh_app):
    Noop().alert('hi')


def test_dispatch_alerts_when_run_raises_and_reraises(fresh_app):
    ps = Recorder()
    fresh_app.set('ps', ps)

    class Boom(Job):
        def run(self, **kwargs):
            raise ValueError('bad params')
    with pytest.raises(ValueError, match='bad params'):
        Boom().dispatch(x=1)
    assert ps.sent == ['Boom job failed: ValueError: bad params']


def test_dispatch_does_not_alert_twice(fresh_app):
    ps = Recorder()
    fresh_app.set('ps', ps)

    class Alerted(Job):
        def run(self, **kwargs):
            self.alert('summary')
            raise RuntimeError('failed')
    with pytest.raises(RuntimeError):
        Alerted().dispatch()
    assert ps.sent == ['summary']


def test_dispatch_names_an_interruption(fresh_app):
    ps = Recorder()
    fresh_app.set('ps', ps)

    class Stopped(Job):
        def run(self, **kwargs):
            raise JobInterrupted('SIGTERM')
    with pytest.raises(JobInterrupted):
        Stopped().dispatch()
    assert ps.sent == ['Stopped job interrupted: SIGTERM']


def test_worker_logs_a_failing_job_and_goes_on(fresh_app):
    errors = []
    fresh_app.set('logger', quiet_logger(errors=errors))

    class Boom(Job):
        def run(self, **kwargs):
            raise ValueError('x')
    worker = Worker.__new__(Worker)
    worker.app = fresh_app
    worker.namespace = types.SimpleNamespace(Boom=Boom)
    worker.handle_message(json.dumps({'job': 'boom', 'params': {}}))
    assert any('Boom' in e and 'x' in e for e in errors)
