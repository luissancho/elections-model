"""Tests del worker sin cola ni base de datos."""
import json
import types

import pytest

from mtpy.core.worker import Job, JobInterrupted, Pipeline, PipelineEnd, SequenceJob, Worker


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


class Alpha(Counter):
    def work_task(self, chunk=None, **kwargs):
        if chunk == 'stop':
            self.calls.append(('work', chunk))
            raise JobInterrupted('SIGTERM')
        if chunk == 'many':
            self.result = [{'status': 'done'}, {'status': 'failed', 'reason': 'second'}]
            return
        super().work_task(chunk=chunk, **kwargs)

    def summary(self, chunk, **kwargs):
        if isinstance(self.result, list):
            return ['{}: {}'.format(chunk, r['status']) for r in self.result]
        return super().summary(chunk, **kwargs)


class Beta(Counter):
    chunked = False
    label = 'beta'

    def finish(self, **kwargs):
        self.finished += 1
        if kwargs.get('break_finish'):
            self.line = 'beta: finish failed (boom)'
            return 'beta-close'
        self.line = 'beta closed'


class Gamma(Counter):
    label = None

    def start(self, **kwargs):
        if kwargs.get('break_start'):
            raise RuntimeError('no fs')
        super().start(**kwargs)


class Seq(SequenceJob):
    sequence = {'grp': ['alpha', 'beta'], 'other': ['gamma']}
    stop_on_error = False
    chunk_noun = 'scopes'

    def __init__(self):
        super().__init__()
        self.made = []
        make = lambda cls: (lambda: self.made.append(cls()) or self.made[-1])
        self.namespace = types.SimpleNamespace(**{'grp.Alpha': make(Alpha), 'grp.Beta': make(Beta),
                                                  'other.Gamma': make(Gamma)})


def test_select_accepts_groups_and_group_dot_pipeline_in_sequence_order(fresh_app):
    job = Seq()
    assert job.select_pipelines('other,grp.beta') == [('grp', 'beta', 'grp.Beta'), ('other', 'gamma', 'other.Gamma')]
    assert [p for _, p, _ in job.select_pipelines(None)] == ['alpha', 'beta', 'gamma']
    assert job.select_pipelines(['grp']) == [('grp', 'alpha', 'grp.Alpha'), ('grp', 'beta', 'grp.Beta')]
    with pytest.raises(ValueError, match='seq: unknown select "grp.nope"'):
        job.select_pipelines('grp.nope')


def test_continue_mode_runs_every_chunk_and_reports_at_the_end(fresh_app, capsys):
    ps, infos = Recorder(), []
    fresh_app.set('ps', ps)
    fresh_app.set('logger', quiet_logger(infos))
    job = Seq()
    with pytest.raises(RuntimeError, match=r'seq: failed scopes: bad \(count\), boom \(count\)$'):
        job.run(select='grp', chunks=['a', 'bad', 'boom', 'c'])
    alpha, beta = job.made
    assert [c for _, c in alpha.calls if _ == 'work'] == ['a', 'bad', 'boom', 'c']
    assert beta.calls == [('check', None), ('work', None), ('close', None)] and beta.kwargs['position'] == (1, 1)
    assert alpha.finished == beta.finished == 1
    assert job.results[('grp.alpha', 'a')] == {'status': 'done', 'chunk': 'a'}
    assert job.results[('grp.alpha', 'boom')] == {'status': 'failed', 'reason': "KeyError: 'x'"}
    assert ('grp.beta', None) in job.results
    assert job.lines[:5] == ['a: done', 'bad: failed', "boom: failed (KeyError: 'x')", 'c: done', 'closed']
    text = ps.sent[-1].splitlines()
    assert text[0].startswith('Seq job executed in ') and text[0].endswith(' with 2 failures')
    assert text[1:3] == ['bad: failed', "boom: failed (KeyError: 'x')"] and text[3] == 'a: done'
    out = capsys.readouterr().out
    assert 'bad: failed' in out and out.index('a: done') < out.index('bad: failed')  # verbose falso: se imprime


def test_every_pipeline_of_the_plan_gets_every_chunk_with_its_position(fresh_app):
    job = Seq()
    job.run(chunks=['a', 'c'])
    alpha, beta, gamma = job.made
    assert [c for _, c in gamma.calls if _ == 'work'] == ['a', 'c'] and gamma.kwargs['position'] == (1, 2)
    assert job.results[('other.gamma', 'c')]['status'] == 'done' and gamma.failure_name('c') == 'c'


def test_summary_is_not_printed_when_the_logger_echoes_the_console(fresh_app, capsys):
    fresh_app.set('verbose', 1)
    Seq().run(select='grp.alpha', chunks=['a'])
    assert 'a: done' not in capsys.readouterr().out


def test_stop_mode_finishes_the_pipeline_alerts_and_reraises(fresh_app):
    ps = Recorder()
    fresh_app.set('ps', ps)
    job = Seq()
    with pytest.raises(KeyError):
        job.run(chunks=['a', 'boom', 'c'], stop_on_error=True)
    alpha = job.made[0]
    assert len(job.made) == 1 and alpha.finished == 1
    assert [c for _, c in alpha.calls if _ == 'work'] == ['a', 'boom']
    assert ps.sent[-1].splitlines()[0].endswith(' with 1 failure')
    job = Seq()
    with pytest.raises(RuntimeError, match=r'seq: failed scopes: bad \(count\)'):
        job.run(chunks=['bad'], stop_on_error=True)


def test_stop_on_error_must_be_a_boolean(fresh_app):
    with pytest.raises(ValueError, match='stop_on_error must be true or false'):
        Seq().run(chunks=['a'], stop_on_error='false')


def test_a_finish_failure_is_a_failure_of_the_job_with_its_own_line(fresh_app):
    job = Seq()
    with pytest.raises(RuntimeError, match='seq: failed scopes: beta-close'):
        job.run(select='grp.beta', break_finish=True)
    assert job.lines == ['None: done', 'beta: finish failed (boom)']


def test_a_finish_exception_is_logged_and_counted(fresh_app):
    class Broken(Beta):
        def finish(self, **kwargs):
            raise OSError('s3 down')
    job = Seq()
    job.namespace = types.SimpleNamespace(**{'grp.Beta': Broken})
    with pytest.raises(RuntimeError, match='seq: failed scopes: grp.beta'):
        job.run(select='grp.beta')
    assert job.lines[-1] == 'grp.beta: finish failed (OSError: s3 down)'


def test_list_results_fail_when_any_item_failed(fresh_app):
    job = Seq()
    with pytest.raises(RuntimeError, match=r'many \(count\)'):
        job.run(select='grp.alpha', chunks=['many'])
    assert job.lines == ['many: done', 'many: failed', 'closed']


def test_a_pipeline_whose_start_raises_fails_every_chunk(fresh_app):
    job = Seq()
    with pytest.raises(RuntimeError, match='seq: failed scopes: a, b'):
        job.run(select='other', chunks=['a', 'b'], break_start=True)
    assert job.lines[:2] == ['a: failed (RuntimeError: no fs)', 'b: failed (RuntimeError: no fs)']


def test_an_interruption_alerts_and_propagates_without_running_more_chunks(fresh_app):
    ps = Recorder()
    fresh_app.set('ps', ps)
    job = Seq()
    with pytest.raises(JobInterrupted):
        job.run(chunks=['a', 'stop', 'c'])
    alpha = job.made[0]
    assert [c for _, c in alpha.calls if _ == 'work'] == ['a', 'stop'] and len(job.made) == 1
    text = ps.sent[-1].splitlines()
    assert text[0].startswith('Seq job interrupted after ') and text[1] == 'interrumpido en grp.alpha.stop: SIGTERM'
    assert job.alerted


def test_existing_sequence_jobs_keep_their_behaviour(fresh_app):
    ps = Recorder()
    fresh_app.set('ps', ps)

    class Plain(Pipeline):
        def __init__(self):
            super().__init__()
            self.pipes = {'update': {'Setup': []}}

    class Old(SequenceJob):
        sequence = {'grp': ['plain']}

        def __init__(self):
            super().__init__()
            self.namespace = types.SimpleNamespace(**{'grp.Plain': Plain})
    job = Old()
    assert job.stop_on_error is True and job.run(chunks='a, b') == {}
    assert len(ps.sent) == 1 and ps.sent[0].startswith('Old job executed in ') and ps.sent[0].endswith(' :-)')
