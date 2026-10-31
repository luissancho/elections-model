"""Tests de los lanzadores de la raíz (`job.py`, `pipeline.py`) y del job `Pipeline`, sin base ni Pushover."""
import signal
import types

import pytest

import job
import pipeline
from mtpy.core.worker import Job, JobInterrupted, Pipeline as CorePipeline, SequenceJob


@pytest.fixture
def booted(monkeypatch, fresh_app):
    monkeypatch.setattr(job.mtpy, 'run', lambda: fresh_app)
    monkeypatch.setattr(pipeline.mtpy, 'run', lambda: fresh_app)
    saved = signal.getsignal(signal.SIGTERM)
    yield fresh_app
    signal.signal(signal.SIGTERM, saved)


def test_job_main_dispatches_the_json_and_returns_0(booted):
    seen = []

    class FakeJob(Job):
        def dispatch(self, **kwargs):
            seen.append(kwargs)
    assert job.main('fake_job', '{"a": 1}', namespace=types.SimpleNamespace(FakeJob=FakeJob)) == 0
    assert job.main('fake_job', None, namespace=types.SimpleNamespace(FakeJob=FakeJob)) == 0
    assert seen == [{'a': 1}, {}]


def test_job_main_returns_1_with_the_traceback_when_the_job_fails(booted, capsys):
    class Boom(Job):
        def run(self, **kwargs):
            raise ValueError('bad')
    assert job.main('boom', '{}', namespace=types.SimpleNamespace(Boom=Boom)) == 1
    assert 'ValueError: bad' in capsys.readouterr().err


def test_job_main_installs_a_sigterm_handler_that_interrupts(booted, capsys):
    class Waits(Job):
        def run(self, **kwargs):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    assert job.main('waits', '{}', namespace=types.SimpleNamespace(Waits=Waits)) == 1
    assert 'waits: interrupted (SIGTERM)' in capsys.readouterr().err
    with pytest.raises(JobInterrupted):
        job.interrupt(signal.SIGTERM, None)


def test_pipeline_job_requires_group_dot_pipeline(fresh_app):
    from mtpy.jobs.Pipeline import Pipeline
    with pytest.raises(ValueError, match='group.pipeline'):
        Pipeline().run(select='forecast')
    ran = []

    class Fake(CorePipeline):
        def __init__(self):
            super().__init__()
            self.pipes = {'update': {'Setup': ['mark']}}

        def mark_task(self, chunk=None, **kwargs):
            ran.append(chunk)
    launcher = Pipeline()
    launcher.namespace = types.SimpleNamespace(**{'grp.Fake': Fake})
    launcher.run(select='grp.fake', chunks='a,b')
    assert launcher.sequence == {'grp': ['fake']} and ran == ['a', 'b']


def test_pipeline_main_runs_each_chunk_with_its_position_finishes_and_prints_the_lines(booted, capsys):
    seen = []

    class Fake(CorePipeline):
        def __init__(self):
            super().__init__()
            self.pipes = {'update': {'Setup': ['mark']}}

        def mark_task(self, chunk=None, position=None, **kwargs):
            seen.append((chunk, position))
            self.result, self.line = {'status': 'done'}, '{}: done'.format(chunk)

        def finish(self, **kwargs):
            seen.append('finish')
    assert pipeline.main('grp.fake', 'update', ['a', 'b'], namespace=types.SimpleNamespace(**{'grp.Fake': Fake})) == 0
    assert seen == [('a', (1, 2)), ('b', (2, 2)), 'finish']
    assert capsys.readouterr().out.splitlines()[-2:] == ['a: done', 'b: done']


def test_pipeline_job_runs_a_publish_pipeline_standalone_on_the_app_fs(booted, tmp_path):
    from mtpy.core.io import FileSystem
    from mtpy.jobs.Pipeline import Pipeline
    booted.set('fs', FileSystem(str(tmp_path)))
    job_ = Pipeline()
    job_.run(select='publish.manifest', freeze={'active': False})
    assert (tmp_path / 'site' / 'v1' / 'manifest.json').exists()
    assert job_.results[('publish.manifest', None)]['status'] == 'published'


def test_pipeline_main_returns_1_when_a_chunk_or_finish_failed(booted, capsys):
    class BadChunk(CorePipeline):
        def __init__(self):
            super().__init__()
            self.pipes = {'update': {'Setup': ['mark']}}

        def mark_task(self, chunk=None, position=None, **kwargs):
            self.result, self.line = {'status': 'failed' if chunk == 'bad' else 'done'}, '{}'.format(chunk)

    class BadFinish(BadChunk):
        def finish(self, **kwargs):
            return 'closed-badly'
    assert pipeline.main('grp.bad', 'update', ['a', 'bad'], namespace=types.SimpleNamespace(**{'grp.Bad': BadChunk})) == 1
    assert pipeline.main('grp.bad', 'update', ['a'], namespace=types.SimpleNamespace(**{'grp.Bad': BadFinish})) == 1
    assert pipeline.main('grp.bad', 'update', ['a'], namespace=types.SimpleNamespace(**{'grp.Bad': BadChunk})) == 0
