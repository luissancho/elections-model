"""Tests de `ScopePipeline.collect` con un pipeline mínimo."""
import warnings

from mtpy.pipelines.base import ScopePipeline


class Minimal(ScopePipeline):
    """Pipeline de una tarea cuyo `work` se inyecta."""

    pipes = {'update': {'Load': ['work']}}

    def work_task(self, chunk=None, work=None, **kwargs):
        self.result = self.collect(chunk, work)

    def line_of(self, chunk, result, **kwargs):
        return '{}: {}'.format(chunk, result['status'])


def test_collect_reraises_an_early_end(fresh_app):
    p = Minimal()

    def work():
        p.end({'status': 'skipped', 'reason': 'x'})

    p.run('update', chunk='a', work=work, verbose=0)
    assert p.result == {'status': 'skipped', 'reason': 'x'}


def test_collect_works_without_a_progress_object(fresh_app, capsys, recwarn):
    p = Minimal()

    def work():
        warnings.warn('w')
        return {'status': 'done'}

    p.run('update', chunk='a', work=work, verbose=0)
    assert p.progress is None
    assert p.result == {'status': 'done', 'warnings': ['w']}
    assert capsys.readouterr().err == '' and len(recwarn) == 0
