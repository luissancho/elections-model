"""Tests del worker sin cola ni base de datos."""
import types

from mtpy.core.worker import Worker


def test_job_class_unwraps_a_module_bound_on_the_namespace():
    """Como `Router.controller_class`: si el paquete `jobs` tiene enlazado el submódulo en vez de la clase."""
    class Publish:
        pass

    module = types.ModuleType('mtpy.jobs.Publish')
    module.Publish = Publish
    worker = Worker.__new__(Worker)
    worker.namespace = types.SimpleNamespace(Publish=module, Other=Publish)
    assert worker.job_class('Publish') is Publish and worker.job_class('Other') is Publish
