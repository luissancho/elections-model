"""Configuración común de los tests.

Los tests unitarios no necesitan base de datos. Los de `tests/integration/` usan la fixture `app`,
que arranca `mtpy.run()` y se salta el test si la base de datos local no está disponible.
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope='session')
def app():
    try:
        from mtpy import mtpy
        app = mtpy.run()

        from mtpy.models import elections as models
        models.Events().get_results(formatted=True)
    except Exception as e:  # pragma: no cover - depende del entorno
        pytest.skip('Base de datos no disponible: {}'.format(e))

    return app
