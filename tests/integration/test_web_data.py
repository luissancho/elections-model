"""Tests de integración de las rutas de base de la web: las columnas que prometen `polls`, `results` y `drift`
(`docs/web/contrato.md`) salen de verdad de los lectores. Solo leen la base."""
import pytest

from mtpy.lib import webapi
from mtpy.models.elections import Drift

pytestmark = pytest.mark.integration


@pytest.fixture
def queries(app):
    """Caché de consultas vacía antes y después: cada test lee la base."""
    webapi._QUERIES.clear()
    yield app
    webapi._QUERIES.clear()


def test_web_data_readers_return_the_promised_columns(queries):
    polls = webapi.query_polls('es', limit=5)
    assert 0 < len(polls) <= 5 and {'date', 'pollster_id', 'event_date'} <= set(polls.columns)
    assert {'PP', 'PSOE'} <= set(polls.columns)  # una columna por partido
    assert polls['date'].is_monotonic_decreasing

    results = webapi.query_results('es', '2023-07-23')
    assert {'date', 'region_id', 'party', 'votes', 'pct', 'seats'} <= set(results.columns)
    assert (results['region_id'] == 0).all() and {'PP', 'PSOE'} <= set(results['party'])
    assert results['seats'].sum() == 350
    region = webapi.query_results('es', '2023-07-23', 28)
    assert len(region) and (region['region_id'] == 28).all()

    drift = webapi.query_drift('es')
    assert list(drift.columns) == Drift().columns and len(drift)
