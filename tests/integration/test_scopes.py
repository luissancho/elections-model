"""Tests de integración de los catálogos de ámbitos y circunscripciones (M11)."""
import pytest

pytestmark = pytest.mark.integration


def test_catalogues_are_synced_and_es_params_unchanged(app):
    from mtpy.lib.data import get_districts, get_event_params, get_scopes, save_catalogues
    from mtpy.models.elections import Districts, Scopes
    out = save_catalogues()
    assert out['scopes'] == 18 and out['districts'] == 118
    from mtpy.lib.quality import read_quality
    from mtpy.models.elections import Pollsters, PollstersQuality
    assert out['quality'] == read_quality().shape[0] == PollstersQuality().get_results(formatted=True).shape[0]
    assert out['pollsters'] == Pollsters().get_results(formatted=True).shape[0]
    assert Scopes().get_results(formatted=True).shape[0] == get_scopes().shape[0] == 18
    assert Districts().get_results(formatted=True).shape[0] == get_districts().shape[0] == 118
    assert get_event_params('es', '2026-11-29', path='.')['bmaps']['main'][:2] == ['PP', 'PSOE']
