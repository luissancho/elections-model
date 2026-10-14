"""Tests de integración de los ratings transversales (M11)."""
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def test_es_ratings_are_invariant_when_regional_weights_are_zero(app):
    from mtpy.lib.computer import Computer
    from mtpy.lib.data import get_scopes
    zero = {s: (1. if s == 'es' else 0.) for s in get_scopes().index}
    base = Computer(scope='es', path='.').build_series()
    base.compute_ratings()
    alone = base.ratings.copy()
    cross = Computer(scope='es', path='.').build_series()
    cross.compute_ratings(scopes=list(zero), rating_weights=zero)
    pd.testing.assert_frame_equal(alone, cross.ratings)


def test_regional_weights_move_ratings_within_bounds(app):
    from mtpy.lib.computer import Computer

    def last_ratings(**kwargs):
        comp = Computer(scope='es', path='.').build_series()
        comp.compute_ratings(**kwargs)
        return comp.ratings.xs(comp.ratings.index.get_level_values('event_date').max(), level='event_date')

    alone, cross = last_ratings(), last_ratings(scopes='all')
    assert cross['rating'].between(0, 100).all()
    # Entran casas que sólo publican en autonómicas
    assert (cross['num_polls'] > 0).sum() > (alone['num_polls'] > 0).sum()
