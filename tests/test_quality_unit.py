import os
import pandas as pd
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib.quality import (COLUMNS, CRITERIA_FILE, QUALITY_FILE, criteria_keys, level_text, load_criteria,
                              max_points, pollsters_quality_values, quality_from_points, read_quality)

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
GOOD_ROW = {'pollster_id': 1, 'pollster': 'CIS', 'fieldwork': 3, 'fieldwork_source': 'https://www.cis.es/ficha',
            'transparency': 3, 'transparency_source': 'https://www.cis.es/microdatos', 'independence': 0,
            'independence_source': '', 'identity': 1, 'identity_source': 'https://www.cis.es', 'standards': 0,
            'standards_source': '', 'notes': 'Organismo público', 'status': 'revisada', 'checked_at': '2026-10-16'}


@pytest.fixture(autouse=True)
def data_dir(fresh_app):
    """El `App` nuevo lee el directorio `data/` versionado, sin base de datos."""
    fresh_app.set('data', FileSystem(DATA))


def csv_of(rows):
    """CSV sintético con las columnas de la rúbrica."""
    return pd.DataFrame(rows, columns=list(COLUMNS)).to_csv(index=False)


def patch_data(monkeypatch, fresh_app, files):
    """`files`: {nombre: texto}; los nombres que no estén se leen de `data/`."""
    def read(name):
        if name in files:
            return files[name]
        with open(os.path.join(DATA, name), encoding='utf-8') as handle:
            return handle.read()
    monkeypatch.setattr(fresh_app.data, 'read', read)


def patch_csv(monkeypatch, fresh_app, text):
    """Sustituye solo el CSV de evaluaciones."""
    patch_data(monkeypatch, fresh_app, {QUALITY_FILE: text})


def test_catalogue_has_the_five_criteria_in_order_and_ten_points(fresh_app):
    criteria = load_criteria()
    assert criteria_keys(criteria) == ['fieldwork', 'transparency', 'independence', 'identity', 'standards']
    assert max_points(criteria) == 10 and criteria['scale'] == {'per_point': 8, 'floor': 5}
    assert level_text(criteria, 'independence', 1) == 'Un único cliente, un medio.'
    for key, entry in criteria['criteria'].items():
        assert sorted(entry['levels']) == [str(i) for i in range(entry['max'] + 1)], key


def test_quality_from_points_applies_the_floor_and_the_scale(fresh_app):
    scale = load_criteria()['scale']
    assert [quality_from_points(p, scale) for p in (0, 1, 7, 10)] == [5, 8, 56, 80]


def test_read_quality_adds_points_and_quality(fresh_app, monkeypatch):
    patch_csv(monkeypatch, fresh_app, csv_of([GOOD_ROW]))
    frame = read_quality(today=pd.Timestamp('2026-10-16').date())
    assert list(frame.columns) == list(COLUMNS) + ['points', 'quality']
    assert frame.loc[0, 'points'] == 7 and frame.loc[0, 'quality'] == 56.0
    assert frame.loc[0, 'checked_at'] == pd.Timestamp('2026-10-16')


@pytest.mark.parametrize('change, reason', [
    ({'fieldwork': 4}, 'fieldwork'),                 # por encima del máximo
    ({'transparency': -1}, 'transparency'),          # negativo
    ({'fieldwork_source': ''}, 'fieldwork_source'),  # puntos sin fuente
    ({'status': 'ok'}, 'status'),                    # estado desconocido
    ({'checked_at': '2099-01-01'}, 'checked_at'),    # fecha futura
    ({'checked_at': 'ayer'}, 'checked_at'),          # fecha no ISO
])
def test_read_quality_rejects_invalid_rows(fresh_app, monkeypatch, change, reason):
    patch_csv(monkeypatch, fresh_app, csv_of([GOOD_ROW | change]))
    with pytest.raises(ValueError, match=reason):
        read_quality(today=pd.Timestamp('2026-10-16').date())


def test_read_quality_rejects_duplicated_ids_and_wrong_columns(fresh_app, monkeypatch):
    patch_csv(monkeypatch, fresh_app, csv_of([GOOD_ROW, GOOD_ROW]))
    with pytest.raises(ValueError, match='pollster_id'):
        read_quality()
    patch_csv(monkeypatch, fresh_app, 'pollster_id,pollster\n1,CIS\n')
    with pytest.raises(ValueError, match='columns'):
        read_quality()


def test_load_criteria_rejects_a_missing_level(fresh_app, monkeypatch):
    broken = '{"scale": {"per_point": 8, "floor": 5}, "criteria": {"fieldwork": {"name": "Campo", "max": 2, "levels": {"0": "a", "2": "c"}}}}'
    patch_data(monkeypatch, fresh_app, {CRITERIA_FILE: broken})
    with pytest.raises(ValueError, match='fieldwork'):
        load_criteria()


def test_pollsters_quality_values_fills_the_floor_and_checks_ids(fresh_app, monkeypatch):
    patch_csv(monkeypatch, fresh_app, csv_of([GOOD_ROW]))
    criteria = load_criteria()
    evaluated = read_quality(criteria, today=pd.Timestamp('2026-10-16').date())
    pollsters = pd.DataFrame({'id': [1, 2], 'name': ['CIS', 'GAD3'], 'quality': [10.0, 76.0]})
    values = pollsters_quality_values(pollsters, evaluated, criteria['scale'])
    assert values.tolist() == [56.0, 5.0] and values.index.equals(pollsters.index)
    with pytest.raises(ValueError, match='7'):
        pollsters_quality_values(pollsters, evaluated.assign(pollster_id=7), criteria['scale'])


def test_pollsters_quality_values_warns_on_name_mismatch(fresh_app, monkeypatch):
    patch_csv(monkeypatch, fresh_app, csv_of([GOOD_ROW | {'pollster': 'C.I.S.'}]))
    criteria = load_criteria()
    evaluated = read_quality(criteria, today=pd.Timestamp('2026-10-16').date())
    with pytest.warns(UserWarning, match='C.I.S.'):
        pollsters_quality_values(pd.DataFrame({'id': [1], 'name': ['CIS']}), evaluated, criteria['scale'])
