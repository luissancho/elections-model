import os
import pandas as pd
import pytest

from mtpy.core.io import FileSystem
from mtpy.lib.quality import (COLUMNS, CRITERIA_FILE, EVIDENCE_KEYS, QUALITY_FILE, count_national_events, evaluated_keys,
                              implantation_points, level_text, load_criteria, manual_points, max_points,
                              pollsters_quality_values, prior_quality, quality_breakdown, quality_from_points,
                              read_quality)

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
    """Sustituye sólo el CSV de evaluaciones."""
    patch_data(monkeypatch, fresh_app, {QUALITY_FILE: text})


def test_criteria_are_four_and_sum_ten(fresh_app):
    criteria = load_criteria()
    assert list(criteria['criteria']) == ['fieldwork', 'transparency', 'independence', 'implantation']
    implantation = criteria['criteria']['implantation']
    assert implantation['max'] == 2 and implantation['computed'] == 'national'
    assert implantation['levels']['0'] == 'Sólo ha sondeado elecciones autonómicas o locales.'
    assert implantation['levels']['1'] == 'Ha sondeado al menos una elección general.'
    assert 'ESOMAR' in implantation['levels']['2'] and 'ISO 20252' in implantation['levels']['2']
    assert evaluated_keys(criteria) == ['fieldwork', 'transparency', 'independence']
    assert EVIDENCE_KEYS == ('identity', 'standards')
    assert max_points(criteria) == 10 and criteria['scale'] == {'per_point': 8, 'floor': 5}
    assert level_text(criteria, 'independence', 1) == 'Un único cliente, un medio.'
    for key, entry in criteria['criteria'].items():
        assert sorted(entry['levels']) == [str(i) for i in range(entry['max'] + 1)], key


def test_quality_from_points_applies_the_floor_and_the_scale(fresh_app):
    scale = load_criteria()['scale']
    assert [quality_from_points(p, scale) for p in (0, 1, 7, 10)] == [5, 8, 56, 80]


def test_read_quality_adds_the_manual_points(fresh_app, monkeypatch):
    patch_csv(monkeypatch, fresh_app, csv_of([GOOD_ROW]))
    frame = read_quality(today=pd.Timestamp('2026-10-16').date())
    assert list(frame.columns) == list(COLUMNS) + ['manual']
    assert frame.loc[0, 'manual'] == 6
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
    national_events = pd.Series({1: 2}, name='num_national_events')
    values = pollsters_quality_values(pollsters, evaluated, criteria['scale'], national_events)
    assert values.tolist() == [56.0, 5.0] and values.index.equals(pollsters.index)
    with pytest.raises(ValueError, match='7'):
        pollsters_quality_values(pollsters, evaluated.assign(pollster_id=7), criteria['scale'], national_events)


def test_pollsters_quality_values_warns_on_name_mismatch(fresh_app, monkeypatch):
    patch_csv(monkeypatch, fresh_app, csv_of([GOOD_ROW | {'pollster': 'C.I.S.'}]))
    criteria = load_criteria()
    evaluated = read_quality(criteria, today=pd.Timestamp('2026-10-16').date())
    with pytest.warns(UserWarning, match='C.I.S.'):
        pollsters_quality_values(pd.DataFrame({'id': [1], 'name': ['CIS']}), evaluated, criteria['scale'],
                                 pd.Series(dtype=int))


def test_identity_voids_transparency():
    assert manual_points(3, 2, 1, identity=0) == 4
    assert manual_points(3, 2, 1, identity=1) == 6
    series = manual_points(pd.Series([3, 3]), pd.Series([2, 2]), pd.Series([1, 1]), pd.Series([0, 1]))
    assert series.tolist() == [4, 6]


def test_implantation_needs_a_general_and_standards_on_top():
    assert implantation_points(0, 1) == 0
    assert implantation_points(1, 0) == 1
    assert implantation_points(3, 1) == 2
    assert implantation_points(pd.Series([0, 2]), pd.Series([1, 1])).tolist() == [0, 2]


def test_count_national_events_ignores_same_day_and_later_elections():
    events = pd.DataFrame({'pollster_id': [1, 1, 1, 2], 'event_date': pd.to_datetime(
        ['2019-11-10', '2023-07-23', '2023-07-23', '2023-07-23'])})
    assert count_national_events(events, pd.Timestamp('2023-07-23')).to_dict() == {1: 1}
    assert count_national_events(events, pd.Timestamp('2023-07-24')).to_dict() == {1: 2, 2: 1}
    assert count_national_events(events).to_dict() == {1: 2, 2: 1}
    assert count_national_events(events, pd.NaT).to_dict() == {1: 2, 2: 1}
    assert 3 not in count_national_events(events).index


def test_prior_quality_values():
    scale = {'per_point': 8, 'floor': 5}
    evaluations = pd.DataFrame({
        'pollster_id': [10, 11], 'fieldwork': [3, 3], 'transparency': [3, 1], 'independence': [0, 2],
        'identity': [1, 1], 'standards': [0, 1]})  # CENTRA and GfK
    none = pd.Series(dtype=int)
    before = prior_quality([10, 11, 12], evaluations, scale, none)
    assert before['quality'].tolist() == [48.0, 48.0, 5.0]
    after = prior_quality([10, 11, 12], evaluations, scale, pd.Series({10: 1, 11: 2, 12: 1}))
    assert after['quality'].tolist() == [56.0, 64.0, 8.0]
    assert after.loc[11, ['num_national_events', 'manual', 'implantation', 'points']].tolist() == [2, 6, 2, 8]
    assert after.index.name == 'pollster_id'


@pytest.mark.parametrize('change, reason', [
    ({'standards': 2}, 'standards'),
    ({'identity': 1, 'identity_source': ''}, 'identity_source'),
])
def test_read_quality_validates_identity_and_standards_as_evidence(fresh_app, monkeypatch, change, reason):
    patch_csv(monkeypatch, fresh_app, csv_of([GOOD_ROW | change]))
    with pytest.raises(ValueError, match=reason):
        read_quality(today=pd.Timestamp('2026-10-16').date())


def test_load_criteria_rejects_an_unknown_computed_value(fresh_app, monkeypatch):
    broken = ('{"scale": {"per_point": 8, "floor": 5}, "criteria": {"x": {"name": "X", "max": 1, "computed": "other", '
              '"levels": {"0": "a", "1": "b"}}}}')
    patch_data(monkeypatch, fresh_app, {CRITERIA_FILE: broken})
    with pytest.raises(ValueError, match='computed'):
        load_criteria()


def test_quality_breakdown_splits_implantation(fresh_app):
    criteria = load_criteria()
    record = GOOD_ROW | {'transparency': 3, 'identity': 1, 'standards': 1, 'points': 8}
    out = quality_breakdown(record, criteria)
    assert [row['key'] for row in out['rows']] == ['fieldwork', 'transparency', 'independence']
    assert (out['manual'], out['implantation'], out['national'], out['transparency_void']) == (6, 2, True, False)
    void = quality_breakdown(GOOD_ROW | {'identity': 0, 'identity_source': '', 'points': 3}, criteria)
    assert (void['manual'], void['implantation'], void['national'], void['transparency_void']) == (3, 0, False, True)
    assert void['rows'][1]['points'] == 0
    assert quality_breakdown(None, criteria)['rows'] == []
