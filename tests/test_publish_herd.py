"""Tests de `mtpy/lib/publish/herd.py`: los exportadores del estudio del rebano y su publicación genérica."""
import dataclasses
import os
from unittest.mock import ANY

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle, publish
from mtpy.lib.publish.herd import CHANCE_COLUMNS, ROBUSTNESS_COLUMNS, WEEK_COLUMNS, _subtitle
from mtpy.lib.studies.herd.study import compute_herd
from tests.fakes import write_herd_bundle
from tests.test_publish_unit import fake_prov, make_writer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def valid(schema, data):
    """El `data` exportado, limpio como lo escribe el paquete, pasa `validate` con el sobre de un estudio."""
    env = bundle.jsonable(bundle.envelope(schema, data, None, generated_at='2026-10-08T12:00:00Z'))
    return bundle.validate(schema, env)['data']


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    """El estudio calculado una vez sobre el paquete sintético."""
    fs = FileSystem(str(tmp_path_factory.mktemp('herd')))
    write_herd_bundle(fs)
    return compute_herd(bundle.BundleReader(fs), FileSystem(os.path.join(ROOT, 'data')))


SIG = {'far_final': {'n': 23, 'k': 19, 'sign_p': 0.0013, 'clear': 3, 'median_change': -0.15, 'since_year': '1986'},
       'far_mid': {'n': 17, 'k': 9, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.1},
       'mid_final': {'n': 29, 'k': 23, 'sign_p': 0.0012, 'clear': 2, 'median_change': -0.15}}


def test_subtitle_when_the_final_weeks_are_significant_and_the_middle_is_not():
    s = {'n_elections': 33, 'since_year': '1977', 'comparisons': SIG}
    # the year is the first of far_final (1986), not the first of any comparison (1977)
    assert _subtitle(s) == (
        'En 19 de las 23 elecciones con sondeos lejos y cerca de la votación, desde 1986, las mismas encuestadoras se '
        'parecen más entre sí en las tres últimas semanas que lejos de ella: por azar, 1 entre 769. Hasta tres semanas '
        'antes, no: 9 de 17, como una moneda.')


def test_subtitle_without_significance():
    s = {'n_elections': 2, 'since_year': '2019', 'comparisons': {
        'far_final': {'n': 4, 'k': 3, 'sign_p': 0.31, 'clear': 0, 'median_change': 0.62, 'since_year': '2019'},
        'far_mid': {'n': 1, 'k': 1, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.25},
        'mid_final': {'n': 1, 'k': 1, 'sign_p': 0.5, 'clear': 0, 'median_change': -0.67}}}
    assert _subtitle(s) == ('En 3 de las 4 elecciones con sondeos lejos y cerca de la votación, desde 2019, las mismas '
                            'encuestadoras se parecen más entre sí en las tres últimas semanas que lejos de ella, una '
                            'proporción que podría deberse al azar.')


def test_subtitle_other_branches():
    far_mid = {'n': 17, 'k': 15, 'sign_p': 0.001, 'clear': 2, 'median_change': -0.2}
    final = dict(SIG['far_final'], sign_p=0.0002)
    text = _subtitle({'since_year': '1977', 'comparisons': {**SIG, 'far_mid': far_mid, 'far_final': final}})
    assert text.endswith('por azar, menos de 1 entre 1.000. Y también hasta tres semanas antes: 15 de 17.')
    no_mid = _subtitle({'since_year': '1977', 'comparisons': {**SIG, 'far_mid': dict(far_mid, n=0, k=0)}})
    assert no_mid.endswith('por azar, 1 entre 769.')
    assert 'copi' not in text.lower()


def test_subtitle_without_a_second_sentence_when_the_last_stretch_is_not_significant():
    quiet = dict(SIG['mid_final'], k=15, sign_p=0.6)
    text = _subtitle({'since_year': '1977', 'comparisons': {**SIG, 'mid_final': quiet}})
    assert text.endswith('por azar, 1 entre 769.')


def test_subtitle_when_most_elections_do_not_get_closer():
    far_final = {'n': 10, 'k': 3, 'sign_p': 0.01, 'clear': 0, 'median_change': 0.3, 'since_year': '2008'}
    assert _subtitle({'since_year': '2008', 'comparisons': {**SIG, 'far_final': far_final}}) == (
        'En sólo 3 de las 10 elecciones con sondeos lejos y cerca de la votación, desde 2008, las mismas '
        'encuestadoras se parecen más entre sí en las tres últimas semanas que lejos de ella; en las demás, no.')


def test_subtitle_without_elections():
    empty = {'n': 0, 'k': 0, 'sign_p': None, 'clear': 0, 'median_change': None}
    comparisons = {'far_final': empty, 'far_mid': empty, 'mid_final': empty}
    assert _subtitle({'since_year': None, 'comparisons': comparisons}) == (
        'Todavía no hay elecciones con sondeos suficientes para comparar las mismas encuestadoras lejos y cerca de la '
        'votación.')


def test_export_herd_parts_validate(study):
    summary, frames = publish.export_herd_summary(study)
    assert valid('rebano-summary', summary) and set(frames) == {'robustness'}
    elections, frames = publish.export_herd_elections(study)
    valid('rebano-elections', elections)
    frame = frames['elections']
    assert len(frame) == 4 and set(frame.columns) == set(study.elections[0])
    assert len(list(study.elections[0])) == 13
    current, frames = publish.export_herd_current(study)
    valid('rebano-current', current)
    assert current['current']['days_left'] == 24 and current['current']['mid_final'] is None and frames == {}
    meta, frames = publish.export_herd_meta(study, prov=fake_prov())
    valid('rebano-meta', meta)
    assert frames == {}
    assert meta['title'] == '¿Se comportan las encuestadoras como un rebaño?'
    assert meta['election'] == study.election and meta['scope'] == 'es'
    assert 'copi' not in meta['subtitle'].lower()
    assert [s['url'] for s in meta['sources']] == ['/elecciones', '/analisis']


def test_export_herd_current_is_none_without_an_election_in_progress(study):
    idle = dataclasses.replace(study, current=None, election=None)
    assert publish.export_herd_current(idle)[0] == {'current': None}
    valid('rebano-current', {'current': None})
    meta = publish.export_herd_meta(idle, prov=fake_prov())[0]
    assert meta['election'] is None and valid('rebano-meta', meta)


def test_publish_study_rebano_writes_every_part(tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    out = publish.publish_study('rebano', make_writer(tmp_path, fs), prov=fake_prov)
    assert out['study'] == 'rebano' and out['registered_at'] is None and out['checked'] is False
    reader = bundle.BundleReader(fs)
    for part in bundle.STUDY_PARTS['rebano']:
        assert reader.read_json(bundle.path_study('rebano', part))['data'] is not None
    assert reader.read_json(bundle.path_study('rebano', 'summary'))['data']['n_elections'] == 2
    assert fs.exists(f"{bundle.PREFIX}/{bundle.path_study_csv('rebano', 'elections')}")
    assert fs.exists(f"{bundle.PREFIX}/{bundle.path_study_csv('rebano', 'chance')}")
    assert reader.read_json(bundle.path_manifest())['data']['studies']['rebano'] == {
        'published_at': ANY, 'registered_at': None, 'checked': False}
    with pytest.raises(publish.PublishRefused, match='nothing to register'):
        publish.publish_study('rebano', make_writer(tmp_path, fs), register=True, prov=fake_prov)


def test_study_runners_know_which_studies_are_live():
    assert {s: r['live'] for s, r in publish.STUDY_RUNNERS.items()} == {
        'voto-cera': False, 'sesgo-cis': True, 'rebano': True, 'voto-oculto': True, 'andorra': True}


def test_export_herd_weeks_validates_and_flattens(study):
    data, frames = publish.STUDY_EXPORTERS['rebano']['weeks'](study)
    valid('rebano-weeks', data)
    frame = frames['weeks']
    assert list(frame.columns) == WEEK_COLUMNS and len(frame) == len(data['rows']) + len(data['mean'])
    mean = frame[frame.label == 'Media']
    assert len(mean) == len(data['mean']) and mean.event_date.isna().all() and not mean.current.any()
    assert mean.pairs.isna().all() and mean.comparisons.isna().all()
    assert frame.label.iloc[0] != 'Media'
    ratios = mean.ratio.astype(object).where(mean.ratio.notna(), None).tolist()
    assert ratios == [m['ratio'] for m in data['mean']]


def test_export_herd_weeks_csv_counts_are_integers(study):
    frame = publish.STUDY_EXPORTERS['rebano']['weeks'](study)[1]['weeks']
    lines = frame.to_csv(index=False).splitlines()
    first = frame.iloc[0]
    cells = lines[1].split(',')
    assert cells[5] == str(int(first.pairs)) and cells[6] == str(int(first.comparisons))
    media = [line.split(',') for line in lines if ',Media,' in line]
    assert media and all(row[5] == '' and row[6] == '' for row in media)


def test_publish_study_rebano_writes_weeks(tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    publish.publish_study('rebano', make_writer(tmp_path, fs), prov=fake_prov)
    reader = bundle.BundleReader(fs)
    assert 'weeks' in bundle.STUDY_PARTS['rebano']
    assert reader.read_json(bundle.path_study('rebano', 'weeks'))['data']['buckets']
    assert fs.exists(f"{bundle.PREFIX}/{bundle.path_study_csv('rebano', 'weeks')}")


def test_export_herd_chance_validates_and_flattens_the_bins(study):
    chance, frames = publish.export_herd_chance(study)
    assert valid('rebano-chance', chance) and chance['chance']['permutations'] == 20000
    frame = frames['chance']
    assert list(frame.columns) == CHANCE_COLUMNS and str(frame['count'].dtype) == 'int64'
    assert len(frame) == len(chance['chance']['null']['bins']) and int(frame['count'].sum()) == 20000


def test_export_herd_chance_without_shared_pairs_is_null(tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    params = {'chance': {'periods': {'final': [0, 1], 'rest': [1, 2]}, 'permutations': 100, 'bin_width': 0.01,
                         'resamples': 100}}
    empty = compute_herd(bundle.BundleReader(fs), FileSystem(os.path.join(ROOT, 'data')), params=params)
    chance, frames = publish.export_herd_chance(empty)
    assert valid('rebano-chance', chance) and chance['chance'] is None
    assert list(frames['chance'].columns) == CHANCE_COLUMNS and frames['chance'].empty


def test_export_herd_summary_validates_and_flattens_the_robustness_grid(study):
    summary, frames = publish.export_herd_summary(study)
    assert valid('rebano-summary', summary)
    frame = frames['robustness']
    assert list(frame.columns) == ROBUSTNESS_COLUMNS and len(frame) == 24
    assert str(frame['n'].dtype) == 'int64' and str(frame['k'].dtype) == 'int64' and frame['default'].sum() == 3
    assert frame.loc[(frame.pair_days == 3) & (frame.min_rows == 6), 'change'].isna().all()


def test_publish_study_rebano_writes_the_robustness_csv(tmp_path):
    fs = FileSystem(str(tmp_path))
    write_herd_bundle(fs)
    publish.publish_study('rebano', make_writer(tmp_path, fs), prov=fake_prov)
    path = f"{bundle.PREFIX}/{bundle.path_study_csv('rebano', 'robustness')}"
    assert fs.exists(path)
    with open(os.path.join(str(tmp_path), path)) as f:
        text = f.read()
    assert text.splitlines()[0] == ','.join(ROBUSTNESS_COLUMNS) and ',3.0,' not in text  # counts as integers
