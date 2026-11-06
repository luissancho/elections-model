"""Tests de `mtpy/lib/studies/shy/data.py`: parámetros, tabla del Gobierno, catálogo y lectura de las elecciones y del
run actual desde el paquete."""
import os

import pandas as pd

from mtpy.core.io import FileSystem
from mtpy.lib import bundle
from mtpy.lib.studies.shy import data
from tests.fakes import write_shy_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def copy_event(fs, scope, source, target):
    """Las tres partes de un evento bajo otra fecha, con las fechas de los sondeos desplazadas lo mismo."""
    shift = pd.Timestamp(target) - pd.Timestamp(source)
    writer = bundle.BundleWriter(fs)
    reader = bundle.BundleReader(fs)
    for part, schema in (('meta', 'event-meta'), ('cycle', 'event-cycle'), ('results', 'event-results')):
        content = reader.read_json(bundle.path_event(scope, source, part))['data']
        if part == 'meta':
            content = content | {'event_date': target}
        if part == 'cycle':
            for poll in content['polls']:
                for key in ('date', 'start_date', 'end_date'):
                    poll[key] = (pd.Timestamp(poll[key]) + shift).strftime('%Y-%m-%d')
        writer.write_json(bundle.path_event(scope, target, part), schema, content, scope)


def rewrite_run_meta(fs, bmaps):
    """La `meta` del run actual con otros `bmaps`."""
    reader = bundle.BundleReader(fs)
    run_id = reader.read_json(bundle.path_manifest())['data']['scopes']['es']['latest']
    path = bundle.path_part('es', run_id, 'meta')
    content = reader.read_json(path)['data'] | {'bmaps': bmaps}
    bundle.BundleWriter(fs).write_json(path, 'meta', content, 'es', run_id=run_id)


def test_load_params_and_catalogue_read_the_repository_files():
    data_fs = FileSystem(os.path.join(ROOT, 'data'))
    params = data.load_params(data_fs)
    assert (params['window_days'], params['recent_from'], params['size_threshold'], params['big_threshold']) == (14, '2018-09-01', 2, -3)
    assert params['lineages']['UP'] == ['SUMAR', 'FA', 'UP', 'IU', 'PCE'] and params['pp_early'] == ['1982-01-01', '1996-01-01']
    codes = [code for code, _ in data.load_catalogue(data_fs)]
    assert 'es' in codes and 'es-md' in codes and len(codes) >= 18 and all(code == 'es' or code.startswith('es-') for code in codes)
    assert dict(data.load_catalogue(data_fs))['es-md'] == 'Madrid'
    incumbents = data.load_incumbents(data_fs)
    assert incumbents['es']['2026-11-29'] == ['PSOE'] and incumbents['es-cb']['2023-05-28'] == ['PRC'] and len(incumbents['es']) == 17


def test_final_polls_window_and_weight():
    polls = pd.DataFrame({'date': ['2023-07-10', '2023-07-09', '2023-07-23', '2023-07-20', '2023-07-01'],
                          'end_date': ['2023-07-10', None, '2023-07-23', '2023-07-20', '2023-07-09'], 'weight': [1, 1, 1, 0, 1]})
    out = data.final_polls(polls, '2023-07-23', 14)
    assert out['date'].tolist() == ['2023-07-10', '2023-07-09', '2023-07-23', '2023-07-01']   # end 13, date 14, end 0, end 14 days before; weight 0 out


def test_read_elections_reads_the_published_elections_of_the_catalogue(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); made = write_shy_bundle(fs)
    elections, inputs = data.read_elections(bundle.BundleReader(fs), data.load_catalogue(made['data']), data.load_params(made['data']))
    assert [(e.scope, e.event_date) for e in elections] == [('es', '1989-10-29'), ('es-md', '2015-05-24'), ('es-md', '2021-05-04'), ('es', '2023-07-23')]
    assert [e.label for e in elections] == ['Generales 1989', 'Madrid 2015', 'Madrid 2021', 'Generales 2023'] and elections[1].short == 'MD 15'
    assert elections[0].polls.shape[0] == 2 and elections[0].n_cycle == 4 and elections[0].results['AP'] == 25.8 and elections[0].general
    assert elections[1].blocks == {'Izquierda': ['PSOE', 'UP', 'IU'], 'Derecha': ['PP', 'Cs']} and elections[1].scope_name == 'Madrid'
    assert inputs['es'] == {'events': 2, 'skipped': [{'date': '2016-06-26', 'reason': 'parts'}]}
    assert inputs['es-md'] == {'events': 2, 'skipped': [{'date': '2019-05-26', 'reason': 'no final polls'}]}


def test_read_elections_skips_a_scope_without_events(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); made = write_shy_bundle(fs)
    _, inputs = data.read_elections(bundle.BundleReader(fs), data.load_catalogue(made['data']), data.load_params(made['data']))
    assert inputs['es-xx'] == {'skipped': 'no events'}


def test_read_elections_adds_the_month_when_the_year_repeats(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); made = write_shy_bundle(fs)
    copy_event(fs, 'es', '2023-07-23', '2023-11-19')   # helper of the test: the same three parts under another date
    elections, _ = data.read_elections(bundle.BundleReader(fs), data.load_catalogue(made['data']), data.load_params(made['data']))
    assert [e.label for e in elections if e.event_date.startswith('2023')] == ['Generales jul. 2023', 'Generales nov. 2023']
    assert [e.short for e in elections if e.event_date.startswith('2023')] == ['GEN 23J', 'GEN 23N']


def test_read_current_reads_the_latest_run(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); made = write_shy_bundle(fs)
    current, inputs = data.read_current(bundle.BundleReader(fs), data.load_params(made['data']))
    assert (current['run_id'], current['as_of'], current['event_date']) == ('20261105-120000', '2026-11-05', '2026-11-29')
    assert current['polls']['date'].tolist() == ['2026-10-25', '2026-11-03'] and current['main'] == ['PP', 'PSOE', 'VOX', 'FA']
    assert current['blocks'] == {'Izquierda': ['PSOE', 'FA', 'UP'], 'Derecha': ['PP', 'VOX']} and inputs == {'run_id': '20261105-120000', 'as_of': '2026-11-05'}


def test_read_current_without_blocks(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); made = write_shy_bundle(fs)
    rewrite_run_meta(fs, bmaps={'main': ['PP', 'PSOE']})   # helper of the test: the run meta without `blocks`
    current, _ = data.read_current(bundle.BundleReader(fs), data.load_params(made['data']))
    assert current['blocks'] is None and current['main'] == ['PP', 'PSOE']


def test_read_current_is_none_without_a_run_or_once_the_election_is_published(fresh_app, tmp_path):
    fs = FileSystem(str(tmp_path)); made = write_shy_bundle(fs); params = data.load_params(made['data'])
    writer = bundle.BundleWriter(fs)
    writer.write_json(bundle.path_event('es', '2026-11-29', 'results'), 'event-results', {'n_seats': 350, 'rows': []}, 'es')
    assert data.read_current(bundle.BundleReader(fs), params) == (None, {'skipped': 'election published'})
    fs.remove(bundle.BundleReader(fs).path(bundle.path_manifest()))   # the bundle lives under its prefix
    assert data.read_current(bundle.BundleReader(fs), params) == (None, {'skipped': 'no manifest'})
