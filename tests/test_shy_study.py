"""Tests de `mtpy/lib/studies/shy/study.py`: el estudio entero del voto oculto desde las partes publicadas."""
import json
import shutil

import pytest

from mtpy.core.io import FileSystem
from mtpy.lib import bundle
from mtpy.lib.publish import POLL_COLUMNS
from mtpy.lib.studies.shy import study as study_mod
from mtpy.lib.studies.shy.data import MissingInput
from mtpy.lib.studies.shy.study import compute_shy
from tests.fakes import bundle_fixture, write_shy_bundle


def publish_current(fs):
    """Publica las tres partes de la elección en curso (`es/2026-11-29`) con los sondeos y partidos del run."""
    names = ['PP', 'PSOE', 'VOX', 'FA', 'UP']
    figures = [('2026-11-20', (32, 26, 20, 6, 3)), ('2026-11-25', (30, 28, 18, 8, None))]
    polls = [{'date': date, 'pollster_id': 1, 'pollster': 'GAD3', 'sponsor': None, 'start_date': date,
              'end_date': date, 'sample_size': 1000, 'mtype': None, 'rating': None, 'weight': 1}
             | {name: (None if value is None else float(value)) for name, value in zip(names, values)}
             for date, values in figures]
    blocks = {'Izquierda': ['PSOE', 'FA', 'UP'], 'Derecha': ['PP', 'VOX']}
    meta = bundle_fixture('event-meta') | {'scope': 'es', 'event_date': '2026-11-29', 'name': 'Generales', 'blocks': blocks}
    rows = [{'party_id': i, 'party': party, 'votes': 1000, 'pct': pct, 'seats': 1}
            for i, (party, pct) in enumerate(zip(names, (31.5, 28.4, 17.2, 7.0, 2.8)), 1)]
    writer = bundle.BundleWriter(fs)
    for part, schema, content in (('meta', 'event-meta', meta),
                                  ('cycle', 'event-cycle', {'parties': names, 'columns': list(POLL_COLUMNS), 'polls': polls}),
                                  ('results', 'event-results', {'n_seats': 350, 'rows': rows})):
        writer.write_json(bundle.path_event('es', '2026-11-29', part), schema, content, 'es')


def compute(tmp_path, incumbents=None, without_manifest=False, without_events=False, publish=False):
    """El estudio sobre el paquete sintético con el cambio pedido."""
    fs = FileSystem(str(tmp_path))
    made = write_shy_bundle(fs)
    reader = bundle.BundleReader(fs)
    params = None
    if incumbents is not None:
        with open(made['data'].get_path('voto-oculto/incumbents.json'), 'w', encoding='utf-8') as fh:
            json.dump(incumbents, fh)
    if without_manifest:
        fs.remove(reader.path(bundle.path_manifest()))
    if without_events:
        shutil.rmtree(fs.get_path(reader.path('events')))
    if publish:
        publish_current(fs)
        params = {'lineages': {'PSOE': ['PSOE', 'PSC'], 'PP': ['PP', 'AP'], 'VOX': ['VOX'],
                               'UP': ['SUMAR', 'FA', 'UP', 'IU']}}
    return compute_shy(reader, made['data'], params)


def test_compute_shy_on_the_synthetic_bundle(fresh_app, tmp_path):
    study = compute(tmp_path)
    s = study.summary
    assert (s['n_elections'], s['n_general'], s['n_regional'], s['n_scopes']) == (4, 2, 2, 2) and study.scopes == ['es', 'es-md']
    assert (s['first_year'], s['first_year_regional'], s['last_year']) == (1989, 2015, 2023)
    assert s['left']['all'] == {'n': 4, 'under': 2, 'over': 2, 'mean': -0.13, 'median': -0.25, 'p_t': 0.928, 'p_sign': 0.688, 'size': 1.83, 'size_over': 0.5}
    assert s['left']['general'] == {'n': 2, 'under': 1, 'over': 1, 'mean': -0.02, 'median': -0.02, 'p_t': None, 'p_sign': 0.75, 'size': 3.32, 'size_over': 1.0}
    assert (s['left']['regional']['mean'], s['left']['recent']['mean']) == (-0.25, -1.62)
    assert (s['right']['all']['mean'], s['right']['regional']['mean'], s['lean']['all']['mean'], s['lean']['general']['mean']) == (0.15, 0.25, -0.28, -0.06)
    assert s['vox'] == {'n': 2, 'under': 0, 'over': 2, 'mean': 0.92, 'median': 0.92, 'p_t': None, 'p_sign': 0.25, 'size': 0.92, 'size_over': 0.0,
                        'extremes': {'under': [], 'over': [{'label': 'Generales 2023', 'error': 0.93}, {'label': 'Madrid 2021', 'error': 0.9}]}}
    parties = {row['key']: row for row in s['parties']}
    assert [row['key'] for row in s['parties']] == ['PSOE', 'PP', 'VOX', 'UP', 'Izquierda', 'Derecha', 'incumbent_general', 'incumbent_regional']
    assert parties['PSOE']['figures'] == {'n': 4, 'under': 2, 'over': 2, 'mean': 0.5, 'median': 1.15, 'p_t': 0.79, 'p_sign': 0.688, 'size': 2.8, 'size_over': 0.75}
    assert (parties['PP']['figures']['mean'], parties['PP']['label'], parties['UP']['figures']['mean'], parties['UP']['label']) == (-0.55, 'PP (AP, CD)', -0.16, 'IU / Podemos / Sumar')
    assert (parties['incumbent_general']['figures']['mean'], parties['incumbent_regional']['figures']['mean']) == (-0.15, -0.4)
    assert s['size'] == {'n_pairs': 14, 'mean': 1.56, 'share_over': 0.36}
    assert s['psoe_general']['big'] == ['Generales 2023'] and s['psoe_general']['mean'] == -0.15
    assert s['pp']['early'] == {'from': '1982-01-01', 'to': '1996-01-01', 'errors': [{'label': 'Generales 1989', 'error': -2.3}]} and s['pp']['late']['n'] == 1
    assert (s['podemos']['before']['mean'], s['podemos']['general_recent']['mean'], s['podemos']['regional_recent']['mean']) == (0.1, 0.37, -1.2)
    assert s['current'] == {'as_of': '2026-11-05', 'days_left': 24, 'n_polls': 2, 'from': '2026-10-25', 'to': '2026-11-03',
                            'values': {'Izquierda': 35.5, 'Derecha': 50.0, 'PP': 31.0, 'PSOE': 27.0, 'VOX': 19.0, 'FA': 7.0}, 'names': ['PP', 'PSOE', 'VOX', 'FA']}
    assert s['verdicts'] == {'p1': 'NO', 'p2': 'NO', 'p3': 'DE NADIE'}
    assert study.election == {'date': '2026-11-29', 'days_left': 24, 'as_of': '2026-11-05', 'run_id': '20261105-120000'}


def test_compute_shy_election_rows(fresh_app, tmp_path):
    rows = {(r['scope'], r['event_date']): r for r in compute(tmp_path).elections}
    r89 = rows[('es', '1989-10-29')]
    assert (r89['label'], r89['short'], r89['general'], r89['n_cycle'], r89['n_final']) == ('Generales 1989', 'GEN 89', True, 4, 2)
    assert (r89['Izquierda'], r89['Derecha'], r89['lean'], r89['PSOE'], r89['PP'], r89['VOX'], r89['UP']) == (3.3, -1.75, 5.05, 3.4, -2.3, None, -0.1)
    assert r89['names'] == {'PSOE': 'PSOE', 'PP': 'AP', 'VOX': None, 'UP': 'IU'} and (r89['incumbent'], r89['incumbent_error']) == ('PSOE', 3.4)
    r15 = rows[('es-md', '2015-05-24')]
    assert (r15['UP'], r15['names']['UP'], r15['Izquierda'], r15['Derecha'], r15['lean']) == (0.3, 'UP+IU', -0.6, 1.8, -2.4)
    assert (r15['scope_name'], r15['incumbent'], r15['incumbent_error']) == ('Madrid', 'PP', 1.4)
    assert (r89['results']['Izquierda'], r89['polls']['Izquierda'], r15['results']['UP'], r15['polls']['UP'], r89['polls']['VOX']) == (48.7, 52.0, 22.8, 23.1, None)
    assert rows[('es', '2023-07-23')]['PSOE'] == -3.7 and rows[('es-md', '2021-05-04')]['blocks'] == {'Izquierda': ['PSOE', 'MM', 'UP'], 'Derecha': ['PP', 'VOX']}
    assert [k for k in rows] == [('es', '1989-10-29'), ('es-md', '2015-05-24'), ('es-md', '2021-05-04'), ('es', '2023-07-23')]


def test_compute_shy_groups_and_inputs(fresh_app, tmp_path):
    study = compute(tmp_path)
    tables = {(g['table'], g['key']) for g in study.groups}
    assert {('left', 'all'), ('right', 'recent'), ('lean', 'general'), ('vox', 'all'), ('parties', 'UP'), ('podemos', 'before'), ('pp', 'late')} <= tables
    assert next(g for g in study.groups if (g['table'], g['key']) == ('parties', 'PSOE'))['mean'] == 0.5
    assert next(g for g in study.groups if (g['table'], g['key']) == ('left', 'general'))['label'] == 'Izquierda: Generales'
    assert study.inputs['es']['skipped'] == [{'date': '2016-06-26', 'reason': 'parts'}] and study.inputs['es-xx'] == {'skipped': 'no events'}
    assert study.inputs['run'] == {'run_id': '20261105-120000', 'as_of': '2026-11-05'} and study.inputs['incumbents'] == []


def test_compute_shy_notes_an_election_without_an_incumbent_row(fresh_app, tmp_path):
    study = compute(tmp_path, incumbents={'es': {'2023-07-23': ['PSOE']}})
    assert study.inputs['incumbents'] == [{'scope': 'es', 'date': '1989-10-29'}, {'scope': 'es-md', 'date': '2015-05-24'}, {'scope': 'es-md', 'date': '2021-05-04'}]
    assert study.summary['parties'][6]['figures']['n'] == 1 and study.summary['parties'][7]['figures'] is None


def test_compute_shy_without_a_run_and_without_elections(fresh_app, tmp_path):
    study = compute(tmp_path, without_manifest=True)
    assert study.summary['current'] is None and study.election is None and study.inputs['run'] == {'skipped': 'no manifest'}
    with pytest.raises(MissingInput) as info:
        compute(tmp_path / 'other', without_events=True)
    assert info.value.part == 'events'


def test_compute_shy_moves_the_current_election_to_the_past_once_published(fresh_app, tmp_path):
    study = compute(tmp_path, publish=True)
    assert study.summary['current'] is None and study.election is None
    last = study.elections[-1]
    assert (last['event_date'], last['label'], last['n_final'], last['incumbent'], last['incumbent_error']) == ('2026-11-29', 'Generales 2026', 2, 'PSOE', -1.4)
    assert (last['PP'], last['UP'], last['names']['UP'], last['Izquierda']) == (-0.5, 0.1, 'FA+UP', -1.3)
    assert (last['results']['Izquierda'], last['polls']['Izquierda']) == (38.2, 36.9)
    assert study.summary['n_elections'] == 5 and study.summary['last_year'] == 2026 and study.summary['verdicts']['p1'] == 'NO'


def test_verdicts_in_the_three_cases():
    no = {'mean': 0.47, 'p_t': 0.128}
    yes = {'mean': -1.2, 'p_t': 0.01}
    few = {'mean': -1.2, 'p_t': None}
    labels = {'UP': 'IU / Podemos / Sumar', 'PP': 'PP (AP, CD)', 'VOX': 'Vox'}
    assert study_mod.verdicts(no, no, [{'key': 'PSOE', 'figures': no}], labels) == {'p1': 'NO', 'p2': 'NO', 'p3': 'DE NADIE'}
    assert study_mod.verdicts(yes, few, [{'key': 'PSOE', 'figures': yes}, {'key': 'UP', 'figures': yes}, {'key': 'Izquierda', 'figures': yes}], labels) == \
        {'p1': 'SÍ', 'p2': 'NO', 'p3': 'DE PSOE Y DE IU / PODEMOS / SUMAR'}
    assert study_mod.verdicts(no, None, [{'key': 'VOX', 'figures': None}], labels)['p2'] == 'SIN DATOS'
