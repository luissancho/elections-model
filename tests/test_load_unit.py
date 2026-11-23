"""Tests de `mtpy.lib.load`: la secuencia de carga de sondeos de Wikipedia, con un loader y un `Computer` falsos."""
import json
import warnings

import pandas as pd
import pytest

from mtpy.lib import load as load_module
from mtpy.lib.load import (
    LOAD_STEPS, add_catalogue_entries, has_polls_article, load_polls, new_poll_list, poll_changes,
    unevaluated_quality, unmapped_reason
)
from mtpy.models.elections import Pollsters, Sponsors

URLS = {'es': {'2026-11-29': {'polls': 'u'}, '2030-01-01': {'results': 'r'}}}
CLASSES = ('parties', 'pollsters', 'sponsors')


class FakeLoader:
    """Fábrica con la firma de `WikipediaLoader`: anota las construcciones (`inits`) y las llamadas (`calls`)."""

    def __init__(self, data=5, missing=None, skipped=None, fixes_missing=None, new_polls=2, changes=None):
        self.inits, self.calls, self.instance = [], [], None
        self.changes = changes or (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        self.results_error = None
        self.n, self.new_polls = data, new_polls
        self.missing, self.skipped, self.fixes = missing or {}, skipped or {}, fixes_missing or []

    def __call__(self, scope, event_date, years=None, exclude=None, verbose=0, path=None):
        self.inits.append((scope, event_date, years, verbose))
        factory = self

        class Loader:
            pages = {}
            data = pd.DataFrame()
            polls = pd.DataFrame()
            fixes_missing = []
            parties_checks = pd.DataFrame({'PP': [1, 1]})
            reloaded = False

            def read_data(self, pages=None):
                factory.calls.append(('read_data', pages))
                self.data = pd.DataFrame({'x': range(factory.n)})
                self.pages = {'u': b''}
                for cls in CLASSES:
                    names = [] if self.reloaded else factory.missing.get(cls, [])
                    setattr(self, cls + '_missing', list(names))
                    setattr(self, cls + '_skipped', list(factory.skipped.get(cls, [])))
                return self

            def reload_catalogues(self):
                factory.calls.append(('reload_catalogues',))
                self.reloaded = True
                return self

            def build_series(self):
                factory.calls.append(('build_series',))
                self.fixes_missing = list(factory.fixes)
                return self

            def select_series(self, overwrite):
                factory.calls.append(('select_series', overwrite))
                self.new_polls = factory.new_polls if isinstance(factory.new_polls, pd.DataFrame) else pd.DataFrame({
                    'pollster': ['Casa {}'.format(i) for i in range(factory.new_polls)],
                    'sponsor': ['Medio' if i % 2 == 0 else None for i in range(factory.new_polls)],
                    'start_date': pd.to_datetime(['2026-11-{:02d}'.format(10 - i) for i in range(factory.new_polls)]),
                    'end_date': pd.to_datetime(['2026-11-{:02d}'.format(12 - i) for i in range(factory.new_polls)])})
                self.polls = pd.DataFrame({'x': range(factory.n)}) if overwrite else self.new_polls
                self.read_polls, self.read_results, self.cur_polls, _ = factory.changes
                return self

            def select_changed(self, keys):
                factory.calls.append(('select_changed', list(keys)))
                self.polls = pd.DataFrame({'x': range(len(self.polls) + len(keys))})
                return self

            def delete_polls(self, keys):
                factory.calls.append(('delete_polls', list(keys)))
                return self

            def fetch_cur_results(self):
                factory.calls.append(('fetch_cur_results',))
                if factory.results_error is not None:
                    raise factory.results_error
                self.cur_results = factory.changes[3]
                return self.cur_results

            def save_polls(self, overwrite):
                factory.calls.append(('save_polls', overwrite))

            def save_results(self, overwrite):
                factory.calls.append(('save_results', overwrite))

        self.instance = Loader()
        return self.instance


class FakeComputer:
    """Fábrica con la firma de `Computer`: `compute_weights` anota `(save, overwrite)`."""

    def __init__(self):
        self.inits, self.calls = [], []

    def __call__(self, scope, event_dates, verbose=0, path=None):
        self.inits.append((scope, event_dates, verbose, path))
        factory = self

        class Comp:
            def build_series(self):
                return self

            def compute_weights(self, save=False, overwrite=False):
                factory.calls.append((save, overwrite))
                return pd.DataFrame({'a': [1, 2, 3]})

        return Comp()


class FakeCatalogues:
    """`catalogues(pollsters, sponsors)`: anota las llamadas y devuelve lo que recibe."""

    def __init__(self):
        self.calls = []

    def __call__(self, pollsters, sponsors):
        self.calls.append((pollsters, sponsors))
        return {'pollsters': list(pollsters), 'sponsors': list(sponsors)}


def run(loader=None, **kwargs):
    loader = FakeLoader() if loader is None else loader
    computer, catalogues, steps, lines = FakeComputer(), FakeCatalogues(), [], []
    options = dict(progress=lambda step, seconds: steps.append(step), report=lambda text, level: lines.append((text, level)),
                   loader=loader, computer=computer, catalogues=catalogues, urls=URLS)
    options.update(kwargs)
    result = load_polls('es', '2026-11-29', **options)
    return result, loader, computer, catalogues, steps, lines


def test_load_polls_runs_the_sequence_of_the_notebook_and_reports_each_phase():
    result, loader, computer, _, steps, lines = run()
    assert steps == list(LOAD_STEPS)
    assert loader.calls == [('read_data', None), ('build_series',), ('select_series', False), ('fetch_cur_results',),
                            ('save_polls', False), ('save_results', False)]
    assert computer.calls == [(True, True)] and computer.inits == [('es', ['2026-11-29'], 0, '.')]
    assert (result['status'], result['action'], result['n_read'], result['n_saved'], result['overwrite']) == \
        ('loaded', 'polls', 5, 2, False)
    assert result['event_date'] == '2026-11-29' and 'reason' not in result
    assert set(result['seconds']) == set(LOAD_STEPS) | {'total'}
    assert ('# Polls: 5', 1) in lines and ('Missing Pollsters: set()', 1) in lines
    assert ('Missing Parties: set()', 1) in lines and ('Missing Sponsors: set()', 1) in lines
    assert (loader.instance.parties_checks.to_string(), 2) in lines


def test_new_poll_list_sorts_by_end_date_and_cleans_missing_values():
    polls = pd.DataFrame({'pollster': ['B', 'A', 'C'], 'sponsor': ['X', None, float('nan')],
                          'start_date': pd.to_datetime(['2026-11-01', '2026-11-05', None]),
                          'end_date': pd.to_datetime(['2026-11-03', '2026-11-03', '2026-11-02'])})
    assert new_poll_list(polls) == [
        {'pollster': 'C', 'sponsor': None, 'start_date': None, 'end_date': '2026-11-02'},
        {'pollster': 'A', 'sponsor': None, 'start_date': '2026-11-05', 'end_date': '2026-11-03'},
        {'pollster': 'B', 'sponsor': 'X', 'start_date': '2026-11-01', 'end_date': '2026-11-03'}]
    assert new_poll_list(None) == [] and new_poll_list(polls.iloc[:0]) == []


def test_load_polls_overwrite_saves_the_whole_cycle():
    result, loader, computer, *_ = run(overwrite=True)
    assert ('select_series', True) in loader.calls and ('save_polls', True) in loader.calls
    assert ('save_results', True) in loader.calls
    assert result['n_saved'] == 5 and result['overwrite'] is True and computer.calls == [(True, True)]


def test_load_polls_dry_run_selects_but_saves_nothing():
    result, loader, computer, _, steps, _ = run(save=False)
    names = [call[0] for call in loader.calls]
    assert 'select_series' in names and 'save_polls' not in names and 'save_results' not in names
    assert computer.calls == [] and 'weights' not in steps
    assert result['status'] == 'checked' and result['n_saved'] == 2


def test_load_polls_stops_before_saving_when_names_are_unmapped():
    loader = FakeLoader(missing={'pollsters': ['UJA', 'Sumar'], 'sponsors': ['Artículo 14']})
    result, loader, _, catalogues, *_ = run(loader)
    assert result['status'] == 'failed'
    assert result['reason'] == 'unmapped pollsters: Sumar, UJA; unmapped sponsors: Artículo 14'
    assert loader.calls == [('read_data', None)] and catalogues.calls == []
    assert result['missing'] == {'parties': [], 'pollsters': ['Sumar', 'UJA'], 'sponsors': ['Artículo 14']}


def test_load_polls_force_adds_pollsters_and_sponsors_and_reads_again_from_the_pages():
    loader = FakeLoader(missing={'pollsters': ['Sumar', 'UJA'], 'sponsors': ['Artículo 14']})
    with pytest.warns(UserWarning) as record:
        result, loader, _, catalogues, *_ = run(loader, force=True)
    assert catalogues.calls == [(['Sumar', 'UJA'], ['Artículo 14'])]
    assert loader.calls[:3] == [('read_data', None), ('reload_catalogues',), ('read_data', {'u': b''})]
    assert result['added'] == {'pollsters': ['Sumar', 'UJA'], 'sponsors': ['Artículo 14']}
    assert result['status'] == 'loaded'
    texts = [str(w.message) for w in record]
    assert 'new pollster «Sumar» without mtype: check it' in texts
    assert 'new pollster «UJA» without mtype: check it' in texts


def test_load_polls_force_fails_when_something_is_still_unmapped_after_adding():
    class Stubborn(FakeLoader):
        def __call__(self, *args, **kwargs):
            loader = super().__call__(*args, **kwargs)
            read = loader.read_data

            def read_data(pages=None):
                read(pages)
                loader.pollsters_missing = ['Sumar']
                return loader

            loader.read_data = read_data
            return loader

    with pytest.warns(UserWarning):
        result, *_ = run(Stubborn(missing={'pollsters': ['Sumar']}), force=True)
    assert result['status'] == 'failed' and result['reason'] == 'unmapped after adding: unmapped pollsters: Sumar'


def test_load_polls_force_never_adds_parties_nor_anything_before_failing_on_a_party():
    loader = FakeLoader(missing={'parties': ['Valencian_Front'], 'pollsters': ['Sumar']})
    result, _, _, catalogues, *_ = run(loader, force=True)
    assert result['status'] == 'failed'
    assert result['reason'] == 'unmapped parties: Valencian_Front; unmapped pollsters: Sumar'
    assert catalogues.calls == []


def test_load_polls_force_in_dry_run_adds_nothing_and_says_what_it_would_add():
    loader = FakeLoader(missing={'pollsters': ['Sumar'], 'sponsors': ['Artículo 14']})
    result, _, _, catalogues, *_ = run(loader, save=False, force=True)
    assert catalogues.calls == [] and result['status'] == 'checked'
    assert result['added'] == {'pollsters': ['Sumar'], 'sponsors': ['Artículo 14']}


def test_load_polls_empty_article_loads_zero_and_touches_nothing():
    result, loader, computer, *_ = run(FakeLoader(data=0), overwrite=True)
    assert (result['status'], result['n_read'], result['n_saved']) == ('loaded', 0, 0)
    assert [call[0] for call in loader.calls] == ['read_data'] and computer.calls == []


def test_load_polls_fixes_missing_become_a_warning():
    with pytest.warns(UserWarning, match='Poll fixes without a poll read: Celeste Tel 2026-09-21'):
        run(FakeLoader(fixes_missing=['Celeste Tel 2026-09-21']))


def test_load_polls_skipped_names_go_to_level_2():
    result, _, _, _, _, lines = run(FakeLoader(skipped={'pollsters': ['CIS (Sondaxe)']}))
    assert ("Skipped pollsters: {'CIS (Sondaxe)'}", 2) in lines
    assert not any(text.startswith('Skipped parties') for text, _ in lines)
    assert result['skipped'] == {'parties': [], 'pollsters': ['CIS (Sondaxe)'], 'sponsors': []}


def test_load_polls_without_report_prints_nothing(capsys):
    run(report=None)
    assert capsys.readouterr().out == ''


def test_load_polls_fails_without_a_polls_article():
    loader, computer = FakeLoader(), FakeComputer()
    result = load_polls('es', '2030-01-01', loader=loader, computer=computer, urls=URLS)
    assert result['status'] == 'failed' and result['reason'] == 'no polls article for 2030-01-01'
    assert loader.inits == [] and computer.calls == []


def test_has_polls_article_and_unmapped_reason():
    assert has_polls_article('es', '2026-11-29', URLS) is True
    assert has_polls_article('es', '2030-01-01', URLS) is False
    assert has_polls_article('es', '1999-01-01', URLS) is False
    assert has_polls_article('fr', '2026-11-29', URLS) is False
    assert unmapped_reason({'parties': ['X'], 'pollsters': ['Z', 'Y'], 'sponsors': ['W']}) == \
        'unmapped parties: X; unmapped pollsters: Y, Z; unmapped sponsors: W'
    assert unmapped_reason({'parties': [], 'pollsters': [], 'sponsors': ['W']}) == 'unmapped sponsors: W'


def test_add_catalogue_entries_inserts_without_truncating(monkeypatch):
    inserted = {}
    monkeypatch.setattr(load_module, 'load_criteria', lambda: {'scale': {'per_point': 5, 'floor': 20}})

    def fake(label):
        def insert(self, df, truncate=True):
            inserted[label] = (df, truncate)
        return insert

    monkeypatch.setattr(Pollsters, 'insert', fake('pollsters'))
    monkeypatch.setattr(Sponsors, 'insert', fake('sponsors'))
    added = add_catalogue_entries(['Sumar'], ['Artículo 14'])
    assert added == {'pollsters': ['Sumar'], 'sponsors': ['Artículo 14']}
    pollsters, truncate = inserted['pollsters']
    assert truncate is False and list(pollsters.columns) == ['id', 'name', 'mtype', 'quality', 'rating']
    assert pollsters['id'].isnull().all() and pollsters['mtype'].isnull().all() and pollsters['rating'].isnull().all()
    assert pollsters['name'].tolist() == ['Sumar'] and pollsters['quality'].tolist() == [unevaluated_quality()] == [20]
    sponsors, truncate = inserted['sponsors']
    assert truncate is False and list(sponsors.columns) == ['id', 'name'] and sponsors['id'].isnull().all()

    inserted.clear()
    assert add_catalogue_entries([], []) == {'pollsters': [], 'sponsors': []} and inserted == {}


def test_load_polls_refuses_years_with_overwrite_before_anything_runs():
    loader, computer = FakeLoader(), FakeComputer()
    with pytest.raises(ValueError, match='"years" cannot be combined with "overwrite"'):
        load_polls('es', '2030-01-01', overwrite=True, years=['2026'], loader=loader, computer=computer, urls=URLS)
    assert loader.inits == [] and computer.calls == []


def test_load_polls_survives_a_failing_parties_check():
    def broken(self):
        raise KeyError('PP')

    class BrokenChecks(FakeLoader):
        """`parties_checks` (a database query in the real loader) raises after the save."""

        def __call__(self, *args, **kwargs):
            loader = super().__call__(*args, **kwargs)
            type(loader).parties_checks = property(broken)
            return loader

    with pytest.warns(UserWarning) as record:
        result, loader, computer, *_ = run(BrokenChecks())
    assert result['status'] == 'loaded' and computer.calls == [(True, True)]
    assert [str(w.message) for w in record] == ["parties checks failed: KeyError: 'PP'"]


# --- parties y bmaps de params.json (`load_params`) ---

from mtpy.lib.load import NO_EVENTS, changed_keys, load_params, merge_event_params  # noqa: E402

DERIVED = {'parties': {'event': ['PP', 'PSOE'], 'polls': ['PP', 'PSOE', 'SUMAR']},
           'bmaps': {'main': ['PP', 'PSOE'], 'blocks': {'Derecha': ['PP'], 'Izquierda': ['PSOE', 'SUMAR']},
                     'vs': {'Derecha': ['PP'], 'Izquierda': ['PSOE', 'SUMAR']}}, 'smap': {'SUMAR': []}}
FILE = {'es': {'2026-11-29': {'parties': {'event': ['PP'], 'polls': ['PP']},
                              'bmaps': {'max': ['PP'], 'min': {'PP': 'PP'}, 'main': ['PP', 'PSOE'], 'blocks': {}, 'vs': {}},
                              'smap': {'FA': {'type': 'agg', 'names': ['SUMAR']}}},
               '2011-05-22': {'threshold': None, 'threshold_scope': 5}},
        'es-pv': {'2024-04-21': {'smap': {}}}}


class FakeParamsStore:
    def __init__(self, data):
        self.data = json.loads(json.dumps(data))
        self.written = None

    def read(self):
        return json.loads(json.dumps(self.data))

    def write(self, data):
        self.written = data


def test_merge_event_params_keeps_manual_keys_and_their_order():
    merged = merge_event_params(FILE['es']['2026-11-29'], DERIVED)
    assert list(merged) == ['parties', 'bmaps', 'smap']
    assert list(merged['bmaps']) == ['max', 'min', 'main', 'blocks', 'vs']
    assert merged['bmaps']['max'] == ['PP'] and merged['smap'] == FILE['es']['2026-11-29']['smap']
    assert merged['parties'] == DERIVED['parties'] and merged['bmaps']['vs'] == DERIVED['bmaps']['vs']
    assert list(merge_event_params({'threshold': 3}, DERIVED)) == ['parties', 'bmaps', 'threshold']
    assert changed_keys(FILE['es']['2026-11-29'], merged) == ['parties.event', 'parties.polls', 'bmaps.blocks', 'bmaps.vs']
    assert changed_keys({}, merged) == ['parties.event', 'parties.polls', 'bmaps.main', 'bmaps.blocks', 'bmaps.vs']


def test_merge_event_params_keeps_the_key_dates():
    """Las fechas clave de una elección, escritas a mano, sobreviven al autorrelleno."""
    entry = {'parties': {'event': ['A'], 'polls': ['A']}, 'bmaps': {'main': ['A'], 'blocks': {}, 'vs': {}},
             'dates': {'call': '2026-10-05', 'campaign': '2026-11-13'}}
    derived = {'parties': {'event': ['B'], 'polls': ['B']}, 'bmaps': {'main': ['B'], 'blocks': {}, 'vs': {}}}
    merged = merge_event_params(entry, derived)
    assert merged['dates'] == {'call': '2026-10-05', 'campaign': '2026-11-13'} and list(merged)[-1] == 'dates'


def test_load_params_rewrites_the_scope_and_reports_each_change():
    store = FakeParamsStore(FILE)
    seen = []
    derived = {'2026-11-29': DERIVED, '2023-07-23': DERIVED}

    def deriver(scope, overrides):
        seen.append((scope, overrides))
        return derived

    out = load_params('es', deriver=deriver, reader=store.read, writer=store.write,
                      report=lambda text, level=1: seen.append((level, text)))
    assert seen[0] == ('es', FILE['es'])
    assert out['status'] == 'loaded' and out['n_events'] == 2 and out['n_changed'] == 2 and out['n_skipped'] == 0
    assert out['changed'] == {'2026-11-29': ['parties.event', 'parties.polls', 'bmaps.blocks', 'bmaps.vs'],
                              '2023-07-23': ['parties.event', 'parties.polls', 'bmaps.main', 'bmaps.blocks', 'bmaps.vs']}
    assert set(out['seconds']) == {'read', 'derive', 'save', 'total'}
    assert list(store.written['es']) == ['2026-11-29', '2023-07-23', '2011-05-22']
    assert store.written['es']['2011-05-22'] == FILE['es']['2011-05-22'] and store.written['es-pv'] == FILE['es-pv']
    assert store.written['es']['2023-07-23'] == {'parties': DERIVED['parties'], 'bmaps': DERIVED['bmaps']}
    assert (1, 'es 2023-07-23: bmaps.main null -> ["PP", "PSOE"]') in seen
    assert (1, 'es 2026-11-29: parties.event ["PP"] -> ["PP", "PSOE"]') in seen


def test_load_params_dry_run_writes_nothing():
    store = FakeParamsStore(FILE)
    out = load_params('es', save=False, deriver=lambda scope, overrides: {'2026-11-29': DERIVED},
                      reader=store.read, writer=store.write)
    assert out['status'] == 'checked' and store.written is None and 'save' not in out['seconds']


def test_load_params_skips_an_election_without_matrix_and_keeps_its_entry():
    store = FakeParamsStore(FILE)
    empty = {'parties': {'event': [], 'polls': []}, 'bmaps': {'main': [], 'blocks': {}, 'vs': {}}, 'smap': {}}
    out = load_params('es', deriver=lambda scope, overrides: {'2026-11-29': empty, '2011-05-22': empty},
                      reader=store.read, writer=store.write)
    assert out['n_skipped'] == 2 and out['skipped'] == ['2026-11-29', '2011-05-22'] and out['n_changed'] == 0
    assert store.written['es'] == FILE['es']


def test_load_params_creates_a_missing_scope_with_dates_descending():
    store = FakeParamsStore(FILE)
    load_params('es-md', deriver=lambda scope, overrides: {'2023-05-28': DERIVED, '2027-05-23': DERIVED},
                reader=store.read, writer=store.write)
    assert list(store.written) == ['es', 'es-pv', 'es-md']
    assert list(store.written['es-md']) == ['2027-05-23', '2023-05-28']
    assert store.written['es-md']['2027-05-23'] == {'parties': DERIVED['parties'], 'bmaps': DERIVED['bmaps']}


def test_load_params_fails_without_elections():
    with pytest.raises(ValueError, match=NO_EVENTS.format('es-xx')):
        load_params('es-xx', deriver=lambda scope, overrides: {}, reader=lambda: {}, writer=lambda data: None)


# Comparación de lo leído con la base (`poll_changes`)

NO_CHANGES = {'changed': [], 'replaced': [], 'removed': [], 'keys': {'changed': [], 'replaced': []}}
CASAS = {15: '40dB', 3: 'Sigma Dos', 40: 'Hamalgama', 50: 'SW Demoscopia'}
MEDIOS = {1: 'Prisa', 2: 'El País', 3: 'El Mundo', 4: 'OKdiario', 5: 'La Séptima', None: None}


def poll_frame(rows):
    """Sondeos de `es` para el 29-N: cada fila es `(pollster_id, sponsor_id, start, end, sample_size, ctype)`."""
    frame = pd.DataFrame([{
        'event_date': pd.Timestamp('2026-11-29'), 'event_scope': 'es', 'date': pd.Timestamp(end),
        'pollster_id': pollster_id, 'sponsor_id': sponsor_id, 'pollster': CASAS[pollster_id],
        'sponsor': MEDIOS[sponsor_id], 'start_date': pd.Timestamp(start), 'end_date': pd.Timestamp(end),
        'sample_size': sample_size, 'ctype': ctype
    } for pollster_id, sponsor_id, start, end, sample_size, ctype in rows])
    return frame


def result_frame(rows):
    """Resultados: cada fila es `(pollster_id, sponsor_id, end, party, pct, seats_min, seats_max)`."""
    return pd.DataFrame([{
        'event_date': pd.Timestamp('2026-11-29'), 'event_scope': 'es', 'date': pd.Timestamp(end),
        'pollster_id': pollster_id, 'sponsor_id': sponsor_id, 'party_id': index, 'party': party, 'pct': pct,
        'seats_min': seats_min, 'seats_max': seats_max
    } for index, (pollster_id, sponsor_id, end, party, pct, seats_min, seats_max) in enumerate(rows)])


POLL_40DB = (15, 1, '2026-11-10', '2026-11-14', 2000, None)
POLL_SIGMA = (3, 3, '2026-11-03', '2026-11-13', 5370, None)
RES_40DB = [(15, 1, '2026-11-14', 'PP', 31.4, 125, 136), (15, 1, '2026-11-14', 'PSOE', 29.9, 117, 123)]
RES_SIGMA = [(3, 3, '2026-11-13', 'PP', 31.0, 126, 132), (3, 3, '2026-11-13', 'PSOE', 25.6, 105, 111)]
LABEL_40DB = {'pollster': '40dB', 'sponsor': 'Prisa', 'start_date': '2026-11-10', 'end_date': '2026-11-14'}


def test_poll_changes_without_differences_is_empty():
    polls, results = poll_frame([POLL_40DB, POLL_SIGMA]), result_frame(RES_40DB + RES_SIGMA)
    assert poll_changes(polls, results, polls.copy(), results.copy()) == NO_CHANGES


def test_poll_changes_reports_poll_fields():
    read = poll_frame([(15, 1, '2026-11-09', '2026-11-14', 2000, 'exit')])
    cur = poll_frame([(15, 1, '2026-11-10', '2026-11-14', None, None)])
    out = poll_changes(read, result_frame(RES_40DB), cur, result_frame(RES_40DB))
    assert out['replaced'] == [] and out['removed'] == [] and len(out['changed']) == 1
    assert {k: v for k, v in out['changed'][0].items() if k != 'changes'} == LABEL_40DB | {'start_date': '2026-11-09'}
    assert out['changed'][0]['changes'] == [
        {'field': 'start_date', 'party': None, 'old': '2026-11-10', 'new': '2026-11-09'},
        {'field': 'sample_size', 'party': None, 'old': None, 'new': 2000},
        {'field': 'ctype', 'party': None, 'old': None, 'new': 'exit'}]


def test_poll_changes_reports_parties_shares_and_seats():
    polls = poll_frame([POLL_40DB])
    read = result_frame([(15, 1, '2026-11-14', 'PP', 31.4, 125, 136), (15, 1, '2026-11-14', 'PSOE', 29.9, 117, 123),
                         (15, 1, '2026-11-14', 'SALF', 1.8, None, None)])
    cur = result_frame([(15, 1, '2026-11-14', 'PP', 31.0, None, None), (15, 1, '2026-11-14', 'PSOE', 29.9, 117, 123),
                        (15, 1, '2026-11-14', 'UP', 1.3, None, None)])
    out = poll_changes(polls, read, polls.copy(), cur)
    assert out['changed'] == [LABEL_40DB | {'changes': [
        {'field': 'pct', 'party': 'PP', 'old': 31.0, 'new': 31.4},
        {'field': 'seats', 'party': 'PP', 'old': None, 'new': [125, 136]},
        {'field': 'party', 'party': 'SALF', 'old': None, 'new': 1.8},
        {'field': 'party', 'party': 'UP', 'old': 1.3, 'new': None}]}]


def test_poll_changes_ignores_dtype_differences():
    read = poll_frame([POLL_40DB])
    read['sample_size'] = read['sample_size'].astype(float)
    read[['date', 'start_date', 'end_date']] = read[['date', 'start_date', 'end_date']].apply(lambda col: col.dt.date)
    read_results = result_frame([(15, 1, '2026-11-14', 'PP', 31.400000001, float('nan'), float('nan')),
                                 (15, 1, '2026-11-14', 'PSOE', 29.9, 117.0, 123.0)])
    cur = poll_frame([POLL_40DB])
    cur['sample_size'] = cur['sample_size'].astype('Int64')
    cur['pollster'], cur['sponsor'] = cur['pollster'].astype('category'), cur['sponsor'].astype('category')
    cur[['pollster_id', 'sponsor_id']] = cur[['pollster_id', 'sponsor_id']].astype('Int64')
    cur_results = result_frame([(15, 1, '2026-11-14', 'PP', 31.4, None, None),
                                (15, 1, '2026-11-14', 'PSOE', 29.9, 117, 123)])
    cur_results[['seats_min', 'seats_max']] = cur_results[['seats_min', 'seats_max']].astype('Int64')
    cur_results['party'] = cur_results['party'].astype('category')
    assert poll_changes(read, read_results, cur, cur_results) == NO_CHANGES


def test_poll_changes_matches_polls_without_sponsor():
    row, res = (15, None, '2026-11-10', '2026-11-14', 2000, None), [(15, None, '2026-11-14', 'PP', 31.4, 125, 136)]
    cur, cur_results = poll_frame([row]), result_frame(res)
    cur['sponsor_id'] = pd.array([pd.NA], dtype='Int64')
    cur_results['sponsor_id'] = pd.array([pd.NA], dtype='Int64')
    assert poll_changes(poll_frame([row]), result_frame(res), cur, cur_results) == NO_CHANGES


def test_poll_changes_returns_the_keys_to_update():
    read = poll_frame([POLL_40DB, POLL_SIGMA])
    cur = poll_frame([(15, 1, '2026-11-10', '2026-11-14', None, None), (3, 3, '2026-11-03', '2026-11-13', None, None),
                      (40, 4, '2026-10-20', '2026-10-22', 1000, None)])
    out = poll_changes(read, result_frame(RES_40DB + RES_SIGMA), cur,
                       result_frame(RES_40DB + RES_SIGMA + [(40, 4, '2026-10-22', 'PP', 33.0, None, None)]))
    # En el orden de `changed` (fin de campo): los retirados sin pareja no dan clave
    assert [poll['pollster'] for poll in out['changed']] == ['Sigma Dos', '40dB'] and len(out['removed']) == 1
    assert out['keys'] == {'changed': [('2026-11-13', 3, 3), ('2026-11-14', 15, 1)], 'replaced': []}


def test_poll_changes_new_and_removed_polls():
    gone = (40, 4, '2026-10-20', '2026-10-22', 1000, None)
    read, read_results = poll_frame([POLL_40DB, POLL_SIGMA]), result_frame(RES_40DB + RES_SIGMA)
    cur = poll_frame([POLL_40DB, gone])
    cur_results = result_frame(RES_40DB + [(40, 4, '2026-10-22', 'PP', 33.0, None, None)])
    out = poll_changes(read, read_results, cur, cur_results)
    assert out['changed'] == [] and out['replaced'] == []
    assert out['removed'] == [{'pollster': 'Hamalgama', 'sponsor': 'OKdiario', 'start_date': '2026-10-20',
                               'end_date': '2026-10-22'}]


def sponsor_change():
    """El 40dB de la base lo patrocina El País y el leído, Prisa: mismo sondeo con otro medio."""
    old = [(15, 2, '2026-11-14', party, pct, low, high) for _, _, _, party, pct, low, high in RES_40DB]
    return (poll_frame([POLL_40DB]), result_frame(RES_40DB),
            poll_frame([(15, 2, '2026-11-10', '2026-11-14', 2000, None)]), result_frame(old))


def test_poll_changes_pairs_a_sponsor_change():
    out = poll_changes(*sponsor_change())
    assert out['changed'] == [] and out['removed'] == []
    assert out['replaced'] == [LABEL_40DB | {'old': LABEL_40DB | {'sponsor': 'El País'}}]
    assert out['keys'] == {'changed': [], 'replaced': [('2026-11-14', 15, 2)]}


def test_poll_changes_pairs_a_longer_fieldwork():
    read = poll_frame([(50, 5, '2026-10-26', '2026-11-13', 1500, None)])
    cur = poll_frame([(50, 5, '2026-10-26', '2026-11-06', 1200, None)])
    out = poll_changes(read, result_frame([(50, 5, '2026-11-13', 'PP', 30.0, None, None)]),
                       cur, result_frame([(50, 5, '2026-11-06', 'PP', 30.5, None, None)]))
    assert out['removed'] == [] and len(out['replaced']) == 1
    assert out['replaced'][0]['end_date'] == '2026-11-13' and out['replaced'][0]['old']['end_date'] == '2026-11-06'


def test_poll_changes_leaves_ambiguous_pairs_apart():
    read = poll_frame([POLL_40DB, (15, 3, '2026-11-10', '2026-11-14', 2000, None)])
    read_results = result_frame(RES_40DB + [(15, 3, '2026-11-14', 'PP', 31.4, 125, 136)])
    cur = poll_frame([(15, 2, '2026-11-10', '2026-11-14', 2000, None)])
    out = poll_changes(read, read_results, cur, result_frame([(15, 2, '2026-11-14', 'PP', 31.4, 125, 136)]))
    assert out['replaced'] == [] and out['removed'] == [LABEL_40DB | {'sponsor': 'El País'}]


def test_poll_changes_without_removed():
    out = poll_changes(*sponsor_change(), removed=False)
    assert out == NO_CHANGES


def test_poll_changes_with_empty_database():
    read, read_results = poll_frame([POLL_40DB]), result_frame(RES_40DB)
    for cur, cur_results in ((None, None), (read.iloc[:0], read_results.iloc[:0])):
        assert poll_changes(read, read_results, cur, cur_results) == NO_CHANGES
    assert poll_changes(None, None, None, None) == NO_CHANGES
    assert poll_changes(read.iloc[:0], read_results.iloc[:0], read, read_results)['removed'] == [LABEL_40DB]


def test_poll_changes_with_duplicate_keys():
    read = poll_frame([POLL_40DB, (15, 1, '2026-11-10', '2026-11-14', 2500, None)])
    out = poll_changes(read, result_frame(RES_40DB + RES_40DB), poll_frame([POLL_40DB]), result_frame(RES_40DB))
    assert out == NO_CHANGES


def changed_sample():
    """Los cuatro frames de un 40dB que en la base no tiene muestra y en el artículo sí."""
    return (poll_frame([POLL_40DB]), result_frame(RES_40DB),
            poll_frame([(15, 1, '2026-11-10', '2026-11-14', None, None)]), result_frame(RES_40DB))


def test_load_polls_returns_the_changes():
    result = run(FakeLoader(changes=changed_sample()))[0]
    assert result['changed'] == [LABEL_40DB | {'changes': [
        {'field': 'sample_size', 'party': None, 'old': None, 'new': 2000}]}]
    assert result['replaced'] == [] and result['removed'] == []


def test_load_polls_with_years_skips_removed():
    frames = (poll_frame([POLL_40DB]).iloc[:0], result_frame(RES_40DB).iloc[:0], poll_frame([POLL_40DB]),
              result_frame(RES_40DB))
    assert run(FakeLoader(changes=frames))[0]['removed'] == [LABEL_40DB]
    assert run(FakeLoader(changes=frames), years=['2026'])[0]['removed'] == []


def test_load_polls_survives_a_failing_comparison(monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError('x')

    monkeypatch.setattr(load_module, 'poll_changes', boom)
    with pytest.warns(UserWarning, match='poll changes failed: ValueError: x'):
        result, loader = run()[:2]
    assert result['status'] == 'loaded' and result['changed'] == [] and result['removed'] == []
    assert result['updated'] is False and not {'select_changed', 'delete_polls'} & {call[0] for call in loader.calls}
    assert ('save_polls', False) in loader.calls and ('save_results', False) in loader.calls


def test_load_polls_failed_results_carry_empty_change_lists():
    result = run(FakeLoader(missing={'parties': ['X']}))[0]
    assert result['status'] == 'failed'
    assert result['changed'] == [] and result['replaced'] == [] and result['removed'] == []


def test_load_polls_dry_run_compares_too():
    result = run(FakeLoader(changes=changed_sample()), save=False)[0]
    assert result['status'] == 'checked' and len(result['changed']) == 1


def test_load_polls_survives_a_failing_read_of_the_current_results():
    loader = FakeLoader(changes=changed_sample())
    loader.results_error = TimeoutError('rds')
    with pytest.warns(UserWarning, match='poll changes failed: TimeoutError: rds'):
        result = run(loader)[0]
    assert result['status'] == 'loaded' and result['changed'] == [] and result['removed'] == []
    assert ('save_polls', False) in loader.calls and ('save_results', False) in loader.calls


KEY_40DB = ('2026-11-14', 15, 1)


def call_names(loader):
    return [call[0] for call in loader.calls]


def test_load_polls_updates_the_changed_polls():
    result, loader = run(FakeLoader(changes=changed_sample()))[:2]
    names = call_names(loader)
    assert ('select_changed', [KEY_40DB]) in loader.calls and ('delete_polls', [KEY_40DB]) in loader.calls
    assert names.index('select_changed') < names.index('delete_polls') < names.index('save_polls')
    # Dos nuevos del doble y el cambiado
    assert result['updated'] is True and result['n_saved'] == 3 and 'keys' not in result
    assert len(result['changed']) == 1


def test_load_polls_without_changes_touches_nothing_else():
    result, loader = run()[:2]
    assert result['updated'] is False and result['n_saved'] == 2
    assert not {'select_changed', 'delete_polls'} & set(call_names(loader))


def test_load_polls_deletes_the_poll_a_new_one_replaces():
    result, loader = run(FakeLoader(changes=sponsor_change()))[:2]
    assert ('delete_polls', [('2026-11-14', 15, 2)]) in loader.calls and ('select_changed', []) in loader.calls
    assert result['updated'] is True and result['n_saved'] == 2 and len(result['replaced']) == 1


def test_load_polls_dry_run_updates_nothing():
    result, loader = run(FakeLoader(changes=changed_sample()), save=False)[:2]
    names = call_names(loader)
    assert 'delete_polls' not in names and 'save_polls' not in names
    # Lo que se guardaría: los nuevos y el cambiado
    assert result['status'] == 'checked' and result['updated'] is False and result['n_saved'] == 3


def test_load_polls_overwrite_marks_updated_without_deleting():
    result, loader = run(FakeLoader(changes=changed_sample()), overwrite=True)[:2]
    assert not {'select_changed', 'delete_polls'} & set(call_names(loader))
    assert result['updated'] is True and result['n_saved'] == 5
    assert run(FakeLoader(changes=changed_sample()), overwrite=True, save=False)[0]['updated'] is False
    assert run(overwrite=True)[0]['updated'] is False


def many_polls(n_total, n_changed):
    """`n_total` sondeos de octubre en los dos lados; a los `n_changed` primeros les falta la muestra en la base."""
    rows = [(15, 1, '2026-10-{:02d}'.format(day), '2026-10-{:02d}'.format(day), 1000, None)
            for day in range(1, n_total + 1)]
    res = [(15, 1, '2026-10-{:02d}'.format(day), 'PP', 30.0, None, None) for day in range(1, n_total + 1)]
    cur = poll_frame(rows)
    cur['sample_size'] = [None] * n_changed + [1000] * (n_total - n_changed)
    return FakeLoader(changes=(poll_frame(rows), result_frame(res), cur, result_frame(res)))


def test_load_polls_brake_leaves_the_changes_unapplied():
    loader = many_polls(30, 11)
    with pytest.warns(UserWarning, match=r'11 changed polls not updated \(more than 10\): check the article or load '
                                         r'with overwrite'):
        result = run(loader)[0]
    assert not {'select_changed', 'delete_polls'} & set(call_names(loader)) and ('save_polls', False) in loader.calls
    assert result['status'] == 'loaded' and result['updated'] is False
    assert len(result['changed']) == 11 and result['n_saved'] == 2
    # Con diez, se aplican
    assert run(many_polls(30, 10))[0]['updated'] is True


def test_load_polls_brake_is_relative_in_a_short_cycle():
    # Un ciclo de 8 sondeos en el que cambian todos (una columna corrida en el artículo): más de la mitad
    loader = many_polls(8, 8)
    with pytest.warns(UserWarning, match=r'8 changed polls not updated \(more than 4\)'):
        result = run(loader)[0]
    assert 'delete_polls' not in call_names(loader) and result['updated'] is False
    assert run(many_polls(8, 4))[0]['updated'] is True
    # Con muy pocos sondeos en la base, hasta dos cambios se aplican
    assert run(many_polls(2, 2))[0]['updated'] is True


def test_poll_changes_rounds_shares_as_the_database_does():
    polls = poll_frame([POLL_40DB])
    read = result_frame([(15, 1, '2026-11-14', 'PP', 31.125, None, None),
                         (15, 1, '2026-11-14', 'UP', 2.675, None, None)])
    cur = result_frame([(15, 1, '2026-11-14', 'PP', 31.13, None, None),
                        (15, 1, '2026-11-14', 'UP', 2.68, None, None)])
    assert poll_changes(polls, read, polls.copy(), cur) == NO_CHANGES
    # También con los reales de 32 bits que devuelve la base y los que deja el formateo de lo leído
    read['pct'], cur['pct'] = read['pct'].astype('float32'), cur['pct'].astype('float32')
    assert poll_changes(polls, read, polls.copy(), cur) == NO_CHANGES
