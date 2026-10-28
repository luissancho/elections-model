"""Tests de `mtpy.lib.load`: la secuencia de carga de sondeos de Wikipedia, con un loader y un `Computer` falsos."""
import json
import warnings

import pandas as pd
import pytest

from mtpy.lib import load as load_module
from mtpy.lib.load import (
    LOAD_STEPS, add_catalogue_entries, has_polls_article, load_polls, unevaluated_quality, unmapped_reason
)
from mtpy.models.elections import Pollsters, Sponsors

URLS = {'es': {'2026-11-29': {'polls': 'u'}, '2030-01-01': {'results': 'r'}}}
CLASSES = ('parties', 'pollsters', 'sponsors')


class FakeLoader:
    """Fábrica con la firma de `WikipediaLoader`: anota las construcciones (`inits`) y las llamadas (`calls`)."""

    def __init__(self, data=5, missing=None, skipped=None, fixes_missing=None, new_polls=2):
        self.inits, self.calls, self.instance = [], [], None
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
                self.polls = pd.DataFrame({'x': range(factory.n if overwrite else factory.new_polls)})
                return self

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
    assert loader.calls == [('read_data', None), ('build_series',), ('select_series', False), ('save_polls', False),
                            ('save_results', False)]
    assert computer.calls == [(True, True)] and computer.inits == [('es', ['2026-11-29'], 0, '.')]
    assert (result['status'], result['action'], result['n_read'], result['n_saved'], result['overwrite']) == \
        ('loaded', 'polls', 5, 2, False)
    assert result['event_date'] == '2026-11-29' and 'reason' not in result
    assert set(result['seconds']) == set(LOAD_STEPS) | {'total'}
    assert ('# Polls: 5', 1) in lines and ('Missing Pollsters: set()', 1) in lines
    assert ('Missing Parties: set()', 1) in lines and ('Missing Sponsors: set()', 1) in lines
    assert (loader.instance.parties_checks.to_string(), 2) in lines


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


def test_merge_event_params_keeps_the_manual_party_lists():
    entry = {'parties': {'event': ['PP'], 'polls': ['PP', 'AC'], 'include': ['COMPROMIS'], 'exclude': ['AC']},
             'bmaps': DERIVED['bmaps']}
    merged = merge_event_params(entry, DERIVED)
    assert list(merged['parties']) == ['event', 'polls', 'include', 'exclude']
    assert merged['parties']['exclude'] == ['AC'] and merged['parties']['include'] == ['COMPROMIS']
    assert changed_keys(entry, merged) == ['parties.event', 'parties.polls']
