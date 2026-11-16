"""Tests de los pipelines `load.Polls` y `load.Params` ejecutados sueltos, sin base."""
import pytest

from mtpy.lib.load import NO_EVENTS
from mtpy.pipelines.load.Params import Params
from mtpy.pipelines.load.Polls import MAX_CHANGES, MAX_POLLS, Polls, change_text, changed_line, poll_line, removed_line, replaced_line
from tests.test_load_unit import (
    POLL_40DB, POLL_SIGMA, RES_40DB, RES_SIGMA, FakeCatalogues, FakeComputer, FakeLoader, poll_frame, result_frame
)

EVENTS = {'es': '2026-11-29', 'es-md': '2027-05-23'}

PARAMS_RESULT = {'status': 'loaded', 'n_events': 3, 'n_changed': 2, 'n_skipped': 1, 'changed': {}, 'skipped': [],
                 'seconds': {'read': 0.1, 'derive': 0.2, 'save': 0.1, 'total': 0.4}}


class FakeParamsLoader:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or {}

    def __call__(self, scope, save=True, verbose=0, progress=None, report=None, **kwargs):
        self.calls.append((scope, save, verbose))
        if scope in self.fail:
            raise self.fail[scope]
        if report is not None:
            report('{} 2023-07-23: bmaps.vs null -> {{}}'.format(scope), 1)
        return dict(PARAMS_RESULT, status='loaded' if save else 'checked')


@pytest.fixture
def events(monkeypatch):
    monkeypatch.setattr('mtpy.pipelines.load.Polls.next_event_date', lambda scope: EVENTS.get(scope))
    monkeypatch.setattr('mtpy.lib.load.has_polls_article', lambda scope, event_date, urls=None: event_date != '2027-05-23')


def polls(scope, position=(1, 1), action='update', loader=None, **kwargs):
    p = Polls()
    p.run(action, chunk=scope, position=position, loader=loader or FakeLoader(), computer=FakeComputer(),
          catalogues=FakeCatalogues(), **{'verbose': 0, **kwargs})
    return p


def params(scope, loader, position=(1, 1), **kwargs):
    p = Params()
    p.run('update', chunk=scope, position=position, params_loader=loader, **kwargs)
    return p


def test_polls_loads_a_scope_and_builds_its_line(fresh_app, events, capsys):
    p = polls('es', position=(1, 2))
    assert p.result['status'] == 'loaded' and p.result['action'] == 'polls' and p.result['warnings'] == []
    assert p.line == ['es: polls loaded (5 read, 2 saved, 2026-11-29) in 0 s',
                      'es: + Casa 1, 2026-11-09 → 2026-11-11',
                      'es: + Casa 0 / Medio, 2026-11-10 → 2026-11-12']
    assert 'load polls: 2 ámbitos' in capsys.readouterr().out


def test_polls_lists_the_new_polls_below_the_line_from_verbose_1(fresh_app, events, capsys):
    polls('es', verbose=0)
    assert '+ Casa' not in capsys.readouterr().out
    polls('es', verbose=1)
    out = capsys.readouterr().out.splitlines()
    done = next(i for i, line in enumerate(out) if 'es: polls loaded' in line)
    assert out[done + 1].endswith('es: + Casa 1, 2026-11-09 → 2026-11-11')
    assert out[done + 2].endswith('es: + Casa 0 / Medio, 2026-11-10 → 2026-11-12')
    p = polls('es', loader=FakeLoader(new_polls=0))
    assert p.line == 'es: polls loaded (5 read, 0 saved, 2026-11-29) in 0 s' and p.result['new'] == []


def test_polls_dry_run_and_overwrite_list_only_the_polls_missing_from_the_database(fresh_app, events):
    p = polls('es', dry_run=True)
    assert p.line[1:] == ['es: + Casa 1, 2026-11-09 → 2026-11-11', 'es: + Casa 0 / Medio, 2026-11-10 → 2026-11-12']
    p = polls('es', action='rebulk', loader=FakeLoader(new_polls=1))
    assert p.line == ['es: polls loaded (5 read, 5 saved, 2026-11-29, overwrite) in 0 s',
                      'es: + Casa 0 / Medio, 2026-11-10 → 2026-11-12']


def test_poll_line_without_sponsor_or_with_one_day_of_fieldwork():
    assert poll_line('es', {'pollster': 'CIS', 'sponsor': None, 'start_date': '2026-11-03',
                            'end_date': '2026-11-03'}) == 'es: + CIS, 2026-11-03'
    assert poll_line('es', {'pollster': '40dB', 'sponsor': 'El País', 'start_date': None,
                            'end_date': '2026-11-12'}) == 'es: + 40dB / El País, 2026-11-12'


def test_change_text_formats():
    assert change_text({'field': 'sample_size', 'party': None, 'old': None, 'new': 2000}) == 'muestra — → 2000'
    assert change_text({'field': 'start_date', 'party': None, 'old': '2026-11-10', 'new': '2026-11-09'}) \
        == 'inicio 2026-11-10 → 2026-11-09'
    assert change_text({'field': 'ctype', 'party': None, 'old': None, 'new': 'exit'}) == 'tipo — → exit'
    assert change_text({'field': 'party', 'party': 'SALF', 'old': None, 'new': 1.8}) == '+ SALF 1,8'
    assert change_text({'field': 'party', 'party': 'UP', 'old': 1.3, 'new': None}) == '- UP'
    assert change_text({'field': 'pct', 'party': 'PP', 'old': 31.0, 'new': 31.4}) == 'PP 31 → 31,4'
    assert change_text({'field': 'pct', 'party': 'AA', 'old': 1.85, 'new': None}) == 'AA 1,85 → —'
    assert change_text({'field': 'seats', 'party': 'PP', 'old': None, 'new': [125, 136]}) == 'PP escaños — → 125–136'
    assert change_text({'field': 'seats', 'party': 'CC', 'old': [1, 1], 'new': [0, 1]}) == 'CC escaños 1 → 0–1'
    assert change_text({'field': 'seats', 'party': 'CC', 'old': [None, 2], 'new': [1, None]}) == 'CC escaños —–2 → 1–—'


def test_changed_replaced_and_removed_lines():
    poll = {'pollster': '40dB', 'sponsor': 'Prisa', 'start_date': '2026-11-10', 'end_date': '2026-11-14'}
    assert changed_line('es', poll | {'changes': [
        {'field': 'sample_size', 'party': None, 'old': None, 'new': 2000},
        {'field': 'party', 'party': 'SALF', 'old': None, 'new': 1.8}]}) \
        == 'es: ~ 40dB / Prisa, 2026-11-10 → 2026-11-14: muestra — → 2000; + SALF 1,8'
    assert removed_line('es', poll) == 'es: - 40dB / Prisa, 2026-11-10 → 2026-11-14'
    assert replaced_line('es', poll | {'old': poll | {'end_date': '2026-11-12'}}) \
        == 'es: + 40dB / Prisa, 2026-11-10 → 2026-11-14 (sustituye a 2026-11-10 → 2026-11-12)'
    assert replaced_line('es', poll | {'old': poll | {'sponsor': 'El País'}}) \
        == 'es: + 40dB / Prisa, 2026-11-10 → 2026-11-14 (sustituye a 40dB / El País, 2026-11-10 → 2026-11-14)'


def test_changed_line_caps_the_differences_of_a_poll():
    poll = {'pollster': '40dB', 'sponsor': 'Prisa', 'start_date': '2026-11-10', 'end_date': '2026-11-14'}
    changes = [{'field': 'party', 'party': 'P{}'.format(i), 'old': 1.0, 'new': None} for i in range(MAX_CHANGES + 3)]
    line = changed_line('es', poll | {'changes': changes})
    assert line.endswith('- P{}; … y 3 más'.format(MAX_CHANGES - 1)) and line.count(';') == MAX_CHANGES
    assert '…' not in changed_line('es', poll | {'changes': changes[:MAX_CHANGES]})


def test_polls_line_of_caps_the_changed_and_removed_polls(fresh_app, events):
    p = polls('es', loader=FakeLoader(new_polls=0))
    many = [{'pollster': 'Casa {:02d}'.format(i), 'sponsor': None, 'start_date': None, 'end_date': '2026-11-01',
             'changes': [{'field': 'sample_size', 'party': None, 'old': None, 'new': 1000}]}
            for i in range(MAX_POLLS + 2)]
    lines = p.line_of('es', p.result | {'changed': many, 'removed': many[:MAX_POLLS]})
    assert lines[0].endswith(' · {} changed, {} removed'.format(MAX_POLLS + 2, MAX_POLLS))
    changed = [line for line in lines if line.startswith('es: ~ ')]
    assert len(changed) == MAX_POLLS + 1 and changed[-1] == 'es: ~ … y 2 sondeos más'
    assert len([line for line in lines if line.startswith('es: - ')]) == MAX_POLLS and '… y' not in lines[-1]


def changes_loader():
    """Un 40dB que cambia de medio (nuevo y sustituto), un Sigma Dos sin muestra en la base y un Hamalgama retirado."""
    old_40db = [(15, 2, '2026-11-14', party, pct, low, high) for _, _, _, party, pct, low, high in RES_40DB]
    read, read_results = poll_frame([POLL_40DB, POLL_SIGMA]), result_frame(RES_40DB + RES_SIGMA)
    cur = poll_frame([(15, 2, '2026-11-10', '2026-11-14', 2000, None), (3, 3, '2026-11-03', '2026-11-13', None, None),
                      (40, 4, '2026-10-20', '2026-10-22', 1000, None)])
    cur_results = result_frame(old_40db + RES_SIGMA + [(40, 4, '2026-10-22', 'PP', 33.0, None, None)])
    return FakeLoader(new_polls=poll_frame([POLL_40DB]), changes=(read, read_results, cur, cur_results))


def test_polls_lists_changes_below_the_line_and_counts_them(fresh_app, events):
    p = polls('es', loader=changes_loader())
    assert p.line == [
        'es: polls loaded (5 read, 2 saved, 2026-11-29) in 0 s · 1 updated, 1 removed',
        'es: + 40dB / Prisa, 2026-11-10 → 2026-11-14 (sustituye a 40dB / El País, 2026-11-10 → 2026-11-14)',
        'es: ~ Sigma Dos / El Mundo, 2026-11-03 → 2026-11-13: muestra — → 5370',
        'es: - Hamalgama / OKdiario, 2026-10-20 → 2026-10-22']


def test_polls_lists_changes_without_new_polls(fresh_app, events):
    loader = changes_loader()
    loader.new_polls = 0
    p = polls('es', loader=loader)
    assert p.line[0] == 'es: polls loaded (5 read, 1 saved, 2026-11-29) in 0 s · 1 updated, 1 removed'
    assert [line.split(' ')[1] for line in p.line[1:]] == ['+', '~', '-']


def test_polls_dry_run_says_changed_and_overwrite_says_updated(fresh_app, events):
    p = polls('es', loader=changes_loader(), dry_run=True)
    # Uno nuevo y uno cambiado: se guardarían dos, pero nuevo sólo es uno
    assert p.line[0] == 'es: polls checked (5 read, 1 new, 2026-11-29, dry run) in 0 s · 1 changed, 1 removed'
    p = polls('es', action='rebulk', loader=changes_loader())
    assert p.line[0] == 'es: polls loaded (5 read, 5 saved, 2026-11-29, overwrite) in 0 s · 1 updated, 1 removed'


def test_polls_line_of_tolerates_a_result_without_change_lists(fresh_app, events):
    p = polls('es')
    result = {key: value for key, value in p.result.items() if key not in ('changed', 'replaced', 'removed')}
    assert p.line_of('es', result) == p.line


def test_polls_rebulk_overwrites_and_flags_the_header(fresh_app, events, capsys):
    p = polls('es', action='rebulk', force=True, dry_run=True)
    assert p.line[0] == 'es: polls checked (5 read, 5 new, 2026-11-29, overwrite, dry run) in 0 s'
    assert 'load polls: 1 ámbito · overwrite · force · dry run' in capsys.readouterr().out
    assert polls('es', action='rebulk').line[0] == 'es: polls loaded (5 read, 5 saved, 2026-11-29, overwrite) in 0 s'


def test_polls_without_article_fails_or_skips_with_all(fresh_app, events):
    result = polls('es-md').result
    assert result['status'] == 'failed' and result['action'] == 'polls' and result['warnings'] == []
    assert result['reason'] == 'no polls article for 2027-05-23'
    p = polls('es-md', skip_missing=True)
    assert p.result['status'] == 'skipped' and p.line == 'es-md: polls skipped (no polls article for 2027-05-23)'


def test_polls_without_upcoming_event_ends_in_setup(fresh_app, events):
    loader = FakeLoader()
    p = polls('es-xx', loader=loader)
    assert p.result == {'status': 'failed', 'action': 'polls', 'reason': 'no upcoming event', 'warnings': []}
    assert loader.inits == []
    assert polls('es-xx', skip_missing=True).result['status'] == 'skipped'


def test_polls_event_lookup_error_fails_the_scope(fresh_app, events, monkeypatch):
    monkeypatch.setattr('mtpy.pipelines.load.Polls.next_event_date', lambda scope: {}['x'])
    p = polls('es')
    assert p.result['status'] == 'failed' and p.result['reason'] == "KeyError: 'x'" and 'failed' in p.line


def test_polls_event_date_parameter(fresh_app, events):
    loader = FakeLoader()
    polls('es', loader=loader, event_date='2023-07-23')
    assert loader.inits[0][:2] == ('es', '2023-07-23')


def test_polls_failure_name_is_scope_and_action(fresh_app):
    assert Polls().failure_name('es') == 'es (polls)' and Params().failure_name('es') == 'es (params)'


def test_polls_years_are_passed_to_the_loader(fresh_app, events):
    loader = FakeLoader()
    polls('es', loader=loader, years=['2025', '2026'])
    assert loader.inits[0][2] == ['2025', '2026']


def test_polls_unmapped_names_fail_the_scope_and_force_adds_them(fresh_app, events, capsys):
    p = polls('es', loader=FakeLoader(missing={'pollsters': ['Sumar']}), verbose=1)
    out = capsys.readouterr().out
    assert p.result['status'] == 'failed' and p.line == 'es: polls failed (unmapped pollsters: Sumar)'
    assert 'es: polls failed (unmapped pollsters: Sumar)' in out
    assert "Missing Pollsters: {'Sumar'}" in out
    p = polls('es', loader=FakeLoader(missing={'pollsters': ['Sumar']}), force=True)
    assert p.line[0] == 'es: polls loaded (5 read, 2 saved, 2026-11-29) in 0 s · 1 pollster added · 1 aviso'
    assert p.result['added']['pollsters'] == ['Sumar'] and len(p.result['warnings']) == 1


def test_polls_line_counts_added_pollsters_and_sponsors_to_add(fresh_app, events):
    missing = {'pollsters': ['Sumar', 'UJA'], 'sponsors': ['Artículo 14']}
    assert ' · 2 pollsters added, 1 sponsor added' in polls('es', loader=FakeLoader(missing=missing), force=True).line[0]
    p = polls('es', loader=FakeLoader(missing=missing), force=True, dry_run=True)
    assert 'dry run) in 0 s · 2 pollsters to add, 1 sponsor to add' in p.line[0]


def test_polls_verbosity_levels(fresh_app, events, capsys):
    polls('es', verbose=0)
    out = capsys.readouterr().out
    assert 'empezando' not in out and 'Missing' not in out
    polls('es', verbose=1)
    out = capsys.readouterr().out
    assert '[1/1] es: empezando' in out and '# Polls: 5' in out and 'Missing Parties: set()' in out
    assert 'lectura (' not in out
    polls('es', loader=FakeLoader(skipped={'pollsters': ['CIS (x)']}), verbose=2)
    out = capsys.readouterr().out
    for label in ('lectura', 'nombres', 'series', 'guardado', 'pesos'):
        assert '[1/1] es: {} ('.format(label) in out, label
    assert 'Skipped pollsters' in out
    loader = FakeLoader()
    polls('es', loader=loader, verbose=3)
    assert loader.inits[0][3] == 1  # el loader recibe verbose - 2


def test_polls_a_failing_loader_becomes_a_failed_result(fresh_app, events, monkeypatch):
    monkeypatch.setattr('mtpy.lib.load.has_polls_article', lambda scope, event_date, urls=None: True)

    class Exploding(FakeLoader):
        def __call__(self, scope, *args, **kwargs):
            raise RuntimeError('boom')

    p = polls('es-md', loader=Exploding())
    assert p.result['status'] == 'failed' and p.result['reason'] == 'RuntimeError: boom'
    assert p.line == 'es-md: polls failed (RuntimeError: boom)'


def test_params_loads_a_scope_and_builds_its_line(fresh_app, capsys):
    loader = FakeParamsLoader()
    p = params('es', loader, position=(1, 2), verbose=1)
    assert loader.calls == [('es', True, 0)] and p.result['status'] == 'loaded'
    assert p.line == 'es: params loaded (3 elections, 2 changed, 1 skipped) in 0 s'
    out = capsys.readouterr().out
    assert 'load params: 2 ámbitos' in out and 'es 2023-07-23: bmaps.vs null -> {}' in out


def test_params_dry_run_checks_and_flags_the_header(fresh_app, capsys):
    loader = FakeParamsLoader()
    p = params('es', loader, dry_run=True)
    assert loader.calls == [('es', False, 0)]
    assert 'load params: 1 ámbito · dry run' in capsys.readouterr().out
    assert p.line.startswith('es: params checked (3 elections, 2 changed, 1 skipped, dry run)')


def test_params_scope_without_elections_is_skipped_with_all_and_fails_when_named(fresh_app):
    loader = FakeParamsLoader(fail={'es-md': ValueError(NO_EVENTS.format('es-md'))})
    p = params('es-md', loader, skip_missing=True)
    assert p.result['status'] == 'skipped' and p.line == 'es-md: params skipped (no elections for es-md)'
    assert params('es-md', loader).result['status'] == 'failed'


def test_params_isolates_an_exception(fresh_app):
    p = params('es', FakeParamsLoader(fail={'es': KeyError('boom')}))
    assert p.result['status'] == 'failed' and p.line == "es: params failed (KeyError: 'boom')"
