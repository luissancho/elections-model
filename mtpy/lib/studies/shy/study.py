"""
The shy-vote study in one call: `compute_shy` reads the published parts through the bundle reader (`data`) and chains
the measures of `stats` for every election with a result and final polls: the error of the left and right blocks, of
the four lineages (PSOE, PP, Vox and the left of the PSOE) and of the party in government, then the summary over all of
them, the long table of figures and the three verdicts. `ShyStudy` carries the results; this module only reads and
wires, every figure comes from `data` and `stats`.

The rows publish the errors rounded to 2 decimals, but the figures (means, p values, sizes) are computed over the
unrounded errors and rounded last. Negative means underestimated.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ...bundle import BundleReader
from ....core.io import FileSystem
from .data import (Election, MissingInput, load_catalogue, load_incumbents, load_params, read_current,
                   read_elections)
from .stats import extremes, figures, group_error, size

# The keys of the error of an election row, blocks first
BLOCKS = ('Izquierda', 'Derecha')
LINEAGES = ('PSOE', 'PP', 'VOX', 'UP')
KEYS = BLOCKS + LINEAGES
# Labels of the rows of ``summary['parties']`` that are not a lineage
PARTY_LABELS = {'Izquierda': 'Izquierda (bloque)', 'Derecha': 'Derecha (bloque)',
                'incumbent_general': 'Gobierno, generales', 'incumbent_regional': 'Gobierno, autonómicas'}
PARTY_KEYS = ('PSOE', 'PP', 'VOX', 'UP', 'Izquierda', 'Derecha', 'incumbent_general', 'incumbent_regional')
MONTHS = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre',
          'noviembre', 'diciembre']


@dataclass
class ShyStudy:
    """
    Results of the shy-vote study, the blocks of the published parts.

    Attributes
    ----------
    params : dict
        Parameters of the study, ``data/voto-oculto/params.json`` updated with the overrides.
    scopes : list of str
        The catalogue scopes with at least one election entered, in catalogue order.
    election : dict or None
        ``{date, days_left, as_of, run_id}`` of the election in progress; ``None`` without it.
    summary : dict
        ``n_elections``, ``n_general``, ``n_regional``, ``n_scopes``, ``first_year``, ``first_year_regional``,
        ``last_year``, ``left``, ``right`` and ``lean`` (four figures blocks each), ``vox``, ``parties``, ``size``,
        ``psoe_general``, ``pp``, ``podemos``, ``current`` and ``verdicts``. ``current['values']`` holds the main
        parties of the run (mean of their non-null shares) and the blocks of ``params['blocks']`` (sum of the means
        of their members, see ``_current``).
    elections : list of dict
        One row per election, chronological (``election_row``).
    groups : list of dict
        The figures blocks of ``summary`` flattened: ``table``, ``key``, ``label`` and the figures.
    inputs : dict
        ``inputs`` of ``read_elections`` by scope, ``run`` (of ``read_current``) and ``incumbents`` (the
        ``{scope, date}`` of the elections without a party in government found).
    """

    params: dict
    scopes: list[str]
    election: Optional[dict]
    summary: dict
    elections: list[dict]
    groups: list[dict]
    inputs: dict


def _round(value: Optional[float]) -> Optional[float]:
    """``value`` rounded to 2 decimals, ``None`` kept."""
    return None if value is None else round(float(value), 2)


def _present_sum(names: list, results: dict) -> Optional[float]:
    """Sum of the results of the members present in ``results``; ``None`` when none is."""
    present = [results[name] for name in names if name in results]

    return sum(present) if present else None


def _measure_row(election: Election, params: dict, incumbents: dict) -> dict:
    """The row of ``election_row`` with the errors and sums unrounded (the figures are computed over them)."""
    polls, results = election.polls, election.results
    members = {block: (election.blocks or {}).get(block, []) for block in BLOCKS} | dict(params['lineages'])
    errors, names = {}, {}

    for key in KEYS:
        error, joined = group_error(polls, members[key], results)
        errors[key] = error
        if key in params['lineages']:
            names[key] = joined

    lean = None
    if None not in (errors['Izquierda'], errors['Derecha']):
        lean = errors['Izquierda'] - errors['Derecha']

    row_names = incumbents.get(election.scope, {}).get(election.event_date, [])
    incumbent = next((name for name in row_names if name in results), None)
    incumbent_error = None
    if incumbent is not None:
        incumbent_error = group_error(polls, [incumbent], results)[0]

    sums = {key: _present_sum(members[key], results) for key in KEYS}

    return {
        'scope': election.scope,
        'scope_name': election.scope_name,
        'event_date': election.event_date,
        'label': election.label,
        'short': election.short,
        'general': election.general,
        'n_cycle': election.n_cycle,
        'n_final': int(len(polls)),
        'blocks': election.blocks,
        **errors,
        'lean': lean,
        'names': names,
        'incumbent': incumbent,
        'incumbent_error': incumbent_error,
        'results': sums,
        'polls': {key: None if errors[key] is None else sums[key] + errors[key] for key in KEYS}
    }


def election_row(election: Election, params: dict, incumbents: dict) -> dict:
    """
    One row of the study: the errors of an election.

    Parameters
    ----------
    election : Election
        The election with its final polls and result.
    params : dict
        Parameters of the study (``lineages``).
    incumbents : dict
        ``{scope: {date: [names]}}`` (``load_incumbents``).

    Returns
    -------
    dict
        Identification (``scope``, ``scope_name``, ``event_date``, ``label``, ``short``, ``general``, ``n_cycle``,
        ``n_final``, ``blocks``), the errors ``Izquierda``, ``Derecha``, ``lean`` (left minus right), ``PSOE``, ``PP``,
        ``VOX`` and ``UP`` (``None`` when no poll estimates the group), ``names`` (the members of every lineage found
        in the result joined with ``+``, ``None`` if none), ``incumbent`` and ``incumbent_error`` (the first name of
        the table of the party in government present in the result; ``None`` without it), ``results`` (the sum of
        the results of the present members of each key) and ``polls`` (``results`` plus the error: the mean of the
        final polls counting the result of the members a poll does not estimate); every figure with 2 decimals.
    """
    return _rounded(_measure_row(election, params, incumbents))


def _rounded(row: dict) -> dict:
    """A row of ``_measure_row`` with its errors, results and polls rounded to 2 decimals."""
    out = {key: _round(value) if isinstance(value, float) else value for key, value in row.items()}
    out['results'] = {key: _round(value) for key, value in row['results'].items()}
    out['polls'] = {key: None if value is None else _round(out['results'][key] + out[key])
                    for key, value in row['polls'].items()}

    return out


def verdicts(left_all: Optional[dict], vox: Optional[dict], parties: list, labels: dict) -> dict:
    """
    The three answers of the study.

    Parameters
    ----------
    left_all : dict or None
        Figures of the error of the left block over every election.
    vox : dict or None
        Figures of the error of Vox; ``None`` without data.
    parties : list of dict
        The ``summary['parties']`` rows (``key`` and ``figures``).
    labels : dict
        ``{key: public label}`` of the parties.

    Returns
    -------
    dict
        ``p1`` («SÍ» if the left is significantly underestimated, else «NO»), ``p2`` (the same for Vox, «SIN DATOS»
        without figures) and ``p3`` («DE NADIE» or one «DE {LABEL}» per underestimated party joined by « Y »).
    """
    def under(figs: Optional[dict]) -> bool:
        return bool(figs) and figs['mean'] < 0 and figs['p_t'] is not None and figs['p_t'] < 0.05

    hidden = [f"DE {labels.get(row['key'], row['key']).upper()}" for row in parties
              if row['key'] in LINEAGES and under(row['figures'])]

    return {'p1': 'SÍ' if under(left_all) else 'NO',
            'p2': 'SIN DATOS' if vox is None else ('SÍ' if under(vox) else 'NO'),
            'p3': ' Y '.join(hidden) if hidden else 'DE NADIE'}


def _figures_of(rows: list, key: str, threshold: float) -> Optional[dict]:
    """Figures of the errors under ``key`` of the rows."""
    return figures([row[key] for row in rows], threshold)


def _subsets(rows: list, recent_from: str) -> dict:
    """The four subsets of the rows: ``all``, ``general``, ``regional`` and ``recent``."""
    return {'all': rows, 'general': [row for row in rows if row['general']],
            'regional': [row for row in rows if not row['general']],
            'recent': [row for row in rows if row['event_date'] >= recent_from]}


def _current(current: Optional[dict], blocks: list) -> Optional[dict]:
    """
    The block ``summary['current']`` from the election in progress; ``None`` without it.

    ``values`` holds, rounded to 1 decimal, the mean of the non-null shares of each ``main`` party of the run and,
    for each block of ``blocks`` (``params['blocks']``) in the ``bmaps`` of the run, the sum of the means of its
    members with an estimate: a poll that omits a member does not pull the block down. A block is present only when
    one of its members has an estimate; the other blocks of the run are left out.
    """
    if current is None:
        return None

    polls = current['polls']
    as_of = pd.Timestamp(current['as_of'])
    days_left = int((pd.Timestamp(current['event_date']) - as_of).days)
    values = {}

    if not polls.empty:
        run_blocks = current['blocks'] or {}
        for block in blocks:
            means = [polls[name].mean() for name in run_blocks.get(block, [])
                     if name in polls.columns and polls[name].notna().any()]
            if means:
                values[block] = round(float(sum(means)), 1)
        for name in current['main']:
            if name in polls.columns and polls[name].notna().any():
                values[name] = round(float(polls[name].mean()), 1)
        ends = polls['end_date'].where(polls['end_date'].notna(), polls['date'])
        first, last = ends.min(), ends.max()
    else:
        first = last = None

    return {'as_of': current['as_of'], 'days_left': days_left, 'n_polls': int(len(polls)), 'from': first, 'to': last,
            'values': values, 'names': list(current['main'])}


def _recent_label(recent_from: str) -> str:
    """«Desde septiembre de 2018» from the date ``recent_from``."""
    date = pd.Timestamp(recent_from)

    return f'Desde {MONTHS[date.month - 1]} de {date.year}'


def _groups(summary: dict, params: dict) -> list:
    """The figures blocks of the summary as long rows ``{table, key, label, <figures>}``."""
    subsets = {'all': 'Todas las elecciones', 'general': 'Generales', 'regional': 'Autonómicas',
               'recent': _recent_label(params['recent_from'])}
    up = params['labels'].get('UP', 'UP')
    year = params['recent_from'][:4]
    entries = []

    for table, prefix in (('left', 'Izquierda'), ('right', 'Derecha'), ('lean', 'Izquierda − Derecha')):
        entries += [(table, key, f'{prefix}: {text}', summary[table][key]) for key, text in subsets.items()]
    entries.append(('vox', 'all', params['labels'].get('VOX', 'VOX'), summary['vox']))
    entries += [('parties', row['key'], row['label'], row['figures']) for row in summary['parties']]
    entries += [('podemos', 'before', f'{up}: Antes de {year}', summary['podemos']['before']),
                ('podemos', 'general_recent', f'{up}: Generales desde {year}', summary['podemos']['general_recent']),
                ('podemos', 'regional_recent', f'{up}: Autonómicas desde {year}',
                 summary['podemos']['regional_recent']),
                ('pp', 'late', f"PP: Desde {params['pp_early'][1][:4]}", summary['pp']['late']),
                ('pp', 'all', 'PP: Todas las elecciones', summary['pp']['all'])]

    return [{'table': table, 'key': key, 'label': label, **{name: figs[name] for name in
             ('n', 'under', 'over', 'mean', 'median', 'p_t', 'p_sign', 'size', 'size_over')}}
            for table, key, label, figs in entries if figs is not None]


def compute_shy(reader: BundleReader, data: FileSystem, params: Optional[dict] = None) -> ShyStudy:
    """
    Compute the shy-vote study from the published bundle.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    data : FileSystem
        The ``data/`` directory of the repository (parameters, party in government, scopes).
    params : dict, optional
        Overrides of the parameters file.

    Returns
    -------
    ShyStudy
        The results.

    Raises
    ------
    MissingInput
        ``'events'`` when no election has published parts and final polls.
    """
    params = load_params(data) | (params or {})
    elections, inputs = read_elections(reader, load_catalogue(data), params)

    if not elections:
        raise MissingInput('events')

    table = load_incumbents(data)
    current, run_inputs = read_current(reader, params)
    raw = [_measure_row(election, params, table) for election in elections]
    rows = [_rounded(row) for row in raw]
    threshold = params['size_threshold']
    labels = params['labels']
    recent_from = params['recent_from']

    def block(key: str) -> dict:
        return {name: _figures_of(subset, key, threshold) for name, subset in _subsets(raw, recent_from).items()}

    general = [row for row in raw if row['general']]
    regional = [row for row in raw if not row['general']]
    vox = _figures_of(raw, 'VOX', threshold)
    if vox is not None:
        vox = vox | {'extremes': extremes(rows, 'VOX', params['extremes'])}

    parties = [{'key': key, 'label': PARTY_LABELS.get(key, labels.get(key, key)),
                'figures': _figures_of({'incumbent_general': general, 'incumbent_regional': regional}.get(key, raw),
                                       'incumbent_error' if key.startswith('incumbent') else key, threshold)}
               for key in PARTY_KEYS]

    early_from, early_to = params['pp_early']
    psoe_general = _figures_of(general, 'PSOE', threshold)
    big = [row['label'] for row in rows
           if row['general'] and row['PSOE'] is not None and row['PSOE'] < params['big_threshold']]
    before = [row for row in raw if row['event_date'] < recent_from]
    since = [row for row in raw if row['event_date'] >= recent_from]

    summary = {
        'n_elections': len(rows),
        'n_general': len(general),
        'n_regional': len(regional),
        'n_scopes': len({row['scope'] for row in rows}),
        'first_year': int(rows[0]['event_date'][:4]),
        'first_year_regional': int(regional[0]['event_date'][:4]) if regional else None,
        'last_year': int(rows[-1]['event_date'][:4]),
        'left': block('Izquierda'),
        'right': block('Derecha'),
        'lean': block('lean'),
        'vox': vox,
        'parties': parties,
        'size': size(raw, params['size_parties'], threshold),
        'psoe_general': None if psoe_general is None else psoe_general | {'big': big},
        'pp': {'early': {'from': early_from, 'to': early_to,
                         'errors': [{'label': row['label'], 'error': row['PP']} for row in rows if row['general']
                                    if early_from <= row['event_date'] < early_to and row['PP'] is not None]},
               'late': _figures_of([row for row in general if row['event_date'] >= early_to], 'PP', threshold),
               'all': _figures_of(raw, 'PP', threshold)},
        'podemos': {'before': _figures_of(before, 'UP', threshold),
                    'general_recent': _figures_of([row for row in since if row['general']], 'UP', threshold),
                    'regional_recent': _figures_of([row for row in since if not row['general']], 'UP', threshold)},
        'current': _current(current, params['blocks'])
    }
    summary['verdicts'] = verdicts(summary['left']['all'], summary['vox'], parties, labels)

    election = None
    if current is not None:
        election = {'date': current['event_date'], 'days_left': summary['current']['days_left'],
                    'as_of': current['as_of'], 'run_id': current['run_id']}

    missing = [{'scope': row['scope'], 'date': row['event_date']} for row in rows if row['incumbent'] is None]
    scopes = [scope for scope, found in inputs.items() if found.get('events')]

    return ShyStudy(params=params, scopes=scopes, election=election, summary=summary, elections=rows,
                    groups=_groups(summary, params), inputs=inputs | {'run': run_inputs, 'incumbents': missing})
