"""
The ban study in one call: `compute_ban` reads the published elections through the bundle reader (`data`) and chains
the measures of `stats`: the pairs of every clandestine poll with the last legal one of its house, the clandestine
consensus against the frozen average of the general elections with a model, and the rows by election, by house and
of the big samples, with the summary over all of them. `BanStudy` carries the results; this module only reads and
wires, every figure comes from `data` and `stats`, unrounded (the exporters round).
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ...bundle import BundleReader
from ....core.io import FileSystem
from .data import Event, MissingInput, load_params, read_events, read_next
from .stats import (as_float, as_int, as_str, big_rows, cell_rows, consensus, election_row, house_rows, model_errors,
                    official_figures, pair_rows, period_row, summarize)


@dataclass
class BanStudy:
    """
    Results of the ban study, the blocks of the published parts.

    Attributes
    ----------
    params : dict
        Parameters of the study, ``data/andorra/params.json`` updated with the overrides.
    inputs : dict
        By scope, the elections read with the parts found and those skipped with the reason (``read_events``).
    next : dict or None
        ``{event_date, open}`` of the election of the latest national run (``read_next``).
    summary : dict
        The summary (``stats.summarize``).
    pairs : list of dict
        Every pair (``stats.pair_rows``), by ``event_date`` descending and ``d`` ascending.
    elections : list of dict
        One row per election with at least one clandestine poll (``stats.election_row``), by ``event_date``
        descending.
    period : dict
        ``{since, k, n, mean_d}`` of the general elections before ``chart_since`` (``stats.period_row``).
    cells : list of dict
        The cells of the general elections with a model (``stats.cell_rows``), by election ascending and party in
        the order of ``main``.
    houses : list of dict
        The houses with ``min_pairs`` pairs or more (``stats.house_rows``).
    big : list of dict
        The pairs of the big samples (``stats.big_rows``).
    latest : dict or None
        The last national election with clandestine polls: ``{scope, event_date, label, parties, polls, consensus,
        frozen, official, error}``; ``None`` without any.
    """

    params: dict
    inputs: dict
    next: Optional[dict]
    summary: dict
    pairs: list
    elections: list
    period: dict
    cells: list
    houses: list
    big: list
    latest: Optional[dict]


def _figures(series) -> Optional[dict]:
    """A ``{party: share}`` of a Series of shares, ``None`` for ``None``."""
    return None if series is None else {party: float(value) for party, value in series.items()}


def _latest_polls(event: Event) -> list:
    """The clandestine polls of an election as the ``latest`` block lists them, in the order of the part."""
    return [{'pollster_id': int(poll['pollster_id']), 'pollster': as_str(poll['pollster']),
             'sponsor': as_str(poll['sponsor']), 'date': as_str(poll['date']), 'days': as_int(poll['days']),
             'sample_size': as_int(poll['sample_size']), 'mtype': as_str(poll['mtype']),
             'error_avg': as_float(poll['error_avg']),
             'pct': {party: float(poll[party]) for party in event.main if pd.notna(poll[party])}}
            for _, poll in event.polls.iterrows()]


def latest_block(event: Event, figures: Optional[dict], params: dict) -> dict:
    """
    The ``latest`` block of an election.

    Parameters
    ----------
    event : Event
        The election.
    figures : dict or None
        Its clandestine consensus (``stats.consensus``).
    params : dict
        Parameters of the study.

    Returns
    -------
    dict
        ``scope``, ``event_date``, ``label``, ``parties`` (``main``), ``polls`` (every clandestine poll:
        ``pollster_id``, ``pollster``, ``sponsor``, ``date``, ``days``, ``sample_size``, ``mtype``, ``error_avg``
        and ``pct``), ``consensus``, ``frozen`` (``{party: model}`` or ``None``), ``official`` (the figures the
        errors compare with) and ``error`` (``{legal, frozen, consensus, with}``, ``stats.model_errors``).
    """
    official = official_figures(event)
    frozen = event.frozen['model'] if event.frozen is not None else None

    return {'scope': event.scope, 'event_date': event.event_date, 'label': event.label, 'parties': list(event.main),
            'polls': _latest_polls(event), 'consensus': figures, 'frozen': _figures(frozen),
            'official': _figures(official) or {}, 'error': model_errors(event, figures, params)}


def compute_ban(reader: BundleReader, data: FileSystem, params: Optional[dict] = None) -> BanStudy:
    """
    Compute the ban study from the published bundle.

    Parameters
    ----------
    reader : BundleReader
        Reader of the bundle.
    data : FileSystem
        The ``data/`` directory of the repository (parameters and the scopes catalogue ``es-scopes.csv``, for the
        names of the regional scopes in the labels).
    params : dict, optional
        Overrides of the parameters file.

    Returns
    -------
    BanStudy
        The results.

    Raises
    ------
    MissingInput
        ``'pairs'`` when no election has a pair of polls.
    """
    params = load_params(data) | (params or {})
    names = data.read_csv('es-scopes.csv').set_index('scode')['name'].to_dict()
    events, inputs = read_events(reader, params, names)

    pairs, elections, cells = [], [], []
    latest = None

    for event in events:
        own = pair_rows(event, params)
        figures = consensus(event, params)
        pairs += own
        cells += cell_rows(event, figures, params)

        if len(event.polls):
            elections.append(election_row(event, own, figures, params))

            if event.national:
                latest = latest_block(event, figures, params)

    if not pairs:
        raise MissingInput('pairs')

    return BanStudy(
        params=params, inputs=inputs, next=read_next(reader, params),
        summary=summarize(events, pairs, elections, cells, params),
        pairs=sorted(pairs, key=lambda pair: (pair['event_date'], -pair['d']), reverse=True),
        elections=sorted(elections, key=lambda row: row['event_date'], reverse=True),
        period=period_row(pairs, params), cells=cells, houses=house_rows(pairs, params), big=big_rows(pairs, params),
        latest=latest,
    )
