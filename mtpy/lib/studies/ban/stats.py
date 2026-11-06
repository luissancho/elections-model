"""
Measures of the ban study: the pairs of a clandestine poll with the last legal one of its house, the sign test of
the pairs, the clandestine consensus of an election, the cells of the main parties against the frozen average, and
the rows by election, by house and of the big samples, with the summary over all of them.

Every figure is unrounded (the exporters round). Figures that are equal but for floating noise (the mean of 0,7,
0,7 and 0,9 against that of 0,6, 0,5 and 1,2) compare as equal: ``TOLERANCE`` absorbs the noise in every
comparison of two errors or distances, so an exact tie never counts as closer, beaten or crossed.
"""
from typing import Optional

import numpy as np
import pandas as pd

from .data import Event

# Absolute tolerance of the comparisons of errors and distances (points of vote share)
TOLERANCE = 1e-9
# Pairs a scope needs to be the exception of the summary (the scope with the lowest share of closer pairs)
EXCEPTION_MIN_PAIRS = 10


def as_float(value) -> Optional[float]:
    """``value`` as a float, ``None`` for null or NaN."""
    return None if value is None or pd.isna(value) else float(value)


def as_int(value) -> Optional[int]:
    """``value`` as an int, ``None`` for null or NaN."""
    return None if value is None or pd.isna(value) else int(value)


def as_str(value) -> Optional[str]:
    """``value`` as a string, ``None`` for null or NaN."""
    return None if value is None or pd.isna(value) else str(value)


def _mean(values) -> Optional[float]:
    """Plain mean of the values that are not ``None``, unrounded; ``None`` without any."""
    values = [value for value in values if value is not None]

    return float(np.mean(values)) if values else None


def _sign(value: float) -> int:
    """Sign of ``value``, zero within ``TOLERANCE``."""
    return 0 if abs(value) <= TOLERANCE else int(np.sign(value))


def _less(a: float, b: float) -> bool:
    """Whether ``a < b`` beyond ``TOLERANCE``."""
    return a < b - TOLERANCE


def is_big(sample_size: Optional[int], params: dict) -> bool:
    """Whether a poll has a big sample: ``sample_size >= params['big_sample']`` (an unknown sample is not big)."""
    return sample_size is not None and sample_size >= params['big_sample']


def last_clandestine(polls: pd.DataFrame) -> pd.DataFrame:
    """
    The last clandestine poll of each house: its row with the latest ``date``, the last row on a tie.

    Parameters
    ----------
    polls : pandas.DataFrame
        The ``polls`` of an ``Event``.

    Returns
    -------
    pandas.DataFrame
        One row per ``pollster_id``, ascending by ``date`` (rows of the same date in the order of ``polls``).
    """
    ordered = polls.sort_values('date', kind='stable')

    return ordered[~ordered['pollster_id'].duplicated(keep='last')]


def pair_rows(event: Event, params: dict) -> list:
    """
    The pairs of an election: the last clandestine poll of each house with the last legal one of the same house.

    A pair needs the legal poll at ``max_days`` days or less from the election and both errors; the houses whose
    clandestine row has a ``mtype`` of ``exclude_mtypes_pairs`` (the aggregators) stay out, the online panels in.

    Parameters
    ----------
    event : Event
        The election.
    params : dict
        Parameters of the study; uses ``max_days``, ``big_sample`` and ``exclude_mtypes_pairs``.

    Returns
    -------
    list of dict
        One row per pair in the order of ``last_clandestine``: ``scope``, ``event_date``, ``label``, ``national``,
        ``pollster_id``, ``pollster``, ``sponsor``, ``sample_size``, ``mtype`` (of the clandestine poll),
        ``date_legal``, ``days_legal``, ``error_legal``, ``date_ban``, ``days_ban``, ``error_ban``, ``d``
        (``error_ban − error_legal``), ``closer`` (``d < 0``) and ``big`` (``sample_size >= big_sample``).
    """
    legal = event.legal.drop_duplicates('pollster_id', keep='last').set_index('pollster_id')
    rows = []

    for _, poll in last_clandestine(event.polls).iterrows():
        if poll['mtype'] in params['exclude_mtypes_pairs'] or poll['pollster_id'] not in legal.index:
            continue

        before = legal.loc[poll['pollster_id']]

        if pd.isna(before['days']) or before['days'] > params['max_days']:
            continue

        if pd.isna(poll['error_avg']) or pd.isna(before['error_avg']):
            continue

        d = float(poll['error_avg'] - before['error_avg'])
        sample_size = as_int(poll['sample_size'])
        rows.append({
            'scope': event.scope, 'event_date': event.event_date, 'label': event.label, 'national': event.national,
            'pollster_id': int(poll['pollster_id']), 'pollster': as_str(poll['pollster']),
            'sponsor': as_str(poll['sponsor']), 'sample_size': sample_size, 'mtype': as_str(poll['mtype']),
            'date_legal': as_str(before['date']), 'days_legal': as_int(before['days']),
            'error_legal': float(before['error_avg']), 'date_ban': as_str(poll['date']),
            'days_ban': as_int(poll['days']), 'error_ban': float(poll['error_avg']), 'd': d, 'closer': d < 0,
            'big': is_big(sample_size, params),
        })

    return rows


def sign_test(k: int, n: int) -> Optional[float]:
    """
    How likely ``k`` or more closer pairs of ``n`` (the ties left out) are by chance: the upper tail of a binomial of
    a fair coin.

    Parameters
    ----------
    k : int
        Pairs with ``d < 0``.
    n : int
        Pairs without the ties (``d == 0``).

    Returns
    -------
    float or None
        ``binomtest(k, n, 0.5, alternative='greater').pvalue``, unrounded; ``None`` without pairs.
    """
    from scipy.stats import binomtest

    if n <= 0:
        return None

    return float(binomtest(k, n, 0.5, alternative='greater').pvalue)


def consensus(event: Event, params: dict) -> Optional[dict]:
    """
    The clandestine consensus of an election: the plain mean, by main party, of the last clandestine poll of each
    house outside ``exclude_mtypes_consensus`` (aggregators and online panels).

    Parameters
    ----------
    event : Event
        The election.
    params : dict
        Parameters of the study; uses ``exclude_mtypes_consensus``.

    Returns
    -------
    dict or None
        ``{party: share}`` in the order of ``main``, each over the polls that carry the party (a party no poll
        carries is left out); ``None`` without any poll left.
    """
    polls = last_clandestine(event.polls)
    polls = polls[~polls['mtype'].isin(params['exclude_mtypes_consensus'])]

    if polls.empty:
        return None

    means = polls[event.main].mean()

    return {party: float(value) for party, value in means.items() if pd.notna(value)}


def official_figures(event: Event) -> Optional[pd.Series]:
    """The official figures the errors and the cells compare with: those of the frozen rows (the model's own
    aggregation of the parties) when there are, else the ``results`` of the election."""
    return event.frozen['official'] if event.frozen is not None else event.official


def mean_error(figures: Optional[dict], official: Optional[pd.Series]) -> Optional[float]:
    """
    Mean absolute error of party figures against the official result.

    Parameters
    ----------
    figures : dict or None
        ``{party: share}``.
    official : pandas.Series or None
        Official share by party.

    Returns
    -------
    float or None
        The mean of ``|figure − official|`` over the parties of both; ``None`` without any.
    """
    if not figures or official is None:
        return None

    return _mean([abs(value - official[party]) for party, value in figures.items()
                  if party in official.index and pd.notna(official[party])])


def error_legal(event: Event, params: dict) -> Optional[float]:
    """
    The error of the legal polls of an election: the mean ``last_error_avg`` of the houses of the ``pollsters``
    part (those of the rating) whose last poll ended ``max_days`` days or less from the election.

    Parameters
    ----------
    event : Event
        The election.
    params : dict
        Parameters of the study; uses ``max_days``.

    Returns
    -------
    float or None
        The mean, unrounded; ``None`` without the ``pollsters`` part or without a house in the window.
    """
    if event.pollsters is None or not {'last_days', 'last_error_avg'} <= set(event.pollsters.columns):
        return None

    rows = event.pollsters[pd.to_numeric(event.pollsters['last_days'], errors='coerce') <= params['max_days']]

    return _mean([as_float(value) for value in rows['last_error_avg']])


def model_errors(event: Event, figures: Optional[dict], params: dict) -> dict:
    """
    The four errors of an election: ``legal`` (``error_legal``), ``frozen`` (the frozen average), ``consensus``
    (the clandestine consensus ``figures``) and ``with`` (the backtest with the clandestine polls in), each
    ``None`` when its input is missing.
    """
    official = official_figures(event)
    frozen = event.frozen['model'].to_dict() if event.frozen is not None else None
    with_ban = event.with_ban

    return {'legal': error_legal(event, params), 'frozen': mean_error(frozen, official),
            'consensus': mean_error(figures, official),
            'with': mean_error(with_ban['model'].to_dict(), with_ban['official']) if with_ban is not None else None}


def has_cells(event: Event, params: dict) -> bool:
    """Whether an election enters the cells and the comparison with the frozen average: national, since
    ``chart_since`` and with frozen rows."""
    return event.national and event.event_date >= params['chart_since'] and event.frozen is not None


def cell_rows(event: Event, consensus: Optional[dict], params: dict) -> list:
    """
    The cells of an election: each main party, frozen average against clandestine consensus and result.

    Parameters
    ----------
    event : Event
        The election; only one of ``has_cells`` has cells.
    consensus : dict or None
        Its clandestine consensus (``consensus``).
    params : dict
        Parameters of the study; uses ``chart_since`` and ``half``.

    Returns
    -------
    list of dict
        In the order of ``main``, for the parties with a frozen and a consensus figure: ``event_date``, ``label``,
        ``party``, ``frozen``, ``consensus``, ``official``, ``move`` (``consensus − frozen``), ``gap`` (``official −
        frozen``), ``closer`` (``|consensus − official| < |frozen − official|``), ``direction`` (``move`` with the
        sign of ``gap`` and ``|move| > 0.05``), ``crossed`` (the consensus ended on the other side of the result:
        an exact hit is not crossed) and ``half`` (closer by ``half`` of the distance or more). Empty for an
        election without cells or without consensus.
    """
    if not consensus or not has_cells(event, params):
        return []

    rows = []

    for party, frozen in event.frozen.iterrows():
        if party not in consensus:
            continue

        model, official, figure = float(frozen['model']), float(frozen['official']), consensus[party]
        move, gap = figure - model, official - model
        before, after = abs(model - official), abs(figure - official)
        closer = _less(after, before)
        rows.append({
            'event_date': event.event_date, 'label': event.label, 'party': party, 'frozen': model,
            'consensus': figure, 'official': official, 'move': move, 'gap': gap, 'closer': closer,
            'direction': abs(move) > 0.05 + TOLERANCE and _sign(move) == _sign(gap),
            'crossed': _sign(figure - official) * _sign(model - official) < 0,
            'half': closer and (before - after) / before >= params['half'] - TOLERANCE,
        })

    return rows


def election_row(event: Event, pairs: list, consensus: Optional[dict], params: dict) -> dict:
    """
    The row of an election.

    Parameters
    ----------
    event : Event
        The election.
    pairs : list of dict
        Its pairs (``pair_rows``).
    consensus : dict or None
        Its clandestine consensus (``consensus``).
    params : dict
        Parameters of the study.

    Returns
    -------
    dict
        ``scope``, ``event_date``, ``label``, ``national``, ``n_ban`` (clandestine polls of the part), ``n_pairs``,
        ``k_closer``, ``mean_d`` (``None`` without pairs) and ``model``: for an election of ``has_cells`` with a
        consensus, ``{error_legal, error_frozen, error_consensus, error_with, beaten, beaten_with}`` (``beaten``:
        the consensus erred less than the frozen average; ``error_with`` and ``beaten_with`` ``None`` without the
        ``ban`` rows), else ``None``.
    """
    model = None

    if consensus and has_cells(event, params):
        errors = model_errors(event, consensus, params)
        beaten_with = None if errors['with'] is None else _less(errors['with'], errors['frozen'])
        model = {'error_legal': errors['legal'], 'error_frozen': errors['frozen'],
                 'error_consensus': errors['consensus'], 'error_with': errors['with'],
                 'beaten': _less(errors['consensus'], errors['frozen']), 'beaten_with': beaten_with}

    return {'scope': event.scope, 'event_date': event.event_date, 'label': event.label, 'national': event.national,
            'n_ban': int(len(event.polls)), 'n_pairs': len(pairs), 'k_closer': sum(pair['closer'] for pair in pairs),
            'mean_d': _mean([pair['d'] for pair in pairs]), 'model': model}


def _pair_group(pairs: list) -> dict:
    """``n``, ``k`` (closer) and ``mean_d`` of a group of pairs."""
    return {'n': len(pairs), 'k': sum(pair['closer'] for pair in pairs), 'mean_d': _mean([p['d'] for p in pairs])}


def house_rows(pairs: list, params: dict) -> list:
    """
    The houses with ``min_pairs`` pairs or more.

    Parameters
    ----------
    pairs : list of dict
        Every pair (``pair_rows``).
    params : dict
        Parameters of the study; uses ``min_pairs``.

    Returns
    -------
    list of dict
        ``pollster_id``, ``pollster`` (the name of its latest pair), ``n``, ``k``, ``share`` (``k / n``) and
        ``mean_d``, by ``share`` descending, ``n`` descending and ``mean_d`` ascending.
    """
    houses = {}

    for pair in sorted(pairs, key=lambda pair: pair['event_date']):
        houses.setdefault(pair['pollster_id'], []).append(pair)

    rows = []
    for pollster_id, own in houses.items():
        if len(own) >= params['min_pairs']:
            group = _pair_group(own)
            rows.append({'pollster_id': pollster_id, 'pollster': own[-1]['pollster'], 'n': group['n'],
                         'k': group['k'], 'share': group['k'] / group['n'], 'mean_d': group['mean_d']})

    return sorted(rows, key=lambda row: (-row['share'], -row['n'], row['mean_d'], row['pollster']))


def big_rows(pairs: list, params: dict) -> list:
    """
    The pairs of the big samples (``big``).

    Parameters
    ----------
    pairs : list of dict
        Every pair (``pair_rows``).
    params : dict
        Parameters of the study; uses ``big_sample`` (``is_big``).

    Returns
    -------
    list of dict
        ``scope``, ``event_date``, ``label``, ``pollster_id``, ``pollster``, ``sponsor``, ``sample_size``,
        ``error_legal``, ``error_ban`` and ``d``, by ``sample_size`` descending.
    """
    keys = ('scope', 'event_date', 'label', 'pollster_id', 'pollster', 'sponsor', 'sample_size', 'error_legal',
            'error_ban', 'd')
    rows = [{key: pair[key] for key in keys} for pair in pairs if is_big(pair['sample_size'], params)]

    return sorted(rows, key=lambda row: (-row['sample_size'], row['event_date']))


def period_row(pairs: list, params: dict) -> dict:
    """
    The pairs of the general elections before ``chart_since`` (the row «1996 – 2011» of the table).

    Returns
    -------
    dict
        ``{since, k, n, mean_d}``, ``mean_d`` ``None`` without pairs.
    """
    early = [pair for pair in pairs if pair['national'] and pair['event_date'] < params['chart_since']]
    group = _pair_group(early)

    return {'since': params['since'], 'k': group['k'], 'n': group['n'], 'mean_d': group['mean_d']}


def _summary_pairs(pairs: list, scopes: list) -> dict:
    """The ``pairs`` block of the summary."""
    ties = sum(pair['d'] == 0 for pair in pairs)
    whole = _pair_group(pairs)
    blocks = {}

    for name, national in (('generals', True), ('regional', False)):
        own = [pair for pair in pairs if pair['national'] == national]
        blocks[name] = _pair_group(own) | {'n_events': len({(p['scope'], p['event_date']) for p in own})}

    by_scope = [{'scope': scope} | _pair_group([pair for pair in pairs if pair['scope'] == scope])
                for scope in scopes if any(pair['scope'] == scope for pair in pairs)]
    by_scope.sort(key=lambda row: -row['n'])

    return {'n': whole['n'], 'k': whole['k'], 'ties': ties, 'share': whole['k'] / whole['n'] if pairs else None,
            'mean_d': whole['mean_d'], 'p': sign_test(whole['k'], whole['n'] - ties), **blocks, 'by_scope': by_scope}


def _summary_elections(elections: list) -> Optional[dict]:
    """The ``elections`` block of the summary, from the rows with a ``model`` block; ``None`` without any."""
    models = [row['model'] for row in elections if row['model'] is not None]

    if not models:
        return None

    with_ban = [model for model in models if model['error_with'] is not None]

    return {'n': len(models), 'k_beaten': sum(model['beaten'] for model in models),
            'mean_legal': _mean([model['error_legal'] for model in models]),
            'mean_frozen': _mean([model['error_frozen'] for model in models]),
            'mean_consensus': _mean([model['error_consensus'] for model in models]),
            'mean_with': _mean([model['error_with'] for model in with_ban]),
            'k_with': sum(model['beaten_with'] for model in with_ban) if with_ban else None}


def _exception(by_scope: list) -> Optional[dict]:
    """The scope with the lowest share of closer pairs among those with ``EXCEPTION_MIN_PAIRS`` pairs or more."""
    candidates = [row for row in by_scope if row['n'] >= EXCEPTION_MIN_PAIRS]

    if not candidates:
        return None

    worst = min(candidates, key=lambda row: row['k'] / row['n'])

    return {'scope': worst['scope'], 'n': worst['n'], 'k': worst['k']}


def summarize(events: list, pairs: list, elections: list, cells: list, params: dict) -> dict:
    """
    The summary of the study.

    Parameters
    ----------
    events : list of Event
        The elections read (their scopes give the order of ``by_scope`` on a tie).
    pairs : list of dict
        Every pair (``pair_rows``).
    elections : list of dict
        Every election row (``election_row``).
    cells : list of dict
        Every cell (``cell_rows``).
    params : dict
        Parameters of the study; uses ``big_sample``.

    Returns
    -------
    dict
        ``pairs`` (``{n, k, ties, share, mean_d, p, generals, regional, by_scope}``, ``generals`` and ``regional``
        with ``{n, k, mean_d, n_events}``, ``by_scope`` ``[{scope, n, k, mean_d}]`` by ``n`` descending),
        ``elections`` (``{n, k_beaten, mean_legal, mean_frozen, mean_consensus, mean_with, k_with}`` over the rows
        with a model, the means of the unrounded errors, ``mean_with`` and ``k_with`` ``None`` without ``ban``
        rows; ``None`` without any model), ``cells`` (``{n, k_closer, k_direction, k_crossed, k_half, missed}``,
        ``missed`` the ``{label, party}`` of the cells not closer), ``big`` (``{threshold, n, k, mean_d}``) and
        ``exception`` (``{scope, n, k}`` or ``None``). Every figure unrounded.
    """
    scopes = list(dict.fromkeys([event.scope for event in events if event.national]
                                + [event.scope for event in events]))
    summary_pairs = _summary_pairs(pairs, scopes)
    big = _pair_group([pair for pair in pairs if is_big(pair['sample_size'], params)])

    return {
        'pairs': summary_pairs,
        'elections': _summary_elections(elections),
        'cells': {'n': len(cells), 'k_closer': sum(cell['closer'] for cell in cells),
                  'k_direction': sum(cell['direction'] for cell in cells),
                  'k_crossed': sum(cell['crossed'] for cell in cells), 'k_half': sum(cell['half'] for cell in cells),
                  'missed': [{'label': cell['label'], 'party': cell['party']} for cell in cells if not cell['closer']]},
        'big': {'threshold': params['big_sample']} | big,
        'exception': _exception(summary_pairs['by_scope']),
    }
