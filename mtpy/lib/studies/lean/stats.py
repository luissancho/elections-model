"""Pure statistics and tables of the study of the lean of a pollster, over the rows of ``data.read_error_rows``."""
import math
from typing import Optional


def floor_odds(p: float) -> int:
    """
    ``1 / p`` floored to two significant figures, so that "less than one in ``floor_odds(p)``" is true.

    Parameters
    ----------
    p : float
        A probability in (0, 1].

    Returns
    -------
    int
        The odds: ``7200`` for a p of ``1 / 7234``.
    """
    odds = 1 / p
    scale = 10 ** max(0, int(math.log10(odds)) - 1)

    return int(odds // scale * scale)


def sign_chance(n: int, k: int) -> Optional[dict]:
    """
    How likely a pollster without lean errs to the same side in ``k`` or more of ``n`` elections: the upper tail of a
    binomial of a fair coin, as each election is equally likely to be over or under the result.

    Parameters
    ----------
    n : int
        Elections with an error in the block.
    k : int
        Elections erring to the side counted.

    Returns
    -------
    dict or None
        ``n``, ``k``, ``ways`` (the outcomes with ``k`` or more on that side, the sum of the binomial coefficients),
        ``total`` (the 2ⁿ outcomes), ``p`` (``ways / total``), ``odds_exact`` (``1 / p`` rounded to the unit) and
        ``odds`` (``1 / p`` floored to two significant figures, so "less than one in ``odds``" is true). ``None``
        without elections.
    """
    if n <= 0:
        return None

    ways = sum(math.comb(n, j) for j in range(k, n + 1))
    total = 2 ** n
    p = ways / total

    return {'n': n, 'k': k, 'ways': ways, 'total': total, 'p': p, 'odds_exact': round(1 / p), 'odds': floor_odds(p)}


def t_test(values: list) -> Optional[dict]:
    """
    One-sample Student t test of the mean of ``values`` against zero, two-sided.

    Parameters
    ----------
    values : list of float
        Observations.

    Returns
    -------
    dict or None
        ``n``, ``mean``, ``sd`` (sample, ddof 1) and ``t`` rounded to 2 decimals and ``p`` with three significant
        figures (rounded to decimals, a tiny p would become zero). ``None`` with fewer than two values or without
        variance.
    """
    from scipy import stats

    n = len(values)

    if n < 2:
        return None

    mean = sum(values) / n
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))

    if sd == 0:
        return None

    t = mean / (sd / math.sqrt(n))
    p = 2 * stats.t.sf(abs(t), n - 1)

    return {'n': n, 'mean': round(mean, 2), 'sd': round(sd, 2), 't': round(t, 2), 'p': float(f'{float(p):.3g}')}


def _mean(values: list) -> Optional[float]:
    """Mean of ``values``, ``None`` when empty."""
    return sum(values) / len(values) if values else None


def _round(value: Optional[float]) -> Optional[float]:
    """Round to 2 decimals, keeping ``None``."""
    return None if value is None else round(value, 2)


def _is_number(value) -> bool:
    """Whether ``value`` is a usable number."""
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def summarize_period(rows: list, pollster: str) -> dict:
    """
    Figures of one pollster over the rows of every house inside a period.

    Parameters
    ----------
    rows : list of dict
        Rows of ``read_error_rows`` of every house in the period (the caller filters by date).
    pollster : str
        Pollster studied.

    Returns
    -------
    dict
        ``n``, ``k_left`` (errors with ``Izquierda > 0``), ``n_right``, ``k_right`` (``Derecha < 0``), ``mean_left``,
        ``mean_right``, ``n_lean``, ``k_lean`` (``lean > 0``), ``mean_lean``, ``exceptions`` (labels with
        ``Izquierda <= 0``, newest first), ``consensus`` (``n``, ``k`` elections above the others' mean, ``mean_diff``),
        ``dates`` (``n``, ``k`` dates with a positive mean), ``chance`` (``coin``, ``dates``, ``consensus`` of
        ``sign_chance``), ``t`` (``all``, ``dates``, ``consensus`` of ``t_test``), ``industry`` (``n_events``, ``left``,
        ``right``, ``mean`` of the elections with three or more other houses) and ``rows`` (one per election of the
        pollster: ``scope``, ``event_date``, ``consensus``, ``n_others``, ``others_left``).
    """
    own = [row for row in rows if row['pollster'] == pollster]
    others = {}

    for row in rows:
        if row['pollster'] != pollster:
            others.setdefault((row['scope'], row['event_date']), []).append(row['Izquierda'])

    left = [row['Izquierda'] for row in own]
    right = [row['Derecha'] for row in own if _is_number(row.get('Derecha'))]
    lean = [row['lean'] for row in own if _is_number(row.get('lean'))]

    by_date = {}
    for row in own:
        by_date.setdefault(row['event_date'], []).append(row['Izquierda'])
    date_means = [_mean(values) for values in by_date.values()]

    consensus_rows, differences = [], []
    for row in own:
        rest = others.get((row['scope'], row['event_date']), [])
        consensus = _mean(rest)
        consensus_rows.append({'scope': row['scope'], 'event_date': row['event_date'], 'consensus': consensus,
                               'n_others': len(rest), 'others_left': sum(1 for value in rest if value > 0)})
        if consensus is not None:
            differences.append(row['Izquierda'] - consensus)

    means = [_mean(rest) for rest in others.values() if len(rest) >= 3]
    k_dates = sum(1 for value in date_means if value > 0)
    k_consensus = sum(1 for value in differences if value > 0)
    k_left = sum(1 for value in left if value > 0)

    return {
        'n': len(own), 'k_left': k_left,
        'n_right': len(right), 'k_right': sum(1 for value in right if value < 0),
        'mean_left': _round(_mean(left)), 'mean_right': _round(_mean(right)),
        'n_lean': len(lean), 'k_lean': sum(1 for value in lean if value > 0), 'mean_lean': _round(_mean(lean)),
        'exceptions': [row['label'] for row in sorted(own, key=lambda row: row['event_date'], reverse=True)
                       if row['Izquierda'] <= 0],
        'consensus': {'n': len(differences), 'k': k_consensus, 'mean_diff': _round(_mean(differences))},
        'dates': {'n': len(date_means), 'k': k_dates},
        'chance': {'coin': sign_chance(len(left), k_left), 'dates': sign_chance(len(date_means), k_dates),
                   'consensus': sign_chance(len(differences), k_consensus)},
        't': {'all': t_test(left), 'dates': t_test(date_means), 'consensus': t_test(differences)},
        'industry': {'n_events': len(means), 'left': sum(1 for value in means if value > 0),
                     'right': sum(1 for value in means if value < 0), 'mean': _round(_mean(means))},
        'rows': consensus_rows,
    }


def house_table(rows: list, pollster: str, min_elections: int, active: set) -> list:
    """
    One row per house with enough elections and still active, ordered from the most to the least right-leaning error.

    Parameters
    ----------
    rows : list of dict
        Rows of ``read_error_rows`` of every house in the period.
    pollster : str
        Pollster studied, flagged with ``cis``.
    min_elections : int
        Minimum rows (elections) of a house.
    active : set of str
        Houses with polls in the current cycle.

    Returns
    -------
    list of dict
        ``pollster``, ``n``, ``k_left``, ``share_left``, ``n_lean``, ``mean_lean``, ``mean_left``, ``active_2026`` and
        ``cis``, sorted by ``mean_lean``, ``share_left`` and name.
    """
    by_house = {}
    for row in rows:
        by_house.setdefault(row['pollster'], []).append(row)

    houses = []
    for house, own in by_house.items():
        if len(own) < min_elections or house not in active:
            continue

        left = [row['Izquierda'] for row in own]
        lean = [row['lean'] for row in own if _is_number(row.get('lean'))]
        k_left = sum(1 for value in left if value > 0)
        houses.append({'pollster': house, 'n': len(own), 'k_left': k_left, 'share_left': k_left / len(own),
                       'n_lean': len(lean), 'mean_lean': _round(_mean(lean)), 'mean_left': _round(_mean(left)),
                       'active_2026': True, 'cis': house == pollster})

    houses.sort(key=lambda house: (house['mean_lean'] is None, house['mean_lean'] or 0, house['share_left'],
                                   house['pollster']))

    return houses


def matrix_table(rows: list, houses: list) -> dict:
    """
    The matrix of elections by house with the error of the left block in each cell.

    Parameters
    ----------
    rows : list of dict
        Rows of ``read_error_rows``.
    houses : list of dict
        Output of ``house_table``.

    Returns
    -------
    dict
        ``events`` (``scope``, ``event_date``, ``label``, ``short``; chronological, same date in the order of the
        rows), ``houses`` (names, the studied one first, then by ``share_left`` descending and name) and ``cells``
        (``[x, y, Izquierda]``, x the event index and y the house index).
    """
    ordered = sorted(houses, key=lambda house: (not house['cis'], -house['share_left'], house['pollster']))
    names = [house['pollster'] for house in ordered]

    events = {}
    for row in rows:
        events.setdefault((row['scope'], row['event_date']),
                          {'scope': row['scope'], 'event_date': row['event_date'], 'label': row['label'],
                           'short': row['short']})
    event_list = sorted(events.values(), key=lambda event: event['event_date'])
    x_of = {(event['scope'], event['event_date']): x for x, event in enumerate(event_list)}
    y_of = {name: y for y, name in enumerate(names)}

    cells = [[x_of[row['scope'], row['event_date']], y_of[row['pollster']], row['Izquierda']]
             for row in rows if row['pollster'] in y_of]

    return {'events': event_list, 'houses': names, 'cells': cells}


def election_table(election: dict, pollster: str, max_days: int, blocks: list) -> dict:
    """
    The last poll of each house before one election, in block shares and errors, ordered by distance to the result.

    Parameters
    ----------
    election : dict
        ``meta``, ``polls`` and ``results`` as returned by ``read_election``.
    pollster : str
        Pollster studied.
    max_days : int
        Only polls this close to the election.
    blocks : list of str
        The two blocks, left first (``['Izquierda', 'Derecha']``).

    Returns
    -------
    dict
        ``blocks`` (block to parties), ``max_days``, ``result`` (``Izquierda``, ``Derecha``, ``gap``), ``cis`` and
        ``consensus`` (``days``, ``Izquierda``, ``Derecha``, ``gap``, ``e_left``, ``e_right``, ``e_gap``, ``dist``;
        ``consensus`` without ``days`` but with ``n``; ``cis`` with ``date`` if the rows carry it) and ``houses``
        (``pollster``, ``days``, ``Izquierda``, ``Derecha``, ``gap``, ``e_left``, ``e_right``, ``dist``).
    """
    left, right = blocks
    groups = {block: election['meta']['blocks'][block] for block in blocks}
    pct = {row['party']: row['pct'] for row in election['results']['rows']}
    result = {block: sum(pct.get(party, 0) for party in parties) for block, parties in groups.items()}
    result_gap = result[left] - result[right]

    closest = {}
    for row in election['polls']['rows']:
        house = row['pollster']
        if row['days'] <= max_days and (house not in closest or row['days'] < closest[house]['days']):
            closest[house] = row

    def build(e_left, e_right):
        """Shares, gap and distance of a poll with the given block errors."""
        share_left, share_right = result[left] + e_left, result[right] + e_right

        return {left: round(share_left, 2), right: round(share_right, 2), 'gap': round(share_left - share_right, 2),
                'e_left': round(e_left, 2), 'e_right': round(e_right, 2),
                'e_gap': round(share_left - share_right - result_gap, 2),
                'dist': round(math.hypot(e_left, e_right), 2)}

    houses = []
    for house, row in closest.items():
        built = build(row[left], row[right])
        houses.append({'pollster': house, 'days': row['days']} | {
            key: built[key] for key in (left, right, 'gap', 'e_left', 'e_right', 'dist')})
    houses.sort(key=lambda house: (house['dist'], house['pollster']))

    own = None
    if pollster in closest:
        own = {'days': closest[pollster]['days']} | build(closest[pollster][left], closest[pollster][right])
        if 'date' in closest[pollster]:
            own['date'] = closest[pollster]['date']

    rest = [row for house, row in closest.items() if house != pollster]
    consensus = None
    if rest:
        consensus = build(_mean([row[left] for row in rest]), _mean([row[right] for row in rest])) | {'n': len(rest)}

    return {'blocks': groups, 'max_days': max_days,
            'result': {left: round(result[left], 2), right: round(result[right], 2), 'gap': round(result_gap, 2)},
            'cis': own, 'consensus': consensus, 'houses': houses}
