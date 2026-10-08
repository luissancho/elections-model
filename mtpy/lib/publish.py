"""Pure exporters of a finished simulation into JSON-ready dicts and CSV twin frames.

Every ``export_*`` function takes an already executed ``Simulator`` and returns ``(data, frame)``:
``data`` follows the bundle schema of the same name and ``frame`` is its tabular twin for the CSV.
Nothing here keeps state or touches storage.
"""
import pandas as pd

PCT_DECIMALS = 2
PROB_DECIMALS = 3
SEATS_DECIMALS = 1

PCT_COLS = ('pct', 'pct_mean', 'pct_lo', 'pct_hi', 'mean', 'lo', 'hi', 'sd')
PROB_COLS = ('p_seats', 'p_majority', 'p_first')
SEATS_COLS = ('seats_mean', 'seats_median', 'seats_lo', 'seats_hi', 'seats_min', 'seats_max')
SUMMARY_COLS = ('pct', 'pct_lo', 'pct_hi', 'seats', 'seats_mean', 'seats_lo', 'seats_hi', 'p_seats')


def records(frame, index_name='name'):
    """Convert a frame into a list of row dicts, with its index as a leading column.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to convert.
    index_name : str
        Name given to the index column.

    Returns
    -------
    list of dict
        One dict per row; NaN values are kept (the bundle writer turns them into null).
    """
    return frame.rename_axis(index_name).reset_index().to_dict(orient='records')


def round_cols(frame, pct=(), prob=(), seats=()):
    """Round the present columns of each group to its number of decimals.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to round (not modified).
    pct, prob, seats : iterable of str
        Column names rounded to ``PCT_DECIMALS``, ``PROB_DECIMALS`` and ``SEATS_DECIMALS``.

    Returns
    -------
    pandas.DataFrame
        A rounded copy.
    """
    out = frame.copy()
    for cols, decimals in ((pct, PCT_DECIMALS), (prob, PROB_DECIMALS), (seats, SEATS_DECIMALS)):
        present = [c for c in cols if c in out.columns]
        if present:
            out[present] = out[present].round(decimals)
    return out


def _summary_table(sim, names=None):
    """Return ``sim.summary(names)`` with every statistic rounded for publication."""
    return round_cols(sim.summary(names), pct=PCT_COLS, prob=PROB_COLS, seats=SEATS_COLS)


def _p_majority(sim):
    """Return the probability of an absolute majority per block, rounded."""
    return sim.probabilities('vs').round(PROB_DECIMALS)


def export_vote(sim):
    """Export the vote share forecast.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``vote`` schema and its CSV twin.
    """
    table = round_cols(sim.vote_forecast(), pct=PCT_COLS)[['pct', 'sd', 'lo', 'hi']]
    rows = records(table, 'name')
    data = {'horizon': int(sim.horizon), 'when': sim.when.strftime('%Y-%m-%d'), 'rows': rows}
    return data, pd.DataFrame(rows, columns=['name', 'pct', 'sd', 'lo', 'hi'])


def export_summary(sim):
    """Export the seat summary for parties, vote-share blocks and seat blocks.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``summary`` schema and its CSV twin (tables stacked with a ``group`` column).
    """
    tables = {'parties': _summary_table(sim), 'vs': _summary_table(sim, 'vs'),
              'blocks': _summary_table(sim, 'blocks')}
    data = {
        'n_seats': int(sim.n_seats),
        'majority': int(sim.n_seats) // 2 + 1,
        **{group: records(table) for group, table in tables.items()},
        'p_majority': _p_majority(sim).to_dict(),
        'totals': {name: int(seats) for name, seats in sim.totals().items()},
    }
    frame = pd.concat([table.rename_axis('name').reset_index().assign(group=group)
                       for group, table in tables.items()], ignore_index=True)
    frame.insert(0, 'group', frame.pop('group'))
    return data, frame


def export_dist(sim):
    """Export the distribution of seats per simulation.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``dist`` schema and its CSV twin (simulations by parties).
    """
    frame = sim.dist()
    data = {'n_seats': int(sim.n_seats), 'parties': list(frame.columns), 'seats': frame.values.tolist()}
    return data, frame


def export_districts(sim):
    """Export votes and seats per district, leaving out the scope total.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``districts`` schema and its CSV twin.
    """
    region_ids = [r for r in sim.params['regions'] if r != sim.default_region]
    regions = [{'id': r, 'name': sim.region_names[r], 'seats': int(sim.reg_totals.loc[r, 'seats'])}
               for r in region_ids]
    rows = []
    for r in region_ids:
        table = round_cols(sim.unit_summary(r), pct=PCT_COLS, prob=PROB_COLS, seats=SEATS_COLS)
        for row in records(table[list(SUMMARY_COLS)]):
            rows.append({'region_id': r, 'region': sim.region_names[r], **row})
    columns = ['region_id', 'region', 'name', *SUMMARY_COLS]
    data = {'parties': list(sim.params['names']), 'regions': regions, 'rows': rows}
    return data, pd.DataFrame(rows, columns=columns)


def export_scenario(sim):
    """Export one randomly chosen simulation as a coherent seat scenario.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``scenario`` schema and its CSV twin.
    """
    i = sim.scenario()
    table = sim.result(i)
    parties = list(table.columns)
    region_ids = list(sim.params['regions'])
    rows = [{'region_id': region_id, 'region': region, 'seats': [int(s) for s in values]}
            for region_id, region, values in zip(region_ids, table.index, table.values)]
    frame = table.copy()
    frame.insert(0, 'region', list(table.index))
    frame.insert(0, 'region_id', region_ids)
    frame = frame.reset_index(drop=True)
    return {'simulation': int(i), 'parties': parties, 'rows': rows}, frame


def export_projection(sim):
    """Export the projected vote share over time for parties and both block groupings.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Data for the ``projection`` schema and its long-format CSV twin.
    """
    groups, dates, long = {}, None, []
    for group, names in (('parties', None), ('vs', 'vs'), ('blocks', 'blocks')):
        out = sim.projection(names).round(PCT_DECIMALS)
        if dates is None:
            dates = list(out.index.strftime('%Y-%m-%d'))
        columns = list(out['mean'].columns)
        groups[group] = {'names': columns,
                         **{stat: {c: out[(stat, c)].tolist() for c in columns} for stat in ('mean', 'lo', 'hi')}}
        for c in columns:
            long.append(pd.DataFrame({'date': dates, 'group': group, 'name': c, 'mean': out[('mean', c)].values,
                                      'lo': out[('lo', c)].values, 'hi': out[('hi', c)].values}))
    frame = pd.concat(long, ignore_index=True)
    return {'dates': dates, 'groups': groups}, frame


def headline_mode(sim):
    """Build the headline of one mode: vote share, seats and win probability per party.

    Parameters
    ----------
    sim : Simulator
        Simulator already run.

    Returns
    -------
    dict
        ``{'parties': [...], 'p_majority': {block: p}}`` with parties in the order of the vote forecast.
    """
    vote = round_cols(sim.vote_forecast(), pct=PCT_COLS)
    summary = round_cols(sim.summary(), prob=PROB_COLS, seats=SEATS_COLS)
    parties = [{'name': name, 'pct': vote.loc[name, 'pct'], 'lo': vote.loc[name, 'lo'], 'hi': vote.loc[name, 'hi'],
                'seats': int(summary.loc[name, 'seats']), 'seats_lo': summary.loc[name, 'seats_lo'],
                'seats_hi': summary.loc[name, 'seats_hi'], 'p_first': summary.loc[name, 'p_first']}
               for name in vote.index]
    return {'parties': parties, 'p_majority': _p_majority(sim).to_dict()}
