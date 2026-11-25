"""
The votes of each scenario of the CERA study and the snapshot that measures them on the simulations of the model.

Every scenario is a frame of votes by `region_id` and candidature of 2023 added to the votes of the model (`delta`,
once `majority.to_model` has put it in the order of the simulator): the votes the study expects from the new
electors in each census scenario, the new electors at a participation and a left share, the whole exterior vote
and the optimal placement of the pool. `build_snapshot` measures them all with `majority.measure`.
"""
import pandas as pd

from .majority import LINEAGE, Draws, applied_lineage, base_counts, measure, province_counts, to_model
from .seats import increment_votes
from .stress import _biased_split, exterior_votes, placement_plan
from .study import (
    GRID_LEFT_FRACS,
    GRID_PARTICIPATIONS,
    PLACEMENT_CANDIDATURE,
    PLACEMENT_FRACS,
    PLACEMENT_PARTICIPATIONS,
    CeraStudy,
)


def _baseline(study: CeraStudy):
    """The results of the baseline election of the study."""
    return study.results[study.params['election']['baseline']]


def expected_votes(study: CeraStudy, name: str) -> pd.DataFrame:
    """
    The votes of the new electors the study expects in a census scenario.

    Parameters
    ----------
    study : CeraStudy
        The study.
    name : str
        The scenario, `convocatoria` or `sin_litigio`.

    Returns
    -------
    pd.DataFrame
        Votes by `region_id` and candidature of 2023.
    """
    return study.scenarios[name]['extra']


def expected_table(study: CeraStudy, name: str) -> dict:
    """
    The census, voters and votes of a scenario in 2023 and in the election of 29-N.

    The same figures as the published scenario of the study, with 2023's census the scenario's minus its new
    electors.

    Parameters
    ----------
    study : CeraStudy
        The study.
    name : str
        The scenario, `convocatoria` or `sin_litigio`.

    Returns
    -------
    dict
        `census`, `voters` and `votes` as `[2023, 29-N]`, and `new_electors`; integers.
    """
    scenario, baseline = study.scenarios[name], study.params['election']['baseline']
    new = scenario['new']
    rates = new['origin'].map(study.params['participation']).astype(float)
    census = int(scenario['census']['electors'].sum())
    new_electors = int(new['electors'].sum())
    voters = int(study.voters[baseline].sum())
    votes = int(round(float(study.cera[baseline].values.sum())))

    return {
        'census': [census - new_electors, census],
        'new_electors': new_electors,
        'voters': [voters, int(round(voters + float((new['electors'] * rates).sum())))],
        'votes': [votes, int(round(votes + scenario['model']['votes_extra']))]
    }


def new_elector_votes(study: CeraStudy, participation: float, left_frac: float) -> pd.DataFrame:
    """
    The votes of the new electors of the convocation at a participation and a left share.

    Parameters
    ----------
    study : CeraStudy
        The study.
    participation : float
        Participation rate (0-1), the same for both origins.
    left_frac : float
        Fraction (0-1) of the left + right vote that goes to the left.

    Returns
    -------
    pd.DataFrame
        Votes by `region_id` and candidature of 2023.
    """
    split = _biased_split(study.split, left_frac, _baseline(study).siglas, study.blocks)

    return increment_votes(study.scenarios['convocatoria']['new'], {'latam': participation, 'rest': participation},
                           split, valid=study.params['participation']['valid'])


def exterior_change(study: CeraStudy, participation: float, left_frac: float) -> pd.DataFrame:
    """
    The change of the exterior vote when the whole exterior census votes at a participation and a left share.

    Parameters
    ----------
    study : CeraStudy
        The study.
    participation : float
        Participation rate (0-1).
    left_frac : float
        Fraction (0-1) of the left + right vote that goes to the left.

    Returns
    -------
    pd.DataFrame
        The whole exterior vote minus the CERA vote of 2023, by `region_id` and candidature of 2023.
    """
    results = _baseline(study)
    cera = study.cera[study.params['baseline_scrutiny']]
    whole = exterior_votes(cera, results, study.census_by_province, participation, left_frac, results.siglas,
                           study.blocks)

    return whole - cera.reindex(index=whole.index, columns=whole.columns, fill_value=0)


def placement_change(study: CeraStudy, participation: float, frac: float) -> pd.DataFrame:
    """
    The votes of the optimal placement of the pool of new electors.

    Parameters
    ----------
    study : CeraStudy
        The study.
    participation : float
        Participation rate (0-1).
    frac : float
        Fraction (0-1) of the votes that go to the candidature.

    Returns
    -------
    pd.DataFrame
        Votes by `region_id`, with the candidature `PLACEMENT_CANDIDATURE` as the only column.
    """
    plan = placement_plan(_baseline(study), study.placement['electors'], participation, frac, PLACEMENT_CANDIDATURE)

    return plan.to_frame(PLACEMENT_CANDIDATURE)


def build_snapshot(draws: Draws, study: CeraStudy, meta: dict) -> dict:
    """
    Measure every scenario on the simulations of a run and return the snapshot.

    Parameters
    ----------
    draws : Draws
        The simulations of the run.
    study : CeraStudy
        The study.
    meta : dict
        The published `meta` of the run: `scope`, `event_date`, `run_id`, `as_of`, `n_sim`, `seed`, `majority` and
        `bmaps` (with its `blocks`).

    Returns
    -------
    dict
        `run`, `majority`, `blocks`, `lineage`, `base`, `expected`, `new_electors`, `grid` and `placement`, plain
        Python types; the run's `commit` and `taken_on` are for the caller to add.
    """
    siglas = _baseline(study).siglas
    blocks, majority = meta['bmaps']['blocks'], meta['majority']

    def delta_of(frame: pd.DataFrame):
        return to_model(frame, siglas, draws.names, draws.regions, LINEAGE)

    def table(frames: list) -> list:
        return [[measure(draws, delta_of(frame), blocks, majority) for frame in row] for row in frames]

    expected = {}
    for name in study.scenarios:
        delta = delta_of(expected_votes(study, name))
        provinces = [{**row, 'name': study.province_names.get(row['region_id'])}
                     for row in province_counts(draws, delta, blocks)]
        expected[name] = {'table': expected_table(study, name), 'counts': measure(draws, delta, blocks, majority),
                          'provinces': provinces}

    return {
        'run': {key: meta[key] for key in ('scope', 'event_date', 'run_id', 'as_of', 'n_sim', 'seed')},
        'majority': majority,
        'blocks': blocks,
        'lineage': applied_lineage(siglas, draws.names, LINEAGE),
        'base': base_counts(draws, blocks, majority),
        'expected': expected,
        'new_electors': {
            'participations': GRID_PARTICIPATIONS, 'left_fracs': GRID_LEFT_FRACS,
            'electors': int(study.scenarios['convocatoria']['new']['electors'].sum()),
            'cells': table([[new_elector_votes(study, p, f) for f in GRID_LEFT_FRACS] for p in GRID_PARTICIPATIONS])
        },
        'grid': {
            'participations': GRID_PARTICIPATIONS, 'left_fracs': GRID_LEFT_FRACS,
            'electors': int(study.census_by_province.sum()),
            'cells': table([[exterior_change(study, p, f) for f in GRID_LEFT_FRACS] for p in GRID_PARTICIPATIONS])
        },
        'placement': {
            'participations': PLACEMENT_PARTICIPATIONS, 'fracs': PLACEMENT_FRACS,
            'pool': int(study.placement['electors'].sum()),
            'cells': table([[placement_change(study, p, f) for f in PLACEMENT_FRACS]
                            for p in PLACEMENT_PARTICIPATIONS])
        }
    }
