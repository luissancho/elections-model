import numpy as np
import pandas as pd
import random
from typing import Any, Optional

from ..core.utils.dataviz import get_color, palette


def build_blocks(
    bmap: Optional[dict[str, Any]] = None,
    parties: Optional[tuple | list | dict] = None
) -> pd.DataFrame:
    """
    Instead of analizing the results of each party individually, we might want to group them into blocks.
    If this is the case, we use `bmap` to aggregate the results of each group of parties.

    Parameters
    ----------
    bmap : dict, optional
        A dictionary mapping the parties and blocks to be included in the analysis.
        If not provided, all the parties will be used as blocks.
    parties : tuple | list | dict, optional
        A tuple, list or dictionary of parties to be included in the analysis.
        If a dictionary is provided, a color should be specified for each party.
        If not provided, random colors will be assigned to each block.

    Returns
    -------
    pd.DataFrame
        Map of blocks and their corresponding party members
    """
    if bmap is None:
        if isinstance(parties, dict):
            bmap = parties
        elif isinstance(parties, (tuple, list)):
            bmap = dict(zip(parties, parties))
        else:
            raise ValueError('At least one of bmap or parties should be provided')
    elif isinstance(bmap, (list, tuple)):
        bmap = dict(zip(bmap, bmap))

    names = list(parties) if parties is not None else list(bmap)
    blocks = {}
    pinc = [n for p in bmap.values() for n in ([p] if isinstance(p, str) else [] if p is None else p)]

    for i, (b, p) in enumerate(bmap.items()):
        if isinstance(p, str):
            p = {'parties': [p]}
        elif isinstance(p, (list, tuple)):
            p = {'parties': p}
        elif p is None:
            p = {'parties': [parties[-1]] + [i for i in names[:-1] if i not in pinc]}

        if 'color' not in p:
            p['color'] = parties[p['parties'][0]] if isinstance(parties, dict) else get_color(random.choice(palette))

        blocks[b] = p

    return pd.DataFrame.from_dict(blocks, orient='index')


def group_results(
    df: pd.DataFrame,
    blocks: Optional[pd.DataFrame] = None,
    bmap: Optional[dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Group the results of the parties into blocks.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the results of the parties.
    blocks : pd.DataFrame, optional
        The dataframe containing the blocks of parties.
    bmap : dict, optional
        The dictionary containing the mapping of the parties to the blocks.

    Returns
    -------
    pd.DataFrame
        The dataframe containing the results of the blocks.
    """
    if blocks is None:
        if bmap is not None:
            blocks = build_blocks(bmap)
        else:
            return df

    block_map = {n: b for b, p in blocks.parties.items() for n in p if n in df.columns and n != b}
    results = df.rename(columns=block_map).groupby(level=0, axis=1).sum(min_count=1)
    for col in blocks.index:
        if col not in results.columns:
            results[col] = np.NaN

    return results[blocks.index.tolist()]


def norm_range(
    drange: Optional[tuple | int] = None,
    dmax: Optional[int] = None
) -> list[int]:
    """
    Normalize drange parameter to a list of two integers clipped to the range [0, `dmax`].

    Parameters
    ----------
    drange : tuple or int, optional
        Range of days to consider.
    dmax : int, optional
        Maximum number of days to consider.

    Returns
    -------
    list[int]
        Normalized range of days to consider.
    """
    if drange is None:
        return 0, dmax
    elif not isinstance(drange, (tuple, list)):
        return int(drange), dmax
    elif len(drange) < 1:
        return 0, dmax
    elif len(drange) < 2:
        return tuple(list(drange) + [dmax])
    elif drange[0] is None:
        return 0, drange[1]
    elif drange[1] is None:
        return drange[0], dmax
    else:
        return tuple(list(drange)[:2])
