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
            default_color = get_color(random.choice(palette))
            p['color'] = parties.get(p['parties'][0], default_color) if isinstance(parties, dict) else default_color

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
    cols = list(dict.fromkeys(n for p in blocks.parties for n in p if n in df.columns))
    results = df[cols].rename(columns=block_map).T.groupby(level=0).sum(min_count=1).T
    for col in blocks.index:
        if col not in results.columns:
            results[col] = np.nan

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


def normal_update(
    mean: float | np.ndarray,
    err: float | np.ndarray,
    prior_mean: float | np.ndarray,
    prior_err: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normal-normal update of a prior `N(prior_mean, prior_err²)` with an estimate `mean ± err`: the posterior
    mean weighs both by their precisions and the posterior error combines them.

    Degenerate cases are handled so that the result is always finite: an estimate without a usable error
    (`err` NaN or infinite, or `mean` NaN) contributes nothing and the prior is returned; a prior without
    uncertainty on the data side (`prior_err` infinite) returns the estimate; `err = 0` is an exact estimate.

    Parameters
    ----------
    mean, err : float or array-like
        Estimate and its standard error.
    prior_mean, prior_err : float or array-like
        Prior mean and standard deviation.

    Returns
    -------
    tuple of np.ndarray
        Posterior mean and posterior standard error (arrays, broadcast to the common shape).
    """
    mean, err, prior_mean, prior_err = np.broadcast_arrays(
        *[np.asarray(x, dtype=float) for x in (mean, err, prior_mean, prior_err)]
    )
    mean, err, prior_mean, prior_err = (np.array(x, dtype=float) for x in (mean, err, prior_mean, prior_err))

    # Estimates without a usable error or value carry no information (infinite error)
    no_data = ~np.isfinite(mean) | ~np.isfinite(err) | (err < 0)
    err = np.where(no_data, np.inf, err)
    mean = np.where(no_data, 0., mean)
    prior_err = np.where(np.isfinite(prior_err) & (prior_err >= 0), prior_err, np.inf)

    with np.errstate(divide='ignore', invalid='ignore'):
        data_lambda = np.where(err > 0, 1. / np.square(err), np.inf)
        prior_lambda = np.where(prior_err > 0, 1. / np.square(prior_err), np.inf)
        post_lambda = data_lambda + prior_lambda

        post_mean = np.where(
            np.isinf(data_lambda), mean,
            np.where(np.isinf(prior_lambda), prior_mean, (data_lambda * mean + prior_lambda * prior_mean) / post_lambda)
        )
        post_err = np.where(np.isinf(post_lambda), 0., np.sqrt(1. / post_lambda))

    # Neither side informative: keep the prior mean with an infinite error
    none = (post_lambda == 0)
    post_mean = np.where(none, prior_mean, post_mean)
    post_err = np.where(none, np.inf, post_err)

    return post_mean, post_err
