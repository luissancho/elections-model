"""
Tuning of the polling average on future polls.

The average is tuned by how well a cut of the model predicts the polls published in the days after it: the
average read at the date of each future poll, plus the house effect of its pollster, against the figure the
poll published. This module holds the pure functions of that harness and its orchestration; it reads no
database by itself.
"""
import numpy as np
import pandas as pd

from .forecaster import Forecaster

PREDICTION_COLUMNS = ['date', 'pollster_id', 'pollster', 'sample_size', 'party', 'published', 'predicted', 'residual']


def predict_polls(fc: Forecaster, polls: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """
    Predict the future polls of a cut with its fitted average and house effects.

    The average of a party is read at the date of the poll (`fit_forecast` leaves the straight prolongation up
    to `date_fit_last + max_fc`; after that day the last value is held) and the house effect of the pollster
    is added.

    Parameters
    ----------
    fc : Forecaster
        Forecaster with the average fitted (`fit_forecast`) on the polls of the cut.
    polls : pd.DataFrame
        Polls published after the cut: `date`, `pollster_id`, `sponsor_id`, `pollster`, `sample_size` and one
        column per party with the published figure.
    names : list of str
        Parties or blocks to predict.

    Returns
    -------
    pd.DataFrame
        One row per poll and party with a published figure: `date`, `pollster_id`, `pollster`, `sample_size`,
        `party`, `published`, `predicted` and `residual` (published minus predicted).
    """
    last = fc.date_fit_last + pd.Timedelta(days=fc.max_fc)
    average = {name: fc.forecast[name].ffill() for name in names}

    rows = []
    for poll in polls.to_dict('records'):
        date = min(pd.Timestamp(poll['date']), last)
        for name in names:
            published = poll[name]
            if pd.isnull(published):
                continue

            predicted = float(average[name].asof(date)) + fc.house_effect_of(poll['pollster_id'], name)
            rows.append({
                'date': poll['date'], 'pollster_id': poll['pollster_id'], 'pollster': poll['pollster'],
                'sample_size': poll['sample_size'], 'party': name, 'published': float(published),
                'predicted': predicted, 'residual': float(published) - predicted
            })

    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def score_polls(pred: pd.DataFrame, k: int) -> dict:
    """
    Score the predictions of the future polls of a horizon.

    The skill compares the mean squared error with the floor of pure sampling noise, the mean over the cells
    of `p (1 - p) / n` in points squared: 1 is a perfect prediction and 0 is as bad as the noise of the polls
    themselves. A missing sample size takes the median of the present ones; without any, the skill is NaN.

    Parameters
    ----------
    pred : pd.DataFrame
        Output of `predict_polls` already filtered to the horizon `k`.
    k : int
        Horizon in days, echoed in the result.

    Returns
    -------
    dict
        `k`, `n_polls` (distinct polls), `n_cells`, `rmse` and `mae` (equal weight per cell, in points),
        `skill` and `rmse_by_party` (party to RMSE). Everything is NaN with no cells.
    """
    n_cells = len(pred)
    if n_cells == 0:
        return {'k': k, 'n_polls': 0, 'n_cells': 0, 'rmse': np.nan, 'mae': np.nan, 'skill': np.nan,
                'rmse_by_party': {}}

    residual = pred['residual'].astype(float)
    mse = float((residual ** 2).mean())

    sample = pred['sample_size'].astype(float)
    if sample.notna().any():
        sample = sample.fillna(sample.median())
        p = pred['published'].astype(float) / 100
        floor = float((p * (1 - p) / sample).mean()) * 100 ** 2
    else:
        floor = np.nan
    skill = 1 - mse / floor if floor > 0 else np.nan

    keys = [column for column in ['date', 'pollster_id', 'sponsor_id'] if column in pred.columns]
    return {
        'k': k,
        'n_polls': len(pred[keys].drop_duplicates()),
        'n_cells': n_cells,
        'rmse': float(np.sqrt(mse)),
        'mae': float(residual.abs().mean()),
        'skill': skill,
        'rmse_by_party': {party: float(np.sqrt((group ** 2).mean())) for party, group in residual.groupby(pred['party'])}
    }


def autocorrelation(pred: pd.DataFrame) -> float:
    """
    Lag-1 autocorrelation of the residuals, ordered by date within each party and pooled over parties.

    A positive value means the residuals persist from one poll to the next: the average reacts too slowly.

    Parameters
    ----------
    pred : pd.DataFrame
        Output of `predict_polls`: `party`, `date` and `residual`.

    Returns
    -------
    float
        Pearson correlation between each residual and the next one of the same party; NaN with fewer than
        3 such pairs.
    """
    current, following = [], []
    for _, group in pred.sort_values('date', kind='stable').groupby('party'):
        values = group['residual'].astype(float).to_numpy()
        current.extend(values[:-1])
        following.extend(values[1:])

    if len(current) < 3 or np.std(current) == 0 or np.std(following) == 0:
        return np.nan

    return float(np.corrcoef(current, following)[0, 1])


def roughness(forecast: pd.DataFrame, names: list[str]) -> float:
    """
    Mean absolute day-to-day change of the fitted series, in points per day.

    Only consecutive days with a fitted value count; NaN gaps are not bridged.

    Parameters
    ----------
    forecast : pd.DataFrame
        Daily series, one column per party or block.
    names : list of str
        Columns to average over.

    Returns
    -------
    float
        Mean over `names` of the mean absolute daily change; 0 for flat series, NaN without any pair of
        consecutive fitted days.
    """
    changes = [forecast[name].astype(float).diff().abs().mean() for name in names]
    changes = [change for change in changes if not np.isnan(change)]

    return float(np.mean(changes)) if changes else np.nan
