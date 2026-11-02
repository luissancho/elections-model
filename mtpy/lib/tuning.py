"""
Tuning of the polling average on future polls.

The average is tuned by how well a cut of the model predicts the polls published in the days after it: the
average read at the date of each future poll, plus the house effect of its pollster, against the figure the
poll published. This module holds the pure functions of that harness and its orchestration; it reads no
database by itself.
"""
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
    for poll in polls.itertuples(index=False):
        date = min(pd.Timestamp(poll.date), last)
        for name in names:
            published = getattr(poll, name)
            if pd.isnull(published):
                continue

            predicted = float(average[name].asof(date)) + fc.house_effect_of(poll.pollster_id, name)
            rows.append({
                'date': poll.date, 'pollster_id': poll.pollster_id, 'pollster': poll.pollster,
                'sample_size': poll.sample_size, 'party': name, 'published': float(published),
                'predicted': predicted, 'residual': float(published) - predicted
            })

    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
