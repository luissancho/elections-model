import pandas as pd
import requests

from ...models.sources import elections as sources
from ...models import elections as models
from ...core.worker import Pipeline
from ...core.utils import helpers


class Events(Pipeline):

    def __init__(self):
        super().__init__()

        self.pipes = {
            'update': {
                'Extract': ['read_update'],
                'Transform': ['transform'],
                'Load': ['remove', 'upsert']
            },
            'bulk': {
                'Extract': ['read_bulk'],
                'Transform': ['transform'],
                'Load': ['insert']
            },
            'rebulk': {
                'Extract': ['read_bulk'],
                'Transform': ['transform'],
                'Load': ['remove', 'upsert']
            }
        }

        self.source = sources.Events()
        self.target = models.Events()

        self.path = self.app.config.elections.path
        self.key = self.app.config.elections.key
        self.names = self.app.config.elections.names.to_dict()

        self.nseq = None
        self.headers = {
            'Authorization': 'Basic {}'.format(self.key)
        }

    def get_current_nseq(self):
        r = requests.get(
            '{}/{}'.format(self.path, self.names['seq_id']),
            headers=self.headers,
            verify=False
        )

        nseq = int(r.text)
        self.app.logger.debug('nSeq {}...'.format(nseq))

        return nseq

    def parse_data(self, data):
        col_map = [
            'date', 'f1', 'f2', 'f3', 'event_id', 'f4', 'event', 'f5', 'f6', 'stations', 'registered',
            'counted', 'counted_rate', 'votes', 'votes_rate', 'abstentions', 'abstentions_rate',
            'blank', 'blank_rate', 'nulls', 'nulls_rate'
        ]
        seq = data.split(';')

        d = []
        i = 0
        while True:
            d.append({col_map[j]: v.strip() for j, v in enumerate(seq[i:i + len(col_map)])})

            i += len(col_map) + 200
            if i >= len(seq) - 1:
                break

        df = pd.DataFrame(d)

        return df

    def read_bulk_task(self, **kwargs):
        chunk = kwargs.get('chunk', None)

        if chunk is not None:
            path = self.names['local_{}'.format(chunk)]
        else:
            self.nseq = self.get_current_nseq()
            path = self.names['local'].format(self.nseq)

        r = requests.get(
            '{}/{}'.format(self.path, path),
            headers=self.headers,
            verify=False
        )

        self.data = self.parse_data(r.text)

        return self

    def read_update_task(self, **kwargs):
        rep_nseq = self.get_update_ts()
        self.nseq = self.get_current_nseq()

        if self.nseq <= rep_nseq:
            return self

        path = self.names['local'].format(self.nseq)

        r = requests.get(
            '{}/{}'.format(self.path, path),
            headers=self.headers,
            verify=False
        )

        self.data = self.parse_data(r.text)

        return self

    def transform_task(self, **kwargs):
        if self.rows == 0:
            return self

        df = self.data

        for col in df.columns[1:]:
            if helpers.is_numeric(df[col]):
                df[col] = df[col].astype(float) / 100 if '_rate' in col else df[col].astype(int)

        df['nseq'] = self.nseq
        df['updated_at'] = pd.to_datetime(df.date.str.replace('000000', '010000'), format='%y%m%d%H%M')
        df['year'] = df.updated_at.dt.year

        self.data = df[self.target.columns].sort_values(self.target.key)

        return self

    def remove_task(self, **kwargs):
        if self.rows == 0:
            return self

        filters = [
            'year = {}'.format(self.data.iloc[0]['year']),
            'event_id = {}'.format(self.data.iloc[0]['event_id'])
        ]

        self.app.logger.debug('Delete...')
        self.target.dal.query('DELETE FROM {} WHERE {}'.format(self.target.table, ' AND '.join(filters)))

        return self
