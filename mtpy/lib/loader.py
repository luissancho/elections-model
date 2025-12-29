import json
from lxml import html
import numpy as np
import os
import pandas as pd
import re
import requests

from typing import Any, Optional
from typing_extensions import Self

from ..core.app import Core
from ..models.elections import (
    Events, EventsData, EventsResults, Parties, Polls, PollsResults, Pollsters, Sponsors
)
from ..core.utils.helpers import (
    array_shift, format_number, is_number
)

from .computer import Computer


class InfoElectoralLoader(Core):

    fbase = 'PROV_02_{}_1'

    def __init__(
        self,
        scope: str,
        event_date: str,
        verbose: int = 0,
        path: Optional[str] = None
    ) -> None:
        super().__init__()

        self.scope = scope
        self.event_date = event_date
        self.verbose = verbose
        self.path = path or os.getcwd()

        self.headers = None
        self.names = None

        self.data = None
        self.totals = None
        self.results = None

        self.filters = [
            f"scope = '{self.scope}'",
            f"date = '{self.event_date}'"
        ]
        self.fname = self.fbase.format(self.event_date.replace('-', '')[:6])

        self.m_totals = EventsData()
        self.m_results = EventsResults()

        self.regions = self.app.fs.read_csv(f'{self.path}/es-regions.csv').set_index('code').region
        self.provinces = self.app.fs.read_csv(f'{self.path}/es-provinces.csv').set_index('code').province

        self.parties = Parties().get_results(formatted=True)
        if self.app.fs.exists(f'{self.path}/infoelectoral/{self.scope}/{self.fname}.json'):
            self.parties_colmap = json.loads(self.app.fs.read(f'{self.path}/infoelectoral/{self.scope}/{self.fname}.json'))
        else:
            self.parties_colmap = {}
        self.parties_idmap = self.parties.set_index('name').id.to_dict()

        self.parties_missing = []

    def read_file(self) -> pd.DataFrame:
        xls = pd.ExcelFile(f'{self.app.fspath}/{self.path}/infoelectoral/{self.scope}/{self.fname}.xlsx')
        df = xls.parse(xls.sheet_names[0]).dropna(how='all')

        loc_headers = df.loc[df[df.columns[1]].notnull()].index[0]
        loc_totals = df.loc[df[df.columns[1]].isnull()].index[-1]

        self.headers = df.loc[loc_headers].str.strip().tolist()
        self.names = df.loc[loc_headers - 2].dropna().str.strip().tolist()

        df = df.loc[loc_headers + 1:loc_totals + 1]

        return df

    def read_data(self) -> Self:
        col_map = {
            'Nombre de Comunidad': 'state',
            'Código de Provincia': 'region_id',
            'Nombre de Provincia': 'region',
            'Población': 'population',
            'Número de mesas': 'stations',
            'Censo electoral sin CERA': 'registered_non_cera',
            'Censo CERA': 'registered_cera',
            'Total censo electoral': 'registered',
            'Solicitudes voto CERA aceptadas': 'accepted_cera',
            'Total votantes CER': 'counted_cer',
            'Total votantes CERA': 'counted_cera',
            'Total votantes': 'counted',
            'Votos válidos': 'votes',
            'Votos a candidaturas': 'votes_candidates',
            'Votos en blanco': 'blank',
            'Votos nulos': 'invalid'
        }
        key_col = 'region_id'
        total_cols = [
            'stations', 'population', 'registered', 'counted', 'votes', 'blank', 'invalid'
        ]
        self.parties_missing = []

        df = self.read_file()

        main_cols = [v for k, v in col_map.items() if k.lower() in list(map(str.lower, self.headers))]
        party_cols = []
        for name in self.names:
            if name in main_cols:
                continue

            if name in self.parties_colmap:
                if self.parties_colmap[name]:
                    party_cols.append(self.parties_colmap[name])
                else:
                    party_cols.append('-')
            else:
                self.parties_missing.append(name)

        if '-' not in party_cols:
            ncols = df.shape[1]
            df.loc[:, [f'Unnamed: {ncols}', f'Unnamed: {ncols + 1}']] = 0
            party_cols += ['-']

        df = df.set_index(df[df.columns[1]].fillna(0)).rename_axis(key_col).sort_index()

        totals = df[df.columns[:len(main_cols)]]
        totals.columns = main_cols
        totals = totals.drop(columns=[key_col])[total_cols]
        totals.columns = pd.MultiIndex.from_product([['totals'], totals.columns])

        results = df[df.columns[len(main_cols):]]
        results_cols = []
        for c in party_cols:
            if c not in results_cols and c != '-':
                results_cols.append(c)
        results_cols.append('-')

        votes = results[[results.columns[i] for i in range(0, len(results.columns), 2)]]
        votes.columns = party_cols
        votes = votes.groupby(level=0, axis=1).sum()[results_cols]
        votes.columns = pd.MultiIndex.from_product([['votes'], votes.columns])

        seats = results[[results.columns[i] for i in range(1, len(results.columns), 2)]]
        seats.columns = party_cols
        seats = seats.groupby(level=0, axis=1).sum()[results_cols]
        seats.columns = pd.MultiIndex.from_product([['seats'], seats.columns])

        self.data = pd.concat([totals, votes, seats], axis=1).fillna(0).astype(int)

        return self

    def build_series(self) -> Self:
        totals = self.data['totals'].reset_index()
        totals['scope'] = self.scope
        totals['date'] = self.event_date
        totals['region'] = totals.region_id.map(self.provinces)
        totals['abstentions'] = totals.registered - totals.counted

        votes = self.data['votes'].rename_axis('party', axis=1).stack().rename('votes')
        seats = self.data['seats'].rename_axis('party', axis=1).stack().rename('seats')

        results = pd.concat([votes, seats], axis=1).reset_index()
        results['scope'] = self.scope
        results['date'] = self.event_date
        results['region'] = results.region_id.map(self.provinces)
        results['party_id'] = results.party.map(self.parties_idmap)
        results['pct'] = (100 * results['votes'] / results['region_id'].map(totals.set_index('region_id')['votes'])).round(2)

        totals['seats'] = results.groupby('region_id')['seats'].sum().values

        self.totals = self.m_totals.format_data(totals, int_type='nullable', bin_type='nullable', sort=True)
        self.results = self.m_results.format_data(results, int_type='nullable', bin_type='nullable', sort=True)

        return self

    def save_totals(self) -> Self:
        if self.totals.shape[0] == 0:
            return self

        if self.verbose > 0:
            print('Overwrite...')
        self.m_totals.execute('DELETE FROM {} WHERE {}'.format(self.m_totals.table, ' AND '.join(self.filters)))

        if self.verbose > 0:
            print('Save...')
        self.m_totals.stage_write(self.totals, compression='gz')
        nrows = self.m_totals.upsert()
        self.m_totals.stage_clean()
        self.m_totals.vacuum()
        if self.verbose > 0:
            print(f'{nrows} rows updated...')

        return self

    def save_results(self) -> Self:
        if self.results.shape[0] == 0:
            return self

        if self.verbose > 0:
            print('Overwrite...')
        self.m_results.execute('DELETE FROM {} WHERE {}'.format(self.m_results.table, ' AND '.join(self.filters)))

        if self.verbose > 0:
            print('Save...')
        self.m_results.stage_write(self.results, compression='gz')
        nrows = self.m_results.upsert()
        self.m_results.stage_clean()
        self.m_results.vacuum()
        if self.verbose > 0:
            print(f'{nrows} rows updated...')

        return self
    
    def show_summary(self) -> None:
        t_votes = self.totals['votes'].iloc[0] - self.totals['blank'].iloc[0]
        n_votes = self.results[self.results['region_id'] == 0]['votes'].sum()
        n_seats = self.results[self.results['region_id'] == 0]['seats'].sum()

        print('Total: {} | Votes: {} | Diff: {} | Seats: {}'.format(
            format_number(t_votes),
            format_number(n_votes),
            format_number(n_votes - t_votes),
            format_number(n_seats)
        ))

        n_voted = self.data['votes'].loc[0][self.data['votes'].loc[0] > 0].shape[0]
        n_seated = self.data['seats'].loc[0][self.data['seats'].loc[0] > 0].shape[0]

        print('Voted: {} | Seated: {}'.format(
            format_number(n_voted),
            format_number(n_seated)
        ))
    
    @property
    def votes(self) -> pd.DataFrame:
        df = self.data['votes'].rename_axis(None)
        df = df.where(df > 0, np.NaN)
        df.index = df.index.map(self.provinces).fillna('TOTAL')

        return df
    
    @property
    def seats(self) -> pd.DataFrame:
        df = self.data['seats'].rename_axis(None)
        df = df.where(df > 0, np.NaN)
        df.index = df.index.map(self.provinces).fillna('TOTAL')

        return df
    
    @property
    def pcts(self) -> pd.DataFrame:
        df = self.totals.set_index('region')['votes']
        df.index = df.index.fillna('TOTAL')
        df = self.votes.div(df, axis=0) * 100

        return df


class WikipediaLoader(Core):

    def __init__(
        self,
        scope: str,
        event_date: str,
        years: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        verbose: int = 0,
        path: Optional[str] = None
    ) -> None:
        super().__init__()

        self.scope = scope
        self.event_date = event_date
        self.years = years
        self.exclude = exclude
        self.verbose = verbose
        self.path = path or os.getcwd()

        self.data = None
        self.polls = None
        self.results = None
        self.computed = None

        self.filters = [
            f"event_scope = '{self.scope}'",
            f"event_date = '{self.event_date}'"
        ]
        self.charsep = '–'

        self.m_events = Events()
        self.m_polls = Polls()
        self.m_results = PollsResults()

        self.event = self.m_events.get_row(query={
            'filters': [
                f.replace('event_', '') for f in self.filters
            ]
        })

        self.urls = json.loads(self.app.fs.read(f'{self.path}/wikipedia/wp-urls.json'))
        self.maps = json.loads(self.app.get_().fs.read(f'{self.path}/wikipedia/wp-maps.json'))
        self.params = self.urls[self.scope][self.event_date]

        self.parties = Parties().get_results(formatted=True)
        self.parties_colmap = self.get_colmap('parties')
        self.parties_idmap = self.get_idmap('parties')
        self.parties_missing = []

        self.pollsters = Pollsters().get_results(formatted=True)
        self.pollsters_colmap = self.get_colmap('pollsters')
        self.pollsters_idmap = self.get_idmap('pollsters')
        self.pollsters_missing = []

        self.sponsors = Sponsors().get_results(formatted=True)
        self.sponsors_colmap = self.get_colmap('sponsors')
        self.sponsors_idmap = self.get_idmap('sponsors')
        self.sponsors_missing = []

    def get_colmap(self, key: str) -> dict[str, str]:
        return {i: p for p, v in self.maps[key].items() for i in v}

    def get_idmap(self, key: str) -> dict[str, str]:
        return getattr(self, key).set_index('name').id.to_dict()

    def parse_party(
        self,
        x: str
    ) -> str:
        if not x:
            return

        return re.sub(r'^(.*\/wiki\/)?(.+)$', r'\2', x.strip())

    def parse_dates(
        self,
        x: str,
        year: str
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        if not x:
            return

        parts = x.lower().split(self.charsep)
        start = parts[0].split()
        end = parts[1].split() if len(parts) > 1 else start
        if len(start) == 1 and len(end) > 1:
            start.append(end[1])
        if len(end) > 2 and len(end[2]) == 4:
            year = end[2]

        started_at = pd.to_datetime('{}-{}-{}'.format(year, start[1], start[0].zfill(2)))
        ended_at = pd.to_datetime('{}-{}-{}'.format(year, end[1], end[0].zfill(2)))
        if started_at > ended_at:
            started_at -= pd.DateOffset(years=1)

        return started_at, ended_at

    def read_cols(
        self,
        cols: list[html.HtmlElement],
        parties: list[str],
        year: str
    ) -> dict[str, Any]:
        is_election = (array_shift(cols[0].xpath('./b//text()')) or '').strip()

        if is_election:
            return {}

        data = {
            'pollster_id': None,
            'pollster': None,
            'sponsor_id': None,
            'sponsor': None,
            'name': None,
            'date': None,
            'start_date': None,
            'end_date': None,
            'sample_size': None,
            'results': []
        }

        poll = array_shift(cols[0].xpath('.//text()')).strip().replace(self.charsep, '-').split('/')
        if len(poll) > 1:
            data['pollster'], data['sponsor'] = poll
        else:
            note = (array_shift(cols[0].xpath('./span/text()')) or '').strip()
            data['pollster'] = ' '.join([poll[0], note]).strip()
            data['sponsor'] = None

        if data['pollster'] in self.pollsters_colmap:
            data['pollster'] = self.pollsters_colmap[data['pollster']]
        if data['pollster'] in self.pollsters_idmap:
            data['pollster_id'] = self.pollsters_idmap[data['pollster']]
        elif data['pollster'] is not None:
            self.pollsters_missing.append(data['pollster'])

        if data['sponsor'] in self.sponsors_colmap:
            data['sponsor'] = self.sponsors_colmap[data['sponsor']]
        if data['sponsor'] in self.sponsors_idmap:
            data['sponsor_id'] = self.sponsors_idmap[data['sponsor']]
        elif data['sponsor'] is not None:
            self.sponsors_missing.append(data['sponsor'])

        data['name'] = data['pollster'] + (' / ' + data['sponsor'] if data['sponsor'] is not None else '')

        if data['pollster_id'] is None:
            return {}
        if data['sponsor_id'] is None:
            data['sponsor_id'] = 0

        dates = array_shift(cols[1].xpath('.//text()')).strip()
        data['start_date'], data['end_date'] = self.parse_dates(dates, year)
        data['date'] = data['end_date']

        sample_size = array_shift(cols[2].xpath('.//text()')).replace(',', '').strip()
        data['sample_size'] = pd.to_numeric(sample_size, errors='coerce') if is_number(sample_size) else None

        lead = array_shift(cols[-1].xpath('.//text()'))
        if not is_number(lead):
            return {}

        for i, col in enumerate(cols[4:-1]):
            result = {
                'party_id': None,
                'party': parties[i],
                'pct': None,
                'seats': None,
                'seats_min': None,
                'seats_max': None
            }

            if result['party'] in self.parties_colmap:
                result['party'] = self.parties_colmap[result['party']]
            if result['party'] in self.parties_idmap:
                result['party_id'] = self.parties_idmap[result['party']]
            elif result['party'] is not None:
                self.parties_missing.append(result['party'])

            result['pct'] = pd.to_numeric(
                array_shift(col.xpath('.//text()')).strip(' ' + self.charsep),
                errors='coerce'
            )

            if pd.isnull(result['pct']) or result['pct'] <= 0:
                continue

            seats = array_shift(col.xpath('./span/text()'))
            if seats:
                seats_parts = [
                    pd.to_numeric(i, errors='coerce') for i in seats.strip().split('/') if is_number(i)
                ]
                if len(seats_parts) > 0:
                    result['seats_min'] = seats_parts[0]
                    result['seats_max'] = seats_parts[1] if len(seats_parts) > 1 else result['seats_min']
                    result['seats'] = np.floor(np.mean([result['seats_min'], result['seats_max']]))

            data['results'].append(result)

        if len(data['results']) == 0:
            return {}

        return data

    def read_rows(
        self,
        rows: list[html.HtmlElement],
        parties: list[str],
        year: str
    ) -> list[dict[str, Any]]:
        data = []  # list of rows
        remainder = []  # list of (index, text, nrows)

        for tr in rows:
            cols = []
            next_remainder = []

            context = None
            if 'style' in tr.attrib:
                color = array_shift(re.findall(r'background:\s*#([0-9a-fA-F]{6});', tr.attrib['style']))
                if color == 'EAFFEA':
                    context = 'exit'
                elif color == 'FFEAEA':
                    context = 'wban'

            index = 0
            tds = tr.xpath('./td')

            if len(tds) < 2:
                continue

            for td in tds:
                while remainder and remainder[0][0] <= index:
                    prev_i, prev_text, prev_rowspan = remainder.pop(0)
                    cols.append(prev_text)
                    if prev_rowspan > 1:
                        next_remainder.append((prev_i, prev_text, prev_rowspan - 1))
                    index += 1

                rowspan = int(array_shift(td.xpath('./@rowspan')) or 1)
                colspan = int(array_shift(td.xpath('./@colspan')) or 1)

                for _ in range(colspan):
                    cols.append(td)
                    if rowspan > 1:
                        next_remainder.append((index, td, rowspan - 1))
                    index += 1

            for prev_i, prev_text, prev_rowspan in remainder:
                cols.append(prev_text)
                if prev_rowspan > 1:
                    next_remainder.append((prev_i, prev_text, prev_rowspan - 1))

            cols = self.read_cols(cols, parties, year)
            if len(cols) > 0:
                cols['context'] = context
                data.append(cols)
            remainder = next_remainder

        while remainder:
            cols = []
            next_remainder = []

            for prev_i, prev_text, prev_rowspan in remainder:
                cols.append(prev_text)
                if prev_rowspan > 1:
                    next_remainder.append((prev_i, prev_text, prev_rowspan - 1))

            cols = self.read_cols(cols, parties, year)
            if len(cols) > 0:
                data.append(cols)
            remainder = next_remainder

        return data

    def read_table(
        self,
        table: html.HtmlElement,
        year: Optional[str] = None
    ) -> list[dict[str, Any]]:
        parties = [self.parse_party(th.xpath('.//a[1]/@href | ./text()')[0].strip()) for th in table.xpath('.//tr[1]/th')[4:-1]]
        if year is None:
            year = array_shift(table.xpath('./preceding-sibling::div[1]/*[self::h5 or self::h4 or self::h3]//text()'), '')[:4]

        if not is_number(year):
            return

        rows = self.read_rows(table.xpath('.//tr')[2:], parties, year)

        return rows

    def read_data(self) -> Self:
        data = {}

        self.parties_missing = []
        self.pollsters_missing = []
        self.sponsors_missing = []

        urls = self.params['polls']
        if not isinstance(urls, list):
            urls = [urls]

        if self.years is not None:
            years = self.years
        elif 'years' in self.params:
            years = self.params['years']
        else:
            years = []
        
        headers = {
            'User-Agent': 'ManyThings/1.0 (https://manythings.pro/; info@manythings.pro) mtpy/elections/1.0'
        }

        for url in urls:
            r = html.fromstring(requests.get(url, headers=headers).content)
            tables = r.xpath("//table[contains(@class, 'wikitable')]")

            for table in tables:
                year = array_shift(table.xpath('./preceding-sibling::div[1]/*[self::h5 or self::h4 or self::h3]//text()'), '')[:4]

                if len(years) > 0 and year not in years:
                    continue

                if not is_number(year):
                    year = str(self.event['date'].year)

                rows = self.read_table(table, year)

                if self.verbose > 0:
                    print(year, len(rows))

                if rows is None:
                    continue

                for row in rows:
                    data[(row['date'], row['pollster_id'], row['sponsor_id'])] = row

                if len(years) == 0:
                    break

        self.data = list(data.values())

        return self

    def build_series(self) -> Self:
        polls = []
        results = []

        if self.exclude is not None:
            exclude = self.exclude
        elif 'exclude' in self.params:
            exclude = self.params['exclude']
        else:
            exclude = []

        for row in self.data:
            poll = {k: v for k, v in row.items() if k != 'results'}

            n_parties = 0
            for result in row['results']:
                if result['party_id'] is None or result['party'] in exclude:
                    continue

                results.append({
                    'event_date': self.event['date'],
                    'event_scope': self.event['scope'],
                    'date': row['date'],
                    'pollster_id': row['pollster_id'],
                    'sponsor_id': row['sponsor_id']
                } | result)

                n_parties += 1

            poll['parties'] = n_parties

            polls.append(poll)

        polls = pd.DataFrame(polls)
        polls['event_date'] = pd.to_datetime(self.event['date'])
        polls['event_scope'] = self.event['scope']
        polls['pub_date'] = polls.end_date
        polls['mtype'] = polls.pollster_id.map(self.pollsters.set_index('id').mtype)
        polls['computed'] = False
        polls['internal'] = False
        polls['partisan'] = False
        polls['days'] = (polls['event_date'] - polls['date']).dt.days

        results = pd.DataFrame(results)
        results = results.set_index(self.m_results.key[:-1]).loc[polls.set_index(self.m_polls.key).index].reset_index()

        self.polls = self.m_polls.format_data(polls, int_type='nullable', bin_type='nullable', sort=True)
        self.results = self.m_results.format_data(results, int_type='nullable', bin_type='nullable', sort=True)

        return self

    def select_series(self, overwrite: bool = False) -> Self:
        if not overwrite:
            cur_polls = self.m_polls.get_results(query={'filters': self.filters}, formatted=True).set_index(self.m_polls.key)
            self.polls = self.polls.set_index(self.m_polls.key).drop(cur_polls.index, errors='ignore').reset_index()

        self.results = self.results.set_index(self.m_results.key[:-1]).loc[self.polls.set_index(self.m_polls.key).index].reset_index()

        return self

    def save_polls(self, overwrite: bool = False) -> Self:
        if self.polls.shape[0] == 0:
            return self

        if overwrite:
            if self.verbose > 0:
                print('Overwrite...')
            self.m_polls.execute('DELETE FROM {} WHERE {}'.format(self.m_polls.table, ' AND '.join(self.filters)))

        if self.verbose > 0:
            print('Save...')
        self.m_polls.stage_write(self.polls, compression='gz')
        nrows = self.m_polls.upsert()
        self.m_polls.stage_clean()
        self.m_polls.vacuum()
        if self.verbose > 0:
            print(f'{nrows} rows updated...')

        return self

    def save_results(self, overwrite: bool = False) -> Self:
        if self.results.shape[0] == 0:
            return self

        if overwrite:
            if self.verbose > 0:
                print('Overwrite...')
            self.m_results.execute('DELETE FROM {} WHERE {}'.format(self.m_results.table, ' AND '.join(self.filters)))

        if self.verbose > 0:
            print('Save...')
        self.m_results.stage_write(self.results, compression='gz')
        nrows = self.m_results.upsert()
        self.m_results.stage_clean()
        self.m_results.vacuum()
        if self.verbose > 0:
            print(f'{nrows} rows updated...')

        return self

    def compute_series(self, save: bool = False, overwrite: bool = False) -> Self:
        self.computed = Computer(
            scope=self.scope,
            event_dates=[self.event_date],
            verbose=self.verbose,
            path=self.path
        ).build_series().compute_weights(
            save=save,
            overwrite=overwrite
        )

        if self.verbose > 0:
            print(f'{self.computed.shape[0]} rows computed...')

        return self
    
    @property
    def parties_checks(self) -> pd.DataFrame:
        events = EventsResults().get_results(query=dict(
            filters=[
                f"scope = '{self.scope}'",
                f"date = '{self.event_date}'",
                'region_id = 0',
                'party_id > 0'
            ]
        ), formatted=True)

        if events.shape[0] > 0:
            event_parties = events.sort_values('votes', ascending=False)['party'].tolist()
        else:
            event_parties = []

        polls_parties = self.results['party'].unique().tolist()
        parties = event_parties + [p for p in polls_parties if p not in event_parties]

        df = pd.DataFrame({
            p: (
                int(p in event_parties),
                int(p in polls_parties)
            )
            for p in parties
        }, index=['event', 'polls'])
        df = df.where(df > 0, np.NaN)

        return df
