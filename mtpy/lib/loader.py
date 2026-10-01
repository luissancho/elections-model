import json
from lxml import html
import numpy as np
import os
import pandas as pd
import re
import requests

from typing import Any, Optional
from typing_extensions import Self
from urllib.parse import quote, unquote

from ..core.app import Core
from ..models.elections import (
    Events, EventsData, EventsResults, Parties, Polls, PollsResults, Pollsters, Sponsors
)
from .data import get_districts, get_scopes
from ..core.utils.helpers import (
    array_shift, format_number, is_number
)

from .computer import Computer

HEADERS = {
    'User-Agent': 'ManyThings/1.0 (https://manythings.pro/; info@manythings.pro) mtpy/elections/1.0'
}


def scope_colmap(maps: dict, scope: str, key: str) -> dict[str, str]:
    """
    Aliases of `key` (`parties`, `pollsters` or `sponsors`) of `wp-maps.json` mapped to their canonical names:
    the global ones plus those of the scope (`maps['scopes'][scope][key]`), which prevail. Every alias is
    percent-decoded.
    """
    colmap = {unquote(i): p for p, v in maps[key].items() for i in v}
    scoped = maps.get('scopes', {}).get(scope, {}).get(key, {})
    colmap.update({unquote(i): p for p, v in scoped.items() for i in v})

    return colmap


def fetch_page(
    url: str,
    cache_dir: Optional[str] = None,
    refresh: bool = False
) -> bytes:
    """
    Download a Wikipedia article, keeping a copy in `cache_dir` so that repeated loads (the curation of
    aliases is iterative) do not hit the network again.

    Parameters
    ----------
    url : str
        Address of the article.
    cache_dir : str, optional
        Directory of the cache; no cache when `None`.
    refresh : bool, optional
        Download again even if the article is cached.

    Returns
    -------
    bytes
        HTML of the article. A response with an error status raises `requests.HTTPError`.
    """
    fname = None
    if cache_dir is not None:
        fname = os.path.join(cache_dir, quote(unquote(url.split('/wiki/')[-1]), safe='()_-,') + '.html')
        if not refresh and os.path.exists(fname):
            with open(fname, 'rb') as fh:
                return fh.read()

    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()

    if fname is not None:
        os.makedirs(cache_dir, exist_ok=True)
        with open(fname, 'wb') as fh:
            fh.write(r.content)

    return r.content


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
        self.path = path or '.'  # Relative to the app file system root (files/)

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

        self.regions = self.app.data.read_csv('es-regions.csv').set_index('code').region
        self.provinces = self.app.data.read_csv('es-provinces.csv').set_index('code').province

        self.parties = Parties().get_results(formatted=True)
        if self.app.data.exists(f'infoelectoral/{self.scope}/{self.fname}.json'):
            self.parties_colmap = json.loads(self.app.data.read(f'infoelectoral/{self.scope}/{self.fname}.json'))
        else:
            self.parties_colmap = {}
        self.parties_idmap = self.parties.set_index('name').id.to_dict()

        self.parties_missing = []

    def read_file(self) -> pd.DataFrame:
        xls = pd.ExcelFile(self.app.data.get_path(f'infoelectoral/{self.scope}/{self.fname}.xlsx'))
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
        votes = votes.T.groupby(level=0).sum().T[results_cols]
        votes.columns = pd.MultiIndex.from_product([['votes'], votes.columns])

        seats = results[[results.columns[i] for i in range(1, len(results.columns), 2)]]
        seats.columns = party_cols
        seats = seats.T.groupby(level=0).sum().T[results_cols]
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
        df = df.where(df > 0, np.nan)
        df.index = df.index.map(self.provinces).fillna('TOTAL')

        return df
    
    @property
    def seats(self) -> pd.DataFrame:
        df = self.data['seats'].rename_axis(None)
        df = df.where(df > 0, np.nan)
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
        self.path = path or '.'  # Relative to the app file system root (files/)
        self.cache_dir = None  # Directory where the downloaded articles are cached (see `fetch_page`)

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

        self.urls = json.loads(self.app.data.read('wikipedia/wp-urls.json'))
        self.maps = json.loads(self.app.data.read('wikipedia/wp-maps.json'))
        self.params = self.urls[self.scope][self.event_date]  # `polls` is optional: an event may have results only

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
        """
        Aliases of `key` (`parties`, `pollsters` or `sponsors`) mapped to their canonical names: the global
        ones of `wp-maps.json` plus those of the scope (`scopes[scope][key]`), which prevail. Wikipedia links
        come percent-encoded or not, so every alias is decoded.
        """
        return scope_colmap(self.maps, self.scope, key)

    def get_idmap(self, key: str) -> dict[str, str]:
        return getattr(self, key).set_index('name').id.to_dict()

    def parse_party(
        self,
        x: str
    ) -> str:
        if not x:
            return

        return unquote(re.sub(r'^(.*\/wiki\/)?(.+)$', r'\2', x.strip()))

    @staticmethod
    def find_poll_table(doc: html.HtmlElement) -> Optional[html.HtmlElement]:
        """
        Table of voting intention polls of an election article: the first `wikitable` whose first row starts
        with "Polling firm" and whose closest previous heading (h2, h3 or h4) contains "Voting intention";
        otherwise, the first one hanging directly from the heading "Opinion polls" (small articles have no
        subsections).

        Parameters
        ----------
        doc : html.HtmlElement
            Parsed article.

        Returns
        -------
        html.HtmlElement or None
            The table, or `None` when the article has no polls table.
        """
        fallback = None
        for table in doc.xpath("//table[contains(@class, 'wikitable')]"):
            first = table.xpath('.//tr[1]')
            if len(first) == 0 or not ' '.join(first[0].text_content().split()).startswith('Polling firm'):
                continue

            heading = table.xpath('preceding::*[self::h2 or self::h3 or self::h4][1]')
            heading = ' '.join(heading[0].text_content().split()) if len(heading) > 0 else ''
            if 'Voting intention' in heading:
                return table
            if fallback is None and heading.startswith('Opinion polls'):
                fallback = table

        return fallback

    @staticmethod
    def read_header(table: html.HtmlElement) -> tuple[dict[int, str], Optional[int]]:
        """
        Party columns and "Lead" column of a polls table, from the cells of its first row.

        Parameters
        ----------
        table : html.HtmlElement
            Polls table.

        Returns
        -------
        tuple
            `{column index: wiki key}` of the cells that link to an article (not to a `File:`), and the index
            of the cell whose text is "Lead" (`None` when it is not in that row). Indexes count `colspan`.
        """
        parties = {}
        lead = None
        index = 0
        for th in table.xpath('.//tr[1]/th'):
            hrefs = [unquote(re.sub(r'^(.*\/wiki\/)?(.+)$', r'\2', h.strip())) for h in th.xpath('.//a/@href')]
            hrefs = [h for h in hrefs if h and not h.startswith('File:') and not h.startswith('#')]
            text = ' '.join(th.text_content().split())

            if text == 'Lead':
                lead = index
            elif len(hrefs) > 0:
                parties[index] = hrefs[0]

            index += int(array_shift(th.xpath('./@colspan')) or 1)

        return parties, lead

    def parse_dates(
        self,
        x: str,
        year: str
    ) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Fieldwork dates of a poll from the text of its cell ("5–7 Apr 2005", "30 Mar–6 Apr 2005",
        "17 Apr 2005"). An unknown start ("?–16 May 2015") takes the end date.

        Parameters
        ----------
        x : str
            Text of the cell.
        year : str
            Year used when the cell does not state it.

        Returns
        -------
        tuple of pd.Timestamp or None
            Start and end dates; `None` when the cell is empty or holds no day (e.g. "Dec 2019", "?").
        """
        if not x:
            return

        parts = x.lower().split(self.charsep)
        start = parts[0].split()
        end = parts[1].split() if len(parts) > 1 else start
        if len(start) == 1 and len(end) > 1:
            start.append(end[1])
        if len(end) > 2 and len(end[2]) == 4:
            year = end[2]
        if len(end) < 2 or not end[0].isdigit():
            return
        if len(start) < 2 or not start[0].isdigit():
            start = end

        try:
            started_at = pd.to_datetime('{}-{}-{}'.format(year, start[1], start[0].zfill(2)))
            ended_at = pd.to_datetime('{}-{}-{}'.format(year, end[1], end[0].zfill(2)))
        except ValueError:
            return
        if started_at > ended_at:
            started_at -= pd.DateOffset(years=1)

        return started_at, ended_at

    def read_cols(
        self,
        cols: list[html.HtmlElement],
        parties: dict[int, str],
        lead_col: Optional[int],
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

        dates = self.parse_dates((array_shift(cols[1].xpath('.//text()')) or '').strip(), year)
        if dates is None:
            return {}
        data['start_date'], data['end_date'] = dates
        data['date'] = data['end_date']

        sample_size = array_shift(cols[2].xpath('.//text()')).replace(',', '').strip()
        data['sample_size'] = pd.to_numeric(sample_size, errors='coerce') if is_number(sample_size) else None

        # The "Lead" cell tells polls from other rows; it is the last one when the header does not place it
        lead_col = lead_col if lead_col is not None and lead_col < len(cols) else len(cols) - 1
        lead = array_shift(cols[lead_col].xpath('.//text()'))
        if not is_number(lead):
            return {}

        for i, party in parties.items():
            if i >= len(cols):
                continue

            col = cols[i]
            result = {
                'party_id': None,
                'party': party,
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
                (array_shift(col.xpath('.//text()')) or '').strip(' ' + self.charsep),
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
        parties: dict[int, str],
        lead_col: Optional[int],
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

            cols = self.read_cols(cols, parties, lead_col, year)
            if len(cols) > 0:
                cols['ctype'] = context
                data.append(cols)
            remainder = next_remainder

        while remainder:
            cols = []
            next_remainder = []

            for prev_i, prev_text, prev_rowspan in remainder:
                cols.append(prev_text)
                if prev_rowspan > 1:
                    next_remainder.append((prev_i, prev_text, prev_rowspan - 1))

            cols = self.read_cols(cols, parties, lead_col, year)
            if len(cols) > 0:
                data.append(cols)
            remainder = next_remainder

        return data

    def read_table(
        self,
        table: html.HtmlElement,
        year: Optional[str] = None
    ) -> list[dict[str, Any]]:
        parties, lead_col = self.read_header(table)
        if year is None:
            year = array_shift(table.xpath('./preceding-sibling::div[1]/*[self::h5 or self::h4 or self::h3]//text()'), '')[:4]

        if not is_number(year):
            return

        # Header rows have no `td`, so `read_rows` skips them
        rows = self.read_rows(table.xpath('.//tr'), parties, lead_col, year)

        return rows

    def read_data(self) -> Self:
        data = {}

        self.parties_missing = []
        self.pollsters_missing = []
        self.sponsors_missing = []

        urls = self.params.get('polls', [])
        if not isinstance(urls, list):
            urls = [urls]

        if self.years is not None:
            years = self.years
        elif 'years' in self.params:
            years = self.params['years']
        else:
            years = []
        
        for url in urls:
            r = html.fromstring(fetch_page(url, self.cache_dir))
            tables = r.xpath("//table[contains(@class, 'wikitable')]")

            if len(years) == 0:
                # A single table: the voting intention one (the first table of the page when not found)
                table = self.find_poll_table(r)
                tables = [table] if table is not None else tables[:1]

            for table in tables:
                year = array_shift(table.xpath('./preceding-sibling::div[1]/*[self::h5 or self::h4 or self::h3]//text()'), '')[:4]

                if len(years) > 0 and year not in years:
                    continue

                if not is_number(year):
                    year = self.event_date[:4]

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
                    'event_date': pd.to_datetime(self.event_date),
                    'event_scope': self.scope,
                    'date': row['date'],
                    'pollster_id': row['pollster_id'],
                    'sponsor_id': row['sponsor_id']
                } | result)

                n_parties += 1

            # A poll without any mapped party has no rows in `polls_results`: left out
            if n_parties == 0:
                continue

            poll['parties'] = n_parties

            polls.append(poll)

        polls = pd.DataFrame(polls)
        polls['event_date'] = pd.to_datetime(self.event_date)
        polls['event_scope'] = self.scope
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
        df = df.where(df > 0, np.nan)

        return df


def table_grid(rows: list[html.HtmlElement]) -> list[list[Optional[html.HtmlElement]]]:
    """
    Expand the rows of an HTML table into a rectangular grid: a cell with `colspan` or `rowspan` is repeated
    in every position it covers, so that columns can be addressed by index.

    Parameters
    ----------
    rows : list of html.HtmlElement
        `tr` elements.

    Returns
    -------
    list of list
        One list of cells (`th` or `td`) per row; `None` where the row has no cell.
    """
    grid = []
    pending = {}  # column index -> (cell, rows left)

    for tr in rows:
        line = []
        cells = tr.xpath('./th|./td')
        index = 0
        pos = 0
        while pos < len(cells) or index in pending:
            if index in pending:
                cell, left = pending[index]
                line.append(cell)
                if left > 1:
                    pending[index] = (cell, left - 1)
                else:
                    del pending[index]
                index += 1
                continue

            cell = cells[pos]
            pos += 1
            rowspan = int(array_shift(cell.xpath('./@rowspan')) or 1)
            colspan = int(array_shift(cell.xpath('./@colspan')) or 1)
            for _ in range(colspan):
                line.append(cell)
                if rowspan > 1:
                    pending[index] = (cell, rowspan - 1)
                index += 1

        grid.append(line)

    return grid


def cell_text(cell: Optional[html.HtmlElement]) -> str:
    """Text of a table cell with the white space collapsed (empty for a missing cell)."""
    return ' '.join(cell.text_content().split()) if cell is not None else ''


def cell_key(cell: html.HtmlElement) -> Optional[str]:
    """Decoded wiki key of the first article linked from a cell (`None` when it links none)."""
    for href in cell.xpath('.//a/@href'):
        key = unquote(re.sub(r'^(.*\/wiki\/)?(.+)$', r'\2', href.strip()))
        if key and not key.startswith('File:') and not key.startswith('#') and 'cite_note' not in key:
            return key

    return None


def cell_number(text: str) -> Optional[float]:
    """Number of a table cell ("1,217,164", "31.40", "−" for none); `None` when it is not a number."""
    text = text.replace(',', '').replace('−', '-').strip()

    return float(text) if is_number(text) else None


class WikipediaResultsLoader(Core):
    """
    Official results of a regional election from its article in the English Wikipedia: *Results › Overall*
    (votes, share and seats of every candidacy, and the totals of the scope) and *Results › Distribution by
    constituency* (share and seats per district, without votes).
    """

    OTHERS = '-'  # Candidacies without a mapped party (id 0), as in `InfoElectoralLoader`
    REGIONAL_LIST = 100  # `region_id` of the Canarian regional list: the whole scope votes in it
    MONTHS = [
        'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre',
        'Noviembre', 'Diciembre'
    ]

    def __init__(
        self,
        scope: str,
        event_date: str,
        verbose: int = 0,
        path: Optional[str] = None
    ) -> None:
        """
        Parameters
        ----------
        scope : str
            Regional scope (`es-*`).
        event_date : str
            Date of the election, as listed in `data/wikipedia/wp-urls.json`.
        verbose : int, optional
            Level of verbosity.
        path : str, optional
            Relative to the app file system root (`files/`).
        """
        super().__init__()

        self.scope = scope
        self.event_date = event_date
        self.verbose = verbose
        self.path = path or '.'
        self.cache_dir = None  # Directory where the downloaded articles are cached (see `fetch_page`)

        self.overall = None  # Candidacies of the *Overall* table (see `parse_overall`)
        self.overall_totals = None
        self.constituencies = None  # Rows of the table by constituency (see `parse_constituencies`)

        self.event = None
        self.totals = None
        self.results = None

        self.filters = [
            f"scope = '{self.scope}'",
            f"date = '{self.event_date}'"
        ]

        self.m_events = Events()
        self.m_totals = EventsData()
        self.m_results = EventsResults()

        self.urls = json.loads(self.app.data.read('wikipedia/wp-urls.json'))
        self.maps = json.loads(self.app.data.read('wikipedia/wp-maps.json'))
        self.params = self.urls[self.scope][self.event_date]

        self.scope_info = get_scopes().loc[self.scope]
        self.districts = get_districts(self.scope)

        self.parties = Parties().get_results(formatted=True)
        self.parties_colmap = scope_colmap(self.maps, self.scope, 'parties')
        self.parties_idmap = self.parties.set_index('name').id.to_dict()
        self.parties_missing = []

    @property
    def is_upcoming(self) -> bool:
        """Whether the event has not been held yet (no results to read)."""
        return pd.Timestamp(self.event_date) > pd.Timestamp.now().normalize()

    def read_data(self) -> Self:
        """
        Download the article of the election and read its results tables. Nothing is read for an upcoming
        event. `parties_missing` lists the candidacies without a mapped party that won seats or at least 1 %
        of the valid votes (the rest add to "others" silently).
        """
        self.parties_missing = []
        if self.is_upcoming:
            return self

        doc = html.fromstring(fetch_page(self.params['results'], self.cache_dir))
        overall, constituencies = self.find_results_tables(doc)
        if overall is None:
            raise ValueError('No overall results table in {}'.format(self.params['results']))

        self.overall, self.overall_totals = self.parse_overall(overall)
        self.constituencies = self.parse_constituencies(constituencies) if constituencies is not None else None

        names = self.overall.apply(lambda r: self.party_name(r['key'], r['abbr'], self.parties_colmap), axis=1)
        relevant = (self.overall['seats'] > 0) | (self.overall['pct'] >= 1)
        missing = self.overall.loc[(names == self.OTHERS) & relevant]
        self.parties_missing = [k if k is not None else a for k, a in zip(missing['key'], missing['abbr'].fillna(missing['label']))]

        return self

    def build_series(self) -> Self:
        """
        Build the rows of `events`, `events_data` and `events_results` of the event: from the tables read
        (see `build_frames`) or, for an upcoming event, the seats of the districts in force without votes.
        """
        date = pd.Timestamp(self.event_date)
        name = 'Próximas' if self.is_upcoming else '{} {}'.format(self.MONTHS[date.month - 1], date.year)
        self.event = pd.DataFrame([{
            'date': date, 'scope': self.scope, 'name': 'Elecciones {} {}'.format(self.scope_info['name'], name),
            'featured': False
        }])

        if self.is_upcoming:
            totals = self.districts.loc[self.districts['seats'].notnull(), ['region_id', 'name', 'population', 'seats']]
            totals = totals.rename(columns={'name': 'region'})
            total = pd.DataFrame([{'region_id': 0, 'region': None, 'population': None, 'seats': totals['seats'].sum()}])
            totals = pd.concat([total, totals], ignore_index=True)
            totals['date'] = date
            totals['scope'] = self.scope
            results = pd.DataFrame(columns=['date', 'scope', 'region_id', 'region', 'party', 'votes', 'pct', 'seats'])
        else:
            official = None
            fname = 'results/{}/{}.csv'.format(self.scope, self.event_date)
            if self.app.data.exists(fname):
                official = self.app.data.read_csv(fname)

            totals, results = self.build_frames(
                self.scope, self.event_date, self.overall, self.overall_totals, self.constituencies,
                self.districts, self.parties_colmap, official=official
            )

        results['party_id'] = results['party'].map(lambda n: 0 if n == self.OTHERS else self.parties_idmap.get(n))
        unknown = sorted(set(results.loc[results['party_id'].isnull(), 'party']))
        if len(unknown) > 0:
            raise ValueError('Parties mapped in wp-maps.json but missing in the parties table: {}'.format(unknown))

        self.totals = self.m_totals.format_data(totals, int_type='nullable', bin_type='nullable', sort=True)
        self.results = self.m_results.format_data(results, int_type='nullable', bin_type='nullable', sort=True)

        return self

    def save_event(self) -> Self:
        """
        Create the row of the event in `events`, or update its name; `featured` of an existing row is kept
        (it is set by `mtpy.lib.data.update_featured`).
        """
        current = self.m_events.get_results(query={'filters': self.filters}, formatted=True)
        event = self.event.copy()
        if current.shape[0] > 0:
            event['featured'] = bool(current['featured'].iloc[0])

        self.m_events.upsert(self.m_events.format_data(event, int_type='nullable', bin_type='nullable', sort=True))

        return self

    def save_totals(self) -> Self:
        """Replace the rows of the event in `events_data`."""
        return self._save(self.m_totals, self.totals)

    def save_results(self) -> Self:
        """Replace the rows of the event in `events_results`."""
        return self._save(self.m_results, self.results)

    def _save(self, model: Any, data: pd.DataFrame) -> Self:
        """Delete the rows of the event from the table of `model` and write `data`."""
        if data is None or data.shape[0] == 0:
            return self

        model.execute('DELETE FROM {} WHERE {}'.format(model.table, ' AND '.join(self.filters)))
        nrows = model.upsert(data)
        if self.verbose > 0:
            print(f'{nrows} rows updated...')

        return self

    def show_summary(self) -> None:
        """Print the consistency checks of the event: votes of the parties against the valid votes, and seats."""
        if self.results is None or self.results.shape[0] == 0:
            print('Upcoming event: {} districts, {} seats'.format(
                self.totals.shape[0] - 1, format_number(self.totals.loc[self.totals['region_id'] == 0, 'seats'].iloc[0])
            ))
            return

        total = self.totals.loc[self.totals['region_id'] == 0].iloc[0]
        t_votes = total['votes'] - total['blank']
        n_votes = self.results.loc[self.results['region_id'] == 0, 'votes'].sum()
        n_seats = self.results.loc[self.results['region_id'] == 0, 'seats'].sum()
        d_seats = self.results.loc[self.results['region_id'] > 0, 'seats'].sum()

        print('Total: {} | Votes: {} | Diff: {} | Seats: {} | District seats: {}'.format(
            format_number(t_votes), format_number(n_votes), format_number(n_votes - t_votes),
            format_number(n_seats), format_number(d_seats)
        ))

    @staticmethod
    def party_name(key: Optional[str], abbr: Optional[str], party_names: dict[str, str]) -> str:
        """
        Party of a candidacy: the alias of its wiki key or, failing that, of its abbreviation; "others" when
        neither is mapped (never the party table directly: abbreviations collide between communities).
        """
        if key is not None and key in party_names:
            return party_names[key]
        if abbr is not None and abbr in party_names:
            return party_names[abbr]

        return WikipediaResultsLoader.OTHERS

    @staticmethod
    def build_frames(
        scope: str,
        event_date: str,
        overall: pd.DataFrame,
        totals: dict[str, int],
        constituencies: Optional[pd.DataFrame],
        districts: pd.DataFrame,
        party_names: dict[str, str],
        official: Optional[pd.DataFrame] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Rows of `events_data` and `events_results` of a past election from its parsed tables.

        The total of the scope (`region_id = 0`) is exact. The article gives shares and seats per district but
        no votes: the valid votes of the scope are split among the districts in proportion to their population
        (`estimated = True`) unless `official` brings the figure; the votes of a party in a district are its
        share of them. The Canarian regional list (`REGIONAL_LIST`) is voted by the whole scope, so it takes
        the valid votes of the total. A single-district community gets one district row equal to the total.

        Parameters
        ----------
        scope : str
            Regional scope.
        event_date : str
            Date of the election.
        overall : pd.DataFrame
            Candidacies of the scope (see `parse_overall`).
        totals : dict
            Totals of the scope (see `parse_overall`).
        constituencies : pd.DataFrame, optional
            Shares and seats per district (see `parse_constituencies`); `None` in a single-district scope.
        districts : pd.DataFrame
            Districts of the scope (`data/es-districts.csv`).
        party_names : dict
            Party of each wiki key or abbreviation; candidacies in neither add to "others".
        official : pd.DataFrame, optional
            Official valid votes per district: `region_id`, `votes` and, optionally, `blank`.

        Returns
        -------
        tuple of pd.DataFrame
            `events_data` rows (`date`, `scope`, `region_id`, `region`, `seats`, `population`, `registered`,
            `counted`, `votes`, `abstentions`, `blank`, `invalid`, `estimated`) and `events_results` rows
            (`date`, `scope`, `region_id`, `region`, `party`, `votes`, `pct`, `seats`).
        """
        cls = WikipediaResultsLoader
        valid, blank = int(totals['votes']), int(totals.get('blank', 0))

        # --- Total of the scope
        parties = overall.assign(party=[cls.party_name(k, a, party_names) for k, a in zip(overall['key'], overall['abbr'])])
        res0 = parties.groupby('party', sort=False)[['votes', 'seats']].sum().reset_index()
        res0['pct'] = (100. * res0['votes'] / valid).round(2)
        res0['region_id'] = 0
        res0['region'] = None

        data = [{
            'region_id': 0, 'region': None, 'seats': int(totals['seats']), 'population': None,
            'registered': totals.get('registered'), 'counted': totals.get('counted'), 'votes': valid,
            'abstentions': totals.get('abstentions'), 'blank': blank, 'invalid': totals.get('invalid'),
            'estimated': False
        }]
        results = [res0]

        if constituencies is None:
            # --- Single district: one unit of allocation equal to the total
            single = districts.loc[districts['seats'].notnull()]
            if single.shape[0] != 1:
                raise ValueError('{} has no table by constituency and {} districts in force'.format(scope, single.shape[0]))
            single = single.iloc[0]
            data.append(data[0] | {'region_id': int(single['region_id']), 'region': single['name'], 'population': single['population']})
            results.append(res0.assign(region_id=int(single['region_id']), region=single['name']))
        else:
            # --- Districts: resolved by name or alias
            lookup = {}
            for _, d in districts.iterrows():
                lookup[d['name']] = d
                for alias in str(d['aliases']).split('|') if isinstance(d['aliases'], str) else []:
                    if alias:
                        lookup[alias] = d
            unknown = sorted(set(constituencies['district']) - set(lookup))
            if len(unknown) > 0:
                raise ValueError('Constituencies of {} {} not in es-districts.csv (name or alias): {}'.format(scope, event_date, unknown))

            const = constituencies.assign(
                region_id=[int(lookup[n]['region_id']) for n in constituencies['district']],
                party=[cls.party_name(k, a, party_names) for k, a in zip(constituencies['key'], constituencies['abbr'])]
            )
            regions = districts.set_index('region_id').loc[sorted(const['region_id'].unique())]

            split = regions.loc[regions.index != cls.REGIONAL_LIST]
            votes = cls.split_votes(valid, split['population'])
            estimated = pd.Series(True, index=regions.index)
            blanks = (blank * votes / valid).round()
            if cls.REGIONAL_LIST in regions.index:
                votes.loc[cls.REGIONAL_LIST] = valid
                blanks.loc[cls.REGIONAL_LIST] = blank
            if official is not None:
                off = official.set_index('region_id')
                for rid in [r for r in off.index if r in regions.index]:
                    votes.loc[rid] = int(off.loc[rid, 'votes'])
                    estimated.loc[rid] = False
                    if 'blank' in off.columns and pd.notnull(off.loc[rid, 'blank']):
                        blanks.loc[rid] = int(off.loc[rid, 'blank'])
                    else:
                        blanks.loc[rid] = round(blank * votes.loc[rid] / valid)

            res = const.groupby(['region_id', 'party'], sort=False)[['pct', 'seats']].sum().reset_index()
            res['votes'] = (res['pct'] * res['region_id'].map(votes) / 100.).round().astype(int)

            # "Others" of each district: what is left of its valid votes, net of the blank ballots
            listed = res.loc[res['party'] != cls.OTHERS].groupby('region_id')['votes'].sum()
            rest = (votes - blanks - listed.reindex(votes.index).fillna(0)).clip(lower=0)
            others = res.loc[res['party'] == cls.OTHERS].set_index('region_id')['seats'].reindex(votes.index).fillna(0)
            res = pd.concat([
                res.loc[res['party'] != cls.OTHERS],
                pd.DataFrame({
                    'region_id': votes.index, 'party': cls.OTHERS, 'votes': rest.astype(int).values,
                    'pct': (100. * rest / votes).round(2).values, 'seats': others.astype(int).values
                })
            ], ignore_index=True)
            res['region'] = res['region_id'].map(regions['name'])
            results.append(res)

            seats = res.groupby('region_id')['seats'].sum()
            for rid, d in regions.iterrows():
                data.append({
                    'region_id': int(rid), 'region': d['name'], 'seats': int(seats.loc[rid]),
                    'population': d['population'] if pd.notnull(d['population']) else None,
                    'registered': None, 'counted': None, 'votes': int(votes.loc[rid]), 'abstentions': None,
                    'blank': int(blanks.loc[rid]), 'invalid': None, 'estimated': bool(estimated.loc[rid])
                })

        data = pd.DataFrame(data)
        results = pd.concat(results, ignore_index=True)
        for df in (data, results):
            df.insert(0, 'scope', scope)
            df.insert(0, 'date', pd.Timestamp(event_date))

        return data, results[['date', 'scope', 'region_id', 'region', 'party', 'votes', 'pct', 'seats']]

    @staticmethod
    def find_results_tables(doc: html.HtmlElement) -> tuple[Optional[html.HtmlElement], Optional[html.HtmlElement]]:
        """
        Results tables of an election article.

        Parameters
        ----------
        doc : html.HtmlElement
            Parsed article.

        Returns
        -------
        tuple
            The *Overall* table (the `wikitable` whose caption contains "Summary of") and the table by
            constituency (first header cell "Constituency"); `None` for the one that is missing, as the
            second in the single-district communities.
        """
        overall = None
        constituencies = None
        for table in doc.xpath("//table[contains(@class, 'wikitable')]"):
            caption = table.xpath('./caption')
            if overall is None and len(caption) > 0 and 'Summary of' in cell_text(caption[0]):
                overall = table
                continue

            first = table.xpath('.//tr[1]/th[1]')
            if constituencies is None and len(first) > 0 and cell_text(first[0]) == 'Constituency':
                constituencies = table

        return overall, constituencies

    @staticmethod
    def parse_overall(table: html.HtmlElement) -> tuple[pd.DataFrame, dict[str, int]]:
        """
        Read the *Overall* table: one row per candidacy and the totals of the scope.

        The columns are located by their header cells ("Votes", "%" and the "Total" of the seats), not by
        position; with two blocks of votes (Canary Islands: island and regional constituencies) the first
        one is read.

        Parameters
        ----------
        table : html.HtmlElement
            The *Overall* table.

        Returns
        -------
        tuple
            A frame with `key` (wiki key of the linked article, or `None`), `label`, `abbr` (text of the last
            parentheses), `votes`, `pct` and `seats`; and a dict with `votes` (valid), `blank`, `invalid`,
            `counted`, `abstentions`, `registered` and `seats`.
        """
        grid = table_grid(table.xpath('.//tr'))
        rows = table.xpath('.//tr')

        # Header: the last row made only of `th` cells before the first candidacy
        cols = {}
        for tr, line in zip(rows, grid):
            if len(tr.xpath('./td')) > 0 and len(cols) > 0:
                break
            labels = [cell_text(c) for c in line]
            if 'Votes' in labels and '%' in labels:
                cols = {
                    'votes': labels.index('Votes'), 'pct': labels.index('%'),
                    'seats': max(i for i, v in enumerate(labels) if v in ('Total', 'Won', 'Seats'))
                }
        if len(cols) == 0:
            raise ValueError('Header of the overall results table not found')

        totals_map = {
            'Blank ballots': 'blank', 'Total': 'total', 'Valid votes': 'votes', 'Invalid votes': 'invalid',
            'Votes cast / turnout': 'counted', 'Abstentions': 'abstentions', 'Registered voters': 'registered'
        }
        data = []
        totals = {}
        for tr, line in zip(rows, grid):
            tds = tr.xpath('./td')
            if len(tds) < 2:
                continue

            first = cell_text(tds[0])
            if first in totals_map:
                name = totals_map[first]
                if name == 'total':
                    seats = cell_number(cell_text(line[cols['seats']]))
                    if seats is not None:
                        totals['seats'] = int(seats)
                else:
                    value = cell_number(cell_text(line[cols['votes']]))
                    if value is not None:
                        totals[name] = int(value)
                continue

            # Rows of the members of a coalition hang from the colour cell of its row (`rowspan`): skipped,
            # their votes and seats are already in the row of the coalition
            if len(line) <= cols['seats'] or line[0] is not tds[0]:
                continue

            votes = cell_number(cell_text(line[cols['votes']]))
            label = cell_text(line[1])
            if votes is None or not label:
                continue

            abbr = re.findall(r'\(([^()]*)\)[^()]*$', label)
            pct = cell_number(cell_text(line[cols['pct']]))
            seats = cell_number(cell_text(line[cols['seats']]))
            data.append({
                'key': cell_key(line[1]), 'label': label, 'abbr': abbr[0] if len(abbr) > 0 else None,
                'votes': int(votes), 'pct': pct if pct is not None else 0., 'seats': int(seats) if seats is not None else 0
            })

        return pd.DataFrame(data, columns=['key', 'label', 'abbr', 'votes', 'pct', 'seats']), totals

    @staticmethod
    def parse_constituencies(table: html.HtmlElement) -> pd.DataFrame:
        """
        Read the table by constituency: share and seats of each candidacy in each district.

        Parameters
        ----------
        table : html.HtmlElement
            The *Distribution by constituency* table (three header rows, two columns per party: `%` and `S`).

        Returns
        -------
        pd.DataFrame
            Long format with `district`, `key` (wiki key of the party), `abbr` (header text), `pct` and `seats`
            (0 where the cell is "−"). The row "Total" and the empty cells (candidacies that did not run in
            the district) are left out.
        """
        rows = table.xpath('.//tr')
        grid = table_grid(rows)

        header = grid[0]
        parties = {}  # first column of each party -> (key, abbr)
        for i, cell in enumerate(header):
            if i == 0 or cell is None or (i > 1 and header[i - 1] is cell):
                continue
            parties[i] = (cell_key(cell), cell_text(cell))

        data = []
        for tr, line in zip(rows, grid):
            if len(tr.xpath('./td')) < 2:
                continue

            district = cell_text(line[0])
            if not district or district == 'Total':
                continue

            for i, (key, abbr) in parties.items():
                if i + 1 >= len(line):
                    continue
                pct = cell_number(cell_text(line[i]))
                if pct is None:
                    continue
                seats = cell_number(cell_text(line[i + 1]))
                data.append({
                    'district': district, 'key': key, 'abbr': abbr, 'pct': pct,
                    'seats': int(seats) if seats is not None else 0
                })

        return pd.DataFrame(data, columns=['district', 'key', 'abbr', 'pct', 'seats'])

    @staticmethod
    def split_votes(total: int, weights: pd.Series) -> pd.Series:
        """
        Split an integer total in proportion to `weights` by largest remainders, so that the parts add up
        exactly to `total`.

        Parameters
        ----------
        total : int
            Amount to split.
        weights : pd.Series
            Positive weights (e.g. population of each district).

        Returns
        -------
        pd.Series
            Integer parts, with the index of `weights`.
        """
        w = weights.astype(float)
        exact = total * w / w.sum()
        out = np.floor(exact).astype(int)
        rest = int(total - out.sum())
        if rest > 0:
            order = (exact - out).sort_values(ascending=False, kind='stable').index[:rest]
            out.loc[order] += 1

        return out
