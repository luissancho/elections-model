import pandas as pd

from ..core.api import HttpError
from ..lib.webapi import EVENT_PARTS, check_date, check_format, check_part, check_scope, query_events, site
from .Data import Data

MODEL_TABLES = ('shares', 'seats', 'blocks')


class Events(Data):
    """
    Read-only endpoints over the past elections: the list (database crossed with the bundle) and
    the published files of each one (``site/v1/events/{scope}/{date}``).
    """

    async def index_action(self, scope: str) -> bytes:
        """
        Elections of a scope, with whether each one has a published page.

        Parameters
        ----------
        scope : str
            Scope code.

        Returns
        -------
        bytes
            ``{'contract', 'scope', 'rows'}`` with ``date`` (ISO), ``name``, ``featured`` and
            ``has_page``, ascending by date, or ``events-{scope}.csv``.
        """
        check_scope(scope)
        check_format(self.query.get('format'))
        events = query_events(scope)
        published = site().events(scope)
        frame = pd.DataFrame({
            'date': pd.to_datetime(events['date']).dt.strftime('%Y-%m-%d'),
            'name': events['name'],
            'featured': events['featured'].astype(bool),
        }).sort_values('date').reset_index(drop=True)
        frame['has_page'] = frame['date'].isin(published)

        return self.serve(frame, 'events-' + scope, scope=scope)

    async def part_action(self, scope: str, date: str, part: str) -> bytes:
        """
        A part of a past election, as JSON or, with ``?format=csv``, as its CSV twin.

        ``meta`` has no CSV twin; ``model`` has three, chosen with ``&table=`` (``shares``,
        ``seats`` or ``blocks``); the other parts take no ``table``. The files are overwritten
        on every publication, so they are never served as immutable.

        Parameters
        ----------
        scope : str
            Scope code.
        date : str
            Election date.
        part : str
            Part name, one of ``EVENT_PARTS``.

        Returns
        -------
        bytes
            The JSON file verbatim, or the CSV file as ``{scope}-{date}-{name}.csv``.

        Raises
        ------
        HttpError
            404 ``event not published`` when the election has no published page; 404 ``no csv
            for this part`` for ``meta``, a missing or unknown ``table`` of ``model`` or a
            ``table`` in any other part.
        """
        check_scope(scope)
        check_date(date)
        check_part(part, EVENT_PARTS)
        fmt = check_format(self.query.get('format'))
        table = self.query.get('table') or None
        self.notice_freeze()

        if date not in site().events(scope):
            raise HttpError(404, 'event not published')

        if fmt == 'csv':
            has_twin = part != 'meta' and (table in MODEL_TABLES if part == 'model' else table is None)

            if not has_twin:
                raise HttpError(404, 'no csv for this part')

            name = part if table is None else f'{part}-{table}'

            return self.csv_bytes(site().event_csv(scope, date, name), f'{scope}-{date}-{name}.csv')

        return self.json_bytes(site().event(scope, date, part))
