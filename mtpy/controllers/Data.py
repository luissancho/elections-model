from typing import Optional

import pandas as pd

from ..core.api import HttpError
from ..lib import bundle
from ..lib.webapi import (
    check_date, check_format, check_limit, check_pollster, check_region, check_scope, frame_to_csv,
    frame_to_records, polls_event, query_drift, query_herding, query_house_effects, query_parties,
    query_pollster, query_pollsters, query_polls, query_ratings, query_results, ratings_event
)
from .Base import Base


class Data(Base):
    """
    Read-only endpoints over the live database tables (catalogues, ratings, polls, results,
    house effects, herding and drift).

    Tables are served as ``{'contract': 1, <validated request fields>, 'rows': [...]}`` or, with
    ``?format=csv``, as a CSV download. Every input passes a closed validator before reaching
    the readers, which build SQL from strings.
    """

    def scope(self) -> str:
        """
        Scope of the request, ``es`` by default.

        Returns
        -------
        str
            Validated scope code.
        """
        return check_scope(self.query.get('scope') or 'es')

    def optional(self, check, name: str) -> Optional[object]:
        """
        Validate an optional query parameter.

        Parameters
        ----------
        check : callable
            Validator of the parameter.
        name : str
            Parameter name.

        Returns
        -------
        object or None
            The validated value, or ``None`` when the parameter is missing or empty.
        """
        value = self.query.get(name)

        if value is None or value == '':
            return None

        return check(value)

    def serve(self, frame: pd.DataFrame, filename: str, **fields) -> bytes:
        """
        Serve a table as JSON records or, with ``?format=csv``, as a CSV download.

        Parameters
        ----------
        frame : pandas.DataFrame
            The table.
        filename : str
            Name of the CSV file, without extension.
        **fields
            Validated request fields echoed in the JSON before ``rows``.

        Returns
        -------
        bytes
            The JSON document or the CSV file.
        """
        self.notice_freeze()

        if check_format(self.query.get('format')) == 'csv':
            return self.csv_bytes(frame_to_csv(frame), filename + '.csv')

        content = {'contract': bundle.CONTRACT, **fields, 'rows': frame_to_records(frame)}

        return self.json_bytes(bundle.dumps(content))

    @staticmethod
    def filename(*parts) -> str:
        """
        Name of a CSV download from its non-empty parts.

        Parameters
        ----------
        *parts
            Route name and the validated request values that select the table.

        Returns
        -------
        str
            The parts joined with ``-``, skipping ``None``.
        """
        return '-'.join(str(part) for part in parts if part is not None)

    async def parties_action(self) -> bytes:
        """
        Parties catalogue.

        Returns
        -------
        bytes
            ``{'contract', 'rows'}`` or ``parties.csv``.
        """
        check_format(self.query.get('format'))

        return self.serve(query_parties(), 'parties')

    async def pollsters_action(self) -> bytes:
        """
        Pollsters catalogue.

        Returns
        -------
        bytes
            ``{'contract', 'rows'}`` or ``pollsters.csv``.
        """
        check_format(self.query.get('format'))

        return self.serve(query_pollsters(), 'pollsters')

    async def pollster_action(self, id: str) -> bytes:
        """
        Profile of a pollster in a scope (``?scope=``, ``es`` by default).

        Parameters
        ----------
        id : str
            Pollster id from the path.

        Returns
        -------
        bytes
            ``{'contract', 'scope', 'pollster', 'ratings', 'house_effects', 'herding',
            'quality'}`` (see ``webapi.query_pollster``).

        Raises
        ------
        HttpError
            404 ``no csv for this part`` with ``?format=csv``: the profile has no CSV twin.
        """
        pollster_id = check_pollster(id)
        scope = self.scope()

        if check_format(self.query.get('format')) == 'csv':
            raise HttpError(404, 'no csv for this part')

        profile = query_pollster(pollster_id, scope)
        self.notice_freeze()

        return self.json_bytes(bundle.dumps({'contract': bundle.CONTRACT, 'scope': scope, **profile}))

    async def ratings_action(self) -> bytes:
        """
        Pollster ratings of an event (``?event=``, the next election by default).

        Returns
        -------
        bytes
            ``{'contract', 'scope', 'event', 'rows'}`` (``event`` is the resolved date, see
            ``webapi.ratings_event``) or ``ratings-{scope}[-{event}].csv``.
        """
        scope = self.scope()
        event = self.optional(check_date, 'event')
        check_format(self.query.get('format'))
        frame = query_ratings(scope, event)

        return self.serve(frame, self.filename('ratings', scope, event), scope=scope, event=ratings_event(scope, event))

    async def polls_action(self) -> bytes:
        """
        Polls of an event (``?event=``, ``?pollster=``, ``?limit=``), most recent first.

        Returns
        -------
        bytes
            ``{'contract', 'scope', 'event', 'pollster', 'limit', 'rows'}`` (``event`` is the
            resolved date, see ``webapi.polls_event``) or ``polls-{scope}[-{event}][-{pollster}].csv``.
        """
        scope = self.scope()
        event = self.optional(check_date, 'event')
        pollster = self.optional(check_pollster, 'pollster')
        limit = check_limit(self.query.get('limit'))
        check_format(self.query.get('format'))

        frame = query_polls(scope, event, pollster, limit)

        return self.serve(
            frame, self.filename('polls', scope, event, pollster),
            scope=scope, event=polls_event(scope, event), pollster=pollster, limit=limit
        )

    async def results_action(self, scope: str, date: str) -> bytes:
        """
        Results of an election in a region (``?region=``, 0 for the whole scope by default).

        Parameters
        ----------
        scope : str
            Scope code from the path.
        date : str
            Election date from the path.

        Returns
        -------
        bytes
            ``{'contract', 'scope', 'date', 'region', 'rows'}`` or
            ``results-{scope}-{date}[-{region}].csv``.
        """
        check_scope(scope)
        check_date(date)
        region = check_region(self.query.get('region'))
        check_format(self.query.get('format'))
        frame = query_results(scope, date, region or 0)
        filename = self.filename('results', scope, date, region)

        return self.serve(frame, filename, scope=scope, date=date, region=region or 0)

    async def house_effects_action(self) -> bytes:
        """
        House effects table of a scope.

        Returns
        -------
        bytes
            ``{'contract', 'scope', 'rows'}`` or ``house-effects-{scope}.csv``.
        """
        scope = self.scope()
        check_format(self.query.get('format'))

        return self.serve(query_house_effects(scope), self.filename('house-effects', scope), scope=scope)

    async def herding_action(self) -> bytes:
        """
        Herding table of a scope.

        Returns
        -------
        bytes
            ``{'contract', 'scope', 'rows'}`` or ``herding-{scope}.csv``.
        """
        scope = self.scope()
        check_format(self.query.get('format'))

        return self.serve(query_herding(scope), self.filename('herding', scope), scope=scope)

    async def drift_action(self) -> bytes:
        """
        Drift table of a scope.

        Returns
        -------
        bytes
            ``{'contract', 'scope', 'rows'}`` or ``drift-{scope}.csv``.
        """
        scope = self.scope()
        check_format(self.query.get('format'))

        return self.serve(query_drift(scope), self.filename('drift', scope), scope=scope)
