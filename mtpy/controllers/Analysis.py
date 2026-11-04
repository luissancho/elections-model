from ..core.api import HttpError
from ..lib.webapi import ANALYSIS_PARTS, check_format, check_part, check_scope, site
from .Base import Base

# the `table` of the second CSV twin of each part that has two
CSV_TABLES = {'herding': 'events', 'rating-check': 'elections'}


class Analysis(Base):
    """
    Read-only endpoints over the published houses analysis (``site/v1/analysis/{scope}``).
    """

    async def part_action(self, scope: str, part: str) -> bytes:
        """
        A part of the analysis of a scope, as JSON or, with ``?format=csv``, as its CSV twin.

        ``meta`` has no CSV twin; ``herding`` has two: ``herding`` and, with ``&table=events``,
        ``herding-events``; ``rating-check`` has two: ``rating-check`` (the pollster-election pairs) and, with
        ``&table=elections``, ``rating-check-elections`` (the simple average of each election). The
        analysis is overwritten on every publication, so it is never served as immutable.

        Parameters
        ----------
        scope : str
            Scope code.
        part : str
            Part name, one of ``ANALYSIS_PARTS``.

        Returns
        -------
        bytes
            The JSON file verbatim, or the CSV file as ``{scope}-{name}.csv``.

        Raises
        ------
        HttpError
            404 ``no csv for this part`` for ``meta`` or a ``table`` that is not one of the part
            (``events`` of ``herding``, ``elections`` of ``rating-check``); 404 ``analysis not
            published`` when the file is missing.
        """
        check_scope(scope)
        check_part(part, ANALYSIS_PARTS)
        fmt = check_format(self.query.get('format'))
        table = self.query.get('table') or None
        self.notice_freeze()

        if fmt == 'csv':
            if part == 'meta' or (table is not None and CSV_TABLES.get(part) != table):
                raise HttpError(404, 'no csv for this part')

            name = part if table is None else part + '-' + table

            return self.csv_bytes(site().analysis_csv(scope, name), f'{scope}-{name}.csv')

        return self.json_bytes(site().analysis(scope, part))
