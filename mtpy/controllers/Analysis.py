from ..core.api import HttpError
from ..lib.webapi import ANALYSIS_PARTS, check_format, check_part, check_scope, site
from .Base import Base


class Analysis(Base):
    """
    Read-only endpoints over the published houses analysis (``site/v1/analysis/{scope}``).
    """

    async def part_action(self, scope: str, part: str) -> bytes:
        """
        A part of the analysis of a scope, as JSON or, with ``?format=csv``, as its CSV twin.

        ``meta`` has no CSV twin; ``herding`` has two: ``herding`` and, with ``&table=events``,
        ``herding-events``. The analysis is overwritten on every publication, so it is never
        served as immutable.

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
            404 ``no csv for this part`` for ``meta`` or an unknown ``table``; 404 ``analysis
            not published`` when the file is missing.
        """
        check_scope(scope)
        check_part(part, ANALYSIS_PARTS)
        fmt = check_format(self.query.get('format'))
        table = self.query.get('table') or None
        self.notice_freeze()

        if fmt == 'csv':
            if part == 'meta' or table not in (None, 'events') or (table == 'events' and part != 'herding'):
                raise HttpError(404, 'no csv for this part')

            name = part if table is None else part + '-' + table

            return self.csv_bytes(site().analysis_csv(scope, name), f'{scope}-{name}.csv')

        return self.json_bytes(site().analysis(scope, part))
