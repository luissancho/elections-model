from ..core.api import HttpError
from ..lib.webapi import BACKTEST_PARTS, check_format, check_part, check_scope, site
from .Base import Base


class Backtest(Base):
    """
    Read-only endpoints over the published model backtest (``site/v1/backtest/{scope}``).
    """

    async def part_action(self, scope: str, part: str) -> bytes:
        """
        A part of the backtest of a scope, as JSON or, with ``?format=csv``, as its CSV twin.

        ``meta`` has no CSV twin. The backtest is overwritten on every publication, so it is
        never served as immutable.

        Parameters
        ----------
        scope : str
            Scope code.
        part : str
            Part name, one of ``BACKTEST_PARTS``.

        Returns
        -------
        bytes
            The JSON file verbatim, or the CSV file as ``{scope}-backtest-{name}.csv``.

        Raises
        ------
        HttpError
            404 ``no csv for this part`` for ``meta``; 404 ``backtest not published`` when the
            file is missing.
        """
        check_scope(scope)
        check_part(part, BACKTEST_PARTS)
        fmt = check_format(self.query.get('format'))
        self.notice_freeze()

        if fmt == 'csv':
            if part == 'meta':
                raise HttpError(404, 'no csv for this part')

            return self.csv_bytes(site().backtest_csv(scope, part), f'{scope}-backtest-{part}.csv')

        return self.json_bytes(site().backtest(scope, part))
