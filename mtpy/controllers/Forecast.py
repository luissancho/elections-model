from ..core.api import HttpError
from ..lib import bundle
from ..lib.webapi import (
    CSV_PARTS, MODE_PARTS, RUN_PARTS, check_format, check_mode, check_part, check_run,
    check_scope, site
)
from .Base import Base


class Forecast(Base):
    """
    Read-only endpoints over the published forecast bundle.
    """

    async def health_action(self) -> dict:
        """
        Liveness probe; touches neither the file system nor the database.

        Returns
        -------
        dict
            ``{'status': 'ok', 'contract': int}``.
        """
        self.no_store()

        return {'status': 'ok', 'contract': bundle.CONTRACT}

    async def manifest_action(self) -> bytes:
        """
        Manifest of the published bundle, verbatim.

        Returns
        -------
        bytes
            The manifest JSON.
        """
        return self.json_bytes(site().manifest())

    async def scopes_action(self) -> bytes:
        """
        Scopes catalogue crossed with the manifest.

        Returns
        -------
        bytes
            Catalogue rows (see ``Site.scopes``) serialised as JSON.
        """
        self.notice_freeze()

        return self.json_bytes(bundle.dumps(site().scopes()))

    async def meta_action(self, scope: str):
        """
        Metadata of a run; shortcut of the ``meta`` part.

        Parameters
        ----------
        scope : str
            Scope code.

        Returns
        -------
        bytes
            The ``meta`` JSON.
        """
        return await self.part_action(scope, 'meta')

    async def runs_action(self, scope: str) -> bytes:
        """
        History of the published runs of a scope.

        Parameters
        ----------
        scope : str
            Scope code.

        Returns
        -------
        bytes
            The history JSON.
        """
        check_scope(scope)
        self.notice_freeze()

        return self.json_bytes(site().history(scope))

    def resolve_run(self, scope: str) -> tuple:
        """
        Run requested with ``?run=`` or the latest one of the scope.

        Parameters
        ----------
        scope : str
            Scope code.

        Returns
        -------
        tuple of (str, bool)
            Run id and whether it was requested explicitly.
        """
        run = check_run(self.query.get('run'))

        if run:
            return run, True

        return site().latest_run(scope), False

    async def part_action(self, scope: str, part: str) -> bytes:
        """
        A part of a run, as JSON or, with ``?format=csv``, as its CSV twin.

        Parameters
        ----------
        scope : str
            Scope code.
        part : str
            Part name, one of ``RUN_PARTS``.

        Returns
        -------
        bytes
            The JSON or CSV file.
        """
        check_scope(scope)
        check_part(part, RUN_PARTS)
        fmt = check_format(self.query.get('format'))
        run, explicit = self.resolve_run(scope)
        self.notice_freeze()

        if fmt == 'csv':
            if part not in CSV_PARTS:
                raise HttpError(404, 'no csv for this part')

            return self.csv_bytes(site().run_csv(scope, run, part), f'{scope}-{run}-{part}.csv', explicit)

        return self.json_bytes(site().run_file(scope, run, part), explicit)

    async def mode_part_action(self, scope: str, mode: str, part: str) -> bytes:
        """
        A part of a simulation mode of a run, as JSON or CSV.

        Parameters
        ----------
        scope : str
            Scope code.
        mode : str
            Simulation mode.
        part : str
            Part name, one of ``MODE_PARTS``.

        Returns
        -------
        bytes
            The JSON or CSV file.
        """
        check_scope(scope)
        check_mode(mode)
        check_part(part, MODE_PARTS)
        fmt = check_format(self.query.get('format'))
        run, explicit = self.resolve_run(scope)
        self.notice_freeze()

        if fmt == 'csv':
            return self.csv_bytes(
                site().run_csv(scope, run, part, mode), f'{scope}-{run}-{mode}-{part}.csv', explicit
            )

        return self.json_bytes(site().run_file(scope, run, part, mode), explicit)
