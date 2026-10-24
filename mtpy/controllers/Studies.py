import json

from ..core.api import HttpError
from ..lib import bundle
from ..lib.webapi import STUDY_PARTS, check_format, check_part, check_study, site
from .Base import Base


class Studies(Base):
    """
    Read-only endpoints over the published studies (``site/v1/studies/{study}``).
    """

    async def index_action(self) -> bytes:
        """
        The published studies, with the figures of their ``meta``.

        Returns
        -------
        bytes
            ``{'contract', 'studies'}``; each study is ``{study, title, subtitle, registered_at,
            checked_at}``, in the order of ``bundle.STUDIES``. Studies without a published ``meta``
            are left out.
        """
        self.notice_freeze()
        rows = []

        for study in site().studies():
            data = json.loads(site().study(study, 'meta'))['data']
            rows.append({key: data[key] for key in ('study', 'title', 'subtitle', 'registered_at', 'checked_at')})

        return self.json_bytes(bundle.dumps({'contract': bundle.CONTRACT, 'studies': rows}))

    async def part_action(self, study: str, part: str) -> bytes:
        """
        A part of a study, as JSON or, with ``?format=csv``, as one of its CSV twins.

        ``meta`` and ``placement`` have no CSV twin; the other parts have the ones of
        ``bundle.STUDY_CSV_OF``, chosen with ``&table=`` (the first by default). The files are
        overwritten on every publication, so they are never served as immutable.

        Parameters
        ----------
        study : str
            Study slug, one of ``STUDIES``.
        part : str
            Part name, one of ``STUDY_PARTS[study]``.

        Returns
        -------
        bytes
            The JSON file verbatim, or the CSV file as ``{study}-{name}.csv``.

        Raises
        ------
        HttpError
            404 ``unknown study``; 404 ``no csv for this part`` for a part without CSV or a
            ``table`` that does not belong to it; 404 ``study not published`` when the file is missing.
        """
        check_study(study)
        check_part(part, STUDY_PARTS[study])
        fmt = check_format(self.query.get('format'))
        table = self.query.get('table') or None
        self.notice_freeze()

        if fmt == 'csv':
            names = bundle.STUDY_CSV_OF[study].get(part, ())

            if not names or (table is not None and table not in names):
                raise HttpError(404, 'no csv for this part')

            name = table or names[0]

            return self.csv_bytes(site().study_csv(study, name), f'{study}-{name}.csv')

        return self.json_bytes(site().study(study, part))
