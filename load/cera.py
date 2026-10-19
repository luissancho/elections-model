"""
Download of the INE inputs of the CERA study: the monthly workbook of the current year and the tables of the
census closed for the election (`data/cera/ine/`).
"""
import os
import tempfile

from typing import Callable

import requests

from mtpy.core.io import FileSystem
from mtpy.lib.studies.cera.data import read_monthly_workbook, read_provinces

MONTHLY_URL = 'https://www.ine.es/oficina_censo/censo_cerrado/cerapropais_{yy:02d}.xlsx'
ELECTION_URLS = {
    'elecgen2026_tab{}.xlsx'.format(n): 'https://www.ine.es/oficina_censo/elecgen2026/Elecgral_2026_cc_tab{}.xlsx'.format(n)
    for n in (6, 7, 8)
}
USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/124.0.0.0 Safari/537.36'
)
INE_FOLDER = 'cera/ine'


def _download(url: str) -> bytes:
    """
    Download a file with a browser User-Agent (the INE rejects the default one).

    Parameters
    ----------
    url : str
        The URL to download.

    Returns
    -------
    bytes
        The content of the response.

    Raises
    ------
    requests.HTTPError
        If the response status is an error.
    """
    response = requests.get(url, headers={'User-Agent': USER_AGENT}, timeout=60)
    response.raise_for_status()

    return response.content


def _read_months(path: str, provinces: dict) -> set:
    """
    Read the months (ISO dates) a monthly workbook has.

    Parameters
    ----------
    path : str
        The full path to the workbook.
    provinces : dict
        The map returned by `read_provinces`.

    Returns
    -------
    set
        The ISO dates of the first day of each month in the workbook.
    """
    frame = read_monthly_workbook(path, provinces)

    return {value.strftime('%Y-%m-%d') for value in frame['date'].unique()}


def _new_months(old_path: str, new_bytes: bytes, provinces: dict) -> list:
    """
    Months the new workbook has and the previous copy did not.

    Parameters
    ----------
    old_path : str
        The full path to the previous workbook; it may not exist.
    new_bytes : bytes
        The content of the new workbook.
    provinces : dict
        The map returned by `read_provinces`.

    Returns
    -------
    list of str
        The new months as ISO dates, ascending; all of them if there is no previous copy.
    """
    old = _read_months(old_path, provinces) if os.path.exists(old_path) else set()

    with tempfile.TemporaryDirectory() as folder:
        new_path = os.path.join(folder, 'new.xlsx')
        with open(new_path, 'wb') as fh:
            fh.write(new_bytes)
        new = _read_months(new_path, provinces)

    return sorted(new - old)


def fetch_cera(data: FileSystem, year: int, fetch: Callable[[str], bytes] = _download) -> dict:
    """
    Download the INE monthly workbook of a year, and the election census tables missing in `cera/ine/`.

    The workbook is downloaded every time (it gains a month with each close); the election tables are fixed once
    the census of the convocation is closed, so only the ones without a local copy are downloaded, and a failure
    to download them cannot block the monthly refresh once they are in place. What is downloaded is downloaded
    first and written only if every download succeeded, so a failure leaves the previous files untouched.

    Parameters
    ----------
    data : FileSystem
        The file system rooted at `data/`.
    year : int
        The year of the monthly workbook.
    fetch : Callable[[str], bytes], optional
        Function returning the content of a URL; an exception it raises propagates.

    Returns
    -------
    dict
        `files`: the names written, relative to `data/`; `months_new`: the ISO dates of the months the new
        workbook has and the previous copy did not.
    """
    monthly_name = 'cerapropais_{:02d}.xlsx'.format(year % 100)
    missing = {name: url for name, url in ELECTION_URLS.items()
               if not data.exists('{}/{}'.format(INE_FOLDER, name))}
    urls = {monthly_name: MONTHLY_URL.format(yy=year % 100), **missing}
    contents = {name: fetch(url) for name, url in urls.items()}

    months_new = _new_months(
        data.get_path('{}/{}'.format(INE_FOLDER, monthly_name)), contents[monthly_name], read_provinces(data)
    )

    files = []
    for name, content in contents.items():
        path = '{}/{}'.format(INE_FOLDER, name)
        data.write_bytes(content, path)
        files.append(path)

    return {'files': files, 'months_new': months_new}
