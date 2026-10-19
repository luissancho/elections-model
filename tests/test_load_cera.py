"""La carga del INE del estudio del voto CERA (`load/cera.py`)."""
import importlib.util
import os
import shutil

import openpyxl
import pytest

from load.cera import ELECTION_URLS, MONTHLY_URL, fetch_cera
from mtpy.core.io import FileSystem
from mtpy.lib.studies.cera.data import read_monthly_workbook, read_provinces

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

INE = os.path.join(ROOT, 'data', 'cera', 'ine')


def real_bytes(name):
    """Bytes of a real file of `data/cera/ine/`."""
    with open(os.path.join(INE, name), 'rb') as fh:
        return fh.read()


def fake_fetch(url):
    """Serve the real files by URL."""
    if url == MONTHLY_URL.format(yy=26):
        return real_bytes('cerapropais_26.xlsx')
    for name, election_url in ELECTION_URLS.items():
        if url == election_url:
            return real_bytes(name)
    raise AssertionError('Unexpected URL {}'.format(url))


def make_data(tmp_path):
    """File system with the 2026 workbook trimmed to eight months and the provinces map."""
    data = FileSystem(str(tmp_path))
    os.makedirs(str(tmp_path / 'cera' / 'ine'))
    shutil.copy(os.path.join(ROOT, 'data', 'cera', 'provinces.json'), str(tmp_path / 'cera' / 'provinces.json'))
    workbook = openpyxl.load_workbook(os.path.join(INE, 'cerapropais_26.xlsx'))
    try:
        sheet = workbook.worksheets[0]
        assert sheet.cell(row=5, column=4).value.startswith('Censo cerrado a 1-')   # el último cierre, el más nuevo
        sheet.delete_cols(4)
        workbook.save(str(tmp_path / 'cera' / 'ine' / 'cerapropais_26.xlsx'))
    finally:
        workbook.close()
    return data


def test_fetch_cera_writes_the_files_and_reports_new_months(tmp_path):
    data = make_data(tmp_path)

    result = fetch_cera(data, 2026, fetch=fake_fetch)

    assert sorted(result['files']) == sorted(
        ['cera/ine/cerapropais_26.xlsx'] + ['cera/ine/' + name for name in ELECTION_URLS]
    )
    provinces = read_provinces(FileSystem(os.path.join(ROOT, 'data')))
    newest = read_monthly_workbook(os.path.join(INE, 'cerapropais_26.xlsx'), provinces)
    assert result['months_new'] == [newest['date'].max().strftime('%Y-%m-%d')]   # el mes recortado, sea cual sea
    for name in result['files']:
        assert os.path.exists(str(tmp_path / name))
    assert (tmp_path / 'cera' / 'ine' / 'cerapropais_26.xlsx').read_bytes() == real_bytes('cerapropais_26.xlsx')


def test_fetch_cera_propagates_http_errors(tmp_path):
    data = FileSystem(str(tmp_path))
    shutil.copy(os.path.join(ROOT, 'data', 'cera', 'provinces.json'), str(tmp_path / 'provinces.json'))
    os.makedirs(str(tmp_path / 'cera'))
    shutil.move(str(tmp_path / 'provinces.json'), str(tmp_path / 'cera' / 'provinces.json'))

    def failing(url):
        raise RuntimeError('HTTP 503')

    with pytest.raises(RuntimeError):
        fetch_cera(data, 2026, fetch=failing)

    assert not os.path.exists(str(tmp_path / 'cera' / 'ine')) or os.listdir(str(tmp_path / 'cera' / 'ine')) == []


def test_fetch_cera_writes_nothing_when_an_election_table_fails(tmp_path):
    data = make_data(tmp_path)
    before = (tmp_path / 'cera' / 'ine' / 'cerapropais_26.xlsx').read_bytes()

    def partial(url):
        if url.endswith('tab8.xlsx'):
            raise RuntimeError('HTTP 404')
        return fake_fetch(url)

    with pytest.raises(RuntimeError):
        fetch_cera(data, 2026, fetch=partial)

    assert (tmp_path / 'cera' / 'ine' / 'cerapropais_26.xlsx').read_bytes() == before
    assert sorted(os.listdir(str(tmp_path / 'cera' / 'ine'))) == ['cerapropais_26.xlsx']


def test_run_load_accepts_the_cera_step():
    spec = importlib.util.spec_from_file_location('run_load', os.path.join(ROOT, 'load', 'run_load.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert 'cera' in module.STEPS
