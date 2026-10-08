"""Tests del job `check_s3` (`mtpy/jobs/CheckS3.py`) sobre un sistema de ficheros local."""
import pytest

from mtpy.core.io import FileSystem


def test_check_s3_round_trip_on_local_fs(fresh_app, tmp_path, capsys):
    from mtpy.jobs.CheckS3 import CheckS3

    result = CheckS3().run(fs=FileSystem(str(tmp_path)))
    assert result == {'write': True, 'read': True, 'exists': True, 'listdir': True, 'remove': True, 'gone': True}
    assert 'write OK' in capsys.readouterr().out
    assert not (tmp_path / 'site' / 'v1' / '_smoke' / 'check.json').exists()


def test_check_s3_raises_when_a_step_fails(fresh_app, tmp_path):
    from mtpy.jobs.CheckS3 import CheckS3

    class Broken(FileSystem):
        def read_bytes(self, name):
            raise IOError('boom')

    with pytest.raises(RuntimeError, match='read'):
        CheckS3().run(fs=Broken(str(tmp_path)))


def test_check_s3_rejects_local_app_fs_unless_allowed(fresh_app, tmp_path):
    """Con `app.fs` local el job se niega, salvo que se pase `allow_local`."""
    from mtpy.jobs.CheckS3 import CheckS3

    fresh_app.set('fs', FileSystem(str(tmp_path)))
    with pytest.raises(RuntimeError, match='local'):
        CheckS3().run()

    result = CheckS3().run(allow_local=True)
    assert all(result.values()) and len(result) == 6


def test_check_s3_requires_a_file_system(fresh_app):
    """Sin sistema de ficheros configurado el job falla."""
    from mtpy.jobs.CheckS3 import CheckS3

    with pytest.raises(RuntimeError, match='no file system'):
        CheckS3().run()
