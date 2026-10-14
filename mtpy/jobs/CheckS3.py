import json
from datetime import datetime, timezone

from ..core.services.s3 import S3
from ..core.worker import Job


class CheckS3(Job):
    """
    Smoke test of the file system (S3 bucket or local root) used by the app.

    Writes, reads, lists and removes a small file under a prefix, to detect
    early whether the pinned S3 client still works against the bucket.
    It only diagnoses; it changes nothing else.
    """

    def run(self, prefix='site/v1/_smoke', fs=None, allow_local=False, **kwargs):
        """
        Run the round trip and report each step.

        Parameters
        ----------
        prefix : str, default 'site/v1/_smoke'
            Folder where the temporary file is written.
        fs : FileSystem, optional
            File system to check. Defaults to ``self.app.fs``.
        allow_local : bool, default False
            Accept a local file system as ``self.app.fs``. Without it, the job refuses
            to run against the local disk, which would be a false all-clear for S3.
            Not needed when ``fs`` is injected.
        **kwargs
            Ignored extra job arguments.

        Returns
        -------
        dict[str, bool]
            Success flag of each step, in execution order.

        Raises
        ------
        RuntimeError
            If no file system is configured, ``self.app.fs`` is local and not allowed,
            or any step fails.
        """
        injected = fs is not None
        fs = fs if injected else self.app.fs
        if fs is None:
            raise RuntimeError(
                'check_s3: no file system configured '
                '(set S3_BUCKET or run with a local files/ root)'
            )

        if not injected and not allow_local and not isinstance(fs, S3):
            raise RuntimeError(
                'check_s3: app.fs is a local FileSystem; set S3_BUCKET or pass allow_local=true'
            )

        print('fs: s3://{}'.format(fs.bucket) if isinstance(fs, S3)
              else 'fs: local {}'.format(fs.path))

        name = prefix + '/check.json'
        payload = json.dumps({
            'checked_at': datetime.now(timezone.utc).isoformat()
        }).encode('utf-8')

        def read():
            if fs.read_bytes(name) != payload:
                raise ValueError('content read differs from content written')

        def listdir():
            if 'check.json' not in fs.listdir(prefix):
                raise ValueError('check.json not listed in ' + prefix)

        def exists():
            if not fs.exists(name):
                raise ValueError('file does not exist')

        def gone():
            if fs.exists(name):
                raise ValueError('file still exists after removal')

        steps = [
            ('write', lambda: fs.write_bytes(payload, name)),
            ('read', read),
            ('exists', exists),
            ('listdir', listdir),
            ('remove', lambda: fs.remove(name)),
            ('gone', gone)
        ]

        result = {}
        errors = {}
        for step, action in steps:
            try:
                action()
                result[step] = True
            except Exception as e:
                result[step] = False
                errors[step] = '{}: {}'.format(type(e).__name__, e)
            line = '{} OK'.format(step) if result[step] else \
                '{} FAIL: {}'.format(step, errors[step])
            print(line)
            if self.app.logger is not None:
                self.app.logger.info('check_s3 ' + line)

        failed = [step for step, ok in result.items() if not ok]
        if failed:
            raise RuntimeError('check_s3 failed: ' + ', '.join(failed))

        return result
