import glob
import gzip
import io
import joblib
import os
import pandas as pd
from s3fs import S3FileSystem
import tempfile

from typing import Any, Literal, Optional
from typing_extensions import Self

from ..app import App
from ..io import FileSystem


class S3(FileSystem):

    def __init__(self, aws_params, bucket):
        super().__init__(path='/')

        self._client = None

        self.aws_params = aws_params
        self.bucket = bucket

    def is_connected(self) -> bool:
        return self._client is not None

    def connect(self, force: bool = False) -> Self:
        if self.is_connected():
            if force:
                self.disconnect()
            else:
                return self
        
        self.connection()
        
        return self

    def disconnect(self) -> Self:
        if self.is_connected():
            self._client = None

        return self

    def connection(self) -> Self:
        self._client = S3FileSystem(
            key=self.aws_params['key'],
            secret=self.aws_params['secret']
        )

    def get_path(self, name: str, force: bool = False) -> str:
        path = '{}/{}'.format(self.bucket, name)

        if force:
            dirname = os.path.dirname(path)
            if not self._client.isdir(dirname):
                self._client.makedirs(dirname)

        return path

    def isdir(self, name: str) -> bool:
        self.connect()

        path = self.get_path(name)

        return self._client.isdir(path)

    def isfile(self, name: str) -> bool:
        self.connect()

        path = self.get_path(name)

        return self._client.isfile(path)

    def exists(self, name: str) -> bool:
        self.connect()

        path = self.get_path(name)

        return self._client.exists(path)

    def copy(self, name: str, target: str) -> None:
        self.connect()

        src_path = self.get_path(name)
        tgt_path = self.get_path(target, force=True)

        self._client.copy(src_path, tgt_path)

        return tgt_path

    def move(self, name: str, target: str) -> None:
        self.connect()

        src_path = self.get_path(name)
        tgt_path = self.get_path(target, force=True)

        self._client.move(src_path, tgt_path)

        return tgt_path

    def remove(self, name: str) -> None:
        self.connect()

        path = self.get_path(name)

        self._client.rm(path, recursive=True)

    def listdir(self, name: Optional[str] = None) -> list[str]:
        self.connect()

        path = self.get_path(name)

        return [
            i.replace(path + '/', '')
            for i in self._client.ls(path)
            if i.rstrip('/') != path
        ]
    
    def makedir(self, name: str) -> str:
        self.connect()

        path = self.get_path(name)
        self._client.makedirs(path, exist_ok=True)

        return path
    
    def glob(self, pattern: str) -> list[str]:
        self.connect()

        path = self.get_path(pattern)

        return [
            i.replace(self.bucket + '/', '')
            for i in sorted(self._client.glob(path))
        ]

    @staticmethod
    def assure_remote_file(name: str, force: bool = False) -> str:
        """
        Assure the existence of the given file in the remote file system.

        Parameters
        ----------
        name : str
            The name of the file to check.
        force : bool, optional
            Whether to force the copy of the file.

        Returns
        -------
        str
            The full path to the file.
        """
        app = App.get_()
        fs_rem = app.fs

        if not fs_rem.exists(name) or force:
            fs_loc = FileSystem(app.fspath)
            if fs_loc.exists(name):
                fs_rem.write_bytes(fs_loc.read_bytes(name), name)
            else:
                return

        return fs_rem.get_path(name)

    def read(
        self,
        name: str,
        format: Optional[Literal['csv', 'json', 'excel']] = None,
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> pd.DataFrame:
        if self.is_url(name):
            return self.read_url(
                name=name,
                format=format,
                compression=compression,
                **kwargs
            )

        self.connect()

        path = self.get_path(name)
        encoding = kwargs.get('encoding')
        mode = 'rb' if compression == 'gzip' or format == 'excel' else 'r'

        with self._client.open(path, mode, encoding=encoding) as fh:
            if compression == 'gzip':
                with gzip.GzipFile(mode='rb', fileobj=fh) as gz:
                    buffer = io.StringIO(gz.read().decode('utf-8'))
            elif format == 'excel':
                buffer = io.BytesIO(fh.read())
            else:
                buffer = io.StringIO(fh.read())

        if format == 'csv':
            df = pd.read_csv(buffer, **kwargs)
        elif format == 'json':
            df = pd.read_json(buffer, **kwargs)
        elif format == 'excel':
            df = pd.read_excel(buffer, **kwargs)
        else:
            df = buffer.getvalue()

        return df

    def write(
        self,
        df: pd.DataFrame,
        name: str,
        format: Optional[Literal['csv', 'json', 'excel']] = None,
        compression: Optional[Literal['gzip']] = None,
        **kwargs
    ) -> str:
        self.connect()

        path = self.get_path(name, force=True)

        if format == 'csv':
            buffer = io.StringIO()
            df.to_csv(buffer, **kwargs)
        elif format == 'json':
            buffer = io.StringIO()
            df.to_json(buffer, **kwargs)
        elif format == 'excel':
            buffer = io.BytesIO()
            df.to_excel(buffer, **kwargs)
        else:
            buffer = io.StringIO(df)

        if compression == 'gzip':
            gz_buffer = io.BytesIO()
            with gzip.GzipFile(mode='wb', fileobj=gz_buffer) as gz:
                if isinstance(buffer, io.StringIO):
                    gz.write(bytes(buffer.getvalue(), 'utf-8'))
                else:
                    gz.write(buffer.getvalue())
            buffer = gz_buffer

        mode = 'wb' if isinstance(buffer, io.BytesIO) else 'w'

        with self._client.open(path, mode) as fh:
            fh.write(buffer.getvalue())

        return path

    def read_bytes(self, name: str) -> bytes:
        self.connect()

        path = self.get_path(name)

        with self._client.open(path, 'rb') as fh:
            content = fh.read()

        return content

    def write_bytes(self, content: bytes, name: str) -> str:
        self.connect()

        path = self.get_path(name, force=True)

        with self._client.open(path, 'wb') as fh:
            fh.write(content)

        return path

    def save_object(self, value: Any, name: str) -> str:
        with tempfile.TemporaryFile() as tf:
            joblib.dump(value, tf)
            tf.seek(0)
            path = self.write_bytes(tf.read(), name)

        return path

    def load_object(self, name: str) -> Any:
        with tempfile.TemporaryFile() as tf:
            tf.write(self.read_bytes(name))
            tf.seek(0)
            value = joblib.load(tf)

        return value
