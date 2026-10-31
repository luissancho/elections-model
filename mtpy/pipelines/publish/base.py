"""
Base of the ``publish`` pipelines: they all write to the web bundle through a `BundleWriter`.
"""
from ...lib import bundle
from ..base import ScopePipeline


class PublishPipeline(ScopePipeline):
    """
    Scope pipeline that writes to the web bundle.

    Attributes
    ----------
    writer : BundleWriter
        Bundle writer (under the dry-run prefix in a dry run), created by `writer_for` in `start`.
    """

    def writer_for(self, fs=None, dry_run=False) -> bundle.BundleWriter:
        """
        Build the bundle writer of the run.

        Parameters
        ----------
        fs : FileSystem, optional
            File system to write to; the one of the app when not given (a pipeline launched on its own, without
            the ``Publish`` job, receives none).
        dry_run : bool, default False
            Write under the dry-run prefix (``site-dry/v1``) instead of the real one.

        Returns
        -------
        BundleWriter
            Writer over the file system.

        Raises
        ------
        RuntimeError
            If neither ``fs`` nor the app has a file system.
        """
        fs = fs if fs is not None else self.app.fs
        if fs is None:
            raise RuntimeError(
                'publish: no file system configured (set S3_BUCKET or run with a local files/ root)'
            )
        return bundle.BundleWriter(fs, prefix=bundle.DRY_PREFIX if dry_run else bundle.PREFIX)
