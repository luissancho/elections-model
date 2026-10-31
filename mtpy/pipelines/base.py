"""
Base of the pipelines that work scope by scope (``load``, ``compute``, ``backtest``, ``publish``).

`ScopePipeline` gives a `Pipeline` the live progress lines, the verbosity levels and the failure handling of
`mtpy.jobs.progress.ScopeReporter`: each chunk is a scope, its outcome is a result dict (a failure is a result with
``status`` ``failed``, not an exception) and its summary line is built by `line_of`.
"""
from ..core.worker import Pipeline, PipelineEnd
from ..jobs.progress import ScopeProgress, ScopeReporter, collect_warnings


class ScopePipeline(ScopeReporter, Pipeline):
    """
    Pipeline that works scope by scope: live progress, collected warnings and failures as results.

    Attributes
    ----------
    steps : dict, optional
        Labels of the phases reported at verbose 2 (see `ScopeProgress`).
    verbose : int
        Verbosity of the run (0 to 3), set by `start`.
    model_verbose : int
        Verbosity passed to the units of work (``verbose - 2``, never negative).
    total : int
        Number of chunks of the run.
    progress : ScopeProgress or None
        Progress printer of the current chunk, created by the first task that needs it.
    """

    steps = None

    def start(self, verbose=1, position=(1, 1), **kwargs) -> None:
        """
        Set the verbosity and the size of the run, on the first run.

        Parameters
        ----------
        verbose : int, default 1
            Verbosity of the console output.
        position : tuple of int, default (1, 1)
            ``(index, total)`` of the first chunk; ``total`` is the number of chunks of the run.
        **kwargs
            Other arguments of the run, ignored here.
        """
        self.verbose = verbose
        self.model_verbose = max(0, verbose - 2)
        self.total = position[1]
        self.progress = None

    def progress_of(self, chunk, position=(1, 1), day=None) -> ScopeProgress:
        """
        Build the progress printer of one chunk.

        Parameters
        ----------
        chunk : str
            Scope being worked.
        position : tuple of int, default (1, 1)
            ``(index, total)`` of the chunk.
        day : str, optional
            Day of a retrospective run, added to the prefix.

        Returns
        -------
        ScopeProgress
            Printer with the phase labels of the pipeline.
        """
        return self._progress(chunk, position[0], position[1], day, self.steps)

    def start_progress(self, chunk, position=(1, 1), day=None) -> ScopeProgress:
        """
        Create the progress printer of one chunk, keep it in ``self.progress`` and report that the chunk starts.

        Parameters
        ----------
        chunk : str
            Scope being worked.
        position : tuple of int, default (1, 1)
            ``(index, total)`` of the chunk.
        day : str, optional
            Day of a retrospective run, added to the prefix.

        Returns
        -------
        ScopeProgress
            The started progress.
        """
        self.progress = self.progress_of(chunk, position, day)
        self.progress.start()
        return self.progress

    def collect(self, chunk, work, **extra) -> dict:
        """
        Run the unit of work of one chunk, collecting its warnings and turning an exception into a failed result.

        A `PipelineEnd` raised by the work is not a failure and goes through untouched; a pipeline without
        progress object (``self.progress`` is ``None``) collects the warnings without printing them.

        Parameters
        ----------
        chunk : str
            Scope being worked.
        work : callable
            Zero-argument callable returning the result dict.
        **extra
            Further keys of the result when the work fails (see `failed`).

        Returns
        -------
        dict
            The result of ``work`` or the failed result, with the ``warnings`` raised.
        """
        notes = []
        try:
            with collect_warnings(notes, self.progress.note if self.progress is not None else None):
                result = work()
        except PipelineEnd:
            raise
        except Exception as e:
            result = self.failed(chunk, e, **extra)
        result['warnings'] = notes
        return result

    def failed(self, scope, e: Exception, **extra) -> dict:
        """
        Log the exception being handled and turn it into a failed result.

        Parameters
        ----------
        scope : str
            Scope that failed.
        e : Exception
            The exception.
        **extra
            Further keys of the result.

        Returns
        -------
        dict
            ``{'status': 'failed', 'reason': 'Type: message', **extra}``.
        """
        self._log_error(scope)
        return {'status': 'failed', 'reason': '{}: {}'.format(type(e).__name__, e), **extra}

    def line_of(self, chunk, result, **kwargs):
        """
        Build the summary line of one result (abstract).

        Parameters
        ----------
        chunk : str
            Scope of the result.
        result : dict
            Result of the run.

        Returns
        -------
        str or list of str
            The summary line(s).
        """
        raise NotImplementedError

    def summary(self, chunk, **kwargs):
        """
        Build the summary line of the run and report it in the progress of the chunk, if there is one.

        Pipelines that print several lines per chunk override this method.
        """
        line = self.line_of(chunk, self.result, **kwargs)
        if self.progress is not None and isinstance(line, str):
            self.progress.done(line)
        return line
