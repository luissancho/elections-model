from datetime import datetime
from importlib import import_module
import json
import pandas as pd
import signal
import time
from types import ModuleType

from typing import NoReturn, Optional
from typing_extensions import Self

from .app import Core
from .data import Model

from .utils.dates import strfdelta
from .utils.strings import to_camel
from .utils.helpers import is_empty


class JobInterrupted(BaseException):
    """Stop signal of a job turned into an exception (not an `Exception`, so no chunk handler swallows it)."""


class Worker(Core):
    """
    Worker class to manage jobs execution. Listens to a queue and dispatches jobs.

    Parameters
    ----------
    queue : str
        Queue name to listen to.
    """

    def __init__(self, queue: Optional[str] = None):
        super().__init__()

        self.handler = self.app.qs
        self.namespace = import_module('...jobs', package=__name__)

        self.queue = None
        self._listen = False

        self.set_queue(queue if queue is not None else self.app.config.app.queue)

    def _register_signals(self):
        for SIG in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT, signal.SIGHUP]:
            signal.signal(SIG, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f'Signal {signum} received...')
        self._listen = False

    def set_namespace(self, namespace: str) -> Self:
        if isinstance(namespace, str):
            self.namespace = import_module(namespace)
        else:
            self.namespace = namespace

        return self

    def set_queue(self, queue: str) -> Self:
        if not is_empty(queue):
            self.queue = self.handler.get_queue(queue)
        else:
            self.queue = None
        
        self._listen = self.queue is not None

        return self

    def listen(self):
        """
        Listen to the queue and dispatch requested jobs.
        """
        self._register_signals()

        self.app.logger.debug('Waiting for new messages...')

        while self._listen:
            messages = self.handler.read_messages(self.queue)

            if len(messages) > 0:
                self.app.logger.info(f'Processing {len(messages)} messages...')

                for msg in messages:
                    self.handle_message(msg)

                self.app.logger.debug('Waiting for new messages...')
            else:
                time.sleep(self.handler.interval)

    def job_class(self, job: str) -> type:
        """
        Resolve a job name into its class in the namespace.

        A job module imported with ``from .Name import Name`` is bound on the package as the
        submodule, which hides the lazy class lookup of the package; in that case the class
        is taken from the submodule (same rule as ``Router.controller_class``).

        Parameters
        ----------
        job : str
            Job name of the message, in snake case (``'publish'``).

        Returns
        -------
        type
            The job class.
        """
        name = to_camel(job)
        attribute = getattr(self.namespace, name)

        if isinstance(attribute, ModuleType):
            attribute = getattr(attribute, name)

        return attribute

    def handle_message(self, msg: str):
        """
        Handle a message received from the queue.

        Parameters
        ----------
        msg : str
            Message to be processed.
        """
        data = json.loads(msg)

        job = self.job_class(data['job'])()
        params = data['params'] if 'params' in data and isinstance(data['params'], dict) else {}

        try:
            job.dispatch(**params)
        except Exception as e:
            self.log('error', 'job {} failed: {}'.format(job.name, e))


class Job(Core):
    """
    Job base class for job execution.
    """

    ALERT_LIMIT = 1024
    TRUNCATED = '… (resumen recortado; el texto completo está en el log)'

    def __init__(self):
        if type(self) is Job:
            raise TypeError('Abstract class should not be instantiated.')

        super().__init__()

        self.namespace = import_module('...pipelines', package=__name__)
        self.name = self.__class__.__name__

        self.params = {}
        self.alerted = False

    def dispatch(self, **kwargs):
        """
        Dispatch the job with the selected parameters.

        If `run` raises anything and the job has not alerted yet, an alert naming the failure
        (or the interruption) is sent before the exception propagates. `after_dispatch` only
        runs when `run` finishes normally.
        """
        self.params = dict(kwargs)
        self.alerted = False

        self.before_dispatch()

        try:
            self.run(**self.params)
        except BaseException as e:
            if not self.alerted:
                self.log('error', '{} job failed'.format(self.name))
                if isinstance(e, (JobInterrupted, KeyboardInterrupt)):
                    self.alert('{} job interrupted: {}'.format(self.name, e))
                else:
                    self.alert('{} job failed: {}: {}'.format(self.name, type(e).__name__, e))
            raise

        self.after_dispatch()

    def run(self, **kwargs):
        """
        Run the job with the selected parameters.
        Will be defined in each subclass.
        """
        pass

    def before_dispatch(self):
        """
        Actions to be executed before the job is dispatched.
        Will be defined in each subclass.
        """
        pass

    def after_dispatch(self):
        """
        Actions to be executed after the job is dispatched.
        Will be defined in each subclass.
        """
        pass

    def alert(self, msg: str) -> None:
        """
        Send an alert message to the configured push service and to the logging system.

        The whole text goes to the log. The push message is cut to `ALERT_LIMIT` characters,
        and a failing push service is logged, never raised. Marks the job as `alerted`.

        Parameters
        ----------
        msg : str
            Alert text.
        """
        self.log('info', msg)

        ps = self.app.ps  # Pushover Service
        if ps:
            text = msg
            if len(msg) > self.ALERT_LIMIT:
                text = msg[:self.ALERT_LIMIT - len(self.TRUNCATED) - 1].rstrip() + '\n' + self.TRUNCATED
            try:
                ps.send(text)
            except Exception as e:
                self.log('error', 'alert: {}: {}'.format(type(e).__name__, e))

        self.alerted = True


class SequenceJob(Job):
    """
    SequenceJob base class to manage a sequence of pipelines execution.

    Attributes
    ----------
    sequence : dict
        Dictionary with the sequence of pipelines to be executed.
    default_action : str
        Default action to be executed.
    """

    sequence = {}
    default_action = 'update'

    def __init__(self):
        if type(self) is Job:
            raise TypeError('Abstract class should not be instantiated.')

        super().__init__()

    def run(
        self,
        select: Optional[str] = None,
        action: Optional[str] = None,
        chunks: Optional[str] = None,
        **kwargs
    ):
        """
        Run the sequence of pipelines with the selected action.

        Parameters
        ----------
        select : str, optional
            Sequence of pipelines to be executed.
        action : str, optional
            Action within each pipeline to be executed.
            If not provided, the default action will be executed.
        chunks : str, optional
            Chunks to be processed.
        """
        action = action or self.default_action

        tstart = datetime.now()

        if isinstance(select, (list, tuple)):
            sequence = {k: self.sequence[k] for k in select}
        elif isinstance(select, str) and select.strip() != '':
            sequence = {k.strip(): self.sequence[k.strip()] for k in select.strip().split(',')}
        else:
            sequence = self.sequence

        if isinstance(chunks, str) and chunks.strip() != '':
            chunks = [c.strip() for c in chunks.strip().split(',')]
        elif not isinstance(chunks, (list, tuple)) or len(chunks) == 0:
            chunks = []

        for group, pipelines in sequence.items():
            try:
                for pname in pipelines:
                    cname = group + '.' + to_camel(pname)
                    pipeline = getattr(self.namespace, cname)()
                    if len(chunks) > 0:
                        for chunk in chunks:
                            pipeline.run(action, chunk=chunk, **kwargs)
                    else:
                        pipeline.run(action, chunk=None, **kwargs)
            except Exception as e:
                self.app.logger.error(e)
                self.alert(
                    f'{self.name} job [{cname}] executed with errors: {e}'
                )

                return

        tend = datetime.now()
        delta = (tend - tstart).seconds

        self.alert(
            f'{self.name} job executed in {strfdelta(delta)} :-)'
        )


class PipelineEnd(Exception):
    """
    Raised by a task to end the pipeline run early, skipping the remaining stages.

    Attributes
    ----------
    result : dict
        Result of the run, with at least a ``status`` and, optionally, a ``reason``.
    """

    def __init__(self, result: dict):
        super().__init__(result)
        self.result = result


class Pipeline(Core):
    """
    Pipeline base class to manage data processing tasks.

    Attributes
    ----------
    pipes : dict
        Dictionary with all actions and tasks to be executed.

        It is a dictionary with 3 levels of nesting, where the first level is the action to be executed,
        the second level is the stage of the action (Setup, Extract, Transform, Load, Build, Set),
        and the third level is the list of tasks to be executed in each stage.

        Example:
        {
            'action1': {
                'Setup': ['task1', 'task2', ...],
                'Extract': ['task3', 'task4', ...],
                'Transform': ['task5', 'task6', ...],
                'Load': ['task7', 'task8', ...],
                'Build': ['task9', 'task10', ...],
                'Set': ['task11', 'task12', ...]
            },
            'action2': {
                ...
            },
            ...
        }

        In every executuin of the pipeline, an action is selected and
        all the tasks in each stage within that action will be executed sequentially.

        Common actions are:
            - 'update': perform an incremental update of the target model.
            - 'bulk': read all data from source, transform it and load it into target.
            - 'rebulk': read all data from source, transform it and upsert it into target.
            - 'load': load data from stage file into target.
            - 'unload': unload data from target into stage file.
            ...

        The tasks within each stage are defined as object methods with '_task' suffix.
        i.e. 'task1' -> 'task1_task()'

        A set of main default tasks are defined in the base class, but can be overridden in the subclass.
    source : str or Model
        Source model to read data from.
    target : str or Model
        Target model to write data to.
    chunked : bool
        Whether the pipeline runs once per chunk (default) or only once.
    label : str, optional
        Short label of the pipeline, used to name its failures.
    result : dict or list, optional
        Result of the last run (``None`` at the start of every run).
    line : str or list of str, optional
        Summary line(s) of the last run (``None`` at the start of every run).
    """

    pipes: dict = {}
    chunked: bool = True
    label: Optional[str] = None

    source: str | Model = None
    target: str | Model = None

    def __init__(self):
        if type(self) is Pipeline:
            raise TypeError('Abstract class should not be instantiated.')

        super().__init__()

        # Name of the pipeline
        self.name = self.get_name()
        # Compression method to use when staging data
        self.compression = getattr(getattr(self.app.config, 'app', None), 'compression', None)

        # Action and stages to be executed
        self._action = None
        self._stages = None

        # Result and summary line of the last run, and whether `start` was already called
        self.result = None
        self.line = None
        self._started = False

        # The dataframe that stores the data which will be processed by the pipeline:
        #   - First, it will be loaded from the source model in the Extract phase
        #   - Then, it will be updated by each task in the Transform phase
        #   - Finally, it will be loaded into the target model in the Load phase
        self._data = pd.DataFrame.from_records({})
        # In case it is a query pipeline, the query to be executed in the Build phase
        self._query = None

        # Number of rows loaded into the target model
        self._loaded_rows = 0

        # Source and target models
        self.source = self.get_model(self.source)
        self.target = self.get_model(self.target)

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()

    @data.setter
    def data(self, data: pd.DataFrame):
        self._data = data
    
    @property
    def rows(self) -> int:
        return self._data.shape[0]

    def get_name(self) -> str:
        prefix = import_module('...pipelines', package=__name__).__name__ + '.'
        name = self.__class__.__module__

        if name.startswith(prefix):
            name = name[len(prefix):]

        return name

    def get_model(self, model: str | Model) -> Model:
        if isinstance(model, str):
            module = import_module('...models', package=__name__)
            model = getattr(module, model)()
        elif callable(model):
            model = model()

        return model

    def get_update_ts(self, filters: Optional[list] = None) -> pd.Timestamp | int:
        """
        Get the last update timestamp from the target model.
        """
        return self.target.get_update_ts(filters)

    def start(self, **kwargs) -> None:
        """
        Hook called once, on the first run, after the action is set and before the first stage.

        Parameters
        ----------
        **kwargs
            Keyword arguments of that first run, except ``chunk``.
        """
        pass

    def finish(self, **kwargs) -> Optional[str]:
        """
        Hook called by the caller once all chunks are done.

        Returns
        -------
        str or None
            A failure label if the closing failed; otherwise ``None``.
        """
        return None

    def end(self, result: dict) -> NoReturn:
        """
        End the run early, skipping the remaining stages.

        Parameters
        ----------
        result : dict
            Result of the run, with a ``status`` and, optionally, a ``reason``.
        """
        raise PipelineEnd(result)

    def summary(self, chunk, **kwargs) -> Optional[str | list[str]]:
        """
        Hook that builds the summary line(s) of a run that left a result and no line.

        Parameters
        ----------
        chunk : str, optional
            Chunk of the run.

        Returns
        -------
        str or list of str or None
            The summary line(s), or ``None`` for no line.
        """
        return None

    def failure_name(self, chunk) -> str:
        """
        Name to report a failure of this pipeline.

        Parameters
        ----------
        chunk : str, optional
            Chunk that failed.

        Returns
        -------
        str
            ``'chunk (label)'`` with chunk and label, the chunk without label,
            or the label (or pipeline name) without chunk.
        """
        if chunk is not None:
            return f'{chunk} ({self.label})' if self.label else str(chunk)

        return self.label or self.name

    def run(
        self,
        action: Optional[str] = None,
        **kwargs
    ):
        """
        Run the pipeline with the selected action.

        Parameters
        ----------
        action : str, optional
            Action to be executed.
            If not provided, the first action found in `pipes` dictionary will be executed.
        """
        self.result = None
        self.line = None

        self._action = action if action in self.pipes else list(self.pipes)[0]
        self._stages = self.pipes[self._action]

        if not self._started:
            self.start(**{k: v for k, v in kwargs.items() if k != 'chunk'})
            self._started = True

        chunk = kwargs.get('chunk', None)
        if chunk is not None:
            self.log('info', f'Run {self.name}.{self._action}.{chunk} pipeline...')
        else:
            self.log('info', f'Run {self.name}.{self._action} pipeline...')

        tstart = datetime.now()

        try:
            # Setup stage -> tasks that must be executed before the data is loaded from the source
            if 'Setup' in self._stages and len(self._stages['Setup']) > 0:
                self.log('info', '--> Setup...')

                for task in self._stages['Setup']:
                    self.log('info', f'{to_camel(task)}...')

                    fn = f'{task}_task'
                    getattr(self, fn)(**kwargs)

        
            # Extract stage -> tasks needed to load data from the source
            if 'Extract' in self._stages and len(self._stages['Extract']) > 0:
                self.log('info', '--> Extract...')

                for task in self._stages['Extract']:
                    self.log('info', f'{to_camel(task)}...')

                    fn = f'{task}_task'
                    getattr(self, fn)(**kwargs)

                # Once the data is loaded, it will be formatted using the source model meta definitions
                if self.source is not None:
                    self.data = self.source.format_data(
                        self.data,
                        int_type='nullable', bin_type='nullable'
                    )

            # Transform stage -> tasks that process the data in order to be loaded into the target table
            if 'Transform' in self._stages and len(self._stages['Transform']) > 0:
                self.log('info', '--> Transform {:,} rows...'.format(self.rows))

                for task in self._stages['Transform']:
                    self.log('info', f'{to_camel(task)}...')

                    fn = f'{task}_task'
                    getattr(self, fn)(**kwargs)

                # Once the data is transformed, it will be formatted using the target model meta definitions
                if self.target is not None:
                    self.data = self.target.format_data(
                        self.data,
                        int_type='nullable', bin_type='nullable',
                        obj_type='json', sort=True
                    )

            # Load stage -> tasks needed to load the data into the target database table
            if 'Load' in self._stages and len(self._stages['Load']) > 0:
                if self.rows > 0:
                    self.log('info', '--> Load {:,} rows...'.format(self.rows))
                elif self.target is None:
                    self.log('info', '--> Load...')
                else:
                    self.log('info', '--> Load staged data...')

                for task in self._stages['Load']:
                    self.log('info', '{}...'.format(to_camel(task)))

                    fn = f'{task}_task'
                    getattr(self, fn)(**kwargs)

            # Build stage -> tasks needed to build the query to be executed in the target database
            # The `query` must be a SELECT query that returns the data to be loaded into the target table
            if 'Build' in self._stages and len(self._stages['Build']) > 0:
                self.log('info', '--> Build...')

                self.target.begin()
                try:
                    for task in self._stages['Build']:
                        self.log('info', '{}...'.format(to_camel(task)))

                        fn = f'{task}_task'
                        getattr(self, fn)(**kwargs)
                except Exception as e:
                    self.target.rollback()
                    raise e
                else:
                    self.target.commit()

            # Set stage -> tasks that must be executed after the data is loaded into the target table
            # This stage is used to perform additional operations in the target table (eg: vacuum, analyze, etc.)
            if 'Set' in self._stages and len(self._stages['Set']) > 0:
                self.log('info', '--> Set...')

                for task in self._stages['Set']:
                    self.log('info', '{}...'.format(to_camel(task)))

                    fn = f'{task}_task'
                    getattr(self, fn)(**kwargs)
        except PipelineEnd as end:
            self.result = end.result
            self.log('info', 'Pipeline {}.{} ended: {} ({})'.format(
                self.name, self._action, self.result.get('status'), self.result.get('reason')
            ))

        if self.result is not None and self.line is None:
            self.line = self.summary(chunk, **{k: v for k, v in kwargs.items() if k != 'chunk'})

        tend = datetime.now()
        delta = (tend - tstart).seconds

        if self._loaded_rows > 0:
            self.log('info', 'Pipeline {}.{} executed loading {:,} rows in {} :-)'.format(
                self.name, action, self._loaded_rows, strfdelta(delta)
            ))
        else:
            self.log('info', 'Pipeline {}.{} executed in {} :-)'.format(
                self.name, action, strfdelta(delta)
            ))
    
    ### Default predefined tasks that can be used in the pipelines ###

    def read_task(self, **kwargs) -> Self:
        """
        Read the data from the source and load it into the `data` object.
        """
        query = kwargs.get('query', {})

        self.data = self.source.get_results(query)

        return self

    def stage_task(self, **kwargs) -> Self:
        """
        Write the content of the `data` object into the stage area,
        from where it will be later loaded into the target table.
        """
        chunk = kwargs.get('chunk', None)

        if self.rows == 0:
            return self

        if self.target.stage:
            self.app.logger.debug('Stage Write...')
            self.target.stage_write(self.data, chunk=chunk, compression=self.compression)

        return self

    def unload_task(self, **kwargs) -> Self:
        """
        Unload and write the content of the target table into the stage area.
        """
        query = kwargs.get('query', {})

        if self.target.stage:
            self.app.logger.debug('Stage Unload...')
            self.target.stage_unload(query, header=True, **kwargs)

        return self

    def insert_task(self, **kwargs):
        """
        Insert data into the target table.

        If the target table is staged, the data will be loaded from the stage area.
        Otherwise, the data will be directly inserted into the target table.

        If the target table already exists, it will be recreated and replaced with the new data.
        """
        chunk = kwargs.get('chunk', None)
        stage = kwargs.get('stage', False)

        if self.target.stage:
            if stage:
                if self.rows > 0:
                    self.app.logger.debug('Stage...')
                    self.target.stage_write(self.data, chunk=chunk, compression=self.compression)
                else:
                    return self

            self.app.logger.debug('Write...')
            self._loaded_rows = self.target.create().insert()

            self.app.logger.debug('Cleanup...')
            self.target.stage_clean()
        elif self.rows > 0:
            self.app.logger.debug('Write...')
            self._loaded_rows = self.target.create().insert(self.data)

        return self

    def query_insert_task(self, **kwargs):
        """
        Insert the result of a SELECT `query` into the target model.

        If the target table already exists, it will be recreated and replaced with the new data.
        """
        if self.query is None:
            return self

        self.app.logger.debug('Write...')
        self._loaded_rows = self.target.create().insert(self.query)

        return self

    def upsert_task(self, **kwargs):
        """
        Upsert data into the target table.

        An upsert operation is a combination of insert and update, this method will insert the data into the table,
        and if a row with the same key already exists, it will update the existing row.

        If the target table is staged, the data will be loaded from the stage area.
        Otherwise, the data will be directly upserted into the target table.
        """
        chunk = kwargs.get('chunk', None)
        stage = kwargs.get('stage', True)

        if self.target.stage:
            if stage:
                if self.rows > 0:
                    self.app.logger.debug('Stage...')
                    self.target.stage_write(self.data, chunk=chunk, compression=self.compression)
                else:
                    return self

            self.app.logger.debug('Write...')
            self._loaded_rows = self.target.upsert()

            self.app.logger.debug('Cleanup...')
            self.target.stage_clean()
        elif self.rows > 0:
            self.app.logger.debug('Write...')
            self._loaded_rows = self.target.upsert(self.data)

        return self

    def query_upsert_task(self, **kwargs):
        """
        Upsert the result of a SELECT `query` into the target table.

        An upsert operation is a combination of insert and update, this method will insert the data into the table,
        and if a row with the same key already exists, it will update the existing row.
        """
        if self.query is None:
            return self

        self.app.logger.debug('Write...')
        self._loaded_rows = self.target.upsert(self.query)

        return self

    def vacuum_task(self, **kwargs):
        """
        Vacuum the target table, in order to reclaim storage
        and reorganize the physical storage of the table.
        """
        self.app.logger.debug('DB Cleanup...')
        self.target.vacuum()

        return self
