from datetime import datetime

from ..core.worker import SequenceJob
from ..lib import tuning
from ..lib.publish import resolve_scopes

MODES = ('smoothing', 'shrinkage', 'check', 'evaluate', 'merge')

# Modes a grid of the search applies to
GRID_MODES = ('smoothing', 'shrinkage', 'merge')

# Switches of the average a configuration to evaluate may set besides the keys of the grid
SWITCHES = ('house_effects', 'dispersion', 'herding')


class Tune(SequenceJob):
    """
    Tune the configuration of the poll average on the polls published after each cut, or evaluate given
    configurations, and write the tables of each scope to ``backtest/tuning`` of the app file system (S3 in ECS;
    the job form of ``backtest/run_tuning.py``).

    Scope by scope (`tuning.tuning_scope`), each mode rewrites ``cuts.csv``, ``polls.csv``, ``scores.csv``,
    ``trials.csv`` and ``run.json`` (``es`` at the root of the folder, the other scopes in ``{scope}/``): the
    search modes ``smoothing`` (phase 1: bandwidth, cap, degree, prolongation and window), ``shrinkage`` (phase 2:
    house effects and herding, from the ``best.json`` of phase 1) and ``check`` (the ``best.json`` against the
    default on the held-out cycles) also write ``trials-{mode}.csv`` and, the first two, ``best.json``;
    ``evaluate`` scores the configurations given. The search modes also keep ``scores-{mode}.csv`` beside
    ``trials-{mode}.csv``. ``shrinkage`` with ``cycles`` is a part of the phase (named after the cycles kept: it
    evaluates the whole product of the values of ``tau``, ``he_n_iter`` and ``herd_prior`` from the start, 18
    configurations with the grid of the spec instead of the 6 of the descent, so a part costs three times its share
    of the whole phase; it writes ``cuts-``, ``scores-`` and ``trials-shrinkage-{part}.csv`` and
    ``run-shrinkage-{part}.json``, never ``best.json``), so the phase can run one cycle per task; ``merge`` joins the
    parts of the folder, repeats the descent on the joined table from the start the parts recorded and writes
    ``best.json`` (`tuning.merge_parts`): the same result as the whole phase run at once. ``workers`` spreads the
    cuts of each evaluation over that many processes, each with its own app and database engine (the tables are
    those of the sequential run). Scopes are isolated: one that fails never stops the others (unless ``stop_on_error``),
    and the job exits with an error at the end if any failed. A scope with nothing to tune, or without the
    ``best.json`` that ``shrinkage``, ``check`` and ``merge`` start from, ends ``skipped`` and nothing is written
    for it; a tuning that cannot be scored (a failed cut of the start of a search or of the check, or every cut
    failed) ends ``failed``.

    ``push`` copies the tables of ``out`` (``backtest/tuning`` of the repository by default) to the app file system
    and ``pull`` the reverse, mirroring the tuning files of each scope (as the ``backtest`` job does).

    The console output has the levels of ``backtest``: 0, the header and the final summary; 1, also
    ``[i/n] scope: empezando``, the result line and the warnings as they happen; 2, also the line of each cut.

    Examples
    --------
    From the command line::

        python job.py tune '{"scopes":["es"],"mode":"smoothing","workers":8}'
        python job.py tune '{"scopes":["es"],"mode":"shrinkage","cycles":["2019-11-10"]}'
        python job.py tune '{"scopes":["es"],"mode":"merge"}'
        python job.py tune '{"scopes":["es"],"mode":"check"}'
        python job.py tune '{"action":"pull","scopes":["es"]}'
        python job.py tune '{"scopes":["es"],"mode":"evaluate","configs":["default",{"bw":["fixed",45]}]}'
        python job.py tune '{"scopes":["es"],"mode":"smoothing","grid":{"bw":[["fixed",45],["adaptive",3]]}}'

    In Docker (JSON without spaces)::

        RUN_JOB='tune {"scopes":["es"],"mode":"smoothing"}'
    """

    sequence = {'tune': ['tune']}
    default_action = 'run'
    stop_on_error = False
    chunk_noun = 'scopes'

    def run(self, scopes=('es',), mode='smoothing', configs=None, grid=None, cycles=None, cut_days=None,
            horizons=(7, 28), rounds=None, out=None, verbose=1, runner=None, stats=None, stop_on_error=None,
            action='run', workers=1, **kwargs):
        """
        Run one mode of the tuning of the requested scopes.

        Parameters
        ----------
        scopes : str or list of str, default ('es',)
            Scopes to tune, or ``'all'`` (``es`` and every regional scope, see ``publish.resolve_scopes``).
        mode : {'smoothing', 'shrinkage', 'check', 'evaluate', 'merge'}, default 'smoothing'
            Phase of the search, the evaluation of ``configs`` or the merge of the parts of the shrinkage phase.
        configs : list, optional
            Configurations of the ``evaluate`` mode (required there, refused in the others): ``'default'`` (the
            live average, `tuning.default_config`) or a dict of settings applied in order on the default, each a
            key of `tuning.GRID` with one of its values (JSON lists as tuples: ``{"bw": ["fixed", 45]}``) or one
            of ``house_effects``, ``dispersion`` and ``herding`` with a boolean. The label is ``key=value`` of
            each setting, joined by commas.
        grid : dict, optional
            Values of each key of the search (``{key: [values]}``, a subset of `tuning.GRID`); the whole grid by
            default. Only in ``smoothing``, ``shrinkage`` and ``merge``.
        cycles : list of str, optional
            Elections (``YYYY-MM-DD``) the tuning is restricted to, in their partition (search or check); every
            cycle of the scope by default. A date that is not a cycle of a scope is ignored with a warning. In
            ``shrinkage`` they make the run a part of the phase.
        cut_days : int, optional
            Days between two cuts; the default of the mode.
        horizons : list of int, default (7, 28)
            Horizons in days (the search modes need 7 and 28).
        rounds : int, optional
            Rounds of the descent; the default of the phase.
        out : str, optional
            Local folder of the tables. For ``run`` the app file system (``backtest/tuning``, S3 in ECS) by
            default; for ``push`` (its origin) and ``pull`` (its destination) the repository copy by default.
        verbose : int, default 1
            Verbosity of the console output (see the class).
        runner : callable, optional
            Replacement of `tuning.tuning_scope` (for tests).
        stats : callable, optional
            Replacement of `backtest.poll_stats` (for tests).
        stop_on_error : bool, optional
            Stop at the first failed scope instead of going on with the others; ``False`` by default.
        action : {'run', 'push', 'pull'}, default 'run'
            ``run`` runs the mode; ``push`` copies the tables from ``out`` to the app file system; ``pull`` from
            the app file system to ``out``.
        workers : int, default 1
            Processes running the cuts of each evaluation (a positive integer).
        **kwargs
            Ignored extra job arguments.

        Returns
        -------
        dict[str, dict]
            Result of each scope: ``status`` (``completed``, ``pushed``, ``pulled``, ``skipped`` or ``failed``);
            when completed, the figures of `tuning.tuning_scope` (``n_configs``, ``n_cuts``, ``cycles``,
            ``folder``, ``seconds`` and, in the search modes and ``merge``, ``best``); when pushed or pulled,
            ``n_files``, ``n_removed`` and ``folder`` (the destination); otherwise the ``reason``. Every result
            carries its ``warnings``.

        Raises
        ------
        ValueError
            If an argument is not valid (``tune: ...``), before anything runs, or there are no scopes to run.
        RuntimeError
            If any scope failed (raised after the summary and the alert).
        """
        if action not in ('run', 'push', 'pull'):
            raise ValueError('tune: unknown action "{}" (expected run, push or pull)'.format(action))
        if mode not in MODES:
            raise ValueError('tune: mode must be {} or {} (got {!r})'.format(', '.join(MODES[:-1]), MODES[-1], mode))
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError('tune: workers must be a positive integer (got {!r})'.format(workers))
        configs = self._configs(configs, mode)
        grid = self._grid(grid, mode)
        cycles = self._cycles(cycles)
        cut_days = self._positive('cut_days', cut_days)
        rounds = self._positive('rounds', rounds)
        horizons = self._horizons(horizons)
        targets = resolve_scopes(scopes)
        if not targets:
            raise ValueError('tune: no scopes to run')

        super().run(select='tune', chunks=targets, action=action, stop_on_error=stop_on_error, mode=mode,
                    configs=configs, grid=grid, cycles=cycles, cut_days=cut_days, horizons=horizons, rounds=rounds,
                    out=out, verbose=verbose, runner=runner, stats=stats, workers=workers)

        return {chunk: result for (key, chunk), result in self.results.items()}

    @staticmethod
    def _configs(values, mode):
        """
        Normalise the ``configs`` argument into configurations of the average.

        Parameters
        ----------
        values : list or None
            ``'default'`` or dicts of settings (see `run`).
        mode : str
            Mode of the run.

        Returns
        -------
        list of dict or None
            The configurations (``None`` outside ``evaluate``).

        Raises
        ------
        ValueError
            Configurations outside ``evaluate``, none in it, or one that is not ``'default'`` nor a non-empty
            dict of valid settings.
        """
        if mode != 'evaluate':
            if values is not None:
                raise ValueError('tune: configs only apply to the evaluate mode (got mode {!r})'.format(mode))
            return None
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError('tune: the evaluate mode needs configs ("default" or {key: value} dicts)')
        out = []
        for value in values:
            if value == 'default':
                out.append(tuning.default_config())
                continue
            if not isinstance(value, dict) or len(value) == 0:
                raise ValueError('tune: a config must be "default" or a non-empty {{key: value}} dict (got {!r})'
                                 .format(value))
            config, labels = tuning.default_config(), []
            for key, setting in value.items():
                setting = tuple(setting) if isinstance(setting, list) else setting
                if key in SWITCHES:
                    if not isinstance(setting, bool):
                        raise ValueError('tune: bad config {!r}: {} must be true or false'.format(value, key))
                    config[key] = setting
                else:
                    try:
                        config = tuning.apply_setting(config, key, setting)
                    except ValueError as e:
                        raise ValueError('tune: bad config {!r}: {}'.format(value, e)) from None
                labels.append('{}={}'.format(key, setting))
            config['label'] = ', '.join(labels)
            out.append(config)
        return out

    @staticmethod
    def _grid(value, mode):
        """
        Normalise the ``grid`` argument: JSON lists of ``bw`` to tuples, every value checked against `tuning.GRID`.

        Raises
        ------
        ValueError
            A grid in ``check`` or ``evaluate``, or a key or value outside `tuning.GRID`.
        """
        if value is None:
            return None
        if mode not in GRID_MODES:
            raise ValueError('tune: grid only applies to the {} modes (got mode {!r})'.format(
                ', '.join(GRID_MODES), mode))
        if not isinstance(value, dict) or len(value) == 0:
            raise ValueError('tune: grid must be a non-empty {{key: [values]}} dict (got {!r})'.format(value))
        out = {}
        for key, values in value.items():
            if key not in tuning.GRID or not isinstance(values, (list, tuple)) or len(values) == 0:
                raise ValueError('tune: grid key {!r} must be one of {} with a list of values'.format(
                    key, list(tuning.GRID)))
            values = [tuple(v) if isinstance(v, list) else v for v in values]
            for v in values:
                if v not in tuning.GRID[key]:
                    raise ValueError('tune: grid value {!r} of {} is not in {}'.format(v, key, tuning.GRID[key]))
            out[key] = values
        return out

    @staticmethod
    def _cycles(value):
        """
        Validate the ``cycles`` argument: ``None`` or a non-empty list of ``YYYY-MM-DD`` dates.

        Raises
        ------
        ValueError
            Anything else (a single string included).
        """
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) == 0:
            raise ValueError('tune: cycles must be a list of YYYY-MM-DD dates or null (got {!r})'.format(value))
        for date in value:
            try:
                valid = isinstance(date, str) and datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d') == date
            except ValueError:
                valid = False
            if not valid:
                raise ValueError('tune: cycles must be a list of YYYY-MM-DD dates or null (got {!r})'.format(date))
        return list(value)

    @staticmethod
    def _positive(name, value):
        """
        Validate ``cut_days`` or ``rounds``: ``None`` or a positive integer.

        Raises
        ------
        ValueError
            Anything else (a boolean or a float included).
        """
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('tune: {} must be a positive integer or null (got {!r})'.format(name, value))
        return value

    @staticmethod
    def _horizons(value):
        """
        Normalise the ``horizons`` argument: a non-empty list of positive integers, as a tuple.

        Raises
        ------
        ValueError
            Anything else.
        """
        if (not isinstance(value, (list, tuple)) or len(value) == 0
                or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in value)):
            raise ValueError('tune: horizons must be a list of positive integers (got {!r})'.format(value))
        return tuple(value)
