#!/usr/bin/env python
"""
Tune the configuration of the poll average on the polls published after each cut and write the tables to
`backtest/tuning/`.

Usage (from the repository root):
    python backtest/run_tuning.py [--scope es | --scopes es es-md ... | --scopes all]
                                  [--mode smoothing|shrinkage|check|evaluate|merge]
                                  [--config default] [--config bw=fixed,45 bw_cap=1.5] [--config house_effects=false]
                                  [--cycles 2019-11-10 ...] [--cut-days 7] [--horizons 7 28] [--rounds 2]
                                  [--out backtest/tuning] [--verbose 1] [--workers 8]

The search runs in three steps, each reading the `best.json` the previous one left in the folder of the scope:
`--mode smoothing` (phase 1), `--mode shrinkage` (phase 2) and `--mode check` (the best configuration against
the default on the held-out cycles). `--config` (repeatable, one configuration each: `default`, or settings
`key=value` applied in order on the default, a comma making a tuple: `bw=fixed,45`) evaluates the configurations
given (`--mode evaluate`, the mode when `--config` is given). `--workers` spreads the cuts of each evaluation over
that many processes, each with its own database connection (1, the default, runs them one after the other). The
shrinkage phase can also run in parts, one per `--cycles` (each writes `trials-shrinkage-{cycles}.csv` and its
companions, never `best.json`), joined afterwards by `--mode merge`. The national scope (`es`) writes to
`backtest/tuning/`; every other scope to `backtest/tuning/{scope}/`; `--scopes all` is `es` and every regional
scope.

This script is a thin wrapper of the `tune` job (`python job.py tune '{...}'`, `mtpy/jobs/Tune.py`), which does
the work through `mtpy.lib.tuning.tuning_scope`; it only translates the command line options and, at the end,
prints the folder written and, in the search modes, the configuration kept and the scores of the `check`
partition of `default` and of the configuration checked against it when the folder has them (also for the
completed scopes when another one fails).
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

MODES = ['smoothing', 'shrinkage', 'check', 'evaluate', 'merge']


def build_parser():
    """
    Command line options of the script.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    parser = argparse.ArgumentParser(description='Tuning of the poll average on the future polls')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--scope', default='es')
    group.add_argument('--scopes', nargs='+', default=None,
                       help="several scopes (es es-md ...) or 'all' for es and every regional one")
    parser.add_argument('--mode', choices=MODES, default=None,
                        help='phase of the search or evaluate; evaluate with --config, smoothing otherwise')
    parser.add_argument('--config', nargs='+', action='append', default=None, metavar='KEY=VALUE',
                        help="one configuration to evaluate: 'default' or settings key=value applied in order on "
                             "the default (bw=fixed,45 bw_cap=1.5 wspan=7 max_fc=0 house_effects=false); repeatable")
    parser.add_argument('--cycles', nargs='+', default=None, help='elections (YYYY-MM-DD) to restrict the tuning to')
    parser.add_argument('--cut-days', type=int, default=None, help='days between two cuts; the default of the mode')
    parser.add_argument('--horizons', nargs='+', type=int, default=[7, 28])
    parser.add_argument('--rounds', type=int, default=None, help='rounds of the descent; the default of the phase')
    parser.add_argument('--out', default=os.path.join(ROOT, 'backtest', 'tuning'))
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1,
                        help='processes running the cuts of each evaluation, each with its own database connection')
    return parser


def parse_value(text):
    """
    Value of a `key=value` setting.

    Parameters
    ----------
    text : str
        The value as written: a comma makes a tuple of values (`fixed,45`); `null`/`None`, `true` and `false`
        are None and the booleans; integers and floats are numbers; anything else stays a string.

    Returns
    -------
    Any
        The parsed value.
    """
    if ',' in text:
        return tuple(parse_value(part.strip()) for part in text.split(','))
    if text.lower() in ('null', 'none'):
        return None
    if text.lower() in ('true', 'false'):
        return text.lower() == 'true'
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def parse_configs(groups):
    """
    Configurations of the `--config` options, in the form of the `configs` of the job.

    Parameters
    ----------
    groups : list of list of str or None
        The tokens of each `--config` option.

    Returns
    -------
    list or None
        `'default'` or a `{key: value}` dict per option; None without options.

    Raises
    ------
    ValueError
        A token without `=`, or `default` with other tokens.
    """
    if groups is None:
        return None
    configs = []
    for tokens in groups:
        if 'default' in tokens:
            if len(tokens) > 1:
                raise ValueError('--config "default" goes alone (got {})'.format(' '.join(tokens)))
            configs.append('default')
            continue
        config = {}
        for token in tokens:
            key, sep, value = token.partition('=')
            if not sep or not key:
                raise ValueError('--config expects key=value settings (got {!r})'.format(token))
            config[key] = parse_value(value)
        configs.append(config)
    return configs


def job_arguments(args):
    """
    Arguments of the `tune` job from the command line options.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed options (see `build_parser`).

    Returns
    -------
    dict
        `scopes`, `mode`, `configs`, `cycles`, `cut_days`, `horizons`, `rounds`, `out`, `verbose` and `workers`.

    Raises
    ------
    ValueError
        A bad `--config` option (see `parse_configs`).
    """
    configs = parse_configs(args.config)
    if args.scopes is None:
        scopes = [args.scope]
    elif args.scopes == ['all']:
        scopes = 'all'
    else:
        scopes = args.scopes
    return {
        'scopes': scopes, 'mode': args.mode or ('evaluate' if configs else 'smoothing'), 'configs': configs,
        'cycles': args.cycles, 'cut_days': args.cut_days, 'horizons': tuple(args.horizons), 'rounds': args.rounds,
        'out': args.out, 'verbose': args.verbose, 'workers': args.workers
    }


def check_scores(scores):
    """
    Scores of `default` and of the configuration checked against it on the `check` partition, one row each.

    Parameters
    ----------
    scores : pandas.DataFrame
        `scores.csv` of a scope. The checked configuration is stored under its own label (the settings that differ
        from the default, as `best.json` names it, or `best`): every label of the partition other than `default`.

    Returns
    -------
    pandas.DataFrame
        `config` (the label as `scores.csv` has it), `n_cycles`, `rmse7` and `rmse28` (the mean over the cycles of
        the RMSE of each cycle, every cycle weighing the same, as the search compares them) and `roughness` (the
        mean over the rows), `default` first; empty when the table has no such rows.
    """
    import pandas as pd

    rows = scores.loc[scores['partition'] == 'check']
    labels = list(rows['config'].drop_duplicates())
    labels = [label for label in labels if label == 'default'] + [label for label in labels if label != 'default']
    out = []
    for label in labels:
        own = rows.loc[rows['config'] == label]
        out.append({'config': label, 'n_cycles': own['event_date'].nunique(),
                    'rmse7': own.loc[own['k'] == 7, 'rmse'].astype(float).mean(),
                    'rmse28': own.loc[own['k'] == 28, 'rmse'].astype(float).mean(),
                    'roughness': own['roughness'].astype(float).mean()})
    return pd.DataFrame(out, columns=['config', 'n_cycles', 'rmse7', 'rmse28', 'roughness'])


def report(results, out):
    """
    Print the folder written by each completed scope and, in the search modes, the configuration kept and the
    scores of `default` and of the configuration checked on the `check` partition when its `scores.csv` has them.

    Parameters
    ----------
    results : dict[str, dict]
        Results of the `tune` job by scope.
    out : str
        Local folder of the tables.
    """
    from mtpy.lib.backtest import results_path
    from mtpy.lib.tuning import tuning_fs

    for scope, result in results.items():
        if result['status'] != 'completed':
            continue
        print('{}: tables written to {}'.format(scope, result['folder']))
        best = result.get('best')
        if best is None:
            continue
        print('{}: best = {}'.format(scope, best.get('label')))
        print('  ' + json.dumps({key: value for key, value in best.items() if key != 'label'}))
        fs, folder = tuning_fs(out, scope)
        path = results_path(folder, 'scores.csv')
        if fs.exists(path):
            table = check_scores(fs.read_csv(path))
            if not table.empty:
                print(table.round(3).to_string(index=False))


def run_job(job, arguments, out):
    """
    Run the `tune` job and report its scopes, also the completed ones when another scope failed.

    Parameters
    ----------
    job : mtpy.jobs.Tune.Tune
        The job.
    arguments : dict
        Its arguments (see `job_arguments`).
    out : str
        Local folder of the tables.

    Raises
    ------
    RuntimeError
        The error of the job when a scope failed, after the report.
    """
    try:
        job.run(**arguments)
    finally:
        report({chunk: result for (key, chunk), result in getattr(job, 'results', {}).items()}, out)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        arguments = job_arguments(args)
    except ValueError as e:
        parser.error(str(e))

    from mtpy import mtpy
    mtpy.run()

    from mtpy.jobs.Tune import Tune

    run_job(Tune(), arguments, args.out)


if __name__ == '__main__':
    main()
