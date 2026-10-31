#!/usr/bin/env python
"""
Run the model backtest on past elections and write the results to `backtest/results/`.

Usage (from the repository root):
    python backtest/run_backtest.py [--scope es | --scopes es-md es-cl ... | --scopes all]
                                    [--events 2019-04-28 2023-07-23] [--horizons 6 30] [--n-sim 1000] [--seed 42] [--nowcast-only]
                                    [--no-house-effects] [--recompute-ratings] [--industry-bias] [--no-dispersion] [--no-herding] [--composition auto|0.25|1]

The national scope (`es`) writes to `backtest/results/`; every other scope to `backtest/results/{scope}/`.
Without `--events`, a regional scope evaluates its featured elections held since 2019.

This script is a thin wrapper of the `backtest` job (`python job.py backtest '{...}'`, `mtpy/jobs/Backtest.py`),
which does the work through `mtpy.lib.backtest.backtest_scope`; it only translates the command line options. Note
that `--scopes all` here means every regional scope (not `es`), while `"scopes": "all"` in the job means `es` and
every regional scope.

The tables go to the repository copy (`backtest/results`, or `--out`), while `publish backtest` and `publish event`
read the app file system (S3 with `S3_BUCKET`): after a local run, `push` the tables (the hint printed at the end) or
pass `"source":"backtest/results"` to `publish`.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description='Backtest of the elections model on past elections')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--scope', default='es')
    group.add_argument('--scopes', nargs='+', default=None, help="several scopes (es-md es-cl ...) or 'all' for every regional one; each writes to its own directory")
    parser.add_argument('--events', nargs='*', default=None)
    parser.add_argument('--horizons', nargs='*', type=int, default=None)
    parser.add_argument('--n-sim', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-fc', type=int, default=10)
    parser.add_argument('--nowcast-only', action='store_true', help='skip the horizon run of each case (columns `_h`)')
    parser.add_argument('--no-house-effects', dest='house_effects', action='store_false', help='do not subtract the house effects (M6)')
    parser.add_argument('--industry-bias', action='store_true', help='shift the average by the industry-wide bias of past elections (M6)')
    parser.add_argument('--no-dispersion', dest='dispersion', action='store_false', help='do not weigh each pollster by its effective sample (M12)')
    parser.add_argument('--no-herding', dest='herding', action='store_false', help='do not reduce the weight of the pollsters less dispersed than their samples allow (M13)')
    parser.add_argument('--composition', default=None, help="joint noise of the national parties (M7): 'auto' or a ratio in (0, 1]; independent draws by default")
    parser.add_argument('--no-regional-noise', dest='regional_noise', action='store_false', help='deterministic proportional swing per province (no M9 shocks)')
    parser.add_argument('--min-events', type=int, default=None, help='featured elections of a regional scope needed to fit the estimators on the scope itself instead of on `es` (M11); by default always `es`, 1 forces the own ones')
    parser.add_argument('--recompute-ratings', action='store_true', help='recompute the pollster ratings in memory with the current code (nothing is saved) instead of using the stored ones')
    default_out = os.path.join(ROOT, 'backtest', 'results')
    parser.add_argument('--out', default=default_out)
    args = parser.parse_args()

    from mtpy import mtpy
    mtpy.run()

    from mtpy.jobs.Backtest import Backtest

    if args.scopes is None:
        scopes = [args.scope]
    elif args.scopes == ['all']:
        from mtpy.lib.data import get_scopes

        catalogue = get_scopes()
        scopes = catalogue.loc[catalogue['parent'].notnull()].index.tolist()
    else:
        scopes = args.scopes

    Backtest().run(
        scopes=scopes, events=args.events, horizons=args.horizons, n_sim=args.n_sim, seed=args.seed, max_fc=args.max_fc,
        nowcast_only=args.nowcast_only, house_effects=args.house_effects, industry_bias=args.industry_bias,
        dispersion=args.dispersion, herding=args.herding, composition=args.composition,
        regional_noise=args.regional_noise, min_events=args.min_events,
        recompute_ratings=args.recompute_ratings, out=args.out
    )
    print(push_hint(args.out, scopes, default_out))


def push_hint(out, scopes, default_out):
    """
    Line telling where the tables went and how to make `publish` see them.

    Parameters
    ----------
    out : str
        Results folder of the run.
    scopes : list of str
        Scopes evaluated.
    default_out : str
        The repository copy (`backtest/results`), the folder `push` reads without `out`.

    Returns
    -------
    str
        The hint, with the `push` command of these scopes (and `out` when it is not the repository copy).
    """
    params = {'action': 'push', 'scopes': list(scopes)}
    if os.path.abspath(out) != os.path.abspath(default_out):
        params['out'] = out
    command = "python job.py backtest '{}'".format(json.dumps(params, separators=(',', ':')))

    return ('Tables written to {}. publish backtest/event read the app file system: run {} (with S3_BUCKET set) '
            'or pass "source" to publish.'.format(out, command))


if __name__ == '__main__':
    main()
