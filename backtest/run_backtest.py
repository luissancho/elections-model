#!/usr/bin/env python
"""
Run the model backtest on past elections and write the results to `backtest/results/`.

Usage (from the repository root):
    python backtest/run_backtest.py [--scope es | --scopes es-md es-cl ... | --scopes all]
                                    [--events 2019-04-28 2023-07-23] [--horizons 6 30] [--n-sim 500] [--seed 42] [--nowcast-only]
                                    [--no-house-effects] [--industry-bias] [--composition auto|0.25|1]

The national scope (`es`) writes to `backtest/results/`; every other scope to `backtest/results/{scope}/`.
Without `--events`, a regional scope evaluates its featured elections held since 2019.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def results_dir(root: str, scope: str) -> str:
    """
    Output directory of the backtest of a scope: `root` for the national one (`es`), a subdirectory named
    after the scope for the others.
    """
    return root if scope == 'es' else os.path.join(root, scope)


def run_scope(args: argparse.Namespace, scope: str, composition) -> bool:
    """
    Run the backtest of one scope and write its tables and `run.json` to its output directory.

    Returns
    -------
    bool
        `False` when the scope has no election to evaluate (nothing is written).
    """
    from mtpy.lib.backtest import default_events, run_backtest
    from mtpy.models import elections as models

    events = args.events or default_events(scope)
    if len(events) == 0:
        print('{}: no elections to evaluate, skipped'.format(scope))
        return False

    results = run_backtest(
        scope=scope, events=events, horizons=args.horizons,
        n_sim=args.n_sim, seed=args.seed, max_fc=args.max_fc, nowcast_only=args.nowcast_only,
        house_effects=args.house_effects, industry_bias=args.industry_bias, composition=composition,
        regional_noise=args.regional_noise, verbose=1
    )
    if results['metrics'].shape[0] == 0:
        print('{}: no case could be evaluated, skipped'.format(scope))
        return False

    out = results_dir(args.out, scope)
    os.makedirs(out, exist_ok=True)
    for key, df in results.items():
        df.to_csv(os.path.join(out, '{}.csv'.format(key)), index=False)

    polls = models.Polls().get_results(query=dict(filters=["event_scope = '{}'".format(scope)]), formatted=True)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT).decode().strip()
    except Exception:
        commit = None
    meta = {
        'run_at': datetime.now().isoformat(timespec='seconds'), 'commit': commit, 'scope': scope,
        'n_sim': args.n_sim, 'seed': args.seed, 'max_fc': args.max_fc, 'nowcast_only': args.nowcast_only,
        'house_effects': args.house_effects, 'industry_bias': args.industry_bias, 'composition': composition,
        'regional_noise': args.regional_noise,
        'events': results['metrics']['event_date'].unique().tolist(), 'horizons': sorted(results['metrics']['horizon'].unique().tolist()),
        'db_polls': int(polls.shape[0]), 'db_last_poll': str(polls['date'].max().date())
    }
    with open(os.path.join(out, 'run.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)

    print(scope)
    print(results['by_horizon'].round(3).to_string(index=False))

    return True


def main():
    parser = argparse.ArgumentParser(description='Backtest of the elections model on past elections')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--scope', default='es')
    group.add_argument('--scopes', nargs='+', default=None, help="several scopes (es-md es-cl ...) or 'all' for every regional one; each writes to its own directory")
    parser.add_argument('--events', nargs='*', default=None)
    parser.add_argument('--horizons', nargs='*', type=int, default=None)
    parser.add_argument('--n-sim', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-fc', type=int, default=10)
    parser.add_argument('--nowcast-only', action='store_true', help='skip the horizon run of each case (columns `_h`)')
    parser.add_argument('--no-house-effects', dest='house_effects', action='store_false', help='do not subtract the house effects (M6)')
    parser.add_argument('--industry-bias', action='store_true', help='shift the average by the industry-wide bias of past elections (M6)')
    parser.add_argument('--composition', default=None, help="joint noise of the national parties (M7): 'auto' or a ratio in (0, 1]; independent draws by default")
    parser.add_argument('--no-regional-noise', dest='regional_noise', action='store_false', help='deterministic proportional swing per province (no M9 shocks)')
    parser.add_argument('--out', default=os.path.join(ROOT, 'backtest', 'results'))
    args = parser.parse_args()

    from mtpy import mtpy
    mtpy.run()

    from mtpy.lib.data import get_scopes

    composition = None if args.composition is None else (args.composition if args.composition == 'auto' else float(args.composition))

    if args.scopes is None:
        scopes = [args.scope]
    elif args.scopes == ['all']:
        catalogue = get_scopes()
        scopes = catalogue.loc[catalogue['parent'].notnull()].index.tolist()
    else:
        scopes = args.scopes

    for scope in scopes:
        run_scope(args, scope, composition)


if __name__ == '__main__':
    main()
