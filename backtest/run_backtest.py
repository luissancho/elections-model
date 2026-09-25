#!/usr/bin/env python
"""
Run the model backtest on past elections and write the results to `backtest/results/`.

Usage (from the repository root):
    python backtest/run_backtest.py [--events 2019-04-28 2023-07-23] [--horizons 6 30] [--n-sim 500] [--seed 42] [--nowcast-only]
                                    [--no-house-effects] [--industry-bias] [--composition auto|0.25|1]
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description='Backtest of the elections model on past elections')
    parser.add_argument('--scope', default='es')
    parser.add_argument('--events', nargs='*', default=None)
    parser.add_argument('--horizons', nargs='*', type=int, default=None)
    parser.add_argument('--n-sim', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-fc', type=int, default=10)
    parser.add_argument('--nowcast-only', action='store_true', help='skip the horizon run of each case (columns `_h`)')
    parser.add_argument('--no-house-effects', dest='house_effects', action='store_false', help='do not subtract the house effects (M6)')
    parser.add_argument('--industry-bias', action='store_true', help='shift the average by the industry-wide bias of past elections (M6)')
    parser.add_argument('--composition', default=None, help="joint noise of the national parties (M7): 'auto' or a ratio in (0, 1]; independent draws by default")
    parser.add_argument('--out', default=os.path.join(ROOT, 'backtest', 'results'))
    args = parser.parse_args()

    from mtpy import mtpy
    mtpy.run()

    from mtpy.lib.backtest import run_backtest
    from mtpy.models import elections as models

    composition = None if args.composition is None else (args.composition if args.composition == 'auto' else float(args.composition))

    results = run_backtest(
        scope=args.scope, events=args.events, horizons=args.horizons,
        n_sim=args.n_sim, seed=args.seed, max_fc=args.max_fc, nowcast_only=args.nowcast_only,
        house_effects=args.house_effects, industry_bias=args.industry_bias, composition=composition, verbose=1
    )

    os.makedirs(args.out, exist_ok=True)
    for key, df in results.items():
        df.to_csv(os.path.join(args.out, '{}.csv'.format(key)), index=False)

    polls = models.Polls().get_results(formatted=True)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT).decode().strip()
    except Exception:
        commit = None
    meta = {
        'run_at': datetime.now().isoformat(timespec='seconds'), 'commit': commit,
        'n_sim': args.n_sim, 'seed': args.seed, 'max_fc': args.max_fc, 'nowcast_only': args.nowcast_only,
        'house_effects': args.house_effects, 'industry_bias': args.industry_bias, 'composition': composition,
        'events': results['metrics']['event_date'].unique().tolist(), 'horizons': sorted(results['metrics']['horizon'].unique().tolist()),
        'db_polls': int(polls.shape[0]), 'db_last_poll': str(polls['date'].max().date())
    }
    with open(os.path.join(args.out, 'run.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)

    print(results['by_horizon'].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
