#!/usr/bin/env python
"""
Compare two backtest results folders case by case and print the summary as a markdown table.

Usage (from the repository root):
    python backtest/compare_results.py A B [--scopes es es-md ...] [--label-a "con rating"] [--label-b "plano"]

A and B are results folders (`backtest/results_xxx`): the national scope `es` has its `metrics.csv` at the root and
every other scope in a subfolder. The cases present in both (same scope, election and horizon) are paired, and per
population (generales, autonómicas, todas) and metric the table gives the means of each folder, their difference and
the number of cases where B is better / worse than A (ties in brackets). A scope without `metrics.csv` is skipped
with a warning. By default the scopes are `es es-md es-an es-ct es-vc es-ga`.

This script is a thin wrapper of `mtpy.lib.backtest.load_results`, `pair_results`, `compare_paired` and
`compare_markdown`.
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

DEFAULT_SCOPES = ['es', 'es-md', 'es-an', 'es-ct', 'es-vc', 'es-ga']


def main():
    parser = argparse.ArgumentParser(description='Pair two backtest results folders and compare them')
    parser.add_argument('a', help='reference results folder (A)')
    parser.add_argument('b', help='candidate results folder (B)')
    parser.add_argument('--scopes', nargs='+', default=DEFAULT_SCOPES, help='scopes to compare')
    parser.add_argument('--label-a', default=None, help='name of A in the table (default: its folder name)')
    parser.add_argument('--label-b', default=None, help='name of B in the table (default: its folder name)')
    args = parser.parse_args()

    from mtpy.lib.backtest import compare_markdown, compare_paired, load_results, pair_results

    paired = pair_results(load_results(args.a, args.scopes), load_results(args.b, args.scopes))
    label_a = args.label_a or os.path.basename(os.path.normpath(args.a))
    label_b = args.label_b or os.path.basename(os.path.normpath(args.b))
    print(compare_markdown(compare_paired(paired), label_a, label_b))


if __name__ == '__main__':
    main()
