#!/usr/bin/env python
"""
Batch load of the regional elections (scopes `es-*`) from the English Wikipedia: inventory of articles,
polls, results, computed series and ratings.

Usage (from the repository root):
    python load/run_load.py --scopes es-md es-cl | all --what urls polls results compute ratings
                            [--since 2009] [--save] [--overwrite] [--refresh] [--cached]

Without `--save` nothing is written to the database (dry run): the report lists, per scope and event, what
was read and the parties, pollsters and sponsors that are not mapped yet in `data/wikipedia/wp-maps.json`.
The report is saved to `files/stage/load_report.csv` and the downloaded articles are cached in
`files/stage/wikipedia/`. A dry run reads the cache (the curation of aliases is iterative); with `--save` the
articles are downloaded again unless `--cached` is given. Only regional scopes are accepted.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Optional

import pandas as pd
import requests
from lxml import html

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

WIKI = 'https://en.wikipedia.org/wiki/'
URLS = os.path.join(ROOT, 'data', 'wikipedia', 'wp-urls.json')
STEPS = ['urls', 'polls', 'results', 'compute', 'ratings']


def cache_dir() -> str:
    """Directory of the cache of downloaded articles, under the app file system root (`files/`)."""
    from mtpy.core.app import App

    return os.path.join(App.get_().fspath, 'stage', 'wikipedia')


def read_urls() -> dict:
    """Read `data/wikipedia/wp-urls.json`."""
    with open(URLS) as fh:
        return json.load(fh)


def dump_urls(urls: dict) -> str:
    """
    Serialise the inventory of articles as it is kept under version control: four spaces of indentation
    and the lists of years on one line.
    """
    text = json.dumps(urls, indent=4, ensure_ascii=False)

    def inline(match):
        return '[' + ', '.join(re.findall(r'"\d{4}"', match.group(0))) + ']'

    return re.sub(r'\[\s*(?:"\d{4}",?\s*)+\]', inline, text) + '\n'


def infobox_event(doc: html.HtmlElement) -> tuple[Optional[pd.Timestamp], Optional[str]]:
    """
    Date of the election of an article and address of the article of the previous one, from the header
    row of its infobox ("← 2023 8 February 2026 Next →").

    Returns
    -------
    tuple
        Date (`None` when the row has no date) and URL of the previous election (`None` when not linked).
    """
    for tr in doc.xpath("//table[contains(@class, 'infobox')]//tr"):
        text = ' '.join(tr.text_content().split())
        if not text.startswith('←'):
            continue

        date = re.search(r'(\d{1,2} \w+ \d{4})', text)
        date = pd.to_datetime(date.group(1), format='%d %B %Y') if date else None
        links = [a.get('href') for a in tr.xpath('.//a') if re.fullmatch(r'\d{4}', a.text_content().strip())]
        previous = links[0] if len(links) > 0 else None
        if previous is not None and previous.startswith('/wiki/'):
            previous = 'https://en.wikipedia.org' + previous

        return date, previous

    return None, None


def build_urls(scope: str, since: int = 2009, cache: Optional[str] = None, refresh: bool = False) -> dict[str, dict[str, str]]:
    """
    Inventory of the election articles of a regional scope: from `Next_{demonym}_regional_election` back
    through the "previous election" link of each infobox, down to the first election held in `since`.

    The date of the next event is the legal limit of the legislature given by the infobox; when the article
    of the next election does not exist yet (a community that has just voted), it is set four years after
    the last election. `polls` is only listed for the articles with a voting intention table.

    Parameters
    ----------
    scope : str
        Regional scope (`es-*`).
    since : int, optional
        First year of the inventory.
    cache : str, optional
        Directory of the cache of downloaded articles.
    refresh : bool, optional
        Ignore the cache.

    Returns
    -------
    dict
        `{event date: {'results': url, 'polls': url}}`, most recent first.
    """
    from mtpy.lib.data import get_scopes
    from mtpy.lib.loader import WikipediaLoader, fetch_page

    demonym = get_scopes().loc[scope, 'demonym']
    url_next = '{}Next_{}_regional_election'.format(WIKI, demonym)
    events = {}

    def read(url):
        doc = html.fromstring(fetch_page(url, cache, refresh))
        date, previous = infobox_event(doc)
        entry = {'results': url}
        if WikipediaLoader.find_poll_table(doc) is not None:
            entry['polls'] = url

        return date, previous, entry

    try:
        date, url, entry = read(url_next)
        events[date.strftime('%Y-%m-%d')] = entry
    except requests.HTTPError:
        # No article for the next election yet: start at the last one held
        url = None
        for year in range(pd.Timestamp.now().year, pd.Timestamp.now().year - 5, -1):
            candidate = '{}{}_{}_regional_election'.format(WIKI, year, demonym)
            try:
                date, _, _ = read(candidate)
            except requests.HTTPError:
                continue
            url = candidate
            events[(date + pd.DateOffset(years=4)).strftime('%Y-%m-%d')] = {'results': url_next}
            break

    while url is not None:
        date, previous, entry = read(url)
        if date is None or date.year < since:
            break
        events[date.strftime('%Y-%m-%d')] = entry
        url = previous

    return dict(sorted(events.items(), reverse=True))


def check_scopes(scopes: list[str], catalogue: pd.DataFrame) -> list[str]:
    """
    Validate the scopes of a batch load: only the regional ones of the catalogue (those with a parent). The
    national scope has its own loaders and curated inventory, which this script would overwrite.

    Raises
    ------
    ValueError
        When a scope is unknown or has no parent.
    """
    bad = [s for s in scopes if s not in catalogue.index or pd.isnull(catalogue.loc[s, 'parent'])]
    if len(bad) > 0:
        raise ValueError('run_load only handles regional scopes of data/es-scopes.csv, not {}'.format(bad))

    return list(scopes)


def merge_urls(existing: dict, rebuilt: dict, today: Optional[str] = None) -> tuple[dict, Optional[tuple[str, str]]]:
    """
    Merge a rebuilt inventory of a scope into the one already kept. The past events of both are kept (the
    rebuilt ones prevail). The date of an upcoming event already inventoried is respected, because it may
    have been corrected by hand and the polls stored are keyed by it; it takes the addresses just rebuilt.

    Returns
    -------
    tuple
        The merged inventory (most recent first) and, when the date of the upcoming event was kept instead
        of the rebuilt one, the pair `(kept, rebuilt)`; `None` otherwise.
    """
    today = today or pd.Timestamp.now().strftime('%Y-%m-%d')
    merged = {d: conf for d, conf in existing.items() if d <= today} | {d: conf for d, conf in rebuilt.items() if d <= today}

    kept = None
    old = sorted(d for d in existing if d > today)
    new = sorted(d for d in rebuilt if d > today)
    if len(old) > 0 and len(new) > 0 and old[0] != new[0]:
        merged[old[0]] = rebuilt[new[0]]
        kept = (old[0], new[0])
    elif len(new) > 0:
        merged[new[0]] = rebuilt[new[0]]
    elif len(old) > 0:
        merged[old[0]] = existing[old[0]]

    return dict(sorted(merged.items(), reverse=True)), kept


def effective_refresh(refresh: bool, save: bool, cached: bool) -> bool:
    """
    Whether the articles are downloaded again: always with `--refresh`, and by default when saving (a load
    must not write a stale copy without saying so) unless `--cached` asks for the cache.
    """
    return bool(refresh or (save and not cached))


def top(names: list[str]) -> list[str]:
    """Distinct names ordered by frequency, as `name (n)`."""
    return ['{} ({})'.format(name, n) for name, n in Counter(names).most_common()]


def load_polls(
    scope: str, event_date: str, save: bool, overwrite: bool, cache: Optional[str], verbose: int, refresh: bool = False
) -> dict:
    """
    Read the polls of one event and, with `save`, write them to the database.

    Returns
    -------
    dict
        Report columns: `polls_read`, `polls_saved` and the names without mapping.
    """
    from mtpy.lib.loader import WikipediaLoader

    loader = WikipediaLoader(scope, event_date, verbose=max(verbose - 1, 0))
    loader.cache_dir, loader.refresh = cache, refresh
    loader.read_data()

    out = {
        'polls_read': len(loader.data), 'polls_saved': 0,
        'parties_missing': top(loader.parties_missing),
        'pollsters_missing': top(loader.pollsters_missing),
        'sponsors_missing': top(loader.sponsors_missing)
    }
    if len(loader.data) == 0 or not save:
        return out

    loader.build_series()
    loader.select_series(overwrite=overwrite)
    out['polls_saved'] = int(loader.polls.shape[0])
    loader.save_polls(overwrite=overwrite).save_results(overwrite=overwrite)

    return out


def load_results(scope: str, event_date: str, save: bool, cache: Optional[str], verbose: int, refresh: bool = False) -> dict:
    """
    Read the results of one event (or the districts and seats of an upcoming one) and, with `save`, write
    its rows of `events`, `events_data` and `events_results`.

    Returns
    -------
    dict
        Report columns: `candidacies_missing` (unmapped candidacies with seats or at least 1 %), `votes_diff`
        (votes of the candidacies minus the valid votes net of blank ballots), `seats` and `district_seats`.
    """
    from mtpy.lib.loader import WikipediaResultsLoader

    loader = WikipediaResultsLoader(scope, event_date, verbose=max(verbose - 1, 0))
    loader.cache_dir, loader.refresh = cache, refresh
    loader.read_data().build_series()

    total = loader.totals.loc[loader.totals['region_id'] == 0].iloc[0]
    out = {'candidacies_missing': top(loader.parties_missing), 'seats': int(total['seats'])}
    if loader.results.shape[0] > 0:
        res = loader.results
        out['votes_diff'] = int(res.loc[res['region_id'] == 0, 'votes'].sum() - (total['votes'] - total['blank']))
        out['district_seats'] = int(res.loc[res['region_id'] > 0, 'seats'].sum())

    if save:
        loader.save_event().save_totals().save_results()

    return out


def compute_scope(scope: str, save: bool, verbose: int) -> None:
    """
    Compute the poll series of a scope: weights, featured events, errors, deviations, drift, house effects and
    herding (the sequence of `notebooks/data-load/PollsCompute.ipynb` without the ratings). The weights are computed
    twice because `polls.featured` copies the flag of the event, which in turn needs the weights.
    """
    from mtpy.lib.computer import Computer
    from mtpy.lib.data import update_featured

    Computer(scope=scope, verbose=max(verbose - 1, 0)).build_series().compute_weights(save=save, overwrite=True)
    if save:
        n = update_featured(scope)
        if verbose > 0:
            print('{}: {} featured events'.format(scope, n))

    comp = Computer(scope=scope, verbose=max(verbose - 1, 0)).build_series()
    comp.compute_weights(save=save, overwrite=True)
    comp.compute_errors(save=save)
    comp.compute_deviations(save=save)
    if not save:
        return

    # Drift and house effects read the errors stored by the steps above: a fresh series is needed
    comp = Computer(scope=scope, verbose=max(verbose - 1, 0)).build_series()
    comp.compute_drift(save=save)
    comp.compute_house_effects(save=save)
    comp.compute_herding(save=save)


def rate_scope(scope: str, save: bool, verbose: int) -> None:
    """Compute the cross-scope pollster ratings of the events of a scope (see `Computer.compute_ratings`)."""
    from mtpy.lib.computer import Computer

    Computer(scope=scope, verbose=max(verbose - 1, 0)).build_series().compute_ratings(save=save, scopes='all')


def run_load(
    scopes: list[str],
    what: list[str],
    since: int = 2009,
    save: bool = False,
    overwrite: bool = False,
    refresh: bool = False,
    cached: bool = False,
    verbose: int = 1
) -> pd.DataFrame:
    """
    Run the steps of `what` (in the order of `STEPS`) for every scope.

    Parameters
    ----------
    scopes : list of str
        Regional scopes to load.
    what : list of str
        Steps: `urls` (rebuild the inventory of articles in `wp-urls.json`), `polls`, `results`, `compute`
        (weights, errors, deviations, featured events, drift and house effects) and `ratings`.
    since : int, optional
        First year of the events loaded.
    save : bool, optional
        Write to the database; a dry run otherwise.
    overwrite : bool, optional
        Replace the polls already stored for the event instead of adding only the new ones.
    refresh : bool, optional
        Download the articles again instead of using the cache.
    cached : bool, optional
        With `save`, use the cached articles instead of downloading them again (see `effective_refresh`).
    verbose : int, optional
        Level of verbosity.

    Returns
    -------
    pd.DataFrame
        One row per scope and event with what was read and saved and what is left to map.
    """
    unknown = [w for w in what if w not in STEPS]
    if len(unknown) > 0:
        raise ValueError('Unknown steps: {}'.format(unknown))

    from mtpy.lib.data import get_scopes

    scopes = check_scopes(scopes, get_scopes())
    cache = cache_dir()
    refresh = effective_refresh(refresh, save, cached)

    if 'urls' in what:
        urls = read_urls()
        for scope in scopes:
            if verbose > 0:
                print('Inventory of {}...'.format(scope))
            urls[scope], kept = merge_urls(urls.get(scope, {}), build_urls(scope, since=since, cache=cache, refresh=refresh))
            if kept is not None:
                print('{}: upcoming event kept at {} (the article now says {}); review it by hand'.format(scope, *kept))
        with open(URLS, 'w') as fh:
            fh.write(dump_urls(urls))

    urls = read_urls()
    rows = []
    for scope in scopes:
        for event_date in sorted(urls.get(scope, {})):
            if int(event_date[:4]) < since:
                continue

            row = {'scope': scope, 'event_date': event_date}
            if 'polls' in what:
                if verbose > 0:
                    print('Polls of {} {}...'.format(scope, event_date))
                row.update(load_polls(scope, event_date, save, overwrite, cache, verbose, refresh))
            if 'results' in what:
                if verbose > 0:
                    print('Results of {} {}...'.format(scope, event_date))
                row.update(load_results(scope, event_date, save, cache, verbose, refresh))
            rows.append(row)

        if 'compute' in what:
            if verbose > 0:
                print('Compute {}...'.format(scope))
            compute_scope(scope, save, verbose)

    # Ratings last: every event sees the polls of all the scopes held before it
    if 'ratings' in what:
        for scope in scopes:
            if verbose > 0:
                print('Ratings of {}...'.format(scope))
            rate_scope(scope, save, verbose)

    return pd.DataFrame(rows)


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description='Batch load of the regional elections from Wikipedia')
    parser.add_argument('--scopes', nargs='+', required=True, help="regional scopes (es-md es-cl ...) or 'all'")
    parser.add_argument('--what', nargs='+', required=True, choices=STEPS)
    parser.add_argument('--since', type=int, default=2009)
    parser.add_argument('--save', action='store_true', help='write to the database (dry run otherwise)')
    parser.add_argument('--overwrite', action='store_true', help='replace the polls already stored')
    parser.add_argument('--refresh', action='store_true', help='download the articles again')
    parser.add_argument('--cached', action='store_true', help='with --save, use the cached articles instead of downloading them again')
    args = parser.parse_args()

    from mtpy import mtpy
    app = mtpy.run()

    from mtpy.lib.data import get_scopes

    scopes = get_scopes()
    scopes = scopes.loc[scopes['parent'].notnull()].index.tolist() if args.scopes == ['all'] else args.scopes

    report = run_load(
        scopes, args.what, since=args.since, save=args.save, overwrite=args.overwrite, refresh=args.refresh,
        cached=args.cached
    )

    out = os.path.join(app.fspath, 'stage', 'load_report.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    report.to_csv(out, index=False)

    pd.set_option('display.width', 250, 'display.max_colwidth', 90, 'display.max_rows', 500)
    print(report.to_string(index=False))


if __name__ == '__main__':
    main()
