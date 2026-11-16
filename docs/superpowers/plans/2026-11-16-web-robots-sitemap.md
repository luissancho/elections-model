# Web: `robots.txt` y `sitemap.xml` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The site answers `/robots.txt` and `/sitemap.xml`, the sitemap listing every canonical page that is published.

**Architecture:** Two routes of the page router answered by a new `Crawlers` controller; the texts are built in a new module `mtpy/lib/pages/crawlers.py` with the same link helpers the pages use (`event_href`, `pollster_slugs`, `poll_detail_href`, `study_cards`), so the sitemap never drifts from what is published. Nothing is stored: both answers are computed per request and cached one hour.

**Tech Stack:** Python 3.11, the project's own router and controllers (`mtpy/core/api.py`), pytest. No new dependency (`xml.sax.saxutils.escape` from the standard library).

**Spec:** `docs/superpowers/specs/2026-11-16-web-robots-sitemap-design.md`

## Global Constraints

- PEP8, lines ≤ 120, numpydoc docstring on every function; identifiers and comments in neutral English, public texts in Spanish.
- Web code lives inside mtpy: HTTP in `mtpy/controllers`, logic in `mtpy/lib/pages`. No standalone scripts.
- Tests never touch the RDS nor write to `backtest/results`.
- The working tree carries someone else's uncommitted work in `mtpy/lib/pages/common.py`, `mtpy/lib/pages/__init__.py`, `tests/test_pages.py` and `deploy/README.md` (and more). For those four files stage **only this plan's hunks**: build the blob «`HEAD` version + this plan's edit» in the scratchpad and put it in the index with `git update-index --cacheinfo 100644,$(git hash-object -w <blob>),<path>`; check that `git diff --cached <path>` shows only this plan's hunk. Never `git add` them whole, never `git commit -a`, never stash or revert. Run `git status --short` and `git log -1` before each commit.
- Commit messages in Spanish, in the style of the log, ending with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`. No push.
- `robots.txt` is exactly: `User-agent: *`, `Disallow: /api/`, an empty line, `Sitemap: <site_url>/sitemap.xml`.
- Both answers: ETag (MD5 of the UTF-8 body) and `cache-control: public, max-age=3600`.
- Sitemap: only canonical URLs, absolute with `site_url`, no query strings, no `<changefreq>` nor `<priority>`; `<lastmod>` only on `/` (`updated_at` of the manifest) and on `/{scope}` (its `run_at`).

## Review Focus

- A published scope whose latest run has no `polls` part: the sitemap still answers, without poll pages for it. (Task 2, `test_sitemap_skips_the_scopes_without_polls_part`)
- Two polls with the same day, house and sponsor, or a poll without a valid date: no URL (the page of the former answers 404; the plan first said «one URL», corrected after the final review). (Task 2, `test_poll_paths_list_only_the_polls_their_page_resolves`)
- A character that XML reserves in a URL (`&` in `site_url`): escaped. (Task 2, `test_sitemap_xml_escapes_the_urls`)
- Nothing published yet: `/sitemap.xml` is a 503, `/robots.txt` still a 200. (Task 2, `test_sitemap_without_bundle_or_database_is_a_503_and_robots_still_answers`)
- A path that only looks like the file (`/robotsxtxt`): a 404, because the dot of the route is escaped. (Task 1, `test_robots_route_takes_the_dot_literally`)

---

### Task 1: `/robots.txt`

**Files:**
- Create: `mtpy/lib/pages/crawlers.py`
- Create: `mtpy/controllers/Crawlers.py`
- Create: `tests/test_pages_crawlers.py`
- Modify: `mtpy/lib/pages/common.py` (`ROUTES`, after the `/analisis/{study}` route)
- Modify: `mtpy/lib/pages/__init__.py` (re-export)
- Modify: `tests/test_pages.py` (`test_pages_package_layout`: `'crawlers'` after `'common'`)

**Interfaces:**
- Produces: `pages.crawlers.CRAWLERS_TTL = 3600`, `SITEMAP_PATH = '/sitemap.xml'`, `DISALLOWED = ('/api/',)`, `robots_txt() -> str`; `Crawlers(Page)` with `respond(body: str, content_type: str) -> str` (sets the content type, the ETag and the cache; returns `body`) and `async robots_action() -> str`; routes `(r'/robots\.txt', 'crawlers', 'robots', ['GET'])` and `(r'/sitemap\.xml', 'crawlers', 'sitemap', ['GET'])`.

- [ ] **Step 1: Write the failing tests** in `tests/test_pages_crawlers.py`

```python
"""Tests de lo que la web dice a los rastreadores: `/robots.txt` y `/sitemap.xml`."""
from mtpy.core.app import Config
from mtpy.lib import pages
from tests.test_api_asgi import call
from tests.test_pages import api, html  # noqa: F401


def test_robots_txt_closes_the_api_and_names_the_sitemap(api, fresh_app):
    status, headers, body = call(api, '/robots.txt')
    assert status == 200 and headers['content-type'] == 'text/plain; charset=utf-8'
    assert headers['cache-control'] == 'public, max-age=3600' and headers['etag']
    assert html(body) == 'User-agent: *\nDisallow: /api/\n\nSitemap: https://demoscopio.es/sitemap.xml\n'
    fresh_app.set('config', Config({'web': {'site_url': 'https://demoscopio.dev/'}}))
    assert html(call(api, '/robots.txt')[2]).endswith('Sitemap: https://demoscopio.dev/sitemap.xml\n')


def test_robots_route_takes_the_dot_literally(api):
    status, headers, _ = call(api, '/robotsxtxt')
    assert status == 404 and headers['content-type'].startswith('text/html')
```

(Adapt the header access to what `call` returns if it is not a dict with lower-case keys; keep the assertions.)

- [ ] **Step 2: Run them** — `python -m pytest tests/test_pages_crawlers.py -q` — Expected: the first FAILS with a 404 (no route), the second passes already (today everything with a dot is a 404): it pins the escaped dot once the route exists.

- [ ] **Step 3: Implement.** `crawlers.py` with the constants and `robots_txt()` (reads `settings()['site_url']`); `Crawlers.py` with `respond` and `robots_action`; the two routes in `ROUTES` with a comment saying the dot is escaped because the patterns are regular expressions; the re-export (`CRAWLERS_TTL`, `robots_txt`) in `__init__.py`; `'crawlers'` in the module list of `test_pages_package_layout`.

- [ ] **Step 4: Run** — `python -m pytest tests/test_pages_crawlers.py tests/test_web_routes.py "tests/test_pages.py::test_pages_package_layout" -q` — Expected: all pass (`/sitemap.xml` has a route but no action yet: nothing asks for it).

- [ ] **Step 5: Commit** (shared files staged by blob, see Global Constraints).

### Task 2: `/sitemap.xml`, the docs and the deploy check

**Files:**
- Modify: `mtpy/lib/pages/crawlers.py`, `mtpy/controllers/Crawlers.py`, `mtpy/lib/pages/__init__.py`, `tests/test_pages_crawlers.py`
- Modify: `deploy/README.md` (a paragraph next to the canonical / `og:image` one), `deploy/check-api.sh` (two `check` lines)

**Interfaces:**
- Consumes: `Crawlers.respond`, `CRAWLERS_TTL`, `SITEMAP_PATH` (Task 1).
- Produces, in `pages.crawlers`: `SITEMAP_NS = 'http://www.sitemaps.org/schemas/sitemap/0.9'`; `scope_entries() -> list` (`/` and one per published scope, `{'path', 'lastmod'}`); `election_paths()`, `pollster_paths()`, `method_paths()`, `study_paths()`, `poll_paths() -> list of str`; `sitemap_entries() -> list of dict` (`{'path': str, 'lastmod': str or None}`, in the order of the spec); `sitemap_xml() -> str`. `Crawlers.sitemap_action() -> str` (`application/xml`).

- [ ] **Step 1: Write the failing tests** (added to `tests/test_pages_crawlers.py`; `houses` gives the fake database, the houses analysis and the 2023 election; `write_study` a published study)

```python
SITE = 'https://demoscopio.es'
NS = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}


def sitemap(api):
    """`{loc: lastmod o None}` del sitemap, en su orden, tras comprobar que responde como XML cacheable."""
    status, headers, body = call(api, '/sitemap.xml')
    assert status == 200 and headers['content-type'] == 'application/xml; charset=utf-8'
    assert headers['cache-control'] == 'public, max-age=3600' and headers['etag']
    assert body.startswith(b'<?xml version="1.0" encoding="UTF-8"?>\n')
    root = ET.fromstring(body)
    assert root.tag == '{%s}urlset' % NS['s']
    return {url.findtext('s:loc', namespaces=NS): url.findtext('s:lastmod', namespaces=NS) for url in root}


def test_sitemap_lists_the_published_pages_in_order(houses, fresh_app):
    write_study(fresh_app.fs)
    urls = sitemap(houses)
    paths = [loc[len(SITE):] for loc in urls]
    assert all(loc.startswith(SITE + '/') for loc in urls) and len(set(paths)) == len(paths)
    assert not any('?' in path or path.startswith('/api/') or path == '/privacidad' for path in paths)
    assert paths[:3] == ['/', '/es', '/elecciones']
    assert urls[SITE + '/'] == '2026-10-08T12:00:00Z' and urls[SITE + '/es'] == '2026-10-08T12:00:00Z'
    assert urls[SITE + '/elecciones'] is None
    expected = ['/elecciones', '/elecciones/es/generales/2023-07', '/casas', '/casas/agregador', '/casas/gad3',
                '/metodo', '/metodo/promedio', '/metodo/escanos', '/metodo/rating', '/metodo/backtest', '/analisis',
                '/analisis/voto-cera', '/es/sondeos/2026-10-05/sigma-dos']
    assert [path for path in paths if path in expected] == expected
    assert paths[-1] == '/es/sondeos/2026-10-05/sigma-dos'
    assert [p for p in paths if p.startswith('/casas/')] == sorted(p for p in paths if p.startswith('/casas/'))


def test_every_sitemap_url_is_a_live_page_and_its_own_canonical(houses, fresh_app):
    write_study(fresh_app.fs)
    for loc in sitemap(houses):
        status, _, body = call(houses, loc[len(SITE):])
        assert status == 200, loc
        assert '<link rel="canonical" href="{}">'.format(loc) in html(body), loc


def test_sitemap_without_bundle_or_database_is_a_503_and_robots_still_answers(api, fresh_app, tmp_path):
    status, headers, body = call(api, '/sitemap.xml')  # a bundle, no database
    assert status == 503 and headers['content-type'].startswith('text/html') and headers['cache-control'] == 'no-store'
    fresh_app.set('fs', FileSystem(str(tmp_path / 'empty')))  # nothing published
    assert call(api, '/sitemap.xml')[0] == 503 and call(api, '/robots.txt')[0] == 200


def test_sitemap_skips_the_scopes_without_polls_part(houses, fresh_app):
    os.remove(fresh_app.fs.path(bundle.path_part('es', RUN, 'polls')))  # adapt to how FileSystem resolves a path
    webapi.site().files.clear()  # adapt: drop the cached run files
    assert not any('/sondeos/' in loc for loc in sitemap(houses))


def test_poll_paths_are_unique_and_skip_the_polls_without_page(houses, monkeypatch):
    poll = {'date': '2026-10-05', 'pollster': 'Sigma Dos', 'pollster_id': 7, 'sponsor': 'El Mundo'}
    polls = {'polls': [poll, dict(poll), {'date': None, 'pollster': 'GAD3'}, {'date': '2026-10-06', 'pollster': None}]}
    monkeypatch.setattr(pages.crawlers, 'bundle_part', lambda *args: polls)
    assert pages.crawlers.poll_paths() == ['/es/sondeos/2026-10-05/sigma-dos/el-mundo']


def test_sitemap_xml_escapes_the_urls(houses, fresh_app):
    fresh_app.set('config', Config({'web': {'site_url': 'https://demoscopio.dev/a&b'}}))
    xml = pages.sitemap_xml()
    assert '<loc>https://demoscopio.dev/a&amp;b/es</loc>' in xml and 'a&b' not in xml
    ET.fromstring(xml)
```

(Imports to add: `os`, `xml.etree.ElementTree as ET`, `FileSystem`, `bundle`, `webapi`, `RUN`, `write_study`, `houses`, `houses_without_analysis`. `use_database` after the `Config` swap if the fake database hangs from the config: adapt the last test, keep its assertions.)

- [ ] **Step 2: Run** — `python -m pytest tests/test_pages_crawlers.py -q` — Expected: the six new tests FAIL (no `sitemap` action, no `poll_paths`, no `sitemap_xml`); the two of Task 1 pass.

- [ ] **Step 3: Implement** the functions of the Interfaces block and `sitemap_action`. Order of `sitemap_entries`: `scope_entries()`, then elections, pollsters, method, studies and polls, each path with `lastmod` `None`. `election_paths`: `/elecciones`, then, per catalogue scope with `site().events(code)`, `event_href(code, date, dates)` (after one `site().warm_events(codes)`, as `elecciones_context`). `pollster_paths`: `/casas`, then `/casas/{slug}` for the sorted slugs of `pollster_slugs()`. `method_paths`: `/metodo` and the `METODO_SECTIONS` with a template. `study_paths`: `/analisis` and the `href` of `study_cards()`. `poll_paths`: per published scope, `bundle_part(site().run_file, code, latest, 'polls')`, `poll_detail_href`, first occurrence only. `sitemap_xml`: one `<url>` per line, `loc` and `lastmod` through `xml.sax.saxutils.escape`.

- [ ] **Step 4: Run** — `python -m pytest tests/test_pages_crawlers.py tests/test_web_routes.py tests/test_deploy_files.py "tests/test_pages.py::test_pages_package_layout" -q` — Expected: all pass. If `test_every_sitemap_url_is_a_live_page_and_its_own_canonical` names a URL whose canonical differs, the sitemap must list the canonical one (rule on it and note it in the ledger).

- [ ] **Step 5: Docs and deploy check.** `deploy/README.md`: a short paragraph in «La web» (what the two files are, where they come from, the hour of cache, the 503 without database). `deploy/check-api.sh`: `check /robots.txt 200 none text/plain` and `check_db_only /sitemap.xml 200 none` (or the form its helpers allow), next to the page checks; `bash -n deploy/check-api.sh` — Expected: no output.

- [ ] **Step 6: Commit** (shared files staged by blob).
