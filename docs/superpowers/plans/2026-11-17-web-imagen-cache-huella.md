# The day of cache of a link image only for the declared address — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The image of a link preview answers a day of cache only when the `v` asked is the fingerprint of what the server draws; otherwise the cache time of the pages.

**Architecture:** The rule lives in `Page.png`, which both image actions (`Sondeos.image_action`, `Elecciones.image_action`) already use: they pass the `version` of the preview they drew and `png` compares it with the `v` of the request.

**Tech Stack:** Python 3.11, the project's own controllers (`mtpy/core/api.py`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-17-web-imagen-cache-huella-design.md`

## Global Constraints

- PEP8, lines ≤ 120, a numpydoc docstring on every function; identifiers and comments in English, public texts and test docstrings in Spanish.
- The long time is `Page.IMAGE_TTL` (86400); the short one is `int(webapi.settings()['cache_ttl'])` (60 by default), as `Page.render` sets it. No new constant.
- Nothing else changes: the PNG bytes, the ETag, `image/png`, the 400/404 error pages with `no-store`, and that the query does not change the drawing.
- Tests never touch the network or a database. Commit only the files of the task, with `git add <files>` (the author commits in parallel in this tree); message in Spanish in the style of the log, ending with the `Co-Authored-By` line.

## Review Focus

- A request without `v`, with an empty `v` or with `v` given twice: the short time. (Task 1, `test_poll_image_keeps_a_day_only_under_its_fingerprint`)
- The right `v` next to other parameters (`utm_source`, `mode`, `run`): still a day. (Task 1, same test)
- The address each page declares in its `og:image`, asked as declared: a day — poll, election, and the page with a pinned run. (Task 1, `test_declared_image_addresses_keep_a_day`)
- The `v` of a drawing that is no longer the current one (an old link, or a worker behind): the current image, the short time, its ETag. (Task 1, `test_election_image_under_an_old_fingerprint_keeps_a_minute`)
- A `cache_ttl` other than 60, and a HEAD request: the short time follows the setting, and HEAD answers the same header as GET. (Task 1, `test_poll_image_keeps_a_day_only_under_its_fingerprint`)

---

### Task 1: `Page.png` gives the day only under the fingerprint

**Files:**
- Modify: `mtpy/controllers/Page.py` (`png`, docstring of `IMAGE_TTL`), `mtpy/controllers/Sondeos.py` (`image_action`), `mtpy/controllers/Elecciones.py` (`image_action`)
- Modify: `deploy/README.md` (the two paragraphs on the images: the day only under the declared address)
- Test: `tests/test_pages_images.py`, `tests/test_pages_event_images.py`

**Interfaces:**
- Produces: `Page.png(self, image: bytes, version: str) -> bytes` — sets `image/png`, the ETag and `public, max-age=<IMAGE_TTL>` when `self.request.query.get('v') == version`, else `public, max-age=<cache_ttl>`.
- Consumes: `preview['version']` of `pages.poll_preview(...)` and `pages.event_preview(...)` (the fingerprint the page declares through `Page.preview_image`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_pages_images.py` (constants `IMAGE_PATH`, `POLL_PATH`, `SITE`, helpers `page_of`, `meta_of`; fixture `houses`):

```python
def test_poll_image_keeps_a_day_only_under_its_fingerprint(houses, fresh_app, monkeypatch):
    version = pages.latest_poll_preview('es', '2026-10-05', 'sigma-dos')['version']
    day, minute = 'public, max-age=86400', 'public, max-age=60'
    _, headers, body = call(houses, IMAGE_PATH, query='v=' + version)
    assert headers['cache-control'] == day
    for query in ('v={}&utm_source=x&mode=forecast&run=nope'.format(version), 'zz=1&v=' + version):
        status, headers, other = call(houses, IMAGE_PATH, query=query)
        assert status == 200 and headers['cache-control'] == day and other == body, query
    for query in ('', 'v=', 'v=abc', 'v=' + version.upper() + 'x', 'v=abc&v=' + version, 'V=' + version,
                  'utm_source=x'):
        status, headers, other = call(houses, IMAGE_PATH, query=query)
        assert status == 200 and headers['cache-control'] == minute and other == body, query
        assert headers['etag'] == '"{}"'.format(hashlib.md5(body).hexdigest())
    assert call(houses, IMAGE_PATH, query='v=' + version, method='HEAD')[1]['cache-control'] == day
    assert call(houses, IMAGE_PATH, method='HEAD')[1]['cache-control'] == minute
    # the short time is the one of the pages
    monkeypatch.setitem(webapi.settings(), 'cache_ttl', 300.0)   # or the way the tests of the pages set it
    assert call(houses, IMAGE_PATH)[1]['cache-control'] == 'public, max-age=300'
    assert call(houses, IMAGE_PATH, query='v=' + version)[1]['cache-control'] == day
```

In `tests/test_pages_event_images.py`:

```python
def test_declared_image_addresses_keep_a_day(houses, fresh_app):
    # the og:image of the poll page, of the election page and of the election page with a pinned run
    for path, query in ((POLL_PATH, ''), (EVENT_PATH, ''), (EVENT_PATH, 'run=' + RUN)):
        declared = meta_of(page_of(houses, path, query), 'og:image')
        assert declared.startswith(SITE) and '.png?v=' in declared
        address, _, asked = declared[len(SITE):].partition('?')
        status, headers, _ = call(houses, address, query=asked)
        assert status == 200 and headers['cache-control'] == 'public, max-age=86400', declared


def test_election_image_under_an_old_fingerprint_keeps_a_minute(houses, fresh_app):
    # the fingerprint of the same election before its previous one was published: an old link, or a worker behind
    old = pages.latest_event_preview('es', '2023-07-23')['version']
    write_event(fresh_app.fs, date='2019-11-10', model=False)   # before the first read of the list of elections
    ...
```

The second test must end with: the image asked with the old fingerprint is the current drawing (`pages.images.result_png(pages.latest_event_preview(...))`), with `public, max-age=60` and its ETag, and asked with the current fingerprint keeps `public, max-age=86400`. If the site caches the list of elections so that the order above cannot be used, build the old fingerprint with `elecciones.result_preview(..., None, HOST)` as `test_election_image_version_follows_the_result_before` does.

Update the two existing tests that expect a day without `v` (`test_poll_image_route_answers_a_cacheable_png`, `test_event_image_route_answers_a_cacheable_png`): they ask with the fingerprint, and their loop over odd queries keeps comparing status, type and bytes.

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_pages_images.py tests/test_pages_event_images.py -q`
Expected: the new tests FAIL on `cache-control` (`public, max-age=86400` where a minute is expected); `test_declared_image_addresses_keep_a_day` passes already (a guard).

- [ ] **Step 3: Implement `Page.png(self, image: bytes, version: str) -> bytes`** and pass `preview['version']` from both image actions (compute the preview, draw it, answer it).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pages_images.py tests/test_pages_event_images.py tests/test_pages.py tests/test_pages_descriptions.py tests/test_pages_previews.py tests/test_web_routes.py tests/test_deploy_files.py -q`
Expected: all pass.

- [ ] **Step 5: Update `deploy/README.md`** (the paragraph of the poll image and the one of the election image: the day of cache only under the address the page declares; any other address of the image, the minute of the pages) and the docstring of `IMAGE_TTL`.

- [ ] **Step 6: Commit**

```bash
git add mtpy/controllers/Page.py mtpy/controllers/Sondeos.py mtpy/controllers/Elecciones.py deploy/README.md tests/test_pages_images.py tests/test_pages_event_images.py
git commit
```
