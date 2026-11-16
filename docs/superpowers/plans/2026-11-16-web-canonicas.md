# Web: la URL canónica de cada página — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every page declares one canonical URL and one `og:url`: its own path, with no query string.

**Architecture:** `Page.canonical_url()` stops copying the query of the request: it returns the public origin plus the path. The controllers that pass their own clean canonical keep doing so. `base.html` gains `twitter:creator`.

**Tech Stack:** the page controllers of `mtpy/controllers`, Jinja2, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-16-web-canonicas-design.md`

## Global Constraints

- PEP8, lines ≤ 120, numpydoc docstring on every function; identifiers and comments in neutral English, public texts in Spanish.
- Web code lives inside mtpy; no standalone scripts. Tests never touch the RDS nor write to `backtest/results`.
- Luis commits in parallel in the same tree: before each commit `git status` and `git log -1`, add only the files of the task, never `git commit -a`.
- Commit messages in Spanish, in the style of the log, ending with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`. No push.
- `<link rel="canonical">` and `og:url` carry the same URL: `web.site_url` (without its trailing slash) + `/` + the path of the request.
- The internal links keep their `?scope=`; nothing visible changes.

## Review Focus

- A tracking parameter (`utm_source`, `fbclid`, `gclid`) on every kind of page: the canonical and `og:url` stay clean. (Task 1, `test_every_sitemap_page_declares_its_path_whatever_the_query`)
- `?scope=` on a global page (`/casas`, `/elecciones`, `/metodo/*`, `/analisis/*`, `/privacidad`): the canonical is the path. (Task 1, same test and the two older ones)
- A page that still passes its own canonical (profile, past election, poll) asked through a non-clean path (`/casas/2`, `/es/sondeos/2026-10-05/7`, `/elecciones/es/generales/2023-07-23`): its canonical stays the clean one. (Task 1, `test_page_asked_by_an_id_declares_its_clean_url` and `test_profile_and_detail_canonicals_are_the_clean_urls`)
- A regional scope on the two global pages whose cards change with it (`/casas`, `/metodo/backtest`): the canonical is the national page. (Task 1, `test_regional_scope_of_a_global_page_declares_the_path`)
- `web.site_url` configured with a trailing slash: no doubled slash. (Task 1, `test_canonical_is_the_path_without_the_query`)
- A query value with reserved characters (`"`, `<`, `&`): nothing of it reaches the head. (Task 1, `test_no_query_value_reaches_the_head`)

---

### Task 1: the canonical is the path

**Files:**
- Modify: `mtpy/controllers/Page.py` (`canonical_url`, docstring of `render`), `mtpy/controllers/Promedios.py` (default canonical), `web/templates/base.html` (`twitter:creator`)
- Modify: `tests/test_pages_crawlers.py` (new tests), `tests/test_pages.py` (two older tests and the head assertions), `deploy/README.md`

**Interfaces:**
- Produces: `Page.canonical_url(self) -> str` — `'<site_url>/<path>'`, no parameters.

- [ ] **Step 1: Write the failing tests**: in `tests/test_pages_crawlers.py`, `test_every_sitemap_page_declares_its_path_whatever_the_query(houses, fresh_app)` (every sitemap URL and `/privacidad` with `scope=es&utm_source=x&utm_medium=social&fbclid=abc&gclid=1`: 200, canonical and `og:url` equal to the clean URL) and `test_no_query_value_reaches_the_head(houses, fresh_app)` (`/metodo` with `x="><script>alert(1)</script>&y=a%26b`: the head has no `script>alert`, no `x=` and the clean canonical); in `tests/test_pages.py`, rename the older test to `test_canonical_is_the_path_without_the_query` and turn `/metodo?scope=es` and `/casas?scope=es` into the clean URLs; assert `<meta name="twitter:creator" content="@luis_sancho">` next to `twitter:card`.
- [ ] **Step 2: Run** `python -m pytest -q -p no:cacheprovider tests/test_pages_crawlers.py tests/test_pages.py` — Expected: the new and the changed tests FAIL (the canonical keeps the query; no `twitter:creator`).
- [ ] **Step 3: Implement** `canonical_url` without the query, drop the explicit canonical of `Promedios`, add the tag.
- [ ] **Step 4: Update** `deploy/README.md` (the two sentences on the canonical).
- [ ] **Step 5: Run** `python -m pytest -q -p no:cacheprovider tests/test_pages_crawlers.py tests/test_pages.py tests/test_pages_titles.py tests/test_pages_descriptions.py tests/test_pages_consent.py tests/test_pages_ejemplo.py` — Expected: PASS. Then the whole suite on a clean copy of the commit.
- [ ] **Step 6: Commit.**
