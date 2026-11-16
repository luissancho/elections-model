# Web: títulos de las páginas y `<h1>` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every page has a title that leads with what the page is and ends with the brand, and exactly one `<h1>`, its own.

**Architecture:** `base.html` composes `<title>` from the `title` block plus a `title_brand` block (the home empties it and writes the brand first) and gives every page a screen-reader-only `<h1>` with the same text (`heading` block), which the pages with a visible `<h1>` empty. The header wordmark stops being a heading. Nothing visible changes.

**Tech Stack:** Jinja2 templates, the project's CSS, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-16-web-titulos-h1-design.md`

## Global Constraints

- PEP8, lines ≤ 120, numpydoc docstring on every function; identifiers and comments in neutral English, public texts in Spanish («sólo» with tilde, «encuestadora», never «empresa»).
- Web code lives inside mtpy; no standalone scripts. Tests never touch the RDS nor write to `backtest/results`.
- Luis commits in parallel in the same tree: before each commit `git status` and `git log -1`, add only the files of the task, never `git commit -a`.
- Commit messages in Spanish, in the style of the log, ending with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`. No push.
- `<title>` is `{title} · Demoscopio`; the home is exactly `Demoscopio · Promedio de encuestas electorales en España`; the error page, `Error {status} · Demoscopio`.
- `og:title` is the `title` block, without the brand (the home's block carries it).
- The titles of the spec table, verbatim.
- No visible change: the header and the pages look the same before and after (checked with screenshots in Task 2).

## Review Focus

- A template that extends `base.html` and has a visible `<h1>` but forgets to empty `heading` (two `<h1>`), or empties it without having one (none). (Task 2, `test_templates_empty_the_hidden_heading_exactly_when_they_show_one`)
- A scope without a known election date: the title of the event page goes without the year, not with «None». (Task 1, `test_event_title_without_a_date_has_no_year`)
- A pollster without rating (aggregator, online panel): its title must not promise a rating. (Task 1, `TITLES['/casas/agregador']`)
- A name with a character HTML reserves (`&`, `"`): escaped once in `<title>`, in `og:title` and in the hidden `<h1>`. (Task 2, `test_reserved_characters_are_escaped_once_in_title_og_and_heading`)
- A regional scope: «elecciones autonómicas {nombre}», without the middle dot of the event bar. (Task 1, `test_event_phrase_names_the_election_in_running_text`)

---

### Task 1: titles

**Files:**
- Create: `tests/test_pages_titles.py`
- Modify: `web/templates/base.html` (`<title>`, `og:title`), `web/templates/error.html` (`<title>`)
- Modify: `web/templates/{index,promedios,sondeo,elecciones,eleccion,casas,casa,metodo,analisis}.html` (block `title`; `index.html` also `title_brand`)
- Modify: `mtpy/lib/pages/common.py` (`event_phrase`, key `phrase` of `event_info`), `mtpy/lib/pages/__init__.py` (export)
- Modify: `tests/test_pages.py`, `tests/test_pages_consent.py` (the title and `og:title` assertions)

**Interfaces:**
- Produces: `pages.event_phrase(scope: str, scopes: list) -> str` — `'elecciones generales'` for `es`, otherwise `'elecciones autonómicas {name}'` (the code when the scope is not in `scopes`); `event_info(...)['phrase']`. Blocks of `base.html`: `title` (page text), `title_brand` (default ` · {{ site_title }}`).

- [ ] **Step 1: Write the failing tests** in `tests/test_pages_titles.py`: module constants `BRAND = ' · Demoscopio'`, `HOME_TITLE`, `TITLES` (path → full `<title>`, one per row of the spec table plus the unchanged ones of the fixture: the four method subpages, `/analisis/voto-cera`, `/privacidad`); helpers `title_of(page)`, `og_title_of(page)`; tests:
  - `test_titles_say_what_each_page_is(houses, fresh_app)`: every `TITLES` path answers 200 with exactly that `<title>`.
  - `test_every_title_ends_with_the_brand_and_og_title_drops_it(houses, fresh_app)`: for every sitemap path plus `/privacidad`: the home is `HOME_TITLE` and its `og:title` the same; the rest end with `BRAND`, do not start with `Demoscopio`, and `og:title == title[:-len(BRAND)]`; no title repeats.
  - `test_event_phrase_names_the_election_in_running_text()`: `'elecciones generales'`, `'elecciones autonómicas Madrid'`, `'elecciones autonómicas es-xx'`; `event_info('es-md', …)['phrase']`.
  - `test_event_title_without_a_date_has_no_year(...)`: the event page of a run whose meta has no `event_date` is titled `Encuestas elecciones generales: promedio y escaños · Demoscopio`.
  - `test_error_page_title_ends_with_the_brand(api)`: 404 → `<title>Error 404 · Demoscopio</title>`.

- [ ] **Step 2: Run** `python -m pytest -q -p no:cacheprovider tests/test_pages_titles.py` — Expected: FAIL (titles start with the brand; `event_phrase` missing).

- [ ] **Step 3: Implement** `event_phrase` and the `phrase` key in `common.py`, export it; `base.html`: `<title>{% block title %}{% endblock %}{% block title_brand %} · {{ site_title }}{% endblock %}</title>` and `og:title` = `{{ self.title() }}`; `error.html`; the nine title blocks.

- [ ] **Step 4: Update the old assertions** (`grep -n "Demoscopio ·" tests/`) to the new titles.

- [ ] **Step 5: Run** `python -m pytest -q -p no:cacheprovider tests/test_pages_titles.py tests/test_pages.py tests/test_pages_consent.py tests/test_pages_crawlers.py` — Expected: PASS.

- [ ] **Step 6: Commit.**

### Task 2: one `<h1>` per page

**Files:**
- Modify: `web/templates/base.html` (block `heading`), `web/templates/_brand.html`, `web/templates/error.html`
- Modify: `web/templates/{index,analisis,analisis_andorra,analisis_rebano,analisis_sesgo_cis,analisis_voto_cera,analisis_voto_oculto,metodo_promedio,metodo_escanos,metodo_rating}.html` (empty `heading`)
- Modify: `web/dist/css/site.css` (`#site-header h1` → `#site-header .wordmark`)
- Modify: `tests/test_pages_titles.py`, `tests/test_pages.py` (the wordmark assertion), `deploy/README.md`

**Interfaces:**
- Consumes: the `title` block of Task 1.
- Produces: block `heading` of `base.html`, default `<h1 class="sr-only">{{ self.title() }}</h1>`, between the header and the event bar.

- [ ] **Step 1: Write the failing tests** in `tests/test_pages_titles.py`:
  - `test_every_page_has_a_single_h1_that_is_not_the_wordmark(houses, fresh_app)`: every sitemap path plus `/privacidad`: one `<h1`, the wordmark is `<p class="wordmark"><a href="/"><img src="/dist/img/logo.svg" alt="Demoscopio" width="574" height="100"></a></p>`.
  - `test_hidden_heading_carries_the_title_before_the_event_bar(houses, fresh_app)`: `/es` → `</header>`, `<h1 class="sr-only">Encuestas elecciones generales 2026: promedio y escaños</h1>`, `<div id="event-bar"` in that order with only whitespace between; `/casas` → the same before `<main id="main">`.
  - `test_pages_with_a_visible_heading_do_not_get_the_hidden_one(houses, fresh_app)`: `/`, `/analisis`, `/analisis/voto-cera`, `/metodo/promedio`: no `sr-only` `<h1>`, their visible one present.
  - `test_templates_empty_the_hidden_heading_exactly_when_they_show_one()`: static, over every template that extends `base.html`.
  - `test_reserved_characters_are_escaped_once_in_title_og_and_heading(...)`: a pollster named `Sondeos & Datos` → `Sondeos &amp; Datos: …` in `<title>`, `og:title` and `<h1>`, never `&amp;amp;`.
  - `test_error_page_has_its_status_as_heading(api)`: one `<h1 class="sr-only">Error 404</h1>`.
  - `test_header_css_styles_the_wordmark_not_a_heading()`: `site.css` has no `#site-header h1` and has `#site-header .wordmark`.

- [ ] **Step 2: Run** the file — Expected: the new tests FAIL (two `<h1>` or the logo as the only one).

- [ ] **Step 3: Implement**: `_brand.html` (`<p class="wordmark">`), the CSS rename (four rules, same declarations), `heading` block in `base.html`, the ten templates that empty it, `error.html`.

- [ ] **Step 4: Run** `python -m pytest -q -p no:cacheprovider tests/test_pages_titles.py tests/test_pages.py tests/test_pages_ejemplo.py tests/test_deploy_files.py` — Expected: PASS after updating the wordmark assertion of `tests/test_pages.py`.

- [ ] **Step 5: Visual check**: render `/`, `/es`, `/casas`, a poll and the 404 page from the fixture at the commit before the task and after it, screenshot both at 1280 and 390 px wide with headless Chrome and compare pixels — Expected: no difference.

- [ ] **Step 6: Document** in `deploy/README.md` (the paragraph of the head metadata): how `<title>`, `og:title` and the `<h1>` are composed, and what a new template must do.

- [ ] **Step 7: Run the whole suite** on a clean copy of the commit — Expected: PASS. **Commit.**
