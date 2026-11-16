# Web: datos estructurados — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The home says what the site is called and who publishes it, a study says it is an article with its author and date, and every second-level page says which section it belongs to, as schema.org JSON-LD.

**Architecture:** `mtpy/lib/pages/structured.py` builds each block as a plain dict from texts the page already shows; the page controller passes them as `structured` and `base.html` prints one `<script type="application/ld+json">` per block with Jinja's `tojson`.

**Tech Stack:** Jinja2 (`tojson`), the page controllers, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-16-web-datos-estructurados-design.md`

## Global Constraints

- PEP8, lines ≤ 120, numpydoc docstring on every function; identifiers and comments in neutral English, public texts in Spanish.
- Web code lives inside mtpy; no standalone scripts. Tests never touch the RDS nor write to `backtest/results`.
- Luis commits in parallel in the same tree: before each commit `git status` and `git log -1`, add only the files of the task, never `git commit -a`.
- Commit messages in Spanish, in the style of the log, ending with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`. No push.
- Every URL in a block is absolute and canonical: `web.site_url` + the clean path, no query.
- No new public text: names and descriptions are the ones the page already shows (menu labels, titles, descriptions, the footer's author).
- Nothing visible changes.

## Review Focus

- A name from the data with `</script>`, quotes or `&` (a pollster, an outlet, a study title): the block parses back to the exact name and cannot close the tag. (Task 1, `test_block_survives_reserved_characters`)
- A study whose manifest entry has no `published_at`, or a site without the entry: the article has no `dateModified` and the page renders. (Task 2, `test_study_without_a_publication_date_has_no_date`)
- A past election without a name: the breadcrumb names it by its date. (Task 2, `test_breadcrumbs_of_each_kind_of_page`)
- `web.site_url` with a trailing slash or another origin: every URL of every block follows it, without a doubled slash. (Task 2, `test_blocks_follow_the_configured_origin`)
- A page asked with `?scope=`, `utm_*` or by an id (`/casas/2`): the URLs in its blocks are the clean ones. (Task 2, `test_blocks_carry_canonical_urls_whatever_the_request`)

---

### Task 1: the builders and the base template

**Files:**
- Create: `mtpy/lib/pages/structured.py`, `tests/test_pages_structured.py`
- Modify: `mtpy/lib/pages/common.py` (`AUTHOR_NAME = 'Luis Sancho'`), `mtpy/lib/pages/__init__.py` (exports), `mtpy/controllers/Page.py` (`render`: `structured` defaults to `[]`), `web/templates/base.html`

**Interfaces:**
- Produces, in `mtpy.lib.pages.structured` (re-exported from `mtpy.lib.pages` as `pages.structured`):
  - `website(origin: str) -> dict` — `WebSite`: `name`, `url` (`origin + '/'`), `inLanguage` `'es'`, `description` (`SITE_TAGLINE`).
  - `organization(origin: str) -> dict` — `Organization`: `name`, `url`, `logo` (`origin + LOGO_PATH`, `LOGO_PATH = '/dist/img/apple-touch-icon.png'`), `description`, `founder` (`person()`).
  - `person() -> dict` — `Person`: `name` (`AUTHOR_NAME`), `url` (`AUTHOR_URL`); no `@context`.
  - `article(origin: str, href: str, headline: str, description: Optional[str], modified: Optional[str]) -> dict` — `Article`: `headline`, `url` and `mainEntityOfPage` (`origin + href`), `inLanguage`, `author`, `publisher` (name, url, logo), plus `description` and `dateModified` only when given.
  - `breadcrumbs(origin: str, trail: list) -> dict` — `BreadcrumbList` from `[(name, href or None), ...]`: `position` from 1, `item` (`origin + href`) only when the href is given.
  - `block_text(template: str, block: str, context: dict) -> str` — the text of a block of a template rendered with that context, unescaped, whitespace collapsed.
  - Every top-level block carries `'@context': 'https://schema.org'`. `origin` never ends in a slash (`settings()['site_url']`).

- [ ] **Step 1: Write the failing tests** in `tests/test_pages_structured.py`: `test_website_and_organization_blocks` (exact dicts for `origin='https://demoscopio.es'`; the logo file exists under `web/dist` and is at least 112×112, read with Pillow if installed, else by its PNG header), `test_article_block_keeps_only_what_it_is_given` (with and without `description` and `modified`), `test_breadcrumbs_number_the_trail_and_leave_the_page_without_item`, `test_block_text_reads_a_template_block` (`metodo_promedio.html`, block `title`, empty context → `'¿Cómo promediamos las encuestas?'`), `test_block_survives_reserved_characters` (render `base.html`-style snippet `{% for item in structured %}<script type="application/ld+json">{{ item|tojson }}</script>{% endfor %}` through `pages.environment()` with a breadcrumb named `'</script><script>alert(1)</script> "A" & B'`: one script element, no `</script><script>` inside it, and `json.loads` of its content gives the name back).
- [ ] **Step 2: Run** `python -m pytest -q -p no:cacheprovider tests/test_pages_structured.py` — Expected: FAIL (no module `structured`).
- [ ] **Step 3: Implement** the module, the constant, the export, the default in `render` and the loop in `base.html` (after the `twitter:` tags).
- [ ] **Step 4: Run** the file and `tests/test_pages.py` — Expected: PASS; no page has a block yet.
- [ ] **Step 5: Commit.**

### Task 2: the pages

**Files:**
- Modify: `mtpy/controllers/{Index,Analisis,Metodo,Casas,Elecciones,Sondeos}.py`
- Modify: `tests/test_pages_structured.py`, `deploy/README.md`

**Interfaces:**
- Consumes: the builders of Task 1; `pages.PAGES` (menu labels by `href`); `context['event_bar']['election']`, `context['poll_title']`, `context['poll_date']`, `context['detail_href']` (poll); `context['event']` (`name`, `event_date`) and `pages.event_href` (election); `context['profile']['pollster']['name']` and `pages.pollster_href` (profile); `pages.STUDIES[study]['template']`, `pages.METODO_SECTIONS[section]['template']`; `published_at` of the study in the manifest (`site().manifest_data()['studies'][study]['published_at']`).
- Produces: `Page.section_trail(self, href: str, name: str) -> dict` — the `BreadcrumbList` of a second-level page: the menu label of `href` linking to it, then `name` without link.

- [ ] **Step 1: Write the failing tests**: `test_every_page_carries_the_blocks_of_its_kind(houses, fresh_app)` (sitemap sweep + `/privacidad`: every block parses, has the schema.org context, and the list of `@type` per page is exactly the one of the spec table), `test_home_names_the_site_and_its_publisher`, `test_study_is_an_article_with_what_the_page_shows` (headline == visible `<h1>`, description == meta description, url == canonical, `dateModified` == manifest `published_at`, author and publisher), `test_study_without_a_publication_date_has_no_date`, `test_breadcrumbs_of_each_kind_of_page` (study, method section, profile, named election, election without a name, poll), `test_blocks_follow_the_configured_origin` (`site_url` `https://demoscopio.dev/`), `test_blocks_carry_canonical_urls_whatever_the_request` (`/casas/2?scope=es&utm_source=x`, a poll by house id, an election by its full date).
- [ ] **Step 2: Run** the file — Expected: the new tests FAIL (no page passes blocks).
- [ ] **Step 3: Implement** `section_trail` and the `structured=` of the six controllers.
- [ ] **Step 4: Document** in `deploy/README.md` (the paragraph of the head metadata): which pages carry which block and where the texts come from.
- [ ] **Step 5: Run** `python -m pytest -q -p no:cacheprovider tests/test_pages_structured.py tests/test_pages.py tests/test_pages_titles.py tests/test_pages_descriptions.py tests/test_pages_crawlers.py tests/test_pages_consent.py tests/test_pages_ejemplo.py` — Expected: PASS. Then the whole suite on a clean copy of the commit.
- [ ] **Step 6: Commit.**
