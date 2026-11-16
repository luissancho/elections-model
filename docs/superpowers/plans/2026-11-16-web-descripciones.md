# Web: descripciones de las páginas — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The description of every page says exactly what the page holds and reads whole in a search result or a link preview (160 characters at most).

**Architecture:** A template filter `snippet` cuts a text to 160 characters at a sentence end or, failing that, at a whole word with an ellipsis; `base.html` captures the `description` block once and applies it for `<meta name="description">` and `og:description`. The descriptions that fail today are rewritten in their templates.

**Tech Stack:** Jinja2 templates and filters, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-16-web-descripciones-design.md`

## Global Constraints

- PEP8, lines ≤ 120, numpydoc docstring on every function; identifiers and comments in neutral English, public texts in Spanish («sólo» with tilde, «encuestadora», never «empresa»).
- Web code lives inside mtpy; no standalone scripts. Tests never touch the RDS nor write to `backtest/results`.
- Luis commits in parallel in the same tree: before each commit `git status` and `git log -1`, add only the files of the task, never `git commit -a`.
- Commit messages in Spanish, in the style of the log, ending with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`. No push.
- The texts of the spec table, verbatim. Limit: 160 characters.
- `<meta name="description">` and `og:description` carry the same text.
- Unchanged: the home, `/metodo/backtest`, `/analisis`, `/analisis/voto-cera`, `/privacidad`, the default description.

## Review Focus

- A description that is already HTML-escaped (`D&#39;Hondt`, `Sondeos &amp; Datos`): the cut counts characters, not entities, never splits one and the result is escaped once. (Task 1, `test_snippet_cuts_escaped_text_without_splitting_an_entity`)
- A text whose first sentence alone is longer than the limit, or with no sentence end at all: a whole word and «…», never over the limit. (Task 1, `test_snippet_cuts_at_a_whole_word_when_no_sentence_fits`)
- A poll with a long house and sponsor: the description ends at its first sentence. (Task 2, `test_long_poll_description_ends_at_its_first_sentence`)
- An election whose page has no model card: no «y el modelo». (Task 2, `test_election_description_only_names_the_model_when_the_page_has_it`)
- The regional scope with the longest name: its event description still fits whole. (Task 2, `test_event_description_fits_for_the_longest_scope_name`)

---

### Task 1: the cut

**Files:**
- Create: `tests/test_pages_descriptions.py`
- Modify: `mtpy/lib/pages/filters.py` (`snippet`, `SNIPPET_LENGTH = 160`), `mtpy/lib/pages/common.py` (register the filter, docstring of `environment`), `web/templates/base.html`

**Interfaces:**
- Produces: `filters.snippet(text, limit: int = SNIPPET_LENGTH) -> str` — plain text (unescaped when it gets `Markup`), whitespace collapsed, cut as the spec says; Jinja filter `snippet`.

- [ ] **Step 1: Write the failing tests**: `test_snippet_keeps_a_text_that_fits`, `test_snippet_keeps_the_leading_sentences_that_fit`, `test_snippet_cuts_at_a_whole_word_when_no_sentence_fits` (also: the kept sentences shorter than half the limit do not count; trailing words of three letters or fewer are dropped; never over the limit), `test_snippet_cuts_escaped_text_without_splitting_an_entity`, `test_every_description_fits_a_snippet_and_matches_og(houses, fresh_app)` (every sitemap path plus `/privacidad`: not empty, ≤ 160, equal to `og:description`).
- [ ] **Step 2: Run** `python -m pytest -q -p no:cacheprovider tests/test_pages_descriptions.py` — Expected: FAIL (`snippet` missing; three method descriptions over 160).
- [ ] **Step 3: Implement** `snippet`, register it, and in `base.html` capture the block (`{% set meta_description %}{% block description %}…{% endblock %}{% endset %}`) and print `{{ meta_description|snippet }}` in both tags.
- [ ] **Step 4: Run** the file and `tests/test_pages.py` — Expected: PASS.
- [ ] **Step 5: Commit.**

### Task 2: the texts

**Files:**
- Modify: `web/templates/{promedios,sondeo,elecciones,eleccion,casas,casa,metodo,metodo_promedio,metodo_escanos,metodo_rating}.html` (block `description`)
- Modify: `tests/test_pages_descriptions.py`, `tests/test_pages.py` (three old assertions), `deploy/README.md`

**Interfaces:**
- Consumes: `snippet` (Task 1); `event_bar.phrase` and `event_bar.year`; `quality.informative`; `horizon_tables`.

- [ ] **Step 1: Write the failing tests**: `test_descriptions_say_what_each_page_is(houses, fresh_app)` (the exact text of each row of the spec table that the fixture can render, and that it ends with a full stop), `test_no_description_repeats_or_leads_with_a_menu_label(houses, fresh_app)`, `test_election_description_only_names_the_model_when_the_page_has_it(api, fresh_app)`, `test_long_poll_description_ends_at_its_first_sentence(houses, fresh_app)`, `test_event_description_fits_for_the_longest_scope_name()` (the `description` block of `promedios.html` with the phrase of «Comunidad Valenciana»).
- [ ] **Step 2: Run** the file — Expected: the new tests FAIL.
- [ ] **Step 3: Rewrite** the ten description blocks.
- [ ] **Step 4: Update** the old assertions of `tests/test_pages.py` and document the rule in `deploy/README.md` (the paragraph of the head metadata).
- [ ] **Step 5: Run** `python -m pytest -q -p no:cacheprovider tests/test_pages_descriptions.py tests/test_pages_titles.py tests/test_pages.py tests/test_pages_ejemplo.py tests/test_pages_consent.py tests/test_pages_crawlers.py` — Expected: PASS. Then the whole suite on a clean copy of the commit.
- [ ] **Step 6: Commit.**
