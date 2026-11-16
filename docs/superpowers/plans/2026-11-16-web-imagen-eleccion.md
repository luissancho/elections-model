# Web: imagen propia de una elección pasada y descripciones con cifras — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A link to a past election shows an image with its official result (the card of a poll, with the change against the previous election and the seats won), and the descriptions of a poll and of a past election say their figures.

**Architecture:** A new `mtpy/lib/pages/previews.py` holds the rules both link previews share (six rows, design version, fingerprint, change mark, figures that fit a description). `sondeos.preview_of` keeps composing the preview of a poll; `elecciones.result_preview` composes the one of a past election in the same shape, so `images.poll_card` draws it unchanged. The `Elecciones` controller answers the PNG at the address of the page ending in `.png` and declares it as `og:image`, as `Sondeos` does.

**Tech Stack:** Pillow 10.3, Jinja2, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-16-web-imagen-eleccion-design.md`

## Global Constraints

- PEP8, lines ≤ 120, numpydoc docstring on every function; identifiers and comments in neutral English, public texts in Spanish («sólo» with tilde, «encuestadora/casa», never «empresa»). Shares with the sign attached («33,1%»).
- Web code lives inside `mtpy`. Tests never touch the RDS, the network nor `backtest/results`.
- Luis commits in parallel in the same tree: before each commit `git status` and `git log -1`, add only the files of the task, never `git commit -a`. Commit messages in Spanish, in the style of the log, ending with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`. No push.
- The drawing (`images.poll_card`) does not change: `PREVIEW_DESIGN` stays 1 and the image of every poll keeps its bytes and its fingerprint.
- Six rows at most: the six parties with most vote, by vote. The row «-» (the rest of the candidacies) is not a party.
- A description never exceeds 160 characters (`SNIPPET_LENGTH`) and never cuts a figure: whole figures or none.
- Pages other than the poll page and the detail of a past election stay byte-identical.
- A whole-suite run on a clean copy closes the work (templates are touched).

## Review Focus

- The first published election of a scope, or one whose six parties are all new: no triangle and no «Cambio frente a…» in the grey line. (Task 2, `test_result_preview_without_a_previous_election`)
- A party that did not stand under that name in the previous election, next to others that did: only its row lacks the triangle. (Task 2, `test_result_preview_marks_the_change_against_the_previous_election`)
- The row «-» among the six most voted, and a result with dozens of parties: «-» is skipped and the footnote counts only the parties with a seat that do not fit. (Task 2, `test_result_preview_skips_the_rest_row_and_counts_the_seated`)
- An election without a name, or without published results: the title falls back to its date, the card is drawn with its header only and the description keeps the sentence it had. (Task 2, `test_result_preview_of_odd_elections`; Task 3, `test_election_without_results_keeps_its_sentence`)
- A scope without a published forecast, and a page with a pinned run: the image is drawn with the catalogue of the national run, and the pinned page declares the fingerprint of the latest run. (Task 3, `test_event_image_of_a_scope_without_forecast`, `test_pinned_election_page_declares_the_latest_image`)

---

### Task 1: the rules the previews share, and the description of a poll with its figures

**Files:**
- Create: `mtpy/lib/pages/previews.py`
- Modify: `mtpy/lib/pages/sondeos.py` (drop `PREVIEW_ROWS`, `PREVIEW_DESIGN`, `_preview_change`; `preview_of` uses `previews` and returns `description`), `mtpy/lib/pages/__init__.py` (export from `.previews`), `web/templates/sondeo.html` (block `description`)
- Test: `tests/test_pages_previews.py` (new), `tests/test_pages_images.py` (the patch of `PREVIEW_DESIGN`), `tests/test_pages_descriptions.py` (the poll)

**Interfaces:**
- Produces, in `previews.py`:
  - `PREVIEW_ROWS = 6`, `PREVIEW_DESIGN = 1`
  - `preview_change(change: Optional[float]) -> Optional[dict]` — the old `sondeos._preview_change`, unchanged.
  - `preview_version(drawn: dict) -> str` — 8 hex digits of the MD5 of `json.dumps([PREVIEW_DESIGN, drawn], sort_keys=True, ensure_ascii=False)`; reads `PREVIEW_DESIGN` of the module at call time.
  - `with_figures(lead: str, figures: Sequence[str], limit: int = SNIPPET_LENGTH) -> str` — `lead` plus the leading figures that fit whole, joined with `', '` and closed with `'.'`; `lead.rstrip()` when none fits or there is none.
- `sondeos.preview_of(...)['description']`: `with_figures('{election}: sondeo de {title} ({day}). ', ['PP 33,1%', …])` with the shares of its rows.

- [ ] **Step 1: failing tests** — `tests/test_pages_previews.py`:
  - `test_preview_change`: `None → None`; `0 → {'direction': 'same', 'text': '='}`; `0.4 → up '0,4'`; `-12.3 → down '12,3'`.
  - `test_preview_version_follows_the_design_and_the_data`: 8 hex digits; equal for equal data whatever the key order; differs with any change of the data and with `PREVIEW_DESIGN + 1` (monkeypatched on `previews`).
  - `test_with_figures_keeps_whole_figures_that_fit`: every figure when they fit (`'A: x, y.'`); with `limit` one character short of the last figure, that figure is left out and the text still ends `'.'` and `len <= limit`; no figure fits → `lead.rstrip()`; no figures → `lead.rstrip()`; default limit is `SNIPPET_LENGTH == 160`.
  - `tests/test_pages_descriptions.py`: `DESCRIPTIONS[POLL_PATH] = 'Elecciones generales 2026: sondeo de Sigma Dos (5 oct 2026). PP 33,1%.'`; `test_description_that_fits_is_escaped_once` expects `…sondeo de Sondeos &amp; Datos (5 oct 2026). PP 33,1%.`; replace `test_poll_description_says_what_the_page_holds_and_fits` by `test_poll_description_says_its_figures_and_fits` (through `sondeos.preview_of` with the thirteen-party GAD3 poll of `tests/test_pages_images.py`: six figures, in order, `len <= 160`, `snippet(text) == text`; with the election «Elecciones autonómicas Comunidad Valenciana 2027» and the title «Lápiz Estratégico para Prensa Ibérica» fewer figures, never a cut one); `test_long_poll_description_ends_at_its_first_sentence` keeps its house and outlet made long enough that no figure fits (the description is the first sentence).
- [ ] **Step 2:** run `python -m pytest tests/test_pages_previews.py tests/test_pages_descriptions.py -q` — Expected: FAIL (`No module named 'mtpy.lib.pages.previews'`, old description texts).
- [ ] **Step 3:** write `previews.py`; move the constants and `_preview_change` out of `sondeos.py`; `preview_of` adds `description`; `sondeo.html`: `{% block description %}{{ preview.description }}{% endblock %}`; `tests/test_pages_images.py` patches `pages.previews.PREVIEW_DESIGN`.
- [ ] **Step 4:** run `python -m pytest tests/test_pages_previews.py tests/test_pages_descriptions.py tests/test_pages_images.py tests/test_pages_titles.py tests/test_pages_crawlers.py -q` — Expected: PASS.
- [ ] **Step 5:** commit.

### Task 2: what the image of a past election says

**Files:**
- Modify: `mtpy/lib/pages/elecciones.py`, `mtpy/lib/pages/images.py` (`result_png`), `mtpy/lib/pages/__init__.py`
- Test: `tests/test_pages_event_images.py` (new)

**Interfaces:**
- Consumes: `previews.PREVIEW_ROWS`, `preview_change`, `preview_version`, `with_figures`.
- Produces, in `elecciones.py`:
  - `REST_PARTY = '-'`
  - `previous_result(scope: str, date: str) -> Optional[dict]` — `{'date': 'YYYY-MM-DD', 'pct': {party: pct}}` of the published election of the scope right before `date` (`site().events(scope)`), `None` for the first one.
  - `result_preview(title: str, date: str, rows: list, previous: Optional[dict], host: str) -> dict` — pure. `rows`: result rows as published (`party`, `pct`, `seats`, `color`). Returns `title`; `subtitle` (`'Resultado oficial del 23 jul 2023'` plus `' · Cambio frente a la elección anterior (10 nov 2019)'` when a shown row has a change); `rows` (`name`, `color`, `value`, `share`, `change`, `seats` as text or `None`); `has_seats`; `more` (`'y 5 partidos más con escaño en demoscopio.es'`, `'y 1 partido más con escaño…'`, `None`); `alt` (`'{title}, resultado oficial del {day}: PP 33,1%, PSOE 31,7%.'`); `description` (`with_figures('{title}: ', ['PP 33,1% (137 escaños)', 'PSOE 31,7% (121)', …])`, `None` without rows; `'(1 escaño)'` in the singular, no parenthesis without seats); `version` (`preview_version` of title, subtitle, rows and more).
  - `event_preview(state: dict, date: str) -> dict` — reads the catalogue of the run (the national one for a scope without a forecast), `meta`, `results` and `previous_result`; raises the 400 and 404 of the page.
  - `latest_event_preview(scope: str, date: str) -> Optional[dict]` — `event_preview` with the latest run, `None` on a 404.
  - `eleccion_context(...)['preview']` — the same data, from the rows the page already reads.
- Produces, in `images.py`: `result_png(preview: dict) -> bytes` (`encode(poll_card(preview))`).

- [ ] **Step 1: failing tests** — `tests/test_pages_event_images.py`, with `ROWS` = PP 33.06/137, PSOE 31.68/121, VOX 12.38/33, SUMAR 12.33/31, ERC 1.89/7, JxCat 1.6/7, `-` 1.2/0, EHB 1.36/6, PNV 1.12/5, BNG 0.62/1, CC 0.47/1, UPN 0.21/1, PACMA 0.67/0 and `BEFORE = {'date': '2019-11-10', 'pct': {'PP': 20.81, 'PSOE': 28.0, 'VOX': 15.08, 'ERC': 3.63, 'JxCat': 2.19}}`:
  - `test_result_preview_marks_the_change_against_the_previous_election`: six rows PP…JxCat by vote; shares `'33,1%'…`; changes `up 12,3`, `up 3,7`, `down 2,7`, `None` (SUMAR), `down 1,7`, `down 0,6`; seats `'137'…`; subtitle with the legend and `(10 nov 2019)`; `'='` for an equal share.
  - `test_result_preview_without_a_previous_election`: `previous=None`, and a previous without any of the six: no change, subtitle `'Resultado oficial del 23 jul 2023'`.
  - `test_result_preview_skips_the_rest_row_and_counts_the_seated`: with `-` as the third most voted it is not a row; `more == 'y 5 partidos más con escaño en demoscopio.es'`; one left → `'y 1 partido más con escaño en demoscopio.es'`; none → `None`.
  - `test_result_preview_texts`: `alt`, `description` (six figures, `'(137 escaños)'` only on the first, `<= 160`), singular seat, a long title where fewer figures fit.
  - `test_result_preview_of_odd_elections`: no rows → `rows == []`, `description is None`, `has_seats` false, `result_png` still a 1200×675 PNG; a row without `pct` is dropped; a row without seats has no parenthesis.
  - `test_result_preview_version_follows_what_is_drawn`: each of title, date, a share, a seat, the previous share and the footnote changes the version; `host` only through the footnote.
  - `test_result_png_is_the_card_of_a_poll`: `images.result_png(data) == images.poll_png(data)`; the bars carry the party colours.
  - `test_previous_result_reads_the_election_before(houses, fresh_app)`: with 2019-11-10 and 2015-12-20 written, 2023 → 2019, 2019 → 2015, 2015 → `None`.
  - `test_event_preview_reads_the_published_election(houses, fresh_app)`: equals `eleccion_context(...)['preview']`; title «Generales 2023»; rows of the fixture; with a previous election written with other shares, the changes follow.
- [ ] **Step 2:** run `python -m pytest tests/test_pages_event_images.py -q` — Expected: FAIL (`AttributeError: … 'result_preview'`).
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run `python -m pytest tests/test_pages_event_images.py tests/test_pages_images.py tests/test_pages.py -q` — Expected: PASS.
- [ ] **Step 5:** commit.

### Task 3: the address of the image, the tags of the page, the description and the deploy checks

**Files:**
- Modify: `mtpy/lib/pages/common.py` (route), `mtpy/controllers/Page.py` (`IMAGE_TTL`, `png`), `mtpy/controllers/Sondeos.py` (uses `Page.png`), `mtpy/controllers/Elecciones.py` (`image_action`, `og:image` in `detail_action`), `web/templates/eleccion.html` (block `description`), `deploy/check-api.sh`, `deploy/README.md`
- Test: `tests/test_pages_event_images.py`, `tests/test_pages_descriptions.py`, `tests/test_pages_images.py` (`test_other_pages_keep_the_brand_image`)

**Interfaces:**
- Consumes: `pages.event_preview`, `pages.latest_event_preview`, `eleccion_context(...)['preview']`, `pages.images.result_png`, `pages.images.available`, `pages.resolve_event_path`, `pages.event_href`.
- Produces: route `(r'/elecciones/{country:loc}/{scope}/{date}\.png', 'elecciones', 'image', ['GET'])`; `Page.IMAGE_TTL = 86400` and `Page.png(self, png: bytes) -> bytes` (sets `image/png`, the ETag and the cache); `Elecciones.image_action(country, scope, date) -> bytes`.

- [ ] **Step 1: failing tests** — `tests/test_pages_event_images.py`, `EVENT_PATH = '/elecciones/es/generales/2023-07'`:
  - `test_event_image_route_answers_a_cacheable_png`: 200, `image/png`, `public, max-age=86400`, ETag, the bytes of `result_png(latest_event_preview('es', '2023-07-23'))`; the same by the full date (`/2023-07-23.png`) and with any query; HEAD.
  - `test_event_image_route_refuses_what_the_page_refuses`: `/elecciones/es/generales/2020-01.png` 404, `/elecciones/es/generales/abc.png` 400, `/elecciones/es/nadie/2023-07.png` 404, HTML error page with `no-store`, nothing drawn; `.jpg`, `.PNG`, `.png.png` are 404.
  - `test_election_page_declares_its_image`: `og:image == SITE + EVENT_PATH + '.png?v=' + version`, `og:image:alt == data['alt']`; the same asked by the full date and with a query; not in the sitemap.
  - `test_pinned_election_page_declares_the_latest_image`: with `?run=RUN` and `pages.latest_event_preview` patched, the patched version and alt; patched to `None`, the brand image.
  - `test_election_page_keeps_the_brand_image_when_nothing_can_be_drawn`: `images.available` false → brand image and alt.
  - `test_event_image_of_a_scope_without_forecast`: an election of `es-md` published without a run of `es-md`: page 200, declares its image, the image answers 200.
  - `tests/test_pages_descriptions.py`: `DESCRIPTIONS['/elecciones/es/generales/2023-07'] = 'Generales 2023: PP 33,1% (160 escaños), PSOE 31,4% (150), VOX 12,3% (40), Cs 0,1% (0).'`; `test_election_without_results_keeps_its_sentence` (results part without rows → the sentence of today, with and without «y el modelo») replaces `test_election_description_only_names_the_model_when_the_page_has_it`; `test_election_without_a_name_is_described_by_its_date` keeps passing without `preview` in the context.
  - `tests/test_pages_images.py::test_other_pages_keep_the_brand_image`: the poll page and the detail of the election are the two pages with an image of their own.
- [ ] **Step 2:** run `python -m pytest tests/test_pages_event_images.py tests/test_pages_descriptions.py tests/test_pages_images.py -q` — Expected: FAIL (404 on the `.png` address, old description).
- [ ] **Step 3:** implement; `deploy/check-api.sh`: `check "/elecciones/es/generales/abc.png" 400 html` and `check_published "/elecciones/es/generales/2023-07.png" none image/png` (`check_published` passes its optional third argument to `check` as the content type); `deploy/README.md`: the paragraph of the image of a poll also tells the one of a past election and where `PREVIEW_DESIGN` lives now.
- [ ] **Step 4:** run `python -m pytest tests/test_pages_event_images.py tests/test_pages_descriptions.py tests/test_pages_images.py tests/test_pages.py tests/test_pages_titles.py tests/test_pages_crawlers.py tests/test_pages_structured.py tests/test_web_routes.py tests/test_deploy_files.py -q` — Expected: PASS.
- [ ] **Step 5:** commit.
