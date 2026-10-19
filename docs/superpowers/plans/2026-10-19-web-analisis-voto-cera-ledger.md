# SDD ledger — plan: docs/superpowers/plans/2026-10-19-web-analisis-voto-cera.md

Spec: docs/superpowers/specs/2026-10-19-web-analisis-voto-cera-design.md (approved 2026-10-19).
Branch: dev (HEAD d27b1fc at start). Luis integrates dev → master himself.

Ruling: work directly on `dev`, no worktree/feature branch — every previous web phase was built on `dev` and Luis integrates it; a worktree would add a merge step he has not asked for — cost if wrong: he branches retroactively from the first task commit.

## Pre-flight scan (2026-10-19)

| Pair / task | Produces vs consumes | Found |
|---|---|---|
| T1 ↔ T2 | CSV/JSON formats; labels normalised by `normalise_label` (T2) | consistent; T1 applies the same rule (uppercase, unidecode, [A-Z0-9]) |
| T2 ↔ T4 | `Results` fields (`totals.n_seats`, `names`, `siglas`) | consistent |
| T3 ↔ T5 | `baseline_date`, `election_growth` | consistent |
| T4 ↔ T6/T8 | `participation_history` lives in T4 (seats.py); T6 uses `study_inputs.hist` | consistent |
| T5 ↔ T6 | `increment_votes` used inside `scenario_table` | consistent |
| T7 ↔ T11 | `with_without` keys → `check` part | consistent |
| T8 ↔ T11 | `ecological`, `autonomic_processes` keys → `vote`/`participation` parts | consistent |
| T9 ↔ T11 | `CeraStudy` attributes → exporters | consistent (names listed in T9) |
| T10 ↔ T11/T12 | `STUDY_PARTS`, `STUDY_CSV_OF`, `path_study`, manifest `studies` | consistent; fixture-validation test added in T12 |
| T11 ↔ T14 | titles in publish `STUDY_TITLES`; pages read them from `meta` | consistent |
| T12 ↔ T14–17 | `Site.study`, `check_study`, `write_study` helper | consistent |
| T13 | uses `read_monthly_workbook` (T2) | consistent |
| T14 ↔ T15–17 | `cera_context` keys; `shade_scale` filter added in T16 to `environment()` | consistent |
| T15/T17 ↔ T18 | container ids and "Ver más" button ids | Ruling: button ids are `provinces-more` (T15) and `check-more` (T17), matching T18's `wireMore` calls — cost if wrong: a selector rename |
| T4 (self) | blocks 152/170 need UPN in `Regionalista` | params blocks already put UPN there; consistent |
| T2 (self) | 245 months Jan-2006..May-2026 | 20·12+5 = 245, consistent |
| Rubric | tests asserting nothing / verbatim duplication mandated? | none |

## Tasks
Task 1: dispatched (BASE d27b1fc, model opus)
Task 1: implementer DONE_WITH_CONCERNS (commit 2f3c91f); pre-existing failure tests/test_pages_ejemplo.py::test_promedio_page_shows_the_example_block fails on a clean tree too (not ours; tell Luis).
Ruling: scrutiny CSVs take `electors` and `voters` from Infoelectoral (`Censo CERA`, `Total votantes CERA`), candidature votes from the Interior file — the Interior PDFs have typos (Albacete 862 for 8.064, Barcelona voters 19.315 for 21.434, truncated 2019 electors) and Infoelectoral is the authority the D'Hondt uses — cost if wrong: two columns to regenerate.
Ruling: the repo's 2023 CERA electors total is Infoelectoral's 2.328.260 (not the OCE's 2.328.261) and the candidature votes sum is whatever the mapped columns give (203.152, not the article's 203.115); Task 2 tests assert those — cost if wrong: a one-elector / 37-vote discrepancy in public figures.
Ruling: the ecological exclusion is by CERA voters, `params.ecological_min_voters = 200` (Ceuta 89, Melilla 177 in 2023), replacing `ecological_min_census` (Melilla's census, 5.971, exceeds Cuenca's) — Task 8 uses `min_voters` — cost if wrong: a parameter rename.
Ruling: where the article's figures do not come out of the correct data (CyL 2026 participation by origin 14,4/5,0/4,8 vs 11,5/7,6/5,0 — the article grouped US consulates under Europe; Aragón 2026 15,8/4,0/5,1; 2023 left share 49,7 over provinces vs 50,0; Madrid 29-N census 498.874 vs the press's 494.799), the data wins: tests and page use the computed values, and the spec's closing section records each deviation — cost if wrong: page figures differ from the article's.
Ruling: exposure membership as fitted by the implementer (baja = Western Europe + Canada, Africa, Asia, Oceania; totals 209.591 / 206.703 / 160.691 vs table 2) and the España Vaciada column covering two candidatures (same sigla EV) are accepted and documented.
Ruling: the 2023 CERA voters figure of the repo is Infoelectoral's 208.431 (sum of `Total votantes CERA`), not the OCE's 233.688 papeletas; the page quotes the repo figure and «Cómo leerlo» explains the difference (Tasks 11, 14, 15) — cost if wrong: a public figure differs from the press one.
Note: Luis committed 0347c23 "Fix" on dev during Task 1; Task 1 range is 0347c23..5118c25.
Task 1: minor (deferred): countries.csv carries 13 codes beyond tab6 (e.g. 155 «SERBIA (155)»); PSC maps to the PSOE candidature (no separate PSC row in the XLSX); exposure membership fitted (±3) rather than the literal rule; ESPAÑA VACIADA column also carries 70 Asturias votes of ASTURIAS EXISTE-ESPAÑA VACIADA (same sigla EV).
Task 1: complete (commits 2f3c91f..5118c25, review clean; ⚠️ article folder untouched and no scratch scripts in the tree verified by the controller)
Task 2: dispatched (BASE 5118c25, model opus)
Task 2: implementer DONE_WITH_CONCERNS (edf0498): misfiled 1-elector rows kept under their own province code (±1 in 2010/11 province totals); renamed countries summed under one name → later tasks join on `country_code`; Feb-2011 duplicate keeps the «(nuevo)» reading (+7.344 method jump); data.py 612 lines (mostly docstrings).
Ruling: subagent commits may carry their own model's Co-Authored-By trailer (Opus 5.5) — honest attribution beats uniformity — cost if wrong: mixed trailers in the log.
Task 2: minor (deferred): groupby().sum() in read_results could hide duplicate candidature headers (unique today); read_results bypasses _read_sheet; test helper path() rebuilds FileSystem; no tests for read_consulates/read_autonomicas/read_countries/load_params/province_series error paths.
Task 2: complete (commits edf0498, review clean; ⚠️ pristine output verified by the controller with -W error)
Task 3: dispatched (BASE edf0498, model opus)
Task 1: minor (deferred, plan-mandated): tests/test_cera_data.py::test_cera_data_folder_is_complete opens JSON files with json.load(open(...)) and leaves them unclosed (ResourceWarning under -W error; silent in normal runs) — fix with context managers in the final wave.
Task 3: implementer DONE_WITH_CONCERNS (c4e916d); article growth figures reproduced exactly (422.253, 189.290, ×2,2, Madrid 124.195, Argentina 73.855, México 54.954).
Ruling: `update_periods` keeps equal lengths (47 + 47 months: 2018-11-01→2022-10-01 and 2022-10-01→2026-09-01); the plan's expected 2019-07-01 was wrong — cost if wrong: none, the spec asks for equal lengths.
Ruling: the trend-break regression over the full series gives p_slope ≈ 0,0004 and p_level ≈ 0,61; the page and the spec claim only the slope change (the article's p ≈ 0,01 for the level came from another specification); the test asserts p_slope < 0,05 and that p_level is a probability in [0, 1], not its sign — cost if wrong: the page omits a claim the article made.
Ruling: `two_periods` and `election_growth` take an optional `province_names` Series (es-provinces.csv) that Task 9 passes — cost if wrong: a signature change.
Task 3: minor (deferred): a 2006 «22 MAURICIO» row is read under code 22 instead of 229 (no effect from 2007 on).
Task 3: minor (deferred): update_periods does not validate last_date (negative length); trend_break with a break_date outside the series fits a degenerate model silently; pct/ratio inf when baseline is 0 (undocumented for pct); test hardcodes 249 series points; _names lacks a countries-None guard.
Task 3: complete (commits c4e916d..b9abd39, review clean; ⚠️ pristine output verified by the controller with -W error)
Task 4: dispatched (BASE b9abd39, model opus)
Task 4: implementer DONE_WITH_CONCERNS (7b81542): D'Hondt reproduces 2019 and 2023 exactly; blocks 152/170.
Ruling: the last-seat gap is the votes the runner-up needs to EQUAL the winner's last quotient, rounded to the nearest integer, minimum 1 (the article's rule; the strict rule gives the article's table 10 plus one in nine of ten provinces) — the spec demands reproducing table 10 — cost if wrong: a one-vote convention in a public table.
Ruling: `allocate_results` feeds `Simulator.alloc_seats` the candidatures ordered by total votes descending, so exact quotient ties go to the larger vote total as the LOREG says (alloc_dhondt picks the first maximum in insertion order) — cost if wrong: none on the official data.
Task 4: minor (deferred): a tie between two seat holders on the smallest quotient picks the first column; a None gap would turn the gap column into object dtype.
Task 4: minor (deferred): participation_history with no CERA columns raises an unclear concat ValueError; round() half-to-even at exact .5; the bare stand-in test couples to .totals.
Task 4: complete (commits 7b81542..bc3a373, review clean; ⚠️ pristine output verified by the controller with -W error)
Task 5: dispatched (BASE bc3a373, model opus)
Task 5: implementer DONE_WITH_CONCERNS (1f89376): article projection 539.260 new electors / 37.006 votes / 0 changes; real census 442.257 / 30.482 / 0; litigation removes exactly 163.020 → 279.237 / 22.412 / 0; Cantabria net +134,8 (×2,99 participation to close), Madrid −1.566.
Ruling: the litigation distribution uses the latam growth from 2022-10-01 to the latest monthly close (2026-09-01), not to 1 August: the monthly series is the only province × country source and the one-month difference is negligible — cost if wrong: a marginal shift of a hypothetical scenario.
Task 5: minor (deferred): net_effect's participation_needed column is object dtype when it holds None (exporters must handle it); new_electors keeps floats for the projected census.
Task 5: minor (deferred): litigation_census with a series starting after `since` raises an opaque KeyError; "items with weight hold only 0" message and missing Raises in the docstring; an unknown country_code silently counts as `rest`; increment_model drops extra columns absent from results; float tie-break by fractional part; no test of the no-runner-up path.
Task 5: complete (commits 1f89376, review clean; ⚠️ pristine output verified by the controller with -W error)
Task 6: dispatched (BASE 1f89376, model opus)
Task 6: implementer DONE_WITH_CONCERNS (cbd0520): pool 181.032 electors; placement grid observed cell (9 %, 50 %) = 5 (article-size pool of 62.800 gives 2); exterior grid up to 4 seats below table 11 in the 50 % column; scenario ×2/×3/0,9 gives 6/−6 (article 7/−6); placement grid 0,78 s.
Ruling: the placement pool stays the observed altas of the four countries since the law (181 k); the page quotes the computed cells (5 at the observed cell), not the article's 2 — the spec's decision on the pool stands and the data win — cost if wrong: a public figure differs from the article.
Ruling: `optimal_placement` gains a trailing optional `split=None` for the contrast cell (Task 9 passes the CERA split); `hist` for the replication is `participation_history` over every Infoelectoral election with CERA columns (1986–2023), built in Task 9 via Task 8's `available_results` — cost if wrong: the replication figures (523 k / 1,114 k votes, 20 / 27 provinces with the full history).
Ruling: Task 7's `with_without` and `verdict` go into a new module `mtpy/lib/studies/cera/check.py` instead of seats.py (995 lines) — file organisation is the builder's call; the spec's "seats.py (4/4)" is a label — cost if wrong: an import path.
Task 6: minor (deferred): _biased_split silently zeroes a province with all-zero weights (none in real data); placement_pool returns the whole census when no date precedes `since`; _seat_cost docstring says "upper bound" for a greedy estimate; stray "(14 of the spec)" in a docstring; grid tests pin full matrices.
Task 6: complete (commits cbd0520, review clean; ⚠️ pristine output and timing verified by the controller with -W error)
Ruling: Task 7 also moves the stress and placement code (peninsular_results, _biased_split, exterior_grid, scenario_table, placement_pool, _seat_cost, _greedy_seats, optimal_placement, report_replication) from seats.py to a new `stress.py` as a pure move in a separate commit, re-exported from the package — seats.py at 995 lines mixes four concerns — cost if wrong: import churn.
Task 7: dispatched (BASE cbd0520, model opus)
Task 7: implementer DONE_WITH_CONCERNS (8e9f423, 43da828): 2023 nine over the gap, Madrid decided (PP +1/PSOE −1, crosses block); 2019 ten over the gap, Bizkaia decided (PP +1/PNV −1); stress.py split done.
Ruling: Task 9 passes the scrutiny `voters` (Infoelectoral) to `with_without`, so `voters_cera` and the page's «votos CERA» are voters, not votes to candidatures — cost if wrong: a 1–5 % difference in a public column.
Task 7: minor (deferred): check.py imports seats._sigla_seats and stress.py imports seats._allocate_row (private names across modules); Task 4's report still lists the strict gaps (superseded by bc3a373).
Task 7: minor (deferred): private cross-module imports (seats._sigla_seats, seats._allocate_row); the shared fixture runs once per test module (twice per session); to_numeric(errors='coerce') hides a non-numeric gap; net_by_block iterates only the "with" blocks.
Task 7: complete (commits 8e9f423..43da828, review clean; ⚠️ pristine output verified by the controller with -W error)
Task 8: dispatched (BASE 43da828, model opus)
Task 8: implementer DONE_WITH_CONCERNS (c968a7e): ecological r −0,743 (50) / −0,767 (43), quartiles 59,8/44,3; sensitive share ARG+CUB+VEN 30,0→26,7, all countries 24,1→17,4; ratios SUMAR 119,7 / PP 99,9 / PSOE 91,6 / VOX 90,5; 16 Infoelectoral files, 13 with CERA columns, read_results raises on 1977–1982; election days from data/wikipedia/wp-urls.json.
Ruling: the sensitive-provinces chart of the page shows the all-countries shares (24,1 → 17,4, as the infographic), and the three-country pair (30,0 → 26,7) goes in the question-2 text; the autonomic margin is the article's figure-6 definition (over voters); madrid_magnet returns None for a non-positive growth; Task 9 catches ValueError from read_results for the elections without CERA columns — cost if wrong: a chart choice and a None in a table.
Task 8: minor (deferred): _latam_share silently treats a code missing from countries.csv as non-latam (participation_by_origin raises instead); reindex over ORIGINS drops a region outside ORIGINS silently; sensitive_share has no zero-growth guard; the test helper _results_all swallows every ValueError; shared fixture imported across test modules (conftest would be cleaner); `only` docstring wording; `codes` parameter holds names.
Task 8: complete (commits c968a7e, review clean; ⚠️ pristine output verified by the controller with -W error)
Task 9: dispatched (BASE c968a7e, model opus)
Task 9: implementer DONE_WITH_CONCERNS (eb56d87): compute_cera 15,5 s; both scenarios 0 changes; replication 523 k / 1.114 k votes, 20 / 27 provinces, mean/max participation 18,9 / 40,2 %; verdict_input None.
Ruling: the presentation constants (grid axes, the six scenario rows, the three sensitive countries, the PSOE name, the ratio siglas) stay in study.py, not params.json — they define what the page shows, not the model — cost if wrong: an edit in code instead of JSON.
Ruling: importing the study package must not load statsmodels into the web: Task 11's publish/studies.py imports the study lazily inside publish_study, and census.py moves its statsmodels import inside trend_break — cost if wrong: slower web start-up.
Task 9: review spec ❌ — Important: computations in the orchestrator (empirical split at study.py:542, report participation at :566-569); minors: skip decision parses list.index message text; tiny filters/groupby in the orchestrator; scrutiny without XLSX skipped silently and candidatures[date] KeyError unclear.
Task 9: fix round 1/5 (1 addressed pending re-review — empirical_split helper, report_replication returns the participation, MissingCeraColumns exception; commits eb56d87..97cbb0f)
Ruling: plain input wiring (a groupby sum for census_by_province, the baseline total, the pool total) may stay in the orchestrator; only derived measures belong to the modules — cost if wrong: none.
Task 9: fix round 1/5 (2 addressed, 0 open — empirical_split helper, report participation inside report_replication, MissingCeraColumns; commits eb56d87..97cbb0f)
Task 9: minor (deferred): a scrutiny without its Infoelectoral XLSX is skipped silently (29-N before the results load); candidatures[date] KeyError unclear when the map lacks an election.
Task 9: complete (commits eb56d87..97cbb0f, review clean after round 1)
Task 10: dispatched (BASE 97cbb0f, model sonnet)
Task 10: implementer DONE_WITH_CONCERNS (3c3aacd): placeholder study-*.json fixtures added because an existing parametrized test validates every schema; Task 12 replaces them with real ones.
Task 10: minor (deferred): _check_study's regex check is redundant with the STUDIES membership; the contract cites publish.publish_study before it exists (Task 11); no negative test for `registered: 'x'`.
Task 10: complete (commits 3c3aacd, review clean)
Ruling: Task 11's exporter and publish tests use the real study (a module-scoped `compute_cera(data_fs)` fixture, ~15 s) and a `compute=` callable returning it, instead of a hand-built synthetic CeraStudy — fidelity over isolation; the publish-flow tests (register, dirty tree, meta last) still run on a temporary file system — cost if wrong: slower tests.
Task 11: dispatched (BASE 3c3aacd, model opus)
Task 11: implementer DONE_WITH_CONCERNS (fefb3a9): Luis has UNCOMMITTED «parties» work in the shared checkout (README, contrato, Publish.py, bundle.py, pages/__init__.py, pages/common.py, promedios.py, publish/__init__.py, webapi/published.py, four test files, promedios.js, new publish/parties.py and a fixture); the implementer staged only its own hunks. `import mtpy.lib.publish` already loads statsmodels via computer→dataviz (pre-existing); the study package and statsmodels.api are not loaded. p-values keep 3 significant digits; source URLs for Interior and the boards are home pages (Luis to check).
Ruling: from Task 12 on, the work moves to an isolated git worktree on a branch from fefb3a9 (`analisis-voto-cera`), because Luis is editing the same files uncommitted in the dev checkout; Luis merges the branch into dev after committing his work — cost if wrong: one merge with conflicts in pages/common.py and the test files.
Task 11: minor (deferred): test_export_study_figures pins live monthly figures (2762829, 2026-09-01…) that break on a data refresh; studies.py imports common._iso_dates/_today; source constants could live beside the study; no dry-run test of the study action; claim date written long ("29 de noviembre de 2026").
Task 11: complete (commits fefb3a9, review clean; schema coverage rests on test_export_study_parts_validate over the real study)
Worktree: /Users/luiss/HD/Proyectos/Code/elections-model/.claude/worktrees/analisis-voto-cera (branch analisis-voto-cera from fefb3a9); Tasks 12–19 run there; ledger stays in this file.
Task 12: dispatched (BASE fefb3a9, model sonnet, worktree)
Worktree baseline: 747 passed, 75 skipped, 1 failed (the pre-existing tests/test_pages_ejemplo.py::test_promedio_page_shows_the_example_block, run with -x so it stopped there; the subagents' full runs give 1058 passed with only that failure).
Task 12: implementer DONE_WITH_CONCERNS (0f5f8d4): report transcribed by the controller (worktree sessions cannot write to the main workspace; from now on briefs/reports live in the worktree workspace .claude/worktrees/analisis-voto-cera/.superpowers/sdd/2026-10-19-web-analisis-voto-cera/); /api/v1/studies returns 200 with an empty list when nothing is published (accepted).
Task 12: minor (deferred): long import line in webapi/__init__.py; study-check.json fixture has n_decided 3 with one decided entry (align in Task 17 if the page cross-checks); an invalid `table` with JSON is ignored (as Analysis); no CSV-404 test for an unpublished study.
Task 12: complete (commits 0f5f8d4, review clean)
Task 13: dispatched (BASE 0f5f8d4, model sonnet, worktree)
Task 13: implementer DONE (7de86b5): load/ is a namespace package (no __init__.py), run_load relies on ROOT in sys.path; `--what cera` writes data/cera/ine files without --save (accepted: inputs refresh, not a database write); not run against ine.es.
Task 13: minor (deferred): the cera step ignores the --save convention (documented); test cosmetics (provinces.json moved; lax final assertion); a corrupt old workbook would block the refresh.
Task 13: complete (commits 7de86b5, review clean; -W error run by the reviewer)
Task 14: dispatched (BASE 7de86b5, model opus, worktree)
Task 14: implementer DONE_WITH_CONCERNS (0d97917): the study meta is exposed as `study_meta` (the footer needs `meta` = run meta); the route is `/analisis/{study}` (`:str` rejects hyphens); the study page loads pages/common.js until Task 18 swaps it; fixture meta.params lacks litigation/participation (figures None); radar left share = CERA left share of the last generales; placement_observed = the 0,5 column; analisis.py 544 lines; KPI arrow in the site's style.
Ruling: Task 15 extends the study-meta fixture with params.litigation and params.participation (as the real export has them) so the KPI supports render in tests; Task 17 builds the «2023 ya pasó la prueba» table from check.rows filtered by over_gap and aligns the fixture's n_decided/decided — cost if wrong: fixture edits.
Task 14: minor (deferred): meta read twice (cards and context); «29-N» and 2026-09-10 as Python text constants (accepted as context facts); import order in inicio.py; task-number comments in the skeleton; analisis.py may deserve a cera.py split as it grows.
Task 14: complete (commits 0d97917, review clean; -W error run by the reviewer)
Task 15: dispatched (BASE 0d97917, model opus, worktree)
Task 15: implementer DONE_WITH_CONCERNS (0637329): province «Altas» column counts from 1-10-2022 (mockup) while the summary counts from 2023; _country_name helper; provinces-more inert until Task 18; analisis.py 744 lines; beyond the brief: KPI notes wrap on the study page, shared classes .note/.grid2/.band/.claim/.pill.warn, write_study and publish_study gained a parts= override for tests.
Task 15: minor (deferred → Task 16 fixes the first two): the Cuba «apenas se mueve» clause is unconditional prose around a live ratio; the «versión actualizada de los periodos» sentence is clumsy; «cambia el reparto en 0 provincias» edge case; .three cards lack aria-label; analisis.py 744 lines.
Task 15: complete (commits 0637329, review clean; -W error run by the reviewer)
Ruling: Task 16 first moves the per-question view models (the question-1 helpers and the ones it adds) from analisis.py to a new mtpy/lib/pages/cera.py as a pure move (separate commit), leaving the section (index, cards, context assembly, island) in analisis.py — cost if wrong: an import path.
Task 16: dispatched (BASE 0637329, model opus, worktree)
Task 16: implementer DONE_WITH_CONCERNS (5592db4, 1f153bb): the move also carried _get and (in commit 2) UNIFORM_SPLIT/SCENARIOS; «Otros» = sum of the other blocks (no n_seats in the parts; omitted with the two-block fixture); grid note reads «uniforme: {grid_observed}; real: {left change}»; scenario names built from rows; CSS :has() swaps the prediction tables without JS; the band is omitted with the fixture because vote.processes lacks the autonomic rows → Task 17 extends the study-vote fixture.
Task 16: minor (deferred): ungated opener/lead sentences («Ni concentración estratégica ni giro a la izquierda», «casi no votan» gated only on latam < rest, «basta relajar una para que el efecto se desplome», «Arrancar un número significativo…»); typed country/block names in prose (Argentina, Cuba y Venezuela; PSOE + SUMAR); aria-labels without None guard; the site.css comment overstates the no-JS swap (:has() only).
Task 16: complete (commits 5592db4..1f153bb, review clean; -W error run by the reviewer)
Task 17: dispatched (BASE 1f153bb, model opus, worktree)
Task 17: implementer DONE_WITH_CONCERNS (6670b48): «cambió uno» wording; 29-N row «Generales 29 nov 2026»; card_start gained a backwards-compatible subtitle_class; earlier Medium/registration assertions narrowed to the page head / before #datos; the 16-provincias/162-pueblos fichas stay qualitative (figures not in the parts); 2008 voters fallback typed only when no pre-rogado election is in the parts.
Task 17: review spec ❌ — Important: the `many` macro compares int to string ("1 escaños"); minors: typed 383.000 fallback and «desde 1986», repeated .claim CSS rules, two pluralisation mechanisms.
Task 17: fix round 1/5 (1 addressed pending re-review — `many` decides on the number itself; the finding was latent, since the site overrides `int` with fmt_int; typed 383.000 dropped; duplicated .claim CSS merged; commits 6670b48..f236b23)
Task 17: fix round 1/5 (3 addressed, 0 open — `many` on the number itself, typed 383.000 dropped, CSS merged; commits 6670b48..f236b23)
Task 17: minor (deferred): «desde 1986» typed in the report ficha; «29-N» typed in several places (accepted); `round` gives the singular to 0,6–1,4 (counts are ints).
Task 17: complete (commits 6670b48..f236b23, review clean after round 1)
Task 18: dispatched (BASE f236b23, model opus, worktree)
Task 18: implementer DONE_WITH_CONCERNS (343198d): new charts/grouped.js (grouped/stacked/horizontal) besides census/scatter/slope/radar; island gains census.periods and winner/runner_up on gap rows; display names of consulate processes and origins come from a map in the page module (a new consulate file shows its raw name); phone: a large negative net label overlaps the province names in the gap chart; slope categories say «Último comicio» (no years in the island).
Task 18: minor (deferred): signed/signedTick duplicated in analisis-cera.js and slope.js; hard-coded light colours in radar.js/grouped.js (dark mode); drawGap leaves a stale chart on an empty scenario; no JS tests of the page module helpers; the island sensitive key carries placement.sensitive whole (small).
Task 18: complete (commits 343198d, review clean; -W error run by the reviewer)
Task 19: dispatched (BASE 343198d, model opus, worktree)
Task 19: implementer DONE_WITH_CONCERNS (8d4fa83, 5e1f028): suite 1096 passed / 75 skipped / 1 pre-existing failure; local publish worked through job.py without a database (DB_ADAPTER=PostgreSQL only); the pages need a published forecast run (503 otherwise, by design); all walkthrough checks passed; the page shows the 2023 baseline as 2.327.388 (monthly series at the baseline date, the model base) while Infoelectoral gives 2.328.260; spec D8 still names ecological_min_census; test_load_cera pin made robust; README cerapropais row updated.
Ruling: the KPI keeps the model baseline (2.327.388, the census the increment is measured against); the final fix wave adds one «Cómo leerlo» line explaining the 872-elector difference with Infoelectoral and fixes spec D8 (ecological_min_voters) — cost if wrong: a line of copy.
Task 19: complete (commits 8d4fa83..5e1f028, review clean; minors: the -W error ResourceWarning of test_cera_data_folder_is_complete (deferred since Task 1), a dense «Estado» sentence in the spec header)
Final review: dispatched on fable over 0347c23..5e1f028 (27 commits) with the ledger copy for the deferred-minors triage.
Final review (fable): ready with fixes — Important: (1) exporters/pages print full candidature names instead of siglas; (2) the page shows the article periods/sensitive shares (article) while the spec says the updated version and a sentence claims they update; (3) the two promised «Cómo leerlo» explanations (2.327.388 vs 2.328.260; 208.431 vs 233.688) are missing. Fix-before-merge minors: unclosed open() in test_cera_data.py; spec D2/D8 still say ecological_min_census; clear ValueError for a scrutiny without candidatures entry and a note when a scrutiny is skipped; plus cheap extras accepted by the controller: optional re-download of the election tables, None-safe gap sort, «desde 1986» from history.from, spec step 4 additions (params.blocks check, pinned tests).
Ruling: the page shows the UPDATED periods/sensitive shares (the `now` versions, as the spec D2 says) and cites the article window figures in «Cómo leerlo» as the replication — cost if wrong: the Q1/Q2 figures move with every monthly refresh (intended).
Final fix wave: dispatched (BASE 5e1f028, model opus, worktree)
Final fix wave: DONE_WITH_CONCERNS (116ddfc, 043dbeb, b024e11, 05d918c): siglas as Infoelectoral writes them (Junts → «JxCat»); Madrid table also on the updated periods; OCE 233.688 as a constant OCE_RECEIVED in pages/cera.py (no part carries it); «se duplica» still fixed text (×2,3 today); skipped scrutinies recorded in meta.skipped_scrutinies; suite 1103 passed / 75 skipped / 1 pre-existing failure.
Final re-review: all findings addressed, no new breakage (47 passed under -W error).
Final review: parked — «se duplica» (resumen, summary, chart title) is fixed text while the ratio is live (×2,3 today) — Ruling: real and deferred; gate it on the ratio in a follow-up — cost if wrong: a wrong verb if the ratio falls below ~1,8.
Final review: parked — skipped scrutinies are recorded in meta.skipped_scrutinies but not printed by the publish job — Ruling: real and deferred (the meta carries it) — cost if wrong: a puzzled republish after the 29-N.
Final review: parked — the spec says «papeletas» where the page says «sobres de voto recibidos» — Ruling: wording only, deferred.
