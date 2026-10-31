# SDD ledger — plan: docs/superpowers/plans/2026-10-31-jobs-pipelines-fase-1.md

Spec: docs/superpowers/specs/2026-10-31-jobs-pipelines-ecs-design.md (approved by Luis 2026-10-31, ce02c16).
Plans committed 710ea47. Execution method chosen by Luis: subagent-driven (2026-10-31).
Phase 2 plan (docs/superpowers/plans/2026-10-31-jobs-pipelines-fase-2-ecs.md) runs after phase 1 is green;
Luis deferred the question about `run-job.sh` ("luego vemos eso") — phase 2 Task 4 may change.

Ruling: work on branch `dev` in the shared checkout, no worktree — Luis's standing workflow (he commits in
parallel with `git commit -a` in this tree; plan Global Constraints say so); `dev` is not main/master.
Cost if wrong: his parallel commits could interleave with task commits; mitigated by `git status` before each
commit and adding only task files.

## Pre-flight scan (2026-10-31)

| Tasks | Shared file / interface | Produces vs consumes | Finding |
| --- | --- | --- | --- |
| T1/T2/T3 | mtpy/core/worker.py, tests/test_worker_unit.py | T1 Core.log, Pipeline hooks → T2 Job.alert/dispatch uses Core.log → T3 SequenceJob uses Pipeline.result/line/finish/failure_name and Job.alert/alerted | consistent; sequential edits of one file |
| T2/T4 | JobInterrupted | T2 defines (BaseException) → T4 job.py interrupt raises it | consistent |
| T3/T4 | SequenceJob.select_pipelines + sequence snake names | T4 jobs/Pipeline.py stores `{group: [snake]}`; T3 camelizes with to_camel | consistent |
| T1/T5 | Pipeline.summary(chunk, **kwargs), start(**kwargs incl. position) | T5 ScopePipeline.summary/start override with same shapes | consistent |
| T3/T5-T9 | SequenceJob.results keyed ('grupo.pipeline', chunk); chunk_noun; failure_name | jobs reshape results by those keys; Backtest label=None → failure label = scope | consistent |
| T8/T9 | mtpy/jobs/Publish.py, tests/test_jobs_publish.py, tests/test_pipelines_publish.py, tests/publish_fixtures.py | T8 creates fixtures file and the job; T9 adds six pipelines and removes T8's temporary skips | consistent; T8 leaves `@pytest.mark.skip(reason='Tarea 9')` on job tests of T9 pipelines |
| T5/T6-T9 | mtpy/pipelines/base.py ScopePipeline | consumed by every group | consistent |
| T7 / phase 2 T2 | backtest.Backtest start header uses shown(out) | phase 2 replaces by describe_source | consistent (later) |
| each task | own tests vs own code | T1 tests use fresh_app with no config/logger (tolerance in scope); T3 tests restricted to `select='grp'` where failure counts matter; T4 tests patch job.mtpy.run | self-consistent |

Ruling: T8's temporary `@pytest.mark.skip(reason='Tarea 9')` on job tests that need pipelines built in T9 is a
plan-mandated transitional state; a T8 reviewer may flag it, it is accepted and verified removed in T9.
Cost if wrong: none beyond one commit with skipped tests.

Ruling: no TodoWrite tool in this session; this ledger is the task tracker.

## Tasks

- [x] Task 1: Pipeline core (result/line, start/finish, end, chunked, tolerance)
- [x] Task 2: Job.alert/dispatch, JobInterrupted, Worker
- [x] Task 3: SequenceJob
- [x] Task 4: launchers (job.py, pipeline.py, jobs/Pipeline.py)
- [x] Task 5: load group + Load job + ScopePipeline
- [x] Task 6: compute group + Compute job
- [x] Task 7: backtest group + Backtest job
- [x] Task 8: publish 1/2 (Forecast, Parties, Manifest) + Publish job
- [x] Task 9: publish 2/2 (Analysis, Event, Backtest, Study, Point, Unpublish)
- [x] Task 10: docs, full suite, manual check

## Log
- 2026-10-31 18:56 Task 1 dispatched: BASE 710ea47, implementer sonnet, brief task-1-brief.md
- 2026-10-31 18:59 Task 1 implementer DONE: commit 8e30900; concern: summary(chunk, **kwargs) got chunk twice → chunk passed positionally and stripped from kwargs (same as start). Ruling: that is the intended reading; later tasks' summary/line_of take chunk positionally.
- Task 1 review (sonnet): spec ✅, quality Approved, no Critical/Important. ⚠️ suite-green claim resolved by the controller from the report's pytest output (1488 passed, 76 skipped, 1 pre-existing matplotlib warning).
- Task 1: minor (deferred): a run ended early still logs the closing `Pipeline X.Y executed in … :-)` line after `ended: …`.
- Task 1: minor (deferred): `PipelineEnd` raised inside `Build` goes through `except Exception` → `target.rollback()`; undocumented, untested.
- Task 1: minor (deferred): tests do not assert the `ended:` log line, the start-after-_action order, nor the explicit logger-None path; `Counter.finish/finished` unused until Task 3.
- Task 1: complete (commits 710ea47..8e30900, review clean)
- 2026-10-31 19:01 Task 2 dispatched: BASE 8e30900, implementer sonnet, brief task-2-brief.md
- 2026-10-31 19:05 Task 2 implementer DONE: commit 3299fa9 (TDD order slip, RED reproduced by stashing); suite 1495 passed, 76 skipped.
- Task 2 review (sonnet): spec ✅, Approved, no Critical/Important. ⚠️ items resolved by the controller: `Core.log` None-tolerance is Task 1's tested behaviour; `Log.log` passes `exc_info=True` for 'error' (app.py); today's jobs alert their summary then raise, so `alerted` suppresses a second alert by design (and Tasks 5-9 rewrite them).
- Task 2: minor (deferred): `self.log(...)` before `alert` in `dispatch`/`alert` is unguarded; a logger that raises masks the original exception.
- Task 2: minor (deferred): untested edges — exactly 1024/1025 chars, KeyboardInterrupt wording, after_dispatch skipped on failure, alerted reset between dispatches, BaseException propagating through handle_message.
- Task 2: complete (commits 8e30900..3299fa9, review clean)
- 2026-10-31 19:06 Task 3 dispatched: BASE 3299fa9, implementer sonnet, brief task-3-brief.md
- 2026-10-31 19:12 Task 3 implementer DONE_WITH_CONCERNS: commit 8befdb6; suite 1507 passed. Concern 1 (test scaffolding records ('work','stop') before raising) accepted. Concern 2 (summary body de-duplicated by text) is a correctness defect → implementer resumed before review to track failed lines by position + test.
- 2026-10-31 19:12 Task 3 correction DONE: commit 3fec973 (failed lines by index, new test); review package 3299fa9..3fec973
- Task 3 review (opus): spec ❌ — Important #1: an exception raised outside a chunk (pipeline instantiation, `chunked`, `failure_name`) sends a `:-)` alert and suppresses dispatch's failure alert. Fix loop entered.
- Task 3 ⚠️ resolved by the controller: non-string chunks — every job passes scope codes (strings), not a gap; `finish` on interruption — the spec routes an interruption through the failure path only (line, alert, exit ≠ 0) and the half-done run is retired with `unpublish`, not a gap.
- Task 3: minor (deferred): interruption line `'.' + chunk` breaks on a non-string chunk (use format).
- Task 3: minor (deferred): stop mode with a chunk failure followed by a failing `finish` raises finish's RuntimeError instead of re-raising the chunk's exception.
- Task 3: minor (deferred): stale type hints of `select`/`chunks`; a `select` of a type other than str/list/tuple silently means "everything"; log message `'grp.beta None: …'` for unchunked runs; untested: select entries with spaces, stop mode chunk+finish failure.
- Task 3: fix round 1/5 dispatched (Important #1 exception outside a chunk; commit 4dbd1ab; suite 1509 passed) → scoped re-review 3fec973..4dbd1ab
- Task 3: fix round 1/5 (1 addressed, 0 open — exception outside a chunk now a counted failure; commits 3fec973..4dbd1ab)
- Task 3: minor (deferred): if `failure_name` raises after a chunk's failed line was recorded, the outer handler adds a second failed line for the same key (over-reports, never hides).
- Task 3: complete (commits 3299fa9..4dbd1ab, review clean after 1 fix round)
- 2026-10-31 19:20 Task 4 dispatched: BASE 4dbd1ab, implementer sonnet, brief task-4-brief.md
- 2026-10-31 19:25 Task 4 implementer DONE: commit 66e1b0e; suite 1514 passed. Deviation accepted: pipeline.py resets pipeline.line to None before finish() (mirrors SequenceJob._finish).
- Task 4 review (sonnet): spec ✅, Approved, no Critical/Important. ⚠️ items are Task 1/2 behaviour already reviewed (dispatch alerts on JobInterrupted; Pipeline.line/finish).
- Task 4: minor (deferred): in `job.py` an unknown job name or malformed JSON raises outside the try (exit 1 with a traceback, no alert); `pipeline.py` installs no SIGTERM handler (by design: laptop launcher); `name.split('.')` in pipeline.py is pre-existing.
- Task 4: complete (commits 4dbd1ab..66e1b0e, review clean)
- 2026-10-31 19:26 Task 5 dispatched: BASE 66e1b0e, implementer sonnet, brief task-5-brief.md
- 2026-10-31 19:33 Task 5 implementer DONE_WITH_CONCERNS: commit 97b74c5; suite 1528 passed. Concerns accepted: no RED run (one over-specified dict assertion relaxed); job-level event-lookup isolation test ported to the pipeline (job isolation still covered by test_load_isolates_a_failing_scope...).
- Task 5 review (opus): spec ❌ — Important: job-level `years` int→str conversion lost its test. Fix loop entered. ⚠️ resolved: suite count 1528+76 = 1604 is 1557 + the tests added by Tasks 1-5; the matplotlib warning was reported pre-existing by every task since Task 1.
- Task 5 Ruling: the duplicated `notes/collect_warnings/except → failed/warnings` block of `Polls.load_task`/`Params.load_task` and the progress creation are moved into `ScopePipeline` helpers now (`start_progress`, `collect`) — later groups (Tasks 6-9) inherit the base, so the copy would spread; `failed(**extra)` used as designed. Included in fix round 1 by controller ruling (minor by severity, load-bearing by position). Cost if wrong: a small refactor of two pipelines.
- Task 5: minor (deferred): job-level test of an exception at Setup on one scope while the other loads exists only at pipeline level (job isolation covered by test_load_isolates_a_failing_scope...).
- Task 5: fix round 1/5 dispatched (years job test + base helpers start_progress/collect + cleanups; commit 1da0074; suite 1529 passed) → scoped re-review 97b74c5..1da0074
- Task 5: fix round 1/5 (3 addressed, 0 open — years job test, base helpers, cleanups; commits 97b74c5..1da0074)
- Task 5: complete (commits 66e1b0e..1da0074, review clean after 1 fix round)
- Task 5 Ruling (carried to Task 6): `ScopePipeline.collect` must re-raise `PipelineEnd` (today it would swallow an early end raised inside `work()`) and tolerate `self.progress is None` (unchunked pipelines like `Catalogues` have no progress object). Task 6 adds both to base.py with tests in tests/test_pipelines_base.py. Cost if wrong: two tiny base lines.
- 2026-10-31 19:41 Task 6 dispatched: BASE 1da0074, implementer sonnet, brief task-6-brief.md
- 2026-10-31 19:47 Task 6 implementer DONE: commit 5a1ab71; suite 1541 passed. Plan test assertion ['es']*3 corrected to *2 (FakeComputer fails at 'errors', the third Computer is never built) — accepted.
- Task 6 review (sonnet): spec ✅, Approved, no Critical/Important. ⚠️ kwargs forwarding by SequenceJob is Task 3 behaviour, reviewed there.
- Task 6: minor (deferred): `mtpy/jobs/Compute.py` docstring line over 120 columns; `Catalogues.line_of` calls `Compute.line_of` unbound (a module-level `format_line` would be cleaner); the empty-scopes ValueError also refuses `what=["catalogues"]` with `scopes=[]` (plan-mandated); `featured` passed to `Ratings.unit` and ignored.
- Task 6: complete (commits 1da0074..5a1ab71, review clean)
- 2026-10-31 19:48 Task 7 dispatched: BASE 5a1ab71, implementer sonnet, brief task-7-brief.md
- 2026-10-31 19:54 Task 7 implementer DONE: commit 7243146; suite 1545 passed (no separate RED run; all old scenarios kept).
- Task 7 review (sonnet): spec ✅, Approved, no Critical/Important. ⚠️ stop_on_error semantics are Task 3 behaviour.
- Task 7: minor (deferred): no test of the header with `out=None` nor of "skipped scope prints no table"; `table` not initialised at class level; `mtpy/jobs/Backtest.py` docstring line over 120 columns; no RED run.
- Task 7: complete (commits 5a1ab71..7243146, review clean)
- 2026-10-31 19:56 Task 8 dispatched: BASE 7243146, implementer opus (Forecast manifest/backfill logic), brief task-8-brief.md
- Task 8 implementer DONE_WITH_CONCERNS: commit e0d6303; suite 1541 passed, 93 skipped (17 'Tarea 9' skips). Concern 1 is real: the job now calls `resolve_scopes` (a DB read) even for manifest/parties/study-only runs, which today need no database (veda rehearsal runs `publish manifest` locally).
- Task 8 Ruling: `Publish.run` resolves the scopes (and applies the empty-list ValueError) only when a chunked action is requested (forecast, analysis, event, backtest, point, unpublish); manifest/parties/study-only runs pass `chunks=[]` and touch no database, as today. Applied by the implementer before review. Cost if wrong: one condition in the job.
- 2026-10-31 20:11 Task 8 pre-review fix DONE: commit 452386e (lazy resolve_scopes + test); review package 7243146..452386e
- Task 8 review (opus): spec ✅, Approved, no Critical/Important; moved Forecast logic verified verbatim; all 57 old scenarios accounted for (44 job, 27 pipeline, 17 skipped 'Tarea 9').
- Task 8 Ruling (carried to Task 9): `Forecast.__init__` initialises `self.entries = {}` (so `finish` never raises AttributeError when `start` failed); the job's results loop uses `self.CHUNKED` instead of a repeated literal; the test `test_manifest_parties_and_study_do_not_resolve_scopes` gains its `study` assertion once Study exists. Cost if wrong: trivial.
- Task 8: minor (deferred): `stop_on_error` type check runs after `resolve_scopes` (one DB read before the rejection).
- Task 8: complete (commits 7243146..452386e, review clean)
- 2026-10-31 20:16 Task 9 dispatched: BASE 452386e, implementer sonnet, brief task-9-brief.md
- Task 9 implementer DONE_WITH_CONCERNS: commit f37691f; suite 1571 passed, 76 skipped, no 'Tarea 9' skips left. Concern: `Pipeline.run` logs `self.result.get('status')` after a `PipelineEnd`, which breaks for list results, so `Event` avoided `self.end([...])` with a workaround.
- Task 9 Ruling: fix the core instead of working around it — `Pipeline.run` logs status/reason only for dict results (a list result logs `ended`), with a core test (`end` with a list), and `Event.resolve_dates_task` uses `self.end([...])` as the brief says (workaround removed). Applied by the implementer before review. Cost if wrong: one core line.
- 2026-10-31 20:28 Task 9 pre-review fix DONE: commit d25504b (core end() with list results, Event uses end); suite 1572 passed; review package 452386e..d25504b
- Task 9 review (opus): spec ✅, Approved, no Critical/Important; moved bodies verified verbatim; 17 reactivated tests intact; core change compatible and tested. ⚠️ resolved: an exception in `Analysis.start` (national_event) now fails every chunk with an alert instead of crashing the job — Task 3 semantics, accepted.
- Task 9: minor (deferred): `monkeypatch.undo()` in `test_event_without_dates_ends_in_setup` undoes the `patched` fakes too (restore only `event_dates_to_publish`); the point/unpublish `ValueError`-as-`str(e)` rule not asserted; no test that point/unpublish print no progress line; key order of `failed(..., event_date=...)` results changed (dict equality unaffected); report said 13 new tests, diff has 11 + 1.
- Task 9: complete (commits 452386e..d25504b, review clean)
- 2026-10-31 20:31 Task 10 dispatched: BASE d25504b, implementer sonnet, brief task-10-brief.md
- 2026-10-31 20:37 Task 10 implementer DONE: commits 7d0441e, f18cabf; suite 1572 passed, 76 skipped; manual check against the RDS: 3 runs exit 0 (es published 28 cases; es-cb published; /tmp/empty → skipped). Concern accepted: 28 cases (the plan's 30 was approximate; the README example already said 28).
- Task 10 review (sonnet): spec ❌ — Important: the `stop_on_error` rows of load/compute/backtest carry the manifest clause (plan-mandated text, but false for those jobs). Ruling: the plan's row text was wrong; the clause stays only in the publish row (spec scopes the manifest to forecast). Fix loop entered.
- Task 10: minor (deferred): «uno que falla no detiene a los demás» sentences are unconditional (true by default); the summary paragraph does not say that in stop mode the chunk's own exception is re-raised instead of the RuntimeError; `APP_VERBOSE=0` behaves like empty; the pipelines table row over 120 columns.
- Task 10: fix round 1/5 dispatched (manifest clause only in the publish row; commit 7691c5d) → scoped re-review f18cabf..7691c5d
- Task 10: fix round 1/5 (1 addressed, 0 open — manifest clause only in the publish row; commits f18cabf..7691c5d)
- Task 10: complete (commits d25504b..7691c5d, review clean after 1 fix round)
- ALL TASKS COMPLETE. Final whole-branch review dispatched (opus) over 710ea47..7691c5d.
- Final review (opus): ready to merge WITH FIXES. Important #1: the publish pipelines never fall back to `app.fs`, so the documented standalone launches (`job.py pipeline`, `pipeline.py`) fail every scope with AttributeError. Minors: pipeline.py always exits 0; `job.py pipeline` stops at the first error (framework default) and the README does not say so; the interruption line is a hard-coded core string that breaks on non-string chunks; exception-path line drops the action; a failed backfill day flags the whole scope's lines; phase-2 hazard: `backtest '{"action":"pull"}'` today runs a full backtest (validate `action` in phase 2 — already in the phase-2 plan Task 2); base.py imports from mtpy/jobs/progress.py. Triage of the ledger minors: all stay deferred except T4 (no alert on unknown job name / bad JSON) promoted to "must fix in phase 2". All rulings agreed.
- Final review Ruling: single fix wave = Important #1 (a `PublishPipeline` base with `writer_for(fs, dry_run)` resolving `app.fs` and raising the job's RuntimeError text; test of `Pipeline` job on `publish.manifest`), Minor 2 (pipeline.py exit 1 on failure), Minor 3 (README note on the standalone launcher's stop default), Minor 4 (interruption line as a class attribute built with format). Minor 5 (exception-path line with the action) stays deferred: it would change the line format for labelled pipelines; Minor 6/8 deferred. Phase-2 carry-overs: T4 alert gap, `action` validation in Backtest. Cost if wrong: small.
- 2026-10-31 20:56 Final fix wave DONE: commits bba80e8, 2bbd057; suite 1577 passed, 76 skipped → scoped re-review 7691c5d..2bbd057
- Final fix wave re-review (sonnet): all 4 findings ADDRESSED, no new breakage (commits 7691c5d..2bbd057). FINAL REVIEW CLEAN.
- Phase-2 carry-overs recorded in the phase-2 plan: alert when job.py cannot even build the job (unknown name / bad JSON); `action` validation in the Backtest job (already its Task 2).
