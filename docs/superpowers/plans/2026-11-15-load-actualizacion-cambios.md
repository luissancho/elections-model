# Job `load`: actualización de los sondeos cambiados — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que un `load` sin `overwrite` reescriba los sondeos cambiados y borre los viejos sustituidos, con un freno de 10.

**Architecture:** `poll_changes` devuelve además las claves a tocar; el cargador gana dos métodos (sumar los cambiados a lo que se guarda, borrar filas por clave); `load_polls` decide y marca `updated`; la línea del pipeline dice `updated` o `changed`.

**Tech Stack:** Python 3.11, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-load-actualizacion-cambios-design.md`

## Global Constraints

- `MAX_UPDATES = 10`: con más de 10 entre cambiados y sustituidos no se aplica ninguno.
- `removed` nunca se borra. `dry_run` nunca escribe ni borra. Con `overwrite` no se llama a los métodos nuevos.
- Clave de trabajo: `(date 'YYYY-MM-DD', pollster_id int, sponsor_id int o None)`.
- Los `DELETE` llevan siempre los filtros del evento (`self.filters`) además de la clave.
- Las pruebas no tocan la RDS; la verificación real es sólo un `dry_run`. Un `load` real lo lanza Luis.
- PEP8, docstrings, líneas ≤ 120, identificadores en inglés. Commits sólo de los ficheros de la tarea.

## Review Focus

1. **`DELETE` demasiado ancho:** una clave mal formada no puede borrar más que su sondeo. → T1 `test_delete_polls_builds_one_filtered_delete_per_table`, lista vacía sin consulta.
2. **Sondeo a la vez nuevo y cambiado** (no debe guardarse dos veces). → T1 `test_select_changed_adds_the_read_rows_once`.
3. **`dry_run` con cambios:** ni borrado ni guardado. → T2 `test_load_polls_dry_run_updates_nothing`.
4. **Freno:** 11 cambios no tocan nada y avisan; los nuevos se guardan. → T2 `test_load_polls_brake_leaves_the_changes_unapplied`.
5. **Comparación fallida:** no se actualiza nada. → T2, ampliando `test_load_polls_survives_a_failing_comparison`.

---

### Task 1: claves en `poll_changes` y los dos métodos del cargador

**Files:** `mtpy/lib/loader.py`, `mtpy/lib/load.py`; tests `tests/test_load_unit.py`, `tests/test_loader_marks.py`.

**Interfaces — Produces:**
- `loader.poll_row_key(row: dict) -> tuple` (sustituye a `load._poll_key`, que lo importa).
- `poll_changes(...)['keys'] == {'changed': [key, ...], 'replaced': [old_key, ...]}`, en el orden de las listas.
- `WikipediaLoader.select_changed(keys: list) -> Self`: suma a `self.polls` y `self.results` las filas de `read_polls`/`read_results` con esas claves que aún no estén en `self.polls`.
- `WikipediaLoader.delete_polls(keys: list) -> Self`: un `DELETE` en `m_results` y otro en `m_polls`, `WHERE {filters} AND ((date = 'd' AND pollster_id = p AND sponsor_id = s) OR ...)` (`sponsor_id IS NULL` para `None`); sin claves, ninguna consulta.

- [ ] Pruebas que fallan: `test_poll_changes_returns_the_keys_to_update`, `test_select_changed_adds_the_read_rows_once`, `test_delete_polls_builds_one_filtered_delete_per_table`.
- [ ] Verlas fallar: `pytest tests/test_load_unit.py tests/test_loader_marks.py -q`.
- [ ] Implementar. Verlas pasar. Commit.

### Task 2: `load_polls` aplica los cambios

**Files:** `mtpy/lib/load.py`; tests `tests/test_load_unit.py` (y `FakeLoader`).

**Interfaces — Produces:** `result['updated']` (bool, `False` por defecto): `True` si los cambios se han escrito (`save` y, o `overwrite`, o hay cambios bajo el freno y se aplicaron). `result['n_saved']` cuenta nuevos y cambiados. `keys` no está en el resultado. Aviso del freno: `'{n} changed polls not updated (more than {MAX_UPDATES}): check the article or load with overwrite'`.

- [ ] Pruebas que fallan: `test_load_polls_updates_the_changed_polls` (llamadas `select_changed`, `delete_polls` antes de `save_polls`; `updated` True; `n_saved`), `test_load_polls_deletes_the_poll_a_new_one_replaces`, `test_load_polls_dry_run_updates_nothing`, `test_load_polls_overwrite_marks_updated_without_deleting`, `test_load_polls_brake_leaves_the_changes_unapplied`, y `survives_a_failing_comparison` ampliada.
- [ ] Verlas fallar, implementar, verlas pasar, commit.

### Task 3: la línea y la documentación

**Files:** `mtpy/pipelines/load/Polls.py`, `mtpy/jobs/Load.py`, `mtpy/jobs/Update.py` (docstrings); tests `tests/test_pipelines_load.py`, `tests/test_jobs_update.py`.

- [ ] Pruebas: la línea dice `· 1 updated, 1 removed` con `updated` y `· 1 changed, 1 removed` sin él; ajustar las existentes (`n_saved`).
- [ ] Implementar, `pytest -q` completo, `dry_run` real de `es` (esperado: sin `~`), commit.
