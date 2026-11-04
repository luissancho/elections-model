# Evolución de las actualizaciones: donuts de cambio y tabla de cambios — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un punto por día con el último run (sin distinguir relleno), donut sólo cuando cambia la cifra visible, y
una tabla de cambios (Hoy / 7 días / Desde el inicio) bajo el gráfico de las tarjetas de Promedios y Escaños.

**Architecture:** Todo en el navegador, sobre los runs diarios que la tarjeta ya recibe (`initial.runs`). La regla
de la cifra visible (`shownValue`) es única y la usan los donuts, el tooltip y la tabla. La tabla es una cadena
HTML pura (probable con `node`) que `wireEvolution` inserta y oculta por filas según la leyenda.

**Tech Stack:** JS ES modules sin build (`web/dist/js`), ECharts 5.6, Jinja2, pruebas `pytest` que lanzan `node`.

**Spec:** `docs/superpowers/specs/2026-11-04-evolucion-cambios-design.md`

## Global Constraints

- Palabra pública «actualización», nunca «publicación»; «sólo» con tilde; nunca «empresa» (lo vigila
  `assert_public_wording`).
- Identificadores en inglés neutro; textos públicos en español.
- Cifras: voto a 1 decimal (`fmtPct`), escaños enteros (`fmtInt`); cambios con la flecha `.dlt` (`▲ 0,2`, `▼ 1`, «–»).
- Pruebas JS nunca escriben en `backtest/results` ni en el bundle; nada de `out=None`.
- Luis hace commits en paralelo en el mismo árbol: antes de cada commit, `git status` y `git log -1`; añadir sólo
  los ficheros de la tarea, nunca `git commit -a`.

## Review Focus

- Deriva de 0,01 diaria sin sondeos (18,87 → 18,88): sin donut ni flecha mientras la cifra a un decimal no cambie
  (Tarea 2, `test_donut_only_when_the_shown_figure_changes`).
- Un nombre ausente en algunos días (VOX nulo un día): su donut y su «Hoy» comparan con su último día con dato, y
  una comparación sin dato da «–» (Tareas 2 y 3).
- Histórico con un solo día: no hay tabla ni donuts, sin errores (Tarea 3, `test_changes_with_a_single_day`).
- Último run que no es de hoy: la columna se titula con su fecha, no «Hoy» (Tarea 3).
- Vista Bloques y selección de la leyenda: la tabla cambia de nombres con la vista y oculta las filas ocultas en el
  gráfico, y la descarga lleva la misma tabla (Tarea 3, comprobación en navegador).

---

### Task 1: `fmtBadge`, espejo de `delta_badge`

**Files:**
- Modify: `web/dist/js/format.js` (junto a `fmtDelta`, ~l. 207-224)
- Test: `tests/test_js_format.py`

**Interfaces:**
- Produces: `fmtBadge(x: number|null, digits = 0) -> {dir: 'up'|'down'|'flat', text: string}`, mismo contrato que
  `delta_badge` de `mtpy/lib/pages/filters.py`. `text` es el de `fmtDelta`.

- [ ] **Step 1: Prueba que falla** — `test_fmt_badge_follows_the_python_filter`: para
  `x in (6, -1, 0.04, -0.06, 1.0, -2.55, 0)` y `digits in (0, 1)`,
  `run_js(f'f.fmtBadge({x}, {digits})') == pages.delta_badge(x, digits)`; y `f.fmtBadge(null)` ==
  `{'dir': 'flat', 'text': '–'}`.
- [ ] **Step 2:** `pytest tests/test_js_format.py::test_fmt_badge_follows_the_python_filter -q` → FAIL (no existe).
- [ ] **Step 3:** Implementar `fmtBadge` en `format.js` sobre `fmtDelta` (`flat` cuando el texto es `DASH`).
- [ ] **Step 4:** `pytest tests/test_js_format.py -q` → PASS.
- [ ] **Step 5: Commit** `web/dist/js/format.js tests/test_js_format.py` — «web: fmtBadge, espejo JS de delta_badge».

### Task 2: un punto por día, donut de cambio y tooltip

**Files:**
- Modify: `web/dist/js/charts/evolution.js` (`beats`, `dailyRuns`, `evolutionOption`, cabecera del módulo)
- Test: `tests/test_js_evolution.py`

**Interfaces:**
- Consumes: `fmtBadge` (Tarea 1).
- Produces (exportados desde `evolution.js`):
  - `FIELD_DIGITS = {pct: 1, seats: 0}`.
  - `shownValue(value: number|null, field: string) -> number|null`: la cifra que se ve,
    `Math.round(value * 10 ** d) / 10 ** d` con `d = FIELD_DIGITS[field]`; `null` si falta.
  - `dailyRuns(runs)`: un run por día, el de `run_at` más tardío **sin mirar `backfill`**.

- [ ] **Step 1: Pruebas que fallan** (ajustar las actuales, añadir las nuevas):
  - Renombrar `test_daily_runs_keep_one_run_per_day_live_over_backfill_and_the_last_live` →
    `test_daily_runs_keep_the_last_run_of_each_day`: con el fijo, `['b05', 'b06', 'l07b', 'x08']` (el 06 gana
    `b06`, 12:00, a `l06`, 09:30).
  - Sustituir `test_evolution_option_draws_a_line_per_party_with_hollow_retrospective_points` por
    `test_evolution_option_draws_a_line_per_party_with_solid_points_and_donuts_on_change`: el primer punto de PP es
    un array plano `[t, 32.71, 0]`; en PP (32.71 → 32.64 → 32.4 → 32.3; cifras 32,7 / 32,6 / 32,4 / 32,3) los
    puntos 1-3 son objetos con `itemStyle == {'color': '#fff', 'borderColor': '#1e5aa8', 'borderWidth': 2}` y
    `symbolSize == 8`; el resto de aserciones de la prueba actual (leyenda, zoom, eje, serie vacía) se mantiene.
  - `test_donut_only_when_the_shown_figure_changes`: con runs propios de 4 días y VOX 18.86, 18.87, 18.88, 18.96
    (cifras 18,9 / 18,9 / 18,9 / 19,0), sólo el cuarto punto es objeto (donut); en `field: "seats"` con 63, 63, 64
    sólo el tercero.
  - `test_donut_compares_with_the_last_day_of_the_same_name`: VOX 17.95, (nulo), 18.1 → el punto de 18,1 es donut
    frente a 18,0 (el día nulo no cuenta).
  - `test_shown_value_rounds_as_the_figure`: `shownValue(18.25, "pct") == 18.3`, `shownValue(63.4, "seats") == 63`,
    `shownValue(null, "pct") is None`.
- [ ] **Step 2:** `pytest tests/test_js_evolution.py -q` → las nuevas y las ajustadas FAIL.
- [ ] **Step 3: Implementar** en `evolution.js`:
  - `beats(run, kept)`: sólo `run_at` posterior.
  - En `evolutionOption`, un punto es donut si su `shownValue` difiere del del punto anterior de la misma serie
    (los puntos de `evolutionPoints` ya saltan los días sin dato). Mismo estilo de donut que hoy.
  - Tooltip: cabecera `fmtDate(runDay(run))` siempre (sin hora ni «estimación retrospectiva»); cada línea añade,
    cuando el cambio de cifra visible frente al punto anterior de esa serie no es nulo,
    ` <span class="dlt ${dir}">${text}</span>` con `fmtBadge(Δ, FIELD_DIGITS[field])`.
  - Actualizar el comentario de cabecera del módulo y el JSDoc de `dailyRuns`/`evolutionOption` (sin «hollow
    retrospective»).
- [ ] **Step 4:** `pytest tests/test_js_evolution.py tests/test_js_blocks.py -q` → PASS.
- [ ] **Step 5: Commit** `web/dist/js/charts/evolution.js tests/test_js_evolution.py` — «web: evolución con un punto
  por día y donut cuando cambia la cifra».

### Task 3: tabla de cambios bajo el gráfico

**Files:**
- Create: `web/dist/js/evolution-changes.js`
- Modify: `web/dist/js/pages/evolution-wire.js` (`wireEvolution`)
- Modify: `web/templates/promedios.html` (~l. 205-221), `web/templates/escanos.html` (~l. 162-172)
- Test: `tests/test_js_evolution.py` (nuevo `CHANGES_JS` en el arnés `run_js`, como `ch`), `tests/test_pages.py`
  (l. 694 y 921)

**Interfaces:**
- Consumes: `dailyRuns`, `runDay`, `shownValue`, `FIELD_DIGITS` (Tarea 2); `fmtBadge` (Tarea 1); `escapeHtml`
  (`charts/base.js`); `fmtDateShort`, `fmtDate` (`format.js`); `isoDay` (`charts/series.js`).
- Produces:
  - `evolutionChanges(runs, names, field = 'pct', today = isoDay(Date.now())) -> object|null`: `runs` son los
    runs diarios de la vista. `null` con menos de dos días. Si no:
    `{days: {last, previous, week, start}` (fechas ISO; `week` null si ningún día ≤ last − 7),
    `todayLabel` («Hoy» si `last === today`, si no `fmtDateShort(last)`),
    `rows: [{name, value, day, week, start}]}`. `value` es la cifra visible del último día; cada cambio es
    `shownValue(último) − shownValue(entonces)`, redondeado a `FIELD_DIGITS[field]` decimales, o `null` sin dato
    en alguno de los dos días. «Hoy» compara con el día anterior a `last` en que el nombre tiene dato, igual que
    el donut.
  - `changesTableHtml(changes, view, {field = 'pct', formatter, shown = null}) -> string`: `view` con `colors` y
    `fullnames`. Una `<table>` con `thead` (Partido | `fmtDateShort(last)` | `todayLabel` | «7 días» |
    «Desde {fmtDateShort(start)}», cada cabecera de cambio con `title="Frente al {fmtDate(día)}"`) y una fila por
    nombre en el orden de `names`. La celda del nombre es
    `<span class="pn"><span class="dot" style="background-color: …"></span>NOMBRE</span>` con `title` = nombre
    completo, la cifra va en `<strong>`, los cambios van en `<span class="dlt {dir}">`, y la fila lleva
    `data-name` y `hidden` si no está en `shown`. La columna «7 días» se omite si `days.week` es null.

- [ ] **Step 1: Pruebas que fallan** en `tests/test_js_evolution.py`:
  - `test_changes_compare_today_a_week_and_the_start`: con runs propios de 05-10 a 15-10 (un hueco el 08-10) y
    `today = '2026-10-15'`: `days == {'last': '2026-10-15', 'previous': '2026-10-14', 'week': '2026-10-07',
    'start': '2026-10-05'}` (el 08 no está, así que el día ≤ 15−7 es el 07), `todayLabel == 'Hoy'`, y en PP
    (32.71 al inicio, 32.44 el 07, 32.04 el 14, 31.93 el 15) `value == 31.9`, `day == -0.1`, `week == -0.5`,
    `start == -0.8`; y un PSOE con 26.55 el 14 y 26.58 el 15 (26,6 ambos) da `day == 0` (celda «–»).
  - `test_changes_label_the_last_day_when_it_is_not_today`: `today = '2026-10-16'` → `todayLabel == '15 oct'`
    (el formato de `fmtDateShort`).
  - `test_changes_with_a_single_day`: un solo día → `None`; `runs = []` → `None`.
  - `test_changes_missing_name_gives_a_dash`: un nombre sin dato al inicio → `start is None` y su celda del HTML
    es `<span class="dlt flat">–</span>`.
  - `test_changes_table_html`: el HTML lleva las 5 cabeceras en orden, `title="Frente al …"`, una fila por nombre
    en el orden dado, la fila fuera de `shown` con `hidden`, el `<strong>` con `formatter(value)`
    (`31,9%` / `129`), y `<span class="dlt down">▼ 0,5</span>`; sin `week` no hay cabecera «7 días».
  - Ajustar `tests/test_pages.py`: l. 694, `'huecos' in evolution` pasa a `'hueco' in evolution and 'id="evolution-changes"' in evolution`;
    l. 921, `'estimaciones retrospectivas'` pasa a `'cambió' in text and '7 días' in text and 'Desde' in text`.
- [ ] **Step 2:** `pytest tests/test_js_evolution.py tests/test_pages.py -k "evolution or escanos_explains" -q`
  → FAIL.
- [ ] **Step 3: Implementar** `evolution-changes.js` (las dos funciones) y conectarlo en `wireEvolution`:
  - El contenedor es `#evolution-changes`. Si falta, no hay tabla. Si `evolutionChanges` da `null`, el
    contenedor queda vacío y `hidden`.
  - En `draw()` se rellena con la vista y la selección actuales. En `legendselectchanged`, sólo se
    activa o quita `hidden` en las filas por `data-name` (los nombres mostrados son
    `state.selected ?? defaults ?? parties`).
  - `wireDownload` recibe `bottom: () => tableImage(changesEl.querySelector('table'))` cuando hay tabla.
    `readTable` ya salta las filas `hidden`.
  - `digits`/`formatter`: los que ya recibe `wireEvolution` (`fmtPct` en Promedios, `fmtInt` en Escaños).
- [ ] **Step 4: Plantillas.** En las dos tarjetas, tras `#evolution-chart`:
  `<div class="table-wrap" id="evolution-changes" hidden></div>`. En el «Cómo leerlo» de las dos:
  - Quitar la frase de los puntos huecos como estimaciones retrospectivas.
  - Decir que un punto hueco marca un día en que la cifra (el voto a un decimal / los escaños) cambió frente al
    día anterior.
  - Explicar la tabla: la cifra del último día y sus cambios frente al día anterior, a hace una semana y al primer
    día de la serie, con flecha verde si sube y roja si baja.

  Mantener «Si un día hubo varias actualizaciones, se muestra la última», «revisa el pasado» y «no se revisan».
- [ ] **Step 5:** `pytest tests/test_js_evolution.py tests/test_js_format.py tests/test_pages.py -q` → PASS; luego
  la suite completa `pytest -q -x --ignore=tests/integration` → PASS.
- [ ] **Step 6: Comprobación en navegador** (técnica de la memoria: ASGI + Chrome sin cabeza por CDP, o el servidor
  local de Luis):
  - `/es` y `/es/escanos` muestran puntos sólidos y donuts sólo donde cambia la cifra, y la tabla con las 5
    columnas.
  - Ocultar un partido en la leyenda oculta su fila; cambiar a Bloques rehace la tabla.
  - La descarga lleva la tabla debajo.
- [ ] **Step 7: Commit** `web/dist/js/evolution-changes.js web/dist/js/pages/evolution-wire.js
  web/templates/promedios.html web/templates/escanos.html tests/test_js_evolution.py tests/test_pages.py` —
  «web: tabla de cambios de la evolución (hoy, 7 días, desde el inicio)».
